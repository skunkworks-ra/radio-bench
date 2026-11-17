"""Backward-compatible `grade_calibrate` wrapper delegating to `extractors` + `rubric`."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import radio_bench.grading.extractors.calibrate as _cal_ext
from radio_bench.grading.extractors.common import (
    has_active_table_lock,
    open_ro_table,
    resolve_ant_ids_by_names,
    resolve_field_ids_by_names,
    resolve_scratch_root,
)
from radio_bench.grading.extractors.common import (
    parse_sky_coord_rad as _parse_sky_coord_rad,
)
from radio_bench.grading.inspect_metrics import resolve_task_configs
from radio_bench.grading.runner import run_verification

PASS_REWARD_MIN_DEFAULT = 0.85
_run_verifier_tclean_recipe = _cal_ext._run_verifier_tclean_recipe


def _materialize_sky_model_vis_npz(
    ms_path: Path,
    sky_model: dict[str, Any],
    target_names: list[str],
    dead_ant_names: list[str],
    rfi_channels: list[int],
    scratch_root: Path,
) -> Path:
    """Predict noiseless target visibilities into a temporary `true_target_vis.npz` when `.npz` is omitted."""
    from casatools import componentlist, measures, simulator

    tmp_dir = Path(tempfile.mkdtemp(prefix="cal_pred_", dir=str(scratch_root)))
    prev_cwd = Path.cwd()
    try:
        os.chdir(tmp_dir)
        tmp_ms = tmp_dir / "pred.ms"
        shutil.copytree(ms_path, tmp_ms)

        ext = sky_model["extended_gaussian"]
        pt = sky_model["point_source"]
        cl_path = tmp_dir / "target.cl"

        me = measures()
        try:
            cl = componentlist()
            try:
                cl.done()
                cl.addcomponent(
                    dir=me.direction("J2000", str(ext["ra"]), str(ext["dec"])),
                    flux=float(ext["flux_jy"]),
                    fluxunit="Jy",
                    freq="4.600GHz",
                    shape="Gaussian",
                    majoraxis=f"{float(ext['major_arcsec'])}arcsec",
                    minoraxis=f"{float(ext['minor_arcsec'])}arcsec",
                    positionangle=f"{float(ext['pa_deg'])}deg",
                    spectrumtype="constant",
                )
                cl.addcomponent(
                    dir=me.direction("J2000", str(pt["ra"]), str(pt["dec"])),
                    flux=float(pt["flux_jy"]),
                    fluxunit="Jy",
                    freq="4.600GHz",
                    shape="point",
                    spectrumtype="constant",
                )
                cl.rename(str(cl_path))
            finally:
                with contextlib.suppress(Exception):
                    cl.close()
                with contextlib.suppress(Exception):
                    cl.done()
        finally:
            with contextlib.suppress(Exception):
                me.done()

        sm = simulator()
        sm_opened = False
        try:
            sm.openfromms(str(tmp_ms))
            sm_opened = True
            sm.setvp(dovp=True, usedefaultvp=True, dosquint=False)
            sm.predict(complist=str(cl_path), incremental=False)
        finally:
            if sm_opened:
                with contextlib.suppress(Exception):
                    sm.close()
            with contextlib.suppress(Exception):
                sm.done()

        target_fids = resolve_field_ids_by_names(tmp_ms, target_names)
        dead_ids = resolve_ant_ids_by_names(tmp_ms, dead_ant_names)
        with open_ro_table(tmp_ms) as tb_p:
            pred_data = np.asarray(tb_p.getcol("DATA"))
            fids = np.asarray(tb_p.getcol("FIELD_ID"), dtype=int)
            a1 = np.asarray(tb_p.getcol("ANTENNA1"), dtype=int)
            a2 = np.asarray(tb_p.getcol("ANTENNA2"), dtype=int)

        tgt_mask = np.isin(fids, target_fids)
        tgt_row_idx = np.flatnonzero(tgt_mask).astype(np.int32)
        dead_sub = np.isin(a1[tgt_mask], dead_ids) | np.isin(a2[tgt_mask], dead_ids)
        clean_tgt_mask_sub = (~dead_sub) & (a1[tgt_mask] != a2[tgt_mask])
        nchan = pred_data.shape[1]
        clean_chan_mask = np.ones(nchan, dtype=bool)
        for ch in rfi_channels:
            if 0 <= int(ch) < nchan:
                clean_chan_mask[int(ch)] = False
        p_hi = 3 if pred_data.shape[0] >= 4 else (pred_data.shape[0] - 1)
        true_rr = pred_data[0, :, :][:, tgt_mask].astype(np.complex128).copy()
        true_ll = pred_data[p_hi, :, :][:, tgt_mask].astype(np.complex128).copy()

        shutil.rmtree(tmp_ms, ignore_errors=True)
        shutil.rmtree(cl_path, ignore_errors=True)
        npz_out = tmp_dir / "true_target_vis.npz"
        np.savez(
            npz_out,
            target_row_indices=tgt_row_idx,
            clean_target_row_mask=clean_tgt_mask_sub,
            clean_chan_mask=clean_chan_mask,
            true_rr=true_rr,
            true_ll=true_ll,
        )
        return tmp_dir
    finally:
        with contextlib.suppress(Exception):
            os.chdir(prev_cwd)


def grade_calibrate(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path | None = None,
    task_id: str,
    scratch_root: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade `/workspace/input.ms` (+ optional `calibrators.ms` & caltables) via `extractors` + `rubric`."""
    ws = Path(workspace_dir)
    resolved_scratch = resolve_scratch_root(scratch_root)
    ec, rb = resolve_task_configs(task_id, "calibrate", ground_truth)
    if rb.stage_blocks:
        # When `grade_calibrate` is invoked standalone with an e2e task_id (e.g. `s2_e2e_synth`),
        # promote the calibrate stage block to a single-stage calibrate rubric.
        cal_blk = next(b for b in rb.stage_blocks if b.stage == "calibrate")
        rb = rb.model_copy(
            update={
                "stages": ["calibrate"],
                "stage_blocks": [],
                "subscores": cal_blk.subscores,
                "core_subscore_floors": {
                    **rb.core_subscore_floors,
                    **cal_blk.core_subscore_floors,
                },
            },
            deep=True,
        )
        ec.stages = ["calibrate"]
        if cal_blk.truth_prefix and isinstance(ground_truth.get(cal_blk.truth_prefix), dict):
            ground_truth = ground_truth[cal_blk.truth_prefix]

    synth_tmp_dir: Path | None = None
    ref_dir: Path | None = Path(masks_npz_path) if masks_npz_path is not None else None
    if (
        ref_dir is None
        and ec.geometry.verifier_tclean_recipe is None
        and isinstance(ground_truth.get("sky_model"), dict)
        and (ws / "input.ms" / "table.dat").is_file()
    ):
        csum = (
            ground_truth.get("corruptions_summary")
            if isinstance(ground_truth.get("corruptions_summary"), dict)
            else {}
        )
        rfi_ch = list(csum.get("rfi_channels", []))
        synth_tmp_dir = _materialize_sky_model_vis_npz(
            ws / "input.ms",
            ground_truth["sky_model"],
            ec.roles.targets,
            ec.roles.dead_antennas,
            rfi_ch,
            resolved_scratch,
        )
        ref_dir = synth_tmp_dir

    orig_ext_runner = _cal_ext._run_verifier_tclean_recipe
    try:
        if _run_verifier_tclean_recipe is not orig_ext_runner:
            _cal_ext._run_verifier_tclean_recipe = _run_verifier_tclean_recipe
        return run_verification(
            workspace_dir=ws,
            tests_dir=(ref_dir if (ref_dir is not None and ref_dir.is_dir()) else ws),
            log_dir=None,
            extract_config=ec,
            rubric=rb,
            ground_truth=ground_truth,
            ref_arrays_dir=ref_dir,
            scratch_root=resolved_scratch,
            compute_raw_hash=(ws / "input.ms" / "table.dat").is_file(),
        )
    finally:
        _cal_ext._run_verifier_tclean_recipe = orig_ext_runner
        if synth_tmp_dir is not None:
            shutil.rmtree(synth_tmp_dir, ignore_errors=True)


__all__ = [
    "PASS_REWARD_MIN_DEFAULT",
    "_parse_sky_coord_rad",
    "_run_verifier_tclean_recipe",
    "grade_calibrate",
    "has_active_table_lock",
    "resolve_scratch_root",
]
