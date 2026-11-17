"""Scientific reference-checkpoint diff across visibilities, caltables, and continuum images."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import tomllib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.analysis.collect import (
    _read_json_dict,
    _resolve_task_name,
    discover_trial_dirs,
    parse_task_variant_with_pack,
    sanitize_json_tree,
)
from radio_bench.config import (
    add_config_cli_args,
    config_from_cli_args,
    load_bench_config,
    prepare_runtime_env,
    reject_slash_tmp,
)
from radio_bench.grading.extractors.common import (
    DiscoveredCaltable,
    _compute_gain_tr_ref_log_jump,
    discover_caltables_by_viscal,
    discover_fluxscale_from_caltables,
    has_active_table_lock,
    open_ro_table,
    parse_sky_coord_rad,
    read_field_intents,
    resolve_scratch_root,
)
from radio_bench.grading.extractors.image import (
    DiscoveredImageSet,
    discover_image_sets,
    read_image_2d,
    sky_to_pix,
)
from radio_bench.grading.schema import ExtractConfig, SuiteManifest

_PARALLEL_CORR_TYPES: frozenset[int] = frozenset({1, 5, 8, 9, 12})
_CROSS_CORR_TYPES: frozenset[int] = frozenset({6, 7, 10, 11})
_MAX_MEDIAN_SAMPLES = 500_000

_VIS_KEY_DTYPE = np.dtype(
    [
        ("sc", "<i4"),
        ("a1", "<i4"),
        ("a2", "<i4"),
        ("tm", "<i8"),
        ("occ", "<i4"),
    ]
)


def _ensure_casa_env() -> None:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    if not os.environ.get("CASASITECONFIG", "").strip():
        etc_cfg = Path("/etc/casa/config.py")
        if etc_cfg.is_file():
            os.environ["CASASITECONFIG"] = str(etc_cfg)
        else:
            with contextlib.suppress(Exception):
                host_cfg = load_bench_config().casaconfig_path
                if host_cfg.is_file():
                    os.environ["CASASITECONFIG"] = str(host_cfg)


def _atomic_write_json(target_path: Path, payload: dict[str, Any]) -> None:
    reject_slash_tmp(target_path, label="output_path")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize_json_tree(payload)
    serialized = json.dumps(clean, indent=2, allow_nan=False) + "\n"
    tmp_path = target_path.parent / f".{target_path.name}.{os.getpid()}.tmp"
    try:
        tmp_path.write_text(serialized, encoding="utf-8")
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def resolve_workspace_dir(path: Path) -> Path:
    """Resolve either a raw `/workspace` directory or a Harbor `<trial_dir>` to its workspace."""
    p = Path(path).resolve()
    if (p / "artifacts" / "workspace").is_dir():
        return p / "artifacts" / "workspace"
    steps_dir = p / "steps"
    if steps_dir.is_dir():
        with contextlib.suppress(OSError):
            step_dirs = sorted(d for d in steps_dir.iterdir() if d.is_dir())
            for sdir in reversed(step_dirs):
                cand = sdir / "artifacts" / "workspace"
                if cand.is_dir():
                    return cand
    return p


def _infer_trial_meta(agent_path: Path) -> tuple[str | None, str | None, str | None]:
    """Infer `(base_task, stage, suite)` when `agent_path` is a Harbor `<trial_dir>`."""
    p = Path(agent_path).resolve()
    res = _read_json_dict(p / "result.json")
    cfg = _read_json_dict(p / "config.json")
    metrics = _read_json_dict(p / "verifier" / "metrics.json")
    if res is None and cfg is None and metrics is None:
        return None, None, None
    tname = _resolve_task_name(p, res, cfg)
    m_stage = (
        str(metrics.get("stage"))
        if isinstance(metrics, dict) and isinstance(metrics.get("stage"), str)
        else None
    )
    m_task = (
        str(metrics.get("task_id")).strip()
        if isinstance(metrics, dict) and isinstance(metrics.get("task_id"), str)
        else None
    )
    m_suite = (
        str(metrics.get("suite_id")).strip()
        if isinstance(metrics, dict) and isinstance(metrics.get("suite_id"), str)
        else None
    )
    base_task, _pack, _arm, suite, stage, _is_e2e = parse_task_variant_with_pack(
        tname, metrics_stage=m_stage
    )
    eff_bt = m_task if (not base_task or base_task == "unknown") and m_task else base_task
    eff_stage = stage if stage != "unknown" else m_stage
    eff_suite = suite if suite != "unknown" else m_suite
    return eff_bt, eff_stage, eff_suite


def _find_primary_ms(directory: Path) -> Path | None:
    if (
        directory.is_dir()
        and directory.name.lower().endswith(".ms")
        and (directory / "table.dat").is_file()
    ):
        return directory
    cand_in = directory / "input.ms"
    if (cand_in / "table.dat").is_file():
        return cand_in
    cand_cal = directory / "calibrators.ms"
    if (cand_cal / "table.dat").is_file():
        return cand_cal
    if not directory.is_dir():
        return None
    with contextlib.suppress(OSError):
        for child in sorted(directory.iterdir(), key=lambda x: x.name):
            if (
                child.is_dir()
                and child.name.lower().endswith(".ms")
                and (child / "table.dat").is_file()
            ):
                return child
    return None


def _load_all_suite_manifests() -> list[SuiteManifest]:
    try:
        cfg = load_bench_config()
        suites_dir = cfg.repo_root / "suites"
        if not suites_dir.is_dir():
            return []
        suite_tomls = sorted(suites_dir.glob("*/suite.toml"))
    except Exception:
        return []

    manifests: list[SuiteManifest] = []
    for st_path in suite_tomls:
        with contextlib.suppress(Exception):
            manifests.append(
                SuiteManifest.model_validate(tomllib.loads(st_path.read_text(encoding="utf-8")))
            )
    return manifests


def _resolve_suite_manifest(
    *,
    base_task: str | None = None,
    suite: str | None = None,
) -> SuiteManifest | None:
    manifests = _load_all_suite_manifests()
    if base_task:
        for mf in manifests:
            if base_task in mf.tasks:
                return mf
    if suite and suite != "unknown":
        s_low = suite.strip().lower()
        for mf in manifests:
            alias = (mf.suite.legacy_suite_alias or "").strip().lower()
            if mf.suite.suite_id.strip().lower() == s_low or (alias and alias == s_low):
                return mf
    return None


def _lookup_suite_checkpoint_dirs(
    ref_root: Path,
    stage: str | None,
    *,
    base_task: str | None = None,
    suite: str | None = None,
) -> tuple[Path | None, Path | None, SuiteManifest | None]:
    """Check `suites/*/suite.toml` checkpoint entries filtered by `base_task` / `suite`."""
    all_manifests = _load_all_suite_manifests()
    if not all_manifests:
        return None, None, None

    matched_mf = _resolve_suite_manifest(base_task=base_task, suite=suite)
    if base_task or (suite and suite != "unknown"):
        manifests = [matched_mf] if matched_mf is not None else []
    else:
        manifests = all_manifests

    eff_stage = (stage or "").strip().lower()
    wanted_suffix = "post_precal" if eff_stage == "precal" else "post_calibrate"

    for manifest in manifests:
        for ck_key, ck_entry in manifest.checkpoints.entries.items():
            if not ck_key.lower().endswith(wanted_suffix):
                continue
            rel = Path(ck_entry.input_ms)
            candidates = [ref_root / rel]
            if len(rel.parts) >= 2:
                candidates.append(ref_root / Path(*rel.parts[1:]))
            if len(rel.parts) >= 3:
                candidates.append(ref_root / Path(*rel.parts[2:]))
            for c_ms in candidates:
                if (c_ms / "table.dat").is_file():
                    return c_ms.parent, None, manifest
    return None, None, matched_mf


def _resolve_reference_layout(
    ref_dir: Path,
    *,
    base_task: str | None = None,
    stage: str | None = None,
    suite: str | None = None,
) -> tuple[Path, Path, Path, SuiteManifest | None]:
    """Resolve `(ms_ref_dir, caltable_ref_dir, image_ref_dir, manifest)` from checkpoint or job dir."""
    r_base = resolve_workspace_dir(Path(ref_dir).resolve())
    matched_mf = _resolve_suite_manifest(base_task=base_task, suite=suite)

    # If `ref_dir` is a Harbor job directory containing trial subdirs, match `base_task`
    if base_task and r_base.is_dir() and _find_primary_ms(r_base) is None:
        with contextlib.suppress(Exception):
            for _jdir, tdir in discover_trial_dirs([r_base]):
                t_bt, _, _ = _infer_trial_meta(tdir)
                if t_bt == base_task:
                    ws = resolve_workspace_dir(tdir)
                    return ws, ws, ws, matched_mf

    eff_stage = (stage or "").strip().lower()
    ms_dir = r_base
    cal_dir = r_base
    img_dir = r_base

    if _find_primary_ms(r_base) is None:
        if eff_stage == "precal" and (r_base / "post_precal").is_dir():
            ms_dir = r_base / "post_precal"
            cal_dir = ms_dir
        elif (r_base / "post_calibrate").is_dir():
            ms_dir = r_base / "post_calibrate"
            cal_dir = ms_dir
        else:
            ck_ms_dir, _, ck_mf = _lookup_suite_checkpoint_dirs(
                r_base, eff_stage, base_task=base_task, suite=suite
            )
            if ck_mf is not None and matched_mf is None:
                matched_mf = ck_mf
            if ck_ms_dir is not None:
                ms_dir = ck_ms_dir
                cal_dir = ck_ms_dir

    if not discover_image_sets(img_dir, max_depth=2):
        for cand_img in (ms_dir.parent / "reference_image", r_base / "reference_image"):
            if cand_img.is_dir() and discover_image_sets(cand_img, max_depth=2):
                img_dir = cand_img
                break
        else:
            for cand_img in (ms_dir.parent / "reference_image", r_base / "reference_image"):
                if cand_img.is_dir():
                    img_dir = cand_img
                    break

    return ms_dir, cal_dir, img_dir, matched_mf


# ==============================================================================
# 1. Visibility-Domain Composite-Key Diff
# ==============================================================================


@dataclass
class _VisStatsAccumulator:
    matched_rows: int = 0
    both_flagged: int = 0
    union_flagged: int = 0
    ref_flagged: int = 0
    ref_unflagged: int = 0
    excess_flagged: int = 0
    par_common_unflagged: int = 0
    par_sum_sq_diff: float = 0.0
    par_ref_amp_samples: list[np.ndarray] = field(default_factory=list)
    par_ref_amp_count: int = 0
    par_phase_count: int = 0
    par_sum_sq_dphi_deg: float = 0.0
    cross_common_unflagged: int = 0
    cross_sum_sq_diff: float = 0.0
    cross_ref_amp_samples: list[np.ndarray] = field(default_factory=list)
    cross_ref_amp_count: int = 0
    cross_phase_count: int = 0
    cross_sum_sq_dphi_deg: float = 0.0

    def add_sample_amps(self, amps: np.ndarray, *, cross: bool = False) -> None:
        if amps.size == 0:
            return
        target_list = self.cross_ref_amp_samples if cross else self.par_ref_amp_samples
        cur_cnt = self.cross_ref_amp_count if cross else self.par_ref_amp_count
        rem = _MAX_MEDIAN_SAMPLES - cur_cnt
        if rem <= 0:
            return
        if amps.size <= rem:
            taken = amps.copy()
        else:
            step = max(1, amps.size // rem)
            taken = amps[::step][:rem].copy()
        target_list.append(taken)
        if cross:
            self.cross_ref_amp_count += int(taken.size)
        else:
            self.par_ref_amp_count += int(taken.size)

    def to_dict(self) -> dict[str, Any]:
        if self.matched_rows == 0:
            jaccard: float | None = None
            recall: float | None = None
            excess: float | None = None
        else:
            jaccard = (
                1.0 if self.union_flagged == 0 else float(self.both_flagged / self.union_flagged)
            )
            recall = 1.0 if self.ref_flagged == 0 else float(self.both_flagged / self.ref_flagged)
            excess = (
                0.0 if self.ref_unflagged == 0 else float(self.excess_flagged / self.ref_unflagged)
            )

        par_nrmse: float | None = None
        par_med_ref: float | None = None
        if self.par_common_unflagged > 0 and self.par_ref_amp_samples:
            all_ref = np.concatenate(self.par_ref_amp_samples)
            if all_ref.size > 0:
                par_med_ref = float(np.median(all_ref))
                par_rms = math.sqrt(self.par_sum_sq_diff / self.par_common_unflagged)
                if par_med_ref > 1e-15:
                    par_nrmse = float(par_rms / par_med_ref)
                elif par_rms <= 1e-15:
                    par_nrmse = 0.0

        par_phs_rms: float | None = None
        if self.par_phase_count > 0:
            par_phs_rms = float(math.sqrt(self.par_sum_sq_dphi_deg / self.par_phase_count))

        cross_nrmse: float | None = None
        if self.cross_common_unflagged > 0 and self.cross_ref_amp_samples:
            c_ref = np.concatenate(self.cross_ref_amp_samples)
            if c_ref.size > 0:
                c_med = float(np.median(c_ref))
                c_rms = math.sqrt(self.cross_sum_sq_diff / self.cross_common_unflagged)
                if c_med > 1e-15:
                    cross_nrmse = float(c_rms / c_med)
                elif c_rms <= 1e-15:
                    cross_nrmse = 0.0

        cross_phs_rms: float | None = None
        if self.cross_phase_count > 0:
            cross_phs_rms = float(math.sqrt(self.cross_sum_sq_dphi_deg / self.cross_phase_count))

        out: dict[str, Any] = {
            "matched_rows": self.matched_rows,
            "flag_jaccard": round(jaccard, 6) if jaccard is not None else None,
            "flag_recall_vs_ref": round(recall, 6) if recall is not None else None,
            "excess_flag_fraction": round(excess, 6) if excess is not None else None,
            "parallel_hand_common_unflagged": self.par_common_unflagged,
            "parallel_hand_ref_median_amp_jy": (
                round(par_med_ref, 6) if par_med_ref is not None else None
            ),
            "parallel_hand_nrmse": round(par_nrmse, 8) if par_nrmse is not None else None,
            "parallel_hand_phase_rms_deg": (
                round(par_phs_rms, 6) if par_phs_rms is not None else None
            ),
        }
        if self.cross_common_unflagged > 0:
            out["cross_hand_nrmse"] = round(cross_nrmse, 8) if cross_nrmse is not None else None
            out["cross_hand_phase_rms_deg"] = (
                round(cross_phs_rms, 6) if cross_phs_rms is not None else None
            )
        return out


def _read_ms_semantic_metadata(ms_path: Path) -> dict[str, Any]:
    with open_ro_table(ms_path / "FIELD") as tb_f:
        field_names = [str(x).strip() for x in tb_f.getcol("NAME")]
    with open_ro_table(ms_path / "ANTENNA") as tb_a:
        ant_names = [str(x).strip() for x in tb_a.getcol("NAME")]

    dd_path = ms_path / "DATA_DESCRIPTION"
    ddid_to_spw: dict[int, int] = {0: 0}
    ddid_to_pol: dict[int, int] = {0: 0}
    if (dd_path / "table.dat").is_file():
        with open_ro_table(dd_path) as tb_dd:
            if int(tb_dd.nrows()) > 0:
                cols = set(tb_dd.colnames())
                if "SPECTRAL_WINDOW_ID" in cols:
                    s_ids = np.asarray(tb_dd.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
                    ddid_to_spw = {i: int(v) for i, v in enumerate(s_ids.tolist())}
                if "POLARIZATION_ID" in cols:
                    p_ids = np.asarray(tb_dd.getcol("POLARIZATION_ID"), dtype=int)
                    ddid_to_pol = {i: int(v) for i, v in enumerate(p_ids.tolist())}

    pol_corr_types: dict[int, list[int]] = {}
    pol_path = ms_path / "POLARIZATION"
    if (pol_path / "table.dat").is_file():
        with open_ro_table(pol_path) as tb_p:
            if "CORR_TYPE" in tb_p.colnames():
                for pid in range(int(tb_p.nrows())):
                    ct = np.asarray(tb_p.getcell("CORR_TYPE", pid), dtype=int).ravel().tolist()
                    pol_corr_types[pid] = [int(x) for x in ct]

    spw_sigs: dict[int, tuple[float, int]] = {}
    spw_freqs_by_id: dict[int, np.ndarray] = {}
    spw_path = ms_path / "SPECTRAL_WINDOW"
    if (spw_path / "table.dat").is_file():
        with open_ro_table(spw_path) as tb_spw:
            for sid in range(int(tb_spw.nrows())):
                cf = np.asarray(tb_spw.getcell("CHAN_FREQ", sid), dtype=np.float64).ravel()
                if cf.size > 0:
                    spw_sigs[sid] = (round(float(cf[0]), 1), int(cf.size))
                    spw_freqs_by_id[sid] = cf.copy()

    ddid_by_spw_sig: dict[tuple[float, int], list[int]] = {}
    spw_chan_freqs_by_sig: dict[tuple[float, int], np.ndarray] = {}
    ddid_corr_types: dict[int, list[int]] = {}
    for ddid, sid in sorted(ddid_to_spw.items()):
        sig = spw_sigs.get(sid, (0.0, 0))
        ddid_by_spw_sig.setdefault(sig, []).append(ddid)
        if sid in spw_freqs_by_id and sig not in spw_chan_freqs_by_sig:
            spw_chan_freqs_by_sig[sig] = spw_freqs_by_id[sid]
        pid = ddid_to_pol.get(ddid, 0)
        ddid_corr_types[ddid] = pol_corr_types.get(pid, [])

    return {
        "field_names": field_names,
        "ant_names": ant_names,
        "ddid_by_spw_sig": ddid_by_spw_sig,
        "spw_chan_freqs_by_sig": spw_chan_freqs_by_sig,
        "ddid_corr_types": ddid_corr_types,
    }


def _match_spw_signatures(
    sigs_a: dict[tuple[float, int], np.ndarray],
    sigs_r: dict[tuple[float, int], np.ndarray],
) -> list[tuple[tuple[float, int], tuple[float, int], np.ndarray | None, np.ndarray | None]]:
    """Pair SpW signatures exactly first, falling back to overlapping channel-frequency grids."""
    exact = sorted(set(sigs_r.keys()) & set(sigs_a.keys()))
    pairs: list[
        tuple[tuple[float, int], tuple[float, int], np.ndarray | None, np.ndarray | None]
    ] = [(s, s, None, None) for s in exact]

    rem_a = [s for s in sorted(sigs_a.keys()) if s not in exact]
    rem_r = [s for s in sorted(sigs_r.keys()) if s not in exact]
    used_r: set[tuple[float, int]] = set()

    for sa in rem_a:
        cfa = sigs_a.get(sa)
        if cfa is None or cfa.size == 0:
            continue
        best_sr: tuple[float, int] | None = None
        best_ia: np.ndarray | None = None
        best_ir: np.ndarray | None = None
        best_n = 0
        min_a, max_a = float(np.min(cfa)), float(np.max(cfa))
        r_cfa = np.round(cfa, 1)

        for sr in rem_r:
            if sr in used_r:
                continue
            cfr = sigs_r.get(sr)
            if cfr is None or cfr.size == 0:
                continue
            min_r, max_r = float(np.min(cfr)), float(np.max(cfr))
            if max(min_a, min_r) > min(max_a, max_r) + 1.0:
                continue
            _vals, ia, ir = np.intersect1d(r_cfa, np.round(cfr, 1), return_indices=True)
            if int(ia.size) > best_n:
                best_n = int(ia.size)
                best_sr = sr
                order_i = np.argsort(ia)
                best_ia = ia[order_i]
                best_ir = ir[order_i]
            elif best_n == 0 and best_sr is None:
                # Overlapping single-channel or rebinned SpW fallback for K/G single-chan tables
                best_sr = sr
                best_ia = None
                best_ir = None

        if best_sr is not None:
            used_r.add(best_sr)
            pairs.append((sa, best_sr, best_ia, best_ir))

    return pairs


def _resolve_pol_hand_indices(npol: int, corr_types: list[int]) -> tuple[list[int], list[int]]:
    if corr_types and len(corr_types) == npol:
        par = [i for i, c in enumerate(corr_types) if c in _PARALLEL_CORR_TYPES]
        crs = [i for i, c in enumerate(corr_types) if c in _CROSS_CORR_TYPES]
        if par:
            return par, crs
    if npol >= 4:
        return [0, npol - 1], [1, 2]
    if npol >= 2:
        return [0, npol - 1], []
    return [0], []


def _classify_field_roles(
    ms_agt: Path,
    ms_ref: Path,
    extract_config: ExtractConfig | None = None,
) -> dict[str, set[str]]:
    """Map `FIELD_NAME.upper() -> set of roles` in `{'flux_cal', 'bandpass_cal', 'phase_cal', 'target'}`."""
    roles_by_upper: dict[str, set[str]] = {}

    if extract_config is not None:
        r = extract_config.roles
        for fn in r.flux_cal:
            roles_by_upper.setdefault(fn.strip().upper(), set()).add("flux_cal")
        for fn in r.bandpass_cal:
            roles_by_upper.setdefault(fn.strip().upper(), set()).add("bandpass_cal")
        for fn in r.phase_cal:
            roles_by_upper.setdefault(fn.strip().upper(), set()).add("phase_cal")
        for fn in r.targets:
            roles_by_upper.setdefault(fn.strip().upper(), set()).add("target")
        if roles_by_upper:
            return roles_by_upper

    for ms_cand in (ms_ref, ms_agt):
        with contextlib.suppress(Exception):
            _fnames, intents_map = read_field_intents(ms_cand)
            for fname, intents in intents_map.items():
                fu = fname.strip().upper()
                joined = " ".join(intents).upper()
                if "CALIBRATE_FLUX" in joined:
                    roles_by_upper.setdefault(fu, set()).add("flux_cal")
                if "CALIBRATE_BANDPASS" in joined:
                    roles_by_upper.setdefault(fu, set()).add("bandpass_cal")
                if "CALIBRATE_PHASE" in joined:
                    roles_by_upper.setdefault(fu, set()).add("phase_cal")
                if "OBSERVE_TARGET" in joined and "CALIBRATE_PHASE" not in joined:
                    roles_by_upper.setdefault(fu, set()).add("target")
            if roles_by_upper:
                break

    return roles_by_upper


def _load_spw_field_slice(
    tb: Any,
    *,
    ddids: list[int],
    fid: int,
    vis_col: str,
    has_scan_col: bool,
    ant_id_to_canon: np.ndarray,
    shared_t0: float,
    corr_types: list[int] | None = None,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Stream one `(spw_sig, field_id)` slice in `50,000`-row chunks and build sorted composite keys."""
    vis_parts: list[np.ndarray] = []
    flg_parts: list[np.ndarray] = []
    a1_parts: list[np.ndarray] = []
    a2_parts: list[np.ndarray] = []
    tm_parts: list[np.ndarray] = []
    sc_parts: list[np.ndarray] = []

    for ddid in ddids:
        q = f"DATA_DESC_ID == {ddid} && FIELD_ID == {fid} && ANTENNA1 != ANTENNA2"
        sub = tb.query(q)
        try:
            sub_nrows = int(sub.nrows())
            if sub_nrows == 0:
                continue
            for sr in range(0, sub_nrows, chunk_size):
                nr = min(chunk_size, sub_nrows - sr)
                v_c = np.asarray(sub.getcol(vis_col, startrow=sr, nrow=nr))
                f_c = np.asarray(sub.getcol("FLAG", startrow=sr, nrow=nr), dtype=bool)
                fr_c = np.asarray(sub.getcol("FLAG_ROW", startrow=sr, nrow=nr), dtype=bool)
                eff_f = f_c | fr_c[np.newaxis, np.newaxis, :]
                a1_c = np.asarray(sub.getcol("ANTENNA1", startrow=sr, nrow=nr), dtype=np.int64)
                a2_c = np.asarray(sub.getcol("ANTENNA2", startrow=sr, nrow=nr), dtype=np.int64)
                tm_c = np.asarray(sub.getcol("TIME", startrow=sr, nrow=nr), dtype=np.float64)
                sc_c = (
                    np.asarray(sub.getcol("SCAN_NUMBER", startrow=sr, nrow=nr), dtype=np.int32)
                    if has_scan_col
                    else np.zeros(nr, dtype=np.int32)
                )
                vis_parts.append(v_c)
                flg_parts.append(eff_f)
                a1_parts.append(a1_c)
                a2_parts.append(a2_c)
                tm_parts.append(tm_c)
                sc_parts.append(sc_c)
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()

    if not vis_parts:
        return None

    vis = np.concatenate(vis_parts, axis=2) if len(vis_parts) > 1 else vis_parts[0].copy()
    flg = np.concatenate(flg_parts, axis=2) if len(flg_parts) > 1 else flg_parts[0].copy()
    a1_raw = np.concatenate(a1_parts, axis=0) if len(a1_parts) > 1 else a1_parts[0]
    a2_raw = np.concatenate(a2_parts, axis=0) if len(a2_parts) > 1 else a2_parts[0]
    tm_raw = np.concatenate(tm_parts, axis=0) if len(tm_parts) > 1 else tm_parts[0]
    sc_raw = np.concatenate(sc_parts, axis=0) if len(sc_parts) > 1 else sc_parts[0]

    a1_m = ant_id_to_canon[a1_raw]
    a2_m = ant_id_to_canon[a2_raw]
    flip = a1_m > a2_m
    if np.any(flip):
        vis[:, :, flip] = np.conj(vis[:, :, flip])
        has_cross_12 = bool(
            vis.shape[0] == 4
            and corr_types
            and len(corr_types) == 4
            and corr_types[1] in _CROSS_CORR_TYPES
            and corr_types[2] in _CROSS_CORR_TYPES
        )
        if has_cross_12:
            vis[[1, 2], :, :] = np.where(
                flip[np.newaxis, np.newaxis, :],
                vis[[2, 1], :, :],
                vis[[1, 2], :, :],
            )
            flg[[1, 2], :, :] = np.where(
                flip[np.newaxis, np.newaxis, :],
                flg[[2, 1], :, :],
                flg[[1, 2], :, :],
            )

    min_a = np.minimum(a1_m, a2_m).astype(np.int32)
    max_a = np.maximum(a1_m, a2_m).astype(np.int32)
    tm_ms = np.rint((tm_raw - shared_t0) * 1000.0).astype(np.int64)

    n = int(tm_ms.size)
    order = np.lexsort((tm_ms, max_a, min_a, sc_raw))
    sc_s = sc_raw[order]
    a1_s = min_a[order]
    a2_s = max_a[order]
    tm_s = tm_ms[order]

    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    if n > 1:
        is_new[1:] = (
            (sc_s[1:] != sc_s[:-1])
            | (a1_s[1:] != a1_s[:-1])
            | (a2_s[1:] != a2_s[:-1])
            | (tm_s[1:] != tm_s[:-1])
        )
    idx_arr = np.arange(n, dtype=np.int64)
    occ_s = (idx_arr - np.maximum.accumulate(np.where(is_new, idx_arr, 0))).astype(np.int32)

    keys_sorted = np.empty(n, dtype=_VIS_KEY_DTYPE)
    keys_sorted["sc"] = sc_s
    keys_sorted["a1"] = a1_s
    keys_sorted["a2"] = a2_s
    keys_sorted["tm"] = tm_s
    keys_sorted["occ"] = occ_s

    return keys_sorted, order, vis, flg


def _accumulate_matched_vis_slice(
    accs: list[_VisStatsAccumulator],
    v_agt: np.ndarray,
    f_agt: np.ndarray,
    v_ref: np.ndarray,
    f_ref: np.ndarray,
    par_idx: list[int],
    cross_idx: list[int],
) -> None:
    n_rows = int(v_agt.shape[2])
    both_f = int(np.sum(f_agt & f_ref))
    union_f = int(np.sum(f_agt | f_ref))
    ref_f = int(np.sum(f_ref))
    ref_uf = int(np.sum(~f_ref))
    exc_f = int(np.sum(f_agt & (~f_ref)))

    va_p = v_agt[par_idx, :, :]
    vr_p = v_ref[par_idx, :, :]
    fa_p = f_agt[par_idx, :, :]
    fr_p = f_ref[par_idx, :, :]
    good_p = (~fa_p) & (~fr_p) & np.isfinite(va_p) & np.isfinite(vr_p)
    n_good_p = int(np.sum(good_p))

    sq_diff_p = 0.0
    ref_amps_p = np.empty(0, dtype=np.float64)
    n_phs_p = 0
    sq_dphi_p = 0.0
    if n_good_p > 0:
        va_g = va_p[good_p].astype(np.complex128)
        vr_g = vr_p[good_p].astype(np.complex128)
        diff = va_g - vr_g
        sq_diff_p = float(np.sum(diff.real**2 + diff.imag**2))
        ref_amps_p = np.abs(vr_g)
        nz = (np.abs(va_g) > 1e-12) & (ref_amps_p > 1e-12)
        if np.any(nz):
            dphi = np.rad2deg(np.angle(va_g[nz] * np.conj(vr_g[nz])))
            n_phs_p = int(dphi.size)
            sq_dphi_p = float(np.sum(dphi**2))

    n_good_c = 0
    sq_diff_c = 0.0
    ref_amps_c = np.empty(0, dtype=np.float64)
    n_phs_c = 0
    sq_dphi_c = 0.0
    if cross_idx and max(cross_idx) < v_agt.shape[0] and max(cross_idx) < v_ref.shape[0]:
        va_c = v_agt[cross_idx, :, :]
        vr_c = v_ref[cross_idx, :, :]
        fa_c = f_agt[cross_idx, :, :]
        fr_c = f_ref[cross_idx, :, :]
        good_c = (~fa_c) & (~fr_c) & np.isfinite(va_c) & np.isfinite(vr_c)
        n_good_c = int(np.sum(good_c))
        if n_good_c > 0:
            vac_g = va_c[good_c].astype(np.complex128)
            vrc_g = vr_c[good_c].astype(np.complex128)
            cdiff = vac_g - vrc_g
            sq_diff_c = float(np.sum(cdiff.real**2 + cdiff.imag**2))
            ref_amps_c = np.abs(vrc_g)
            nz_c = (np.abs(vac_g) > 1e-12) & (ref_amps_c > 1e-12)
            if np.any(nz_c):
                dphi_c = np.rad2deg(np.angle(vac_g[nz_c] * np.conj(vrc_g[nz_c])))
                n_phs_c = int(dphi_c.size)
                sq_dphi_c = float(np.sum(dphi_c**2))

    for acc in accs:
        acc.matched_rows += n_rows
        acc.both_flagged += both_f
        acc.union_flagged += union_f
        acc.ref_flagged += ref_f
        acc.ref_unflagged += ref_uf
        acc.excess_flagged += exc_f
        acc.par_common_unflagged += n_good_p
        acc.par_sum_sq_diff += sq_diff_p
        acc.add_sample_amps(ref_amps_p, cross=False)
        acc.par_phase_count += n_phs_p
        acc.par_sum_sq_dphi_deg += sq_dphi_p
        acc.cross_common_unflagged += n_good_c
        acc.cross_sum_sq_diff += sq_diff_c
        acc.add_sample_amps(ref_amps_c, cross=True)
        acc.cross_phase_count += n_phs_c
        acc.cross_sum_sq_dphi_deg += sq_dphi_c


def _field_has_distinct_corrected_data(tb: Any, ddids: list[int], fid: int) -> bool:
    """Return True if `tb` has `CORRECTED_DATA != DATA` on `(ddids, fid)`."""
    cols = set(tb.colnames())
    if "CORRECTED_DATA" not in cols or "DATA" not in cols:
        return False
    for ddid in ddids:
        sub = tb.query(f"DATA_DESC_ID == {ddid} && FIELD_ID == {fid} && ANTENNA1 != ANTENNA2")
        try:
            nr = min(512, int(sub.nrows()))
            if nr <= 0:
                continue
            d_c = np.asarray(sub.getcol("DATA", startrow=0, nrow=nr), dtype=np.complex128)
            c_c = np.asarray(sub.getcol("CORRECTED_DATA", startrow=0, nrow=nr), dtype=np.complex128)
            f_c = np.asarray(sub.getcol("FLAG", startrow=0, nrow=nr), dtype=bool)
            fr_c = np.asarray(sub.getcol("FLAG_ROW", startrow=0, nrow=nr), dtype=bool)
            uf = (~(f_c | fr_c[np.newaxis, np.newaxis, :])) & np.isfinite(d_c) & np.isfinite(c_c)
            if np.any(uf):
                if float(np.max(np.abs(c_c[uf] - d_c[uf]))) > 1e-6:
                    return True
            else:
                fin = np.isfinite(d_c) & np.isfinite(c_c)
                if np.any(fin) and float(np.max(np.abs(c_c[fin] - d_c[fin]))) > 1e-6:
                    return True
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()
    return False


def _merge_calibrator_flags_into_slice(
    main_slice: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cal_slice: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys_m, order_m, vis_m, flg_m = main_slice
    keys_c, order_c, _vis_c, flg_c = cal_slice
    if flg_m.shape[0] != flg_c.shape[0] or flg_m.shape[1] != flg_c.shape[1] or keys_c.size == 0:
        return main_slice
    pos = np.searchsorted(keys_c, keys_m)
    clamped = np.minimum(pos, keys_c.size - 1)
    hit = (pos < keys_c.size) & (keys_c[clamped] == keys_m)
    if not np.any(hit):
        return main_slice
    merged_flg = flg_m.copy()
    idx_m = order_m[hit]
    idx_c = order_c[pos[hit]]
    merged_flg[:, :, idx_m] |= flg_c[:, :, idx_c]
    return keys_m, order_m, vis_m, merged_flg


def diff_visibilities(
    ms_agt_path: Path,
    ms_ref_path: Path,
    *,
    stage: str | None = None,
    extract_config: ExtractConfig | None = None,
    cal_ms_agt_path: Path | None = None,
    cal_ms_ref_path: Path | None = None,
    chunk_size: int = 50_000,
) -> dict[str, Any]:
    """Compare agent and reference MS visibilities & flags via semantic composite keys."""
    _ensure_casa_env()
    ms_a = Path(ms_agt_path)
    ms_r = Path(ms_ref_path)
    reject_slash_tmp(ms_a, label="ms_agt_path")
    reject_slash_tmp(ms_r, label="ms_ref_path")
    if cal_ms_agt_path is not None:
        reject_slash_tmp(Path(cal_ms_agt_path), label="cal_ms_agt_path")
    if cal_ms_ref_path is not None:
        reject_slash_tmp(Path(cal_ms_ref_path), label="cal_ms_ref_path")

    if not (ms_a / "table.dat").is_file() or not (ms_r / "table.dat").is_file():
        return {"status": "missing_ms", "agent_ms": str(ms_a), "ref_ms": str(ms_r)}
    if has_active_table_lock(ms_a) or has_active_table_lock(ms_r):
        return {"status": "locked_or_unreadable", "agent_ms": str(ms_a), "ref_ms": str(ms_r)}

    eff_cal_a = Path(cal_ms_agt_path) if cal_ms_agt_path is not None else None
    if eff_cal_a is None and ms_a.name.lower() == "input.ms":
        cand_ca = ms_a.parent / "calibrators.ms"
        if (cand_ca / "table.dat").is_file() and not has_active_table_lock(cand_ca):
            eff_cal_a = cand_ca
    elif eff_cal_a is not None and (
        not (eff_cal_a / "table.dat").is_file() or has_active_table_lock(eff_cal_a)
    ):
        eff_cal_a = None

    eff_cal_r = Path(cal_ms_ref_path) if cal_ms_ref_path is not None else None
    if eff_cal_r is None and ms_r.name.lower() == "input.ms":
        cand_cr = ms_r.parent / "calibrators.ms"
        if (cand_cr / "table.dat").is_file() and not has_active_table_lock(cand_cr):
            eff_cal_r = cand_cr
    elif eff_cal_r is not None and (
        not (eff_cal_r / "table.dat").is_file() or has_active_table_lock(eff_cal_r)
    ):
        eff_cal_r = None

    meta_a = _read_ms_semantic_metadata(ms_a)
    meta_r = _read_ms_semantic_metadata(ms_r)
    meta_ca = _read_ms_semantic_metadata(eff_cal_a) if eff_cal_a is not None else None
    meta_cr = _read_ms_semantic_metadata(eff_cal_r) if eff_cal_r is not None else None
    roles_by_upper = _classify_field_roles(ms_a, ms_r, extract_config=extract_config)

    # Unified canonical antenna index map by lowercase antenna name
    extra_ants_ca = {a.lower() for a in meta_ca["ant_names"]} if meta_ca is not None else set()
    extra_ants_cr = {a.lower() for a in meta_cr["ant_names"]} if meta_cr is not None else set()
    all_ants = sorted(
        {a.lower() for a in meta_r["ant_names"]}
        | {a.lower() for a in meta_a["ant_names"]}
        | extra_ants_ca
        | extra_ants_cr
    )
    ant_to_idx = {nm: idx for idx, nm in enumerate(all_ants)}
    ant_map_a = np.array([ant_to_idx[a.lower()] for a in meta_a["ant_names"]], dtype=np.int32)
    ant_map_r = np.array([ant_to_idx[a.lower()] for a in meta_r["ant_names"]], dtype=np.int32)
    ant_map_ca = (
        np.array([ant_to_idx[a.lower()] for a in meta_ca["ant_names"]], dtype=np.int32)
        if meta_ca is not None
        else None
    )
    ant_map_cr = (
        np.array([ant_to_idx[a.lower()] for a in meta_cr["ant_names"]], dtype=np.int32)
        if meta_cr is not None
        else None
    )

    fields_a_upper = {nm.upper(): (idx, nm) for idx, nm in enumerate(meta_a["field_names"])}
    fields_r_upper = {nm.upper(): (idx, nm) for idx, nm in enumerate(meta_r["field_names"])}
    fields_ca_upper = (
        {nm.upper(): (idx, nm) for idx, nm in enumerate(meta_ca["field_names"])}
        if meta_ca is not None
        else {}
    )
    fields_cr_upper = (
        {nm.upper(): (idx, nm) for idx, nm in enumerate(meta_cr["field_names"])}
        if meta_cr is not None
        else {}
    )
    avail_a = set(fields_a_upper) | set(fields_ca_upper)
    avail_r = set(fields_r_upper) | set(fields_cr_upper)
    common_field_keys = [
        k
        for k in list(dict.fromkeys([*fields_r_upper.keys(), *fields_cr_upper.keys()]))
        if k in avail_a and k in avail_r
    ]

    sigs_a_all = dict(meta_a["spw_chan_freqs_by_sig"])
    if meta_ca is not None:
        for sg, cf in meta_ca["spw_chan_freqs_by_sig"].items():
            sigs_a_all.setdefault(sg, cf)
    sigs_r_all = dict(meta_r["spw_chan_freqs_by_sig"])
    if meta_cr is not None:
        for sg, cf in meta_cr["spw_chan_freqs_by_sig"].items():
            sigs_r_all.setdefault(sg, cf)
    matched_spws = _match_spw_signatures(sigs_a_all, sigs_r_all)

    overall_acc = _VisStatsAccumulator()
    by_field_accs: dict[str, _VisStatsAccumulator] = {}
    by_role_accs: dict[str, _VisStatsAccumulator] = {}

    eff_stage = (stage or "").strip().lower()

    with contextlib.ExitStack() as stack:
        tb_a = stack.enter_context(open_ro_table(ms_a))
        tb_r = stack.enter_context(open_ro_table(ms_r))
        tb_ca = stack.enter_context(open_ro_table(eff_cal_a)) if eff_cal_a is not None else None
        tb_cr = stack.enter_context(open_ro_table(eff_cal_r)) if eff_cal_r is not None else None

        nrows_a = int(tb_a.nrows())
        nrows_r = int(tb_r.nrows())
        cols_a = set(tb_a.colnames())
        cols_r = set(tb_r.colnames())
        cols_ca = set(tb_ca.colnames()) if tb_ca is not None else set()
        cols_cr = set(tb_cr.colnames()) if tb_cr is not None else set()

        has_corr_r = "CORRECTED_DATA" in cols_r or "CORRECTED_DATA" in cols_cr
        has_corr_a = "CORRECTED_DATA" in cols_a or "CORRECTED_DATA" in cols_ca

        if eff_stage == "precal" or not has_corr_r:
            col_a = "DATA"
            col_r = "DATA"
        elif has_corr_a:
            col_a = "CORRECTED_DATA"
            col_r = "CORRECTED_DATA"
        else:
            col_a = "DATA"
            col_r = "CORRECTED_DATA"

        if (col_a not in cols_a and col_a not in cols_ca) or (
            col_r not in cols_r and col_r not in cols_cr
        ):
            return {
                "status": "missing_visibility_column",
                "agent_ms": ms_a.name,
                "ref_ms": ms_r.name,
            }

        has_scan_a = "SCAN_NUMBER" in cols_a and (tb_ca is None or "SCAN_NUMBER" in cols_ca)
        has_scan_r = "SCAN_NUMBER" in cols_r and (tb_cr is None or "SCAN_NUMBER" in cols_cr)
        use_scan = has_scan_a and has_scan_r

        t0_r = float(tb_r.getcell("TIME", 0)) if nrows_r > 0 else 0.0
        t0_a = float(tb_a.getcell("TIME", 0)) if nrows_a > 0 else t0_r
        shared_t0 = min(t0_a, t0_r) if (nrows_a > 0 and nrows_r > 0) else (t0_r or t0_a)

        total_agt_cross_rows = 0
        total_ref_cross_rows = 0

        def _load_side_field_slice(
            *,
            tb_main: Any,
            meta_main: dict[str, Any],
            fields_main: dict[str, tuple[int, str]],
            ant_map_main: np.ndarray,
            cols_main: set[str],
            tb_cal: Any | None,
            meta_cal: dict[str, Any] | None,
            fields_cal: dict[str, tuple[int, str]],
            ant_map_cal: np.ndarray | None,
            cols_cal: set[str],
            spw_sig: tuple[float, int],
            f_upper: str,
            wanted_col: str,
        ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None, list[int]]:
            is_cal_field = bool(roles_by_upper.get(f_upper, set()) - {"target"}) or (
                f_upper in fields_cal and "target" not in roles_by_upper.get(f_upper, set())
            )
            main_slice = None
            main_ct: list[int] = []
            ddids_m = meta_main["ddid_by_spw_sig"].get(spw_sig, [])
            if ddids_m and f_upper in fields_main:
                fid_m, _ = fields_main[f_upper]
                eff_col_m = wanted_col if wanted_col in cols_main else "DATA"
                main_ct = meta_main["ddid_corr_types"].get(ddids_m[0], [])
                if eff_col_m in cols_main:
                    main_slice = _load_spw_field_slice(
                        tb_main,
                        ddids=ddids_m,
                        fid=fid_m,
                        vis_col=eff_col_m,
                        has_scan_col=use_scan,
                        ant_id_to_canon=ant_map_main,
                        shared_t0=shared_t0,
                        corr_types=main_ct,
                        chunk_size=chunk_size,
                    )

            if (
                not is_cal_field
                or tb_cal is None
                or meta_cal is None
                or ant_map_cal is None
                or f_upper not in fields_cal
            ):
                return main_slice, main_ct

            ddids_c = meta_cal["ddid_by_spw_sig"].get(spw_sig, [])
            if not ddids_c:
                # Match overlapping SpW inside calibrators.ms if trimmed
                for sg_c, cf_c in meta_cal["spw_chan_freqs_by_sig"].items():
                    cf_m = meta_main["spw_chan_freqs_by_sig"].get(spw_sig)
                    if (
                        cf_m is not None
                        and cf_m.size == cf_c.size
                        and np.allclose(cf_m, cf_c, atol=1.0)
                    ):
                        ddids_c = meta_cal["ddid_by_spw_sig"].get(sg_c, [])
                        break
            if not ddids_c:
                return main_slice, main_ct

            fid_c, _ = fields_cal[f_upper]
            cal_ct = meta_cal["ddid_corr_types"].get(ddids_c[0], [])
            eff_col_c = wanted_col if wanted_col in cols_cal else "DATA"
            if eff_col_c not in cols_cal:
                return main_slice, main_ct

            if wanted_col == "DATA":
                cal_slice = _load_spw_field_slice(
                    tb_cal,
                    ddids=ddids_c,
                    fid=fid_c,
                    vis_col="DATA",
                    has_scan_col=use_scan,
                    ant_id_to_canon=ant_map_cal,
                    shared_t0=shared_t0,
                    corr_types=cal_ct,
                    chunk_size=chunk_size,
                )
                if main_slice is None:
                    return cal_slice, cal_ct
                if cal_slice is None:
                    return main_slice, main_ct
                return _merge_calibrator_flags_into_slice(main_slice, cal_slice), main_ct

            # Calibrated stage (`wanted_col == "CORRECTED_DATA"`): prefer `calibrators.ms` when
            # `calibrators.ms` has calibrated `CORRECTED_DATA != DATA` and `input.ms` does not!
            cal_has_corr = _field_has_distinct_corrected_data(tb_cal, ddids_c, fid_c)
            main_has_corr = bool(
                ddids_m and f_upper in fields_main
            ) and _field_has_distinct_corrected_data(tb_main, ddids_m, fields_main[f_upper][0])
            if main_slice is None or (cal_has_corr and not main_has_corr):
                cal_slice = _load_spw_field_slice(
                    tb_cal,
                    ddids=ddids_c,
                    fid=fid_c,
                    vis_col=eff_col_c,
                    has_scan_col=use_scan,
                    ant_id_to_canon=ant_map_cal,
                    shared_t0=shared_t0,
                    corr_types=cal_ct,
                    chunk_size=chunk_size,
                )
                if cal_slice is not None:
                    return cal_slice, (cal_ct or main_ct)
            return main_slice, main_ct

        for sig_a, sig_r, chan_ia, chan_ir in matched_spws:
            for f_upper in common_field_keys:
                disp_r = (
                    fields_r_upper[f_upper][1]
                    if f_upper in fields_r_upper
                    else fields_cr_upper[f_upper][1]
                )
                slice_a, ct_a = _load_side_field_slice(
                    tb_main=tb_a,
                    meta_main=meta_a,
                    fields_main=fields_a_upper,
                    ant_map_main=ant_map_a,
                    cols_main=cols_a,
                    tb_cal=tb_ca,
                    meta_cal=meta_ca,
                    fields_cal=fields_ca_upper,
                    ant_map_cal=ant_map_ca,
                    cols_cal=cols_ca,
                    spw_sig=sig_a,
                    f_upper=f_upper,
                    wanted_col=col_a,
                )
                slice_r, ct_r = _load_side_field_slice(
                    tb_main=tb_r,
                    meta_main=meta_r,
                    fields_main=fields_r_upper,
                    ant_map_main=ant_map_r,
                    cols_main=cols_r,
                    tb_cal=tb_cr,
                    meta_cal=meta_cr,
                    fields_cal=fields_cr_upper,
                    ant_map_cal=ant_map_cr,
                    cols_cal=cols_cr,
                    spw_sig=sig_r,
                    f_upper=f_upper,
                    wanted_col=col_r,
                )
                if slice_a is not None:
                    total_agt_cross_rows += int(slice_a[0].size)
                if slice_r is not None:
                    total_ref_cross_rows += int(slice_r[0].size)
                if slice_a is None or slice_r is None:
                    continue

                keys_a, order_a, vis_a, flg_a = slice_a
                keys_r, order_r, vis_r, flg_r = slice_r

                pos = np.searchsorted(keys_r, keys_a)
                clamped_pos = np.minimum(pos, keys_r.size - 1)
                matched_mask = (pos < keys_r.size) & (keys_r[clamped_pos] == keys_a)
                if not np.any(matched_mask):
                    continue

                idx_a = order_a[matched_mask]
                idx_r = order_r[pos[matched_mask]]

                va_m = vis_a[:, :, idx_a]
                fa_m = flg_a[:, :, idx_a]
                vr_m = vis_r[:, :, idx_r]
                fr_m = flg_r[:, :, idx_r]

                if (
                    chan_ia is not None
                    and chan_ir is not None
                    and int(np.max(chan_ia)) < va_m.shape[1]
                    and int(np.max(chan_ir)) < vr_m.shape[1]
                ):
                    va_m = va_m[:, chan_ia, :]
                    fa_m = fa_m[:, chan_ia, :]
                    vr_m = vr_m[:, chan_ir, :]
                    fr_m = fr_m[:, chan_ir, :]

                if va_m.shape[0] != vr_m.shape[0] or va_m.shape[1] != vr_m.shape[1]:
                    continue

                par_idx, cross_idx = _resolve_pol_hand_indices(va_m.shape[0], ct_r or ct_a)
                f_acc = by_field_accs.setdefault(disp_r, _VisStatsAccumulator())
                target_accs = [overall_acc, f_acc]
                for rname in sorted(roles_by_upper.get(f_upper, ())):
                    target_accs.append(by_role_accs.setdefault(rname, _VisStatsAccumulator()))

                _accumulate_matched_vis_slice(
                    target_accs, va_m, fa_m, vr_m, fr_m, par_idx, cross_idx
                )

    vis_status = "no_common_rows" if overall_acc.matched_rows == 0 else "ok"
    return sanitize_json_tree(
        {
            "status": vis_status,
            "agent_ms": ms_a.name,
            "ref_ms": ms_r.name,
            "agent_calibrators_ms": eff_cal_a.name if eff_cal_a is not None else None,
            "ref_calibrators_ms": eff_cal_r.name if eff_cal_r is not None else None,
            "agent_column": col_a,
            "ref_column": col_r,
            "agent_total_rows": nrows_a,
            "ref_total_rows": nrows_r,
            "agent_cross_rows_in_common_spws": total_agt_cross_rows,
            "ref_cross_rows_in_common_spws": total_ref_cross_rows,
            "overall": overall_acc.to_dict(),
            "by_role": {k: v.to_dict() for k, v in sorted(by_role_accs.items())},
            "by_field": {k: v.to_dict() for k, v in sorted(by_field_accs.items())},
        }
    )


# ==============================================================================
# 2. Calibration-Table Gauge-Fixed Diff (`VisCal` Keyword Discovery)
# ==============================================================================


@dataclass
class _LoadedCaltable:
    path: Path
    role: str
    viscal: str
    is_complex: bool
    param: np.ndarray  # shape (npol, nchan, nrow)
    flag: np.ndarray  # shape (npol, nchan, nrow)
    ant_names_by_row: list[str]
    field_names_by_row: list[str]
    spw_sigs_by_row: list[tuple[float, int]]
    time_col: np.ndarray
    scan_col: np.ndarray
    field_id_col: np.ndarray
    ant_id_col: np.ndarray
    spw_chan_freqs_by_sig: dict[tuple[float, int], np.ndarray] = field(default_factory=dict)


def _caltable_sort_key(ct_path: Path) -> tuple[float, str]:
    td = ct_path / "table.dat"
    try:
        mt = float(td.stat().st_mtime)
    except OSError:
        mt = 0.0
    return (mt, ct_path.name)


def _load_caltable(ct: DiscoveredCaltable, fallback_ms: Path | None) -> _LoadedCaltable | None:
    p = ct.path
    if has_active_table_lock(p):
        return None

    ant_names: list[str] = []
    for ant_src in (p / "ANTENNA", fallback_ms / "ANTENNA" if fallback_ms else None):
        if ant_src is not None and (ant_src / "table.dat").is_file():
            with contextlib.suppress(Exception), open_ro_table(ant_src) as tb_a:
                if "NAME" in tb_a.colnames():
                    ant_names = [str(x).strip().lower() for x in tb_a.getcol("NAME")]
            if ant_names:
                break

    field_names: list[str] = []
    for fld_src in (p / "FIELD", fallback_ms / "FIELD" if fallback_ms else None):
        if fld_src is not None and (fld_src / "table.dat").is_file():
            with contextlib.suppress(Exception), open_ro_table(fld_src) as tb_f:
                if "NAME" in tb_f.colnames():
                    field_names = [str(x).strip().upper() for x in tb_f.getcol("NAME")]
            if field_names:
                break

    spw_sigs: dict[int, tuple[float, int]] = {}
    spw_chan_freqs_by_sig: dict[tuple[float, int], np.ndarray] = {}
    for spw_src in (
        p / "SPECTRAL_WINDOW",
        fallback_ms / "SPECTRAL_WINDOW" if fallback_ms else None,
    ):
        if spw_src is not None and (spw_src / "table.dat").is_file():
            with contextlib.suppress(Exception), open_ro_table(spw_src) as tb_s:
                if "CHAN_FREQ" in tb_s.colnames():
                    for sid in range(int(tb_s.nrows())):
                        cf = np.asarray(tb_s.getcell("CHAN_FREQ", sid), dtype=np.float64).ravel()
                        if cf.size > 0:
                            sg = (round(float(cf[0]), 1), int(cf.size))
                            spw_sigs[sid] = sg
                            spw_chan_freqs_by_sig.setdefault(sg, cf.copy())
            if spw_sigs:
                break

    try:
        with open_ro_table(p) as tb:
            nrows = int(tb.nrows())
            if nrows == 0:
                return None
            cols = set(tb.colnames())
            if "ANTENNA1" not in cols or "FLAG" not in cols:
                return None
            is_complex = "CPARAM" in cols
            if not is_complex and "FPARAM" not in cols:
                return None
            param = np.asarray(tb.getcol("CPARAM" if is_complex else "FPARAM"))
            flg = np.asarray(tb.getcol("FLAG"), dtype=bool)
            if param.ndim == 2:
                param = param[:, np.newaxis, :]
                flg = flg[:, np.newaxis, :]
            if param.ndim != 3 or param.shape != flg.shape:
                return None

            ant1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int64)
            fids = (
                np.asarray(tb.getcol("FIELD_ID"), dtype=np.int64)
                if "FIELD_ID" in cols
                else np.zeros(nrows, dtype=np.int64)
            )
            sids = (
                np.asarray(tb.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int64)
                if "SPECTRAL_WINDOW_ID" in cols
                else np.zeros(nrows, dtype=np.int64)
            )
            t_col = (
                np.asarray(tb.getcol("TIME"), dtype=np.float64)
                if "TIME" in cols
                else np.zeros(nrows, dtype=np.float64)
            )
            sc_col = (
                np.asarray(tb.getcol("SCAN_NUMBER"), dtype=np.int64)
                if "SCAN_NUMBER" in cols
                else np.zeros(nrows, dtype=np.int64)
            )
    except Exception:
        return None

    row_ants = [
        ant_names[int(a)] if 0 <= int(a) < len(ant_names) else f"ant_{int(a)}"
        for a in ant1.tolist()
    ]
    row_fields = [
        field_names[int(f)] if 0 <= int(f) < len(field_names) else f"FIELD_{int(f)}"
        for f in fids.tolist()
    ]
    nchan = int(param.shape[1])
    row_spws = [spw_sigs.get(int(s), (0.0, nchan)) for s in sids.tolist()]

    return _LoadedCaltable(
        path=p,
        role=ct.role,
        viscal=ct.viscal,
        is_complex=is_complex,
        param=param,
        flag=flg,
        ant_names_by_row=row_ants,
        field_names_by_row=row_fields,
        spw_sigs_by_row=row_spws,
        time_col=t_col,
        scan_col=sc_col,
        field_id_col=fids,
        ant_id_col=ant1,
        spw_chan_freqs_by_sig=spw_chan_freqs_by_sig,
    )


def _ant_unflagged_fractions(ct: _LoadedCaltable) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for r_idx, aname in enumerate(ct.ant_names_by_row):
        f_slice = ct.flag[:, :, r_idx]
        p_slice = ct.param[:, :, r_idx]
        good = (~f_slice) & np.isfinite(p_slice)
        pair = counts.setdefault(aname, [0, 0])
        pair[0] += int(np.sum(good))
        pair[1] += int(f_slice.size)
    return {a: (v[0] / max(v[1], 1)) for a, v in counts.items()}


def _pick_common_refant(ct_a: _LoadedCaltable, ct_r: _LoadedCaltable) -> str | None:
    uf_a = _ant_unflagged_fractions(ct_a)
    uf_r = _ant_unflagged_fractions(ct_r)
    common = sorted(set(uf_a) & set(uf_r))
    if not common:
        return None

    candidates = [a for a in common if uf_a[a] >= 0.80 and uf_r[a] >= 0.80]
    if not candidates:
        candidates = [a for a in common if uf_a[a] >= 0.50 and uf_r[a] >= 0.50]
    if not candidates:
        candidates = [a for a in common if uf_a[a] > 0.0 and uf_r[a] > 0.0]
    if not candidates:
        return None

    def _ref_dispersion(aname: str) -> tuple[float, str]:
        rows = [i for i, nm in enumerate(ct_r.ant_names_by_row) if nm == aname]
        vals = ct_r.param[:, :, rows][~ct_r.flag[:, :, rows]]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (1e9, aname)
        if ct_r.role == "D":
            d0 = ct_r.param[0, :, rows][~ct_r.flag[0, :, rows]]
            d0 = d0[np.isfinite(d0)]
            disp = float(np.median(np.abs(d0))) if d0.size > 0 else 1e9
            return (disp, aname)
        if ct_r.is_complex:
            ang = np.angle(vals)
            disp = float(np.sqrt(np.mean(ang**2)))
            return (disp, aname)
        disp = float(np.median(np.abs(vals.astype(np.float64))))
        return (disp, aname)

    candidates.sort(key=_ref_dispersion)
    return candidates[0]


def _diff_k_tables(ct_a: _LoadedCaltable, ct_r: _LoadedCaltable) -> dict[str, Any]:
    ref_ant = _pick_common_refant(ct_a, ct_r)
    if ref_ant is None:
        return {"status": "no_common_live_refant"}

    def _extract_rel_delays(ct: _LoadedCaltable) -> dict[tuple[tuple[float, int], int, str], float]:
        # Group rows by (spw_sig, round(time, 3))
        groups: dict[tuple[tuple[float, int], float], dict[str, int]] = {}
        for r_idx in range(len(ct.ant_names_by_row)):
            gkey = (ct.spw_sigs_by_row[r_idx], round(float(ct.time_col[r_idx]), 3))
            groups.setdefault(gkey, {})[ct.ant_names_by_row[r_idx]] = r_idx

        by_key: dict[tuple[tuple[float, int], int, str], list[float]] = {}
        npol = ct.param.shape[0]
        for (spw_sig, _t), amap in groups.items():
            if ref_ant not in amap:
                continue
            r_row = amap[ref_ant]
            for pol in range(npol):
                if ct.flag[pol, 0, r_row]:
                    continue
                tau_ref_ant = float(np.real(ct.param[pol, 0, r_row]))
                if not math.isfinite(tau_ref_ant):
                    continue
                for aname, a_row in amap.items():
                    if ct.flag[pol, 0, a_row]:
                        continue
                    tau_val = float(np.real(ct.param[pol, 0, a_row]))
                    if math.isfinite(tau_val):
                        by_key.setdefault((spw_sig, pol, aname), []).append(tau_val - tau_ref_ant)
        return {k: float(np.median(vs)) for k, vs in by_key.items() if vs}

    rel_a = _extract_rel_delays(ct_a)
    rel_r = _extract_rel_delays(ct_r)
    common_keys = sorted(set(rel_a) & set(rel_r))
    if not common_keys and ct_a.spw_chan_freqs_by_sig and ct_r.spw_chan_freqs_by_sig:
        spw_map = {
            sa: sr
            for sa, sr, _ia, _ir in _match_spw_signatures(
                ct_a.spw_chan_freqs_by_sig, ct_r.spw_chan_freqs_by_sig
            )
        }
        remapped_a = {(spw_map.get(sg, sg), pol, ant): val for (sg, pol, ant), val in rel_a.items()}
        rel_a = remapped_a
        common_keys = sorted(set(rel_a) & set(rel_r))
    if not common_keys:
        return {"status": "no_common_solutions", "refant": ref_ant}

    diffs = np.array([rel_a[k] - rel_r[k] for k in common_keys], dtype=np.float64)
    return {
        "status": "ok",
        "agent_table": ct_a.path.name,
        "ref_table": ct_r.path.name,
        "refant": ref_ant,
        "n_compared": int(diffs.size),
        "delay_rms_ns": round(float(np.sqrt(np.mean(diffs**2))), 6),
        "delay_max_abs_ns": round(float(np.max(np.abs(diffs))), 6),
    }


def _diff_b_tables(ct_a: _LoadedCaltable, ct_r: _LoadedCaltable) -> dict[str, Any]:
    ref_ant = _pick_common_refant(ct_a, ct_r)
    if ref_ant is None:
        return {"status": "no_common_live_refant"}

    def _extract_gauge_bp(
        ct: _LoadedCaltable,
    ) -> dict[tuple[tuple[float, int], int, str], tuple[np.ndarray, np.ndarray]]:
        groups: dict[tuple[tuple[float, int], float], dict[str, int]] = {}
        for r_idx in range(len(ct.ant_names_by_row)):
            gkey = (ct.spw_sigs_by_row[r_idx], round(float(ct.time_col[r_idx]), 3))
            groups.setdefault(gkey, {})[ct.ant_names_by_row[r_idx]] = r_idx

        accum: dict[tuple[tuple[float, int], int, str], list[tuple[np.ndarray, np.ndarray]]] = {}
        npol = ct.param.shape[0]
        for (spw_sig, _t), amap in groups.items():
            if ref_ant not in amap:
                continue
            r_row = amap[ref_ant]
            for pol in range(npol):
                r_spec = ct.param[pol, :, r_row].astype(np.complex128)
                r_flg = ct.flag[pol, :, r_row] | (~np.isfinite(r_spec)) | (np.abs(r_spec) <= 1e-12)
                rot = np.where(~r_flg, np.exp(-1j * np.angle(r_spec)), 1.0 + 0.0j)
                for aname, a_row in amap.items():
                    a_spec = ct.param[pol, :, a_row].astype(np.complex128) * rot
                    a_flg = (
                        ct.flag[pol, :, a_row]
                        | r_flg
                        | (~np.isfinite(a_spec))
                        | (np.abs(a_spec) <= 1e-12)
                    )
                    accum.setdefault((spw_sig, pol, aname), []).append((a_spec, a_flg))

        out: dict[tuple[tuple[float, int], int, str], tuple[np.ndarray, np.ndarray]] = {}
        for key, items in accum.items():
            if len(items) == 1:
                out[key] = items[0]
            else:
                specs = np.stack([it[0] for it in items], axis=0)
                flgs = np.stack([it[1] for it in items], axis=0)
                comb_flg = np.all(flgs, axis=0)
                nch = specs.shape[1]
                comb_spec = np.zeros(nch, dtype=np.complex128)
                for ch in range(nch):
                    good = ~flgs[:, ch]
                    if np.any(good):
                        v = specs[good, ch]
                        amp = float(np.median(np.abs(v)))
                        u = np.mean(v / np.maximum(np.abs(v), 1e-12))
                        ang = float(np.angle(u)) if abs(u) > 1e-12 else float(np.angle(v[0]))
                        comb_spec[ch] = amp * np.exp(1j * ang)
                out[key] = (comb_spec, comb_flg)
        return out

    bp_a = _extract_gauge_bp(ct_a)
    bp_r = _extract_gauge_bp(ct_r)
    spw_chan_slices: dict[tuple[float, int], tuple[np.ndarray | None, np.ndarray | None]] = {}
    common_keys = sorted(set(bp_a) & set(bp_r))
    if not common_keys and ct_a.spw_chan_freqs_by_sig and ct_r.spw_chan_freqs_by_sig:
        remapped_bp_a: dict[tuple[tuple[float, int], int, str], tuple[np.ndarray, np.ndarray]] = {}
        spw_map: dict[tuple[float, int], tuple[float, int]] = {}
        for sa, sr, ia, ir in _match_spw_signatures(
            ct_a.spw_chan_freqs_by_sig, ct_r.spw_chan_freqs_by_sig
        ):
            spw_map[sa] = sr
            spw_chan_slices[sr] = (ia, ir)
        for (sg, pol, ant), val in bp_a.items():
            remapped_bp_a[(spw_map.get(sg, sg), pol, ant)] = val
        bp_a = remapped_bp_a
        common_keys = sorted(set(bp_a) & set(bp_r))
    if not common_keys:
        return {"status": "no_common_solutions", "refant": ref_ant}

    dphi_sq_list: list[np.ndarray] = []
    amp_ratios: list[np.ndarray] = []
    norm_amp_sq_diffs: list[np.ndarray] = []

    for k in common_keys:
        sa, fa = bp_a[k]
        sr, fr = bp_r[k]
        if sa.shape != sr.shape:
            ia_s, ir_s = spw_chan_slices.get(k[0], (None, None))
            if (
                ia_s is not None
                and ir_s is not None
                and int(np.max(ia_s)) < sa.size
                and int(np.max(ir_s)) < sr.size
            ):
                sa = sa[ia_s]
                fa = fa[ia_s]
                sr = sr[ir_s]
                fr = fr[ir_s]
        if sa.shape != sr.shape:
            continue
        good = (~fa) & (~fr) & (np.abs(sa) > 1e-12) & (np.abs(sr) > 1e-12)
        if not np.any(good):
            continue
        sa_g = sa[good]
        sr_g = sr[good]
        dphi_deg = np.rad2deg(np.angle(sa_g * np.conj(sr_g)))
        dphi_sq_list.append(dphi_deg**2)
        amp_ratios.append(np.abs(sa_g) / np.abs(sr_g))
        if int(np.sum(good)) >= 4:
            med_a = float(np.median(np.abs(sa_g)))
            med_r = float(np.median(np.abs(sr_g)))
            if med_a > 1e-12 and med_r > 1e-12:
                na = np.abs(sa_g) / med_a
                nr = np.abs(sr_g) / med_r
                norm_amp_sq_diffs.append((na - nr) ** 2)

    if not dphi_sq_list:
        return {"status": "no_unflagged_overlap", "refant": ref_ant}

    all_dphi_sq = np.concatenate(dphi_sq_list)
    all_ratios = np.concatenate(amp_ratios)
    norm_nrmse = (
        float(np.sqrt(np.mean(np.concatenate(norm_amp_sq_diffs)))) if norm_amp_sq_diffs else 0.0
    )

    return {
        "status": "ok",
        "agent_table": ct_a.path.name,
        "ref_table": ct_r.path.name,
        "refant": ref_ant,
        "n_channel_solutions_compared": int(all_dphi_sq.size),
        "bp_phase_rms_deg": round(float(np.sqrt(np.mean(all_dphi_sq))), 6),
        "bp_amp_scale_ratio": round(float(np.median(all_ratios)), 6),
        "bp_norm_amp_nrmse": round(norm_nrmse, 6),
    }


def _classify_g_tables(
    g_tables: list[_LoadedCaltable],
) -> tuple[_LoadedCaltable | None, _LoadedCaltable | None, _LoadedCaltable | None, float | None]:
    """Classify `G`/`T` caltables into `(g0_phase_only, g_ap_unscaled, f_fluxscaled, s_boot_jy)`."""
    phase_only: list[_LoadedCaltable] = []
    ap_tables: list[_LoadedCaltable] = []

    for ct in g_tables:
        if not ct.is_complex:
            continue
        valid = ct.param[~ct.flag]
        valid = valid[np.isfinite(valid) & (np.abs(valid) > 1e-12)]
        if valid.size == 0:
            continue
        dev_p95 = float(np.percentile(np.abs(np.abs(valid) - 1.0), 95.0))
        if dev_p95 < 1e-3:
            phase_only.append(ct)
        else:
            ap_tables.append(ct)

    g0_chosen = phase_only[-1] if phase_only else None
    if not ap_tables:
        return g0_chosen, None, None, None
    if len(ap_tables) == 1:
        return g0_chosen, ap_tables[-1], ap_tables[-1], None

    # Pairwise physical (G_unscaled, F_fluxscaled) orientation without filename reliance (newest first)
    n = len(ap_tables)
    for i in range(n - 1, -1, -1):
        ta = ap_tables[i]
        for j in range(i - 1, -1, -1):
            tb_ct = ap_tables[j]
            if (
                ta.param.shape != tb_ct.param.shape
                or not np.array_equal(ta.field_id_col, tb_ct.field_id_col)
                or not np.array_equal(ta.ant_id_col, tb_ct.ant_id_col)
                or not np.allclose(ta.time_col, tb_ct.time_col, rtol=0.0, atol=1e-3)
            ):
                continue

            valid_ab = (
                (~ta.flag)
                & (~tb_ct.flag)
                & np.isfinite(ta.param)
                & np.isfinite(tb_ct.param)
                & (np.abs(ta.param) > 1e-12)
                & (np.abs(tb_ct.param) > 1e-12)
            )
            uniq_fids = sorted(int(x) for x in np.unique(ta.field_id_col))
            ref_fid: int | None = None
            tr_fid: int | None = None

            for fid in uniq_fids:
                fm = (ta.field_id_col == fid)[np.newaxis, np.newaxis, :] & valid_ab
                if not np.any(fm):
                    continue
                amp_diff = float(np.median(np.abs(np.abs(ta.param[fm]) - np.abs(tb_ct.param[fm]))))
                phs_diff = float(
                    np.median(np.abs(np.angle(ta.param[fm] * np.conj(tb_ct.param[fm]))))
                )
                if amp_diff <= 1e-4 and phs_diff <= 1e-3:
                    if ref_fid is None:
                        ref_fid = fid
                elif phs_diff <= 1e-3:
                    ratios = np.abs(ta.param[fm]) / np.abs(tb_ct.param[fm])
                    med_r = float(np.median(ratios))
                    if med_r > 1e-12 and abs(med_r - 1.0) > 1e-4:
                        rel_mad = float(np.median(np.abs(ratios - med_r))) / med_r
                        if rel_mad <= 0.02 and tr_fid is None:
                            tr_fid = fid

            if ref_fid is None or tr_fid is None:
                continue

            jump_a = _compute_gain_tr_ref_log_jump(
                ta.param[:, :1, :],
                ta.flag[:, :1, :],
                ta.field_id_col,
                ta.ant_id_col,
                ref_fid,
                tr_fid,
            )
            jump_b = _compute_gain_tr_ref_log_jump(
                tb_ct.param[:, :1, :],
                tb_ct.flag[:, :1, :],
                tb_ct.field_id_col,
                tb_ct.ant_id_col,
                ref_fid,
                tr_fid,
            )
            if jump_a <= jump_b:
                f_ct, g_ct = ta, tb_ct
            else:
                f_ct, g_ct = tb_ct, ta

            tr_m = (ta.field_id_col == tr_fid)[np.newaxis, np.newaxis, :] & valid_ab
            s_est = float(np.median((np.abs(g_ct.param[tr_m]) / np.abs(f_ct.param[tr_m])) ** 2))
            return g0_chosen, g_ct, f_ct, s_est

    return g0_chosen, ap_tables[-2], ap_tables[-1], None


def _interp_complex_gain_series(
    t_src: np.ndarray,
    g_src: np.ndarray,
    t_eval: np.ndarray,
) -> np.ndarray:
    """Scan-bounded complex gain interpolation supporting `solint='inf'` (N=1) vs `'30s'` (N>=2)."""
    if t_src.size == 1:
        return np.full(t_eval.shape, g_src[0], dtype=np.complex128)
    order = np.argsort(t_src)
    ts = t_src[order]
    gs = g_src[order]
    # Deduplicate identical timestamps if any
    uniq_t, inv = np.unique(ts, return_inverse=True)
    if uniq_t.size != ts.size:
        dedup_g = np.zeros(uniq_t.size, dtype=np.complex128)
        for idx in range(uniq_t.size):
            sel = gs[inv == idx]
            amp = float(np.median(np.abs(sel)))
            u = np.mean(sel / np.maximum(np.abs(sel), 1e-12))
            dedup_g[idx] = amp * np.exp(1j * float(np.angle(u)))
        ts = uniq_t
        gs = dedup_g
    if ts.size == 1:
        return np.full(t_eval.shape, gs[0], dtype=np.complex128)

    amps = np.abs(gs)
    unwrapped_phs = np.unwrap(np.angle(gs))
    amp_i = np.interp(t_eval, ts, amps)
    phs_i = np.interp(t_eval, ts, unwrapped_phs)
    return amp_i * np.exp(1j * phs_i)


def _diff_g_pair(ct_a: _LoadedCaltable, ct_r: _LoadedCaltable) -> dict[str, Any]:
    ref_ant = _pick_common_refant(ct_a, ct_r)
    if ref_ant is None:
        return {"status": "no_common_live_refant"}

    target_npol = max(int(ct_a.param.shape[0]), int(ct_r.param.shape[0]))
    spw_remap_a: dict[tuple[float, int], tuple[float, int]] = {}
    if ct_a.spw_chan_freqs_by_sig and ct_r.spw_chan_freqs_by_sig:
        for sa, sr, _ia, _ir in _match_spw_signatures(
            ct_a.spw_chan_freqs_by_sig, ct_r.spw_chan_freqs_by_sig
        ):
            spw_remap_a[sa] = sr

    def _build_scan_series(
        ct: _LoadedCaltable,
        *,
        use_scan: bool,
        remap_spw: dict[tuple[float, int], tuple[float, int]] | None = None,
    ) -> dict[tuple[str, tuple[float, int], int, int, str], tuple[np.ndarray, np.ndarray]]:
        # First group by timestamp slot `(field, spw_sig, scan_key, round(t, 3))` to gauge-fix to `ref_ant`
        slots: dict[tuple[str, tuple[float, int], int, float], dict[str, int]] = {}
        for r_idx in range(len(ct.ant_names_by_row)):
            sc_k = int(ct.scan_col[r_idx]) if use_scan else 0
            raw_sg = ct.spw_sigs_by_row[r_idx]
            eff_sg = remap_spw.get(raw_sg, raw_sg) if remap_spw else raw_sg
            skey = (
                ct.field_names_by_row[r_idx],
                eff_sg,
                sc_k,
                round(float(ct.time_col[r_idx]), 3),
            )
            slots.setdefault(skey, {})[ct.ant_names_by_row[r_idx]] = r_idx

        series: dict[tuple[str, tuple[float, int], int, int, str], list[tuple[float, complex]]] = {}
        npol = int(ct.param.shape[0])
        eff_npol = target_npol if (npol == 1 and target_npol > 1) else npol
        for (fld, spw_sig, sc_k, _t_r), amap in sorted(slots.items(), key=lambda kv: kv[0][3]):
            if ref_ant not in amap:
                continue
            r_row = amap[ref_ant]
            t_exact = float(ct.time_col[r_row])
            for pol in range(eff_npol):
                src_pol = 0 if npol == 1 else pol
                if ct.flag[src_pol, 0, r_row]:
                    continue
                g_ref_ant = complex(ct.param[src_pol, 0, r_row])
                if not (
                    math.isfinite(g_ref_ant.real)
                    and math.isfinite(g_ref_ant.imag)
                    and abs(g_ref_ant) > 1e-12
                ):
                    continue
                rot = np.exp(-1j * math.atan2(g_ref_ant.imag, g_ref_ant.real))
                for aname, a_row in amap.items():
                    if ct.flag[src_pol, 0, a_row]:
                        continue
                    gv = complex(ct.param[src_pol, 0, a_row]) * rot
                    if math.isfinite(gv.real) and math.isfinite(gv.imag) and abs(gv) > 1e-12:
                        series.setdefault((fld, spw_sig, sc_k, pol, aname), []).append(
                            (t_exact, gv)
                        )

        return {
            k: (
                np.array([pt[0] for pt in pts], dtype=np.float64),
                np.array([pt[1] for pt in pts], dtype=np.complex128),
            )
            for k, pts in series.items()
            if pts
        }

    ser_a = _build_scan_series(ct_a, use_scan=True)
    ser_r = _build_scan_series(ct_r, use_scan=True)
    common_keys = sorted(set(ser_a) & set(ser_r))
    if not common_keys and spw_remap_a:
        ser_a = _build_scan_series(ct_a, use_scan=True, remap_spw=spw_remap_a)
        common_keys = sorted(set(ser_a) & set(ser_r))
    if not common_keys:
        # Fallback when one side used `combine='scan'` so `SCAN_NUMBER` differs
        ser_a = _build_scan_series(ct_a, use_scan=False, remap_spw=spw_remap_a or None)
        ser_r = _build_scan_series(ct_r, use_scan=False)
        common_keys = sorted(set(ser_a) & set(ser_r))

    if not common_keys:
        return {"status": "no_common_solutions", "refant": ref_ant}

    dphi_sq: list[np.ndarray] = []
    amp_diffs: list[np.ndarray] = []
    ref_amps: list[np.ndarray] = []

    for k in common_keys:
        ta_arr, ga_arr = ser_a[k]
        tr_arr, gr_arr = ser_r[k]
        if ta_arr.size >= tr_arr.size:
            t_eval = ta_arr
            ga_eval = ga_arr
            gr_eval = _interp_complex_gain_series(tr_arr, gr_arr, t_eval)
        else:
            t_eval = tr_arr
            gr_eval = gr_arr
            ga_eval = _interp_complex_gain_series(ta_arr, ga_arr, t_eval)

        dphi_deg = np.rad2deg(np.angle(ga_eval * np.conj(gr_eval)))
        dphi_sq.append(dphi_deg**2)
        amp_diffs.append(np.abs(ga_eval) - np.abs(gr_eval))
        ref_amps.append(np.abs(gr_eval))

    all_dphi_sq = np.concatenate(dphi_sq)
    all_amp_diffs = np.concatenate(amp_diffs)
    all_ref_amps = np.concatenate(ref_amps)
    med_ref_amp = float(np.median(all_ref_amps)) if all_ref_amps.size > 0 else 1.0
    amp_rel_rms = (
        float(np.sqrt(np.mean((all_amp_diffs / med_ref_amp) ** 2))) if med_ref_amp > 1e-12 else 0.0
    )

    return {
        "status": "ok",
        "agent_table": ct_a.path.name,
        "ref_table": ct_r.path.name,
        "refant": ref_ant,
        "n_compared": int(all_dphi_sq.size),
        "gain_phase_rms_deg": round(float(np.sqrt(np.mean(all_dphi_sq))), 6),
        "gain_amp_rel_rms": round(amp_rel_rms, 6),
    }


def _estimate_fluxscale_ratio_from_f_tables(
    f_agt: _LoadedCaltable,
    f_ref: _LoadedCaltable,
    flux_cal_names: list[str],
    phase_cal_names: list[str],
) -> float | None:
    """Fallback `S_agt / S_ref` directly from `(F_agt, F_ref)` when unscaled `G` was overwritten."""
    flux_set = {x.strip().upper() for x in flux_cal_names if x.strip()}
    phase_set = {x.strip().upper() for x in phase_cal_names if x.strip()}
    if not flux_set or not phase_set:
        return None

    def _med_amp_for_fields(ct: _LoadedCaltable, fset: set[str]) -> float | None:
        rows = [i for i, fn in enumerate(ct.field_names_by_row) if fn in fset]
        if not rows:
            return None
        sub_p = ct.param[:, :, rows]
        sub_f = ct.flag[:, :, rows]
        vals = np.abs(sub_p[(~sub_f) & np.isfinite(sub_p)])
        vals = vals[vals > 1e-12]
        return float(np.median(vals)) if vals.size > 0 else None

    fa_ref = _med_amp_for_fields(f_agt, flux_set)
    fa_tr = _med_amp_for_fields(f_agt, phase_set)
    fr_ref = _med_amp_for_fields(f_ref, flux_set)
    fr_tr = _med_amp_for_fields(f_ref, phase_set)
    if (
        fa_ref is None
        or fa_tr is None
        or fr_ref is None
        or fr_tr is None
        or fa_tr <= 1e-12
        or fr_ref <= 1e-12
    ):
        return None
    # |F(tr)| / |F(ref)| = sqrt(S_true / S_est) => (|F_ref(tr)|/|F_ref(ref)|)^2 / (|F_agt(tr)|/|F_agt(ref)|)^2 = S_agt / S_ref
    ratio = ((fr_tr / fr_ref) / (fa_tr / fa_ref)) ** 2
    return float(ratio) if math.isfinite(ratio) else None


def _diff_d_tables(ct_a: _LoadedCaltable, ct_r: _LoadedCaltable) -> dict[str, Any]:
    """Compare `D` / `Df` leakage tables using coupled first-order RIME gauge fix to `r*`."""
    if ct_a.param.shape[0] < 2 or ct_r.param.shape[0] < 2:
        return {"status": "invalid_dterm_pol_shape"}
    ref_ant = _pick_common_refant(ct_a, ct_r)
    if ref_ant is None:
        return {"status": "no_common_live_refant"}

    def _extract_rime_gauge_d(
        ct: _LoadedCaltable,
    ) -> dict[tuple[tuple[float, int], str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        groups: dict[tuple[tuple[float, int], float], dict[str, int]] = {}
        for r_idx in range(len(ct.ant_names_by_row)):
            gkey = (ct.spw_sigs_by_row[r_idx], round(float(ct.time_col[r_idx]), 3))
            groups.setdefault(gkey, {})[ct.ant_names_by_row[r_idx]] = r_idx

        accum: dict[
            tuple[tuple[float, int], str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}
        for (spw_sig, _t), amap in groups.items():
            if ref_ant not in amap:
                continue
            r_row = amap[ref_ant]
            d0_ref_ant = ct.param[0, :, r_row].astype(np.complex128)
            r_flg = ct.flag[0, :, r_row] | (~np.isfinite(d0_ref_ant))
            for aname, a_row in amap.items():
                d0_raw = ct.param[0, :, a_row].astype(np.complex128)
                d1_raw = ct.param[1, :, a_row].astype(np.complex128)
                flg = (
                    ct.flag[0, :, a_row]
                    | ct.flag[1, :, a_row]
                    | r_flg
                    | (~np.isfinite(d0_raw))
                    | (~np.isfinite(d1_raw))
                )
                # Coupled RIME gauge fix: D0_tilde = D0 - D0[r*], D1_tilde = D1 + conj(D0[r*])
                d0_t = d0_raw - d0_ref_ant
                d1_t = d1_raw + np.conj(d0_ref_ant)
                accum.setdefault((spw_sig, aname), []).append((d0_t, d1_t, flg))

        out: dict[tuple[tuple[float, int], str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, items in accum.items():
            if len(items) == 1:
                out[key] = items[0]
                continue
            d0_s = np.stack([it[0] for it in items], axis=0)
            d1_s = np.stack([it[1] for it in items], axis=0)
            fl_s = np.stack([it[2] for it in items], axis=0)
            comb_flg = np.all(fl_s, axis=0)
            nch = d0_s.shape[1]
            c_d0 = np.zeros(nch, dtype=np.complex128)
            c_d1 = np.zeros(nch, dtype=np.complex128)
            for ch in range(nch):
                good = ~fl_s[:, ch]
                if np.any(good):
                    v0 = d0_s[good, ch]
                    v1 = d1_s[good, ch]
                    c_d0[ch] = float(np.median(v0.real)) + 1j * float(np.median(v0.imag))
                    c_d1[ch] = float(np.median(v1.real)) + 1j * float(np.median(v1.imag))
            out[key] = (c_d0, c_d1, comb_flg)
        return out

    da = _extract_rime_gauge_d(ct_a)
    dr = _extract_rime_gauge_d(ct_r)
    spw_chan_slices_d: dict[tuple[float, int], tuple[np.ndarray | None, np.ndarray | None]] = {}
    common_keys = sorted(set(da) & set(dr))
    if not common_keys and ct_a.spw_chan_freqs_by_sig and ct_r.spw_chan_freqs_by_sig:
        remapped_da: dict[
            tuple[tuple[float, int], str], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        spw_map_d: dict[tuple[float, int], tuple[float, int]] = {}
        for sa, sr, ia, ir in _match_spw_signatures(
            ct_a.spw_chan_freqs_by_sig, ct_r.spw_chan_freqs_by_sig
        ):
            spw_map_d[sa] = sr
            spw_chan_slices_d[sr] = (ia, ir)
        for (sg, ant), val in da.items():
            remapped_da[(spw_map_d.get(sg, sg), ant)] = val
        da = remapped_da
        common_keys = sorted(set(da) & set(dr))
    if not common_keys:
        return {"status": "no_common_solutions", "refant": ref_ant}

    diffs_list: list[np.ndarray] = []
    for k in common_keys:
        d0a, d1a, fa = da[k]
        d0r, d1r, fr = dr[k]
        if d0a.shape != d0r.shape:
            ia_d, ir_d = spw_chan_slices_d.get(k[0], (None, None))
            if (
                ia_d is not None
                and ir_d is not None
                and int(np.max(ia_d)) < d0a.size
                and int(np.max(ir_d)) < d0r.size
            ):
                d0a, d1a, fa = d0a[ia_d], d1a[ia_d], fa[ia_d]
                d0r, d1r, fr = d0r[ir_d], d1r[ir_d], fr[ir_d]
        if d0a.shape != d0r.shape:
            continue
        good = (~fa) & (~fr)
        if not np.any(good):
            continue
        diffs_list.append(d0a[good] - d0r[good])
        diffs_list.append(d1a[good] - d1r[good])

    if not diffs_list:
        return {"status": "no_unflagged_overlap", "refant": ref_ant}

    all_diffs = np.concatenate(diffs_list)
    abs_diffs = np.abs(all_diffs)
    return {
        "status": "ok",
        "agent_table": ct_a.path.name,
        "ref_table": ct_r.path.name,
        "refant": ref_ant,
        "n_compared": int(abs_diffs.size),
        "dterm_rms": round(float(np.sqrt(np.mean(abs_diffs**2))), 8),
        "dterm_max_abs": round(float(np.max(abs_diffs)), 8),
    }


def diff_caltables(
    ws_agt_dir: Path,
    ws_ref_dir: Path,
    *,
    ms_agt_path: Path | None = None,
    ms_ref_path: Path | None = None,
    extract_config: ExtractConfig | None = None,
) -> dict[str, Any]:
    """Discover (`VisCal`) and compare `K`, `B`, `G`/`F`, and `D`/`Df` caltables gauge-invariantly."""
    _ensure_casa_env()
    ws_a = Path(ws_agt_dir)
    ws_r = Path(ws_ref_dir)
    reject_slash_tmp(ws_a, label="ws_agt_dir")
    reject_slash_tmp(ws_r, label="ws_ref_dir")
    if ms_agt_path is not None:
        reject_slash_tmp(Path(ms_agt_path), label="ms_agt_path")
    if ms_ref_path is not None:
        reject_slash_tmp(Path(ms_ref_path), label="ms_ref_path")
    ms_a = ms_agt_path or _find_primary_ms(ws_a)
    ms_r = ms_ref_path or _find_primary_ms(ws_r)

    disc_a = sorted(
        discover_caltables_by_viscal(ws_a, max_depth=3), key=lambda d: _caltable_sort_key(d.path)
    )
    disc_r = sorted(
        discover_caltables_by_viscal(ws_r, max_depth=3), key=lambda d: _caltable_sort_key(d.path)
    )
    if not disc_a and not disc_r:
        return {"status": "no_caltables_found"}

    loaded_a = [lc for d in disc_a if (lc := _load_caltable(d, ms_a)) is not None]
    loaded_r = [lc for d in disc_r if (lc := _load_caltable(d, ms_r)) is not None]
    loaded_a.sort(key=lambda c: _caltable_sort_key(c.path))
    loaded_r.sort(key=lambda c: _caltable_sort_key(c.path))

    out: dict[str, Any] = {
        "status": "ok",
        "agent_caltables": [
            {"name": d.path.name, "role": d.role, "viscal": d.viscal} for d in disc_a
        ],
        "ref_caltables": [
            {"name": d.path.name, "role": d.role, "viscal": d.viscal} for d in disc_r
        ],
    }

    # K Jones
    ka = [c for c in loaded_a if c.role == "K"]
    kr = [c for c in loaded_r if c.role == "K"]
    if ka and kr:
        out["K"] = _diff_k_tables(ka[-1], kr[-1])

    # B Jones
    ba = [c for c in loaded_a if c.role == "B"]
    br = [c for c in loaded_r if c.role == "B"]
    if ba and br:
        out["B"] = _diff_b_tables(ba[-1], br[-1])

    # G / T Jones + Fluxscale
    ga_list = [c for c in loaded_a if c.role == "G"]
    gr_list = [c for c in loaded_r if c.role == "G"]
    if ga_list and gr_list:
        g0_a, gap_a, f_a, s_a = _classify_g_tables(ga_list)
        g0_r, gap_r, f_r, s_r = _classify_g_tables(gr_list)

        flux_names: list[str] = []
        phase_names: list[str] = []
        if extract_config is not None:
            flux_names = list(extract_config.roles.flux_cal)
            phase_names = list(extract_config.roles.phase_cal)
        elif ms_a is not None and ms_r is not None:
            roles_map = _classify_field_roles(ms_a, ms_r)
            flux_names = [fn for fn, rset in roles_map.items() if "flux_cal" in rset]
            phase_names = [fn for fn, rset in roles_map.items() if "phase_cal" in rset]

        if (s_a is None or s_r is None) and flux_names and phase_names:
            if s_a is None:
                s_a, _ = discover_fluxscale_from_caltables(
                    ws_a, flux_cal_name=flux_names[0], phase_cal_name=phase_names[0]
                )
            if s_r is None:
                s_r, _ = discover_fluxscale_from_caltables(
                    ws_r, flux_cal_name=flux_names[0], phase_cal_name=phase_names[0]
                )

        g_block: dict[str, Any] = {}
        if g0_a is not None and g0_r is not None:
            g_block["G0_phase_only"] = _diff_g_pair(g0_a, g0_r)
        if gap_a is not None and gap_r is not None:
            g_block["G_ap"] = _diff_g_pair(gap_a, gap_r)
        if f_a is not None and f_r is not None and (f_a is not gap_a or f_r is not gap_r):
            g_block["F_fluxscale"] = _diff_g_pair(f_a, f_r)

        flux_ratio: float | None = None
        if s_a is not None and s_r is not None and s_r > 1e-12:
            flux_ratio = float(s_a / s_r)
        elif f_a is not None and f_r is not None and flux_names and phase_names:
            flux_ratio = _estimate_fluxscale_ratio_from_f_tables(f_a, f_r, flux_names, phase_names)

        g_block["agent_bootstrapped_flux_jy"] = round(s_a, 6) if s_a is not None else None
        g_block["ref_bootstrapped_flux_jy"] = round(s_r, 6) if s_r is not None else None
        g_block["fluxscale_ratio"] = round(flux_ratio, 6) if flux_ratio is not None else None
        out["G"] = g_block

    # D / Df Jones
    da_list = [c for c in loaded_a if c.role == "D"]
    dr_list = [c for c in loaded_r if c.role == "D"]
    if da_list and dr_list:
        out["D"] = _diff_d_tables(da_list[-1], dr_list[-1])

    return sanitize_json_tree(out)


# ==============================================================================
# 3. Image-Domain Common-Beam Diff (`ia.commonbeam()` + Quadrature `imsmooth` + `imregrid`)
# ==============================================================================


def _compute_enclosing_common_beam(
    hdr_a: dict[str, Any],
    hdr_r: dict[str, Any],
    scratch_dir: Path,
) -> tuple[float, float, float]:
    """Compute `(cb_maj_arcsec, cb_min_arcsec, cb_pa_deg)` via a 2-channel `(4,4,1,2)` scratch image."""
    import casatools

    cb_img = scratch_dir / "commonbeam_probe.im"
    ia = casatools.image()
    qa = casatools.quanta()
    try:
        ia.fromshape(str(cb_img), [4, 4, 1, 2], overwrite=True)
        ia.setrestoringbeam(
            major=f"{float(hdr_a['b_major_arcsec']):.8f}arcsec",
            minor=f"{float(hdr_a['b_minor_arcsec']):.8f}arcsec",
            pa=f"{float(hdr_a['b_pa_deg']):.6f}deg",
            channel=0,
            polarization=0,
        )
        ia.setrestoringbeam(
            major=f"{float(hdr_r['b_major_arcsec']):.8f}arcsec",
            minor=f"{float(hdr_r['b_minor_arcsec']):.8f}arcsec",
            pa=f"{float(hdr_r['b_pa_deg']):.6f}deg",
            channel=1,
            polarization=0,
        )
        cb = ia.commonbeam()
        cb_maj = float(qa.convert(cb["major"], "arcsec")["value"])
        cb_min = float(qa.convert(cb["minor"], "arcsec")["value"])
        pa_key = "pa" if "pa" in cb else "positionangle"
        cb_pa = float(qa.convert(cb[pa_key], "deg")["value"]) if pa_key in cb else 0.0
        return cb_maj, cb_min, cb_pa
    finally:
        with contextlib.suppress(Exception):
            ia.close()
        with contextlib.suppress(Exception):
            ia.done()
        with contextlib.suppress(Exception):
            qa.done()
        shutil.rmtree(cb_img, ignore_errors=True)


def _beams_nearly_identical(hdr_a: dict[str, Any], hdr_r: dict[str, Any]) -> bool:
    maj_a, min_a, pa_a = (
        float(hdr_a["b_major_arcsec"]),
        float(hdr_a["b_minor_arcsec"]),
        float(hdr_a["b_pa_deg"]),
    )
    maj_r, min_r, pa_r = (
        float(hdr_r["b_major_arcsec"]),
        float(hdr_r["b_minor_arcsec"]),
        float(hdr_r["b_pa_deg"]),
    )
    rel_maj = abs(maj_a - maj_r) / max(maj_r, 1e-12)
    rel_min = abs(min_a - min_r) / max(min_r, 1e-12)
    if rel_maj > 1e-3 or rel_min > 1e-3:
        return False
    circ_a = abs(maj_a - min_a) / max(maj_a, 1e-12) <= 1e-3
    circ_r = abs(maj_r - min_r) / max(maj_r, 1e-12) <= 1e-3
    if circ_a and circ_r:
        return True
    dpa = abs(((pa_a - pa_r + 90.0) % 180.0) - 90.0)
    return dpa <= 0.5


def _wcs_or_shape_differs(
    shape_a: tuple[int, ...],
    hdr_a: dict[str, Any],
    shape_r: tuple[int, ...],
    hdr_r: dict[str, Any],
) -> bool:
    if shape_a != shape_r:
        return True
    if abs(float(hdr_a["inc"][0]) - float(hdr_r["inc"][0])) > 1e-10:
        return True
    if abs(float(hdr_a["inc"][1]) - float(hdr_r["inc"][1])) > 1e-10:
        return True
    if abs(float(hdr_a["refval"][0]) - float(hdr_r["refval"][0])) > 1e-8:
        return True
    if abs(float(hdr_a["refval"][1]) - float(hdr_r["refval"][1])) > 1e-8:
        return True
    if abs(float(hdr_a["refpix"][0]) - float(hdr_r["refpix"][0])) > 1e-3:
        return True
    return abs(float(hdr_a["refpix"][1]) - float(hdr_r["refpix"][1])) > 1e-3


def _compute_negative_bowl_ratio(iset: DiscoveredImageSet, pb_mask_min: float) -> float | None:
    if iset.model_path is None:
        return None
    try:
        mod_arr, mod_msk, _ = read_image_2d(iset.model_path)
    except Exception:
        return None
    gate = mod_msk & np.isfinite(mod_arr)
    if iset.pb_path is not None:
        with contextlib.suppress(Exception):
            pb_arr, pb_msk, _ = read_image_2d(iset.pb_path)
            if pb_arr.shape == mod_arr.shape:
                gate = gate & pb_msk & np.isfinite(pb_arr) & (pb_arr >= pb_mask_min)
    vals = mod_arr[gate]
    if vals.size == 0:
        return None
    pos_sum = float(np.sum(vals[vals > 0.0]))
    neg_abs = float(np.abs(np.sum(vals[vals < 0.0])))
    if pos_sum <= 1e-12:
        return 1.0
    return float(neg_abs / pos_sum)


def _compute_native_offsource_rms(
    iset: DiscoveredImageSet,
    pb_mask_min: float,
    extract_config: ExtractConfig | None,
) -> float | None:
    target_path = (
        iset.residual_path
        if iset.residual_path is not None
        else (iset.pbcor_path or iset.image_path)
    )
    try:
        arr, msk, hdr = read_image_2d(iset.image_path, need_header=True)
        if target_path != iset.image_path:
            res_arr, res_msk, _ = read_image_2d(target_path)
        else:
            res_arr, res_msk = arr, msk
    except Exception:
        return None
    if hdr is None or res_arr.shape != arr.shape:
        return None

    valid = res_msk & np.isfinite(res_arr) & np.isfinite(arr)
    if iset.pb_path is not None:
        with contextlib.suppress(Exception):
            pb_arr, pb_msk, _ = read_image_2d(iset.pb_path)
            if pb_arr.shape == res_arr.shape:
                valid = valid & pb_msk & np.isfinite(pb_arr) & (pb_arr >= pb_mask_min)

    if int(np.sum(valid)) < 20:
        return None

    nx, ny = res_arr.shape
    ix, iy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    cell_x = float(hdr["cell_x_arcsec"])
    cell_y = float(hdr["cell_y_arcsec"])

    off_mask = valid.copy()
    if (
        extract_config is not None
        and extract_config.geometry.off_source_min_radius_arcsec is not None
    ):
        geom = extract_config.geometry
        if geom.phasecenter:
            ra_c, dec_c = parse_sky_coord_rad(geom.phasecenter)
            px_c, py_c = sky_to_pix(hdr, ra_c, dec_c)
        else:
            px_c, py_c = (nx - 1) / 2.0, (ny - 1) / 2.0
        dist_c = np.hypot((ix - px_c) * cell_x, (iy - py_c) * cell_y)
        cand = valid & (dist_c >= float(geom.off_source_min_radius_arcsec))
        if int(np.sum(cand)) >= 20:
            off_mask = cand
    elif extract_config is not None and extract_config.geometry.source_apertures:
        beam_excl = float(extract_config.geometry.offsource_inner_exclusion_beams) * float(
            hdr["b_major_arcsec"]
        )
        cand = valid.copy()
        for ap in extract_config.geometry.source_apertures:
            ra_r, dec_r = parse_sky_coord_rad(ap.ra, ap.dec)
            ap_px, ap_py = sky_to_pix(hdr, ra_r, dec_r)
            dist_ap = np.hypot((ix - ap_px) * cell_x, (iy - ap_py) * cell_y)
            excl_r = (
                float(ap.offsource_exclusion_arcsec)
                if ap.offsource_exclusion_arcsec is not None
                else max(beam_excl, 1.5 * float(ap.radius_arcsec))
            )
            cand = cand & (dist_ap > excl_r)
        if int(np.sum(cand)) >= 20:
            off_mask = cand
    else:
        masked_img = np.where(valid, arr, -np.inf)
        pk_flat = int(np.argmax(masked_img))
        px_pk, py_pk = np.unravel_index(pk_flat, res_arr.shape)
        dist_pk = np.hypot((ix - float(px_pk)) * cell_x, (iy - float(py_pk)) * cell_y)
        cand = valid & (dist_pk >= 5.0 * float(hdr["b_major_arcsec"]))
        if int(np.sum(cand)) >= 20:
            off_mask = cand

    vals = res_arr[off_mask]
    if vals.size == 0:
        return None
    return float(np.sqrt(np.mean(vals**2)))


def diff_images(
    ws_agt_dir: Path,
    ws_ref_dir: Path,
    *,
    pb_mask_min: float = 0.20,
    extract_config: ExtractConfig | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Compare continuum images at a quadrature-padded common beam on the reference WCS grid."""
    _ensure_casa_env()
    ws_a = Path(ws_agt_dir)
    ws_r = Path(ws_ref_dir)
    reject_slash_tmp(ws_a, label="ws_agt_dir")
    reject_slash_tmp(ws_r, label="ws_ref_dir")
    if scratch_root is not None:
        reject_slash_tmp(Path(scratch_root), label="scratch_root")

    sets_a = discover_image_sets(ws_a, max_depth=3)
    sets_r = discover_image_sets(ws_r, max_depth=3)
    if not sets_a or not sets_r:
        return {
            "status": "missing_image_set",
            "agent_sets_found": len(sets_a),
            "ref_sets_found": len(sets_r),
        }

    chosen_a = sets_a[0]
    chosen_r = sets_r[0]
    if chosen_a.pbcor_path is not None and chosen_r.pbcor_path is not None:
        img_a_src = chosen_a.pbcor_path
        img_r_src = chosen_r.pbcor_path
        product_kind = "pbcor"
    else:
        img_a_src = chosen_a.image_path
        img_r_src = chosen_r.image_path
        product_kind = "image"

    if has_active_table_lock(img_a_src) or has_active_table_lock(img_r_src):
        return {"status": "locked_or_unreadable"}

    eff_pb_min = (
        float(extract_config.geometry.pb_mask_min)
        if extract_config is not None
        else float(pb_mask_min)
    )

    try:
        arr_a0, _msk_a0, hdr_a0 = read_image_2d(img_a_src, need_header=True)
        arr_r0, _msk_r0, hdr_r0 = read_image_2d(img_r_src, need_header=True)
    except Exception:
        return {"status": "missing_restoring_beam_or_wcs"}
    if hdr_a0 is None or hdr_r0 is None:
        return {"status": "missing_restoring_beam_or_wcs"}

    base_scratch = resolve_scratch_root(scratch_root)
    run_scratch = base_scratch / f"diff_img_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    run_scratch.mkdir(parents=True, exist_ok=True)

    try:
        from casatasks import imregrid, imsmooth

        same_beam = _beams_nearly_identical(hdr_a0, hdr_r0)
        if same_beam:
            work_a = img_a_src
            work_r = img_r_src
            cb_maj = float(hdr_r0["b_major_arcsec"])
            cb_min = float(hdr_r0["b_minor_arcsec"])
            cb_pa = float(hdr_r0["b_pa_deg"])
            beam_smoothed = False
        else:
            raw_cb_maj, raw_cb_min, cb_pa = _compute_enclosing_common_beam(
                hdr_a0, hdr_r0, run_scratch
            )
            max_cell = max(
                float(hdr_a0["cell_x_arcsec"]),
                float(hdr_a0["cell_y_arcsec"]),
                float(hdr_r0["cell_x_arcsec"]),
                float(hdr_r0["cell_y_arcsec"]),
            )
            pad_arcsec = 2.2 * max_cell
            major_pad = math.hypot(raw_cb_maj, pad_arcsec)
            minor_pad = math.hypot(raw_cb_min, pad_arcsec)
            if major_pad < minor_pad:
                major_pad, minor_pad = minor_pad, major_pad
            if abs(major_pad - minor_pad) < 1e-6 * major_pad:
                major_pad *= 1.0 + 1e-5

            smooth_a = run_scratch / "agent_commonbeam.im"
            smooth_r = run_scratch / "ref_commonbeam.im"
            imsmooth(
                imagename=str(img_a_src),
                kernel="gauss",
                major=f"{major_pad:.8f}arcsec",
                minor=f"{minor_pad:.8f}arcsec",
                pa=f"{cb_pa:.6f}deg",
                targetres=True,
                outfile=str(smooth_a),
                overwrite=True,
            )
            imsmooth(
                imagename=str(img_r_src),
                kernel="gauss",
                major=f"{major_pad:.8f}arcsec",
                minor=f"{minor_pad:.8f}arcsec",
                pa=f"{cb_pa:.6f}deg",
                targetres=True,
                outfile=str(smooth_r),
                overwrite=True,
            )
            work_a = smooth_a
            work_r = smooth_r
            cb_maj, cb_min = major_pad, minor_pad
            beam_smoothed = True

        need_regrid = _wcs_or_shape_differs(arr_a0.shape, hdr_a0, arr_r0.shape, hdr_r0)
        pb_a_aligned_path: Path | None = chosen_a.pb_path
        if need_regrid:
            regrid_a = run_scratch / "agent_regridded.im"
            imregrid(
                imagename=str(work_a),
                template=str(work_r),
                output=str(regrid_a),
                axes=[0, 1],
                overwrite=True,
            )
            work_a = regrid_a
            if chosen_a.pb_path is not None:
                regrid_pba = run_scratch / "agent_pb_regridded.im"
                with contextlib.suppress(Exception):
                    imregrid(
                        imagename=str(chosen_a.pb_path),
                        template=str(work_r),
                        output=str(regrid_pba),
                        axes=[0, 1],
                        overwrite=True,
                    )
                    if (regrid_pba / "table.dat").is_file():
                        pb_a_aligned_path = regrid_pba

        try:
            arr_a, msk_a, _ = read_image_2d(work_a)
            arr_r, msk_r, hdr_r = read_image_2d(work_r, need_header=True)
        except Exception:
            return {"status": "alignment_failed"}
        if hdr_r is None or arr_a.shape != arr_r.shape:
            return {"status": "alignment_failed"}

        common_mask = msk_a & msk_r & np.isfinite(arr_a) & np.isfinite(arr_r)
        if chosen_r.pb_path is not None:
            with contextlib.suppress(Exception):
                pb_r, pbm_r, _ = read_image_2d(chosen_r.pb_path)
                if pb_r.shape == arr_r.shape:
                    common_mask = common_mask & pbm_r & np.isfinite(pb_r) & (pb_r >= eff_pb_min)
        if pb_a_aligned_path is not None:
            with contextlib.suppress(Exception):
                pb_a, pbm_a, _ = read_image_2d(pb_a_aligned_path)
                if pb_a.shape == arr_r.shape:
                    common_mask = common_mask & pbm_a & np.isfinite(pb_a) & (pb_a >= eff_pb_min)

        n_common_pix = int(np.sum(common_mask))
        if n_common_pix < 16:
            return {"status": "insufficient_common_pb_pixels", "n_common_pixels": n_common_pix}

        va = arr_a[common_mask]
        vr = arr_r[common_mask]
        ca = va - float(np.mean(va))
        cr = vr - float(np.mean(vr))
        da = float(np.sum(ca**2))
        dr = float(np.sum(cr**2))
        if da <= 1e-30 and dr <= 1e-30:
            ncc = 1.0 if float(np.max(np.abs(va - vr))) <= 1e-12 else 0.0
        elif da <= 1e-30 or dr <= 1e-30:
            ncc = 0.0
        else:
            ncc = max(-1.0, min(1.0, float(np.sum(ca * cr) / math.sqrt(da * dr))))

        peak_a = float(np.max(va))
        peak_r = float(np.max(vr))
        peak_ratio = float(peak_a / peak_r) if abs(peak_r) > 1e-15 else None

        # Integrated flux inside configured apertures or reference high-SNR support
        nx, ny = arr_r.shape
        ix, iy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
        cell_x = float(hdr_r["cell_x_arcsec"])
        cell_y = float(hdr_r["cell_y_arcsec"])
        beam_area_pix = (math.pi * cb_maj * cb_min) / max(
            4.0 * math.log(2.0) * cell_x * cell_y, 1e-12
        )

        ap_mask = np.zeros(arr_r.shape, dtype=bool)
        if extract_config is not None and extract_config.geometry.source_apertures:
            for ap in extract_config.geometry.source_apertures:
                ra_r, dec_r = parse_sky_coord_rad(ap.ra, ap.dec)
                ap_px, ap_py = sky_to_pix(hdr_r, ra_r, dec_r)
                dist_ap = np.hypot((ix - ap_px) * cell_x, (iy - ap_py) * cell_y)
                ap_mask = ap_mask | (common_mask & (dist_ap <= float(ap.radius_arcsec)))

        if not np.any(ap_mask):
            med_vr = float(np.median(vr))
            mad_vr = 1.4826 * float(np.median(np.abs(vr - med_vr)))
            thresh = max(med_vr + 5.0 * mad_vr, 0.05 * peak_r)
            ap_mask = common_mask & (arr_r >= thresh)
            if not np.any(ap_mask):
                ap_mask = common_mask & (arr_r > 0.0)

        ap_flux_a_jy: float | None = None
        ap_flux_r_jy: float | None = None
        ap_flux_ratio: float | None = None
        if np.any(ap_mask):
            ap_flux_a_jy = float(np.sum(arr_a[ap_mask]) / beam_area_pix)
            ap_flux_r_jy = float(np.sum(arr_r[ap_mask]) / beam_area_pix)
            if abs(ap_flux_r_jy) > 1e-15:
                ap_flux_ratio = float(ap_flux_a_jy / ap_flux_r_jy)

        rms_a = _compute_native_offsource_rms(chosen_a, eff_pb_min, extract_config)
        rms_r = _compute_native_offsource_rms(chosen_r, eff_pb_min, extract_config)
        rms_ratio = (
            float(rms_a / rms_r)
            if (rms_a is not None and rms_r is not None and rms_r > 1e-15)
            else None
        )

        bowl_a = _compute_negative_bowl_ratio(chosen_a, eff_pb_min)
        bowl_r = _compute_negative_bowl_ratio(chosen_r, eff_pb_min)
        bowl_delta = float(bowl_a - bowl_r) if (bowl_a is not None and bowl_r is not None) else None

        return sanitize_json_tree(
            {
                "status": "ok",
                "product_compared": product_kind,
                "agent_image": img_a_src.name,
                "ref_image": img_r_src.name,
                "beam_smoothed": beam_smoothed,
                "regridded": need_regrid,
                "common_beam_arcsec": {
                    "major": round(cb_maj, 6),
                    "minor": round(cb_min, 6),
                    "pa_deg": round(cb_pa, 4),
                },
                "n_common_pixels": n_common_pix,
                "ncc": round(ncc, 8),
                "agent_peak_jy_per_beam": round(peak_a, 6),
                "ref_peak_jy_per_beam": round(peak_r, 6),
                "peak_flux_ratio": round(peak_ratio, 6) if peak_ratio is not None else None,
                "agent_aperture_flux_jy": (
                    round(ap_flux_a_jy, 6) if ap_flux_a_jy is not None else None
                ),
                "ref_aperture_flux_jy": (
                    round(ap_flux_r_jy, 6) if ap_flux_r_jy is not None else None
                ),
                "aperture_flux_ratio": (
                    round(ap_flux_ratio, 6) if ap_flux_ratio is not None else None
                ),
                "agent_offsource_rms_jy": round(rms_a, 8) if rms_a is not None else None,
                "ref_offsource_rms_jy": round(rms_r, 8) if rms_r is not None else None,
                "offsource_rms_ratio": round(rms_ratio, 6) if rms_ratio is not None else None,
                "agent_negative_bowl_ratio": round(bowl_a, 6) if bowl_a is not None else None,
                "ref_negative_bowl_ratio": round(bowl_r, 6) if bowl_r is not None else None,
                "negative_bowl_delta": round(bowl_delta, 6) if bowl_delta is not None else None,
            }
        )
    finally:
        shutil.rmtree(run_scratch, ignore_errors=True)


# ==============================================================================
# 4. Unified Workspace & Batch Job Diff Entrypoints + CLI
# ==============================================================================


def diff_workspaces(
    agent_dir: Path,
    ref_dir: Path,
    *,
    stage: str | None = None,
    base_task: str | None = None,
    extract_config: ExtractConfig | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Compare an agent `/workspace` (or Harbor `<trial_dir>`) against a reference checkpoint."""
    _ensure_casa_env()
    raw_agent = Path(agent_dir).resolve()
    raw_ref = Path(ref_dir).resolve()
    reject_slash_tmp(raw_agent, label="agent_dir")
    reject_slash_tmp(raw_ref, label="ref_dir")

    inferred_bt, inferred_stage, inferred_suite = _infer_trial_meta(raw_agent)
    eff_bt = base_task or inferred_bt
    eff_stage = stage or inferred_stage
    eff_suite = inferred_suite
    if (not eff_suite or eff_suite == "unknown") and eff_bt:
        _, _, _, parsed_s, parsed_stg, _ = parse_task_variant_with_pack(eff_bt)
        if parsed_s != "unknown":
            eff_suite = parsed_s
        if not eff_stage and parsed_stg != "unknown":
            eff_stage = parsed_stg

    ws_a = resolve_workspace_dir(raw_agent)
    ms_ref_dir, cal_ref_dir, img_ref_dir, matched_mf = _resolve_reference_layout(
        raw_ref, base_task=eff_bt, stage=eff_stage, suite=eff_suite
    )

    eff_ec = extract_config
    if eff_ec is None:
        mf = matched_mf or _resolve_suite_manifest(base_task=eff_bt, suite=eff_suite)
        if mf is not None:
            if eff_bt and eff_bt in mf.tasks:
                eff_ec = mf.to_extract_config(eff_bt)
            else:
                eff_ec = ExtractConfig(
                    suite_id=mf.suite.suite_id,
                    split=mf.suite.split,
                    telescope=mf.suite.telescope,
                    pol_basis=mf.suite.pol_basis,
                    stages=[eff_stage or "e2e"],
                    allow_virtual_model=False,
                    target_retention_excludes_dead_antenna=bool(
                        mf.roles.target_retention_excludes_dead_antenna
                    ),
                    roles=mf.roles.model_copy(deep=True),
                    geometry=mf.geometry.model_copy(deep=True),
                )

    ms_a = _find_primary_ms(ws_a)
    ms_r = _find_primary_ms(ms_ref_dir)

    vis_diff: dict[str, Any] | None = None
    if ms_a is not None and ms_r is not None and eff_stage != "inspect":
        vis_diff = diff_visibilities(ms_a, ms_r, stage=eff_stage, extract_config=eff_ec)

    cal_diff: dict[str, Any] | None = None
    if eff_stage != "inspect":
        cand_cal = diff_caltables(
            ws_a,
            cal_ref_dir,
            ms_agt_path=ms_a,
            ms_ref_path=ms_r,
            extract_config=eff_ec,
        )
        if cand_cal.get("status") != "no_caltables_found":
            cal_diff = cand_cal

    img_diff: dict[str, Any] | None = None
    if eff_stage != "inspect":
        cand_img = diff_images(
            ws_a,
            img_ref_dir,
            extract_config=eff_ec,
            scratch_root=scratch_root,
        )
        if cand_img.get("status") != "missing_image_set":
            img_diff = cand_img

    return sanitize_json_tree(
        {
            "schema_version": "0.2",
            "agent_workspace": str(ws_a),
            "ref_dir": str(raw_ref),
            "base_task": eff_bt,
            "stage": eff_stage,
            "visibilities": vis_diff,
            "caltables": cal_diff,
            "images": img_diff,
        }
    )


def diff_job_directory(
    job_dir: Path,
    ref_dir: Path,
    *,
    output_dir: Path | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Run `diff_workspaces` across all retained trial workspaces in `job_dir` -> `<job_dir>/analysis/diffs/`."""
    jdir = Path(job_dir).resolve()
    rdir = Path(ref_dir).resolve()
    reject_slash_tmp(jdir, label="job_dir")
    reject_slash_tmp(rdir, label="ref_dir")

    diffs_dir = (
        Path(output_dir).resolve() if output_dir is not None else (jdir / "analysis" / "diffs")
    )
    reject_slash_tmp(diffs_dir, label="output_dir")
    diffs_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped_no_workspace: list[str] = []

    for _parent_job, tdir in discover_trial_dirs([jdir]):
        res = _read_json_dict(tdir / "result.json")
        tname = (
            str(res.get("trial_name")).strip()
            if isinstance(res, dict) and isinstance(res.get("trial_name"), str)
            else tdir.name
        )
        ws_cand = resolve_workspace_dir(tdir)
        if ws_cand == tdir or not ws_cand.is_dir():
            # Fallback: export verifier-time `metrics.json["diff_summary"]` if workspace was torn down
            ver_metrics = _read_json_dict(tdir / "verifier" / "metrics.json")
            if ver_metrics is None and (tdir / "steps").is_dir():
                with contextlib.suppress(OSError):
                    for sdir in reversed(
                        sorted(d for d in (tdir / "steps").iterdir() if d.is_dir())
                    ):
                        cand_m = _read_json_dict(sdir / "verifier" / "metrics.json")
                        if cand_m is not None:
                            ver_metrics = cand_m
                            break
            saved_ds = ver_metrics.get("diff_summary") if isinstance(ver_metrics, dict) else None
            if isinstance(saved_ds, dict) and saved_ds:
                exported = dict(saved_ds)
                exported["trial_name"] = tname
                exported.setdefault("source", "verifier_metrics_json")
                out_path = diffs_dir / f"{tname}.json"
                _atomic_write_json(out_path, exported)
                written.append(str(out_path))
                continue
            skipped_no_workspace.append(tname)
            continue
        diff_payload = diff_workspaces(tdir, rdir, scratch_root=scratch_root)
        diff_payload["trial_name"] = tname
        out_path = diffs_dir / f"{tname}.json"
        _atomic_write_json(out_path, diff_payload)
        written.append(str(out_path))

    return sanitize_json_tree(
        {
            "job_dir": str(jdir),
            "ref_dir": str(rdir),
            "diffs_dir": str(diffs_dir),
            "n_written": len(written),
            "n_skipped_no_workspace": len(skipped_no_workspace),
            "written_files": written,
            "skipped_trials": skipped_no_workspace,
        }
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radio-bench diff-run",
        description=(
            "Compare an agent workspace / Harbor trial (or batch Harbor job) against a "
            "frozen reference checkpoint across visibilities, caltables, and images."
        ),
    )
    add_config_cli_args(parser)
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=None,
        help="Agent workspace directory or Harbor <trial_dir> containing artifacts/workspace/.",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help="Batch Harbor <job_dir>; writes <job_dir>/analysis/diffs/<trial_name>.json.",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        required=True,
        help="Reference checkpoint directory (or suite checkpoint root).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Optional explicit stage hint (precal, calibrate, image, e2e).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path (single mode) or output diffs directory (batch --job-dir mode).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if (args.agent_dir is None) == (args.job_dir is None):
        print("error: specify exactly one of --agent-dir or --job-dir", file=sys.stderr)
        return 2

    try:
        cfg = config_from_cli_args(args)
        prepare_runtime_env(cfg, config_path=args.config, mutate_os_environ=True)

        if args.job_dir is not None:
            summary = diff_job_directory(
                args.job_dir,
                args.ref_dir,
                output_dir=args.output,
                scratch_root=cfg.tmp_dir,
            )
            sys.stdout.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            return 0

        payload = diff_workspaces(
            args.agent_dir,
            args.ref_dir,
            stage=args.stage,
            scratch_root=cfg.tmp_dir,
        )
        if args.output is not None:
            _atomic_write_json(args.output, payload)
        sys.stdout.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
