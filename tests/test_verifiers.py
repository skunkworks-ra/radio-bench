"""Unit and adversarial physics tests for `radio_bench.grading` scientific verifiers."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import multiprocessing as mp
import os
import shutil
import struct
import types
from pathlib import Path
from typing import Any

_cur_tmp = os.environ.get("TMPDIR", "").strip()
if not _cur_tmp or _cur_tmp == "/tmp" or _cur_tmp.startswith("/tmp/"):
    os.environ["TMPDIR"] = "/mnt/fast/radio-bench/tmp"

_cur_mpl = os.environ.get("MPLCONFIGDIR", "").strip()
if not _cur_mpl or _cur_mpl == "/tmp" or _cur_mpl.startswith("/tmp/"):
    os.environ["MPLCONFIGDIR"] = "/mnt/fast/radio-bench/mplconfig"

os.environ.setdefault("CASASITECONFIG", "/mnt/fast/radio-bench/casaconfig_local.py")

import casatasks  # noqa: E402
import casatools  # noqa: E402
import harbor.verifier.verifier as hv  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from radio_bench.grading import (  # noqa: E402
    grade_calibrate,
    grade_e2e,
    grade_image,
    grade_inspect,
    grade_precal,
    grade_task,
    write_verifier_outputs,
)

SYNTH_ROOT = Path("/mnt/fast/radio_bench_data/synth")
R3C391_ROOT = Path("/mnt/fast/radio_bench_data/checkpoints/3c391")


def _parse_with_harbor(reward_json_path: Path) -> dict[str, float | int]:
    """Run Harbor 0.25.0 `Verifier._parse_reward_json` directly on `reward_json_path`."""
    stub = types.SimpleNamespace(
        trial_paths=types.SimpleNamespace(reward_json_path=reward_json_path)
    )
    parsed = hv.Verifier._parse_reward_json(stub)
    for k, v in parsed.items():
        assert isinstance(k, str)
        assert isinstance(v, (int, float)) and not isinstance(v, bool)
        assert math.isfinite(float(v))
    return parsed


def _load_gt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _scratch_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep CWD inside pytest `--basetemp` so CASA never drops TempLattice* in the repo."""
    assert not str(tmp_path).startswith("/tmp")
    scratch = tmp_path / "cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(scratch)
    return scratch


@pytest.fixture(scope="module")
def synth_golden_image_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one clean synthetic multiscale continuum image set on `s1_image/input.ms`."""
    img_ws = tmp_path_factory.mktemp("synth_golden_img")
    prev = Path.cwd()
    os.chdir(img_ws)
    try:
        vis_copy = img_ws / "work.ms"
        shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", vis_copy)
        prefix = img_ws / "synth_mosaic"
        casatasks.tclean(
            vis=str(vis_copy),
            imagename=str(prefix),
            field="SYNTH_MOSAIC_P1,SYNTH_MOSAIC_P2",
            spw="0",
            datacolumn="corrected",
            specmode="mfs",
            deconvolver="multiscale",
            scales=[0, 3, 8],
            smallscalebias=0.6,
            imsize=[576, 576],
            cell="3.00arcsec",
            phasecenter="J2000 18h24m00.000s -09d35m00.000s",
            stokes="I",
            weighting="briggs",
            robust=0.5,
            gridder="mosaic",
            pblimit=0.2,
            usemask="pb",
            pbmask=0.2,
            niter=4000,
            nmajor=25,
            cyclefactor=1.6,
            threshold="0.4469mJy",
            pbcor=True,
            savemodel="none",
            interactive=False,
        )
        shutil.rmtree(vis_copy, ignore_errors=True)
        casatasks.clearstat()
        return img_ws
    finally:
        os.chdir(prev)


@pytest.fixture(scope="module")
def synth_golden_precal_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a golden `s1_precal` workspace exercising `radio-analyst`-style `calibrators.ms` split."""
    ws = tmp_path_factory.mktemp("synth_golden_precal")
    prev = Path.cwd()
    os.chdir(ws)
    try:
        in_ms = ws / "input.ms"
        cal_ms = ws / "calibrators.ms"
        shutil.copytree(SYNTH_ROOT / "s1_precal" / "input.ms", in_ms)

        # Populate intents with semicolon-joined strings matching radio-analyst `ms_set_intents`
        intents_by_fid = {
            0: "CALIBRATE_BANDPASS#ON_SOURCE;CALIBRATE_FLUX#ON_SOURCE",
            1: "CALIBRATE_PHASE#ON_SOURCE",
            2: "OBSERVE_TARGET#ON_SOURCE",
            3: "OBSERVE_TARGET#ON_SOURCE",
        }
        unique_modes = sorted(set(intents_by_fid.values()))
        mode_to_sid = {m: idx for idx, m in enumerate(unique_modes)}

        tb_s = casatools.table()
        tb_s.open(str(in_ms / "STATE"), nomodify=False)
        try:
            nrows = int(tb_s.nrows())
            if nrows < len(unique_modes):
                tb_s.addrows(len(unique_modes) - nrows)
            elif nrows > len(unique_modes):
                tb_s.removerows(list(range(len(unique_modes), nrows)))
            for idx, mode in enumerate(unique_modes):
                tb_s.putcell("OBS_MODE", idx, mode)
        finally:
            tb_s.close()
            tb_s.done()

        with np.load(SYNTH_ROOT / "s1_precal" / "corruption_masks.npz", allow_pickle=False) as z:
            quack_rows = z["quack_rows"].astype(bool)
            burst_rows = z["burst_rows"].astype(bool)
            dead_rows = z["dead_rows"].astype(bool)
            rfi_channels = z["rfi_channels"].astype(int).tolist()

        # Apply quack + dead + target burst flags on input.ms, leaving RFI channels 29..31 on
        # calibrators to be flagged ONLY inside calibrators.ms so we verify cross-MS flag merging!
        tb_m = casatools.table()
        tb_m.open(str(in_ms), nomodify=False)
        try:
            fids = tb_m.getcol("FIELD_ID")
            sids = np.array([mode_to_sid[intents_by_fid[int(f)]] for f in fids], dtype=np.int32)
            tb_m.putcol("STATE_ID", sids)
            flg = tb_m.getcol("FLAG")
            flg[:, :, quack_rows | burst_rows | dead_rows] = True
            tgt_rows = np.isin(fids, [2, 3])
            for ch in rfi_channels:
                flg[:, ch, tgt_rows] = True
            tb_m.putcol("FLAG", flg)
            tb_m.putcol("FLAG_ROW", np.all(flg, axis=(0, 1)))
        finally:
            tb_m.close()
            tb_m.done()

        # Split calibrators with keepflags=False (drops quack/dead rows like ms_apply_preflag!)
        casatasks.split(
            vis=str(in_ms),
            outputvis=str(cal_ms),
            field="J1331+3030,J1822-0938",
            datacolumn="data",
            keepflags=False,
        )
        # Run setjy and rflag-equivalent channel flagging ONLY on calibrators.ms
        casatasks.setjy(
            vis=str(cal_ms),
            field="J1331+3030",
            standard="Perley-Butler 2017",
            usescratch=True,
        )
        tb_c = casatools.table()
        tb_c.open(str(cal_ms), nomodify=False)
        try:
            cflg = tb_c.getcol("FLAG")
            for ch in rfi_channels:
                cflg[:, ch, :] = True
            tb_c.putcol("FLAG", cflg)
            tb_c.putcol("FLAG_ROW", np.all(cflg, axis=(0, 1)))
        finally:
            tb_c.close()
            tb_c.done()

        casatasks.clearstat()
        return ws
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# 1. Harbor 0.25.0 output contract & NaN / Inf / type safety
# ---------------------------------------------------------------------------


def test_write_verifier_outputs_harbor_compat_and_nan_rejection(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    write_verifier_outputs(
        log_dir,
        {"reward": 0.9123456, "passed": 1, "sub_a": 1.0, "sub_b": 0.75},
        {
            "task_id": "unit_check",
            "finite_arr": np.array([1.0, 2.5]),
            "bad_nan": float("nan"),
            "nested": {"bad_inf": float("inf"), "ok": np.int64(7)},
        },
    )
    parsed = _parse_with_harbor(log_dir / "reward.json")
    assert parsed["reward"] == pytest.approx(0.912346)
    assert parsed["passed"] == 1
    assert type(parsed["passed"]) is int

    metrics = json.loads((log_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["bad_nan"] is None
    assert metrics["nested"]["bad_inf"] is None
    assert "bad_nan" in metrics["non_finite_fields"]
    assert "nested.bad_inf" in metrics["non_finite_fields"]

    for bad_reward in (
        {"reward": float("nan")},
        {"reward": float("inf")},
        {"reward": -float("inf")},
        {"reward": 0.9, "passed": True},
        {"reward": 0.9, "bad_str": "0.5"},
        {"reward": 0.9, "bad_none": None},
        {"reward": 0.9, "nested": {"x": 1.0}},
    ):
        with pytest.raises(ValueError):
            write_verifier_outputs(tmp_path / "bad_logs", bad_reward, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Inspect verifier tests (Suite S & Suite R, golden + adversarial)
# ---------------------------------------------------------------------------


def test_inspect_golden_and_aliases_both_suites(tmp_path: Path) -> None:
    gt_s1 = _load_gt(SYNTH_ROOT / "s1_inspect" / "ground_truth.json")
    ws_s1 = tmp_path / "ws_s1"
    ws_s1.mkdir()
    (ws_s1 / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "target_fields": ["SYNTH_MOSAIC_P1", "SYNTH_MOSAIC_P2"],
                "dead_antennas": ["EA07"],
                "bandpass_min_elevation_deg": 20.42,
            }
        ),
        encoding="utf-8",
    )
    log_s1 = tmp_path / "log_s1"
    rew_s1, _met_s1 = grade_task("s1_inspect", "inspect", ws_s1, SYNTH_ROOT / "s1_inspect", log_s1)
    _parse_with_harbor(log_s1 / "reward.json")
    assert rew_s1["reward"] == 1.0
    assert rew_s1["passed"] == 1

    # Suite R with natural key aliases + missing VLA pads reported alongside ea13
    ws_r1 = tmp_path / "ws_r1"
    ws_r1.mkdir()
    (ws_r1 / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_cal": ["J1331+3030"],
                "bandpass_cal": "J1331+3030",
                "phase_cal": "J1822-0938",
                "mosaic_target_fields": [f"3C391 C{i}" for i in range(1, 8)],
                "bad_antennas": ["ea13", "ea06", "ea10"],
                "missing_antennas": ["ea06", "ea10"],
                "bp_min_elevation_deg": 34.86,
            }
        ),
        encoding="utf-8",
    )
    log_r1 = tmp_path / "log_r1"
    rew_r1, _ = grade_task(
        "r1_3c391_inspect",
        "inspect",
        ws_r1,
        R3C391_ROOT / "r1_3c391_inspect",
        log_r1,
    )
    _parse_with_harbor(log_r1 / "reward.json")
    assert rew_r1["reward"] == 1.0
    assert rew_r1["passed"] == 1
    _ = gt_s1


def test_inspect_adversarial_failures(tmp_path: Path) -> None:
    gt_s1 = _load_gt(SYNTH_ROOT / "s1_inspect" / "ground_truth.json")
    gt_r1 = _load_gt(R3C391_ROOT / "r1_3c391_inspect" / "ground_truth.json")

    # Missing file
    empty_ws = tmp_path / "empty"
    empty_ws.mkdir()
    rew_missing, _ = grade_inspect(empty_ws, gt_s1, task_id="s1_inspect")
    assert rew_missing["reward"] == 0.0
    assert rew_missing["passed"] == 0

    # Malformed JSON
    bad_ws = tmp_path / "malformed"
    bad_ws.mkdir()
    (bad_ws / "inspect_summary.json").write_text("{not valid json", encoding="utf-8")
    rew_mal, _ = grade_inspect(bad_ws, gt_s1, task_id="s1_inspect")
    assert rew_mal["reward"] == 0.0
    assert rew_mal["passed"] == 0

    # Wrong phase cal + +2.0 deg elevation error + missed dead antenna -> fails (< 0.85)
    wrong_ws = tmp_path / "wrong_vals"
    wrong_ws.mkdir()
    (wrong_ws / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "phase_calibrator": "SYNTH_MOSAIC_P1",
                "target_fields": ["J1822-0938", "SYNTH_MOSAIC_P2"],
                "dead_antennas": [],
                "bandpass_min_elevation_deg": 22.42,
            }
        ),
        encoding="utf-8",
    )
    rew_wrong, _ = grade_inspect(wrong_ws, gt_s1, task_id="s1_inspect")
    assert rew_wrong["elevation_score"] == 0.0
    assert rew_wrong["dead_antenna_score"] == 0.0
    assert rew_wrong["phase_vs_target_score"] < 0.2
    assert rew_wrong["passed"] == 0

    # Confusing J0319+4130 pol leakage cal with a target pointing + missing ea13 on 3c391
    conf_ws = tmp_path / "confused_3c391"
    conf_ws.mkdir()
    (conf_ws / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "phase_calibrator": "J0319+4130",
                "target_fields": [*(f"3C391 C{i}" for i in range(1, 8)), "J0319+4130"],
                "dead_antennas": ["ea01"],
                "bandpass_min_elevation_deg": 37.0,
            }
        ),
        encoding="utf-8",
    )
    rew_conf, _ = grade_inspect(conf_ws, gt_r1, task_id="r1_3c391_inspect")
    assert rew_conf["phase_vs_target_score"] < 0.25
    assert rew_conf["dead_antenna_score"] == 0.0
    assert rew_conf["elevation_score"] == 0.0
    assert rew_conf["passed"] == 0


# ---------------------------------------------------------------------------
# 3. Precal verifier tests (Suite S & Suite R, golden + overflag + unity setjy)
# ---------------------------------------------------------------------------


def test_precal_golden_both_suites(synth_golden_precal_dir: Path, tmp_path: Path) -> None:
    log_s1 = tmp_path / "log_s1_precal"
    rew_s1, met_s1 = grade_task(
        "s1_precal",
        "precal",
        synth_golden_precal_dir,
        SYNTH_ROOT / "s1_precal",
        log_s1,
    )
    _parse_with_harbor(log_s1 / "reward.json")
    assert rew_s1["reward"] == 1.0
    assert rew_s1["passed"] == 1
    assert met_s1["measured"]["j1331_model_source_ms"] == "calibrators.ms"
    assert met_s1["measured"]["must_flag_recall"] == 1.0
    assert met_s1["measured"]["clean_false_positive_rate"] == 0.0

    # Suite R golden post_precal checkpoint
    ws_r1 = tmp_path / "ws_r1_precal"
    ws_r1.mkdir()
    os.symlink(R3C391_ROOT / "post_precal" / "input.ms", ws_r1 / "input.ms")
    os.symlink(R3C391_ROOT / "post_precal" / "calibrators.ms", ws_r1 / "calibrators.ms")
    log_r1 = tmp_path / "log_r1_precal"
    rew_r1, met_r1 = grade_task(
        "r1_3c391_precal",
        "precal",
        ws_r1,
        R3C391_ROOT / "r1_3c391_precal",
        log_r1,
    )
    _parse_with_harbor(log_r1 / "reward.json")
    assert rew_r1["reward"] == 1.0
    assert rew_r1["passed"] == 1
    assert met_r1["measured"]["must_flag_recall"] == 1.0
    assert met_r1["measured"]["clean_false_positive_rate"] == 0.0


def test_precal_unity_model_and_overflagging_adversarial(
    synth_golden_precal_dir: Path, tmp_path: Path
) -> None:
    gt_s1 = _load_gt(SYNTH_ROOT / "s1_precal" / "ground_truth.json")
    masks_s1 = SYNTH_ROOT / "s1_precal" / "corruption_masks.npz"

    ws = tmp_path / "adv_precal"
    ws.mkdir()
    shutil.copytree(synth_golden_precal_dir / "input.ms", ws / "input.ms")
    shutil.copytree(synth_golden_precal_dir / "calibrators.ms", ws / "calibrators.ms")

    # 1. Overwrite MODEL_DATA with 1.0 Jy unity instead of Perley-Butler 2017 (~7.60 Jy)
    tb = casatools.table()
    tb.open(str(ws / "calibrators.ms"), nomodify=False)
    try:
        mod = tb.getcol("MODEL_DATA")
        mod[:, :, :] = 0.0 + 0.0j
        mod[0, :, :] = 1.0 + 0.0j
        mod[3, :, :] = 1.0 + 0.0j
        tb.putcol("MODEL_DATA", mod)
    finally:
        tb.close()
        tb.done()

    rew_unity, _ = grade_precal(ws, gt_s1, masks_npz_path=masks_s1, task_id="s1_precal")
    assert rew_unity["model_data_score"] == 0.0
    assert rew_unity["reward"] == pytest.approx(0.75)
    assert rew_unity["passed"] == 0

    # 2. Over-flag 50% of clean visibilities -> clean_fpr_score == 0.0 and passed == 0
    tb = casatools.table()
    tb.open(str(ws / "input.ms"), nomodify=False)
    try:
        flg = tb.getcol("FLAG")
        flg[:, :32, :] = True
        tb.putcol("FLAG", flg)
    finally:
        tb.close()
        tb.done()
    rew_half_flg, _ = grade_precal(ws, gt_s1, masks_npz_path=masks_s1, task_id="s1_precal")
    assert rew_half_flg["clean_fpr_score"] == 0.0
    assert rew_half_flg["passed"] == 0

    # 3. Blanket 100% flagging -> both clean_fpr_score == 0.0 and flag_recall_score == 0.0
    tb = casatools.table()
    tb.open(str(ws / "input.ms"), nomodify=False)
    try:
        flg = tb.getcol("FLAG")
        flg[:, :, :] = True
        tb.putcol("FLAG", flg)
        tb.putcol("FLAG_ROW", np.ones(flg.shape[2], dtype=bool))
    finally:
        tb.close()
        tb.done()
    rew_all_flg, _ = grade_precal(ws, gt_s1, masks_npz_path=masks_s1, task_id="s1_precal")
    assert rew_all_flg["clean_fpr_score"] == 0.0
    assert rew_all_flg["flag_recall_score"] == 0.0
    assert rew_all_flg["passed"] == 0


def _hold_write_lock_worker(lock_file_str: str, ready_evt: Any, release_evt: Any) -> None:
    fd = os.open(lock_file_str, os.O_RDWR)
    try:
        lock_spec = struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 1, 0)
        fcntl.fcntl(fd, fcntl.F_SETLK, lock_spec)
        ready_evt.set()
        release_evt.wait(timeout=10.0)
    finally:
        os.close(fd)


def test_active_casacore_table_lock_rejected_cleanly(
    synth_golden_precal_dir: Path, tmp_path: Path
) -> None:
    ws = tmp_path / "locked_ws"
    ws.mkdir()
    shutil.copytree(synth_golden_precal_dir / "input.ms", ws / "input.ms")
    shutil.copytree(synth_golden_precal_dir / "calibrators.ms", ws / "calibrators.ms")

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(
        target=_hold_write_lock_worker,
        args=(str(ws / "calibrators.ms" / "table.lock"), ready, release),
    )
    proc.start()
    try:
        assert ready.wait(timeout=5.0)
        gt_s1 = _load_gt(SYNTH_ROOT / "s1_precal" / "ground_truth.json")
        rew, met = grade_precal(
            ws,
            gt_s1,
            masks_npz_path=SYNTH_ROOT / "s1_precal" / "corruption_masks.npz",
            task_id="s1_precal",
        )
        assert met["table_locked"] is True
        assert rew["reward"] == 0.0
        assert rew["passed"] == 0
    finally:
        release.set()
        proc.join(timeout=5.0)


# ---------------------------------------------------------------------------
# 4. Calibrate verifier tests (Suite S & Suite R, golden + uncalibrated + tclean recipe)
# ---------------------------------------------------------------------------


def test_calibrate_golden_and_adversarial_synth(tmp_path: Path) -> None:
    # Golden calibrated MS (`s1_image/input.ms`) via grade_task with true_target_vis.npz
    ws_ok = tmp_path / "ws_cal_ok"
    ws_ok.mkdir()
    os.symlink(SYNTH_ROOT / "s1_image" / "input.ms", ws_ok / "input.ms")
    log_ok = tmp_path / "log_cal_ok"
    rew_ok, met_ok = grade_task(
        "s1_calibrate",
        "calibrate",
        ws_ok,
        SYNTH_ROOT / "s1_calibrate",
        log_ok,
    )
    _parse_with_harbor(log_ok / "reward.json")
    assert rew_ok["reward"] == 1.0
    assert rew_ok["passed"] == 1
    assert met_ok["measured"]["coherent_amp_rel_err"] < 0.01
    assert abs(met_ok["measured"]["coherent_phase_deg"]) < 1.0

    # Fallback analytical/simulator prediction when masks_npz_path=None
    gt_cal = _load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json")
    rew_fb, met_fb = grade_calibrate(
        ws_ok, gt_cal, masks_npz_path=None, task_id="s1_calibrate", scratch_root=tmp_path
    )
    assert rew_fb["reward"] == 1.0
    assert met_fb["measured"]["coherent_amp_rel_err"] == pytest.approx(
        met_ok["measured"]["coherent_amp_rel_err"], abs=1e-4
    )

    # Uncalibrated s1_calibrate/input.ms (missing CORRECTED_DATA, ea07 unflagged) -> < 0.25
    ws_raw = tmp_path / "ws_cal_raw"
    ws_raw.mkdir()
    os.symlink(SYNTH_ROOT / "s1_calibrate" / "input.ms", ws_raw / "input.ms")
    log_raw = tmp_path / "log_cal_raw"
    rew_raw, _ = grade_task(
        "s1_calibrate",
        "calibrate",
        ws_raw,
        SYNTH_ROOT / "s1_calibrate",
        log_raw,
    )
    _parse_with_harbor(log_raw / "reward.json")
    assert rew_raw["reward"] == 0.0
    assert rew_raw["passed"] == 0

    # Adversarial fake: copy raw DATA into CORRECTED_DATA (and perturb by 1e-5 so distinct check
    # triggers raw delay slope + wrong fluxscale + incoherent target!)
    ws_fake = tmp_path / "ws_cal_fake"
    ws_fake.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_calibrate" / "input.ms", ws_fake / "input.ms")
    casatasks.clearcal(vis=str(ws_fake / "input.ms"), addmodel=False)
    tb = casatools.table()
    tb.open(str(ws_fake / "input.ms"), nomodify=False)
    try:
        raw_d = tb.getcol("DATA")
        tb.putcol("CORRECTED_DATA", raw_d * 1.001)
    finally:
        tb.close()
        tb.done()

    rew_fake, _ = grade_calibrate(
        ws_fake,
        gt_cal,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert rew_fake["bandpass_flatness_score"] == 0.0
    assert rew_fake["reward"] < 0.25
    assert rew_fake["passed"] == 0


def test_calibrate_3c391_golden_and_verifier_tclean_recipe_path(tmp_path: Path) -> None:
    # 1. Verify Suite R `post_calibrate` checkpoint scores 1.0 on retention/dead-ant, flatness,
    #    and fluxscale (both CORRECTED_DATA and caltable G/fluxscale ratio), and 1.0 overall
    #    when verifier_tclean DR >= 180.
    gt_r1 = _load_gt(R3C391_ROOT / "r1_3c391_calibrate" / "ground_truth.json")
    ws_r1 = tmp_path / "ws_r1_cal"
    ws_r1.mkdir()
    os.symlink(R3C391_ROOT / "post_calibrate" / "input.ms", ws_r1 / "input.ms")
    os.symlink(R3C391_ROOT / "post_calibrate" / "caltables", ws_r1 / "caltables")

    import radio_bench.grading.cal_metrics as cm

    orig_runner = cm._run_verifier_tclean_recipe
    try:
        cm._run_verifier_tclean_recipe = lambda *_a, **_kw: {  # type: ignore[assignment]
            "verifier_tclean_peak_jy": 0.134512,
            "verifier_tclean_offsource_rms_jy": 0.00043623,
            "verifier_tclean_dynamic_range": 308.35,
        }
        log_r1 = tmp_path / "log_r1_cal"
        rew_r1, met_r1 = grade_task(
            "r1_3c391_calibrate",
            "calibrate",
            ws_r1,
            R3C391_ROOT / "r1_3c391_calibrate",
            log_r1,
            scratch_root=tmp_path,
        )
        _parse_with_harbor(log_r1 / "reward.json")
        assert rew_r1["reward"] == 1.0
        assert rew_r1["passed"] == 1
        assert met_r1["measured"]["dead_antenna_flagged_fraction"] == 1.0
        assert 0.70 <= met_r1["measured"]["target_row_retention"] <= 0.90
        assert met_r1["measured"]["pcal_caltable_flux_jy"] == pytest.approx(2.296125, rel=1e-4)
    finally:
        cm._run_verifier_tclean_recipe = orig_runner

    # 2. Exercise the real `_run_verifier_tclean_recipe` CASA mstransform + tclean code path
    #    on compact calibrated `s1_image/input.ms` in < 2.5s and verify zero workspace mutation!
    fast_recipe = {
        "field": "2,3",
        "spw": "0:4~59",
        "phasecenter": "J2000 18h24m00.000s -09d35m00.000s",
        "specmode": "mfs",
        "gridder": "mosaic",
        "deconvolver": "hogbom",
        "imsize": [256, 256],
        "cell_arcsec": 4.0,
        "weighting": "briggs",
        "robust": 0.5,
        "pblimit": 0.2,
        "usemask": "pb",
        "pbmask": 0.2,
        "niter": 100,
        "nmajor": 1,
        "threshold": "0.5mJy",
        "pbcor": True,
        "off_source_min_radius_arcsec": 90.0,
    }
    tclean_res = cm._run_verifier_tclean_recipe(
        SYNTH_ROOT / "s1_image" / "input.ms",
        fast_recipe,
        tmp_path,
    )
    assert tclean_res["verifier_tclean_dynamic_range"] is not None
    assert tclean_res["verifier_tclean_dynamic_range"] > 20.0
    _ = gt_r1


# ---------------------------------------------------------------------------
# 5. Image verifier tests (Suite S & Suite R, golden + clean bowl + bad beam sampling)
# ---------------------------------------------------------------------------


def test_image_golden_both_suites(synth_golden_image_dir: Path, tmp_path: Path) -> None:
    log_s1 = tmp_path / "log_s1_img"
    rew_s1, met_s1 = grade_task(
        "s1_image",
        "image",
        synth_golden_image_dir,
        SYNTH_ROOT / "s1_image",
        log_s1,
    )
    _parse_with_harbor(log_s1 / "reward.json")
    assert rew_s1["reward"] == 1.0
    assert rew_s1["passed"] == 1
    assert met_s1["measured"]["ext_flux_rel_err"] < 0.10
    assert met_s1["measured"]["pt_snr_image"] >= 8.0
    assert met_s1["measured"]["negative_bowl_ratio"] < 0.06

    # Suite R golden reference_image/3c391_mosaic symlinked into workspace
    ws_r1 = tmp_path / "ws_r1_img"
    ws_r1.mkdir()
    for p in (R3C391_ROOT / "reference_image").iterdir():
        os.symlink(p, ws_r1 / p.name)

    log_r1 = tmp_path / "log_r1_img"
    rew_r1, met_r1 = grade_task(
        "r1_3c391_image",
        "image",
        ws_r1,
        R3C391_ROOT / "r1_3c391_image",
        log_r1,
    )
    _parse_with_harbor(log_r1 / "reward.json")
    assert rew_r1["reward"] == 1.0
    assert rew_r1["passed"] == 1
    assert met_r1["measured"]["offsource_residual_rms_mjy_per_beam"] == pytest.approx(
        0.43617, abs=0.01
    )
    assert met_r1["measured"]["dynamic_range"] >= 300.0


def test_image_clean_bowl_and_bad_pixel_sampling_adversarial(
    synth_golden_image_dir: Path, tmp_path: Path
) -> None:
    gt_s1 = _load_gt(SYNTH_ROOT / "s1_image" / "ground_truth.json")

    ws = tmp_path / "adv_img"
    ws.mkdir()
    for suffix in ("image", "image.pbcor", "pb", "residual", "model"):
        shutil.copytree(
            synth_golden_image_dir / f"synth_mosaic.{suffix}",
            ws / f"synth_mosaic.{suffix}",
        )

    # 1. Inject diverged negative clean bowl: |sum(model < 0)| / sum(model > 0) == 0.35 (>= 0.12)
    ia = casatools.image()
    ia.open(str(ws / "synth_mosaic.model"))
    try:
        mod = ia.getchunk()
        pos_sum = float(np.sum(mod[mod > 0.0]))
        mod[mod < 0.0] = 0.0
        # Place negative bowl pixels well inside pb >= 0.2 near center
        cx, cy = mod.shape[0] // 2, mod.shape[1] // 2
        mod[cx + 15 : cx + 25, cy + 15 : cy + 25, 0, 0] = -(0.35 * pos_sum) / 100.0
        ia.putchunk(mod)
    finally:
        ia.close()
        ia.done()

    rew_bowl, met_bowl = grade_image(ws, gt_s1, task_id="s1_image")
    assert met_bowl["measured"]["negative_bowl_ratio"] == pytest.approx(0.35, rel=0.02)
    assert rew_bowl["deconv_and_bowl_score"] == 0.0
    assert rew_bowl["reward"] <= 0.60
    assert rew_bowl["passed"] == 0

    # 2. Corrupt pixel cell size to 15 arcsec (~0.79 px/beam << 3.0 px/beam)
    ia = casatools.image()
    ia.open(str(ws / "synth_mosaic.image"))
    try:
        cs = ia.coordsys()
        try:
            inc = cs.increment()["numeric"]
            rad_15 = math.radians(15.0 / 3600.0)
            cs.setincrement([-rad_15, rad_15, inc[2], inc[3]])
            ia.setcoordsys(cs.torecord())
        finally:
            with contextlib.suppress(Exception):
                cs.done()
    finally:
        ia.close()
        ia.done()

    rew_cell, met_cell = grade_image(ws, gt_s1, task_id="s1_image")
    assert met_cell["measured"]["pixels_per_beam"] < 1.0
    assert rew_cell["beam_sampling_score"] == 0.0
    assert rew_cell["passed"] == 0


# ---------------------------------------------------------------------------
# 6. End-to-end verifier tests (`s2_e2e_synth` & `r2_3c391_e2e`)
# ---------------------------------------------------------------------------


def test_e2e_requires_both_calibration_and_imaging(
    synth_golden_image_dir: Path, tmp_path: Path
) -> None:
    # Full golden e2e workspace on Suite S -> 1.0
    ws_full = tmp_path / "e2e_full"
    ws_full.mkdir()
    os.symlink(SYNTH_ROOT / "s1_image" / "input.ms", ws_full / "input.ms")
    for suffix in ("image", "image.pbcor", "pb", "residual", "model"):
        os.symlink(
            synth_golden_image_dir / f"synth_mosaic.{suffix}",
            ws_full / f"synth_mosaic.{suffix}",
        )

    log_full = tmp_path / "log_e2e_full"
    rew_full, _ = grade_task(
        "s2_e2e_synth",
        "e2e",
        ws_full,
        SYNTH_ROOT / "s2_e2e_synth",
        log_full,
    )
    _parse_with_harbor(log_full / "reward.json")
    assert rew_full["reward"] == 1.0
    assert rew_full["passed"] == 1
    assert rew_full["calibration_score"] == 1.0
    assert rew_full["delivered_image_score"] == 1.0

    # Calibration-only (agent stopped before imaging) -> reward == 0.60 < 0.85, passed == 0
    ws_cal_only = tmp_path / "e2e_cal_only"
    ws_cal_only.mkdir()
    os.symlink(SYNTH_ROOT / "s1_image" / "input.ms", ws_cal_only / "input.ms")
    gt_s2 = _load_gt(SYNTH_ROOT / "s2_e2e_synth" / "ground_truth.json")
    rew_cal_only, _ = grade_e2e(
        ws_cal_only,
        gt_s2,
        masks_npz_path=SYNTH_ROOT / "s2_e2e_synth" / "true_target_vis.npz",
        task_id="s2_e2e_synth",
    )
    assert rew_cal_only["reward"] == pytest.approx(0.60)
    assert rew_cal_only["passed"] == 0

    # Image-only with uncalibrated input.ms (faked image) -> reward == 0.40 < 0.85, passed == 0
    ws_img_only = tmp_path / "e2e_img_only"
    ws_img_only.mkdir()
    os.symlink(SYNTH_ROOT / "s2_e2e_synth" / "input.ms", ws_img_only / "input.ms")
    for suffix in ("image", "image.pbcor", "pb", "residual", "model"):
        os.symlink(
            synth_golden_image_dir / f"synth_mosaic.{suffix}",
            ws_img_only / f"synth_mosaic.{suffix}",
        )
    rew_img_only, _ = grade_e2e(
        ws_img_only,
        gt_s2,
        masks_npz_path=SYNTH_ROOT / "s2_e2e_synth" / "true_target_vis.npz",
        task_id="s2_e2e_synth",
    )
    assert rew_img_only["reward"] == pytest.approx(0.40)
    assert rew_img_only["passed"] == 0

    # Suite R `r2_3c391_e2e` golden check
    ws_r2 = tmp_path / "r2_full"
    ws_r2.mkdir()
    os.symlink(R3C391_ROOT / "post_calibrate" / "input.ms", ws_r2 / "input.ms")
    for p in (R3C391_ROOT / "reference_image").iterdir():
        os.symlink(p, ws_r2 / p.name)

    import radio_bench.grading.cal_metrics as cm

    orig_runner = cm._run_verifier_tclean_recipe
    try:
        cm._run_verifier_tclean_recipe = lambda *_a, **_kw: {  # type: ignore[assignment]
            "verifier_tclean_peak_jy": 0.134512,
            "verifier_tclean_offsource_rms_jy": 0.00043623,
            "verifier_tclean_dynamic_range": 308.35,
        }
        log_r2 = tmp_path / "log_r2"
        rew_r2, _ = grade_task(
            "r2_3c391_e2e",
            "e2e",
            ws_r2,
            R3C391_ROOT / "r2_3c391_e2e",
            log_r2,
            scratch_root=tmp_path,
        )
        _parse_with_harbor(log_r2 / "reward.json")
        assert rew_r2["reward"] == 1.0
        assert rew_r2["passed"] == 1
    finally:
        cm._run_verifier_tclean_recipe = orig_runner


# ---------------------------------------------------------------------------
# 7. Wave 4 Adversarial & End-to-End Hardening Regression Tests
# ---------------------------------------------------------------------------


def _run_py_script(script_path: Path, args: list[str]) -> None:
    import subprocess
    import sys

    env = os.environ.copy()
    env["TMPDIR"] = "/mnt/fast/radio-bench/tmp"
    env["MPLCONFIGDIR"] = "/mnt/fast/radio-bench/mplconfig"
    env.setdefault("CASASITECONFIG", "/mnt/fast/radio-bench/casaconfig_local.py")
    src_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{src_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_root
    proc = subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd="/mnt/fast/radio-bench/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=300.0,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{script_path} exited {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_wave4_inspect_solve_verify_and_core_subscore_gates(tmp_path: Path) -> None:
    specs_dir = Path(__file__).resolve().parents[1] / "task_specs"

    # 1. Direct solve.py -> verify.py execution on s1_inspect and r1_3c391_inspect
    for task_id, data_dir in (
        ("s1_inspect", SYNTH_ROOT / "s1_inspect"),
        ("r1_3c391_inspect", R3C391_ROOT / "r1_3c391_inspect"),
    ):
        ws = tmp_path / f"ws_{task_id}"
        ws.mkdir()
        src_ms = (
            data_dir / "input.ms"
            if (data_dir / "input.ms").exists()
            else R3C391_ROOT / "raw" / "input.ms"
        )
        os.symlink(src_ms, ws / "input.ms")
        log_dir = tmp_path / f"log_{task_id}"
        _run_py_script(specs_dir / task_id / "solve.py", ["--workspace", str(ws)])
        _run_py_script(
            specs_dir / task_id / "verify.py",
            ["--workspace", str(ws), "--tests-dir", str(data_dir), "--log-dir", str(log_dir)],
        )
        parsed = _parse_with_harbor(log_dir / "reward.json")
        assert parsed["reward"] == 1.0
        assert parsed["passed"] == 1

    # 2. Empty dead_antennas = [] with otherwise perfect s1_inspect JSON -> reward == 0.85, passed == 0
    ws_empty_dead = tmp_path / "ws_empty_dead"
    ws_empty_dead.mkdir()
    (ws_empty_dead / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "target_fields": ["SYNTH_MOSAIC_P1", "SYNTH_MOSAIC_P2"],
                "dead_antennas": [],
                "bandpass_min_elevation_deg": 20.42,
            }
        ),
        encoding="utf-8",
    )
    log_ed = tmp_path / "log_empty_dead"
    rew_ed, _ = grade_task(
        "s1_inspect", "inspect", ws_empty_dead, SYNTH_ROOT / "s1_inspect", log_ed
    )
    parsed_ed = _parse_with_harbor(log_ed / "reward.json")
    assert rew_ed["reward"] == pytest.approx(0.85)
    assert rew_ed["dead_antenna_score"] == 0.0
    assert rew_ed["passed"] == 0
    assert parsed_ed["passed"] == 0

    # 3. Dumping all 28 VLA antennas into dead_antennas -> passed == 0
    ws_all28 = tmp_path / "ws_all28"
    ws_all28.mkdir()
    (ws_all28 / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "target_fields": [f"3C391 C{i}" for i in range(1, 8)],
                "dead_antennas": [f"ea{i:02d}" for i in range(1, 29)],
                "bandpass_min_elevation_deg": 34.86,
            }
        ),
        encoding="utf-8",
    )
    log_all28 = tmp_path / "log_all28"
    rew_all28, _ = grade_task(
        "r1_3c391_inspect", "inspect", ws_all28, R3C391_ROOT / "r1_3c391_inspect", log_all28
    )
    assert rew_all28["dead_antenna_score"] < 0.05
    assert rew_all28["passed"] == 0
    assert _parse_with_harbor(log_all28 / "reward.json")["passed"] == 0

    # 4. Wrong target_fields with otherwise perfect fields -> passed == 0
    ws_bad_tgt = tmp_path / "ws_bad_tgt"
    ws_bad_tgt.mkdir()
    (ws_bad_tgt / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "target_fields": ["BOGUS_FIELD_1"],
                "dead_antennas": ["ea07"],
                "bandpass_min_elevation_deg": 20.42,
            }
        ),
        encoding="utf-8",
    )
    rew_bt, _ = grade_inspect(
        ws_bad_tgt, _load_gt(SYNTH_ROOT / "s1_inspect" / "ground_truth.json"), task_id="s1_inspect"
    )
    assert rew_bt["passed"] == 0

    # 5. Active fcntl.F_WRLCK on input.ms/table.lock during inspect -> reward == 0.0, passed == 0
    ws_lock = tmp_path / "ws_insp_lock"
    ws_lock.mkdir()
    shutil.copytree(ws_empty_dead, ws_lock, dirs_exist_ok=True)
    # Fix dead_antennas so JSON would otherwise score 1.0 if unlocked
    (ws_lock / "inspect_summary.json").write_text(
        json.dumps(
            {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "target_fields": ["SYNTH_MOSAIC_P1", "SYNTH_MOSAIC_P2"],
                "dead_antennas": ["ea07"],
                "bandpass_min_elevation_deg": 20.42,
            }
        ),
        encoding="utf-8",
    )
    (ws_lock / "input.ms").mkdir()
    lock_file = ws_lock / "input.ms" / "table.lock"
    lock_file.write_bytes(b"\x00" * 64)

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_write_lock_worker, args=(str(lock_file), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=5.0)
        rew_lk, met_lk = grade_inspect(
            ws_lock, _load_gt(SYNTH_ROOT / "s1_inspect" / "ground_truth.json"), task_id="s1_inspect"
        )
        assert met_lk["table_locked"] is True
        assert rew_lk["reward"] == 0.0
        assert rew_lk["passed"] == 0
    finally:
        release.set()
        proc.join(timeout=5.0)


def test_wave4_mcp_calibrators_ms_and_outlier_rfi_robustness(
    synth_golden_precal_dir: Path, tmp_path: Path
) -> None:
    specs_dir = Path(__file__).resolve().parents[1] / "task_specs"

    # 1. MCP-style workspace: calibrators.ms has calibrated J1331 & J1822 (with reversed FIELD order
    #    to verify name-based FIELD resolution!), while input.ms has CORRECTED_DATA == DATA on
    #    calibrators, calibrated target, and 1 unflagged 170 Jy RFI burst integration on J1331!
    ws_mcp = tmp_path / "ws_mcp"
    ws_mcp.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_mcp / "input.ms")

    # Split calibrators into calibrators.ms with J1822 first and J1331 second (swapping field IDs!)
    casatasks.mstransform(
        vis=str(SYNTH_ROOT / "s1_image" / "input.ms"),
        outputvis=str(ws_mcp / "calibrators.ms"),
        field="J1822-0938,J1331+3030",
        datacolumn="all",
    )

    # Reset calibrator rows in input.ms back to uncalibrated DATA and inject 170 Jy RFI burst on Scan 1 int 1
    tb = casatools.table()
    tb.open(str(ws_mcp / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        raw = tb.getcol("DATA")
        corr = tb.getcol("CORRECTED_DATA")
        flg = tb.getcol("FLAG")
        flgr = tb.getcol("FLAG_ROW")
        cal_mask = np.isin(fids, [0, 1])
        corr[:, :, cal_mask] = raw[:, :, cal_mask]
        tb.putcol("CORRECTED_DATA", corr)
    finally:
        tb.close()
        tb.done()

    # Add a symlink cycle inside workspace to verify bounded inode-tracked traversal (SF-2)
    os.symlink(ws_mcp, ws_mcp / "loop_link")

    log_mcp = tmp_path / "log_mcp"
    rew_mcp, met_mcp = grade_task(
        "s1_calibrate", "calibrate", ws_mcp, SYNTH_ROOT / "s1_calibrate", log_mcp
    )
    _parse_with_harbor(log_mcp / "reward.json")
    assert rew_mcp["reward"] == 1.0
    assert rew_mcp["passed"] == 1
    assert met_mcp["measured"]["j1331_source_ms"] == "calibrators.ms"
    assert met_mcp["measured"]["pcal_source_ms"] == "calibrators.ms"

    # 2. Outlier 170 Jy unflagged RFI integration directly inside calibrated J1331 (no calibrators.ms)
    ws_rfi = tmp_path / "ws_j1331_outlier"
    ws_rfi.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_rfi / "input.ms")
    tb = casatools.table()
    tb.open(str(ws_rfi / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")
        corr = tb.getcol("CORRECTED_DATA")
        flg = tb.getcol("FLAG")
        flgr = tb.getcol("FLAG_ROW")
        j_rows = np.flatnonzero((fids == 0) & (a1 != a2) & (a1 != 6) & (a2 != 6))
        # Unflag 1 integration (28 baselines) and inject 170 Jy phase-corrupted RFI spike
        burst_rows = j_rows[:28]
        flgr[burst_rows] = False
        flg[:, :, burst_rows] = False
        freqs = np.linspace(0.0, 1.0, corr.shape[1])[:, np.newaxis]
        corr[0, :, :][:, burst_rows] = 170.0 * np.exp(1j * 4.5 * freqs)
        corr[3, :, :][:, burst_rows] = 170.0 * np.exp(-1j * 3.8 * freqs)
        tb.putcol("CORRECTED_DATA", corr)
        tb.putcol("FLAG", flg)
        tb.putcol("FLAG_ROW", flgr)
    finally:
        tb.close()
        tb.done()

    rew_rfi, met_rfi = grade_calibrate(
        ws_rfi,
        _load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json"),
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert met_rfi["measured"]["j1331_med_baseline_slope_deg_per_100mhz"] < 5.0
    assert rew_rfi["bandpass_flatness_score"] == 1.0
    assert rew_rfi["reward"] == 1.0
    assert rew_rfi["passed"] == 1

    # 3. Verify s1_calibrate/solve.py -> verify.py works cleanly when calibrators.ms is staged (SF-5)
    ws_sol = tmp_path / "ws_s1_cal_solve"
    ws_sol.mkdir()
    shutil.copytree(synth_golden_precal_dir / "input.ms", ws_sol / "input.ms")
    shutil.copytree(synth_golden_precal_dir / "calibrators.ms", ws_sol / "calibrators.ms")
    log_sol = tmp_path / "log_s1_cal_solve"
    _run_py_script(specs_dir / "s1_calibrate" / "solve.py", ["--workspace", str(ws_sol)])
    _run_py_script(
        specs_dir / "s1_calibrate" / "verify.py",
        [
            "--workspace",
            str(ws_sol),
            "--tests-dir",
            str(SYNTH_ROOT / "s1_calibrate"),
            "--log-dir",
            str(log_sol),
        ],
    )
    parsed_sol = _parse_with_harbor(log_sol / "reward.json")
    assert parsed_sol["reward"] == 1.0
    assert parsed_sol["passed"] == 1


def test_wave4_fromarray_fake_image_bypasses_blocked(
    synth_golden_image_dir: Path, tmp_path: Path
) -> None:
    specs_dir = Path(__file__).resolve().parents[1] / "task_specs"

    # 0. Direct s1_image/solve.py -> s1_image/verify.py golden execution still scores 1.0
    ws_s1_img = tmp_path / "ws_s1_img_solve"
    ws_s1_img.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_s1_img / "input.ms")
    log_s1_img = tmp_path / "log_s1_img_solve"
    _run_py_script(specs_dir / "s1_image" / "solve.py", ["--workspace", str(ws_s1_img)])
    _run_py_script(
        specs_dir / "s1_image" / "verify.py",
        [
            "--workspace",
            str(ws_s1_img),
            "--tests-dir",
            str(SYNTH_ROOT / "s1_image"),
            "--log-dir",
            str(log_s1_img),
        ],
    )
    parsed_s1_img = _parse_with_harbor(log_s1_img / "reward.json")
    assert parsed_s1_img["reward"] == 1.0
    assert parsed_s1_img["passed"] == 1

    # 1. Zero-tclean ia.fromarray fake image with .pb = 1.0, .residual = 1e-6, spike .image/.model
    ia = casatools.image()
    ref_ia = casatools.image()
    ref_ia.open(str(synth_golden_image_dir / "synth_mosaic.image"))
    cs_rec = ref_ia.coordsys().torecord()
    rb_rec = ref_ia.restoringbeam()
    shp = list(ref_ia.shape())
    ref_ia.close()
    ref_ia.done()

    ws_fake_s = tmp_path / "ws_fake_fromarray_s"
    ws_fake_s.mkdir()
    cx, cy = shp[0] // 2, shp[1] // 2
    img_pix = np.zeros(shp, dtype=np.float32)
    img_pix[cx, cy, 0, 0] = 0.075
    pb_pix = np.ones(shp, dtype=np.float32)
    res_pix = np.full(shp, 1.0e-6, dtype=np.float32)

    for sfx, arr in (
        ("image", img_pix),
        ("image.pbcor", img_pix),
        ("model", img_pix),
        ("pb", pb_pix),
        ("residual", res_pix),
    ):
        out_p = ws_fake_s / f"fake_mosaic.{sfx}"
        ia.fromarray(outfile=str(out_p), pixels=arr, csys=cs_rec, overwrite=True)
        if sfx in ("image", "image.pbcor", "residual"):
            ia.setrestoringbeam(beam=rb_rec)
        ia.close()
        ia.done()

    rew_fs, _ = grade_image(
        ws_fake_s, _load_gt(SYNTH_ROOT / "s1_image" / "ground_truth.json"), task_id="s1_image"
    )
    assert rew_fs["products_score"] == 0.0
    assert rew_fs["residual_rms_score"] == 0.0
    assert rew_fs["deconv_and_bowl_score"] == 0.0
    assert rew_fs["reward"] < 0.25
    assert rew_fs["passed"] == 0

    rew_fr, _ = grade_image(
        ws_fake_s,
        _load_gt(R3C391_ROOT / "r1_3c391_image" / "ground_truth.json"),
        task_id="r1_3c391_image",
    )
    assert rew_fr["products_score"] == 0.0
    assert rew_fr["residual_rms_score"] == 0.0
    assert rew_fr["deconv_and_bowl_score"] == 0.0
    assert rew_fr["reward"] < 0.25
    assert rew_fr["passed"] == 0

    # 2. Corrupt .image.pbcor so pbcor * pb != image -> products_score == 0, deconv_and_bowl == 0
    ws_bad_pbcor = tmp_path / "ws_bad_pbcor"
    ws_bad_pbcor.mkdir()
    for suffix in ("image", "image.pbcor", "pb", "residual", "model"):
        shutil.copytree(
            synth_golden_image_dir / f"synth_mosaic.{suffix}",
            ws_bad_pbcor / f"synth_mosaic.{suffix}",
        )
    ia = casatools.image()
    ia.open(str(ws_bad_pbcor / "synth_mosaic.image.pbcor"))
    try:
        ia.putchunk(ia.getchunk() * 0.2)
    finally:
        ia.close()
        ia.done()

    rew_pbc, _ = grade_image(
        ws_bad_pbcor, _load_gt(SYNTH_ROOT / "s1_image" / "ground_truth.json"), task_id="s1_image"
    )
    assert rew_pbc["products_score"] == 0.0
    assert rew_pbc["deconv_and_bowl_score"] == 0.0
    assert rew_pbc["passed"] == 0

    # 3. Sparse-spike residual (zeros + 1 spike to match target RMS -> MAD == 0) -> blocked!
    ws_spike_res = tmp_path / "ws_spike_res"
    ws_spike_res.mkdir()
    for suffix in ("image", "image.pbcor", "pb", "residual", "model"):
        shutil.copytree(
            synth_golden_image_dir / f"synth_mosaic.{suffix}",
            ws_spike_res / f"synth_mosaic.{suffix}",
        )
    ia = casatools.image()
    ia.open(str(ws_spike_res / "synth_mosaic.residual"))
    try:
        r_chunk = np.zeros_like(ia.getchunk())
        r_chunk[cx + 60, cy + 60, 0, 0] = 0.05
        ia.putchunk(r_chunk)
    finally:
        ia.close()
        ia.done()

    rew_spk, _ = grade_image(
        ws_spike_res, _load_gt(SYNTH_ROOT / "s1_image" / "ground_truth.json"), task_id="s1_image"
    )
    assert rew_spk["residual_rms_score"] == 0.0
    assert rew_spk["deconv_and_bowl_score"] == 0.0
    assert rew_spk["passed"] == 0


def test_wave4_constant_corrected_data_and_subprocess_tclean_isolation(tmp_path: Path) -> None:
    gt_s1 = _load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json")

    # 1. Constant fake CORRECTED_DATA = 0.48 + 0j on target (previously gave amp_ratio ~ 0.999)
    ws_const = tmp_path / "ws_const_048"
    ws_const.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_const / "input.ms")
    tb = casatools.table()
    tb.open(str(ws_const / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        corr = tb.getcol("CORRECTED_DATA")
        tgt = np.isin(fids, [2, 3])
        corr[:, :, tgt] = 0.48 + 0.0j
        tb.putcol("CORRECTED_DATA", corr)
    finally:
        tb.close()
        tb.done()

    rew_const, met_const = grade_calibrate(
        ws_const,
        gt_s1,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert met_const["antenna_gain_consistent"] is False
    assert met_const["measured"]["coherent_pearson_r"] == 0.0
    assert rew_const["target_fidelity_score"] == 0.0
    assert rew_const["retention_and_dead_ant_score"] == 0.0
    assert rew_const["passed"] == 0

    # 2. Constant 100 Jy + 10 mJy Gaussian noise hack -> fails rank-1 gain factorization & Pearson r
    rng = np.random.default_rng(12345)
    tb = casatools.table()
    tb.open(str(ws_const / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        corr = tb.getcol("CORRECTED_DATA")
        tgt = np.isin(fids, [2, 3])
        sub_shape = corr[:, :, tgt].shape
        noise = 0.01 * (rng.standard_normal(sub_shape) + 1j * rng.standard_normal(sub_shape))
        corr[:, :, tgt] = (100.0 + noise).astype(corr.dtype)
        tb.putcol("CORRECTED_DATA", corr)
    finally:
        tb.close()
        tb.done()

    rew_noisy, met_noisy = grade_calibrate(
        ws_const,
        gt_s1,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert met_noisy["antenna_gain_consistent"] is False
    assert met_noisy["measured"]["rank1_logamp_rms_median"] > 0.12
    assert rew_noisy["target_fidelity_score"] == 0.0
    assert rew_noisy["passed"] == 0

    # 3. Subprocess-isolated verifier tclean on noiseless constant CORRECTED_DATA = 100.0 + 0j
    #    with usemask='auto-multithresh' (triggers CASA C++ SIGABRT / zero-variance failure):
    #    verify no parent crash, clean reward.json & metrics.json, passed == 0!
    tb = casatools.table()
    tb.open(str(ws_const / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        corr = tb.getcol("CORRECTED_DATA")
        corr[:, :, np.isin(fids, [2, 3])] = 100.0 + 0.0j
        tb.putcol("CORRECTED_DATA", corr)
    finally:
        tb.close()
        tb.done()

    gt_tclean_crash = dict(gt_s1)
    gt_tclean_crash["suite"] = "R"
    gt_tclean_crash["verifier_tclean_recipe"] = {
        "field": "2,3",
        "spw": "0:4~59",
        "phasecenter": "J2000 18h24m00.000s -09d35m00.000s",
        "specmode": "mfs",
        "gridder": "mosaic",
        "deconvolver": "multiscale",
        "scales": [0, 3, 8],
        "imsize": [192, 192],
        "cell_arcsec": 4.0,
        "weighting": "briggs",
        "robust": 0.5,
        "pblimit": 0.2,
        "usemask": "auto-multithresh",
        "pbmask": 0.2,
        "sidelobethreshold": 1.25,
        "noisethreshold": 2.75,
        "lownoisethreshold": 1.35,
        "negativethreshold": 0.0,
        "minbeamfrac": 0.1,
        "growiterations": 10,
        "dogrowprune": False,
        "niter": 100,
        "nmajor": 1,
        "threshold": "0.3mJy",
        "pbcor": True,
        "off_source_min_radius_arcsec": 80.0,
    }
    tests_crash = tmp_path / "tests_crash"
    tests_crash.mkdir()
    (tests_crash / "ground_truth.json").write_text(json.dumps(gt_tclean_crash), encoding="utf-8")
    log_crash = tmp_path / "log_crash"

    rew_crash, met_crash = grade_task(
        "r1_3c391_calibrate",
        "calibrate",
        ws_const,
        tests_crash,
        log_crash,
        scratch_root=tmp_path,
    )
    parsed_crash = _parse_with_harbor(log_crash / "reward.json")
    assert rew_crash["target_fidelity_score"] == 0.0
    assert rew_crash["passed"] == 0
    assert parsed_crash["passed"] == 0
    assert (log_crash / "metrics.json").is_file()
    _ = met_crash


def test_wave4_parallel_hand_and_calibrator_overflagging_blocked(tmp_path: Path) -> None:
    gt_s1 = _load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json")

    # 1. Flag 99.99% of RR and LL on target while leaving cross-hand RL unflagged
    ws_par = tmp_path / "ws_par_overflag"
    ws_par.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_par / "input.ms")
    tb = casatools.table()
    tb.open(str(ws_par / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        flg = tb.getcol("FLAG")
        flgr = tb.getcol("FLAG_ROW")
        tgt_idx = np.flatnonzero(np.isin(fids, [2, 3]))
        # Leave RL (pol 1) unflagged everywhere, flag RR/LL on all but 1 target row
        flgr[tgt_idx] = False
        flg[1, :, :][:, tgt_idx] = False
        flg[0, :, :][:, tgt_idx[1:]] = True
        flg[3, :, :][:, tgt_idx[1:]] = True
        tb.putcol("FLAG", flg)
        tb.putcol("FLAG_ROW", flgr)
    finally:
        tb.close()
        tb.done()

    rew_par, met_par = grade_calibrate(
        ws_par,
        gt_s1,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert met_par["measured"]["target_row_retention"] < 0.01
    assert rew_par["retention_and_dead_ant_score"] == 0.0
    assert rew_par["target_fidelity_score"] == 0.0
    assert rew_par["passed"] == 0

    # 2. Over-flag > 50% of J1331+3030 and J1822-0938 parallel-hand visibilities -> 0.0 subscores
    ws_cal_flg = tmp_path / "ws_cal_overflag"
    ws_cal_flg.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_cal_flg / "input.ms")
    tb = casatools.table()
    tb.open(str(ws_cal_flg / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        flg = tb.getcol("FLAG")
        cal_rows = np.flatnonzero(np.isin(fids, [0, 1]))
        # Flag 40 of 64 channels on all calibrator rows -> unflagged fraction < 0.40 (< 0.50)
        flg[0, :40, :][:, cal_rows] = True
        flg[3, :40, :][:, cal_rows] = True
        tb.putcol("FLAG", flg)
    finally:
        tb.close()
        tb.done()

    rew_cf, _ = grade_calibrate(
        ws_cal_flg,
        gt_s1,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert rew_cf["bandpass_flatness_score"] == 0.0
    assert rew_cf["fluxscale_score"] == 0.0
    assert rew_cf["passed"] == 0


def test_extract_cal_truth_suite_s_and_suite_r_ground_truth() -> None:
    """Regression check for MF-1, MF-2, SF-4, SF-5 on Suite S and Suite R ground_truth.json."""
    from radio_bench.grading.cal_metrics import _extract_cal_truth

    for synth_tid in ("s1_calibrate", "s2_e2e_synth"):
        gt_s = _load_gt(SYNTH_ROOT / synth_tid / "ground_truth.json")
        ct_s = _extract_cal_truth(gt_s)
        assert ct_s["is_suite_r"] is False
        assert ct_s["expected_pcal_flux_jy"] == pytest.approx(2.150366)
        assert ct_s["pcal_flux_rel_tol"] == pytest.approx(0.05)
        assert ct_s["excludes_dead_in_retention"] is True
        assert ct_s["weights"] == {
            "retention_and_dead_ant": pytest.approx(0.20),
            "bandpass_flatness": pytest.approx(0.25),
            "fluxscale_accuracy": pytest.approx(0.25),
            "target_fidelity": pytest.approx(0.30),
        }

    for r_tid in ("r1_3c391_calibrate", "r2_3c391_e2e"):
        gt_r = _load_gt(R3C391_ROOT / r_tid / "ground_truth.json")
        ct_r = _extract_cal_truth(gt_r)
        assert ct_r["is_suite_r"] is True
        assert ct_r["excludes_dead_in_retention"] is False
        assert ct_r["expected_pcal_flux_jy"] == pytest.approx(2.3)
        assert ct_r["pcal_flux_rel_tol"] == pytest.approx(0.08)
        assert ct_r["weights"] == {
            "retention_and_dead_ant": pytest.approx(0.20),
            "bandpass_flatness": pytest.approx(0.25),
            "fluxscale_accuracy": pytest.approx(0.25),
            "target_fidelity": pytest.approx(0.30),
        }

    ct_fallback_s = _extract_cal_truth({"suite": "synth"})
    assert ct_fallback_s["expected_pcal_flux_jy"] == pytest.approx(2.150366)
    assert ct_fallback_s["excludes_dead_in_retention"] is True

    ct_fallback_r = _extract_cal_truth({"suite": "R"})
    assert ct_fallback_r["excludes_dead_in_retention"] is False


def test_wave6_bandpass_flatness_sparse_rfi_channel_and_branch_cut(tmp_path: Path) -> None:
    """Regression for Wave 6 E2E bandpass flatness:
    1. Sparse surviving narrowband RFI cells (2-4 unflagged 110 Jy cells in 99%-flagged ch 29 & 31
       inside otherwise clean 7.6 Jy rows, previously triggering a +360 deg np.unwrap jump ->
       177.17 deg RMS and 420.62 deg/100MHz slope) plus a +-180 deg branch-cut global phase offset
       on J1331+3030 still score bandpass_flatness_score == 1.0.
    2. Scaled uncalibrated DATA (preserving true 14 ns delays with CORRECTED_DATA != DATA) still
       scores bandpass_flatness_score == 0.0.
    """
    gt_s2 = _load_gt(SYNTH_ROOT / "s2_e2e_synth" / "ground_truth.json")

    # 1. Calibrated MS with sparse surviving RFI cells in ch 29 & 31 + pi branch-cut rotation
    ws_sparse = tmp_path / "ws_sparse_rfi"
    ws_sparse.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_sparse / "input.ms")

    rng = np.random.default_rng(20261114)
    tb = casatools.table()
    tb.open(str(ws_sparse / "input.ms"), nomodify=False)
    try:
        fids = tb.getcol("FIELD_ID")
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")
        corr = tb.getcol("CORRECTED_DATA")
        flg = tb.getcol("FLAG")
        flgr = tb.getcol("FLAG_ROW")

        j_live = np.flatnonzero((fids == 0) & (a1 != a2) & (a1 != 6) & (a2 != 6) & (~flgr))
        # Rotate J1331+3030 calibrated visibilities onto +-180 deg branch cut with small 1 deg jitter
        jitter_rad = np.deg2rad(rng.normal(0.0, 1.0, size=corr[:, :, j_live].shape))
        corr[:, :, j_live] = corr[:, :, j_live] * np.exp(1j * (math.pi + jitter_rad))

        # Unflag 2 rows in ch 29 (+158 deg) and 4 rows in ch 31 (-108 deg) inside clean rows
        r29 = j_live[10:12]
        r31 = j_live[12:16]
        flg[0, 29, r29] = False
        flg[3, 29, r29] = False
        corr[0, 29, r29] = 113.0 * np.exp(1j * math.radians(158.06 - 180.0))
        corr[3, 29, r29] = 113.0 * np.exp(1j * math.radians(158.06 - 180.0))

        flg[0, 31, r31] = False
        flg[3, 31, r31] = False
        corr[0, 31, r31] = 95.0 * np.exp(1j * math.radians(-107.77 - 180.0))
        corr[3, 31, r31] = 95.0 * np.exp(1j * math.radians(-107.77 - 180.0))

        # Also leave 2 low-amplitude unflagged stray cells in ch 30 (<10% channel support)
        r30 = j_live[16:18]
        flg[0, 30, r30] = False
        flg[3, 30, r30] = False
        corr[0, 30, r30] = 7.6 * np.exp(1j * math.radians(145.0))
        corr[3, 30, r30] = 7.6 * np.exp(1j * math.radians(-135.0))

        tb.putcol("CORRECTED_DATA", corr)
        tb.putcol("FLAG", flg)
    finally:
        tb.close()
        tb.done()

    rew_sp, met_sp = grade_calibrate(
        ws_sparse,
        gt_s2,
        masks_npz_path=SYNTH_ROOT / "s2_e2e_synth" / "true_target_vis.npz",
        task_id="s2_e2e_synth",
        scratch_root=tmp_path,
    )
    assert met_sp["measured"]["j1331_phase_slope_deg_per_100mhz"] < 5.0
    assert met_sp["measured"]["j1331_chan_phase_rms_deg"] < 8.0
    assert rew_sp["bandpass_flatness_score"] == 1.0
    assert rew_sp["reward"] == 1.0

    # 2. Anti-hacking: scaled uncalibrated DATA (with true 14 ns delays intact) still gets 0.0
    ws_uncal = tmp_path / "ws_uncal_scaled"
    ws_uncal.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_calibrate" / "input.ms", ws_uncal / "input.ms")
    casatasks.clearcal(vis=str(ws_uncal / "input.ms"), addmodel=False)
    tb = casatools.table()
    tb.open(str(ws_uncal / "input.ms"), nomodify=False)
    try:
        raw = tb.getcol("DATA")
        flg = tb.getcol("FLAG")
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")
        flg[:, :, (a1 == 6) | (a2 == 6)] = True
        tb.putcol("FLAG", flg)
        tb.putcol("CORRECTED_DATA", (5.93 * raw).astype(raw.dtype))
    finally:
        tb.close()
        tb.done()

    rew_unc, met_unc = grade_calibrate(
        ws_uncal,
        gt_s2,
        masks_npz_path=SYNTH_ROOT / "s2_e2e_synth" / "true_target_vis.npz",
        task_id="s2_e2e_synth",
        scratch_root=tmp_path,
    )
    assert met_unc["measured"]["j1331_med_baseline_slope_deg_per_100mhz"] > 50.0
    assert rew_unc["bandpass_flatness_score"] == 0.0
    assert rew_unc["passed"] == 0
