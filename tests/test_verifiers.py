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
import sys
import types
from pathlib import Path
from typing import Any

from radio_bench.config import is_under_slash_tmp, load_bench_config, write_casaconfig_local

BENCH_CFG = load_bench_config()

_cur_tmp = os.environ.get("TMPDIR", "").strip()
if not _cur_tmp or is_under_slash_tmp(_cur_tmp):
    os.environ["TMPDIR"] = str(BENCH_CFG.tmp_dir)
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

_cur_mpl = os.environ.get("MPLCONFIGDIR", "").strip()
if not _cur_mpl or is_under_slash_tmp(_cur_mpl):
    os.environ["MPLCONFIGDIR"] = str(BENCH_CFG.mplconfig_dir)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

write_casaconfig_local(BENCH_CFG)
os.environ.setdefault("CASASITECONFIG", str(BENCH_CFG.casaconfig_path))

import harbor.verifier.verifier as hv  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

if sys.platform == "darwin":
    casatasks = pytest.importorskip("casatasks")
    casatools = pytest.importorskip("casatools")
else:
    import casatasks  # noqa: E402
    import casatools  # noqa: E402

from radio_bench.grading import (  # noqa: E402
    grade_calibrate,
    grade_e2e,
    grade_image,
    grade_inspect,
    grade_precal,
    grade_task,
    write_verifier_outputs,
)

SYNTH_ROOT = BENCH_CFG.data_dir / "synth"
R3C391_ROOT = BENCH_CFG.data_dir / "checkpoints" / "3c391"


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
    assert (
        met_s1["measurements"]["precal.flux_cal_model_amp_jy"]["provenance"]
        == "calibrators.ms::MODEL_DATA"
    )
    assert met_s1["measured"]["precal.must_flag_recall"] == 1.0
    assert met_s1["measured"]["precal.clean_false_positive_rate"] == 0.0

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
    assert met_r1["measured"]["precal.must_flag_recall"] == 1.0
    assert met_r1["measured"]["precal.clean_false_positive_rate"] == 0.0


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
        if sys.platform == "darwin":
            # Darwin struct flock: (l_start, l_len, l_pid, l_type, l_whence)
            lock_spec = struct.pack("qqihh", 0, 0, 0, fcntl.F_WRLCK, os.SEEK_SET)
        else:
            # Linux x86_64 struct flock: (l_type, l_whence, l_start, l_len, l_pid)
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
    assert met_ok["measured"]["cal.coherent_vis_amp_rel_err"] < 0.01
    assert abs(met_ok["measured"]["cal.coherent_vis_phase_err_deg"]) < 1.0

    # Fallback analytical/simulator prediction when masks_npz_path=None
    gt_cal = _load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json")
    rew_fb, met_fb = grade_calibrate(
        ws_ok, gt_cal, masks_npz_path=None, task_id="s1_calibrate", scratch_root=tmp_path
    )
    assert rew_fb["reward"] == 1.0
    assert met_fb["measured"]["cal.coherent_vis_amp_rel_err"] == pytest.approx(
        met_ok["measured"]["cal.coherent_vis_amp_rel_err"], abs=1e-4
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
            "verifier_tclean_offsource_mad_over_rms": 1.04,
            "verifier_tclean_dynamic_range": 308.35,
            "verifier_tclean_pb_valid": True,
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
        assert met_r1["measured"]["cal.dead_antenna_flagged_fraction"] == 1.0
        assert 0.70 <= met_r1["measured"]["cal.target_row_retention"] <= 0.90
        assert met_r1["measured"]["cal.fluxscale_table_flux_jy"] == pytest.approx(
            2.296125, rel=1e-4
        )
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
    assert abs(met_s1["measured"]["image.source_integrated_flux_jy"] - 0.075) / 0.075 < 0.10
    assert met_s1["measured"]["image.point_source_snr"] >= 8.0
    assert met_s1["measured"]["image.negative_bowl_ratio"] < 0.06

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
    assert met_r1["measured"]["image.offsource_residual_rms_jy"] * 1e3 == pytest.approx(
        0.43617, abs=0.01
    )
    assert met_r1["measured"]["image.dynamic_range"] >= 300.0


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
    assert met_bowl["measured"]["image.negative_bowl_ratio"] == pytest.approx(0.35, rel=0.02)
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
    assert met_cell["measured"]["image.pixels_per_beam"] < 1.0
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
            "verifier_tclean_offsource_mad_over_rms": 1.04,
            "verifier_tclean_dynamic_range": 308.35,
            "verifier_tclean_pb_valid": True,
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


def _run_py_module(mod_name: str, args: list[str]) -> None:
    import subprocess
    import sys

    env = os.environ.copy()
    BENCH_CFG.tmp_dir.mkdir(parents=True, exist_ok=True)
    BENCH_CFG.mplconfig_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(BENCH_CFG.tmp_dir)
    env["MPLCONFIGDIR"] = str(BENCH_CFG.mplconfig_dir)
    env.setdefault("CASASITECONFIG", str(BENCH_CFG.casaconfig_path))
    src_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{src_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_root
    proc = subprocess.run(
        [sys.executable, "-m", mod_name, *args],
        cwd=str(BENCH_CFG.tmp_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300.0,
        check=False,
    )
    assert proc.returncode == 0, (
        f"python -m {mod_name} exited {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def _stage_runner_tests_dir(task_id: str, stage: str, src_data_dir: Path, dst_dir: Path) -> Path:
    from radio_bench.grading.inspect_metrics import resolve_task_configs

    dst_dir.mkdir(parents=True, exist_ok=True)
    gt_src = src_data_dir / "ground_truth.json"
    gt = _load_gt(gt_src)
    ec, rb = resolve_task_configs(task_id, stage, gt)
    (dst_dir / "extract_config.json").write_text(
        json.dumps(ec.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    (dst_dir / "rubric.json").write_text(
        json.dumps(rb.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    os.symlink(gt_src, dst_dir / "ground_truth.json")
    for npz_p in src_data_dir.glob("*.npz"):
        os.symlink(npz_p, dst_dir / npz_p.name)
    return dst_dir


def _run_py_script(script_path: Path, args: list[str]) -> None:
    import subprocess
    import sys

    env = os.environ.copy()
    BENCH_CFG.tmp_dir.mkdir(parents=True, exist_ok=True)
    BENCH_CFG.mplconfig_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(BENCH_CFG.tmp_dir)
    env["MPLCONFIGDIR"] = str(BENCH_CFG.mplconfig_dir)
    env.setdefault("CASASITECONFIG", str(BENCH_CFG.casaconfig_path))
    src_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{src_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_root
    proc = subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(BENCH_CFG.tmp_dir),
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

    # 1. Direct solve.py -> grading.runner execution on s1_inspect and r1_3c391_inspect
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
        tests_dir = _stage_runner_tests_dir(
            task_id, "inspect", data_dir, tmp_path / f"td_{task_id}"
        )
        _run_py_script(specs_dir / task_id / "solve.py", ["--workspace", str(ws)])
        _run_py_module(
            "radio_bench.grading.runner",
            ["--workspace", str(ws), "--tests-dir", str(tests_dir), "--log-dir", str(log_dir)],
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
    assert (
        met_mcp["measurements"]["cal.bandpass_coh_phase_slope_deg_per_100mhz"]["provenance"]
        == "calibrators.ms::CORRECTED_DATA"
    )
    assert (
        met_mcp["measurements"]["cal.phase_cal_corrected_amp_jy"]["provenance"]
        == "calibrators.ms::CORRECTED_DATA"
    )

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
    assert met_rfi["measured"]["cal.bandpass_med_bl_phase_slope_deg_per_100mhz"] < 5.0
    assert rew_rfi["bandpass_flatness_score"] == 1.0
    assert rew_rfi["reward"] == 1.0
    assert rew_rfi["passed"] == 1

    # 3. Verify s1_calibrate/solve.py -> grading.runner works cleanly when calibrators.ms is staged
    ws_sol = tmp_path / "ws_s1_cal_solve"
    ws_sol.mkdir()
    shutil.copytree(synth_golden_precal_dir / "input.ms", ws_sol / "input.ms")
    shutil.copytree(synth_golden_precal_dir / "calibrators.ms", ws_sol / "calibrators.ms")
    log_sol = tmp_path / "log_s1_cal_solve"
    td_sol = _stage_runner_tests_dir(
        "s1_calibrate", "calibrate", SYNTH_ROOT / "s1_calibrate", tmp_path / "td_s1_cal_solve"
    )
    _run_py_script(specs_dir / "s1_calibrate" / "solve.py", ["--workspace", str(ws_sol)])
    _run_py_module(
        "radio_bench.grading.runner",
        [
            "--workspace",
            str(ws_sol),
            "--tests-dir",
            str(td_sol),
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

    # 0. Direct s1_image/solve.py -> grading.runner golden execution still scores 1.0
    ws_s1_img = tmp_path / "ws_s1_img_solve"
    ws_s1_img.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_s1_img / "input.ms")
    log_s1_img = tmp_path / "log_s1_img_solve"
    td_s1_img = _stage_runner_tests_dir(
        "s1_image", "image", SYNTH_ROOT / "s1_image", tmp_path / "td_s1_img_solve"
    )
    _run_py_script(specs_dir / "s1_image" / "solve.py", ["--workspace", str(ws_s1_img)])
    _run_py_module(
        "radio_bench.grading.runner",
        [
            "--workspace",
            str(ws_s1_img),
            "--tests-dir",
            str(td_s1_img),
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
    assert met_const["measured"]["cal.target_corrected_std_jy"] == 0.0
    assert met_const["measured"]["cal.coherent_vis_pearson_r"] == 0.0
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
    assert met_noisy["measured"]["cal.rank1_logamp_rms_median"] > 0.12
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

    from radio_bench.grading.inspect_metrics import resolve_task_configs

    gt_tclean_crash = dict(gt_s1)
    gt_tclean_crash["j1822_expected_flux_jy"] = 2.150366
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
    ec_crash, _ = resolve_task_configs("s1_calibrate", "calibrate", gt_tclean_crash)
    _, rb_crash = resolve_task_configs("r1_3c391_calibrate", "calibrate", gt_tclean_crash)
    tests_crash = tmp_path / "tests_crash"
    tests_crash.mkdir()
    (tests_crash / "extract_config.json").write_text(
        json.dumps(ec_crash.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    (tests_crash / "rubric.json").write_text(
        json.dumps(rb_crash.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
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
    assert met_par["measured"]["cal.target_row_retention"] < 0.01
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


def test_suite_manifests_and_extract_configs_zero_truth_leakage() -> None:
    """Verify `suites/*/suite.toml` + `rubrics/*.json` schemas and zero truth leakage in `ExtractConfig`."""
    from radio_bench.grading.inspect_metrics import load_suite_task_registry

    reg = load_suite_task_registry()
    assert set(reg.keys()) == {
        "s1_inspect",
        "s1_precal",
        "s1_calibrate",
        "s1_image",
        "s2_e2e_synth",
        "r1_3c391_inspect",
        "r1_3c391_precal",
        "r1_3c391_calibrate",
        "r1_3c391_image",
        "r2_3c391_e2e",
    }
    forbidden_scalar_substrings = (
        "expected_flux",
        "fluxscale_jy",
        "dynamic_range",
        "offsource_rms",
        "pass_reward_min",
        "min_elevation",
        "raw_data_blake2b",
    )
    for tid, (ec, rb) in reg.items():
        assert ec.suite_id in {"synth_vla_d", "vla_3c391_cband"}
        assert rb.task_id == tid
        dumped = json.dumps(ec.model_dump(mode="json")).lower()
        for bad in forbidden_scalar_substrings:
            assert bad not in dumped, f"ExtractConfig for {tid} leaked truth key {bad!r}"


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
    assert met_sp["measured"]["cal.bandpass_coh_phase_slope_deg_per_100mhz"] < 5.0
    assert met_sp["measured"]["cal.bandpass_coh_chan_phase_rms_deg"] < 8.0
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
    assert met_unc["measured"]["cal.bandpass_med_bl_phase_slope_deg_per_100mhz"] > 50.0
    assert rew_unc["bandpass_flatness_score"] == 0.0
    assert rew_unc["passed"] == 0


# ---------------------------------------------------------------------------
# 8. Wave 2 Two-Sided Calibration Matrix (Planted Faults + Alternative Recipes)
# ---------------------------------------------------------------------------


def test_wave2_precal_unflagged_rfi_and_virtual_source_model(
    synth_golden_precal_dir: Path, tmp_path: Path
) -> None:
    """Two-sided `precal` calibration:
    - Planted fault: unflagged RFI (`must_flag_recall == 0.0`) -> `flag_recall_score == 0.0`, `passed == 0`.
    - Alternative recipe: `setjy(usescratch=False)` virtual `SOURCE_MODEL` without `MODEL_DATA`
      scores `0.0` on `model_data_score` when `allow_virtual_model == False` (`v0.1` parity) and
      `1.0` (`reward == 1.0, passed == 1`) when `allow_virtual_model == True`.
    """
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.runner import run_verification

    gt_s1 = _load_gt(SYNTH_ROOT / "s1_precal" / "ground_truth.json")
    npz_s1 = SYNTH_ROOT / "s1_precal" / "corruption_masks.npz"

    # 1. Planted fault: clear all flags on input.ms and split unflagged calibrators.ms -> must_flag_recall == 0.0
    ws_unflag = tmp_path / "ws_precal_unflag"
    ws_unflag.mkdir()
    shutil.copytree(synth_golden_precal_dir / "input.ms", ws_unflag / "input.ms")
    tb = casatools.table()
    tb.open(str(ws_unflag / "input.ms"), nomodify=False)
    try:
        flg = tb.getcol("FLAG")
        flgr = tb.getcol("FLAG_ROW")
        flg[:] = False
        flgr[:] = False
        tb.putcol("FLAG", flg)
        tb.putcol("FLAG_ROW", flgr)
    finally:
        tb.close()
        tb.done()
    casatasks.split(
        vis=str(ws_unflag / "input.ms"),
        outputvis=str(ws_unflag / "calibrators.ms"),
        field="J1331+3030,J1822-0938",
        datacolumn="data",
        keepflags=True,
    )
    casatasks.setjy(
        vis=str(ws_unflag / "calibrators.ms"),
        field="J1331+3030",
        standard="Perley-Butler 2017",
        usescratch=True,
    )

    rew_unf, met_unf = grade_precal(ws_unflag, gt_s1, masks_npz_path=npz_s1, task_id="s1_precal")
    assert met_unf["measured"]["precal.must_flag_recall"] == 0.0
    assert rew_unf["flag_recall_score"] == 0.0
    assert rew_unf["passed"] == 0

    # 2. Alternative recipe: virtual SOURCE_MODEL (`setjy(usescratch=False)`) with MODEL_DATA removed
    ws_virt = tmp_path / "ws_precal_virt"
    ws_virt.mkdir()
    shutil.copytree(synth_golden_precal_dir / "input.ms", ws_virt / "input.ms")
    shutil.copytree(synth_golden_precal_dir / "calibrators.ms", ws_virt / "calibrators.ms")
    for ms_p in (ws_virt / "input.ms", ws_virt / "calibrators.ms"):
        tb = casatools.table()
        tb.open(str(ms_p), nomodify=False)
        try:
            if "MODEL_DATA" in tb.colnames():
                tb.removecols(["MODEL_DATA"])
        finally:
            tb.close()
            tb.done()

    casatasks.setjy(
        vis=str(ws_virt / "calibrators.ms"),
        field="J1331+3030",
        standard="Perley-Butler 2017",
        usescratch=False,
    )

    # Under v0.1 parity (`allow_virtual_model == False`), virtual model scores 0 on model_data_score
    ec_off, rb_s1 = resolve_task_configs("s1_precal", "precal", gt_s1)
    assert ec_off.allow_virtual_model is False
    rew_off, _ = run_verification(
        workspace_dir=ws_virt,
        tests_dir=SYNTH_ROOT / "s1_precal",
        log_dir=None,
        extract_config=ec_off,
        rubric=rb_s1,
        ground_truth=gt_s1,
        ref_arrays_dir=SYNTH_ROOT / "s1_precal",
        compute_raw_hash=False,
    )
    assert rew_off["model_data_score"] == 0.0
    assert rew_off["passed"] == 0

    # Under `allow_virtual_model == True`, SOURCE_MODEL is evaluated and scores 1.0
    ec_on = ec_off.model_copy(update={"allow_virtual_model": True}, deep=True)
    rew_on, met_on = run_verification(
        workspace_dir=ws_virt,
        tests_dir=SYNTH_ROOT / "s1_precal",
        log_dir=None,
        extract_config=ec_on,
        rubric=rb_s1,
        ground_truth=gt_s1,
        ref_arrays_dir=SYNTH_ROOT / "s1_precal",
        compute_raw_hash=False,
    )
    assert met_on["measured"]["precal.flux_cal_model_amp_jy"] == pytest.approx(7.61, rel=0.05)
    assert rew_on["model_data_score"] == 1.0
    assert rew_on["reward"] == 1.0
    assert rew_on["passed"] == 1


def test_wave2_raw_data_blake2b_mutation_gate_and_multi_ddid_streaming(tmp_path: Path) -> None:
    """Verify `proc.raw_data_blake2b` hard gate on mutated `DATA` and `SIGABRT`-safe multi-`DDID` streaming."""
    from radio_bench.analysis.taxonomy import classify_failure
    from radio_bench.grading.extractors.calibrate import extract_calibrate
    from radio_bench.grading.extractors.common import compute_raw_data_blake2b
    from radio_bench.grading.extractors.precal import extract_precal
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.runner import run_verification
    from radio_bench.grading.schema import ExtractConfig, GeometrySpec, RolesSpec

    # 1. Raw DATA mutation hard gate -> reward == 0.0, passed == 0, raw_data_or_calflag_mutation
    ws_mut = tmp_path / "ws_raw_mut"
    ws_mut.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_image" / "input.ms", ws_mut / "input.ms")
    orig_digest = compute_raw_data_blake2b(ws_mut / "input.ms")
    assert isinstance(orig_digest, str) and len(orig_digest) == 64

    tb = casatools.table()
    tb.open(str(ws_mut / "input.ms"), nomodify=False)
    try:
        cell = tb.getcell("DATA", 0)
        cell[0, 0] = cell[0, 0] + (5.0 + 0.0j)
        tb.putcell("DATA", 0, cell)
    finally:
        tb.close()
        tb.done()

    gt_mut = dict(_load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json"))
    gt_mut["raw_data_blake2b"] = orig_digest
    ec_cal, rb_cal = resolve_task_configs("s1_calibrate", "calibrate", gt_mut)
    rew_mut, met_mut = run_verification(
        workspace_dir=ws_mut,
        tests_dir=SYNTH_ROOT / "s1_calibrate",
        log_dir=None,
        extract_config=ec_cal,
        rubric=rb_cal,
        ground_truth=gt_mut,
        ref_arrays_dir=SYNTH_ROOT / "s1_calibrate",
        scratch_root=tmp_path,
        compute_raw_hash=True,
    )
    assert met_mut["measured"]["proc.raw_data_unmodified"] is False
    assert rew_mut["reward"] == 0.0
    assert rew_mut["passed"] == 0
    assert (
        classify_failure(
            passed=rew_mut["passed"],
            stage="calibrate",
            reward=float(rew_mut["reward"]),
            reward_dict=rew_mut,
            metrics_dict=met_mut,
        )
        == "raw_data_or_calflag_mutation"
    )

    # 2. Multi-DDID heterogeneous-nchan MS (spw0=32ch, spw1=64ch) -> zero SIGABRT across extractors
    multi_ms = tmp_path / "multi_ddid.ms"
    sm = casatools.simulator()
    me = casatools.measures()
    try:
        sm.open(str(multi_ms))
        cofa = me.observatory("VLA")
        sm.setconfig(
            telescopename="VLA",
            x=[0.0, 120.0, -90.0],
            y=[0.0, 60.0, 140.0],
            z=[0.0, 0.0, 0.0],
            dishdiameter=[25.0, 25.0, 25.0],
            mount=["alt-az", "alt-az", "alt-az"],
            antname=["ea01", "ea02", "ea03"],
            coordsystem="local",
            referencelocation=cofa,
        )
        sm.setspwindow(
            spwname="spw32",
            freq="4.5GHz",
            deltafreq="2MHz",
            freqresolution="2MHz",
            nchannels=32,
            stokes="RR RL LR LL",
        )
        sm.setspwindow(
            spwname="spw64",
            freq="4.7GHz",
            deltafreq="1MHz",
            freqresolution="1MHz",
            nchannels=64,
            stokes="RR RL LR LL",
        )
        sm.setfeed(mode="perfect R L", pol=[""])
        sm.setfield(
            sourcename="J1331+3030",
            sourcedirection=me.direction("J2000", "13h31m08.288s", "+30d30m32.959s"),
        )
        sm.setfield(
            sourcename="J1822-0938",
            sourcedirection=me.direction("J2000", "18h22m28.000s", "-09d38m00.000s"),
        )
        sm.setfield(
            sourcename="TARGET_A",
            sourcedirection=me.direction("J2000", "18h24m00.000s", "-09d35m00.000s"),
        )
        sm.settimes(
            integrationtime="10s",
            usehourangle=True,
            referencetime=me.epoch("UTC", "2026/04/15/10:00:00"),
        )
        for fname in ("J1331+3030", "J1822-0938", "TARGET_A"):
            sm.observe(fname, "spw32", starttime="-120s", stoptime="-100s")
            sm.observe(fname, "spw64", starttime="-90s", stoptime="-70s")
    finally:
        with contextlib.suppress(Exception):
            sm.close()
        with contextlib.suppress(Exception):
            sm.done()
        with contextlib.suppress(Exception):
            me.done()

    # Populate nonzero per-row DATA & CORRECTED_DATA per DDID safely
    casatasks.clearcal(vis=str(multi_ms), addmodel=True)
    tb = casatools.table()
    tb.open(str(multi_ms), nomodify=False)
    try:
        ddids = tb.getcol("DATA_DESC_ID")
        assert set(np.unique(ddids).tolist()) == {0, 1}
        for ddid, scale in ((0, 7.6), (1, 7.4)):
            sub = tb.query(f"DATA_DESC_ID == {ddid}")
            try:
                d = sub.getcol("DATA")
                d[:] = scale + 0.0j
                sub.putcol("DATA", d)
                sub.putcol("MODEL_DATA", d)
                sub.putcol("CORRECTED_DATA", (1.02 * d).astype(d.dtype))
                sub.putcol("FLAG", np.zeros_like(sub.getcol("FLAG"), dtype=bool))
                sub.putcol("FLAG_ROW", np.zeros_like(sub.getcol("FLAG_ROW"), dtype=bool))
            finally:
                sub.close()
                sub.done()
    finally:
        tb.close()
        tb.done()

    ws_mddid = tmp_path / "ws_mddid"
    ws_mddid.mkdir()
    os.symlink(multi_ms, ws_mddid / "input.ms")
    os.symlink(multi_ms, ws_mddid / "calibrators.ms")

    multi_digest = compute_raw_data_blake2b(ws_mddid / "input.ms")
    assert isinstance(multi_digest, str) and len(multi_digest) == 64

    ec_multi = ExtractConfig(
        suite_id="test_suite",
        split="dev",
        telescope="VLA",
        pol_basis="circular",
        stages=["precal", "calibrate"],
        roles=RolesSpec(
            flux_cal=["J1331+3030"],
            bandpass_cal=["J1331+3030"],
            phase_cal=["J1822-0938"],
            targets=["TARGET_A"],
            dead_antennas=[],
        ),
        geometry=GeometrySpec(dish_diameter_m=25.0),
    )
    obs_pre, errs_pre = extract_precal(ws_mddid, ec_multi, ref_arrays_dir=None)
    assert not errs_pre
    assert obs_pre["precal.flux_cal_model_amp_jy"].status == "measured"
    assert obs_pre["precal.flux_cal_model_amp_jy"].value == pytest.approx(7.5, abs=0.2)

    obs_cal, _ = extract_calibrate(ws_mddid, ec_multi, ref_arrays_dir=None, scratch_root=tmp_path)
    assert obs_cal["cal.bandpass_cal_calibrated"].value is True
    assert obs_cal["cal.bandpass_coh_phase_slope_deg_per_100mhz"].value == pytest.approx(
        0.0, abs=1e-3
    )


def test_wave2_viscal_arbitrary_names_fluxscale_orientation_and_unapplied_fluxscale(
    synth_golden_precal_dir: Path, synth_golden_image_dir: Path, tmp_path: Path
) -> None:
    """Verify:
    1. In-place `input.ms` calibration without `calibrators.ms` scores `1.0` on `calibrate` and `e2e`.
    2. Arbitrary caltable filenames (`ap_flux.tab` < `z_unscaled.tab` alphabetically!) discover `VisCal`
       keywords and orient `(G_unscaled, F_fluxscaled)` physically (`~2.14 Jy`, NEVER reciprocal `0.467 Jy`).
    3. Wrong `fluxscale` (`5.0 Jy`) and unapplied `fluxscale` (`fluxscale` table == `2.14 Jy` while
       `CORRECTED_DATA` has unscaled `1.0 Jy` gains) both fail (`passed == 0`).
    """
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.runner import run_verification

    gt_cal = _load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json")
    gt_e2e = _load_gt(SYNTH_ROOT / "s2_e2e_synth" / "ground_truth.json")

    # 1. In-place `input.ms` without `calibrators.ms` on `s1_calibrate` and `s2_e2e_synth` -> 1.0
    ws_inplace = tmp_path / "ws_inplace"
    ws_inplace.mkdir()
    os.symlink(SYNTH_ROOT / "s1_image" / "input.ms", ws_inplace / "input.ms")
    rew_inp_cal, _ = grade_calibrate(
        ws_inplace,
        gt_cal,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert rew_inp_cal["reward"] == 1.0
    assert rew_inp_cal["passed"] == 1

    ws_inplace_e2e = tmp_path / "ws_inplace_e2e"
    ws_inplace_e2e.mkdir()
    os.symlink(SYNTH_ROOT / "s1_image" / "input.ms", ws_inplace_e2e / "input.ms")
    for sfx in ("image", "image.pbcor", "pb", "residual", "model"):
        os.symlink(
            synth_golden_image_dir / f"synth_mosaic.{sfx}",
            ws_inplace_e2e / f"synth_mosaic.{sfx}",
        )
    rew_inp_e2e, _ = grade_e2e(
        ws_inplace_e2e,
        gt_e2e,
        masks_npz_path=SYNTH_ROOT / "s2_e2e_synth" / "true_target_vis.npz",
        task_id="s2_e2e_synth",
        scratch_root=tmp_path,
    )
    assert rew_inp_e2e["reward"] == 1.0
    assert rew_inp_e2e["passed"] == 1

    # 2. Arbitrary caltable names where fluxscaled table sorts FIRST alphabetically (`ap_flux.tab` < `z_unscaled.tab`)
    ws_arb = tmp_path / "ws_arb_names"
    ws_arb.mkdir()
    shutil.copytree(SYNTH_ROOT / "s1_calibrate" / "input.ms", ws_arb / "input.ms")
    cal_ms = str(ws_arb / "input.ms")
    casatasks.flagdata(vis=cal_ms, mode="manual", antenna="ea07", flagbackup=False)
    casatasks.setjy(vis=cal_ms, field="J1331+3030", standard="Perley-Butler 2017", usescratch=True)

    g0_tab = str(ws_arb / "00_init_phase.tab")
    k_tab = str(ws_arb / "foo_delay.tab")
    b_tab = str(ws_arb / "bar_bp.tab")
    g_unscaled = str(ws_arb / "z_unscaled.tab")
    f_scaled = str(ws_arb / "ap_flux.tab")

    casatasks.gaincal(
        vis=cal_ms,
        caltable=g0_tab,
        field="J1331+3030,J1822-0938",
        spw="0:16~24",
        solint="int",
        refant="ea01",
        calmode="p",
        minsnr=3.0,
    )
    casatasks.gaincal(
        vis=cal_ms,
        caltable=k_tab,
        field="J1331+3030",
        solint="inf",
        combine="scan",
        refant="ea01",
        gaintype="K",
        gaintable=[g0_tab],
        gainfield=["J1331+3030"],
        minsnr=3.0,
    )
    casatasks.bandpass(
        vis=cal_ms,
        caltable=b_tab,
        field="J1331+3030",
        solint="inf",
        combine="scan",
        refant="ea01",
        bandtype="B",
        fillgaps=3,
        solnorm=True,
        gaintable=[g0_tab, k_tab],
        gainfield=["J1331+3030", "J1331+3030"],
        minsnr=3.0,
    )
    casatasks.gaincal(
        vis=cal_ms,
        caltable=g_unscaled,
        field="J1331+3030,J1822-0938",
        spw="0:4~59",
        solint="inf",
        refant="ea01",
        calmode="ap",
        gaintable=[k_tab, b_tab],
        gainfield=["J1331+3030", "J1331+3030"],
        minsnr=3.0,
    )
    casatasks.fluxscale(
        vis=cal_ms,
        caltable=g_unscaled,
        fluxtable=f_scaled,
        reference=["J1331+3030"],
        transfer=["J1822-0938"],
        incremental=False,
    )

    # First verify UNAPPLIED fluxscale (applying `z_unscaled.tab` instead of `ap_flux.tab`!)
    casatasks.applycal(
        vis=cal_ms,
        field="J1331+3030",
        gaintable=[k_tab, b_tab, g_unscaled],
        gainfield=["", "", "J1331+3030"],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
    )
    casatasks.applycal(
        vis=cal_ms,
        field="J1822-0938,SYNTH_MOSAIC_P1,SYNTH_MOSAIC_P2",
        gaintable=[k_tab, b_tab, g_unscaled],
        gainfield=["", "", "J1822-0938"],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
    )

    # Even on v0.1 parity rubric, unapplied fluxscale fails target fidelity -> passed == 0
    rew_unapp, met_unapp = grade_calibrate(
        ws_arb,
        gt_cal,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert met_unapp["measured"]["cal.fluxscale_table_flux_jy"] == pytest.approx(2.14, rel=0.03)
    assert met_unapp["measured"]["cal.phase_cal_corrected_amp_jy"] == pytest.approx(1.0, abs=0.05)
    assert rew_unapp["target_fidelity_score"] == 0.0
    assert rew_unapp["passed"] == 0

    # And on a strict v0.2 rubric requiring `cal.phase_cal_corrected_amp_jy` directly, `fluxscale_score == 0.0`
    from radio_bench.grading.schema import CombineRuleSpec, LeafRuleSpec

    ec_strict, rb_strict = resolve_task_configs("s1_calibrate", "calibrate", gt_cal)
    for sub in rb_strict.subscores:
        if sub.id == "fluxscale_score":
            sub.expr = CombineRuleSpec(
                combine="zero_if_any_zero",
                children=[
                    LeafRuleSpec(
                        observable="cal.phase_cal_unflagged_fraction",
                        primitive="threshold_gate",
                        params={"op": ">=", "limit": 0.5},
                    ),
                    LeafRuleSpec(
                        observable="cal.phase_cal_corrected_amp_jy",
                        primitive="rel_err_ramp",
                        params={
                            "truth_key": "phase_calibrator_truth.spw_mean_flux_jy",
                            "pass_rel_tol": 0.05,
                            "zero_at_rel_err": 0.25,
                        },
                    ),
                ],
            )
    rew_unapp_strict, _ = run_verification(
        workspace_dir=ws_arb,
        tests_dir=SYNTH_ROOT / "s1_calibrate",
        log_dir=None,
        extract_config=ec_strict,
        rubric=rb_strict,
        ground_truth=gt_cal,
        ref_arrays_dir=SYNTH_ROOT / "s1_calibrate",
        scratch_root=tmp_path,
    )
    assert rew_unapp_strict["fluxscale_score"] == 0.0
    assert rew_unapp_strict["passed"] == 0

    # Now apply `ap_flux.tab` properly + target rflag -> physical fluxscale orientation (`~2.14 Jy`) & reward == 1.0!
    casatasks.applycal(
        vis=cal_ms,
        field="J1822-0938,SYNTH_MOSAIC_P1,SYNTH_MOSAIC_P2",
        gaintable=[k_tab, b_tab, f_scaled],
        gainfield=["J1331+3030", "J1331+3030", "J1822-0938"],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
    )
    casatasks.flagdata(
        vis=cal_ms,
        mode="rflag",
        field="SYNTH_MOSAIC_P1,SYNTH_MOSAIC_P2",
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )
    rew_arb_ok, met_arb_ok = grade_calibrate(
        ws_arb,
        gt_cal,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert met_arb_ok["measured"]["cal.fluxscale_table_flux_jy"] == pytest.approx(2.14, rel=0.03)
    assert met_arb_ok["measured"]["cal.phase_cal_corrected_amp_jy"] == pytest.approx(2.14, rel=0.03)
    assert rew_arb_ok["fluxscale_score"] == 1.0
    assert rew_arb_ok["reward"] == 1.0
    assert rew_arb_ok["passed"] == 1

    rew_strict_ok, _ = run_verification(
        workspace_dir=ws_arb,
        tests_dir=SYNTH_ROOT / "s1_calibrate",
        log_dir=None,
        extract_config=ec_strict,
        rubric=rb_strict,
        ground_truth=gt_cal,
        ref_arrays_dir=SYNTH_ROOT / "s1_calibrate",
        scratch_root=tmp_path,
    )
    assert rew_strict_ok["fluxscale_score"] == 1.0
    assert rew_strict_ok["reward"] == 1.0
    assert rew_strict_ok["passed"] == 1

    # 3. Planted fault: wrong ~5.0 Jy fluxscale transfer (wrong primary flux density setjy) -> fluxscale_score == 0.0
    casatasks.setjy(
        vis=cal_ms,
        field="J1331+3030",
        standard="manual",
        fluxdensity=[7.61 * (5.0 / 2.14), 0.0, 0.0, 0.0],
        usescratch=True,
    )
    shutil.rmtree(g_unscaled)
    shutil.rmtree(f_scaled)
    casatasks.gaincal(
        vis=cal_ms,
        caltable=g_unscaled,
        field="J1331+3030,J1822-0938",
        spw="0:4~59",
        solint="inf",
        refant="ea01",
        calmode="ap",
        gaintable=[k_tab, b_tab],
        gainfield=["J1331+3030", "J1331+3030"],
        minsnr=3.0,
    )
    casatasks.fluxscale(
        vis=cal_ms,
        caltable=g_unscaled,
        fluxtable=f_scaled,
        reference=["J1331+3030"],
        transfer=["J1822-0938"],
        incremental=False,
    )
    casatasks.applycal(
        vis=cal_ms,
        field="J1822-0938,SYNTH_MOSAIC_P1,SYNTH_MOSAIC_P2",
        gaintable=[k_tab, b_tab, f_scaled],
        gainfield=["J1331+3030", "J1331+3030", "J1822-0938"],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
    )
    rew_wrong5, met_wrong5 = grade_calibrate(
        ws_arb,
        gt_cal,
        masks_npz_path=SYNTH_ROOT / "s1_calibrate" / "true_target_vis.npz",
        task_id="s1_calibrate",
        scratch_root=tmp_path,
    )
    assert met_wrong5["measured"]["cal.fluxscale_table_flux_jy"] == pytest.approx(5.0, rel=0.05)
    assert met_wrong5["measured"]["cal.phase_cal_corrected_amp_jy"] == pytest.approx(5.0, rel=0.05)
    assert rew_wrong5["fluxscale_score"] == 0.0
    assert rew_wrong5["passed"] == 0


def test_wave2_image_overfine_cell_pbcor_nonstealing_and_narrow_fov_pb(
    synth_golden_image_dir: Path, tmp_path: Path
) -> None:
    """Verify:
    1. Planted fault: overly fine pixel cell (`0.5 arcsec` -> `pixels_per_beam > 20`) -> `beam_sampling_score == 0.0`.
    2. Alternative recipe: multi-image workspace where `dirty_test.*` lacks `.pbcor` alongside complete `final.*`
       and a `*pbcor*.image` decoy -> selects `final` cleanly without stealing `final.image.pbcor`.
    3. Single-pointing narrow-FoV (`pb_min_full > 0.85`): analytical Airy/Gaussian PB passes (`pb_taper_consistent == True`),
       whereas constant `pb == 1.0` on the identical narrow-FoV grid fails (`pb_taper_consistent == False`).
    """
    from radio_bench.grading.extractors.image import extract_image
    from radio_bench.grading.inspect_metrics import resolve_task_configs

    gt_s1 = _load_gt(SYNTH_ROOT / "s1_image" / "ground_truth.json")

    # 1. Overly fine pixel cell (`0.5 arcsec` -> ~23.7 px/beam > 20) -> beam_sampling_score == 0.0
    ws_fine = tmp_path / "ws_fine_cell"
    ws_fine.mkdir()
    for sfx in ("image", "image.pbcor", "pb", "residual", "model"):
        shutil.copytree(
            synth_golden_image_dir / f"synth_mosaic.{sfx}",
            ws_fine / f"synth_mosaic.{sfx}",
        )
    ia = casatools.image()
    ia.open(str(ws_fine / "synth_mosaic.image"))
    try:
        cs = ia.coordsys()
        try:
            inc = cs.increment()["numeric"]
            rad_05 = math.radians(0.5 / 3600.0)
            cs.setincrement([-rad_05, rad_05, inc[2], inc[3]])
            ia.setcoordsys(cs.torecord())
        finally:
            with contextlib.suppress(Exception):
                cs.done()
    finally:
        ia.close()
        ia.done()

    rew_fine, met_fine = grade_image(ws_fine, gt_s1, task_id="s1_image")
    assert met_fine["measured"]["image.pixels_per_beam"] > 20.0
    assert rew_fine["beam_sampling_score"] == 0.0
    assert rew_fine["passed"] == 0

    # 2. Multi-image stem `.pbcor` non-stealing + `*pbcor*` stem exclusion
    ws_multi = tmp_path / "ws_multi_stems"
    ws_multi.mkdir()
    for sfx in ("image", "residual", "pb", "model"):
        shutil.copytree(
            synth_golden_image_dir / f"synth_mosaic.{sfx}",
            ws_multi / f"dirty_test.{sfx}",
        )
    # Zero out dirty_test.model so it mimics an uncleaned dirty image
    ia = casatools.image()
    ia.open(str(ws_multi / "dirty_test.model"))
    try:
        ia.putchunk(np.zeros_like(ia.getchunk()))
    finally:
        ia.close()
        ia.done()

    # Decoy `*pbcor*.image` directory
    shutil.copytree(
        synth_golden_image_dir / "synth_mosaic.image.pbcor",
        ws_multi / "decoy_pbcor_view.image",
    )
    # Complete `final.*` image set
    for sfx in ("image", "image.pbcor", "pb", "residual", "model"):
        shutil.copytree(
            synth_golden_image_dir / f"synth_mosaic.{sfx}",
            ws_multi / f"final.{sfx}",
        )

    rew_multi, met_multi = grade_image(ws_multi, gt_s1, task_id="s1_image")
    assert met_multi["measurements"]["image.products_complete"]["provenance"] == "final"
    assert rew_multi["products_score"] == 1.0
    assert rew_multi["reward"] == 1.0
    assert rew_multi["passed"] == 1

    # 3. Single-pointing narrow-FoV (`64x64`, `1.5 arcsec` cell at `4.6 GHz`, VLA `25m` dish -> `pb_min_full ~ 0.96 > 0.85`)
    ref_ia = casatools.image()
    ref_ia.open(str(synth_golden_image_dir / "synth_mosaic.image"))
    try:
        cs_obj = ref_ia.coordsys()
        try:
            rad_15as = math.radians(1.5 / 3600.0)
            inc = cs_obj.increment()["numeric"]
            cs_obj.setincrement([-rad_15as, rad_15as, inc[2], inc[3]])
            cs_obj.setreferencepixel([32.0, 32.0, 0.0, 0.0])
            cs_narrow = cs_obj.torecord()
        finally:
            with contextlib.suppress(Exception):
                cs_obj.done()
        rb_narrow = ref_ia.restoringbeam()
    finally:
        ref_ia.close()
        ref_ia.done()

    yy, xx = np.meshgrid(
        np.arange(64, dtype=np.float64), np.arange(64, dtype=np.float64), indexing="ij"
    )
    theta_rad = np.hypot(xx - 32.0, yy - 32.0) * math.radians(1.5 / 3600.0)
    lam_m = 299_792_458.0 / 4.599e9
    fwhm_rad = 1.22 * lam_m / 25.0
    pb_airy_2d = np.exp(-4.0 * math.log(2.0) * (theta_rad / fwhm_rad) ** 2).astype(np.float32)
    assert float(np.min(pb_airy_2d)) > 0.90  # Narrow FoV: pb_min_full > 0.85!

    rng = np.random.default_rng(77)
    sky_pbcor = np.zeros((64, 64, 1, 1), dtype=np.float32)
    sky_pbcor[32, 32, 0, 0] = 0.080
    pb_true = pb_airy_2d[:, :, np.newaxis, np.newaxis]
    img_true = (sky_pbcor * pb_true).astype(np.float32)
    res_true = rng.normal(0.0, 5.5e-5, size=(64, 64, 1, 1)).astype(np.float32)

    ec_img, _ = resolve_task_configs("s1_image", "image", gt_s1)
    ec_narrow = ec_img.model_copy(deep=True)
    ec_narrow.geometry.offsource_inner_exclusion_beams = 1.0
    ec_narrow.geometry.source_apertures = []

    ws_narrow_ok = tmp_path / "ws_narrow_ok"
    ws_narrow_ok.mkdir()
    for sfx, arr in (
        ("image", img_true),
        ("image.pbcor", sky_pbcor),
        ("model", sky_pbcor),
        ("pb", pb_true),
        ("residual", res_true),
    ):
        ia = casatools.image()
        ia.fromarray(
            outfile=str(ws_narrow_ok / f"narrow.{sfx}"), pixels=arr, csys=cs_narrow, overwrite=True
        )
        if sfx in ("image", "image.pbcor", "residual"):
            ia.setrestoringbeam(beam=rb_narrow)
        ia.close()
        ia.done()

    obs_narrow_ok, _ = extract_image(ws_narrow_ok, ec_narrow)
    assert obs_narrow_ok["image.pb_taper_consistent"].value is True

    # Replace `.pb` with constant `1.0` (and matching `pbcor = image`) -> MUST FAIL analytical PB check!
    ws_narrow_fake = tmp_path / "ws_narrow_fake"
    shutil.copytree(ws_narrow_ok, ws_narrow_fake)
    for sfx, arr in (("pb", np.ones_like(pb_true)), ("image.pbcor", img_true)):
        ia = casatools.image()
        ia.open(str(ws_narrow_fake / f"narrow.{sfx}"))
        try:
            ia.putchunk(arr)
        finally:
            ia.close()
            ia.done()

    obs_narrow_fake, _ = extract_image(ws_narrow_fake, ec_narrow)
    assert obs_narrow_fake["image.pb_taper_consistent"].value is False


def test_wave2_claim_auditor_report_html_nan_rejection_and_abstention_halt(
    synth_golden_image_dir: Path, tmp_path: Path
) -> None:
    """Verify Layer B Claim Auditor (`stage_output.json` + `report.html`), strict `NaN`/`Inf` rejection,
    and `expect_halt` / `proc.abstention_correct` / `invalid_abstention_or_halt` routing.
    """
    from radio_bench.analysis.taxonomy import classify_failure
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.runner import run_verification
    from radio_bench.grading.schema import HardGateSpec

    gt_img = _load_gt(SYNTH_ROOT / "s1_image" / "ground_truth.json")
    ec_img, rb_base = resolve_task_configs("s1_image", "image", gt_img)

    ws = tmp_path / "ws_claims"
    ws.mkdir()
    for sfx in ("image", "image.pbcor", "pb", "residual", "model"):
        os.symlink(
            synth_golden_image_dir / f"synth_mosaic.{sfx}",
            ws / f"synth_mosaic.{sfx}",
        )

    rb_claims = rb_base.model_copy(deep=True)
    rb_claims.required_reported_metrics = ["image.dynamic_range"]
    rb_claims.hard_gates.extend(
        [
            HardGateSpec(
                id="no_missing_claims",
                observable="proc.missing_required_claim_count",
                primitive="threshold_gate",
                params={"op": "<=", "limit": 0},
                failure_tag="unsupported_or_missing_claim",
            ),
            HardGateSpec(
                id="no_unsupported_claims",
                observable="proc.unsupported_claim_count",
                primitive="threshold_gate",
                params={"op": "<=", "limit": 0},
                failure_tag="unsupported_or_missing_claim",
            ),
        ]
    )

    # 1. Missing required claim -> missing_required_claim_count == 1, reward == 0.0, unsupported_or_missing_claim
    (ws / "stage_output.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "stage": "image",
                "status": "completed",
                "reported_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    rew_miss, met_miss = run_verification(
        workspace_dir=ws,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_claims,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_miss["measured"]["proc.missing_required_claim_count"] == 1
    assert rew_miss["reward"] == 0.0
    assert rew_miss["passed"] == 0
    assert (
        classify_failure(
            passed=rew_miss["passed"],
            stage="image",
            reward=float(rew_miss["reward"]),
            reward_dict=rew_miss,
            metrics_dict=met_miss,
        )
        == "unsupported_or_missing_claim"
    )

    # 2. Accurate claim in stage_output.json + unit-converted `mJy` span in report.html + ignored hidden/script spans -> 1.0!
    true_dr = float(met_miss["measured"]["image.dynamic_range"])
    true_rms_mjy = float(met_miss["measured"]["image.offsource_residual_rms_jy"]) * 1000.0
    (ws / "stage_output.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "stage": "image",
                "status": "completed",
                "reported_metrics": {
                    "image.dynamic_range": {"value": round(true_dr, 2), "unit": "dimensionless"}
                },
            }
        ),
        encoding="utf-8",
    )
    (ws / "report.html").write_text(
        f"""<!doctype html><html><body>
        <p>RMS: <span data-observable="image.offsource_residual_rms_jy" data-unit="mJy">{true_rms_mjy:.4f}</span> mJy/beam</p>
        <div hidden><span data-observable="image.dynamic_range">999999.0</span></div>
        <span style="display: none" data-observable="image.dynamic_range">888888.0</span>
        <script><span data-observable="image.dynamic_range">777777.0</span></script>
        </body></html>""",
        encoding="utf-8",
    )
    rew_acc, met_acc = run_verification(
        workspace_dir=ws,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_claims,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_acc["measured"]["proc.missing_required_claim_count"] == 0
    assert met_acc["measured"]["proc.unsupported_claim_count"] == 0
    assert rew_acc["reward"] == 1.0
    assert rew_acc["passed"] == 1

    # 3. Fabricated visible report.html claim -> unsupported_claim_count == 1, reward == 0.0
    (ws / "report.html").write_text(
        '<html><body><p>Peak DR: <span data-observable="image.dynamic_range">99999.0</span></p></body></html>',
        encoding="utf-8",
    )
    rew_fab, met_fab = run_verification(
        workspace_dir=ws,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_claims,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_fab["measured"]["proc.unsupported_claim_count"] >= 1
    assert rew_fab["reward"] == 0.0
    assert (
        classify_failure(
            passed=rew_fab["passed"],
            stage="image",
            reward=float(rew_fab["reward"]),
            reward_dict=rew_fab,
            metrics_dict=met_fab,
        )
        == "unsupported_or_missing_claim"
    )
    (ws / "report.html").unlink()

    # 4. Strict NaN rejection in stage_output.json -> proc.manifest_schema_valid is False
    (ws / "stage_output.json").write_text(
        '{"schema_version": "0.2", "stage": "image", "status": "completed", '
        '"reported_metrics": {"image.dynamic_range": {"value": NaN}}}',
        encoding="utf-8",
    )
    _, met_nan = run_verification(
        workspace_dir=ws,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_base,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_nan["measured"]["proc.manifest_schema_valid"] is False

    # 5. Invalid halt on solvable task (`expect_halt == False`) -> reward == 0.0, invalid_abstention_or_halt
    (ws / "stage_output.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "stage": "image",
                "status": "halted",
                "halt_code": "insufficient_parallactic_angle",
                "halt_reason": "Spurious halt on solvable continuum imaging task.",
            }
        ),
        encoding="utf-8",
    )
    rew_bad_halt, met_bad_halt = run_verification(
        workspace_dir=ws,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_base,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert rew_bad_halt["reward"] == 0.0
    assert rew_bad_halt["passed"] == 0
    assert (
        classify_failure(
            passed=rew_bad_halt["passed"],
            stage="image",
            reward=float(rew_bad_halt["reward"]),
            reward_dict=rew_bad_halt,
            metrics_dict=met_bad_halt,
        )
        == "invalid_abstention_or_halt"
    )

    # 6. `expect_halt == True` stress rubric:
    #    - Clean workspace WITHOUT downstream `.image` deliverables + valid halt -> scores 1.0!
    #    - Workspace WITH fabricated downstream `.image` deliverables (`ws`) + halt -> hedged halt blocked (0.0)!
    #    - Clean workspace with hedged downstream `image.dynamic_range` claim -> blocked (0.0)!
    #    - Unhalted `"completed"` manifest -> blocked (0.0)!
    rb_halt = rb_base.model_copy(
        update={
            "expect_halt": True,
            "expected_halt_codes": ["insufficient_parallactic_angle"],
        },
        deep=True,
    )
    ws_clean_halt = tmp_path / "ws_clean_halt"
    ws_clean_halt.mkdir()
    (ws_clean_halt / "stage_output.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "stage": "image",
                "status": "halted",
                "halt_code": "insufficient_parallactic_angle",
                "halt_reason": "Parallactic angle coverage is insufficient for polarization solve.",
            }
        ),
        encoding="utf-8",
    )
    rew_good_halt, met_good_halt = run_verification(
        workspace_dir=ws_clean_halt,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_halt,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_good_halt["measured"]["proc.abstention_correct"] is True
    assert rew_good_halt["abstention_score"] == 1.0
    assert rew_good_halt["reward"] == 1.0
    assert rew_good_halt["passed"] == 1

    # Hedged halt: `ws` has `.image` deliverables on disk alongside `status="halted"` -> 0.0!
    rew_hedged_disk, met_hedged_disk = run_verification(
        workspace_dir=ws,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_halt,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_hedged_disk["measured"]["proc.abstention_correct"] is False
    assert rew_hedged_disk["reward"] == 0.0
    assert rew_hedged_disk["passed"] == 0

    # Hedged downstream claim on clean halted workspace -> 0.0!
    (ws_clean_halt / "stage_output.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "stage": "image",
                "status": "halted",
                "halt_code": "insufficient_parallactic_angle",
                "halt_reason": "Halted before imaging.",
                "reported_metrics": {"image.dynamic_range": {"value": 500.0}},
            }
        ),
        encoding="utf-8",
    )
    rew_hedged_claim, met_hedged_claim = run_verification(
        workspace_dir=ws_clean_halt,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_halt,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_hedged_claim["measured"]["proc.abstention_correct"] is False
    assert rew_hedged_claim["reward"] == 0.0
    assert rew_hedged_claim["passed"] == 0

    (ws_clean_halt / "stage_output.json").write_text(
        json.dumps({"schema_version": "0.2", "stage": "image", "status": "completed"}),
        encoding="utf-8",
    )
    rew_unhalted, met_unhalted = run_verification(
        workspace_dir=ws_clean_halt,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_halt,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_unhalted["measured"]["proc.abstention_correct"] is False
    assert rew_unhalted["reward"] == 0.0
    assert rew_unhalted["passed"] == 0
    assert (
        classify_failure(
            passed=rew_unhalted["passed"],
            stage="image",
            reward=float(rew_unhalted["reward"]),
            reward_dict=rew_unhalted,
            metrics_dict=met_unhalted,
        )
        == "invalid_abstention_or_halt"
    )


def test_wave2_extractor_and_repo_purity_invariants() -> None:
    """Static verification of zero dataset branches, zero `j1331`/`j1822` keys in `grading/`,
    and zero hardcoded host paths or em/en dashes across `src/`, `suites/`, `pixi.toml`, `images/`.
    """
    repo_root = Path(__file__).resolve().parents[1]
    extractors_dir = repo_root / "src" / "radio_bench" / "grading" / "extractors"
    grading_dir = repo_root / "src" / "radio_bench" / "grading"

    for py_p in sorted(extractors_dir.glob("*.py")):
        txt = py_p.read_text(encoding="utf-8")
        for forbidden in (
            "ground_truth",
            "is_suite_r",
            "if suite",
            "j1331",
            "j1822",
            "3c391",
            "pass_reward_min",
        ):
            assert forbidden not in txt.lower(), (
                f"{py_p.name} contains forbidden token {forbidden!r}"
            )

    for py_p in sorted(grading_dir.rglob("*.py")):
        txt = py_p.read_text(encoding="utf-8")
        for forbidden in ("is_suite_r", "j1331", "j1822"):
            assert forbidden not in txt.lower(), (
                f"{py_p} contains dataset-specific token {forbidden!r}"
            )

    scan_paths: list[Path] = [repo_root / "pixi.toml"]
    for sub in ("src", "suites", "images"):
        scan_paths.extend(
            p for p in (repo_root / sub).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    forbidden_host_tokens = (
        "/" + "/".join(("home", "pjaganna")),
        "/" + "/".join(("var", "home")),
        "/" + "/".join(("mnt", "fast")),
    )
    for p in scan_paths:
        txt = p.read_text(encoding="utf-8")
        for host_tok in forbidden_host_tokens:
            assert host_tok not in txt, f"{p} contains hardcoded host path {host_tok!r}"
        assert "\u2014" not in txt and "\u2013" not in txt, f"{p} contains em/en dash"


def test_wave2_adversarial_review_regressions(synth_golden_image_dir: Path, tmp_path: Path) -> None:
    """Post-build adversarial & code-review regression suite covering:
    - `cal.coherent_vis_pearson_r` `0.25` parity (`0.30` passes, `0.20` fails).
    - Compile-time `LeafRuleSpec` / `HardGateSpec` parameter-key and `op` validation.
    - `ReportedMetric` `"+Infinity"` / `"1e500"` rejection & malformed `stage_output.json` forcing `passed == 0`.
    - `reported_metrics: {}` missing-claim `passed == 0` & `claim_max_scale_err` non-zero/missing_artifact even without hard gates.
    - HTML5 `<div style="display:none" />`, auto-closed `<p>`, `opacity:0`, `font-size:0px`, `aria-hidden="true"`,
      Unicode micro (`&mu;Jy`, `&#181;Jy`), `deg/GHz` & `deg/MHz`, `mJy km/s`, incompatible units, trailing-garbage scalars,
      and `list[float]` claims.
    - Non-finite `1e200` pixel `float64` overflow (`1e200**2 -> inf`) safe `extractor_error` handling + `e2e` stage crash isolation.
    - Zero `<frozen runpy>:128: RuntimeWarning` under `PYTHONWARNINGS=error python -m radio_bench.grading.runner --help`.
    """
    import subprocess
    import sys

    from pydantic import ValidationError

    from radio_bench.grading.extractors import WorkspaceExtractionResult, extract_workspace
    from radio_bench.grading.extractors.image import extract_image
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.rubric import (
        audit_claims,
        evaluate_rubric,
        extract_visible_html_claims,
        normalize_claim_unit,
    )
    from radio_bench.grading.runner import run_verification
    from radio_bench.grading.schema import (
        HardGateSpec,
        LeafRuleSpec,
        ObservableRecord,
        ReportedMetric,
    )

    # 1. Compile-time primitive parameter validation & ReportedMetric "+Infinity" / "1e500" rejection
    with pytest.raises(ValidationError):
        LeafRuleSpec(
            observable="cal.phase_cal_corrected_amp_jy",
            primitive="rel_err_ramp",
            params={"pass_rel": 0.05},
        )
    with pytest.raises(ValidationError):
        HardGateSpec(
            id="bad_op",
            observable="proc.missing_required_claim_count",
            primitive="threshold_gate",
            params={"op": "!=", "limit": 0},
            failure_tag="unsupported_or_missing_claim",
        )
    for bad_inf in ("+Infinity", "-Infinity", "1e500", "-1e500"):
        with pytest.raises(ValidationError):
            ReportedMetric(value=bad_inf)

    # 2. `cal.coherent_vis_pearson_r` 0.25 parity (`0.30` passes `target_fidelity_score`, `0.20` fails)
    gt_cal = _load_gt(SYNTH_ROOT / "s1_calibrate" / "ground_truth.json")
    _, rb_cal = resolve_task_configs("s1_calibrate", "calibrate", gt_cal)
    base_cal_obs = {
        "proc.table_locked": ObservableRecord(value=False, status="measured"),
        "cal.dead_antenna_flagged_fraction": ObservableRecord(value=1.0, status="measured"),
        "cal.target_row_retention": ObservableRecord(value=0.86, status="measured"),
        "cal.corrected_distinct_from_data": ObservableRecord(value=True, status="measured"),
        "cal.target_corrected_std_jy": ObservableRecord(value=0.08, status="measured", unit="Jy"),
        "cal.rank1_logamp_rms_median": ObservableRecord(value=0.01, status="measured"),
        "cal.bandpass_coh_phase_slope_deg_per_100mhz": ObservableRecord(
            value=0.1, status="measured", unit="deg/100MHz"
        ),
        "cal.bandpass_med_bl_phase_slope_deg_per_100mhz": ObservableRecord(
            value=0.2, status="measured", unit="deg/100MHz"
        ),
        "cal.bandpass_cal_calibrated": ObservableRecord(value=True, status="measured"),
        "cal.bandpass_cal_unflagged_fraction": ObservableRecord(value=0.80, status="measured"),
        "cal.phase_cal_corrected_amp_jy": ObservableRecord(
            value=2.15, status="measured", unit="Jy"
        ),
        "cal.fluxscale_table_flux_jy": ObservableRecord(value=2.15, status="measured", unit="Jy"),
        "cal.phase_cal_unflagged_fraction": ObservableRecord(value=0.80, status="measured"),
        "cal.coherent_vis_amp_rel_err": ObservableRecord(value=0.01, status="measured"),
        "cal.coherent_vis_phase_err_deg": ObservableRecord(
            value=0.5, status="measured", unit="deg"
        ),
        "cal.clean_target_unflagged_fraction": ObservableRecord(value=0.99, status="measured"),
        "cal.coherent_vis_pearson_r": ObservableRecord(value=0.30, status="measured"),
    }
    res_r030 = evaluate_rubric(
        tmp_path, WorkspaceExtractionResult(observables=base_cal_obs), rb_cal, gt_cal
    )
    assert res_r030.subscores["target_fidelity_score"] == 1.0
    assert res_r030.passed == 1

    obs_r020 = dict(base_cal_obs)
    obs_r020["cal.coherent_vis_pearson_r"] = ObservableRecord(value=0.20, status="measured")
    res_r020 = evaluate_rubric(
        tmp_path, WorkspaceExtractionResult(observables=obs_r020), rb_cal, gt_cal
    )
    assert res_r020.subscores["target_fidelity_score"] == 0.0
    assert res_r020.passed == 0

    # 3. `reported_metrics: {}` gaming WITHOUT explicit hard_gates + malformed `stage_output.json` forcing `passed == 0`
    gt_img = _load_gt(SYNTH_ROOT / "s1_image" / "ground_truth.json")
    ec_img, rb_img_base = resolve_task_configs("s1_image", "image", gt_img)
    rb_req_no_hg = rb_img_base.model_copy(
        update={"required_reported_metrics": ["image.dynamic_range"]}, deep=True
    )

    ws_claim_edge = tmp_path / "ws_claim_edge"
    ws_claim_edge.mkdir()
    for sfx in ("image", "image.pbcor", "pb", "residual", "model"):
        os.symlink(
            synth_golden_image_dir / f"synth_mosaic.{sfx}",
            ws_claim_edge / f"synth_mosaic.{sfx}",
        )
    (ws_claim_edge / "stage_output.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "stage": "image",
                "status": "completed",
                "reported_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    rew_empty_rep, met_empty_rep = run_verification(
        workspace_dir=ws_claim_edge,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_req_no_hg,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_empty_rep["measurements"]["proc.claim_max_scale_err"]["status"] == "missing_artifact"
    assert rew_empty_rep["passed"] == 0
    assert "unsupported_or_missing_claim" in met_empty_rep["triggered_failure_tags"]

    # Malformed `stage_output.json` (extra forbidden field + fabricated reported_metrics) forces `passed == 0`
    # AND still audits `raw_manifest_dict["reported_metrics"]`!
    (ws_claim_edge / "stage_output.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "stage": "image",
                "status": "completed",
                "unexpected_extra_key": 123,
                "reported_metrics": {"image.dynamic_range": {"value": 999999.0}},
            }
        ),
        encoding="utf-8",
    )
    rew_mal, met_mal = run_verification(
        workspace_dir=ws_claim_edge,
        tests_dir=SYNTH_ROOT / "s1_image",
        log_dir=None,
        extract_config=ec_img,
        rubric=rb_img_base,
        ground_truth=gt_img,
        compute_raw_hash=False,
    )
    assert met_mal["measured"]["proc.manifest_schema_valid"] is False
    assert met_mal["measured"]["proc.unsupported_claim_count"] == 1
    assert rew_mal["passed"] == 0
    assert "unsupported_or_missing_claim" in met_mal["triggered_failure_tags"]

    # 4. HTML5 DOM anti-spoofing, Unicode micro units, deg/GHz, mJy km/s, incompatible units, & list[float] claims
    html_file = tmp_path / "spoof_test.html"
    html_file.write_text(
        """<!doctype html><html><body>
        <div style="display:none" /><span data-observable="image.dynamic_range">999999</span></div>
        <span style="opacity: 0" data-observable="image.dynamic_range">888888</span>
        <span style="font-size: 0px" data-observable="image.dynamic_range">777777</span>
        <span aria-hidden="true" data-observable="image.dynamic_range">666666</span>
        <p style="display:none">hidden para<p><span data-observable="cal.bandpass_coh_phase_slope_deg_per_100mhz">12.0 deg/GHz</span>
        <span data-observable="image.offsource_residual_rms_jy">127.1 &mu;Jy/beam</span>
        <span data-observable="cube.integrated_flux_jy_kms" data-unit="mJy km/s">2500.0</span>
        </body></html>""",
        encoding="utf-8",
    )
    spans = extract_visible_html_claims(html_file)
    assert [s.observable for s in spans] == [
        "cal.bandpass_coh_phase_slope_deg_per_100mhz",
        "image.offsource_residual_rms_jy",
        "cube.integrated_flux_jy_kms",
    ]

    assert normalize_claim_unit(100.0, "\u00b5Jy/beam", "Jy/beam") == (
        True,
        pytest.approx(1.0e-4),
    )
    assert normalize_claim_unit(5.0, "MHz", "Jy") == (False, 5.0)

    ws_unit_check = tmp_path / "ws_unit_check"
    ws_unit_check.mkdir()
    shutil.copyfile(html_file, ws_unit_check / "report.html")
    unit_obs = {
        "cal.bandpass_coh_phase_slope_deg_per_100mhz": ObservableRecord(
            value=1.2, status="measured", unit="deg/100MHz"
        ),
        "image.offsource_residual_rms_jy": ObservableRecord(
            value=1.271e-4, status="measured", unit="Jy/beam"
        ),
        "cube.integrated_flux_jy_kms": ObservableRecord(
            value=2.5, status="measured", unit="Jy km/s"
        ),
        "precal.flux_cal_model_spectrum_jy": ObservableRecord(
            value=[7.60, 7.55, 7.50], status="measured", unit="Jy"
        ),
    }
    c_obs_ok, _, _ = audit_claims(
        ws_unit_check,
        rb_img_base,
        None,
        unit_obs,
        raw_manifest_dict={
            "reported_metrics": {
                "precal.flux_cal_model_spectrum_jy": {
                    "value": [7600.0, 7550.0, 7500.0],
                    "unit": "mJy",
                }
            }
        },
    )
    assert c_obs_ok["proc.unsupported_claim_count"].value == 0

    # Incompatible unit ("MHz" on Jy) and trailing-garbage string ("1.2 bogus 9.9") both fail
    (ws_unit_check / "report.html").write_text(
        '<html><body><span data-observable="image.offsource_residual_rms_jy" data-unit="MHz">0.0001271</span>'
        '<span data-observable="cal.bandpass_coh_phase_slope_deg_per_100mhz">1.2 bogus 9.9</span></body></html>',
        encoding="utf-8",
    )
    c_obs_bad, _, _ = audit_claims(ws_unit_check, rb_img_base, None, unit_obs)
    assert c_obs_bad["proc.unsupported_claim_count"].value == 2

    # 5. Non-finite `1e200` pixel overflow (`1e200**2 -> inf` in float64) safe `extractor_error` + `e2e` crash isolation
    ws_inf_img = tmp_path / "ws_inf_img"
    ws_inf_img.mkdir()
    for sfx in ("image", "image.pbcor", "pb", "residual", "model"):
        shutil.copytree(
            synth_golden_image_dir / f"synth_mosaic.{sfx}",
            ws_inf_img / f"synth_mosaic.{sfx}",
        )
    ia = casatools.image()
    ia.open(str(ws_inf_img / "synth_mosaic.residual"))
    try:
        r_arr = ia.getchunk().astype(np.float64)
        r_arr[:] = 1.0e200
        # Write huge finite float32 max or float64 image to test overflow resilience
        ia.putchunk(np.full_like(ia.getchunk(), 3.0e38))
    finally:
        ia.close()
        ia.done()

    # Directly verify `1e200**2 -> inf` overflow converts to `extractor_error` via `finite_observable`
    from radio_bench.grading.schema import finite_observable

    with np.errstate(over="ignore"):
        ov_rec = finite_observable(float(np.float64(1e200) ** 2), unit="Jy/beam")
    assert ov_rec.status == "extractor_error" and ov_rec.value is None
    obs_huge, _ = extract_image(ws_inf_img, ec_img)
    assert obs_huge["image.offsource_residual_rms_jy"].status == "measured"

    # Verify `extract_workspace` stage crash isolation on `e2e`
    import radio_bench.grading.extractors as ext_pkg

    ec_e2e, _ = resolve_task_configs("s2_e2e_synth", "e2e", gt_img)
    orig_cal_fn = ext_pkg.extract_calibrate
    try:

        def _boom(*_a: object, **_kw: object) -> tuple[dict[str, ObservableRecord], list[str]]:
            raise RuntimeError("simulated calibrate crash")

        ext_pkg.extract_calibrate = _boom  # type: ignore[assignment]
        ext_iso = extract_workspace(ws_inf_img, ec_e2e, compute_raw_hash=False)
        assert any("simulated calibrate crash" in e for e in ext_iso.errors)
        assert ext_iso.observables["image.valid_product_count"].value == 5
    finally:
        ext_pkg.extract_calibrate = orig_cal_fn

    # 6. `PYTHONWARNINGS=error python -m radio_bench.grading.runner --help` exits 0 cleanly!
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "error"
    src_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{src_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_root
    proc_help = subprocess.run(
        [sys.executable, "-m", "radio_bench.grading.runner", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert proc_help.returncode == 0, f"STDERR:\n{proc_help.stderr}"


def test_part2_precal_role_scoped_observables_and_calibrator_split_rubrics(
    tmp_path: Path,
) -> None:
    """Verify role-scoped precal flag observables, calibrator-split 1.0 parity, and blanket-cal guard."""
    from radio_bench.grading.extractors import WorkspaceExtractionResult
    from radio_bench.grading.extractors.precal import extract_precal
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.rubric import evaluate_rubric
    from radio_bench.grading.schema import ObservableRecord

    gt_s1 = _load_gt(SYNTH_ROOT / "s1_precal" / "ground_truth.json")
    masks_s1 = SYNTH_ROOT / "s1_precal" / "corruption_masks.npz"

    # 1. Build an arm-neutral calibrator-split `s1_precal` workspace:
    #    - `input.ms` gets ONLY deterministic preflags (`quack_rows | dead_rows`), deferring
    #      science-target statistical RFI (`rfi_channels`, target `burst_rows`) to `postcal`.
    #    - `calibrators.ms` gets full calibrator RFI flagging (`burst_rows` + `rfi_channels`) + `setjy`.
    ws_split = tmp_path / "ws_s1_cal_split"
    ws_split.mkdir()
    in_ms = ws_split / "input.ms"
    cal_ms = ws_split / "calibrators.ms"
    shutil.copytree(SYNTH_ROOT / "s1_precal" / "input.ms", in_ms)

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

    with np.load(masks_s1, allow_pickle=False) as z:
        quack_rows = z["quack_rows"].astype(bool)
        burst_rows = z["burst_rows"].astype(bool)
        dead_rows = z["dead_rows"].astype(bool)
        rfi_channels = z["rfi_channels"].astype(int).tolist()

    tb_m = casatools.table()
    tb_m.open(str(in_ms), nomodify=False)
    try:
        fids = tb_m.getcol("FIELD_ID")
        sids = np.array([mode_to_sid[intents_by_fid[int(f)]] for f in fids], dtype=np.int32)
        tb_m.putcol("STATE_ID", sids)
        flg = tb_m.getcol("FLAG")
        # Deterministic preflags ONLY on input.ms (no target rfi_channels or target burst_rows!)
        flg[:, :, quack_rows | dead_rows] = True
        cal_burst = burst_rows & np.isin(fids, [0, 1])
        flg[:, :, cal_burst] = True
        tb_m.putcol("FLAG", flg)
        tb_m.putcol("FLAG_ROW", np.all(flg, axis=(0, 1)))
    finally:
        tb_m.close()
        tb_m.done()

    casatasks.split(
        vis=str(in_ms),
        outputvis=str(cal_ms),
        field="J1331+3030,J1822-0938",
        datacolumn="data",
        keepflags=False,
    )
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

    rew_split, met_split = grade_precal(
        ws_split, gt_s1, masks_npz_path=masks_s1, task_id="s1_precal"
    )
    meas_split = met_split["measured"]
    assert meas_split["precal.calibrator_must_flag_recall"] == pytest.approx(1.0)
    assert 0.85 <= meas_split["precal.target_must_flag_recall"] < 1.0
    assert meas_split["precal.calibrator_clean_false_positive_rate"] == pytest.approx(0.0)
    assert meas_split["precal.target_clean_false_positive_rate"] == pytest.approx(0.0)
    assert rew_split["flag_recall_score"] == pytest.approx(1.0)
    assert rew_split["clean_fpr_score"] == pytest.approx(1.0)
    assert rew_split["reward"] == pytest.approx(1.0)
    assert rew_split["passed"] == 1

    # 2. Verify missing/renamed fields -> `missing_artifact`, empty role list -> `not_applicable`,
    #    missing `input.ms` -> `missing_artifact`, and vacuous submasks -> `1.0` / `0.0` (`measured`).
    ec_s1, _ = resolve_task_configs("s1_precal", "precal", gt_s1)
    ec_renamed_tgt = ec_s1.model_copy(
        update={"roles": ec_s1.roles.model_copy(update={"targets": ["NONEXISTENT_TARGET_FIELD"]})},
        deep=True,
    )
    obs_renamed_tgt, _ = extract_precal(
        ws_split, ec_renamed_tgt, ref_arrays_dir=SYNTH_ROOT / "s1_precal"
    )
    assert obs_renamed_tgt["precal.calibrator_must_flag_recall"].status == "measured"
    assert obs_renamed_tgt["precal.target_must_flag_recall"].status == "missing_artifact"
    assert obs_renamed_tgt["precal.target_clean_false_positive_rate"].status == "missing_artifact"

    ec_empty_tgt = ec_s1.model_copy(
        update={"roles": ec_s1.roles.model_copy(update={"targets": []})},
        deep=True,
    )
    obs_empty_tgt, _ = extract_precal(
        ws_split, ec_empty_tgt, ref_arrays_dir=SYNTH_ROOT / "s1_precal"
    )
    assert obs_empty_tgt["precal.target_must_flag_recall"].status == "not_applicable"
    assert obs_empty_tgt["precal.target_clean_false_positive_rate"].status == "not_applicable"

    ws_empty_dir = tmp_path / "ws_missing_input_ms"
    ws_empty_dir.mkdir()
    obs_no_in, _ = extract_precal(ws_empty_dir, ec_s1, ref_arrays_dir=SYNTH_ROOT / "s1_precal")
    assert obs_no_in["precal.must_flag_recall"].status == "missing_artifact"
    assert obs_no_in["precal.calibrator_must_flag_recall"].status == "missing_artifact"
    assert obs_no_in["precal.target_must_flag_recall"].status == "missing_artifact"

    vac_npz = tmp_path / "vacuous_masks.npz"
    with np.load(masks_s1, allow_pickle=False) as z:
        c_shape = z["corrupted_chan_row"].shape
    np.savez(
        vac_npz,
        must_flag_2d=np.zeros(c_shape, dtype=bool),
        clean_2d=np.zeros(c_shape, dtype=bool),
    )
    obs_vac, _ = extract_precal(ws_split, ec_s1, ref_arrays_dir=vac_npz)
    assert obs_vac["precal.calibrator_must_flag_recall"].status == "measured"
    assert obs_vac["precal.calibrator_must_flag_recall"].value == pytest.approx(1.0)
    assert obs_vac["precal.calibrator_clean_false_positive_rate"].value == pytest.approx(0.0)
    assert obs_vac["precal.target_must_flag_recall"].value == pytest.approx(1.0)
    assert obs_vac["precal.target_clean_false_positive_rate"].value == pytest.approx(0.0)

    # 2b. Verify `keepflags=False` calibrator row-drop:
    #     - Legitimate: calibrator rows in `input.ms` are left completely UNFLAGGED (`False`),
    #       while `calibrators.ms` dropped `quack/dead/cal_burst` rows via `split(..., keepflags=False)`
    #       -> absent calibrator rows are marked `True` in merged `eff_2d` -> `reward == 1.0`!
    ws_kf = tmp_path / "ws_keepflags_false"
    ws_kf.mkdir()
    shutil.copytree(in_ms, ws_kf / "input.ms")
    shutil.copytree(cal_ms, ws_kf / "calibrators.ms")
    tb_kf_in = casatools.table()
    tb_kf_in.open(str(ws_kf / "input.ms"), nomodify=False)
    try:
        kf_fids = tb_kf_in.getcol("FIELD_ID")
        kf_flg = tb_kf_in.getcol("FLAG")
        cal_in_rows = np.isin(kf_fids, [0, 1])
        kf_flg[:, :, cal_in_rows] = False
        tb_kf_in.putcol("FLAG", kf_flg)
        tb_kf_in.putcol("FLAG_ROW", np.all(kf_flg, axis=(0, 1)))
    finally:
        tb_kf_in.close()
        tb_kf_in.done()

    rew_kf_ok, met_kf_ok = grade_precal(ws_kf, gt_s1, masks_npz_path=masks_s1, task_id="s1_precal")
    assert met_kf_ok["measured"]["precal.calibrator_must_flag_recall"] == pytest.approx(1.0)
    assert met_kf_ok["measured"]["precal.calibrator_clean_false_positive_rate"] == pytest.approx(
        0.0
    )
    assert rew_kf_ok["reward"] == pytest.approx(1.0)
    assert rew_kf_ok["passed"] == 1

    #     - Adversarial: dropping clean calibrator rows/baselines from `calibrators.ms` marks those
    #       absent rows `True`, spiking `calibrator_clean_false_positive_rate > 0.50` -> `0.0`!
    shutil.rmtree(ws_kf / "calibrators.ms")
    casatasks.split(
        vis=str(cal_ms),
        outputvis=str(ws_kf / "calibrators.ms"),
        antenna="0&1",
        datacolumn="all",
        keepflags=True,
    )
    casatasks.clearstat()

    rew_kf_drop, met_kf_drop = grade_precal(
        ws_kf, gt_s1, masks_npz_path=masks_s1, task_id="s1_precal"
    )
    assert met_kf_drop["measured"]["precal.calibrator_clean_false_positive_rate"] > 0.50
    assert rew_kf_drop["flag_recall_score"] == 0.0
    assert rew_kf_drop["clean_fpr_score"] == 0.0
    assert rew_kf_drop["passed"] == 0

    # 3. Blanket-flagging `calibrators.ms` (`100%` of calibrators.ms flagged while science targets
    #    remain clean so overall `clean_false_positive_rate < 0.50`) -> 0.0 on BOTH subscores!
    tb_c2 = casatools.table()
    tb_c2.open(str(cal_ms), nomodify=False)
    try:
        cflg2 = tb_c2.getcol("FLAG")
        cflg2[:, :, :] = True
        tb_c2.putcol("FLAG", cflg2)
        tb_c2.putcol("FLAG_ROW", np.ones(cflg2.shape[2], dtype=bool))
    finally:
        tb_c2.close()
        tb_c2.done()

    rew_blanket_cal, met_blanket_cal = grade_precal(
        ws_split, gt_s1, masks_npz_path=masks_s1, task_id="s1_precal"
    )
    assert met_blanket_cal["measured"]["precal.clean_false_positive_rate"] < 0.50
    assert met_blanket_cal["measured"][
        "precal.calibrator_clean_false_positive_rate"
    ] == pytest.approx(1.0)
    assert rew_blanket_cal["flag_recall_score"] == 0.0
    assert rew_blanket_cal["clean_fpr_score"] == 0.0
    assert rew_blanket_cal["passed"] == 0

    # 4. Suite R (`r1_3c391_precal`): verify golden checkpoint observables + calibrator-split 1.0
    #    + blanket-calibrator guard (`0.0` subscores, `passed == 0`)
    ws_r1_gold = tmp_path / "ws_r1_gold_roles"
    ws_r1_gold.mkdir()
    os.symlink(R3C391_ROOT / "post_precal" / "input.ms", ws_r1_gold / "input.ms")
    os.symlink(R3C391_ROOT / "post_precal" / "calibrators.ms", ws_r1_gold / "calibrators.ms")
    rew_r1_g, met_r1_g = grade_task(
        "r1_3c391_precal",
        "precal",
        ws_r1_gold,
        R3C391_ROOT / "r1_3c391_precal",
        tmp_path / "log_r1_gold_roles",
    )
    assert rew_r1_g["reward"] == pytest.approx(1.0)
    assert rew_r1_g["passed"] == 1
    assert met_r1_g["measured"]["precal.calibrator_must_flag_recall"] == pytest.approx(1.0)
    assert met_r1_g["measured"]["precal.target_must_flag_recall"] == pytest.approx(1.0)
    assert met_r1_g["measured"]["precal.calibrator_clean_false_positive_rate"] == pytest.approx(0.0)
    assert met_r1_g["measured"]["precal.target_clean_false_positive_rate"] == pytest.approx(0.0)

    gt_r1 = _load_gt(R3C391_ROOT / "r1_3c391_precal" / "ground_truth.json")
    _, rb_r1 = resolve_task_configs("r1_3c391_precal", "precal", gt_r1)
    r1_split_obs = {
        k: ObservableRecord(**v)
        for k, v in met_r1_g["measurements"].items()
        if isinstance(v, dict) and "status" in v
    }
    # Deterministic preflags (`dead + 10s quack`) on 3C391 targets achieve ~41.88% target recall
    r1_split_obs["precal.calibrator_must_flag_recall"] = ObservableRecord(
        value=1.0, status="measured"
    )
    r1_split_obs["precal.target_must_flag_recall"] = ObservableRecord(
        value=0.4188, status="measured"
    )
    r1_split_obs["precal.must_flag_recall"] = ObservableRecord(value=0.585, status="measured")
    r1_split_obs["precal.clean_false_positive_rate"] = ObservableRecord(
        value=0.002, status="measured"
    )
    r1_split_obs["precal.calibrator_clean_false_positive_rate"] = ObservableRecord(
        value=0.004, status="measured"
    )
    r1_split_obs["precal.target_clean_false_positive_rate"] = ObservableRecord(
        value=0.001, status="measured"
    )
    eval_r1_split = evaluate_rubric(
        ws_r1_gold, WorkspaceExtractionResult(observables=r1_split_obs), rb_r1, gt_r1
    )
    assert eval_r1_split.subscores["flag_recall_score"] == pytest.approx(1.0)
    assert eval_r1_split.subscores["clean_fpr_score"] == pytest.approx(1.0)
    assert eval_r1_split.reward == pytest.approx(1.0)
    assert eval_r1_split.passed == 1

    # Blanket-flagging `calibrators.ms` on 3C391 (overall FPR ~0.16 < 0.50, calibrator FPR == 1.0)
    r1_blanket_obs = dict(r1_split_obs)
    r1_blanket_obs["precal.clean_false_positive_rate"] = ObservableRecord(
        value=0.16, status="measured"
    )
    r1_blanket_obs["precal.calibrator_clean_false_positive_rate"] = ObservableRecord(
        value=1.0, status="measured"
    )
    eval_r1_blanket = evaluate_rubric(
        ws_r1_gold, WorkspaceExtractionResult(observables=r1_blanket_obs), rb_r1, gt_r1
    )
    assert eval_r1_blanket.subscores["flag_recall_score"] == 0.0
    assert eval_r1_blanket.subscores["clean_fpr_score"] == 0.0
    assert eval_r1_blanket.passed == 0
