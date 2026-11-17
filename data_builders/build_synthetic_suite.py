"""Deterministic builder and CASA physics verifier for Suite S synthetic datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from data_builders.sim_common import (  # noqa: E402
    DEAD_ANTENNA_ID,
    DEAD_ANTENNA_NAME,
    EXT_COMP_DEC,
    EXT_COMP_FLUX_JY,
    EXT_COMP_MAJOR,
    EXT_COMP_MINOR,
    EXT_COMP_PA,
    EXT_COMP_RA,
    FLUX_CAL_DEC,
    FLUX_CAL_NAME,
    FLUX_CAL_RA,
    INTEGRATION_SEC,
    MOSAIC_PHASECENTER,
    PHASE_CAL_DEC,
    PHASE_CAL_FLUX_JY,
    PHASE_CAL_NAME,
    PHASE_CAL_RA,
    PHASE_CAL_SPIX,
    PT_COMP_DEC,
    PT_COMP_RA,
    REF_EPOCH_UTC,
    REFANT_NAME,
    SPW_CHAN_WIDTH_HZ,
    SPW_NCHAN,
    SPW_START_HZ,
    SYNTH_ANTENNA_NAMES,
    TARGET_P1_DEC,
    TARGET_P1_NAME,
    TARGET_P1_RA,
    TARGET_P2_DEC,
    TARGET_P2_NAME,
    TARGET_P2_RA,
    TRUE_DELAYS_NS,
    apply_corruptions_to_ms,
    assign_scan_intents,
    clear_ms_history,
    compute_theoretical_mosaic_sigma_jy,
    configure_builder_casa_env,
    create_base_noiseless_ms,
    fixed_scan_intents,
    verify_ms_unlocked,
)
from radio_bench.config import (  # noqa: E402
    BenchConfig,
    add_config_cli_args,
    config_from_cli_args,
    load_bench_config,
)

SYNTH_SEED = 20261113
SIGMA_VIS_JY = 0.0527
PT_SOURCE_SIGMA_MULT = 10.0


def __getattr__(name: str) -> Any:
    if name in {"DEFAULT_OUT_DIR", "DEFAULT_SCRATCH_ROOT"}:
        cfg = load_bench_config()
        if name == "DEFAULT_OUT_DIR":
            return cfg.data_dir / "synth"
        return cfg.tmp_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _reject_tmp_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    s = str(resolved)
    if s == "/tmp" or s.startswith("/tmp/"):
        raise ValueError(f"{label} must never reside under /tmp: {resolved}")
    return resolved


def _dir_size_mb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.exists() and not fpath.is_symlink():
                total += fpath.stat().st_size
    return round(total / (1024.0 * 1024.0), 2)


def _assert_history_empty(ms_path: Path) -> None:
    from casatools import table

    tb = table()
    opened = False
    try:
        tb.open(str(ms_path / "HISTORY"), nomodify=True)
        opened = True
        nrows = int(tb.nrows())
        if nrows != 0:
            raise AssertionError(f"Expected 0 HISTORY rows in {ms_path}, found {nrows}")
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()


def _build_s1_calibrate_checkpoint(
    raw_ms: Path,
    cal_checkpoint_ms: Path,
    masks_npz: Path,
) -> dict[str, Any]:
    """Create post-precal checkpoint: intents fixed, PB2017 set, quack+RFI flagged, ea07 UNFLAGGED."""
    from casatasks import setjy
    from casatools import table

    if cal_checkpoint_ms.exists():
        shutil.rmtree(cal_checkpoint_ms)
    cal_checkpoint_ms.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw_ms, cal_checkpoint_ms)

    assign_scan_intents(cal_checkpoint_ms, fixed_scan_intents())

    with np.load(masks_npz, allow_pickle=False) as masks:
        quack_rows = masks["quack_rows"].astype(bool).copy()
        burst_rows = masks["burst_rows"].astype(bool).copy()
        dead_rows = masks["dead_rows"].astype(bool).copy()
        rfi_channels = masks["rfi_channels"].astype(int).tolist()

    tb = table()
    opened = False
    try:
        tb.open(str(cal_checkpoint_ms), nomodify=False)
        opened = True
        data = tb.getcol("DATA")
        flag = tb.getcol("FLAG")
        fids = tb.getcol("FIELD_ID")
        # Populate dead antenna ea07 with near-zero micro-noise (~1.4e-4 Jy, SNR << 3 in gaincal)
        # and leave unflagged so s1_calibrate tests dead-antenna detection & flagging.
        npol, nchan, _nrow = data.shape
        n_dead = int(np.sum(dead_rows))
        ckpt_rng = np.random.default_rng(SYNTH_SEED + 7)
        data[:, :, dead_rows] = ckpt_rng.normal(
            0.0, 1.0e-4, size=(npol, nchan, n_dead)
        ) + 1j * ckpt_rng.normal(0.0, 1.0e-4, size=(npol, nchan, n_dead))
        tb.putcol("DATA", data)

        flag[:, :, :] = False
        flag[:, :, quack_rows] = True
        flag[:, :, burst_rows] = True
        for ch in rfi_channels:
            flag[:, ch, :] = True
        flag_row = np.all(flag, axis=(0, 1))
        tb.putcol("FLAG", flag)
        tb.putcol("FLAG_ROW", flag_row)

        dead_unflagged_frac = float(np.mean(~flag[:, :, dead_rows]))
        target_live_rows = np.isin(fids, [2, 3]) & (~dead_rows)
        target_clean_rows_unflagged = int(np.sum(~flag_row[target_live_rows]))
        target_live_rows_total = int(np.sum(target_live_rows))
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()

    setjy(
        vis=str(cal_checkpoint_ms),
        field=FLUX_CAL_NAME,
        standard="Perley-Butler 2017",
        usescratch=True,
    )
    tb_post = table()
    opened = False
    try:
        tb_post.open(str(cal_checkpoint_ms), nomodify=False)
        opened = True
        fids_post = tb_post.getcol("FIELD_ID")
        model_post = tb_post.getcol("MODEL_DATA")
        non_f0 = fids_post != 0
        model_post[:, :, non_f0] = 0.0 + 0.0j
        model_post[0, :, non_f0] = 1.0 + 0.0j
        model_post[3, :, non_f0] = 1.0 + 0.0j
        tb_post.putcol("MODEL_DATA", model_post)
        if "CORRECTED_DATA" in tb_post.colnames():
            tb_post.removecols(["CORRECTED_DATA"])
    finally:
        try:
            if opened:
                tb_post.close()
        finally:
            tb_post.done()

    clear_ms_history(cal_checkpoint_ms)
    verify_ms_unlocked(cal_checkpoint_ms)
    return {
        "ea07_unflagged_vis_fraction_in_input": round(dead_unflagged_frac, 6),
        "target_live_rows_total": target_live_rows_total,
        "target_clean_rows_unflagged_in_input": target_clean_rows_unflagged,
        "target_clean_row_fraction_of_live": round(
            target_clean_rows_unflagged / max(target_live_rows_total, 1), 6
        ),
    }


def stage_s1_calibrators_ms(cal_checkpoint_ms: Path, calibrators_ms: Path) -> None:
    """Split calibrator fields from s1_calibrate/input.ms and set PB2017 MODEL_DATA cleanly."""
    from casatasks import setjy, split
    from casatools import table

    if calibrators_ms.exists():
        for root_str, _dirnames, filenames in os.walk(calibrators_ms):
            root_p = Path(root_str)
            os.chmod(root_p, 0o755)
            for fn in filenames:
                os.chmod(root_p / fn, 0o644)
        shutil.rmtree(calibrators_ms)

    calibrators_ms.parent.mkdir(parents=True, exist_ok=True)
    split(
        vis=str(cal_checkpoint_ms),
        outputvis=str(calibrators_ms),
        field=f"{FLUX_CAL_NAME},{PHASE_CAL_NAME}",
        datacolumn="data",
        keepflags=True,
    )
    setjy(
        vis=str(calibrators_ms),
        field=FLUX_CAL_NAME,
        standard="Perley-Butler 2017",
        usescratch=True,
    )

    tb_cal = table()
    opened = False
    try:
        tb_cal.open(str(calibrators_ms), nomodify=False)
        opened = True
        fids_cal = tb_cal.getcol("FIELD_ID")
        model_cal = tb_cal.getcol("MODEL_DATA")
        non_f0 = fids_cal != 0
        model_cal[:, :, non_f0] = 0.0 + 0.0j
        model_cal[0, :, non_f0] = 1.0 + 0.0j
        model_cal[3, :, non_f0] = 1.0 + 0.0j
        tb_cal.putcol("MODEL_DATA", model_cal)
        if "CORRECTED_DATA" in tb_cal.colnames():
            tb_cal.removecols(["CORRECTED_DATA"])
    finally:
        try:
            if opened:
                tb_cal.close()
        finally:
            tb_cal.done()

    clear_ms_history(calibrators_ms)
    verify_ms_unlocked(calibrators_ms)
    for root_str, _dirnames, filenames in os.walk(calibrators_ms):
        root_p = Path(root_str)
        os.chmod(root_p, 0o755)
        for fn in filenames:
            os.chmod(root_p / fn, 0o644)


def _verify_precal_workflow(
    s1_precal_ms: Path,
    masks_npz: Path,
    expected_j1331_amp_jy: float,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Run CASA precal sequence on a scratch copy of s1_precal/input.ms and verify SPEC tolerances."""
    from casatasks import applycal, bandpass, flagdata, gaincal, setjy, split
    from casatools import table

    work_dir = scratch_dir / "verify_precal"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    work_ms = work_dir / "input.ms"
    shutil.copytree(s1_precal_ms, work_ms)

    assign_scan_intents(work_ms, fixed_scan_intents())
    setjy(
        vis=str(work_ms),
        field=FLUX_CAL_NAME,
        standard="Perley-Butler 2017",
        usescratch=True,
    )

    # 1. Quack first integration (5s < 10s integration boundary) at start of each scan + shadow + zero clip
    flagdata(
        vis=str(work_ms),
        mode="quack",
        quackinterval=0.5 * INTEGRATION_SEC,
        quackmode="beg",
        flagbackup=False,
    )
    flagdata(
        vis=str(work_ms),
        mode="shadow",
        tolerance=0.0,
        flagbackup=False,
    )
    # 2. Flag zero-amplitude dead antenna ea07
    flagdata(
        vis=str(work_ms),
        mode="clip",
        clipzeros=True,
        flagbackup=False,
    )
    # 3. Initial phase + delay + bandpass on J1331+3030
    g_init = work_dir / "init.G0"
    k_init = work_dir / "init.K"
    b_init = work_dir / "init.B"
    # Pre-clip extreme >8 Jy raw RFI spikes before initial bandpass so channels 29..31 don't bias K_init
    flagdata(
        vis=str(work_ms),
        mode="clip",
        datacolumn="DATA",
        clipminmax=[0.0, 8.0],
        flagbackup=False,
    )
    gaincal(
        vis=str(work_ms),
        caltable=str(g_init),
        field=FLUX_CAL_NAME,
        spw="0:27~35",
        solint="int",
        refant=REFANT_NAME,
        minsnr=3.0,
        calmode="p",
    )
    gaincal(
        vis=str(work_ms),
        caltable=str(k_init),
        field=FLUX_CAL_NAME,
        solint="inf",
        combine="scan",
        refant=REFANT_NAME,
        minsnr=3.0,
        gaintype="K",
        gaintable=[str(g_init)],
    )
    bandpass(
        vis=str(work_ms),
        caltable=str(b_init),
        field=FLUX_CAL_NAME,
        solint="inf",
        combine="scan",
        refant=REFANT_NAME,
        minsnr=3.0,
        fillgaps=3,
        solnorm=True,
        gaintable=[str(g_init), str(k_init)],
    )
    applycal(
        vis=str(work_ms),
        gaintable=[str(k_init), str(b_init)],
        interp=["nearest", "nearest,linear"],
        applymode="calonly",
        calwt=False,
        flagbackup=False,
    )
    # 4. Residual rflag + grow high-flag rows/channels
    flagdata(
        vis=str(work_ms),
        mode="rflag",
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )
    flagdata(
        vis=str(work_ms),
        mode="extend",
        extendpols=True,
        growtime=80.0,
        growfreq=80.0,
        flagbackup=False,
    )

    cal_ms = work_dir / "calibrators.ms"
    split(
        vis=str(work_ms),
        outputvis=str(cal_ms),
        field=f"{FLUX_CAL_NAME},{PHASE_CAL_NAME}",
        datacolumn="data",
        keepflags=True,
    )
    verify_ms_unlocked(cal_ms)

    with np.load(masks_npz, allow_pickle=False) as masks:
        corrupted_chan_row = masks["corrupted_chan_row"].astype(bool).copy()

    tb = table()
    opened = False
    try:
        tb.open(str(work_ms), nomodify=True)
        opened = True
        flag = tb.getcol("FLAG")
        flag_row = tb.getcol("FLAG_ROW")
        fids = tb.getcol("FIELD_ID")
        model = tb.getcol("MODEL_DATA")
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()

    flagged_chan_row = np.any(flag, axis=0) | flag_row[np.newaxis, :]
    recall = float(np.mean(flagged_chan_row[corrupted_chan_row]))
    clean_fpr = float(np.mean(flagged_chan_row[~corrupted_chan_row]))
    j1331_model_amp = float(
        np.mean(0.5 * (np.abs(model[0, :, fids == 0]) + np.abs(model[3, :, fids == 0])))
    )
    j1331_rel_err = abs(j1331_model_amp - expected_j1331_amp_jy) / expected_j1331_amp_jy

    if recall < 0.95:
        raise AssertionError(f"Precal verification recall {recall:.4f} < 0.95")
    if clean_fpr >= 0.05:
        raise AssertionError(f"Precal verification clean FPR {clean_fpr:.4f} >= 0.05")
    if j1331_rel_err >= 0.01:
        raise AssertionError(f"Precal J1331 MODEL_DATA error {j1331_rel_err:.4%} >= 1.0%")

    # Also verify radio-analyst's deterministic preflag list-mode (quack 5s + shadow + clipzeros + tfcrop + extend)
    ra_ms = work_dir / "ra_preflag.ms"
    shutil.copytree(s1_precal_ms, ra_ms)
    ra_cmds = work_dir / "preflag_cmds.txt"
    ra_cmds.write_text(
        "mode='quack' quackinterval=5.0 quackmode='beg'\n"
        "mode='shadow' tolerance=0.0\n"
        "mode='clip' clipzeros=True\n"
        "mode='tfcrop' timecutoff=3.0 freqcutoff=3.0\n"
        "mode='extend' extendpols=True\n",
        encoding="utf-8",
    )
    flagdata(vis=str(ra_ms), mode="list", inpfile=str(ra_cmds), flagbackup=False)
    tb_ra = table()
    opened = False
    try:
        tb_ra.open(str(ra_ms), nomodify=True)
        opened = True
        ra_flag = tb_ra.getcol("FLAG")
        ra_flag_row = tb_ra.getcol("FLAG_ROW")
    finally:
        try:
            if opened:
                tb_ra.close()
        finally:
            tb_ra.done()
    ra_flag_any = np.any(ra_flag, axis=0) | ra_flag_row[np.newaxis, :]
    ra_recall = float(np.mean(ra_flag_any[corrupted_chan_row]))
    ra_fpr = float(np.mean(ra_flag_any[~corrupted_chan_row]))
    if ra_recall < 0.95:
        raise AssertionError(f"Preflag list-mode recall {ra_recall:.4f} < 0.95")
    if ra_fpr >= 0.05:
        raise AssertionError(f"Preflag list-mode clean FPR {ra_fpr:.4f} >= 0.05")

    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        "corrupted_recall": round(recall, 6),
        "clean_false_positive_rate": round(clean_fpr, 6),
        "preflag_list_recall": round(ra_recall, 6),
        "preflag_list_false_positive_rate": round(ra_fpr, 6),
        "j1331_model_mean_amp_jy": round(j1331_model_amp, 6),
        "j1331_model_rel_err": round(j1331_rel_err, 6),
    }


def _run_and_verify_calibration(
    s1_cal_input_ms: Path,
    s1_image_input_ms: Path,
    target_vis_npz: Path,
    pcal_true_flux_jy: float,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Run G0 -> K -> B -> G -> fluxscale -> applycal, verify SPEC metrics, and stage s1_image/input.ms."""
    from casatasks import applycal, bandpass, flagdata, fluxscale, gaincal
    from casatools import table

    work_dir = scratch_dir / "verify_cal"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    cal_ms = work_dir / "cal.ms"
    shutil.copytree(s1_cal_input_ms, cal_ms)

    # 1. Flag dead antenna ea07
    flagdata(
        vis=str(cal_ms),
        mode="manual",
        antenna=DEAD_ANTENNA_NAME,
        flagbackup=False,
    )

    g0_tab = work_dir / "cal.G0"
    k_tab = work_dir / "cal.K"
    b_tab = work_dir / "cal.B"
    g_tab = work_dir / "cal.G"
    flux_tab = work_dir / "cal.fluxscale"

    # 2. G0: initial phase-only solve on bandpass calibrator central channels
    gaincal(
        vis=str(cal_ms),
        caltable=str(g0_tab),
        field=FLUX_CAL_NAME,
        spw="0:27~35",
        solint="int",
        refant=REFANT_NAME,
        minsnr=3.0,
        calmode="p",
    )
    # 3. K: antenna-based delay solve
    gaincal(
        vis=str(cal_ms),
        caltable=str(k_tab),
        field=FLUX_CAL_NAME,
        solint="inf",
        combine="scan",
        refant=REFANT_NAME,
        minsnr=3.0,
        gaintype="K",
        gaintable=[str(g0_tab)],
    )
    # 4. B: normalized complex bandpass solve
    bandpass(
        vis=str(cal_ms),
        caltable=str(b_tab),
        field=FLUX_CAL_NAME,
        solint="inf",
        combine="scan",
        refant=REFANT_NAME,
        minsnr=3.0,
        bandtype="B",
        fillgaps=3,
        solnorm=True,
        gaintable=[str(g0_tab), str(k_tab)],
    )
    # 5. G: scan-averaged complex amplitude+phase gains on flux + phase calibrators
    gaincal(
        vis=str(cal_ms),
        caltable=str(g_tab),
        field=f"{FLUX_CAL_NAME},{PHASE_CAL_NAME}",
        solint="inf",
        refant=REFANT_NAME,
        minsnr=3.0,
        calmode="ap",
        gaintable=[str(k_tab), str(b_tab)],
    )
    # 6. fluxscale: bootstrap secondary calibrator flux scale from J1331+3030
    flux_dict = fluxscale(
        vis=str(cal_ms),
        caltable=str(g_tab),
        fluxtable=str(flux_tab),
        reference=[FLUX_CAL_NAME],
        transfer=[PHASE_CAL_NAME],
        incremental=False,
    )
    # 7. applycal to flux calibrator, phase calibrator, and mosaic target fields
    applycal(
        vis=str(cal_ms),
        field=FLUX_CAL_NAME,
        gaintable=[str(k_tab), str(b_tab), str(flux_tab)],
        gainfield=["", "", FLUX_CAL_NAME],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )
    applycal(
        vis=str(cal_ms),
        field=f"{PHASE_CAL_NAME},{TARGET_P1_NAME},{TARGET_P2_NAME}",
        gaintable=[str(k_tab), str(b_tab), str(flux_tab)],
        gainfield=["", "", PHASE_CAL_NAME],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )

    # Extract fitted K delays for diagnostic verification
    tb_k = table()
    opened = False
    try:
        tb_k.open(str(k_tab), nomodify=True)
        opened = True
        k_fparam = tb_k.getcol("FPARAM")  # (2, 1, nant) in ns
        k_ant = tb_k.getcol("ANTENNA1")
    finally:
        try:
            if opened:
                tb_k.close()
        finally:
            tb_k.done()
    solved_delays_ns: dict[str, float] = {}
    for idx, ant_id in enumerate(k_ant):
        ant_name = SYNTH_ANTENNA_NAMES[int(ant_id)]
        if ant_name != DEAD_ANTENNA_NAME:
            solved_delays_ns[ant_name] = round(float(np.mean(k_fparam[:, 0, idx])), 4)

    tb_spw = table()
    opened = False
    try:
        tb_spw.open(str(cal_ms / "SPECTRAL_WINDOW"), nomodify=True)
        opened = True
        freqs_hz = tb_spw.getcol("CHAN_FREQ")[:, 0]
    finally:
        try:
            if opened:
                tb_spw.close()
        finally:
            tb_spw.done()

    tb = table()
    opened = False
    try:
        tb.open(str(cal_ms), nomodify=False)
        opened = True
        colnames = tb.colnames()
        if "CORRECTED_DATA" not in colnames:
            raise AssertionError("CORRECTED_DATA missing after applycal")
        corr = tb.getcol("CORRECTED_DATA")
        raw_flag = tb.getcol("FLAG")
        flag_row = tb.getcol("FLAG_ROW")
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        fids = tb.getcol("FIELD_ID")

        # Ensure target MODEL_DATA is clean unity before copying to s1_image/input.ms
        model = tb.getcol("MODEL_DATA")
        tgt_mask_all = np.isin(fids, [2, 3])
        model[:, :, tgt_mask_all] = 0.0 + 0.0j
        model[0, :, tgt_mask_all] = 1.0 + 0.0j
        model[3, :, tgt_mask_all] = 1.0 + 0.0j
        tb.putcol("MODEL_DATA", model)
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()

    flag = raw_flag | flag_row[np.newaxis, np.newaxis, :]

    # Metric 1: dead antenna ea07 flagged + clean target rows retained >= 85%
    dead_rows = (ant1 == DEAD_ANTENNA_ID) | (ant2 == DEAD_ANTENNA_ID)
    ea07_flagged_frac = float(np.mean(flag[:, :, dead_rows]))
    live_tgt_rows = np.isin(fids, [2, 3]) & (~dead_rows) & (ant1 != ant2)
    row_has_unflagged_vis = ~np.all(flag[:, :, live_tgt_rows], axis=(0, 1))
    clean_tgt_row_retention_vs_live = float(np.mean(row_has_unflagged_vis))

    with np.load(target_vis_npz, allow_pickle=False) as tvis:
        clean_tgt_mask_sub = tvis["clean_target_row_mask"].astype(bool).copy()
        tgt_row_idx = tvis["target_row_indices"].astype(int).copy()
        clean_chan_mask = tvis["clean_chan_mask"].astype(bool).copy()
        true_rr = tvis["true_rr"].astype(np.complex128).copy()
        true_ll = tvis["true_ll"].astype(np.complex128).copy()

    clean_tgt_global_rows = tgt_row_idx[clean_tgt_mask_sub]
    clean_tgt_retained_frac = float(
        np.mean(~np.all(flag[:, clean_chan_mask, :][:, :, clean_tgt_global_rows], axis=(0, 1)))
    )

    # Metric 2: J1331+3030 CORRECTED_DATA cross-channel phase slope (< 5.0 deg / 100 MHz) & phase RMS
    j1331_rows = (fids == 0) & (~dead_rows) & (ant1 != ant2)
    j1331_unflagged_ch: list[int] = []
    j1331_vec_spec: list[complex] = []
    for ch in range(len(freqs_hz)):
        m_rr = ~flag[0, ch, j1331_rows]
        m_ll = ~flag[3, ch, j1331_rows]
        vals = []
        if np.any(m_rr):
            vals.append(np.mean(corr[0, ch, j1331_rows][m_rr]))
        if np.any(m_ll):
            vals.append(np.mean(corr[3, ch, j1331_rows][m_ll]))
        if vals:
            j1331_unflagged_ch.append(ch)
            j1331_vec_spec.append(complex(np.mean(vals)))

    ch_freqs = freqs_hz[j1331_unflagged_ch]
    phs_rad = np.unwrap(np.angle(np.array(j1331_vec_spec)))
    phs_deg = np.rad2deg(phs_rad)
    slope_deg_per_hz, _intercept_deg = np.polyfit(ch_freqs - np.mean(ch_freqs), phs_deg, 1)
    slope_deg_per_100mhz = abs(float(slope_deg_per_hz) * 1.0e8)
    j1331_chan_phase_rms_deg = float(np.sqrt(np.mean(phs_deg**2)))

    # Metric 3: Secondary phase calibrator fluxscale accuracy within +-5% of S_true
    fluxscale_reported_jy = float(flux_dict["1"]["0"]["fluxd"][0])
    pcal_rows = (fids == 1) & (~dead_rows) & (ant1 != ant2)
    pcal_rr = corr[0, :, :][:, pcal_rows][~flag[0, :, :][:, pcal_rows]]
    pcal_ll = corr[3, :, :][:, pcal_rows][~flag[3, :, :][:, pcal_rows]]
    pcal_corr_amp_jy = float(0.5 * (np.abs(np.mean(pcal_rr)) + np.abs(np.mean(pcal_ll))))
    fluxscale_rel_err = abs(fluxscale_reported_jy - pcal_true_flux_jy) / pcal_true_flux_jy
    pcal_corr_rel_err = abs(pcal_corr_amp_jy - pcal_true_flux_jy) / pcal_true_flux_jy

    # Metric 4: Coherent target visibility recovery within +-8% amp and +-10 deg phase
    corr_tgt_rr = corr[0, :, :][:, tgt_row_idx]
    corr_tgt_ll = corr[3, :, :][:, tgt_row_idx]
    flag_tgt_rr = flag[0, :, :][:, tgt_row_idx]
    flag_tgt_ll = flag[3, :, :][:, tgt_row_idx]

    valid_rr = (~flag_tgt_rr) & clean_chan_mask[:, None] & clean_tgt_mask_sub[None, :]
    valid_ll = (~flag_tgt_ll) & clean_chan_mask[:, None] & clean_tgt_mask_sub[None, :]

    num = np.sum(corr_tgt_rr[valid_rr] * np.conj(true_rr[valid_rr])) + np.sum(
        corr_tgt_ll[valid_ll] * np.conj(true_ll[valid_ll])
    )
    den = np.sum(np.abs(true_rr[valid_rr]) ** 2) + np.sum(np.abs(true_ll[valid_ll]) ** 2)
    gamma = complex(num / den)
    coherent_amp_ratio = float(abs(gamma))
    coherent_phase_deg = float(math.degrees(math.atan2(gamma.imag, gamma.real)))

    if ea07_flagged_frac < 0.999:
        raise AssertionError(f"ea07 not fully flagged: {ea07_flagged_frac:.4f}")
    if clean_tgt_row_retention_vs_live < 0.85:
        raise AssertionError(
            f"Clean target row retention {clean_tgt_row_retention_vs_live:.4f} < 0.85"
        )
    if slope_deg_per_100mhz >= 5.0:
        raise AssertionError(f"J1331 phase slope {slope_deg_per_100mhz:.4f} deg/100MHz >= 5.0")
    if fluxscale_rel_err >= 0.05 or pcal_corr_rel_err >= 0.05:
        raise AssertionError(
            f"Fluxscale error {fluxscale_rel_err:.4%} / corr {pcal_corr_rel_err:.4%} >= 5.0%"
        )
    if abs(coherent_amp_ratio - 1.0) >= 0.08:
        raise AssertionError(
            f"Target coherent amp ratio {coherent_amp_ratio:.4f} outside [0.92, 1.08]"
        )
    if abs(coherent_phase_deg) >= 10.0:
        raise AssertionError(
            f"Target coherent phase offset {coherent_phase_deg:.4f} deg outside [-10, +10]"
        )

    # Stage clean post-calibrate checkpoint as s1_image/input.ms with empty HISTORY
    if s1_image_input_ms.exists():
        shutil.rmtree(s1_image_input_ms)
    s1_image_input_ms.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cal_ms, s1_image_input_ms)
    clear_ms_history(s1_image_input_ms)
    verify_ms_unlocked(s1_image_input_ms)

    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        "ea07_flagged_fraction": round(ea07_flagged_frac, 6),
        "clean_target_row_retention_of_live": round(clean_tgt_row_retention_vs_live, 6),
        "clean_target_row_retention_of_uncorrupted": round(clean_tgt_retained_frac, 6),
        "j1331_phase_slope_deg_per_100mhz": round(slope_deg_per_100mhz, 4),
        "j1331_chan_phase_rms_deg": round(j1331_chan_phase_rms_deg, 4),
        "fluxscale_pcal_flux_jy": round(fluxscale_reported_jy, 6),
        "pcal_corrected_amp_jy": round(pcal_corr_amp_jy, 6),
        "fluxscale_rel_err": round(fluxscale_rel_err, 6),
        "target_coherent_amp_ratio": round(coherent_amp_ratio, 6),
        "target_coherent_phase_deg": round(coherent_phase_deg, 4),
        "solved_delays_ns": solved_delays_ns,
    }


def _next_casa_fft_size(n: int) -> int:
    """Return smallest even 5-smooth integer >= n (only factors 2, 3, 5 for CASA FFT)."""
    cand = max(n + (n % 2), 64)
    while True:
        rem = cand
        for p in (2, 3, 5):
            while rem % p == 0:
                rem //= p
        if rem == 1:
            return cand
        cand += 2


def _derive_imaging_geometry(ms_path: Path) -> dict[str, Any]:
    """Derive cell_arcsec, imsize, and multiscale scales from unflagged target UVW."""
    from casatools import table

    tb_spw = table()
    opened = False
    try:
        tb_spw.open(str(ms_path / "SPECTRAL_WINDOW"), nomodify=True)
        opened = True
        freqs_hz = tb_spw.getcol("CHAN_FREQ")[:, 0]
    finally:
        try:
            if opened:
                tb_spw.close()
        finally:
            tb_spw.done()
    nu_max_hz = float(np.max(freqs_hz))
    c_m_s = 299792458.0

    tb = table()
    opened = False
    try:
        tb.open(str(ms_path), nomodify=True)
        opened = True
        uvw = tb.getcol("UVW")
        flag_row = tb.getcol("FLAG_ROW")
        flag = tb.getcol("FLAG")
        fids = tb.getcol("FIELD_ID")
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()

    tgt_unflagged = np.isin(fids, [2, 3]) & (~flag_row) & (~np.all(flag, axis=(0, 1)))
    uv_dist_m = np.hypot(uvw[0, tgt_unflagged], uvw[1, tgt_unflagged])
    b_max_m = float(np.max(uv_dist_m))
    b_min_m = float(np.min(uv_dist_m[uv_dist_m > 0.0]))
    theta_res_arcsec = ((c_m_s / nu_max_hz) / b_max_m) * 206264.806247
    cell_arcsec = round(theta_res_arcsec / 4.3, 1)
    # Cover ~28 arcmin so .pb drops to 0.0 well inside image edges (zero PB edge aliasing)
    raw_imsize = int(math.ceil(1700.0 / cell_arcsec))
    imsize = _next_casa_fft_size(raw_imsize)
    scales = [0, 3, 8]
    return {
        "b_max_m": round(b_max_m, 2),
        "b_min_m": round(b_min_m, 2),
        "theta_res_arcsec": round(theta_res_arcsec, 3),
        "recommended_cell_arcsec": cell_arcsec,
        "recommended_imsize": imsize,
        "recommended_scales": scales,
    }


def _hms_dms_to_rad(ra_hms: str, dec_dms: str) -> tuple[float, float]:
    from astropy.coordinates import SkyCoord

    sc = SkyCoord(ra_hms, dec_dms, frame="icrs")
    return float(sc.ra.rad), float(sc.dec.rad)


def _measure_imaging_products(
    imagename: str,
    *,
    cell_arcsec: float,
    sigma_th_jy: float,
    pt_flux_jy: float,
    label: str,
) -> dict[str, Any]:
    """Read tclean products (.image, .image.pbcor, .pb, .residual, .model) and verify margins."""
    from casatools import image, quanta

    pbcor_path = Path(f"{imagename}.image.pbcor")
    if not pbcor_path.is_dir():
        raise AssertionError(f"Expected {pbcor_path} to exist for {label}")

    def _read_2d(path_str: str) -> tuple[np.ndarray, np.ndarray]:
        ia = image()
        ia_opened = False
        try:
            ia.open(path_str)
            ia_opened = True
            arr = ia.getchunk()[:, :, 0, 0].copy()
            msk = ia.getchunk(getmask=True)[:, :, 0, 0].copy()
            return arr, msk
        finally:
            try:
                if ia_opened:
                    ia.close()
            finally:
                ia.done()

    qa = quanta()
    try:
        ia_img = image()
        ia_opened = False
        try:
            ia_img.open(f"{imagename}.image")
            ia_opened = True
            img_arr = ia_img.getchunk()[:, :, 0, 0].copy()
            img_mask = ia_img.getchunk(getmask=True)[:, :, 0, 0].copy()
            rb = ia_img.restoringbeam()
            if "beams" in rb:
                rb = rb["beams"]["*0"]["*0"]
            b_major_arcsec = float(qa.convert(rb["major"], "arcsec")["value"])
            b_minor_arcsec = float(qa.convert(rb["minor"], "arcsec")["value"])
            b_pa_deg = float(qa.convert(rb["positionangle"], "deg")["value"])
            cs = ia_img.coordsys()
            try:
                refpix = cs.referencepixel()["numeric"]
                refval = cs.referencevalue()["numeric"]
                inc = cs.increment()["numeric"]
            finally:
                cs.done()
        finally:
            try:
                if ia_opened:
                    ia_img.close()
            finally:
                ia_img.done()

        pb_arr, pb_mask = _read_2d(f"{imagename}.pb")
        pbcor_arr, pbcor_mask = _read_2d(f"{imagename}.image.pbcor")
        res_arr, res_mask = _read_2d(f"{imagename}.residual")
        mod_arr, _mod_mask = _read_2d(f"{imagename}.model")
    finally:
        qa.done()

    for suffix in (".image", ".image.pbcor", ".pb", ".residual", ".model"):
        verify_ms_unlocked(Path(f"{imagename}{suffix}"))

    # Build sky-offset coordinate grids in arcsec from exact SIN projection parameters
    ra0, dec0 = float(refval[0]), float(refval[1])
    nx, ny = img_arr.shape
    ix, iy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")

    def _sky_to_pix(ra_hms: str, dec_dms: str) -> tuple[float, float]:
        ra_r, dec_r = _hms_dms_to_rad(ra_hms, dec_dms)
        l_rad = math.cos(dec_r) * math.sin(ra_r - ra0)
        m_rad = math.sin(dec_r) * math.cos(dec0) - math.cos(dec_r) * math.sin(dec0) * math.cos(
            ra_r - ra0
        )
        return float(refpix[0] + l_rad / inc[0]), float(refpix[1] + m_rad / inc[1])

    ext_px, ext_py = _sky_to_pix(EXT_COMP_RA, EXT_COMP_DEC)
    pt_px, pt_py = _sky_to_pix(PT_COMP_RA, PT_COMP_DEC)

    dist_ext_arcsec = np.hypot((ix - ext_px) * cell_arcsec, (iy - ext_py) * cell_arcsec)
    dist_pt_arcsec = np.hypot((ix - pt_px) * cell_arcsec, (iy - pt_py) * cell_arcsec)

    valid_pb20 = (
        img_mask & pb_mask & pbcor_mask & res_mask & np.isfinite(pbcor_arr) & (pb_arr >= 0.2)
    )
    offsource_mask = valid_pb20 & (dist_ext_arcsec > 75.0) & (dist_pt_arcsec > 35.0)

    offsource_rms_jy = float(np.sqrt(np.mean(res_arr[offsource_mask] ** 2)))
    rms_over_sigma_th = offsource_rms_jy / sigma_th_jy
    pixels_per_beam = b_minor_arcsec / cell_arcsec

    # Extended Gaussian integrated flux in 50 arcsec radius aperture on .image.pbcor
    beam_area_pix = (math.pi * b_major_arcsec * b_minor_arcsec) / (
        4.0 * math.log(2.0) * (cell_arcsec**2)
    )
    ext_aperture = valid_pb20 & (dist_ext_arcsec <= 50.0)
    ext_integrated_flux_jy = float(np.sum(pbcor_arr[ext_aperture]) / beam_area_pix)
    ext_flux_rel_err = abs(ext_integrated_flux_jy - EXT_COMP_FLUX_JY) / EXT_COMP_FLUX_JY

    # 10-sigma point source peak & SNR
    pt_box = valid_pb20 & (dist_pt_arcsec <= 20.0)
    pt_peak_image_jy = float(np.max(img_arr[pt_box]))
    pt_peak_pbcor_jy = float(np.max(pbcor_arr[pt_box]))
    pt_snr_image = pt_peak_image_jy / offsource_rms_jy
    pt_pb_val = float(pb_arr[int(round(pt_px)), int(round(pt_py))])
    pt_snr_pbcor = pt_peak_pbcor_jy / (offsource_rms_jy / max(pt_pb_val, 0.2))

    # Clean-bowl guard: |sum(model < 0)| / sum(model > 0) < 0.12 (require <= 0.06 2x margin here)
    pos_model_sum = float(np.sum(mod_arr[mod_arr > 0.0]))
    neg_model_abs_sum = float(np.abs(np.sum(mod_arr[mod_arr < 0.0])))
    bowl_ratio = neg_model_abs_sum / max(pos_model_sum, 1e-12)

    if not (3.0 <= pixels_per_beam <= 7.0):
        raise AssertionError(f"[{label}] pixels_per_beam {pixels_per_beam:.3f} outside [3.0, 7.0]")
    if rms_over_sigma_th >= 1.5:
        raise AssertionError(
            f"[{label}] Off-source residual RMS {offsource_rms_jy * 1e6:.2f} uJy/beam "
            f"= {rms_over_sigma_th:.3f} * sigma_th >= 1.5 * sigma_th"
        )
    if ext_flux_rel_err >= 0.10:
        raise AssertionError(
            f"[{label}] Extended Gaussian integrated flux {ext_integrated_flux_jy:.4f} Jy "
            f"rel_err {ext_flux_rel_err:.2%} >= 10% (2x margin inside 15% SPEC gate)"
        )
    if pt_snr_image < 7.5 or pt_snr_pbcor < 7.5:
        raise AssertionError(
            f"[{label}] Point source SNR {pt_snr_image:.2f} (img) / {pt_snr_pbcor:.2f} (pbcor) < 7.5"
        )
    if bowl_ratio > 0.06:
        raise AssertionError(
            f"[{label}] Negative model bowl ratio {bowl_ratio:.4f} > 0.06 (2x margin inside 0.12)"
        )

    return {
        "b_major_arcsec": round(b_major_arcsec, 4),
        "b_minor_arcsec": round(b_minor_arcsec, 4),
        "b_pa_deg": round(b_pa_deg, 3),
        "pixels_per_beam": round(pixels_per_beam, 4),
        "offsource_residual_rms_jy_per_beam": round(offsource_rms_jy, 8),
        "offsource_rms_over_sigma_th": round(rms_over_sigma_th, 4),
        "ext_aperture_radius_arcsec": 50.0,
        "ext_integrated_flux_jy": round(ext_integrated_flux_jy, 6),
        "ext_flux_rel_err": round(ext_flux_rel_err, 6),
        "pt_true_flux_jy": round(pt_flux_jy, 7),
        "pt_peak_image_jy_per_beam": round(pt_peak_image_jy, 7),
        "pt_peak_pbcor_jy_per_beam": round(pt_peak_pbcor_jy, 7),
        "pt_pb_response": round(pt_pb_val, 4),
        "pt_snr_image": round(pt_snr_image, 3),
        "pt_snr_pbcor": round(pt_snr_pbcor, 3),
        "model_positive_sum_jy": round(pos_model_sum, 6),
        "model_negative_abs_sum_jy": round(neg_model_abs_sum, 6),
        "negative_bowl_ratio": round(bowl_ratio, 6),
    }


def _run_and_verify_imaging(
    s1_image_input_ms: Path,
    sigma_th_jy: float,
    pt_flux_jy: float,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Run both reference multiscale tclean and default ms_tclean-equivalent on s1_image/input.ms."""
    from casatasks import tclean

    geom = _derive_imaging_geometry(s1_image_input_ms)
    cell_arcsec = float(geom["recommended_cell_arcsec"])
    imsize = int(geom["recommended_imsize"])
    scales = list(geom["recommended_scales"])

    work_dir = scratch_dir / "verify_img"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    clean_thresh_mjy = 3.0 * sigma_th_jy * 1.0e3

    # 1. Reference tclean (smallscalebias=0.6, threshold=3.0*sigma_th, pbcor=True -> .image.pbcor)
    ref_ms = work_dir / "img_ref.ms"
    shutil.copytree(s1_image_input_ms, ref_ms)
    ref_imagename = str(work_dir / "synth_mosaic_ref")
    tclean(
        vis=str(ref_ms),
        imagename=ref_imagename,
        field=f"{TARGET_P1_NAME},{TARGET_P2_NAME}",
        spw="0",
        datacolumn="corrected",
        specmode="mfs",
        deconvolver="multiscale",
        scales=scales,
        smallscalebias=0.6,
        imsize=[imsize, imsize],
        cell=f"{cell_arcsec:.2f}arcsec",
        phasecenter=MOSAIC_PHASECENTER,
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
        threshold=f"{clean_thresh_mjy:.4f}mJy",
        pbcor=True,
        interactive=False,
    )
    ref_metrics = _measure_imaging_products(
        ref_imagename,
        cell_arcsec=cell_arcsec,
        sigma_th_jy=sigma_th_jy,
        pt_flux_jy=pt_flux_jy,
        label="reference_tclean",
    )

    # 2. Default ms_tclean-equivalent (smallscalebias=0.0, scales=[0, 3, 8], threshold=3.0*sigma_th,
    #    default phasecenter="", savemodel="modelcolumn", pbcor=True -> .image.pbcor)
    mst_ms = work_dir / "img_ms_tclean.ms"
    shutil.copytree(s1_image_input_ms, mst_ms)
    mst_imagename = str(work_dir / "synth_mosaic_ms_tclean")
    tclean(
        vis=str(mst_ms),
        imagename=mst_imagename,
        field=f"{TARGET_P1_NAME},{TARGET_P2_NAME}",
        stokes="I",
        spw="0",
        specmode="mfs",
        deconvolver="multiscale",
        scales=[0, 3, 8],
        smallscalebias=0.0,
        gridder="mosaic",
        cell=[f"{cell_arcsec:.2f}arcsec"],
        imsize=[imsize, imsize],
        weighting="briggs",
        robust=0.5,
        niter=50000,
        threshold=f"{clean_thresh_mjy:.4f}mJy",
        pbcor=True,
        pblimit=0.2,
        usemask="pb",
        pbmask=0.2,
        savemodel="modelcolumn",
        interactive=False,
    )
    ms_tclean_metrics = _measure_imaging_products(
        mst_imagename,
        cell_arcsec=cell_arcsec,
        sigma_th_jy=sigma_th_jy,
        pt_flux_jy=pt_flux_jy,
        label="ms_tclean_default",
    )

    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        **geom,
        "clean_threshold_mjy": round(clean_thresh_mjy, 6),
        "pbcor_product_suffix": ".image.pbcor",
        **ref_metrics,
        "ms_tclean_default_verification": ms_tclean_metrics,
    }


def _common_ground_truth(
    task_id: str,
    *,
    base_info: dict[str, Any],
    corrupt_info: dict[str, Any],
    sigma_info: dict[str, Any],
    pt_flux_jy: float,
) -> dict[str, Any]:
    sigma_th = float(sigma_info["sigma_th_jy_per_beam"])
    sigma_nat = float(sigma_info["sigma_th_natural_jy_per_beam"])
    return {
        "task_id": task_id,
        "suite": "synth",
        "seed": SYNTH_SEED,
        "pass_reward_min": 0.85,
        "target_retention_excludes_dead_antenna": True,
        "target_retention_excludes_autocorr": True,
        "telescope": "VLA",
        "configuration": "D",
        "ref_epoch_utc": REF_EPOCH_UTC,
        "obs_duration_s": base_info["obs_duration_s"],
        "integration_sec": INTEGRATION_SEC,
        "n_antennas": len(SYNTH_ANTENNA_NAMES),
        "antenna_names": list(SYNTH_ANTENNA_NAMES),
        "refant": REFANT_NAME,
        "dead_antenna": DEAD_ANTENNA_NAME,
        "dead_antenna_id": DEAD_ANTENNA_ID,
        "dead_antennas": [DEAD_ANTENNA_NAME],
        "spw": {
            "spw_id": 0,
            "name": "Cband_spw0",
            "nchan": SPW_NCHAN,
            "start_hz": SPW_START_HZ,
            "chan_width_hz": SPW_CHAN_WIDTH_HZ,
            "center_hz": SPW_START_HZ + 0.5 * (SPW_NCHAN - 1) * SPW_CHAN_WIDTH_HZ,
            "total_bw_hz": SPW_NCHAN * SPW_CHAN_WIDTH_HZ,
            "correlations": ["RR", "RL", "LR", "LL"],
        },
        "fields": {
            "flux_calibrator": FLUX_CAL_NAME,
            "flux_calibrator_id": 0,
            "bandpass_calibrator": FLUX_CAL_NAME,
            "bandpass_calibrator_id": 0,
            "phase_calibrator": PHASE_CAL_NAME,
            "phase_calibrator_id": 1,
            "phase_calibrator_missing_intent_in_raw": True,
            "target_fields_missing_intent_in_raw": True,
            "raw_populated_intent_fields": [FLUX_CAL_NAME],
            "raw_unassigned_intent_fields": [PHASE_CAL_NAME, TARGET_P1_NAME, TARGET_P2_NAME],
            "vla_callist_matches": {
                FLUX_CAL_NAME: "1331+305",
                PHASE_CAL_NAME: "1822-096",
            },
            "phase_calibrator_scans": [3, 6, 9, 12],
            "target_fields": [TARGET_P1_NAME, TARGET_P2_NAME],
            "target_field_ids": [2, 3],
            "target_scans_by_field": {
                TARGET_P1_NAME: [4, 7, 10],
                TARGET_P2_NAME: [5, 8, 11],
            },
            "is_mosaic": True,
            "n_mosaic_pointings": 2,
            "mosaic_phasecenter": MOSAIC_PHASECENTER,
            "coordinates_j2000": {
                FLUX_CAL_NAME: {"ra": FLUX_CAL_RA, "dec": FLUX_CAL_DEC},
                PHASE_CAL_NAME: {"ra": PHASE_CAL_RA, "dec": PHASE_CAL_DEC},
                TARGET_P1_NAME: {"ra": TARGET_P1_RA, "dec": TARGET_P1_DEC},
                TARGET_P2_NAME: {"ra": TARGET_P2_RA, "dec": TARGET_P2_DEC},
            },
        },
        "bp_min_elevation_deg": base_info["bp_min_elevation_deg"],
        "bp_min_elevation_casa_azelgeo_deg": base_info["bp_min_elevation_casa_azelgeo_deg"],
        "bp_min_elevation_tolerance_deg": 0.25,
        "j1331_pb2017": {
            "standard": "Perley-Butler 2017",
            "mean_parallel_amp_jy": base_info["j1331_pb2017_mean_amp_jy"],
            "per_chan_parallel_amp_jy": base_info["j1331_pb2017_per_chan_amp_jy"],
            "rel_tolerance": 0.01,
        },
        "phase_calibrator_truth": {
            "field_name": PHASE_CAL_NAME,
            "field_id": 1,
            "ref_freq_hz": 4.6e9,
            "flux_at_ref_freq_jy": PHASE_CAL_FLUX_JY,
            "spectral_index": PHASE_CAL_SPIX,
            "spw_mean_flux_jy": base_info["pcal_spw_mean_amp_jy"],
            "fluxscale_rel_tolerance": 0.05,
        },
        "true_delays_ns": dict(TRUE_DELAYS_NS),
        "noise": {
            "sigma_vis_jy": SIGMA_VIS_JY,
            "sigma_th_jy_per_beam": round(sigma_th, 8),
            "sigma_th_natural_jy_per_beam": round(sigma_nat, 8),
            "sigma_th_single_pointing_jy_per_beam": round(
                float(sigma_info["sigma_th_single_pointing_jy_per_beam"]), 8
            ),
            "residual_rms_max_sigma_th_mult": 1.5,
        },
        "sky_model": {
            "extended_gaussian": {
                "ra": EXT_COMP_RA,
                "dec": EXT_COMP_DEC,
                "flux_jy": EXT_COMP_FLUX_JY,
                "ref_freq_hz": 4.6e9,
                "spectral_index": 0.0,
                "major_arcsec": float(EXT_COMP_MAJOR.replace("arcsec", "")),
                "minor_arcsec": float(EXT_COMP_MINOR.replace("arcsec", "")),
                "pa_deg": float(EXT_COMP_PA.replace("deg", "")),
                "aperture_radius_arcsec": 50.0,
            },
            "point_source": {
                "ra": PT_COMP_RA,
                "dec": PT_COMP_DEC,
                "flux_jy": round(pt_flux_jy, 8),
                "sigma_th_multiple": PT_SOURCE_SIGMA_MULT,
                "min_snr_threshold": 5.0,
            },
            "total_clean_flux_jy": round(EXT_COMP_FLUX_JY + pt_flux_jy, 7),
            "true_target_vis_mean_amp_jy": corrupt_info["true_target_vis_mean_amp_jy"],
            "negative_bowl_max_ratio": 0.12,
        },
        "corruptions_summary": {
            "n_rows": corrupt_info["n_rows"],
            "n_chan": corrupt_info["n_chan"],
            "n_corrupted_chan_rows": corrupt_info["n_corrupted_chan_rows"],
            "n_clean_chan_rows": corrupt_info["n_clean_chan_rows"],
            "rfi_channels": corrupt_info["rfi_channels"],
            "scan1_quack_scans": [1],
            "slew_quack_first_integration_all_scans": True,
            "burst_scan": 5,
            "n_quack_rows": len(corrupt_info["quack_row_indices"]),
            "n_burst_rows": len(corrupt_info["burst_row_indices"]),
            "n_dead_rows": len(corrupt_info["dead_row_indices"]),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    content = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def build_synthetic_suite(
    out_dir: Path | None = None,
    scratch_root: Path | None = None,
    *,
    cfg: BenchConfig | None = None,
) -> dict[str, Any]:
    """Deterministically build and CASA-verify all 5 Suite S datasets."""
    t_start = time.perf_counter()
    resolved_cfg = configure_builder_casa_env(cfg)
    raw_out_dir = out_dir if out_dir is not None else (resolved_cfg.data_dir / "synth")
    raw_scratch_root = scratch_root if scratch_root is not None else resolved_cfg.tmp_dir
    out_dir = _reject_tmp_path(raw_out_dir, "out_dir")
    scratch_root = _reject_tmp_path(raw_scratch_root, "scratch_root")

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = scratch_root / "build_synth_work"
    shutil.rmtree(scratch_dir, ignore_errors=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    orig_cwd = os.getcwd()
    orig_stdout = os.dup(1)
    orig_stderr = os.dup(2)
    console_log_path = scratch_dir / "casa_console.log"
    console_log_fh = console_log_path.open("w", encoding="utf-8")
    failed = False
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(console_log_fh.fileno(), 1)
        os.dup2(console_log_fh.fileno(), 2)
        os.chdir(scratch_dir)
        from casatasks import casalog

        casalog.setlogfile(str(scratch_dir / "casa_synth.log"))
        casalog.showconsole(False)

        sigma_info = compute_theoretical_mosaic_sigma_jy(SIGMA_VIS_JY, include_slew_quack=True)
        sigma_th_jy = float(sigma_info["sigma_th_jy_per_beam"])
        pt_flatnoise_pb = float(sigma_info["pt_mosaic_flatnoise_response"])
        pt_flux_jy = (PT_SOURCE_SIGMA_MULT * sigma_th_jy) / max(pt_flatnoise_pb, 0.3)

        base_ms = scratch_dir / "base_noiseless.ms"
        base_info = create_base_noiseless_ms(
            base_ms,
            point_source_flux_jy=pt_flux_jy,
            work_dir=scratch_dir / "cl_work",
        )

        raw_ms = scratch_dir / "raw_corrupted.ms"
        shutil.copytree(base_ms, raw_ms)
        masks_npz = scratch_dir / "corruption_masks.npz"
        target_vis_npz = scratch_dir / "true_target_vis.npz"

        corrupt_info = apply_corruptions_to_ms(
            raw_ms,
            seed=SYNTH_SEED,
            sigma_vis_jy=SIGMA_VIS_JY,
            dead_ant_mode="zero",
            include_slew_quack=True,
            masks_out_npz=masks_npz,
            target_vis_out_npz=target_vis_npz,
        )

        # Verify actual clean target integration counts match compute_theoretical_mosaic_sigma_jy
        expected_clean_tgt_rows = int(
            (
                sigma_info["clean_target_p1_integrations"]
                + sigma_info["clean_target_p2_integrations"]
            )
            * sigma_info["clean_baselines"]
        )
        with np.load(target_vis_npz, allow_pickle=False) as tvis_check:
            actual_clean_tgt_rows = int(np.sum(tvis_check["clean_target_row_mask"]))
        if actual_clean_tgt_rows != expected_clean_tgt_rows:
            raise AssertionError(
                f"Clean target row count mismatch: {actual_clean_tgt_rows} != {expected_clean_tgt_rows}"
            )

        # Stage raw MS tasks: s1_inspect, s1_precal, s2_e2e_synth
        for task_id in ("s1_inspect", "s1_precal", "s2_e2e_synth"):
            task_ms = out_dir / task_id / "input.ms"
            if task_ms.exists():
                shutil.rmtree(task_ms)
            task_ms.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(raw_ms, task_ms)
            clear_ms_history(task_ms)
            verify_ms_unlocked(task_ms)

        shutil.copy2(masks_npz, out_dir / "s1_precal" / "corruption_masks.npz")
        shutil.copy2(masks_npz, out_dir / "s2_e2e_synth" / "corruption_masks.npz")
        shutil.copy2(target_vis_npz, out_dir / "s2_e2e_synth" / "true_target_vis.npz")

        # Verify precal workflow on s1_precal/input.ms
        precal_ver = _verify_precal_workflow(
            out_dir / "s1_precal" / "input.ms",
            masks_npz=out_dir / "s1_precal" / "corruption_masks.npz",
            expected_j1331_amp_jy=float(base_info["j1331_pb2017_mean_amp_jy"]),
            scratch_dir=scratch_dir,
        )

        # Stage s1_calibrate/input.ms and run CASA G0->K->B->G->fluxscale->applycal -> s1_image/input.ms
        s1_cal_ms = out_dir / "s1_calibrate" / "input.ms"
        cal_ckpt_info = _build_s1_calibrate_checkpoint(
            raw_ms=raw_ms,
            cal_checkpoint_ms=s1_cal_ms,
            masks_npz=masks_npz,
        )
        shutil.copy2(target_vis_npz, out_dir / "s1_calibrate" / "true_target_vis.npz")

        s1_img_ms = out_dir / "s1_image" / "input.ms"
        cal_ver = _run_and_verify_calibration(
            s1_cal_input_ms=s1_cal_ms,
            s1_image_input_ms=s1_img_ms,
            target_vis_npz=target_vis_npz,
            pcal_true_flux_jy=float(base_info["pcal_spw_mean_amp_jy"]),
            scratch_dir=scratch_dir,
        )

        # Verify both reference multiscale tclean and default ms_tclean-equivalent on s1_image/input.ms
        img_ver = _run_and_verify_imaging(
            s1_image_input_ms=s1_img_ms,
            sigma_th_jy=sigma_th_jy,
            pt_flux_jy=pt_flux_jy,
            scratch_dir=scratch_dir,
        )

        # Write ground_truth.json for all 5 Suite S tasks (SPEC.md Section 4 weights + tolerances)
        gt_inspect = _common_ground_truth(
            "s1_inspect",
            base_info=base_info,
            corrupt_info=corrupt_info,
            sigma_info=sigma_info,
            pt_flux_jy=pt_flux_jy,
        )
        gt_inspect["weights"] = {
            "valid_json": 0.20,
            "flux_bandpass_calibrator": 0.20,
            "phase_vs_target_separation": 0.25,
            "dead_antenna": 0.15,
            "bandpass_min_elevation": 0.20,
        }
        gt_inspect["tolerances"] = {
            "elevation_tol_deg": 0.25,
            "pass_reward_min": 0.85,
        }
        _write_json(out_dir / "s1_inspect" / "ground_truth.json", gt_inspect)

        gt_precal = _common_ground_truth(
            "s1_precal",
            base_info=base_info,
            corrupt_info=corrupt_info,
            sigma_info=sigma_info,
            pt_flux_jy=pt_flux_jy,
        )
        gt_precal["corruption_masks_file"] = "corruption_masks.npz"
        gt_precal["corruptions_indices"] = {
            "quack_row_indices": corrupt_info["quack_row_indices"],
            "scan1_quack_row_indices": corrupt_info["scan1_quack_row_indices"],
            "burst_row_indices": corrupt_info["burst_row_indices"],
            "dead_row_indices": corrupt_info["dead_row_indices"],
            "rfi_channels": corrupt_info["rfi_channels"],
        }
        gt_precal["weights"] = {
            "calibrators_ms_valid": 0.20,
            "model_data_accuracy": 0.25,
            "rfi_preflag_recall": 0.30,
            "clean_false_positive_guard": 0.25,
        }
        gt_precal["tolerances"] = {
            "min_corrupted_recall": 0.95,
            "max_clean_false_positive_rate": 0.05,
            "j1331_model_rel_tol": 0.01,
            "pass_reward_min": 0.85,
        }
        gt_precal["reference_verification"] = precal_ver
        _write_json(out_dir / "s1_precal" / "ground_truth.json", gt_precal)

        gt_cal = _common_ground_truth(
            "s1_calibrate",
            base_info=base_info,
            corrupt_info=corrupt_info,
            sigma_info=sigma_info,
            pt_flux_jy=pt_flux_jy,
        )
        gt_cal["true_target_vis_file"] = "true_target_vis.npz"
        gt_cal["checkpoint_input_state"] = cal_ckpt_info
        gt_cal["weights"] = {
            "corrected_dead_ant_target_retention": 0.20,
            "bandpass_delay_flatness": 0.25,
            "fluxscale_accuracy": 0.25,
            "target_coherent_visibility": 0.30,
        }
        gt_cal["tolerances"] = {
            "min_clean_target_row_retention": 0.85,
            "dead_antenna_flagged_min": 0.99,
            "max_j1331_phase_slope_deg_per_100mhz": 5.0,
            "max_fluxscale_rel_err": 0.05,
            "max_target_coherent_amp_rel_err": 0.08,
            "max_target_coherent_phase_deg": 10.0,
            "target_retention_excludes_dead_antenna": True,
            "target_retention_excludes_autocorr": True,
            "pass_reward_min": 0.85,
        }
        gt_cal["reference_verification"] = cal_ver
        _write_json(out_dir / "s1_calibrate" / "ground_truth.json", gt_cal)

        gt_img = _common_ground_truth(
            "s1_image",
            base_info=base_info,
            corrupt_info=corrupt_info,
            sigma_info=sigma_info,
            pt_flux_jy=pt_flux_jy,
        )
        gt_img["weights"] = {
            "products_valid": 0.15,
            "beam_sampling": 0.20,
            "off_source_residual_rms": 0.25,
            "flux_snr_and_bowl_guard": 0.40,
        }
        gt_img["tolerances"] = {
            "min_pixels_per_beam": 3.0,
            "max_pixels_per_beam": 7.0,
            "pb_cutoff": 0.2,
            "max_offsource_rms_sigma_th_mult": 1.5,
            "min_point_source_snr": 5.0,
            "max_ext_flux_rel_err": 0.15,
            "max_negative_bowl_ratio": 0.12,
            "pass_reward_min": 0.85,
        }
        gt_img["reference_verification"] = img_ver
        _write_json(out_dir / "s1_image" / "ground_truth.json", gt_img)

        gt_e2e = _common_ground_truth(
            "s2_e2e_synth",
            base_info=base_info,
            corrupt_info=corrupt_info,
            sigma_info=sigma_info,
            pt_flux_jy=pt_flux_jy,
        )
        gt_e2e["corruption_masks_file"] = "corruption_masks.npz"
        gt_e2e["true_target_vis_file"] = "true_target_vis.npz"
        gt_e2e["weights"] = {
            "calibration_score": 0.60,
            "delivered_image_score": 0.40,
        }
        gt_e2e["calibration_weights"] = dict(gt_cal["weights"])
        gt_e2e["image_weights"] = dict(gt_img["weights"])
        gt_e2e["tolerances"] = {
            "calibration_weight": 0.60,
            "imaging_weight": 0.40,
            "min_clean_target_row_retention": 0.85,
            "dead_antenna_flagged_min": 0.99,
            "max_j1331_phase_slope_deg_per_100mhz": 5.0,
            "max_fluxscale_rel_err": 0.05,
            "max_target_coherent_amp_rel_err": 0.08,
            "max_target_coherent_phase_deg": 10.0,
            "target_retention_excludes_dead_antenna": True,
            "target_retention_excludes_autocorr": True,
            "min_pixels_per_beam": 3.0,
            "max_pixels_per_beam": 7.0,
            "pb_cutoff": 0.2,
            "max_offsource_rms_sigma_th_mult": 1.5,
            "min_point_source_snr": 5.0,
            "max_ext_flux_rel_err": 0.15,
            "max_negative_bowl_ratio": 0.12,
            "pass_reward_min": 0.85,
        }
        gt_e2e["reference_verification"] = {
            "precal": precal_ver,
            "calibrate": cal_ver,
            "image": img_ver,
        }
        _write_json(out_dir / "s2_e2e_synth" / "ground_truth.json", gt_e2e)
    except BaseException:
        failed = True
        raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            try:
                os.dup2(orig_stdout, 1)
                os.dup2(orig_stderr, 2)
            finally:
                console_log_fh.close()
                os.chdir(orig_cwd)
                try:
                    if failed and console_log_path.is_file():
                        tail_lines = console_log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()[-60:]
                        if tail_lines:
                            msg = (
                                "--- Last 60 lines of casa_console.log ---\n"
                                + "\n".join(tail_lines)
                                + "\n-----------------------------------------\n"
                            )
                            os.write(orig_stderr, msg.encode("utf-8", errors="replace"))
                finally:
                    os.close(orig_stdout)
                    os.close(orig_stderr)
                    shutil.rmtree(scratch_dir, ignore_errors=True)

    # Final lock & HISTORY hygiene + size report across all 5 Suite S directories
    task_ids = ("s1_inspect", "s1_precal", "s1_calibrate", "s1_image", "s2_e2e_synth")
    sizes_mb: dict[str, float] = {}
    for tid in task_ids:
        ms_dir = out_dir / tid / "input.ms"
        if not ms_dir.exists():
            raise RuntimeError(f"Missing expected output MS: {ms_dir}")
        _assert_history_empty(ms_dir)
        verify_ms_unlocked(ms_dir)
        if tid == "s1_calibrate":
            cal_split_dir = out_dir / tid / "calibrators.ms"
            if not cal_split_dir.exists():
                raise RuntimeError(f"Missing expected output calibrators.ms: {cal_split_dir}")
            _assert_history_empty(cal_split_dir)
            verify_ms_unlocked(cal_split_dir)
        sizes_mb[tid] = _dir_size_mb(out_dir / tid)

    elapsed_s = round(time.perf_counter() - t_start, 2)
    return {
        "elapsed_sec": elapsed_s,
        "out_dir": str(out_dir),
        "sizes_mb": sizes_mb,
        "total_size_mb": round(sum(sizes_mb.values()), 2),
        "bp_min_elevation_deg": base_info["bp_min_elevation_deg"],
        "bp_min_elevation_casa_azelgeo_deg": base_info["bp_min_elevation_casa_azelgeo_deg"],
        "obs_duration_s": base_info["obs_duration_s"],
        "j1331_pb2017_mean_amp_jy": base_info["j1331_pb2017_mean_amp_jy"],
        "pcal_spw_mean_amp_jy": base_info["pcal_spw_mean_amp_jy"],
        "sigma_th_jy_per_beam": round(sigma_th_jy, 8),
        "point_source_flux_jy": round(pt_flux_jy, 8),
        "precal_verification": precal_ver,
        "calibrate_verification": cal_ver,
        "image_verification": img_ver,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and verify Suite S synthetic MS datasets.")
    add_config_cli_args(parser)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root for Suite S tasks (default: <data_dir>/synth)",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Scratch root directory (never /tmp)",
    )
    args = parser.parse_args(argv)
    cfg = config_from_cli_args(args)
    summary = build_synthetic_suite(
        out_dir=args.out_dir,
        scratch_root=args.scratch_root,
        cfg=cfg,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
