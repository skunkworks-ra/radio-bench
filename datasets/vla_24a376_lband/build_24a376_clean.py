"""Clean Track A VLA 24A-376 Option A checkpoint & ground-truth builder.

Builds a clean, zero-history-leak, 4096x4096 first-sidelobe wideband mosaic
checkpoint set under /mnt/fast/radio_bench_data/24a376_clean/ without modifying
the source MS, the earlier 24a376_test_slice/, or the radio-bench git repo.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

OUT_ROOT = Path("/mnt/fast/radio_bench_data/24a376_clean")
SRC_DIR = Path("/mnt/fast/datasets/24A-376")
SRC_MS = SRC_DIR / "24A-376.sb45387559.eb45519359.60419.ms"
SRC_FLAGONLINE = SRC_DIR / "24A-376.sb45387559.eb45519359.60419.flagonline.txt"

SLICE_SPW = "7,11,21,22"
SLICE_FIELDS = (
    "J0336+3218,0542+498=3C147,0521+166=3C138,PER_FIELD_71,PER_FIELD_75,PER_FIELD_77,PER_FIELD_81"
)
SLICE_SCAN = (
    "2,3,20,24,26,30,32,60,61,62,63,64"  # Drop VLA SYSTEM_CONFIGURATION setup scan (scan 1)
)
CAL_FIELDS = "J0336+3218,0542+498=3C147,0521+166=3C138"
FLUX_CAL = "0542+498=3C147"
PHASE_CAL = "J0336+3218"
POL_CAL = "0521+166=3C138"
TARGET_FIELDS_LIST = ["PER_FIELD_71", "PER_FIELD_75", "PER_FIELD_77", "PER_FIELD_81"]
TARGET_FIELDS = ",".join(TARGET_FIELDS_LIST)
REFANT = "ea16,ea04,ea07"

MOSAIC_PHASECENTER = "J2000 52.11772deg 30.44509deg"
VERIFIER_TCLEAN_RECIPE: dict[str, Any] = {
    "field": TARGET_FIELDS,
    "spw": "",
    "phasecenter": MOSAIC_PHASECENTER,
    "specmode": "mfs",
    "gridder": "mosaic",
    "deconvolver": "mtmfs",
    "nterms": 2,
    "scales": [0, 5, 15],
    "imsize": [4096, 4096],
    "cell_arcsec": 2.5,
    "weighting": "briggs",
    "robust": 0.5,
    "pblimit": 0.2,
    "usemask": "pb",
    "pbmask": 0.2,
    "niter": 4000,
    "cycleniter": 1000,
    "nmajor": 4,
    "threshold": "0.60mJy",
    "pbcor": True,
    "off_source_min_radius_arcsec": 150.0,
    "timeout_sec": 2100.0,
}


def _setup_env() -> None:
    tmp_dir = OUT_ROOT / "_tmp"
    mpl_dir = tmp_dir / "mplconfig"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@contextlib.contextmanager
def open_tb(path: Path | str, nomodify: bool = True) -> Iterator[Any]:
    from casatools import table

    tb = table()
    opened = False
    try:
        tb.open(str(path), nomodify=nomodify)
        opened = True
        yield tb
    finally:
        if opened:
            with contextlib.suppress(Exception):
                tb.close()
        with contextlib.suppress(Exception):
            tb.done()


def clear_history(ms_path: Path) -> None:
    hist = ms_path / "HISTORY"
    if (hist / "table.dat").is_file():
        with open_tb(hist, nomodify=False) as tb:
            n = int(tb.nrows())
            if n > 0:
                tb.removerows(list(range(n)))


def assert_unlocked(ms_path: Path) -> None:
    import casatasks

    casatasks.clearstat()
    with open_tb(ms_path, nomodify=True) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active table lock on {ms_path}")


def dir_size_mb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 2)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inspect_ms_meta(ms_path: Path) -> dict[str, Any]:
    from casatools import msmetadata

    with open_tb(ms_path) as tb:
        nrows = int(tb.nrows())
        cols = sorted(tb.colnames())
        flag = tb.getcol("FLAG")
        flag_row = tb.getcol("FLAG_ROW")
        flagged_vis_frac = float(np.mean(flag))
        flagged_row_frac = float(np.mean(flag_row))
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        active_ant_ids = sorted({int(x) for x in np.unique(np.concatenate([ant1, ant2]))})
    with open_tb(ms_path / "FIELD") as tb:
        field_names = [str(x) for x in tb.getcol("NAME")]
    with open_tb(ms_path / "SPECTRAL_WINDOW") as tb:
        nchan = [int(x) for x in tb.getcol("NUM_CHAN")]
        bw_mhz = [round(float(x) / 1e6, 3) for x in tb.getcol("TOTAL_BANDWIDTH")]
        ref_freq_ghz = [round(float(x) / 1e9, 4) for x in tb.getcol("REF_FREQUENCY")]
    with open_tb(ms_path / "POLARIZATION") as tb:
        pol_nrows = int(tb.nrows())
        pol_corrs = [list(map(int, tb.getcell("CORR_TYPE", i))) for i in range(pol_nrows)]
    with open_tb(ms_path / "DATA_DESCRIPTION") as tb:
        ddid_pols = [int(x) for x in tb.getcol("POLARIZATION_ID")]
    with open_tb(ms_path / "HISTORY") as tb:
        history_nrows = int(tb.nrows())
    syspower_exists = (ms_path / "SYSPOWER" / "table.dat").is_file()
    syspower_nrows = 0
    if syspower_exists:
        with open_tb(ms_path / "SYSPOWER") as tb:
            syspower_nrows = int(tb.nrows())

    msmd = msmetadata()
    try:
        msmd.open(str(ms_path))
        field_intents = {
            name: sorted(str(x) for x in msmd.intentsforfield(i))
            for i, name in enumerate(field_names)
        }
        scans = [int(s) for s in sorted(msmd.scannumbers())]
    finally:
        with contextlib.suppress(Exception):
            msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()

    return {
        "ms_path": str(ms_path),
        "size_mb": dir_size_mb(ms_path),
        "nrows": nrows,
        "columns": cols,
        "scans": scans,
        "n_scans": len(scans),
        "n_fields": len(field_names),
        "field_names": field_names,
        "field_intents": field_intents,
        "n_active_antennas": len(active_ant_ids),
        "spw_nchan": nchan,
        "spw_bw_mhz": bw_mhz,
        "spw_ref_freq_ghz": ref_freq_ghz,
        "pol_nrows": pol_nrows,
        "pol_corrs": pol_corrs,
        "ddid_pols": ddid_pols,
        "syspower_exists": syspower_exists,
        "syspower_nrows": syspower_nrows,
        "history_nrows": history_nrows,
        "flagged_vis_fraction": round(flagged_vis_frac, 6),
        "flagged_row_fraction": round(flagged_row_frac, 6),
    }


def compute_min_elevation_deg(ms_path: Path, field_name: str) -> float:
    from casatools import measures, msmetadata, quanta

    msmd = msmetadata()
    me = measures()
    qa = quanta()
    try:
        msmd.open(str(ms_path))
        me.doframe(me.observatory("VLA"))
        fnames = [str(x) for x in msmd.fieldnames()]
        fid = fnames.index(field_name)
        pdir = msmd.phasecenter(fid)
        scans = msmd.scansforfield(fid)
        times = np.concatenate([msmd.timesforscan(s) for s in scans])
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            azel = me.measure(pdir, "AZEL")
            els.append(float(qa.convert(azel["m1"], "deg")["value"]))
        return float(min(els))
    finally:
        with contextlib.suppress(Exception):
            msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()
        with contextlib.suppress(Exception):
            me.done()
        with contextlib.suppress(Exception):
            qa.done()


def build_extract_config(stages: list[str], *, use_verifier_tclean: bool = False) -> Any:
    from radio_bench.grading.schema import ExtractConfig

    geom: dict[str, Any] = {
        "dish_diameter_m": 25.0,
        "pb_mask_min": 0.20,
        "offsource_inner_exclusion_beams": 5.0,
        "phasecenter": MOSAIC_PHASECENTER,
        "off_source_min_radius_arcsec": 150.0,
    }
    if use_verifier_tclean:
        geom["verifier_tclean_recipe"] = VERIFIER_TCLEAN_RECIPE
    return ExtractConfig.model_validate(
        {
            "suite_id": "vla_24a376_lband",
            "stages": stages,
            "allow_virtual_model": False,
            "target_retention_excludes_dead_antenna": False,
            "roles": {
                "flux_cal": [FLUX_CAL],
                "bandpass_cal": [FLUX_CAL],
                "phase_cal": [PHASE_CAL],
                "pol_angle_cal": [POL_CAL],
                "pol_leakage_cal": [FLUX_CAL],
                "targets": TARGET_FIELDS_LIST,
                "dead_antennas": [],
                "missing_stations": [],
            },
            "geometry": geom,
        }
    )


def main() -> None:
    _setup_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT_ROOT)

    import casatasks

    from radio_bench.grading.extractors.calibrate import extract_calibrate
    from radio_bench.grading.extractors.common import compute_raw_data_blake2b
    from radio_bench.grading.extractors.image import extract_image
    from radio_bench.grading.extractors.precal import extract_precal

    timings: dict[str, float] = {}
    summary: dict[str, Any] = {
        "dataset": "24A-376",
        "option": 'A-clean (SPWs 7,11,21,22, width=2 32ch x 2MHz, scan=!1, 4096x4096 2.5" first-sidelobe mosaic)',
        "source_ms": str(SRC_MS),
        "source_flagonline": str(SRC_FLAGONLINE),
        "out_root": str(OUT_ROOT),
    }

    # --- 1. Clean raw MS ---
    raw_dir = OUT_ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_ms = raw_dir / "input.ms"
    raw_flagonline = raw_dir / "input.flagonline.txt"
    shutil.copy2(SRC_FLAGONLINE, raw_flagonline)

    t0 = time.perf_counter()
    if raw_ms.exists():
        shutil.rmtree(raw_ms)
    casatasks.mstransform(
        vis=str(SRC_MS),
        outputvis=str(raw_ms),
        spw=SLICE_SPW,
        field=SLICE_FIELDS,
        scan=SLICE_SCAN,
        datacolumn="data",
        chanaverage=True,
        chanbin=2,
        keepflags=True,
        reindex=True,
    )
    clear_history(raw_ms)
    assert_unlocked(raw_ms)
    raw_blake2b = compute_raw_data_blake2b(raw_ms)
    timings["slice_raw_s"] = round(time.perf_counter() - t0, 2)

    raw_meta = inspect_ms_meta(raw_ms)
    raw_meta["raw_data_blake2b"] = raw_blake2b
    bp_min_el_deg = round(compute_min_elevation_deg(raw_ms, FLUX_CAL), 4)
    raw_meta["bandpass_min_elevation_deg"] = bp_min_el_deg
    write_json(OUT_ROOT / "raw_summary.json", raw_meta)
    summary["raw"] = raw_meta
    print(
        f"[1/5] Clean raw MS: {raw_meta['size_mb']} MB, {raw_meta['nrows']} rows, "
        f"scans={raw_meta['n_scans']}, hist={raw_meta['history_nrows']}, "
        f"3C147 el_min={bp_min_el_deg} deg in {timings['slice_raw_s']}s",
        flush=True,
    )

    # --- 2. Clean post_precal checkpoint + 3-way precal_masks.npz ---
    pre_dir = OUT_ROOT / "post_precal"
    if pre_dir.exists():
        shutil.rmtree(pre_dir)
    pre_dir.mkdir(parents=True, exist_ok=True)
    pre_ms = pre_dir / "input.ms"
    shutil.copytree(raw_ms, pre_ms)
    shutil.copy2(raw_flagonline, pre_dir / "input.flagonline.txt")

    t0 = time.perf_counter()
    with open_tb(pre_ms) as tb:
        raw_flag = tb.getcol("FLAG").astype(bool)
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        field_id = tb.getcol("FIELD_ID")
        ddid = tb.getcol("DATA_DESC_ID")
    with open_tb(pre_ms / "FIELD") as tb:
        fnames = [str(x) for x in tb.getcol("NAME")]
    flux_fid = fnames.index(FLUX_CAL)

    # Mandatory deterministic preflags: online flags + 5s quack + shadow + clipzeros
    casatasks.flagdata(
        vis=str(pre_ms),
        mode="list",
        inpfile=str(pre_dir / "input.flagonline.txt"),
        flagbackup=False,
    )
    casatasks.flagdata(
        vis=str(pre_ms), mode="quack", quackinterval=5.0, quackmode="beg", flagbackup=False
    )
    casatasks.flagdata(vis=str(pre_ms), mode="shadow", tolerance=0.0, flagbackup=False)
    casatasks.flagdata(vis=str(pre_ms), mode="clip", clipzeros=True, flagbackup=False)
    with open_tb(pre_ms) as tb:
        must_det_flag = tb.getcol("FLAG").astype(bool)

    # 2nd integration quack (10s) -> neutral band
    casatasks.flagdata(
        vis=str(pre_ms), mode="quack", quackinterval=10.0, quackmode="beg", flagbackup=False
    )
    with open_tb(pre_ms) as tb:
        det_all_flag = tb.getcol("FLAG").astype(bool)

    # Conservative tfcrop + extendpols (standard preflag)
    casatasks.flagdata(
        vis=str(pre_ms), mode="tfcrop", timecutoff=3.0, freqcutoff=3.0, flagbackup=False
    )
    casatasks.flagdata(vis=str(pre_ms), mode="extend", extendpols=True, flagbackup=False)
    with open_tb(pre_ms) as tb:
        tfcrop_flag = tb.getcol("FLAG").astype(bool)

    # Setjy Perley-Butler 2017 on 3C147
    casatasks.setjy(vis=str(pre_ms), field=FLUX_CAL, standard="Perley-Butler 2017", usescratch=True)

    # Scratch copy for initial bandpass + rflag classification
    rfi_work = OUT_ROOT / "_tmp" / "precal_rfi"
    if rfi_work.exists():
        shutil.rmtree(rfi_work)
    rfi_work.mkdir(parents=True, exist_ok=True)
    rfi_ms = rfi_work / "work.ms"
    shutil.copytree(pre_ms, rfi_ms)
    g_init = rfi_work / "G_init.cal"
    b_init = rfi_work / "B_init.cal"
    casatasks.gaincal(
        vis=str(rfi_ms),
        caltable=str(g_init),
        field=FLUX_CAL,
        spw="*:6~25",
        refant=REFANT,
        calmode="p",
        solint="int",
        minsnr=3.0,
    )
    casatasks.bandpass(
        vis=str(rfi_ms),
        caltable=str(b_init),
        field=FLUX_CAL,
        refant=REFANT,
        solint="inf",
        combine="scan",
        solnorm=True,
        minsnr=3.0,
        gaintable=[str(g_init)],
    )
    casatasks.applycal(
        vis=str(rfi_ms),
        field=FLUX_CAL,
        gaintable=[str(g_init), str(b_init)],
        interp=["nearest", "nearest,linear"],
        calwt=False,
        flagbackup=False,
    )
    other_fields = ",".join(f for f in fnames if f != FLUX_CAL)
    casatasks.applycal(
        vis=str(rfi_ms),
        field=other_fields,
        gaintable=[str(b_init)],
        gainfield=[FLUX_CAL],
        interp=["nearest,linear"],
        calwt=False,
        flagbackup=False,
    )
    with open_tb(rfi_ms) as tb:
        binit_corr = tb.getcol("CORRECTED_DATA")
        binit_flag = tb.getcol("FLAG").astype(bool)

    casatasks.flagdata(
        vis=str(rfi_ms),
        mode="rflag",
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )
    with open_tb(rfi_ms) as tb:
        golden_precal_flag = tb.getcol("FLAG").astype(bool)

    # Classify strong (>=8 MAD) vs mild (>=3.5 MAD) per field, DDID, channel
    par_amp = 0.5 * (np.abs(binit_corr[0]) + np.abs(binit_corr[3]))
    pre_valid = (~np.any(binit_flag, axis=0)) & (ant1 != ant2)[np.newaxis, :]
    strong_amp = np.zeros_like(par_amp, dtype=bool)
    mild_amp = np.zeros_like(par_amp, dtype=bool)
    for fid in np.unique(field_id):
        for d in np.unique(ddid):
            rows = np.flatnonzero((field_id == fid) & (ddid == d))
            if rows.size < 16:
                continue
            f_amp = par_amp[:, rows]
            f_val = pre_valid[:, rows]
            for ch in range(par_amp.shape[0]):
                v = f_amp[ch, f_val[ch]]
                if v.size < 16:
                    continue
                med = float(np.median(v))
                mad = float(1.4826 * np.median(np.abs(v - med)))
                if mad <= 0.0:
                    continue
                dev = np.abs(f_amp[ch] - med)
                strong_amp[ch, rows] = dev >= 8.0 * mad
                mild_amp[ch, rows] = dev >= 3.5 * mad
    del binit_corr, binit_flag, par_amp

    # Build 3-way disjoint masks (32, nrow)
    raw_2d = np.any(raw_flag, axis=0)
    must_det_2d = np.any(must_det_flag, axis=0)
    det_all_2d = np.any(det_all_flag, axis=0)
    tfcrop_2d = np.any(tfcrop_flag, axis=0)
    golden_2d = np.any(golden_precal_flag, axis=0)

    rfi_only_2d = golden_2d & (~det_all_2d)
    edge_ch_2d = np.zeros_like(raw_2d, dtype=bool)
    edge_ch_2d[[0, 1, 30, 31], :] = True

    must_flag_2d = (must_det_2d & (~raw_2d)) | (rfi_only_2d & strong_amp)
    neutral_2d = (
        raw_2d | (det_all_2d & (~must_det_2d)) | tfcrop_2d | golden_2d | mild_amp | edge_ch_2d
    ) & (~must_flag_2d)
    clean_2d = ~(must_flag_2d | neutral_2d)

    assert not np.any(must_flag_2d & neutral_2d)
    assert not np.any(must_flag_2d & clean_2d)
    assert not np.any(neutral_2d & clean_2d)
    assert np.all(must_flag_2d | neutral_2d | clean_2d)

    # Write golden precal flags to post_precal/input.ms and strip CORRECTED_DATA
    final_flag_row = np.all(golden_precal_flag, axis=(0, 1))
    with open_tb(pre_ms, nomodify=False) as tb:
        tb.putcol("FLAG", golden_precal_flag)
        tb.putcol("FLAG_ROW", final_flag_row)
        if "CORRECTED_DATA" in tb.colnames():
            tb.removecols(["CORRECTED_DATA"])
        model_data = tb.getcol("MODEL_DATA")

    # Measure 3C147 MODEL_DATA mean amp and SPW 0 spectrum (matching precal.py)
    flux_unflg_mask = (field_id == flux_fid) & (~final_flag_row)
    par_mod = 0.5 * (np.abs(model_data[0]) + np.abs(model_data[3]))
    par_good = (~np.any(golden_precal_flag[[0, 3]], axis=0)) & flux_unflg_mask[np.newaxis, :]
    flux_mean_amp_jy = float(np.mean(par_mod[par_good]))
    # Primary DDID (0) per-channel spectrum
    ddid0_flux = (field_id == flux_fid) & (ddid == 0)
    spw0_spec = [
        round(float(np.mean(par_mod[ch, ddid0_flux])), 6) for ch in range(par_mod.shape[0])
    ]
    del model_data, par_mod

    # Split calibrators.ms (with DATA + MODEL_DATA, unflagged rows kept via keepflags=False)
    cal_ms = pre_dir / "calibrators.ms"
    casatasks.mstransform(
        vis=str(pre_ms),
        outputvis=str(cal_ms),
        field=CAL_FIELDS,
        datacolumn="all",
        keepflags=False,
    )
    clear_history(pre_ms)
    clear_history(cal_ms)
    assert_unlocked(pre_ms)
    assert_unlocked(cal_ms)

    # Save bit-packed precal_masks.npz
    masks_npz = pre_dir / "precal_masks.npz"
    np.savez_compressed(
        masks_npz,
        shape_2d=np.array(must_flag_2d.shape, dtype=np.int64),
        must_flag_2d=np.packbits(must_flag_2d, axis=1),
        neutral_2d=np.packbits(neutral_2d, axis=1),
        clean_2d=np.packbits(clean_2d, axis=1),
        ref_flux_cal_spectrum_jy=np.array(spw0_spec, dtype=np.float64),
    )

    # Also generate reference priorcals in pre_dir (for reference audit)
    gc_tab = pre_dir / "gain_curves.gc"
    opac_tab = pre_dir / "opacities.opac"
    rq_tab = pre_dir / "requantizer.rq"
    casatasks.gencal(vis=str(cal_ms), caltable=str(gc_tab), caltype="gc")
    taus = casatasks.plotweather(vis=str(cal_ms), doPlot=False)
    casatasks.gencal(
        vis=str(cal_ms), caltable=str(opac_tab), caltype="opac", spw="0,1,2,3", parameter=taus
    )
    casatasks.gencal(vis=str(cal_ms), caltable=str(rq_tab), caltype="rq")
    casatasks.clearstat()

    precal_blake2b = compute_raw_data_blake2b(pre_ms)
    timings["precal_s"] = round(time.perf_counter() - t0, 2)

    # Verify with extract_precal
    pre_obs, pre_errs = extract_precal(pre_dir, build_extract_config(["precal"]), pre_dir)
    pre_obs_dict = {k: v.value for k, v in pre_obs.items()}
    pre_summary = {
        "input_ms": inspect_ms_meta(pre_ms),
        "calibrators_ms": inspect_ms_meta(cal_ms),
        "raw_data_blake2b": precal_blake2b,
        "flux_cal_model_mean_jy": round(flux_mean_amp_jy, 6),
        "flux_cal_spw0_spectrum_jy": spw0_spec,
        "mask_counts": {
            "must_flag": int(np.sum(must_flag_2d)),
            "neutral": int(np.sum(neutral_2d)),
            "clean": int(np.sum(clean_2d)),
            "total": int(must_flag_2d.size),
        },
        "opacities_taus": [round(float(x), 5) for x in taus],
        "extractor_observables": pre_obs_dict,
        "extractor_errors": pre_errs,
        "elapsed_s": timings["precal_s"],
    }
    write_json(OUT_ROOT / "precal_summary.json", pre_summary)
    summary["precal"] = pre_summary
    print(
        f"[2/5] Clean post_precal done in {timings['precal_s']}s: "
        f"model_amp={pre_obs_dict.get('precal.flux_cal_model_amp_jy')}, "
        f"cal_recall={pre_obs_dict.get('precal.calibrator_must_flag_recall')}, "
        f"clean_fpr={pre_obs_dict.get('precal.clean_false_positive_rate')}, errs={pre_errs}",
        flush=True,
    )

    # --- 3. Clean post_calibrate checkpoint ---
    cal_dir = OUT_ROOT / "post_calibrate"
    if cal_dir.exists():
        shutil.rmtree(cal_dir)
    cal_dir.mkdir(parents=True, exist_ok=True)
    work_ms = cal_dir / "input.ms"
    work_cal_ms = cal_dir / "calibrators.ms"
    shutil.copytree(pre_ms, work_ms)
    shutil.copytree(cal_ms, work_cal_ms)

    t0 = time.perf_counter()
    c_gc = cal_dir / "gain_curves.gc"
    c_opac = cal_dir / "opacities.opac"
    c_rq = cal_dir / "requantizer.rq"
    casatasks.gencal(vis=str(work_cal_ms), caltable=str(c_gc), caltype="gc")
    c_taus = casatasks.plotweather(vis=str(work_cal_ms), doPlot=False)
    casatasks.gencal(
        vis=str(work_cal_ms), caltable=str(c_opac), caltype="opac", spw="0,1,2,3", parameter=c_taus
    )
    casatasks.gencal(vis=str(work_cal_ms), caltable=str(c_rq), caltype="rq")
    priorcals = [str(c_gc), str(c_opac), str(c_rq)]

    g0 = cal_dir / "G0.cal"
    k_tab = cal_dir / "K.cal"
    b_tab = cal_dir / "B.cal"
    g_tab = cal_dir / "G.cal"
    flux_tab = cal_dir / "fluxscale.cal"

    casatasks.gaincal(
        vis=str(work_cal_ms),
        caltable=str(g0),
        field=FLUX_CAL,
        spw="*:6~25",
        refant=REFANT,
        calmode="p",
        solint="int",
        minsnr=3.0,
        gaintable=priorcals,
    )
    casatasks.gaincal(
        vis=str(work_cal_ms),
        caltable=str(k_tab),
        field=FLUX_CAL,
        refant=REFANT,
        gaintype="K",
        solint="inf",
        combine="scan",
        minsnr=3.0,
        gaintable=[*priorcals, str(g0)],
    )
    casatasks.bandpass(
        vis=str(work_cal_ms),
        caltable=str(b_tab),
        field=FLUX_CAL,
        refant=REFANT,
        solint="inf",
        combine="scan",
        solnorm=True,
        minsnr=3.0,
        gaintable=[*priorcals, str(g0), str(k_tab)],
    )
    casatasks.gaincal(
        vis=str(work_cal_ms),
        caltable=str(g_tab),
        field=CAL_FIELDS,
        refant=REFANT,
        gaintype="G",
        calmode="ap",
        solint="inf",
        minsnr=3.0,
        gaintable=[*priorcals, str(k_tab), str(b_tab)],
    )
    flux_res = casatasks.fluxscale(
        vis=str(work_cal_ms),
        caltable=str(g_tab),
        fluxtable=str(flux_tab),
        reference=[FLUX_CAL],
        transfer=[PHASE_CAL, POL_CAL],
        incremental=False,
    )
    full_tables = [*priorcals, str(k_tab), str(b_tab), str(flux_tab)]
    gf_flux = [""] * len(priorcals) + [FLUX_CAL, FLUX_CAL, FLUX_CAL]
    gf_phase = [""] * len(priorcals) + [FLUX_CAL, FLUX_CAL, PHASE_CAL]
    gf_pol = [""] * len(priorcals) + [FLUX_CAL, FLUX_CAL, POL_CAL]
    casatasks.applycal(
        vis=str(work_ms),
        field=FLUX_CAL,
        gaintable=full_tables,
        gainfield=gf_flux,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(work_ms),
        field=PHASE_CAL,
        gaintable=full_tables,
        gainfield=gf_phase,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(work_ms),
        field=POL_CAL,
        gaintable=full_tables,
        gainfield=gf_pol,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(work_ms),
        field=TARGET_FIELDS,
        gaintable=full_tables,
        gainfield=gf_phase,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.flagdata(
        vis=str(work_ms),
        field=TARGET_FIELDS,
        mode="rflag",
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )
    clear_history(work_ms)
    assert_unlocked(work_ms)
    cal_blake2b = compute_raw_data_blake2b(work_ms)
    timings["calibrate_solve_s"] = round(time.perf_counter() - t0, 2)

    flux_clean: dict[str, Any] = {}
    for _k, v in flux_res.items():
        if isinstance(v, dict) and "fieldName" in v:
            spw_flux = {}
            for sk, sv in v.items():
                if isinstance(sv, dict) and "fluxd" in sv:
                    spw_flux[str(sk)] = [round(float(x), 4) for x in sv["fluxd"]]
            flux_clean[str(v["fieldName"])] = {
                "spw_fluxd": spw_flux,
                "fitFluxd": round(float(v["fitFluxd"]), 4)
                if v.get("fitFluxd") is not None
                else None,
                "spidx": [round(float(x), 4) for x in v.get("spidx", [])]
                if v.get("spidx") is not None
                else None,
            }

    # Verify with extract_calibrate INCLUDING 4096x4096 fast verifier tclean
    t_ver = time.perf_counter()
    cal_obs, cal_errs = extract_calibrate(
        cal_dir, build_extract_config(["calibrate"], use_verifier_tclean=True)
    )
    timings["calibrate_verifier_tclean_s"] = round(time.perf_counter() - t_ver, 2)
    cal_obs_dict = {k: v.value for k, v in cal_obs.items()}
    cal_summary = {
        "calibrated_ms": inspect_ms_meta(work_ms),
        "raw_data_blake2b": cal_blake2b,
        "fluxscale": flux_clean,
        "extractor_observables": cal_obs_dict,
        "extractor_errors": cal_errs,
        "elapsed_solve_s": timings["calibrate_solve_s"],
        "elapsed_verifier_tclean_s": timings["calibrate_verifier_tclean_s"],
    }
    write_json(OUT_ROOT / "cal_summary.json", cal_summary)
    summary["calibrate"] = cal_summary
    print(
        f"[3/5] Clean post_calibrate done (solve={timings['calibrate_solve_s']}s, "
        f"ver_tclean={timings['calibrate_verifier_tclean_s']}s): "
        f"pcal={cal_obs_dict.get('cal.phase_cal_corrected_amp_jy')} Jy, "
        f"bp_rms={cal_obs_dict.get('cal.bandpass_coh_chan_phase_rms_deg')} deg, "
        f"ver_DR={cal_obs_dict.get('cal.verifier_tclean_dr')}, errs={cal_errs}",
        flush=True,
    )

    # --- 4. Clean 4096x4096 first-sidelobe reference image ---
    img_dir = OUT_ROOT / "reference_image"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    img_prefix = img_dir / "per_mosaic_4k"

    t0 = time.perf_counter()
    casatasks.tclean(
        vis=str(work_ms),
        imagename=str(img_prefix),
        field=TARGET_FIELDS,
        phasecenter=MOSAIC_PHASECENTER,
        datacolumn="corrected",
        stokes="I",
        specmode="mfs",
        deconvolver="mtmfs",
        nterms=2,
        scales=[0, 5, 15],
        gridder="mosaic",
        cell="2.5arcsec",
        imsize=[4096, 4096],
        weighting="briggs",
        robust=0.5,
        niter=10000,
        nsigma=3.0,
        usemask="auto-multithresh",
        sidelobethreshold=2.0,
        noisethreshold=4.25,
        lownoisethreshold=1.5,
        minbeamfrac=0.3,
        pblimit=0.2,
        pbcor=True,
        interactive=False,
    )
    casatasks.clearstat()
    timings["image_4k_s"] = round(time.perf_counter() - t0, 2)

    img_obs, img_errs = extract_image(img_dir, build_extract_config(["image"]))
    img_obs_dict = {k: v.value for k, v in img_obs.items()}
    img_summary = {
        "imagename_prefix": str(img_prefix),
        "imsize": [4096, 4096],
        "cell": "2.5arcsec",
        "products": sorted(p.name for p in img_dir.iterdir()),
        "image_dir_size_mb": dir_size_mb(img_dir),
        "extractor_observables": img_obs_dict,
        "extractor_errors": img_errs,
        "elapsed_s": timings["image_4k_s"],
    }
    write_json(OUT_ROOT / "image_summary.json", img_summary)
    summary["image"] = img_summary
    print(
        f"[4/5] Clean 4096x4096 reference image done in {timings['image_4k_s']}s: "
        f"peak={img_obs_dict.get('image.peak_jy_per_beam')} Jy, "
        f"rms={img_obs_dict.get('image.offsource_residual_rms_jy')} Jy, "
        f"DR={img_obs_dict.get('image.dynamic_range')}, "
        f"bowl={img_obs_dict.get('image.negative_bowl_ratio')}, errs={img_errs}",
        flush=True,
    )

    # --- 5. Emit per-task ground_truth.json & sidecars ---
    pcal_exp_jy = float(
        cal_obs_dict.get("cal.phase_cal_corrected_amp_jy") or flux_clean[PHASE_CAL]["fitFluxd"]
    )
    ver_dr = float(cal_obs_dict.get("cal.verifier_tclean_dr") or 300.0)
    ver_peak = float(
        cal_obs_dict.get("cal.verifier_tclean_peak_jy")
        or img_obs_dict.get("image.peak_jy_per_beam")
        or 0.09
    )
    ref_rms = float(img_obs_dict.get("image.offsource_residual_rms_jy") or 0.00018)
    ref_dr = float(img_obs_dict.get("image.dynamic_range") or 500.0)
    ref_peak = float(img_obs_dict.get("image.peak_jy_per_beam") or 0.09)
    tgt_ret = float(cal_obs_dict.get("cal.target_row_retention") or 0.70)

    inspect_gt = {
        "raw_data_blake2b": raw_blake2b,
        "flux_calibrator": FLUX_CAL,
        "bandpass_calibrator": FLUX_CAL,
        "phase_calibrator": PHASE_CAL,
        "pol_angle_calibrator": POL_CAL,
        "pol_leakage_calibrator": FLUX_CAL,
        "mosaic_target_fields": TARGET_FIELDS_LIST,
        "target_fields": TARGET_FIELDS_LIST,
        "dead_antennas": [],
        "missing_vla_pads": [],
        "bandpass_min_elevation_deg": bp_min_el_deg,
        "tolerances": {
            "elevation_abs_deg": 1.0,
        },
    }

    precal_gt = {
        "raw_data_blake2b": raw_blake2b,
        "flux_cal_model_parallel_mean_jy": round(
            float(pre_obs_dict["precal.flux_cal_model_amp_jy"]), 6
        ),
        "flux_cal_model_parallel_per_chan_jy": spw0_spec,
        "tolerances": {
            "model_amp_rel_tol": 0.02,
            "flag_recall_min": 0.85,
            "clean_false_positive_max": 0.05,
        },
    }

    calibrate_gt = {
        "raw_data_blake2b": precal_blake2b,
        "phase_cal_expected_flux_jy": round(pcal_exp_jy, 4),
        "fluxscale_truth": flux_clean,
        "reference_image_summary": {
            "peak_image_jy_per_beam": round(ref_peak, 6),
            "verifier_tclean_peak_jy": round(ver_peak, 6),
            "verifier_tclean_dr": round(ver_dr, 2),
        },
        "tolerances": {
            "target_unflagged_rows_min": round(max(0.45, tgt_ret - 0.20), 3),
            "target_unflagged_rows_max": 0.98,
            "bp_channel_phase_rms_max_deg": 5.0,
            "phase_cal_flux_rel_tol": 0.08,
            "verifier_tclean_dr_min": round(max(150.0, 0.70 * ver_dr), 1),
        },
    }

    image_gt = {
        "raw_data_blake2b": cal_blake2b,
        "reference_image_summary": {
            "peak_image_jy_per_beam": round(ref_peak, 6),
            "offsource_residual_rms_jy": round(ref_rms, 8),
            "dynamic_range": round(ref_dr, 2),
            "negative_bowl_ratio": round(
                float(img_obs_dict.get("image.negative_bowl_ratio") or 0.08), 5
            ),
        },
        "tolerances": {
            "pixels_per_beam_min": 2.5,
            "pixels_per_beam_max": 6.0,
            "off_source_residual_rms_max_jy_per_beam": round(ref_rms * 2.0, 6),
            "off_source_residual_rms_min_jy_per_beam": 0.00004,
            "dynamic_range_min": round(max(150.0, 0.55 * ref_dr), 1),
            "negative_model_bowl_ratio_max": 0.12,
        },
    }

    e2e_gt = {
        "raw_data_blake2b": raw_blake2b,
        "calibrate_ground_truth": {**calibrate_gt, "raw_data_blake2b": raw_blake2b},
        "image_ground_truth": {**image_gt, "raw_data_blake2b": raw_blake2b},
    }

    task_gts = {
        "r1_24a376_inspect": inspect_gt,
        "r1_24a376_precal": precal_gt,
        "r1_24a376_calibrate": calibrate_gt,
        "r1_24a376_image": image_gt,
        "r2_24a376_e2e": e2e_gt,
    }
    for tid, gt in task_gts.items():
        tdir = OUT_ROOT / tid
        tdir.mkdir(parents=True, exist_ok=True)
        write_json(tdir / "ground_truth.json", gt)
        if tid in ("r1_24a376_precal", "r2_24a376_e2e"):
            shutil.copy2(masks_npz, tdir / "precal_masks.npz")

    summary["timings_s"] = timings
    summary["ground_truth_tolerances"] = {
        "calibrate": calibrate_gt["tolerances"],
        "image": image_gt["tolerances"],
    }
    summary["total_dir_size_mb"] = dir_size_mb(OUT_ROOT)
    write_json(OUT_ROOT / "summary.json", summary)
    print(
        f"[5/5] ALL CLEAN 24A-376 CHECKPOINTS READY! total_dir={summary['total_dir_size_mb']} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
