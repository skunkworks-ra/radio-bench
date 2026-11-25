"""Deterministic checkpoint and ground-truth builder for the vla_3c75_pol reference suite.

Needs only casatools, casatasks and numpy (runs in the casa-base container). Every target
number written to ground_truth.json is read from REFERENCE.json next to this file; nothing
here invents a target. raw_data_blake2b hashes are added later by `run.py build --stage truth`.

Layout written under --root (= <data_dir>/casaguides/vla_3c75_pol):
  raw/input.ms            all 4 fields, pipeline-calibrated DATA, averaged (inspect + e2e input)
  post_polcal/input.ms    3C75 only, guide polcal applied, split to DATA, statwt (image input)
  polcal_tables/          guide Kcross / Df / Df+QU / Xf tables (future polcal reference)
  <task_id>/ground_truth.json, build_summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE_JSON = HERE / "REFERENCE.json"
SRC_MS_NAME = "TDRW0001_calibrated.ms"

# Field names and refant as used by the guide (REFERENCE.json ids field_*, refant).
# main() asserts these match REFERENCE.json.
FLUX_CAL = "0137+331=3C48"
UNPOL_CAL = "J2355+4950"
PHASE_CAL = "J0259+0747"
TARGET = "3C75"
CAL_FIELDS = f"{FLUX_CAL},{UNPOL_CAL},{PHASE_CAL}"
REFANT = "ea10"

# Size reduction for the task inputs (decision recorded in REFERENCE.json `build_choices`).
CHANBIN = 4
TIMEBIN = "10s"

TASKS = ("g1_3c75pol_inspect", "g1_3c75pol_image", "g2_3c75pol_e2e")


def _setup_env(root: Path) -> None:
    tmp_dir = Path(os.environ.get("TMPDIR") or (root / "_tmp"))
    if str(tmp_dir) == "/tmp" or str(tmp_dir).startswith("/tmp/"):
        tmp_dir = root / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR") or (tmp_dir / "mplconfig"))
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


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


@contextlib.contextmanager
def open_msmd(path: Path | str) -> Iterator[Any]:
    from casatools import msmetadata

    msmd = msmetadata()
    try:
        msmd.open(str(path))
        yield msmd
    finally:
        with contextlib.suppress(Exception):
            msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()


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
    with open_tb(ms_path) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active table lock on {ms_path}")


def dir_size_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1048576.0, 2)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_reference() -> dict[str, Any]:
    ref = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    return ref["entries"]


def rv(entries: dict[str, Any], key: str) -> Any:
    """Value of one REFERENCE.json entry (None stays None: an open item, never filled here)."""
    return entries[key]["value"]


# ---------------------------------------------------------------------------
# Measurements on the guide-shipped MS (all read-only)
# ---------------------------------------------------------------------------


def ms_meta(ms_path: Path) -> dict[str, Any]:
    with open_tb(ms_path) as tb:
        nrows = int(tb.nrows())
        cols = sorted(tb.colnames())
    with open_tb(ms_path / "FIELD") as tb:
        field_names = [str(x) for x in tb.getcol("NAME")]
    with open_tb(ms_path / "SPECTRAL_WINDOW") as tb:
        nchan = [int(x) for x in tb.getcol("NUM_CHAN")]
        ref_freq_ghz = [round(float(x) / 1e9, 4) for x in tb.getcol("REF_FREQUENCY")]
    with open_tb(ms_path / "HISTORY") as tb:
        history_nrows = int(tb.nrows())
    with open_msmd(ms_path) as msmd:
        intents = {
            n: sorted(str(x) for x in msmd.intentsforfield(i)) for i, n in enumerate(field_names)
        }
        scans = [int(s) for s in sorted(msmd.scannumbers())]
    return {
        "ms_path": str(ms_path),
        "size_mb": dir_size_mb(ms_path),
        "nrows": nrows,
        "columns": cols,
        "field_names": field_names,
        "field_intents": intents,
        "scans": scans,
        "spw_nchan": nchan,
        "spw_ref_freq_ghz": ref_freq_ghz,
        "history_nrows": history_nrows,
    }


def min_elevation_deg(ms_path: Path, field_name: str) -> float:
    from casatools import measures, quanta

    me = measures()
    qa = quanta()
    try:
        with open_msmd(ms_path) as msmd:
            fid = [str(x) for x in msmd.fieldnames()].index(field_name)
            pdir = msmd.phasecenter(fid)
            times = np.concatenate([msmd.timesforscan(s) for s in msmd.scansforfield(fid)])
        me.doframe(me.observatory("VLA"))
        els = []
        for t in np.unique(times):
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))
        return float(min(els))
    finally:
        with contextlib.suppress(Exception):
            me.done()
        with contextlib.suppress(Exception):
            qa.done()


def antenna_health(ms_path: Path) -> dict[str, Any]:
    """Per-antenna flagged fraction and median parallel-hand amplitude (cross-correlations)."""
    with open_tb(ms_path / "ANTENNA") as tb:
        names = [str(x) for x in tb.getcol("NAME")]
    nant = len(names)
    flagged = np.zeros(nant)
    total = np.zeros(nant)
    amp_samples: list[list[np.ndarray]] = [[] for _ in range(nant)]
    with open_tb(ms_path) as tb:
        nrow = int(tb.nrows())
        step = 200_000
        for start in range(0, nrow, step):
            n = min(step, nrow - start)
            a1 = tb.getcol("ANTENNA1", start, n)
            a2 = tb.getcol("ANTENNA2", start, n)
            flg = tb.getcol("FLAG", start, n)
            dat = tb.getcol("DATA", start, n)
            cross = a1 != a2
            par = flg[[0, 3]]
            nflg = par.reshape(-1, n).sum(axis=0)
            nvis = par.shape[0] * par.shape[1]
            amp = np.where(par, np.nan, np.abs(dat[[0, 3]]))
            row_med = np.nanmedian(amp.reshape(-1, n), axis=0)
            for ant in range(nant):
                m = cross & ((a1 == ant) | (a2 == ant))
                if not np.any(m):
                    continue
                flagged[ant] += float(nflg[m].sum())
                total[ant] += float(m.sum() * nvis)
                vals = row_med[m]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    amp_samples[ant].append(vals[:: max(1, vals.size // 2000)])
    out: dict[str, Any] = {}
    meds = []
    for ant, name in enumerate(names):
        frac = float(flagged[ant] / total[ant]) if total[ant] > 0 else 1.0
        med = float(np.median(np.concatenate(amp_samples[ant]))) if amp_samples[ant] else 0.0
        out[name] = {"flagged_fraction": round(frac, 6), "median_amp_jy": round(med, 6)}
        meds.append(med)
    arr_med = float(np.median([m for m in meds if m > 0])) if any(m > 0 for m in meds) else 0.0
    # Dead-antenna rule copied from the 24A-376 inspect oracle (median amp < 0.05 x array median),
    # plus antennas whose cross-correlations are fully flagged.
    dead = sorted(
        n
        for n, v in out.items()
        if v["flagged_fraction"] >= 0.999 or v["median_amp_jy"] < 0.05 * arr_med
    )
    return {"per_antenna": out, "array_median_amp_jy": round(arr_med, 6), "dead_antennas": dead}


def field_pol_summary(ms_path: Path, column: str) -> dict[str, Any]:
    """Vector-averaged per-spw I, Q/I, U/I and EVPA per calibrator field (circular feeds).

    Q = Re(RL + LR) / 2, U = Im(RL - LR) / 2, I = (RR + LL) / 2, over unflagged cross-correlations.
    """
    with open_tb(ms_path / "FIELD") as tb:
        fnames = [str(x) for x in tb.getcol("NAME")]
    with open_tb(ms_path / "DATA_DESCRIPTION") as tb:
        dd_spw = [int(x) for x in tb.getcol("SPECTRAL_WINDOW_ID")]
    out: dict[str, Any] = {}
    with open_tb(ms_path) as tb:
        for fname in (FLUX_CAL, UNPOL_CAL, PHASE_CAL):
            fid = fnames.index(fname)
            per_spw: dict[str, Any] = {}
            for dd, spw in enumerate(dd_spw):
                sub = tb.query(f"FIELD_ID=={fid} && DATA_DESC_ID=={dd} && ANTENNA1!=ANTENNA2")
                try:
                    if int(sub.nrows()) == 0:
                        continue
                    vis = sub.getcol(column)
                    flg = sub.getcol("FLAG")
                finally:
                    sub.close()
                good = ~np.any(flg, axis=0)
                if not np.any(good):
                    continue
                rr, rl, lr, ll = (vis[i][good].mean() for i in range(4))
                stokes_i = 0.5 * (rr + ll).real
                q = 0.5 * (rl + lr).real
                u = 0.5 * (rl - lr).imag
                per_spw[str(spw)] = {
                    "I_jy": round(float(stokes_i), 6),
                    "q_frac": round(float(q / stokes_i), 6),
                    "u_frac": round(float(u / stokes_i), 6),
                    "p_frac": round(float(np.hypot(q, u) / stokes_i), 6),
                    "evpa_deg": round(float(np.degrees(0.5 * np.arctan2(u, q))), 3),
                }
            out[fname] = per_spw
    return out


def caltable_amp_stats(path: Path) -> dict[str, Any]:
    with open_tb(path) as tb:
        cols = tb.colnames()
        val = np.abs(tb.getcol("CPARAM")) if "CPARAM" in cols else tb.getcol("FPARAM")
        flg = tb.getcol("FLAG")
    good = val[~flg]
    if good.size == 0:
        return {"n_good": 0}
    return {
        "n_good": int(good.size),
        "median": round(float(np.median(good)), 6),
        "p95": round(float(np.percentile(good, 95)), 6),
        "max": round(float(np.max(good)), 6),
        "flagged_fraction": round(float(np.mean(flg)), 6),
    }


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------


def build_raw(src_ms: Path, root: Path) -> Path:
    import casatasks

    raw_ms = root / "raw" / "input.ms"
    if raw_ms.exists():
        shutil.rmtree(raw_ms)
    raw_ms.parent.mkdir(parents=True, exist_ok=True)
    casatasks.mstransform(
        vis=str(src_ms),
        outputvis=str(raw_ms),
        datacolumn="data",
        chanaverage=True,
        chanbin=CHANBIN,
        timeaverage=True,
        timebin=TIMEBIN,
        keepflags=True,
        reindex=True,
    )
    clear_history(raw_ms)
    assert_unlocked(raw_ms)
    return raw_ms


def build_polcal(src_ms: Path, root: Path, ref: dict[str, Any]) -> dict[str, Any]:
    """Guide sections 'Additional Flagging' through 'Applying the Calibration', at full resolution."""
    import casatasks

    work = root / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    vis = work / SRC_MS_NAME
    shutil.copytree(src_ms, vis)
    v = str(vis)

    # Additional Flagging (guide).
    casatasks.flagdata(
        vis=v,
        flagbackup=False,
        mode="manual",
        antenna="ea12",
        scan="17",
        timerange="07:25:57~07:26:18",
    )
    casatasks.flagdata(
        vis=v,
        mode="tfcrop",
        field=CAL_FIELDS,
        correlation="",
        freqfit="line",
        extendflags=False,
        flagbackup=False,
    )
    casatasks.flagdata(
        vis=v,
        mode="rflag",
        datacolumn="data",
        field=CAL_FIELDS,
        correlation="RL,LR",
        extendflags=True,
        flagbackup=False,
    )

    # Setting the Polarization Calibrator Models (guide values from REFERENCE.json).
    casatasks.setjy(
        vis=v,
        field=FLUX_CAL,
        spw="",
        selectdata=False,
        scalebychan=True,
        standard="manual",
        fluxdensity=[rv(ref, "setjy_3c48_I_jy"), 0, 0, 0],
        spix=rv(ref, "setjy_3c48_spix"),
        reffreq=rv(ref, "setjy_3c48_reffreq"),
        polindex=rv(ref, "setjy_3c48_polindex"),
        polangle=rv(ref, "setjy_3c48_polangle"),
        rotmeas=0,
        usescratch=True,
    )
    casatasks.setjy(
        vis=v,
        field=UNPOL_CAL,
        spw="",
        selectdata=False,
        scalebychan=True,
        standard="manual",
        fluxdensity=[rv(ref, "fluxboot_j2355_I_jy"), 0, 0, 0],
        spix=rv(ref, "fluxboot_j2355_spix"),
        reffreq=rv(ref, "fluxboot_reffreq"),
        polindex=[],
        polangle=[],
        rotmeas=0,
        usescratch=True,
    )
    casatasks.setjy(
        vis=v,
        field=PHASE_CAL,
        spw="",
        selectdata=False,
        scalebychan=True,
        standard="manual",
        fluxdensity=[rv(ref, "fluxboot_j0259_I_jy"), 0, 0, 0],
        spix=rv(ref, "fluxboot_j0259_spix"),
        reffreq=rv(ref, "fluxboot_reffreq"),
        polindex=[],
        polangle=[],
        rotmeas=0,
        usescratch=True,
    )

    tabs = root / "polcal_tables"
    if tabs.exists():
        shutil.rmtree(tabs)
    tabs.mkdir(parents=True)
    kcross = str(tabs / "TDRW0001_calibrated.Kcross_mbd")
    dtab_j2355 = str(tabs / "TDRW0001_calibrated.Df")
    dtab_j0259 = str(tabs / "TDRW0001_calibrated.DfQU")
    xtab = str(tabs / "TDRW0001_calibrated.Xf")
    spwmap8 = [0, 0, 0, 0, 0, 0, 0, 0]

    # Solving for the Cross-Hand delays (multiband, the solution the guide applies).
    casatasks.gaincal(
        vis=v,
        caltable=kcross,
        field=FLUX_CAL,
        spw="0~7:5~58",
        refant=REFANT,
        gaintype="KCROSS",
        solint="inf",
        combine="scan,spw",
        calmode="ap",
        append=False,
        gaintable=[""],
        gainfield=[""],
        interp=[""],
        spwmap=[[]],
        parang=True,
    )
    # Solving for the Leakage Terms.
    casatasks.polcal(
        vis=v,
        caltable=dtab_j2355,
        field=UNPOL_CAL,
        spw="0~7",
        refant=REFANT,
        poltype="Df",
        solint="inf,2MHz",
        combine="scan",
        gaintable=[kcross],
        gainfield=[""],
        spwmap=[spwmap8],
        append=False,
    )
    casatasks.polcal(
        vis=v,
        caltable=dtab_j0259,
        intent="CALIBRATE_POL_LEAKAGE#UNSPECIFIED",
        spw="0~7",
        refant=REFANT,
        poltype="Df+QU",
        solint="inf,2MHz",
        combine="scan",
        gaintable=[kcross],
        gainfield=[""],
        spwmap=[spwmap8],
        append=False,
    )
    for dtab in (dtab_j2355, dtab_j0259):
        casatasks.flagdata(
            vis=dtab,
            mode="clip",
            correlation="ABS_ALL",
            clipminmax=[0.0, 0.25],
            datacolumn="CPARAM",
            clipoutside=True,
            action="apply",
            flagbackup=False,
            savepars=False,
        )
    # Solving for the R-L polarization angle.
    casatasks.polcal(
        vis=v,
        caltable=xtab,
        spw="0~7",
        field=FLUX_CAL,
        solint="inf,2MHz",
        combine="scan",
        poltype="Xf",
        refant=REFANT,
        gaintable=[kcross, dtab_j0259],
        gainfield=["", ""],
        spwmap=[spwmap8, []],
        append=False,
    )
    # Applying the Calibration (flagbackup=False: no .flagversions in the build area).
    casatasks.applycal(
        vis=v,
        field="",
        gainfield=["", "", ""],
        flagbackup=False,
        interp=["", "", ""],
        gaintable=[kcross, dtab_j0259, xtab],
        spw="0~7",
        calwt=[False, False, False],
        applymode="calflagstrict",
        antenna="*&*",
        spwmap=[spwmap8, [], []],
        parang=True,
    )

    with open_tb(kcross) as tb:
        kc_delay_ns = [round(float(x), 5) for x in np.ravel(tb.getcol("FPARAM"))]
    summary = {
        "kcross_mbd_fparam_ns": kc_delay_ns,
        "dterm_amp_df_j2355": caltable_amp_stats(Path(dtab_j2355)),
        "dterm_amp_dfqu_j0259": caltable_amp_stats(Path(dtab_j0259)),
        "corrected_pol_by_field_spw": field_pol_summary(vis, "CORRECTED_DATA"),
        "tables": [kcross, dtab_j2355, dtab_j0259, xtab],
    }

    # split + statwt (guide 'Applying the Calibration'), then the size-reduction average.
    post_ms = root / "post_polcal" / "input.ms"
    if post_ms.exists():
        shutil.rmtree(post_ms)
    post_ms.parent.mkdir(parents=True, exist_ok=True)
    casatasks.mstransform(
        vis=v,
        outputvis=str(post_ms),
        field=TARGET,
        datacolumn="corrected",
        chanaverage=True,
        chanbin=CHANBIN,
        timeaverage=True,
        timebin=TIMEBIN,
        keepflags=True,
        reindex=True,
    )
    casatasks.statwt(vis=str(post_ms), datacolumn="data", minsamp=8)
    clear_history(post_ms)
    assert_unlocked(post_ms)
    casatasks.clearstat()
    shutil.rmtree(work)
    return summary


def ground_truths(ref: dict[str, Any], raw_meas: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inspect_gt = {
        "raw_data_blake2b": None,
        "flux_calibrator": rv(ref, "field_flux_cal"),
        "bandpass_calibrator": rv(ref, "field_bandpass_cal"),
        "phase_calibrator": rv(ref, "field_phase_cal"),
        "pol_angle_calibrator": rv(ref, "field_pol_angle_cal"),
        "pol_leakage_calibrator": rv(ref, "field_pol_leakage_cal"),
        "secondary_pol_leakage_calibrator": rv(ref, "field_secondary_pol_leakage_cal"),
        "target_fields": [rv(ref, "field_target")],
        "dead_antennas": raw_meas["dead_antennas"],
        "bandpass_min_elevation_deg": raw_meas["bandpass_min_elevation_deg"],
        "tolerances": {"elevation_abs_deg": rv(ref, "tol_elevation_abs_deg")},
    }
    pol_reference = {
        k: rv(ref, k)
        for k in ref
        if ref[k].get("group") == "polarization" and ref[k].get("value") is not None
    }
    image_gt = {
        "raw_data_blake2b": None,
        "guide_reference": {
            "peak_jy_per_beam": rv(ref, "guide_final_peak_jy_per_beam"),
            "offsource_rms_jy_per_beam": rv(ref, "guide_final_offsource_rms_jy_per_beam"),
            "peak_initial_jy_per_beam": rv(ref, "guide_initial_peak_jy_per_beam"),
        },
        "tolerances": {
            "pixels_per_beam_min": rv(ref, "tol_pixels_per_beam_min"),
            "pixels_per_beam_max": rv(ref, "tol_pixels_per_beam_max"),
            "off_source_residual_rms_min_jy_per_beam": rv(ref, "tol_rms_floor_jy_per_beam"),
            "negative_model_bowl_ratio_max": rv(ref, "tol_negative_bowl_ratio_max"),
            "peak_rel_tol": rv(ref, "tol_peak_rel_tol"),
        },
        "polarization_reference": pol_reference,
    }
    e2e_gt = {
        "raw_data_blake2b": None,
        "inspect_ground_truth": dict(inspect_gt),
        "image_ground_truth": json.loads(json.dumps(image_gt)),
        "polarization_reference": pol_reference,
    }
    return {TASKS[0]: inspect_gt, TASKS[1]: image_gt, TASKS[2]: e2e_gt}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vla_3c75_pol checkpoints + ground truth")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    src_ms = (args.extracted / SRC_MS_NAME).resolve()
    if not (src_ms / "table.dat").is_file():
        raise SystemExit(f"Missing source MS {src_ms}")
    _setup_env(root)
    os.chdir(root)
    ref = load_reference()
    for const, key in (
        (FLUX_CAL, "field_flux_cal"),
        (UNPOL_CAL, "field_secondary_pol_leakage_cal"),
        (PHASE_CAL, "field_phase_cal"),
        (TARGET, "field_target"),
        (REFANT, "refant"),
    ):
        if rv(ref, key) != const:
            raise SystemExit(f"REFERENCE.json {key}={rv(ref, key)!r} != builder constant {const!r}")
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    raw_ms = build_raw(src_ms, root)
    timings["raw_s"] = round(time.perf_counter() - t0, 1)
    health = antenna_health(raw_ms)
    raw_meas = {
        "bandpass_min_elevation_deg": round(min_elevation_deg(raw_ms, FLUX_CAL), 4),
        "dead_antennas": health["dead_antennas"],
    }
    print(f"[1/3] raw checkpoint {raw_ms} in {timings['raw_s']} s: {raw_meas}", flush=True)

    t0 = time.perf_counter()
    pol_summary = build_polcal(src_ms, root, ref)
    timings["polcal_s"] = round(time.perf_counter() - t0, 1)
    print(f"[2/3] post_polcal checkpoint in {timings['polcal_s']} s", flush=True)

    for tid, gt in ground_truths(ref, raw_meas).items():
        write_json(root / tid / "ground_truth.json", gt)
    summary = {
        "suite_id": "vla_3c75_pol",
        "source_ms": str(src_ms),
        "chanbin": CHANBIN,
        "timebin": TIMEBIN,
        "raw": ms_meta(raw_ms),
        "post_polcal": ms_meta(root / "post_polcal" / "input.ms"),
        "raw_measurements": raw_meas,
        "antenna_health": health,
        "polcal_builder_measurements": pol_summary,
        "timings_s": timings,
    }
    write_json(root / "build_summary.json", summary)
    print(f"[3/3] ground truth written for {list(TASKS)}; summary {root / 'build_summary.json'}")


if __name__ == "__main__":
    main()
