"""Deterministic checkpoint + ground-truth builder for alma_3c286_band6_pol.

Reproduces the ALMA 3C286 Band 6 polarization CASA Guide (CASA 6.6.6 calibration page) up to
the Stokes I calibration, writes HISTORY-cleared checkpoints, measures the guide-shipped
reference images, and emits per-task ground_truth.json files.

Needs only casatools, casatasks, numpy. If radio_bench is importable it also computes the
raw-data BLAKE2b gate hash and runs the bench extractors on the oracle checkpoints.

Layout under --root (= <data_dir>/casaguides/alma_3c286_band6_pol):
  extracted/                  run.py fetch output (read-only here)
  work/                       importasdm / a-priori products (removed unless --keep-work)
  raw/input.ms                a-priori (Tsys + WVR) calibrated, 4 spw x 16 chan, no guide flags
  post_precal/input.ms        raw + guide manual flags + Ceres MODEL_DATA
  post_calibrate/input.ms     post_precal + guide Stokes I calibration in CORRECTED_DATA
  guide_calibrated/input.ms   guide-shipped 3c286_Band6.pol.cal.ms (DATA column only)
  <task_id>/ground_truth.json
  reference_measurements.json, summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SUITE_ID = "alma_3c286_band6_pol"
BASENAMES = ["uid___A002_X85c183_X10a", "uid___A002_X85c183_X51a", "uid___A002_X85c183_X822"]
REFANT = "DV23"
POL_CAL = "J1337-1257"
BP_CAL = "J1256-0547"
FLUX_CAL = "Ceres"
PHASE_CAL = "J1310+3220"
TARGET = "3c286"
GAIN_FIELDS = f"{BP_CAL},{FLUX_CAL},{PHASE_CAL}"  # guide field = '1, 2, 3'
SCI_SPW = "17,19,21,23"
TSYS_EDGE_FLAG_SPW = "9:0~3, 11:0~3, 13:0~3, 15:0~3"
# Guide: tsysmap = tsysspwmap(...)[:25]; tsysmap[21] = 13; tsysmap[23] = 15. The 17->9 and 19->11
# entries are the same-baseband Tsys spws (listobs BBC_NO); asserted against BBC_NO at build time.
TSYS_SCI_MAP = {17: 9, 19: 11, 21: 13, 23: 15}
CHANBIN = 4  # 64 -> 16 channels; guide's own final split uses width='4'
# Guide flags on the 64-channel split: spw='*:0~3, *:60~63'. After chanbin=4 these are channels 0, 15.
EDGE_FLAG_SPW = "*:0;15"
# Guide gaincal G0ph uses spw='*:20~45' on 64 channels; nearest 16-channel bins are 5~11 (20~47).
G0PH_SPW = "*:5~11"
TARGET_RA = "13h31m08.288100s"  # guide listobs field 4: 13:31:08.288100 +30.30.32.96000
TARGET_DEC = "+30d30m32.96000s"

# Guide-quoted numbers (see REFERENCE.json for page, section, quote).
GUIDE = {
    "phase_cal_fitted_flux_jy": 0.897572,
    "bp_cal_fitted_flux_jy": 8.31993,
    "ceres_setjy_fluxd_jy": [1.53510869, 1.56220722, 1.72976518, 1.75838304],
    "initial_image_rms_jy": 0.001051827310962827,
    "initial_image_peak_jy": 0.2975700795650482,
    "initial_image_dr": 282.90773253706163,
    "final_image_rms_jy": 0.0001295028028448405,
    "final_image_peak_jy": 0.33380603790283203,
    "final_image_dr": 2577.597013886801,
    "target_I_mjy": 351.0,
    "target_Q_mjy": 11.9,
    "target_U_mjy": 56.2,
    "target_V_mjy": -0.49,
    "target_pol_int_mjy": 57.5,
    "target_pol_frac_pct": 16.4,
    "target_pol_frac_err_pct": 0.049,
    "target_evpa_deg": 39.05,
    "target_evpa_err_deg": 0.025,
    "target_evpa_sys_err_deg": 2.0,
    "polcal_qu_spwave_before": [0.014114697176655198, 0.04039249521486836],
    "polcal_qu_spwave_after": [0.0005853405152922923, -0.00031083130393098516],
}

# Tolerance factors copied from existing suites (REFERENCE.json names suite + key).
CONV = {
    "elevation_abs_deg": 0.25,  # vla_3c391_cband r1_3c391_inspect tolerances.elevation_abs_deg
    "bp_channel_phase_rms_max_deg": 8.0,  # vla_3c391_cband j1331_channel_phase_rms_max_deg
    "phase_cal_flux_rel_tol": 0.08,  # vla_3c391_cband j1822_flux_rel_tol == vla_24a376 value
    "verifier_dr_factor": 0.70,  # vla_24a376 builder: max(150, 0.70 * ver_dr)
    "verifier_dr_floor": 150.0,
    "retention_min_offset": 0.20,  # vla_24a376 builder: max(0.45, tgt_ret - 0.20)
    "retention_min_floor": 0.45,
    "retention_max": 0.98,  # vla_24a376 builder target_unflagged_rows_max
    "pixels_per_beam_min": 3.0,  # vla_3c391_cband r1_3c391_image tolerances
    "pixels_per_beam_max": 7.0,
    "rms_max_factor": 2.0,  # vla_24a376 builder: ref_rms * 2.0
    "rms_min_jy": 0.00004,  # vla_24a376 builder off_source_residual_rms_min_jy_per_beam
    "dr_min_factor": 0.55,  # vla_24a376 builder: max(150, 0.55 * ref_dr)
    "dr_min_floor": 150.0,
    "negative_model_bowl_ratio_max": 0.12,  # both suites
}

VERIFIER_TCLEAN_RECIPE: dict[str, Any] = {
    "field": TARGET,
    "spw": "",
    "specmode": "mfs",
    "gridder": "standard",
    "deconvolver": "clark",
    "imsize": [250, 250],
    "cell_arcsec": 0.1,
    "weighting": "briggs",
    "robust": 0.5,
    "pblimit": 0.2,
    "usemask": "pb",
    "pbmask": 0.2,
    "niter": 500,
    "threshold": "0.0mJy",
    "pbcor": True,
    "timeout_sec": 1800.0,
}

TASK_IDS = {
    "inspect": "g1_3c286pol_inspect",
    "calibrate": "g1_3c286pol_calibrate",
    "image": "g1_3c286pol_image",
    "e2e": "g2_3c286pol_e2e",
}


# ---------------------------------------------------------------------------
# CASA helpers (every tool handle closed in finally)
# ---------------------------------------------------------------------------


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


def clear_rows(table_path: Path) -> None:
    if (table_path / "table.dat").is_file():
        with open_tb(table_path, nomodify=False) as tb:
            n = int(tb.nrows())
            if n > 0:
                tb.removerows(list(range(n)))


def scrub_checkpoint(ms_path: Path) -> None:
    """Clear HISTORY (as the existing builders do) and empty POINTING (size; unused by tclean)."""
    clear_rows(ms_path / "HISTORY")
    clear_rows(ms_path / "POINTING")


def assert_unlocked(ms_path: Path) -> None:
    import casatasks

    casatasks.clearstat()
    with open_tb(ms_path) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active table lock on {ms_path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rmtree_if(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def fixsyscaltimes(vis: Path, newinterval: float = 2.0) -> bool:
    """Same algorithm as casarecipes.almahelpers.fixsyscaltimes (CASA 6.6.0, v1.4).

    casarecipes is not shipped with modular casatasks 6.7.6.14, so the guide's helper is
    reproduced here with numpy in place of pylab.
    """
    import numpy as np
    from casatools import table

    tb = table()
    fixed = False
    try:
        tb.open(str(vis / "SYSCAL"), nomodify=False)
        for ispw in np.unique(tb.getcol("SPECTRAL_WINDOW_ID")):
            st = tb.query(f"SPECTRAL_WINDOW_ID=={int(ispw)}", name="byspw")
            try:
                times = st.getcol("TIME")
                interval = st.getcol("INTERVAL")
                timestamps = times - interval / 2
                t0 = 86400.0 * math.floor(timestamps[0] / 86400.0)
                n_t = len(np.unique(times - t0))
                utimestamps = np.unique(np.floor(timestamps) - t0)
                if n_t == len(utimestamps):
                    continue
                fixed = True
                for uts in utimestamps:
                    mask = (np.floor(timestamps) - t0) == uts
                    if len(np.unique(times[mask])) > 1:
                        times[mask] = np.mean(times[mask] - interval[mask] / 2) + newinterval / 2
                        interval[mask] = newinterval
                        st.putcol("TIME", times)
                        st.putcol("INTERVAL", interval)
            finally:
                st.close()
    finally:
        with contextlib.suppress(Exception):
            tb.close()
        tb.done()
    return fixed


def build_tsys_spwmap(vis: Path) -> list[int]:
    with open_tb(vis / "SPECTRAL_WINDOW") as tb:
        bbc = [int(x) for x in tb.getcol("BBC_NO")]
    spwmap = list(range(25))
    for sci, tsys in TSYS_SCI_MAP.items():
        if bbc[sci] != bbc[tsys]:
            raise RuntimeError(
                f"Tsys spw {tsys} BBC {bbc[tsys]} != science spw {sci} BBC {bbc[sci]}"
            )
        spwmap[sci] = tsys
    return spwmap


def field_meta(ms_path: Path) -> dict[str, Any]:
    from casatools import msmetadata

    md = msmetadata()
    try:
        md.open(str(ms_path))
        names = [str(x) for x in md.fieldnames()]
        out = {
            "field_names": names,
            "field_intents": {
                n: sorted(str(x) for x in md.intentsforfield(i)) for i, n in enumerate(names)
            },
            "n_scans": int(len(md.scannumbers())),
            "spw_nchan": [int(md.nchan(i)) for i in range(md.nspw())],
            "spw_ref_freq_ghz": [
                round(float(md.reffreq(i)["m0"]["value"]) / 1e9, 6) for i in range(md.nspw())
            ],
            "antenna_names": [str(x) for x in md.antennanames()],
        }
    finally:
        with contextlib.suppress(Exception):
            md.close()
        md.done()
    with open_tb(ms_path) as tb:
        out["nrows"] = int(tb.nrows())
        out["columns"] = sorted(tb.colnames())
    return out


def min_elevation_deg(ms_path: Path, field_name: str) -> float:
    import numpy as np
    from casatools import measures, msmetadata, quanta

    md = msmetadata()
    me = measures()
    qa = quanta()
    try:
        md.open(str(ms_path))
        me.doframe(me.observatory("ALMA"))
        fid = [str(x) for x in md.fieldnames()].index(field_name)
        pdir = md.phasecenter(fid)
        times = np.concatenate([md.timesforscan(int(s)) for s in md.scansforfield(fid)])
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))
        return float(min(els))
    finally:
        with contextlib.suppress(Exception):
            md.close()
        md.done()
        me.done()
        qa.done()


def field_corrected_amp_jy(ms_path: Path, field_name: str) -> float | None:
    """Mean over unflagged parallel-hand (XX, YY) CORRECTED_DATA of one field, per pol then averaged."""
    import numpy as np

    with open_tb(ms_path / "FIELD") as tb:
        fid = [str(x) for x in tb.getcol("NAME")].index(field_name)
    with open_tb(ms_path) as tb:
        sub = tb.query(f"FIELD_ID=={fid} && ANTENNA1!=ANTENNA2", columns="CORRECTED_DATA,FLAG")
        try:
            sums = [0j, 0j]
            cnts = [0, 0]
            for start in range(0, int(sub.nrows()), 50000):
                n = min(50000, int(sub.nrows()) - start)
                cd = sub.getcol("CORRECTED_DATA", start, n)
                fl = sub.getcol("FLAG", start, n)
                for k, p in enumerate((0, 3)):
                    good = ~fl[p]
                    sums[k] += complex(np.sum(cd[p][good]))
                    cnts[k] += int(np.sum(good))
        finally:
            sub.close()
    amps = [abs(s / c) for s, c in zip(sums, cnts, strict=True) if c > 0]
    return float(np.mean(amps)) if amps else None


# ---------------------------------------------------------------------------
# Reference-image measurements (guide imaging page recipe on the guide-shipped images)
# ---------------------------------------------------------------------------


def measure_reference_images(ref_dir: Path) -> dict[str, Any]:
    import casatasks
    from casatools import image

    out: dict[str, Any] = {"reference_dir": str(ref_dir)}
    for stokes in ("I", "Q", "U", "V"):
        name = ref_dir / f"3c286.Stokes{stokes}.image"
        rms = casatasks.imstat(imagename=str(name), box="45,45,95,95")["rms"][0]
        peak = casatasks.imstat(imagename=str(name))["max"][0]
        fit = casatasks.imfit(imagename=str(name), box="110,110,145,145")
        idx = "IQUV".index(stokes)
        comp = fit["results"]["component0"]["flux"]
        out[f"stokes_{stokes}"] = {
            "box_rms_jy": float(rms),
            "peak_jy": float(peak),
            "imfit_flux_jy": float(comp["value"][idx]),
            "imfit_flux_err_jy": float(comp["error"][idx]),
        }
    poli = ref_dir / "3c286.POLI.image"
    out["poli_box_rms_jy"] = float(
        casatasks.imstat(imagename=str(poli), box="45,45,95,95")["rms"][0]
    )
    out["poli_peak_jy"] = float(casatasks.imstat(imagename=str(poli))["max"][0])
    f_i = out["stokes_I"]["imfit_flux_jy"]
    f_q = out["stokes_Q"]["imfit_flux_jy"]
    f_u = out["stokes_U"]["imfit_flux_jy"]
    f_pi = math.sqrt(f_q**2 + f_u**2)
    out["pol_int_jy"] = f_pi
    out["pol_frac"] = f_pi / f_i
    out["evpa_deg"] = 0.5 * math.degrees(math.atan2(f_u, f_q))
    ia = image()
    try:
        ia.open(str(ref_dir / "3c286.StokesI.image"))
        rb = ia.restoringbeam()
        out["beam_major_arcsec"] = float(rb["major"]["value"])
        out["beam_minor_arcsec"] = float(rb["minor"]["value"])
        out["beam_pa_deg"] = float(rb["positionangle"]["value"])
        out["cell_arcsec"] = (
            abs(float(ia.coordsys().increment()["numeric"][0])) * 180 / math.pi * 3600
        )
    finally:
        with contextlib.suppress(Exception):
            ia.close()
        ia.done()
    return out


# ---------------------------------------------------------------------------
# Build stages
# ---------------------------------------------------------------------------


def stage_apriori(extracted: Path, work: Path) -> Path:
    """Guide calibration page: import, a-priori flags, Tsys, WVR, concat, split (spw 17,19,21,23)."""
    import casatasks

    asdm_dir = extracted / "3C286_Band6_UncalibratedData"
    work.mkdir(parents=True, exist_ok=True)
    mss = []
    for name in BASENAMES:
        vis = work / f"{name}.ms"
        rmtree_if(vis)
        rmtree_if(work / f"{name}.ms.flagversions")
        casatasks.importasdm(asdm=str(asdm_dir / name), vis=str(vis), asis="*")
        casatasks.flagdata(vis=str(vis), mode="manual", autocorr=True, flagbackup=False)
        casatasks.flagdata(
            vis=str(vis),
            mode="manual",
            intent="*POINTING*,*SIDEBAND_RATIO*,*ATMOSPHERE*",
            flagbackup=False,
        )
        casatasks.flagdata(vis=str(vis), mode="shadow", flagbackup=False)
        fixsyscaltimes(vis)
        tsys = work / f"{name}.ms.tsys"
        wvr = work / f"{name}.ms.wvr"
        rmtree_if(tsys)
        rmtree_if(wvr)
        casatasks.gencal(vis=str(vis), caltable=str(tsys), caltype="tsys")
        casatasks.flagdata(vis=str(tsys), spw=TSYS_EDGE_FLAG_SPW, flagbackup=False)
        casatasks.wvrgcal(
            vis=str(vis),
            caltable=str(wvr),
            tie=[f"{TARGET},{PHASE_CAL}"],
            statsource=BP_CAL,
            wvrflag=["DA54"],
        )
        mss.append(vis)
    tsysmap = build_tsys_spwmap(mss[0])
    for vis in mss:
        tsys = f"{vis}.tsys"
        wvr = f"{vis}.wvr"
        for fname in (POL_CAL, BP_CAL, TARGET):
            casatasks.applycal(
                vis=str(vis),
                field=fname,
                spw=SCI_SPW,
                gaintable=[tsys, wvr],
                gainfield=[fname, fname],
                interp="linear",
                spwmap=[tsysmap, []],
                calwt=True,
                flagbackup=False,
            )
        casatasks.applycal(
            vis=str(vis),
            field=PHASE_CAL,
            spw=SCI_SPW,
            gaintable=[tsys, wvr],
            gainfield=[TARGET, PHASE_CAL],
            interp="linear",
            spwmap=[tsysmap, []],
            calwt=True,
            flagbackup=False,
        )
    casatasks.applycal(
        vis=str(mss[0]),
        field=FLUX_CAL,
        spw=SCI_SPW,
        gaintable=[f"{mss[0]}.tsys", f"{mss[0]}.wvr"],
        gainfield=[FLUX_CAL, FLUX_CAL],
        interp="linear",
        spwmap=[tsysmap, []],
        calwt=True,
        flagbackup=False,
    )
    concat_ms = work / "3c286_Band6_concat.ms"
    split_ms = work / "3c286_Band6.ms"
    rmtree_if(concat_ms)
    rmtree_if(split_ms)
    casatasks.concat(vis=[str(m) for m in mss], concatvis=str(concat_ms))
    casatasks.split(
        vis=str(concat_ms), outputvis=str(split_ms), datacolumn="corrected", spw=SCI_SPW
    )
    return split_ms


def stage_raw(split_ms: Path, raw_ms: Path) -> None:
    """Channel-average 64 -> 16 and drop (a-priori flagged) autocorrelations; no guide manual flags."""
    import casatasks

    rmtree_if(raw_ms)
    raw_ms.parent.mkdir(parents=True, exist_ok=True)
    casatasks.mstransform(
        vis=str(split_ms),
        outputvis=str(raw_ms),
        datacolumn="data",
        antenna="*&",
        chanaverage=True,
        chanbin=CHANBIN,
        keepflags=True,
    )
    scrub_checkpoint(raw_ms)
    assert_unlocked(raw_ms)


def apply_guide_flags(vis: Path) -> None:
    """Guide calibration page manual flags (channel ranges translated to the 16-channel grid)."""
    import casatasks

    v = str(vis)
    casatasks.flagdata(vis=v, spw=EDGE_FLAG_SPW, flagbackup=False)
    casatasks.flagdata(vis=v, antenna="DV21", flagbackup=False)
    casatasks.flagdata(vis=v, antenna="DA49", flagbackup=False)
    casatasks.flagdata(
        vis=v, antenna="DA45", field=f"{POL_CAL},{PHASE_CAL}", spw="1", flagbackup=False
    )
    casatasks.flagdata(vis=v, antenna="DA45", scan="48", field=POL_CAL, spw="2", flagbackup=False)
    casatasks.flagdata(vis=v, antenna="DV14", flagbackup=False)
    casatasks.flagdata(vis=v, antenna="DV04", scan="93", field=POL_CAL, flagbackup=False)
    casatasks.flagdata(vis=v, timerange="2014/07/02/00:25:00~2014/07/02/00:25:10", flagbackup=False)


def stage_post_precal(raw_ms: Path, pre_ms: Path) -> dict[str, Any]:
    import casatasks

    rmtree_if(pre_ms)
    pre_ms.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw_ms, pre_ms)
    apply_guide_flags(pre_ms)
    sj = casatasks.setjy(
        vis=str(pre_ms),
        field=FLUX_CAL,
        spw="0,1,2,3",
        standard="Butler-JPL-Horizons 2012",
        usescratch=True,
    )
    scrub_checkpoint(pre_ms)
    assert_unlocked(pre_ms)
    fluxd: dict[str, float] = {}
    for fval in (sj or {}).values():
        if isinstance(fval, dict):
            for skey, sval in fval.items():
                if isinstance(sval, dict) and "fluxd" in sval:
                    fluxd[str(skey)] = round(float(sval["fluxd"][0]), 6)
    return {"ceres_setjy_fluxd_jy": fluxd}


def stage_post_calibrate(pre_ms: Path, cal_dir: Path) -> dict[str, Any]:
    """Guide Stokes I calibration (G0ph, Bscan, G2ph, G2amp, fluxscale, applycal; G1 on pol cal)."""
    import casatasks

    rmtree_if(cal_dir)
    cal_dir.mkdir(parents=True, exist_ok=True)
    vis = cal_dir / "input.ms"
    shutil.copytree(pre_ms, vis)
    v = str(vis)
    t = {
        k: str(cal_dir / f"3c286_Band6.ms.{k}")
        for k in ("G0ph", "Bscan", "G2ph", "G2amp", "flux", "G1")
    }
    common = {"refant": REFANT, "refantmode": "strict"}
    casatasks.gaincal(
        vis=v,
        caltable=t["G0ph"],
        field=BP_CAL,
        gaintype="G",
        solint="int",
        calmode="p",
        spw=G0PH_SPW,
        smodel=[1, 0, 0, 0],
        **common,
    )
    casatasks.bandpass(
        vis=v,
        caltable=t["Bscan"],
        field=BP_CAL,
        solint="inf",
        combine="scan,obs",
        refant=REFANT,
        solnorm=True,
        gaintable=[t["G0ph"]],
        interp=["nearest"],
    )
    casatasks.gaincal(
        vis=v,
        caltable=t["G2ph"],
        gaintype="G",
        field=GAIN_FIELDS,
        solint="int",
        calmode="p",
        gaintable=[t["Bscan"]],
        interp=["nearest"],
        **common,
    )
    casatasks.gaincal(
        vis=v,
        caltable=t["G2amp"],
        field=GAIN_FIELDS,
        solint="int",
        gaintype="T",
        calmode="a",
        gaintable=[t["Bscan"], t["G2ph"]],
        interp=["nearest", "nearest"],
        **common,
    )
    fs = casatasks.fluxscale(
        vis=v,
        caltable=t["G2amp"],
        fluxtable=t["flux"],
        reference=FLUX_CAL,
        transfer=f"{BP_CAL},{PHASE_CAL}",
    )
    casatasks.applycal(
        vis=v,
        field=GAIN_FIELDS,
        calwt=True,
        gaintable=[t["Bscan"], t["G2ph"], t["flux"]],
        interp=["nearest", "linear", "linear"],
        parang=False,
        flagbackup=False,
    )
    casatasks.gaincal(
        vis=v,
        caltable=t["G1"],
        field=POL_CAL,
        solint="int",
        smodel=[1, 0, 0, 0],
        gaintable=t["Bscan"],
        interp="nearest",
        **common,
    )
    casatasks.applycal(
        vis=v,
        field=POL_CAL,
        calwt=True,
        gaintable=[t["Bscan"], t["G1"]],
        interp=["nearest", "linear"],
        parang=False,
        flagbackup=False,
    )
    # Stokes I subset of the guide's final target applycal (pol tables omitted, parang=False).
    casatasks.applycal(
        vis=v,
        field=TARGET,
        calwt=True,
        gaintable=[t["Bscan"], t["G2ph"], t["flux"]],
        gainfield=["", PHASE_CAL, PHASE_CAL],
        interp=["nearest", "linear", "linear"],
        parang=False,
        flagbackup=False,
    )
    scrub_checkpoint(vis)
    assert_unlocked(vis)
    fitted: dict[str, Any] = {}
    for val in (fs or {}).values():
        if isinstance(val, dict) and "fieldName" in val:
            fitted[str(val["fieldName"])] = {
                "fitFluxd": float(val["fitFluxd"]) if val.get("fitFluxd") is not None else None,
                "spw_fluxd": {
                    str(sk): round(float(sv["fluxd"][0]), 6)
                    for sk, sv in val.items()
                    if isinstance(sv, dict) and "fluxd" in sv
                },
            }
    return {
        "fluxscale": fitted,
        "phase_cal_corrected_amp_jy": field_corrected_amp_jy(vis, PHASE_CAL),
        "caltables": t,
    }


def stage_guide_calibrated(extracted: Path, out_ms: Path) -> None:
    """Guide-shipped calibrated target MS; DATA holds the split(datacolumn='corrected') output."""
    import casatasks

    src = extracted / "3C286_Band6_CalibratedData" / "3c286_Band6.pol.cal.ms"
    rmtree_if(out_ms)
    out_ms.parent.mkdir(parents=True, exist_ok=True)
    casatasks.mstransform(
        vis=str(src), outputvis=str(out_ms), datacolumn="data", antenna="*&", keepflags=True
    )
    scrub_checkpoint(out_ms)
    assert_unlocked(out_ms)


# ---------------------------------------------------------------------------
# Optional bench verification (only when radio_bench is importable)
# ---------------------------------------------------------------------------


def _extract_cfg(stages: list[str], *, use_tclean: bool) -> Any:
    from radio_bench.grading.schema import ExtractConfig

    geom: dict[str, Any] = {
        "dish_diameter_m": 12.0,
        "pb_mask_min": 0.20,
        "offsource_inner_exclusion_beams": 5.0,
        "source_apertures": [
            {
                "label": TARGET,
                "kind": "point",
                "ra": TARGET_RA,
                "dec": TARGET_DEC,
                "radius_arcsec": 1.5,
            }
        ],
    }
    if use_tclean:
        geom["verifier_tclean_recipe"] = VERIFIER_TCLEAN_RECIPE
    return ExtractConfig.model_validate(
        {
            "suite_id": SUITE_ID,
            "split": "reference",
            "telescope": "ALMA",
            "pol_basis": "linear",
            "stages": stages,
            "target_retention_excludes_dead_antenna": False,
            "roles": {
                "flux_cal": [FLUX_CAL],
                "bandpass_cal": [BP_CAL],
                "phase_cal": [PHASE_CAL],
                "pol_angle_cal": [POL_CAL],
                "pol_leakage_cal": [POL_CAL],
                "targets": [TARGET],
            },
            "geometry": geom,
        }
    )


def bench_verify(root: Path) -> dict[str, Any]:
    try:
        from radio_bench.grading.extractors.calibrate import extract_calibrate
        from radio_bench.grading.extractors.common import compute_raw_data_blake2b
    except ImportError:
        print(
            "radio_bench not importable: skipping gate hashes and extractor verification",
            flush=True,
        )
        return {}
    out: dict[str, Any] = {"raw_data_blake2b": {}}
    for ck in ("raw", "post_precal", "guide_calibrated"):
        out["raw_data_blake2b"][ck] = compute_raw_data_blake2b(root / ck / "input.ms")
    obs, errs = extract_calibrate(
        root / "post_calibrate", _extract_cfg(["calibrate"], use_tclean=True)
    )
    out["oracle_calibrate_observables"] = {k: v.value for k, v in obs.items()}
    out["oracle_calibrate_errors"] = errs
    return out


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def write_ground_truth(root: Path, summary: dict[str, Any]) -> None:
    ref = summary["reference_measurements"]
    hashes = summary.get("bench", {}).get("raw_data_blake2b", {})
    cal_obs = summary.get("bench", {}).get("oracle_calibrate_observables", {})
    oracle_ret = cal_obs.get("cal.target_row_retention")

    pol_reference = {
        "guide": {k: v for k, v in GUIDE.items() if k.startswith(("target_", "polcal_"))},
        "product": {
            "pol_frac": ref["pol_frac"],
            "evpa_deg": ref["evpa_deg"],
            "pol_int_jy": ref["pol_int_jy"],
            "poli_peak_jy": ref["poli_peak_jy"],
            "stokes": {s: ref[f"stokes_{s}"] for s in "IQUV"},
        },
        "pol_basis": "linear",
        "pol_leakage_calibrator": POL_CAL,
        "note": "Not graded yet: no polcal extractor exists. Kept for a future polcal stage.",
    }

    def with_hash(gt: dict[str, Any], ck: str) -> dict[str, Any]:
        if ck in hashes:
            gt["raw_data_blake2b"] = hashes[ck]
        return gt

    inspect_gt = with_hash(
        {
            "flux_calibrator": FLUX_CAL,
            "bandpass_calibrator": BP_CAL,
            "phase_calibrator": PHASE_CAL,
            "pol_leakage_calibrator": POL_CAL,
            "pol_angle_calibrator": POL_CAL,
            "target_fields": [TARGET],
            "mosaic_target_fields": [TARGET],
            "dead_antennas": [],
            "missing_vla_pads": [],
            "bandpass_min_elevation_deg": round(summary["raw"]["bandpass_min_elevation_deg"], 4),
            "tolerances": {"elevation_abs_deg": CONV["elevation_abs_deg"]},
        },
        "raw",
    )
    ver_dr_ref = GUIDE["initial_image_dr"]
    tol_cal: dict[str, Any] = {
        "bp_channel_phase_rms_max_deg": CONV["bp_channel_phase_rms_max_deg"],
        "phase_cal_flux_rel_tol": CONV["phase_cal_flux_rel_tol"],
        "verifier_tclean_dr_min": round(
            max(CONV["verifier_dr_floor"], CONV["verifier_dr_factor"] * ver_dr_ref), 1
        ),
        "target_unflagged_rows_max": CONV["retention_max"],
        "target_unflagged_rows_min": (
            round(
                max(CONV["retention_min_floor"], float(oracle_ret) - CONV["retention_min_offset"]),
                3,
            )
            if oracle_ret is not None
            else CONV["retention_min_floor"]
        ),
    }
    calibrate_gt = with_hash(
        {
            "phase_cal_expected_flux_jy": GUIDE["phase_cal_fitted_flux_jy"],
            "reference_image_summary": {
                "peak_image_jy_per_beam": GUIDE["initial_image_peak_jy"],
                "rms_image_jy_per_beam": GUIDE["initial_image_rms_jy"],
                "dynamic_range": GUIDE["initial_image_dr"],
            },
            "oracle_fluxscale": summary["calibrate"]["fluxscale"],
            "polarization_reference": pol_reference,
            "tolerances": tol_cal,
        },
        "post_precal",
    )
    ref_rms = GUIDE["final_image_rms_jy"]
    ref_dr = GUIDE["final_image_dr"]
    image_gt = with_hash(
        {
            "reference_image_summary": {
                "peak_image_jy_per_beam": GUIDE["final_image_peak_jy"],
                "offsource_rms_jy": ref_rms,
                "dynamic_range": ref_dr,
                "beam_major_arcsec": ref["beam_major_arcsec"],
                "beam_minor_arcsec": ref["beam_minor_arcsec"],
            },
            "polarization_reference": pol_reference,
            "tolerances": {
                "pixels_per_beam_min": CONV["pixels_per_beam_min"],
                "pixels_per_beam_max": CONV["pixels_per_beam_max"],
                "off_source_residual_rms_max_jy_per_beam": round(
                    CONV["rms_max_factor"] * ref_rms, 8
                ),
                "off_source_residual_rms_min_jy_per_beam": CONV["rms_min_jy"],
                "dynamic_range_min": round(
                    max(CONV["dr_min_floor"], CONV["dr_min_factor"] * ref_dr), 1
                ),
                "negative_model_bowl_ratio_max": CONV["negative_model_bowl_ratio_max"],
            },
        },
        "guide_calibrated",
    )
    e2e_gt = with_hash(
        {
            "calibrate_ground_truth": {
                k: v for k, v in calibrate_gt.items() if k != "raw_data_blake2b"
            },
            "image_ground_truth": {k: v for k, v in image_gt.items() if k != "raw_data_blake2b"},
            "polarization_reference": pol_reference,
        },
        "raw",
    )
    for key, gt in (
        ("inspect", inspect_gt),
        ("calibrate", calibrate_gt),
        ("image", image_gt),
        ("e2e", e2e_gt),
    ):
        write_json(root / TASK_IDS[key] / "ground_truth.json", gt)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--keep-work", action="store_true", help="Keep work/ (imports, concat, split).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    extracted = root / "extracted"
    plan = {
        "extracted": str(extracted),
        "work": str(root / "work"),
        "checkpoints": [
            str(root / c / "input.ms")
            for c in ("raw", "post_precal", "post_calibrate", "guide_calibrated")
        ],
        "ground_truth": [str(root / t / "ground_truth.json") for t in TASK_IDS.values()],
        "reference_images": str(extracted / "3C286_Band6_ReferenceImages"),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    for need in (
        extracted / "3C286_Band6_UncalibratedData",
        extracted / "3C286_Band6_CalibratedData",
        extracted / "3C286_Band6_ReferenceImages",
    ):
        if not need.is_dir():
            raise SystemExit(f"missing {need}; run `run.py fetch` first")
    tmp = root / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp))
    os.chdir(tmp)

    timings: dict[str, float] = {}
    summary: dict[str, Any] = {"suite_id": SUITE_ID, "root": str(root)}

    t0 = time.perf_counter()
    split_ms = stage_apriori(extracted, root / "work")
    raw_ms = root / "raw" / "input.ms"
    stage_raw(split_ms, raw_ms)
    summary["raw"] = field_meta(raw_ms)
    summary["raw"]["bandpass_min_elevation_deg"] = min_elevation_deg(raw_ms, BP_CAL)
    timings["apriori_raw_s"] = round(time.perf_counter() - t0, 1)
    print(f"[1/5] raw checkpoint in {timings['apriori_raw_s']} s", flush=True)

    t0 = time.perf_counter()
    pre_ms = root / "post_precal" / "input.ms"
    summary["precal"] = stage_post_precal(raw_ms, pre_ms)
    timings["post_precal_s"] = round(time.perf_counter() - t0, 1)
    print(f"[2/5] post_precal in {timings['post_precal_s']} s: {summary['precal']}", flush=True)

    t0 = time.perf_counter()
    summary["calibrate"] = stage_post_calibrate(pre_ms, root / "post_calibrate")
    timings["post_calibrate_s"] = round(time.perf_counter() - t0, 1)
    print(
        f"[3/5] post_calibrate in {timings['post_calibrate_s']} s: {summary['calibrate']['fluxscale']}",
        flush=True,
    )

    t0 = time.perf_counter()
    gc_ms = root / "guide_calibrated" / "input.ms"
    stage_guide_calibrated(extracted, gc_ms)
    summary["guide_calibrated"] = field_meta(gc_ms)
    summary["reference_measurements"] = measure_reference_images(
        extracted / "3C286_Band6_ReferenceImages"
    )
    write_json(root / "reference_measurements.json", summary["reference_measurements"])
    timings["guide_calibrated_and_ref_s"] = round(time.perf_counter() - t0, 1)
    print(
        f"[4/5] guide_calibrated + reference measurements in {timings['guide_calibrated_and_ref_s']} s",
        flush=True,
    )

    summary["bench"] = bench_verify(root)
    write_ground_truth(root, summary)
    summary["timings_s"] = timings
    write_json(root / "summary.json", summary)
    if not args.keep_work:
        rmtree_if(root / "work")
    print(f"[5/5] ground truth written for {list(TASK_IDS.values())}", flush=True)


if __name__ == "__main__":
    main()
