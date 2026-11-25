"""Standalone CASA 6.7.6.14 oracle solver for g1_g55_calibrate (stages: calibrate)."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

import numpy as np

STAGES = ("calibrate",)
FLUX_CAL = "0542+498=3C147"
PHASE_CAL = "J1925+2106"
TARGET = "G55.7+3.4"
CAL_FIELDS = f"{PHASE_CAL},{FLUX_CAL}"
REFANT = "ea24"
INIT_PHASE_SPW = "0:20~24,1:49~52,2:38~41,3:41~44"


def _configure_temp_env(workspace: Path) -> Path:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur_tmp = os.environ.get("TMPDIR", "").strip()
    if not cur_tmp or cur_tmp == "/tmp" or cur_tmp.startswith("/tmp/"):
        var_tmp = Path("/var/tmp")
        base = var_tmp if (var_tmp.exists() and os.access(var_tmp, os.W_OK)) else workspace
        scratch = base / f".solver_scratch_{os.getpid()}"
    else:
        scratch = Path(cur_tmp) / f"solver_{os.getpid()}"
    (scratch / "mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["MPLCONFIGDIR"] = str(scratch / "mplconfig")
    return scratch


@contextlib.contextmanager
def _open_tb(path: Path, nomodify: bool = True):
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


def _rmtables(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            shutil.rmtree(p)


def repair_flux_intent(ms: Path) -> None:
    """The flux calibrator carries CALIBRATE_AMPLI only; add CALIBRATE_FLUX to its STATE rows."""
    with _open_tb(ms / "FIELD") as tb:
        flux_fid = [str(x) for x in tb.getcol("NAME")].index(FLUX_CAL)
    with _open_tb(ms) as tb:
        fid = np.asarray(tb.getcol("FIELD_ID"))
        sid = np.asarray(tb.getcol("STATE_ID"))
    flux_sids = sorted({int(s) for s in np.unique(sid[fid == flux_fid]) if s >= 0})
    other_sids = {int(s) for s in np.unique(sid[fid != flux_fid]) if s >= 0}
    with _open_tb(ms / "STATE", nomodify=False) as st:
        for s in flux_sids:
            if s in other_sids:
                continue
            mode = str(st.getcell("OBS_MODE", s))
            if "CALIBRATE_FLUX" not in mode:
                st.putcell(
                    "OBS_MODE",
                    s,
                    f"{mode},CALIBRATE_FLUX#UNSPECIFIED" if mode else "CALIBRATE_FLUX#UNSPECIFIED",
                )


def do_precal(ws: Path) -> None:
    import casatasks

    ms = ws / "input.ms"
    repair_flux_intent(ms)
    # Online flags (FLAG_CMD subtable), shadow, zero clip, quack
    casatasks.flagcmd(vis=str(ms), inpmode="table", reason="any", action="apply", flagbackup=False)
    casatasks.flagdata(vis=str(ms), mode="shadow", tolerance=0.0, flagbackup=False)
    casatasks.flagdata(vis=str(ms), mode="clip", clipzeros=True, flagbackup=False)
    casatasks.flagdata(
        vis=str(ms), mode="quack", quackinterval=5.0, quackmode="beg", flagbackup=False
    )
    # Automatic RFI excision: tfcrop on DATA, rflag on initial-bandpass-corrected data
    for spw in ("0", "1~3"):
        casatasks.flagdata(
            vis=str(ms),
            mode="tfcrop",
            spw=spw,
            datacolumn="data",
            action="apply",
            display="",
            flagbackup=False,
        )
    init_ph = ws / "precal_initPh.cal"
    init_bp = ws / "precal_initBP.cal"
    _rmtables(init_ph, init_bp)
    casatasks.gaincal(
        vis=str(ms),
        caltable=str(init_ph),
        field=PHASE_CAL,
        solint="int",
        spw=INIT_PHASE_SPW,
        refant=REFANT,
        minblperant=3,
        minsnr=3.0,
        calmode="p",
    )
    casatasks.bandpass(
        vis=str(ms),
        caltable=str(init_bp),
        field=PHASE_CAL,
        solint="inf",
        combine="scan",
        refant=REFANT,
        minblperant=3,
        minsnr=10.0,
        gaintable=[str(init_ph)],
        interp=["nearest"],
        solnorm=False,
    )
    casatasks.applycal(vis=str(ms), gaintable=[str(init_bp)], calwt=False, flagbackup=False)
    for spw, fdev, tdev in (("0", 2.5, 3.5), ("1,3", 4.0, 4.0), ("2", 2.5, 2.5)):
        casatasks.flagdata(
            vis=str(ms),
            mode="rflag",
            spw=spw,
            datacolumn="corrected",
            freqdevscale=fdev,
            timedevscale=tdev,
            action="apply",
            display="",
            flagbackup=False,
        )
        casatasks.flagdata(
            vis=str(ms),
            mode="extend",
            spw=spw,
            extendpols=True,
            action="apply",
            display="",
            flagbackup=False,
        )
        casatasks.flagdata(
            vis=str(ms),
            mode="extend",
            spw=spw,
            growtime=50.0,
            growfreq=90.0,
            action="apply",
            display="",
            flagbackup=False,
        )
    casatasks.flagdata(vis=str(ms), scan="87,88", antenna="ea16", flagbackup=False)
    for bl in ("ea04&ea16", "ea02&ea27", "ea02&ea08"):
        casatasks.flagdata(vis=str(ms), antenna=bl, spw="1", flagbackup=False)
    with _open_tb(ms, nomodify=False) as tb:
        if "CORRECTED_DATA" in tb.colnames():
            tb.removecols(["CORRECTED_DATA"])
    casatasks.setjy(vis=str(ms), field=FLUX_CAL, standard="Perley-Butler 2017", usescratch=True)
    cal_ms = ws / "calibrators.ms"
    _rmtables(cal_ms)
    casatasks.mstransform(
        vis=str(ms), outputvis=str(cal_ms), field=CAL_FIELDS, datacolumn="all", keepflags=False
    )
    casatasks.clearstat()


def do_calibrate(ws: Path) -> None:
    import casatasks

    ms = ws / "input.ms"
    cal_ms = ws / "calibrators.ms"
    gc = ws / "gain_curves.gc"
    opac = ws / "opacities.opac"
    g0 = ws / "G0.cal"
    k_tab = ws / "K.cal"
    b_tab = ws / "B.cal"
    g_tab = ws / "G.cal"
    flux_tab = ws / "fluxscale.cal"
    _rmtables(gc, opac, g0, k_tab, b_tab, g_tab, flux_tab)
    # Prior calibrations: gain curves and opacities (no SYSPOWER table, so no requantizer gains)
    casatasks.gencal(vis=str(cal_ms), caltable=str(gc), caltype="gc")
    taus = casatasks.plotweather(vis=str(cal_ms), doPlot=False)
    casatasks.gencal(
        vis=str(cal_ms), caltable=str(opac), caltype="opac", spw="0,1,2,3", parameter=taus
    )
    priors = [str(gc), str(opac)]
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(g0),
        field=FLUX_CAL,
        spw=INIT_PHASE_SPW,
        refant=REFANT,
        calmode="p",
        solint="int",
        minsnr=3.0,
        gaintable=priors,
    )
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(k_tab),
        field=FLUX_CAL,
        refant=REFANT,
        gaintype="K",
        solint="inf",
        combine="scan",
        minsnr=3.0,
        gaintable=[*priors, str(g0)],
    )
    casatasks.bandpass(
        vis=str(cal_ms),
        caltable=str(b_tab),
        field=FLUX_CAL,
        refant=REFANT,
        solint="inf",
        combine="scan",
        solnorm=True,
        minsnr=3.0,
        gaintable=[*priors, str(g0), str(k_tab)],
    )
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(g_tab),
        field=CAL_FIELDS,
        refant=REFANT,
        gaintype="G",
        calmode="ap",
        solint="inf",
        minsnr=3.0,
        gaintable=[*priors, str(k_tab), str(b_tab)],
    )
    casatasks.fluxscale(
        vis=str(cal_ms),
        caltable=str(g_tab),
        fluxtable=str(flux_tab),
        reference=[FLUX_CAL],
        transfer=[PHASE_CAL],
        incremental=False,
    )
    tables = [*priors, str(k_tab), str(b_tab), str(flux_tab)]
    gf_flux = ["", "", FLUX_CAL, FLUX_CAL, FLUX_CAL]
    gf_phase = ["", "", FLUX_CAL, FLUX_CAL, PHASE_CAL]
    casatasks.applycal(
        vis=str(ms),
        field=FLUX_CAL,
        gaintable=tables,
        gainfield=gf_flux,
        applymode="calflag",
        flagbackup=False,
    )
    for fld in (PHASE_CAL, TARGET):
        casatasks.applycal(
            vis=str(ms),
            field=fld,
            gaintable=tables,
            gainfield=gf_phase,
            applymode="calflag",
            flagbackup=False,
        )
    casatasks.flagdata(
        vis=str(ms),
        field=TARGET,
        mode="rflag",
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )
    casatasks.clearstat()


def do_image(ws: Path) -> None:
    import casatasks

    ms = ws / "input.ms"
    with _open_tb(ms) as tb:
        datacolumn = "corrected" if "CORRECTED_DATA" in tb.colnames() else "data"
    prefix = ws / "g55_mtmfs_wproj"
    for p in ws.glob(f"{prefix.name}.*"):
        shutil.rmtree(p, ignore_errors=True)
    casatasks.tclean(
        vis=str(ms),
        imagename=str(prefix),
        field=TARGET,
        datacolumn=datacolumn,
        gridder="wproject",
        wprojplanes=-1,
        pblimit=-0.01,
        imsize=1280,
        cell="8arcsec",
        specmode="mfs",
        deconvolver="mtmfs",
        nterms=2,
        scales=[0, 6, 10, 30, 60],
        smallscalebias=0.9,
        interactive=False,
        niter=1000,
        weighting="briggs",
        stokes="I",
        threshold="0.15mJy",
        pbcor=True,
        savemodel="none",
    )
    casatasks.clearstat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    ws = parser.parse_args().workspace.resolve()
    scratch = _configure_temp_env(ws)
    prev = Path.cwd()
    try:
        os.chdir(ws)
        steps = {"precal": do_precal, "calibrate": do_calibrate, "image": do_image}
        for stage in STAGES:
            steps[stage](ws)
    finally:
        with contextlib.suppress(Exception):
            import casatasks

            casatasks.clearstat()
        for temp_lat in ws.glob("TempLattice*"):
            shutil.rmtree(temp_lat, ignore_errors=True)
        with contextlib.suppress(Exception):
            os.chdir(prev)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
