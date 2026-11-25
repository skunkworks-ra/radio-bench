"""Standalone CASA 6.7.6.14 oracle for g2_3c286pol_e2e (reference walkthrough, Stokes I).

Walkthrough manual flags (channel ranges on the 16-channel grid), Ceres flux model, Stokes I
calibration, split of the calibrated target, then the imaging self-cal sequence.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

REFANT = "DV23"
POL_CAL = "J1337-1257"
BP_CAL = "J1256-0547"
FLUX_CAL = "Ceres"
PHASE_CAL = "J1310+3220"
TARGET = "3c286"
GAIN_FIELDS = f"{BP_CAL},{FLUX_CAL},{PHASE_CAL}"
G0PH_SPW = "*:5~11"  # 64-channel '*:20~45' on the 16-channel grid
FINAL_PREFIX = "target_IQUV_selfcal_ap"
MASK = "circle[[125pix,125pix],1.5arcsec]"


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


def flag_and_model(workspace: Path) -> None:
    import casatasks

    v = str(workspace / "input.ms")
    casatasks.flagdata(vis=v, spw="*:0;15", flagbackup=False)  # 64-channel '*:0~3, *:60~63'
    casatasks.flagdata(vis=v, antenna="DV21", flagbackup=False)
    casatasks.flagdata(vis=v, antenna="DA49", flagbackup=False)
    casatasks.flagdata(
        vis=v, antenna="DA45", field=f"{POL_CAL},{PHASE_CAL}", spw="1", flagbackup=False
    )
    casatasks.flagdata(vis=v, antenna="DA45", scan="48", field=POL_CAL, spw="2", flagbackup=False)
    casatasks.flagdata(vis=v, antenna="DV14", flagbackup=False)
    casatasks.flagdata(vis=v, antenna="DV04", scan="93", field=POL_CAL, flagbackup=False)
    casatasks.flagdata(vis=v, timerange="2014/07/02/00:25:00~2014/07/02/00:25:10", flagbackup=False)
    casatasks.setjy(
        vis=v, field=FLUX_CAL, spw="0,1,2,3", standard="Butler-JPL-Horizons 2012", usescratch=True
    )


def calibrate_stokes_i(workspace: Path) -> None:
    import casatasks

    v = str(workspace / "input.ms")
    t = {k: str(workspace / f"cal.{k}") for k in ("G0ph", "Bscan", "G2ph", "G2amp", "flux", "G1")}
    for path in t.values():
        shutil.rmtree(path, ignore_errors=True)
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
    casatasks.fluxscale(
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
    casatasks.clearstat()


def _clean(vis: str, prefix: Path, niter: int, savemodel: str) -> None:
    import casatasks

    for p in prefix.parent.glob(prefix.name + ".*"):
        shutil.rmtree(p, ignore_errors=True)
    casatasks.tclean(
        vis=vis,
        imagename=str(prefix),
        cell=["0.1arcsec"],
        imsize=[250, 250],
        deconvolver="clarkstokes",
        stokes="IQUV",
        weighting="briggs",
        robust=0.5,
        niter=niter,
        usemask="user",
        mask=MASK,
        pbcor=True,
        savemodel=savemodel,
        interactive=False,
    )


def selfcal_and_image(workspace: Path, vis_path: Path) -> None:
    import casatasks

    vis = str(vis_path)
    work = workspace / "selfcal_work"
    work.mkdir(exist_ok=True)
    g1p, g2p, gap = (str(work / n) for n in ("G1p", "G2p", "Gap"))
    for path in (g1p, g2p, gap):
        shutil.rmtree(path, ignore_errors=True)
    sc = {"refant": REFANT, "refantmode": "strict", "minblperant": 4, "minsnr": 4}

    _clean(vis, work / "round0", 500, "modelcolumn")
    casatasks.gaincal(vis=vis, caltable=g1p, solint="300s", calmode="p", **sc)
    casatasks.applycal(vis=vis, gaintable=[g1p], flagbackup=False)
    _clean(vis, work / "round1", 1000, "modelcolumn")
    casatasks.gaincal(vis=vis, caltable=g2p, solint="30s", calmode="p", **sc)
    casatasks.applycal(vis=vis, gaintable=[g2p], flagbackup=False)
    _clean(vis, work / "round2", 1000, "modelcolumn")
    casatasks.gaincal(vis=vis, caltable=gap, gaintable=[g2p], solint="30s", calmode="ap", **sc)
    casatasks.applycal(vis=vis, gaintable=[g2p, gap], flagbackup=False)
    _clean(vis, workspace / FINAL_PREFIX, 1000, "none")
    casatasks.clearstat()
    # Keep only the final image set so the delivered product is unambiguous.
    for p in work.glob("round*"):
        shutil.rmtree(p, ignore_errors=True)


def run_solver(workspace: Path) -> None:
    import casatasks

    flag_and_model(workspace)
    calibrate_stokes_i(workspace)
    target_ms = workspace / "target_calibrated.ms"
    shutil.rmtree(target_ms, ignore_errors=True)
    casatasks.split(
        vis=str(workspace / "input.ms"),
        outputvis=str(target_ms),
        field=TARGET,
        datacolumn="corrected",
    )
    selfcal_and_image(workspace, target_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    ws = parser.parse_args().workspace.resolve()
    scratch = _configure_temp_env(ws)
    prev = Path.cwd()
    try:
        os.chdir(ws)
        run_solver(ws)
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
