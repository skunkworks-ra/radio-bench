"""Standalone CASA 6.7.6.14 oracle for g1_3c75pol_image.

Reproduces the reference guide's imaging chain on the polarization-calibrated, target-only MS:
initial IQUV mtmfs clean -> residual flagging -> Stokes I clean -> one phase + bandpass
self-calibration round -> statwt -> final IQUV mtmfs image (800 px, 3.4 arcsec).
Non-interactive deviations from the guide: auto-multithresh masking replaces interactive masks,
nsigma=3 stops each clean, pblimit=1e-4 with pbcor=True replaces widebandpbcor.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

REFANT = "ea10"
FINAL_PREFIX = "3C75_final"


def _configure_temp_env(workspace: Path) -> tuple[Path, bool]:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur_tmp = os.environ.get("TMPDIR", "").strip()
    if not cur_tmp or cur_tmp == "/tmp" or cur_tmp.startswith("/tmp/"):
        var_tmp = Path("/var/tmp")
        base = var_tmp if (var_tmp.exists() and os.access(var_tmp, os.W_OK)) else workspace
        scratch = base / f".solver_scratch_{os.getpid()}"
    else:
        scratch = Path(cur_tmp) / f"solver_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["MPLCONFIGDIR"] = str(scratch / "mplconfig")
    (scratch / "mplconfig").mkdir(parents=True, exist_ok=True)
    return scratch, True


def _tclean(
    vis: str,
    imagename: str,
    *,
    stokes: str,
    datacolumn: str,
    imsize: int,
    niter: int,
    cycleniter: int,
    savemodel: str,
) -> None:
    import casatasks

    casatasks.tclean(
        vis=vis,
        imagename=imagename,
        datacolumn=datacolumn,
        imsize=imsize,
        cell="3.4arcsec",
        stokes=stokes,
        specmode="mfs",
        reffreq="3.0GHz",
        gridder="standard",
        pblimit=0.0001,
        normtype="flatnoise",
        deconvolver="mtmfs",
        scales=[0, 6, 18],
        nterms=2,
        smallscalebias=0.6,
        weighting="briggs",
        robust=0.5,
        niter=niter,
        gain=0.1,
        nsigma=3.0,
        cycleniter=cycleniter,
        usemask="auto-multithresh",
        sidelobethreshold=2.0,
        noisethreshold=4.25,
        lownoisethreshold=1.5,
        minbeamfrac=0.3,
        pbcor=True,
        savemodel=savemodel,
        interactive=False,
    )


def image_target(vis_path: Path, workspace: Path) -> None:
    """Guide sections 'Multi-scale Clean', 'Self-Calibration', 'Final Polarization Images'."""
    import casatasks

    vis = str(vis_path)
    work = workspace / "_imaging_work"
    work.mkdir(exist_ok=True)

    _tclean(
        vis,
        str(work / "3C75_initial"),
        stokes="IQUV",
        datacolumn="data",
        imsize=480,
        niter=20000,
        cycleniter=1000,
        savemodel="modelcolumn",
    )
    for corr in ("ABS_RR,ABS_LL", "ABS_RL,ABS_LR"):
        casatasks.flagdata(
            vis=vis,
            mode="tfcrop",
            correlation=corr,
            freqfit="line",
            extendflags=False,
            flagbackup=False,
            datacolumn="residual_data",
            flagdimension="freq",
            ntime="scan",
        )
    for corr in ("RR,LL", "RL,LR"):
        casatasks.flagdata(
            vis=vis,
            mode="rflag",
            correlation=corr,
            extendflags=False,
            flagbackup=False,
            datacolumn="residual_data",
            ntime="scan",
        )
    casatasks.flagdata(vis=vis, mode="extend", flagbackup=False)

    casatasks.delmod(vis=vis)
    _tclean(
        vis,
        str(work / "3C75_initial_I"),
        stokes="I",
        datacolumn="data",
        imsize=480,
        niter=3500,
        cycleniter=750,
        savemodel="modelcolumn",
    )
    g0 = str(work / "3C75.ScG0")
    b0 = str(work / "3C75.ScB0")
    casatasks.gaincal(
        vis=vis,
        caltable=g0,
        field="",
        solint="inf",
        refant=REFANT,
        spw="",
        minsnr=3.0,
        gaintype="G",
        parang=False,
        calmode="p",
    )
    casatasks.bandpass(
        vis=vis,
        caltable=b0,
        field="",
        solint="inf",
        refant=REFANT,
        minsnr=3.0,
        spw="",
        parang=False,
        gaintable=[g0],
        interp=[],
    )
    casatasks.applycal(
        vis=vis,
        gaintable=[g0, b0],
        spw="",
        applymode="calflagstrict",
        parang=False,
        flagbackup=False,
    )

    casatasks.statwt(vis=vis, minsamp=8, datacolumn="corrected", flagbackup=False)
    _tclean(
        vis,
        str(workspace / FINAL_PREFIX),
        stokes="IQUV",
        datacolumn="corrected",
        imsize=800,
        niter=20000,
        cycleniter=1000,
        savemodel="none",
    )
    casatasks.clearstat()
    # Leave exactly one image product set in /workspace.
    shutil.rmtree(work, ignore_errors=True)


def run_solver(workspace: Path) -> None:
    image_target(workspace / "input.ms", workspace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    ws = args.workspace.resolve()
    scratch, created = _configure_temp_env(ws)
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
        if created:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
