"""CASA 6.7.6.14 oracle for g1_antennae_image_south: guide continuum tclean, non-interactive."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

# Guide imaging step for this mosaic (phasecenter field id, line-free spw, imsize, cell,
# niter, threshold). interactive=True with a hand-drawn mask is replaced by
# usemask="auto-multithresh" at tclean defaults; pbcor=True adds the .image.pbcor product.
PHASECENTER = 15
LINE_FREE_SPW = "0:1~30;120~164"
IMSIZE = 750
CELL = "0.13arcsec"
RESTFREQ = "345.79599GHz"


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


def run_solver(workspace: Path) -> None:
    import casatasks

    casatasks.tclean(
        vis=str(workspace / "input.ms"),
        imagename=str(workspace / "mosaic_target"),
        datacolumn="data",
        field="",
        phasecenter=PHASECENTER,
        stokes="I",
        specmode="mfs",
        restfreq=RESTFREQ,
        spw=LINE_FREE_SPW,
        gridder="mosaic",
        mosweight=True,
        deconvolver="hogbom",
        imsize=IMSIZE,
        cell=CELL,
        weighting="natural",
        niter=1000,
        threshold="0.4mJy",
        usemask="auto-multithresh",
        pblimit=0.2,
        pbcor=True,
        interactive=False,
    )
    casatasks.clearstat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    ws = args.workspace.resolve()
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
