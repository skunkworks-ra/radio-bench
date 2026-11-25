"""Standalone CASA 6.7.6.14 oracle solver for r1_24a376_image (4096x4096 first-sidelobe mosaic)."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

TARGET_FIELDS = "PER_FIELD_71,PER_FIELD_75,PER_FIELD_77,PER_FIELD_81"
MOSAIC_PHASECENTER = "J2000 52.11772deg 30.44509deg"


def _configure_temp_env(workspace: Path) -> tuple[Path, bool]:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur_tmp = os.environ.get("TMPDIR", "").strip()
    if not cur_tmp or cur_tmp == "/tmp" or cur_tmp.startswith("/tmp/"):
        var_tmp = Path("/var/tmp")
        scratch = (
            var_tmp if (var_tmp.exists() and os.access(var_tmp, os.W_OK)) else workspace
        ) / f".solver_scratch_{os.getpid()}"
    else:
        scratch = Path(cur_tmp) / f"solver_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["MPLCONFIGDIR"] = str(scratch / "mplconfig")
    (scratch / "mplconfig").mkdir(parents=True, exist_ok=True)
    return scratch, True


def run_solver(workspace: Path) -> None:
    import casatasks

    ms_path = workspace / "input.ms"
    img_prefix = workspace / "mosaic_target"

    casatasks.tclean(
        vis=str(ms_path),
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
