"""Standalone CASA 6.7.6.14 oracle for g1_twhya_image, reproducing the guide's continuum image.

Guide steps (First Look at Imaging CASA 6.6.6): split the science target with width=8 from the
calibrated DATA column, then the non-interactive tclean (hogbom, 250 x 0.1 arcsec, briggs 0.5,
box mask [100,100]-[150,150] pix, threshold 15 mJy, niter 10000). pbcor=True replaces the
guide's separate impbcor step so `<prefix>.image.pbcor` exists next to the other products.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

TARGET_FIELD_ID = 5
TARGET_FIELD_NAME = "TW Hya"


def _configure_temp_env(workspace: Path) -> Path:
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
    return scratch


def _check_target_field(ms_path: Path) -> None:
    from casatools import msmetadata

    md = msmetadata()
    opened = False
    try:
        opened = bool(md.open(str(ms_path)))
        names = [str(x) for x in md.fieldnames()]
        if len(names) <= TARGET_FIELD_ID or names[TARGET_FIELD_ID] != TARGET_FIELD_NAME:
            raise RuntimeError(f"field {TARGET_FIELD_ID} is not {TARGET_FIELD_NAME!r}: {names}")
    finally:
        if opened:
            with contextlib.suppress(Exception):
                md.close()
        with contextlib.suppress(Exception):
            md.done()


def run_solver(workspace: Path) -> None:
    import casatasks

    ms_path = workspace / "input.ms"
    smoothed = workspace / "twhya_smoothed.ms"
    prefix = workspace / "twhya_cont"
    _check_target_field(ms_path)

    if smoothed.exists():
        shutil.rmtree(smoothed)
    casatasks.split(
        vis=str(ms_path),
        field=str(TARGET_FIELD_ID),
        width="8",
        outputvis=str(smoothed),
        datacolumn="data",
    )
    casatasks.tclean(
        vis=str(smoothed),
        imagename=str(prefix),
        field="0",
        spw="",
        datacolumn="data",
        specmode="mfs",
        gridder="standard",
        deconvolver="hogbom",
        imsize=[250, 250],
        cell=["0.1arcsec"],
        mask="box [[100pix, 100pix], [150pix, 150pix]]",
        weighting="briggs",
        robust=0.5,
        threshold="15mJy",
        niter=10000,
        interactive=False,
        pbcor=True,
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
