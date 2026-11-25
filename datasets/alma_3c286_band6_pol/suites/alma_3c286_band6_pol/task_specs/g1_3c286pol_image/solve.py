"""Standalone CASA 6.7.6.14 oracle for g1_3c286pol_image (reference walkthrough self-cal sequence).

Non-interactive version of the walkthrough: IQUV clarkstokes imaging, phase self-cal at 300 s and
30 s, amplitude+phase self-cal at 30 s, final IQUV image. The interactive clean mask is replaced
by a fixed 1.5 arcsec circle on the phase centre (the target).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

REFANT = "DV23"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    ws = parser.parse_args().workspace.resolve()
    scratch = _configure_temp_env(ws)
    prev = Path.cwd()
    try:
        os.chdir(ws)
        selfcal_and_image(ws, ws / "input.ms")
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
