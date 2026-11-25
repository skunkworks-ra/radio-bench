"""Standalone CASA 6.7.6.14 oracle solver for r1_24a376_precal."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

FLUX_CAL = "0542+498=3C147"
CAL_FIELDS = "J0336+3218,0542+498=3C147,0521+166=3C138"
REFANT = "ea16,ea04,ea07"


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
    from casatools import table

    ms_path = workspace / "input.ms"
    flagonline = workspace / "input.flagonline.txt"
    cal_ms = workspace / "calibrators.ms"

    if flagonline.is_file():
        casatasks.flagdata(vis=str(ms_path), mode="list", inpfile=str(flagonline), flagbackup=False)
    casatasks.flagdata(
        vis=str(ms_path), mode="quack", quackinterval=5.0, quackmode="beg", flagbackup=False
    )
    casatasks.flagdata(vis=str(ms_path), mode="shadow", tolerance=0.0, flagbackup=False)
    casatasks.flagdata(vis=str(ms_path), mode="clip", clipzeros=True, flagbackup=False)
    casatasks.flagdata(
        vis=str(ms_path), mode="tfcrop", timecutoff=3.0, freqcutoff=3.0, flagbackup=False
    )
    casatasks.flagdata(vis=str(ms_path), mode="extend", extendpols=True, flagbackup=False)

    casatasks.setjy(
        vis=str(ms_path), field=FLUX_CAL, standard="Perley-Butler 2017", usescratch=True
    )

    # Initial bandpass + residual rflag on a scratch copy so input.ms gets clean RFI flags without CORRECTED_DATA
    scratch_rfi = workspace / ".precal_rfi_tmp"
    if scratch_rfi.exists():
        shutil.rmtree(scratch_rfi)
    scratch_rfi.mkdir(parents=True, exist_ok=True)
    try:
        rfi_ms = scratch_rfi / "work.ms"
        shutil.copytree(ms_path, rfi_ms)
        g_init = scratch_rfi / "G_init.cal"
        b_init = scratch_rfi / "B_init.cal"
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
        casatasks.applycal(
            vis=str(rfi_ms),
            field="J0336+3218,0521+166=3C138,PER_FIELD_71,PER_FIELD_75,PER_FIELD_77,PER_FIELD_81",
            gaintable=[str(b_init)],
            gainfield=[FLUX_CAL],
            interp=["nearest,linear"],
            calwt=False,
            flagbackup=False,
        )
        casatasks.flagdata(
            vis=str(rfi_ms),
            mode="rflag",
            datacolumn="corrected",
            timedevscale=5.0,
            freqdevscale=5.0,
            flagbackup=False,
        )
        tb = table()
        try:
            tb.open(str(rfi_ms), nomodify=True)
            new_flag = tb.getcol("FLAG")
            new_flag_row = tb.getcol("FLAG_ROW")
            tb.close()
            tb.open(str(ms_path), nomodify=False)
            tb.putcol("FLAG", new_flag)
            tb.putcol("FLAG_ROW", new_flag_row)
            tb.close()
        finally:
            with contextlib.suppress(Exception):
                tb.done()
    finally:
        shutil.rmtree(scratch_rfi, ignore_errors=True)

    if cal_ms.exists():
        shutil.rmtree(cal_ms)
    casatasks.mstransform(
        vis=str(ms_path), outputvis=str(cal_ms), field=CAL_FIELDS, datacolumn="all", keepflags=False
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
        with contextlib.suppress(Exception):
            os.chdir(prev)
        if created:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
