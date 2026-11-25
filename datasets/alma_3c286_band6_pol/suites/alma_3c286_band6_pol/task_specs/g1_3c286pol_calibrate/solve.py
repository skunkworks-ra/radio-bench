"""Standalone CASA 6.7.6.14 oracle for g1_3c286pol_calibrate (reference walkthrough Stokes I steps)."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    ws = parser.parse_args().workspace.resolve()
    scratch = _configure_temp_env(ws)
    prev = Path.cwd()
    try:
        os.chdir(ws)
        calibrate_stokes_i(ws)
    finally:
        with contextlib.suppress(Exception):
            import casatasks

            casatasks.clearstat()
        with contextlib.suppress(Exception):
            os.chdir(prev)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
