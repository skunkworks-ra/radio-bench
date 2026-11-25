"""Standalone CASA 6.7.6.14 oracle for g1_3c129p_image (CASA Guide 3C129 P-band steps on the slice)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

STAGES = ("image",)
FLUX_CAL = "3C147"
TARGET = "3C129"
REFANT = "ea09"
STANDARD_FLUX_CALS = ("3C147", "3C286", "3C48", "3C138")
# Guide section "Antenna Position Corrections" offsets (m); explicit because the
# archive lookup in gencal(caltype='antpos') needs network access.
ANTPOS_OFFSETS_M: dict[str, tuple[float, float, float]] = {
    "ea02": (0.00000, 0.00000, 0.00160),
    "ea03": (0.00290, 0.00000, 0.00000),
    "ea08": (0.00000, 0.00390, 0.00000),
    "ea10": (-0.00170, 0.00000, 0.00400),
    "ea11": (0.00000, 0.00160, -0.00170),
    "ea13": (0.00000, 0.00050, 0.00000),
    "ea14": (0.00000, -0.00090, -0.00180),
    "ea15": (0.00000, 0.00000, 0.00180),
    "ea16": (-0.00180, 0.00680, 0.00230),
    "ea18": (0.00470, 0.00350, 0.00000),
    "ea21": (-0.00160, 0.00000, 0.00210),
    "ea22": (0.00010, 0.00000, 0.00140),
    "ea23": (0.00140, 0.00200, -0.00150),
    "ea24": (0.00380, 0.00000, -0.00340),
    "ea25": (0.00330, 0.00000, -0.00110),
    "ea26": (0.00000, 0.00190, 0.00000),
    "ea27": (-0.00120, 0.00070, 0.00000),
}


def _configure_temp_env(workspace: Path) -> Path:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur = os.environ.get("TMPDIR", "").strip()
    if not cur or cur == "/tmp" or cur.startswith("/tmp/"):
        var_tmp = Path("/var/tmp")
        base = var_tmp if (var_tmp.exists() and os.access(var_tmp, os.W_OK)) else workspace
        scratch = base / f".solver_scratch_{os.getpid()}"
    else:
        scratch = Path(cur) / f"solver_{os.getpid()}"
    (scratch / "mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["MPLCONFIGDIR"] = str(scratch / "mplconfig")
    return scratch


@contextlib.contextmanager
def open_tb(path: Path, nomodify: bool = True) -> Any:
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


def field_names(ms: Path) -> list[str]:
    with open_tb(ms / "FIELD") as tb:
        return [str(x) for x in tb.getcol("NAME")]


def do_inspect(ws: Path) -> None:
    from casatools import measures, msmetadata, quanta

    ms = ws / "input.ms"
    names = field_names(ms)
    msmd = msmetadata()
    me = measures()
    qa = quanta()
    try:
        msmd.open(str(ms))
        intents = {
            n: {str(x).upper() for x in msmd.intentsforfield(i)} for i, n in enumerate(names)
        }
        flux = next(n for n in names if n.upper() in STANDARD_FLUX_CALS)
        targets = [n for n in names if any(i.startswith("OBSERVE_TARGET") for i in intents[n])]
        phase = [
            n
            for n in names
            if n not in targets and any(i.startswith("CALIBRATE_PHASE") for i in intents[n])
        ]
        fid = names.index(flux)
        me.doframe(me.observatory("VLA"))
        pdir = msmd.phasecenter(fid)
        times = np.concatenate([msmd.timesforscan(s) for s in msmd.scansforfield(fid)])
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))
    finally:
        with contextlib.suppress(Exception):
            msmd.close()
        for tool in (msmd, me, qa):
            with contextlib.suppress(Exception):
                tool.done()

    with open_tb(ms / "ANTENNA") as tb:
        ants = [str(x) for x in tb.getcol("NAME")]
    with open_tb(ms) as tb:
        sub = tb.query(f"FIELD_ID == {fid} && ANTENNA1 != ANTENNA2 && DATA_DESC_ID == 0")
        try:
            a1 = sub.getcol("ANTENNA1")
            a2 = sub.getcol("ANTENNA2")
            amp = np.abs(sub.getcol("DATA")[[0, 3]])
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()
    meds = {}
    for i, n in enumerate(ants):
        m = (a1 == i) | (a2 == i)
        if np.any(m):
            meds[n] = float(np.median(amp[:, :, m]))
    arr_med = float(np.median(list(meds.values()))) if meds else 1.0
    dead = sorted(n for n, v in meds.items() if v < 0.05 * arr_med)
    out = {
        "flux_calibrator": flux,
        "bandpass_calibrator": flux,
        "phase_calibrator": phase[0] if phase else flux,
        "target_fields": targets,
        "dead_antennas": dead,
        "bandpass_min_elevation_deg": round(min(els), 4),
    }
    (ws / "inspect_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def set_intents(ms: Path) -> None:
    names = field_names(ms)
    modes = {
        FLUX_CAL: "CALIBRATE_FLUX#UNSPECIFIED,CALIBRATE_BANDPASS#UNSPECIFIED,CALIBRATE_PHASE#UNSPECIFIED",
        TARGET: "OBSERVE_TARGET#UNSPECIFIED",
    }
    with open_tb(ms / "STATE", nomodify=False) as tb:
        n0 = int(tb.nrows())
        tb.addrows(len(names))
        for i, fn in enumerate(names):
            r = n0 + i
            tb.putcell("OBS_MODE", r, modes[fn])
            for col, val in (
                ("CAL", 0.0),
                ("LOAD", 0.0),
                ("REF", False),
                ("SIG", True),
                ("SUB_SCAN", 0),
                ("FLAG_ROW", False),
            ):
                if col in tb.colnames():
                    tb.putcell(col, r, val)
    with open_tb(ms, nomodify=False) as tb:
        tb.putcol("STATE_ID", (n0 + tb.getcol("FIELD_ID")).astype(np.int32))


def priorcals(vis: Path, ws: Path) -> list[str]:
    import casatasks

    rq = ws / "3C129_pband.rq"
    antpos = ws / "3C129_pband.antpos"
    for p in (rq, antpos):
        if p.exists():
            shutil.rmtree(p)
    ants = sorted(ANTPOS_OFFSETS_M)
    casatasks.gencal(
        vis=str(vis),
        caltable=str(antpos),
        caltype="antpos",
        antenna=",".join(ants),
        parameter=[v for a in ants for v in ANTPOS_OFFSETS_M[a]],
    )
    casatasks.gencal(vis=str(vis), caltable=str(rq), caltype="rq")
    return [str(antpos), str(rq)]


def do_precal(ws: Path) -> None:
    import casatasks

    ms = ws / "input.ms"
    fd = casatasks.flagdata
    set_intents(ms)
    with open_tb(ms / "FLAG_CMD") as tb:
        n_cmd = int(tb.nrows())
    if n_cmd > 0:
        casatasks.flagcmd(vis=str(ms), inpmode="table", action="apply", flagbackup=False)
    fd(vis=str(ms), mode="manual", antenna="ea19", flagbackup=False)
    fd(vis=str(ms), mode="manual", scan="1~2", flagbackup=False)
    fd(vis=str(ms), mode="clip", clipzeros=True, flagbackup=False)
    fd(vis=str(ms), mode="shadow", tolerance=0.0, flagbackup=False)
    fd(
        vis=str(ms),
        field="*",
        mode="tfcrop",
        datacolumn="data",
        timecutoff=4.0,
        freqcutoff=3.0,
        maxnpieces=5,
        combinescans=True,
        ntime="3600s",
        correlation="ABS_XY,ABS_YX",
        flagbackup=False,
    )
    fd(
        vis=str(ms),
        field="*",
        mode="tfcrop",
        datacolumn="data",
        timecutoff=3.0,
        freqcutoff=3.0,
        maxnpieces=2,
        combinescans=True,
        ntime="3600s",
        correlation="ABS_XX,ABS_YY",
        flagbackup=False,
    )
    fd(vis=str(ms), mode="extend", flagbackup=False)
    casatasks.setjy(
        vis=str(ms),
        field=FLUX_CAL,
        model="3C147_P.im",
        standard="Perley-Butler 2017",
        usescratch=True,
        scalebychan=True,
        spw="",
    )

    scratch = ws / ".precal_rfi_tmp"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    try:
        work = scratch / "work.ms"
        shutil.copytree(ms, work)
        prior = priorcals(work, scratch)
        g0, k0, b0 = (str(scratch / n) for n in ("G0", "K0", "B0"))
        casatasks.gaincal(
            vis=str(work),
            caltable=g0,
            gaintype="G",
            calmode="p",
            solint="int",
            field=FLUX_CAL,
            refant=REFANT,
            gaintable=prior,
        )
        casatasks.gaincal(
            vis=str(work),
            caltable=k0,
            gaintype="K",
            solint="inf",
            field=FLUX_CAL,
            refant=REFANT,
            gaintable=[*prior, g0],
        )
        casatasks.bandpass(
            vis=str(work),
            caltable=b0,
            solint="inf",
            field=FLUX_CAL,
            refant=REFANT,
            minsnr=2.0,
            gaintable=[*prior, g0, k0],
        )
        casatasks.applycal(
            vis=str(work),
            field=FLUX_CAL,
            applymode="calflagstrict",
            gaintable=[*prior, g0, k0, b0],
            flagbackup=False,
        )
        fd(
            vis=str(work),
            field=FLUX_CAL,
            mode="rflag",
            datacolumn="corrected",
            timedevscale=4.0,
            freqdevscale=3.0,
            combinescans=True,
            ntime="3600s",
            flagbackup=False,
        )
        with open_tb(work) as tb:
            flag = tb.getcol("FLAG")
            flag_row = tb.getcol("FLAG_ROW")
        with open_tb(ms, nomodify=False) as tb:
            tb.putcol("FLAG", flag)
            tb.putcol("FLAG_ROW", flag_row)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    cal_ms = ws / "calibrators.ms"
    if cal_ms.exists():
        shutil.rmtree(cal_ms)
    casatasks.mstransform(
        vis=str(ms), outputvis=str(cal_ms), field=FLUX_CAL, datacolumn="all", keepflags=False
    )


def do_calibrate(ws: Path) -> None:
    import casatasks

    ms = ws / "input.ms"
    cal_ms = ws / "calibrators.ms"
    solve_ms = cal_ms if (cal_ms / "table.dat").is_file() else ms
    prior = priorcals(solve_ms, ws)
    k1, b1, g1, gs1 = (str(ws / f"3C129_pband.{n}") for n in ("K1", "B1", "G1", "Gs1"))
    for p in (k1, b1, g1, gs1):
        if Path(p).exists():
            shutil.rmtree(p)
    casatasks.gaincal(
        vis=str(solve_ms),
        caltable=k1,
        field=FLUX_CAL,
        solint="inf",
        refant=REFANT,
        gaintype="K",
        gaintable=prior,
        parang=True,
    )
    casatasks.bandpass(
        vis=str(solve_ms),
        caltable=b1,
        field=FLUX_CAL,
        solint="inf",
        refant=REFANT,
        minsnr=3.0,
        parang=True,
        gaintable=[*prior, k1],
        interp=["", "", "nearest,nearestflag"],
    )
    casatasks.gaincal(
        vis=str(solve_ms),
        caltable=g1,
        field=FLUX_CAL,
        solint="int",
        refant=REFANT,
        minsnr=3.0,
        gaintype="G",
        calmode="ap",
        gaintable=[*prior, k1, b1],
        interp=["", "", "nearest,nearestflag", "nearest,nearestflag"],
        parang=True,
    )
    casatasks.smoothcal(
        vis=str(solve_ms), tablein=g1, caltable=gs1, smoothtype="median", smoothtime=3600.0
    )
    tables = [*prior, k1, b1, gs1]
    interp = ["", "", "nearest,nearestflag", "nearest,nearestflag", "nearest,nearestflag"]
    targets = [ms] if solve_ms == ms else [ms, cal_ms]
    for vis in targets:
        casatasks.applycal(
            vis=str(vis),
            parang=True,
            applymode="calflagstrict",
            gaintable=tables,
            interp=interp,
            flagbackup=False,
        )
    casatasks.flagdata(
        vis=str(ms), field=TARGET, mode="rflag", datacolumn="corrected", flagbackup=False
    )
    casatasks.flagdata(vis=str(ms), field=TARGET, mode="extend", flagbackup=False)
    casatasks.statwt(vis=str(ms), field=TARGET, datacolumn="corrected", timebin=30)


def do_image(ws: Path) -> None:
    import casatasks

    prefix = ws / "3C129_initial_clean"
    for p in ws.glob("3C129_initial_clean.*"):
        shutil.rmtree(p)
    # Guide section 14 parameters; imsize 4860 -> 2048 (2.8 deg) and interactive -> automask
    # so the oracle fits the agent timeout.
    casatasks.tclean(
        vis=str(ws / "input.ms"),
        field=TARGET,
        imagename=str(prefix),
        datacolumn="corrected",
        cell=["5.0arcsec", "5.0arcsec"],
        imsize=[2048, 2048],
        deconvolver="mtmfs",
        nterms=2,
        gridder="wproject",
        wprojplanes=128,
        stokes="I",
        niter=10000,
        scales=[0, 20, 30],
        pblimit=0.01,
        weighting="briggs",
        robust=0.0,
        usemask="auto-multithresh",
        pbcor=True,
        interactive=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    ws = args.workspace.resolve()
    scratch = _configure_temp_env(ws)
    prev = Path.cwd()
    try:
        os.chdir(ws)
        steps = {
            "inspect": do_inspect,
            "precal": do_precal,
            "calibrate": do_calibrate,
            "image": do_image,
        }
        for stage in STAGES:
            steps[stage](ws)
    finally:
        with contextlib.suppress(Exception):
            import casatasks

            casatasks.clearstat()
        with contextlib.suppress(Exception):
            os.chdir(prev)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
