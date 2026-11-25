"""Standalone CASA 6.7.6.14 oracle for g2_3c75pol_e2e (inspect + Stokes I/IQUV imaging).

Inspect: write inspect_summary.json. Then the reference guide's polarization calibration on the
pipeline-calibrated MS (extra flagging, setjy models, multiband Kcross, Df+QU on the leakage
calibrator, Xf on the angle calibrator, applycal with parang=True), split + statwt of the target,
and the guide imaging chain with one self-calibration round. Adapted to the 16-channel averaged
input: delay channel range 1~14 (guide 5~58 of 64) and per-channel polcal solint 'inf,8MHz'.
If any polarization step fails, the target is split from DATA so Stokes I imaging still runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REFANT = "ea10"
FINAL_PREFIX = "3C75_final"
FLUX_CAL = "0137+331=3C48"
UNPOL_CAL = "J2355+4950"
PHASE_CAL = "J0259+0747"
TARGET = "3C75"
CAL_FIELDS = f"{FLUX_CAL},{UNPOL_CAL},{PHASE_CAL}"


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


def inspect_ms(ms_path: Path) -> dict[str, Any]:
    from casatools import measures, msmetadata, quanta, table

    msmd = msmetadata()
    me = measures()
    qa = quanta()
    tb = table()
    try:
        msmd.open(str(ms_path))
        fnames = [str(x) for x in msmd.fieldnames()]
        intents = {i: {str(x).upper() for x in msmd.intentsforfield(i)} for i in range(len(fnames))}
        scans = {i: [int(s) for s in msmd.scansforfield(i)] for i in range(len(fnames))}

        def has(fid: int, prefix: str) -> bool:
            return any(x.startswith(prefix) for x in intents[fid])

        targets = [fnames[i] for i in intents if has(i, "OBSERVE_TARGET")]
        flux_ids = [i for i in intents if has(i, "CALIBRATE_FLUX")]
        bp_ids = [i for i in intents if has(i, "CALIBRATE_BANDPASS")] or flux_ids
        # Complex gain calibrator: phase-intent field (not flux) with the most scans (interleaved).
        phase_ids = [i for i in intents if has(i, "CALIBRATE_PHASE") and i not in flux_ids]
        phase_id = max(phase_ids, key=lambda i: len(scans[i]))
        flux_id = flux_ids[0]
        bp_id = bp_ids[0]

        me.doframe(me.observatory("VLA"))
        pdir = msmd.phasecenter(bp_id)
        times = np.unique(np.concatenate([msmd.timesforscan(s) for s in scans[bp_id]]))
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))

        tb.open(str(ms_path / "ANTENNA"), nomodify=True)
        ant_names = [str(x) for x in tb.getcol("NAME")]
        tb.close()
        tb.open(str(ms_path), nomodify=True)
        nrow = int(tb.nrows())
        nant = len(ant_names)
        nflag = np.zeros(nant)
        nvis = np.zeros(nant)
        meds: list[list[np.ndarray]] = [[] for _ in range(nant)]
        step = 200_000
        for start in range(0, nrow, step):
            n = min(step, nrow - start)
            a1 = tb.getcol("ANTENNA1", start, n)
            a2 = tb.getcol("ANTENNA2", start, n)
            flg = tb.getcol("FLAG", start, n)[[0, 3]]
            amp = np.abs(tb.getcol("DATA", start, n)[[0, 3]])
            cross = a1 != a2
            rowf = flg.reshape(-1, n)
            amp = np.where(flg, np.nan, amp).reshape(-1, n)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rowmed = np.nanmedian(amp, axis=0)
            for ant in range(nant):
                m = cross & ((a1 == ant) | (a2 == ant))
                if not np.any(m):
                    continue
                nflag[ant] += float(rowf[:, m].sum())
                nvis[ant] += float(rowf[:, m].size)
                v = rowmed[m]
                v = v[np.isfinite(v)]
                if v.size:
                    meds[ant].append(v)
        tb.close()
        ant_med = {
            a: (float(np.median(np.concatenate(meds[i]))) if meds[i] else 0.0)
            for i, a in enumerate(ant_names)
        }
        arr_med = (
            float(np.median([v for v in ant_med.values() if v > 0]))
            if any(ant_med.values())
            else 1.0
        )
        dead = sorted(
            a
            for i, a in enumerate(ant_names)
            if nvis[i] == 0 or nflag[i] / nvis[i] >= 0.999 or ant_med[a] < 0.05 * arr_med
        )
        return {
            "flux_calibrator": fnames[flux_id],
            "bandpass_calibrator": fnames[bp_id],
            "phase_calibrator": fnames[phase_id],
            "target_fields": targets,
            "dead_antennas": dead,
            "bandpass_min_elevation_deg": round(float(min(els)), 4),
        }
    finally:
        with contextlib.suppress(Exception):
            msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()
        with contextlib.suppress(Exception):
            me.done()
        with contextlib.suppress(Exception):
            qa.done()
        with contextlib.suppress(Exception):
            tb.close()
        with contextlib.suppress(Exception):
            tb.done()


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


def polcal_and_split(ms_path: Path, workspace: Path) -> Path:
    import casatasks

    v = str(ms_path)
    target_ms = workspace / "3C75.ms"
    if target_ms.exists():
        shutil.rmtree(target_ms)
    column = "data"
    try:
        casatasks.flagdata(
            vis=v,
            flagbackup=False,
            mode="manual",
            antenna="ea12",
            scan="17",
            timerange="07:25:57~07:26:18",
        )
        casatasks.flagdata(
            vis=v,
            mode="tfcrop",
            field=CAL_FIELDS,
            correlation="",
            freqfit="line",
            extendflags=False,
            flagbackup=False,
        )
        casatasks.flagdata(
            vis=v,
            mode="rflag",
            datacolumn="data",
            field=CAL_FIELDS,
            correlation="RL,LR",
            extendflags=True,
            flagbackup=False,
        )
        casatasks.setjy(
            vis=v,
            field=FLUX_CAL,
            spw="",
            selectdata=False,
            scalebychan=True,
            standard="manual",
            fluxdensity=[8.55557, 0, 0, 0],
            spix=[-0.8864, -0.1432],
            reffreq="3.0GHz",
            polindex=[0.02152856, 0.03937167, 0.003804, -0.01969663],
            polangle=[-1.68850138, 1.637449, -2.36190333, 1.59809091, -0.38386091],
            rotmeas=0,
            usescratch=True,
        )
        casatasks.setjy(
            vis=v,
            field=UNPOL_CAL,
            spw="",
            selectdata=False,
            scalebychan=True,
            standard="manual",
            fluxdensity=[1.76891, 0, 0, 0],
            spix=[-0.599302, -0.203418],
            reffreq="2984571609.0079317Hz",
            polindex=[],
            polangle=[],
            rotmeas=0,
            usescratch=True,
        )
        casatasks.setjy(
            vis=v,
            field=PHASE_CAL,
            spw="",
            selectdata=False,
            scalebychan=True,
            standard="manual",
            fluxdensity=[0.970688, 0, 0, 0],
            spix=[0.170883, -0.167233],
            reffreq="2984571609.0079317Hz",
            polindex=[],
            polangle=[],
            rotmeas=0,
            usescratch=True,
        )
        kcross = str(workspace / "TDRW0001_calibrated.Kcross_mbd")
        dtab = str(workspace / "TDRW0001_calibrated.DfQU")
        xtab = str(workspace / "TDRW0001_calibrated.Xf")
        spwmap8 = [0, 0, 0, 0, 0, 0, 0, 0]
        casatasks.gaincal(
            vis=v,
            caltable=kcross,
            field=FLUX_CAL,
            spw="0~7:1~14",
            refant=REFANT,
            gaintype="KCROSS",
            solint="inf",
            combine="scan,spw",
            calmode="ap",
            append=False,
            gaintable=[""],
            gainfield=[""],
            interp=[""],
            spwmap=[[]],
            parang=True,
        )
        casatasks.polcal(
            vis=v,
            caltable=dtab,
            intent="CALIBRATE_POL_LEAKAGE#UNSPECIFIED",
            spw="0~7",
            refant=REFANT,
            poltype="Df+QU",
            solint="inf,8MHz",
            combine="scan",
            gaintable=[kcross],
            gainfield=[""],
            spwmap=[spwmap8],
            append=False,
        )
        casatasks.flagdata(
            vis=dtab,
            mode="clip",
            correlation="ABS_ALL",
            clipminmax=[0.0, 0.25],
            datacolumn="CPARAM",
            clipoutside=True,
            action="apply",
            flagbackup=False,
            savepars=False,
        )
        casatasks.polcal(
            vis=v,
            caltable=xtab,
            spw="0~7",
            field=FLUX_CAL,
            solint="inf,8MHz",
            combine="scan",
            poltype="Xf",
            refant=REFANT,
            gaintable=[kcross, dtab],
            gainfield=["", ""],
            spwmap=[spwmap8, []],
            append=False,
        )
        casatasks.applycal(
            vis=v,
            field="",
            gainfield=["", "", ""],
            flagbackup=False,
            interp=["", "", ""],
            gaintable=[kcross, dtab, xtab],
            spw="0~7",
            calwt=[False, False, False],
            applymode="calflagstrict",
            antenna="*&*",
            spwmap=[spwmap8, [], []],
            parang=True,
        )
        column = "corrected"
    except Exception:
        traceback.print_exc()
        print("polarization calibration failed; imaging the pipeline-calibrated DATA column")
    casatasks.split(vis=v, outputvis=str(target_ms), datacolumn=column, field=TARGET)
    casatasks.statwt(vis=str(target_ms), datacolumn="data", minsamp=8)
    casatasks.clearstat()
    return target_ms


def run_solver(workspace: Path) -> None:
    ms_path = workspace / "input.ms"
    out = inspect_ms(ms_path)
    (workspace / "inspect_summary.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    target_ms = polcal_and_split(ms_path, workspace)
    image_target(target_ms, workspace)


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
