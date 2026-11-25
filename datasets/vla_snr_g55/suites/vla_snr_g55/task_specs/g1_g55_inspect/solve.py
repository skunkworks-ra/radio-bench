"""Standalone CASA 6.7.6.14 oracle solver for g1_g55_inspect."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


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
    from casatools import measures, msmetadata, quanta, table

    ms_path = workspace / "input.ms"
    msmd = msmetadata()
    me = measures()
    qa = quanta()
    tb = table()
    try:
        msmd.open(str(ms_path))
        me.doframe(me.observatory("VLA"))
        fnames = [str(x) for x in msmd.fieldnames()]
        targets: list[str] = []
        phase_cal = ""
        flux_cal = ""
        for fid, fname in enumerate(fnames):
            intents = {str(x).upper() for x in msmd.intentsforfield(fid)}
            if any(i.startswith("OBSERVE_TARGET") for i in intents):
                targets.append(fname)
            elif any(
                i.startswith(("CALIBRATE_BANDPASS", "CALIBRATE_FLUX", "CALIBRATE_AMPLI"))
                for i in intents
            ):
                flux_cal = fname
            elif any(i.startswith("CALIBRATE_PHASE") for i in intents):
                phase_cal = fname

        flux_fid = fnames.index(flux_cal)
        pdir = msmd.phasecenter(flux_fid)
        times = np.unique(
            np.concatenate([msmd.timesforscan(s) for s in msmd.scansforfield(flux_fid)])
        )
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))

        tb.open(str(ms_path / "ANTENNA"), nomodify=True)
        ant_names = [str(x) for x in tb.getcol("NAME")]
        tb.close()
        tb.open(str(ms_path), nomodify=True)
        sub = tb.query(f"FIELD_ID=={flux_fid} && ANTENNA1!=ANTENNA2")
        try:
            amp = np.abs(sub.getcol("DATA"))
            a1 = sub.getcol("ANTENNA1")
            a2 = sub.getcol("ANTENNA2")
        finally:
            sub.close()
            tb.close()
        meds = {
            n: float(np.median(amp[:, :, (a1 == i) | (a2 == i)]))
            for i, n in enumerate(ant_names)
            if np.any((a1 == i) | (a2 == i))
        }
        arr_med = float(np.median(list(meds.values())))
        dead = sorted(n for n, v in meds.items() if v < 0.05 * arr_med)

        out = {
            "flux_calibrator": flux_cal,
            "bandpass_calibrator": flux_cal,
            "phase_calibrator": phase_cal,
            "target_fields": targets,
            "dead_antennas": dead,
            "bandpass_min_elevation_deg": round(min(els), 4),
        }
        (workspace / "inspect_summary.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        for closer in (msmd.close, msmd.done, me.done, qa.done, tb.done):
            with contextlib.suppress(Exception):
                closer()


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
        with contextlib.suppress(Exception):
            os.chdir(prev)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
