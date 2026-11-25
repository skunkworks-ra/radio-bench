"""Standalone CASA 6.7.6.14 oracle solver for r1_24a376_inspect."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


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

        flux_cands: list[tuple[float, str, float]] = []
        phase_cal = ""
        targets: list[str] = []

        # Read raw amplitudes per field for flux ranking
        tb.open(str(ms_path), nomodify=True)
        fids_col = tb.getcol("FIELD_ID")
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")
        data = tb.getcol("DATA")
        tb.close()
        cross = a1 != a2
        amp = np.abs(data)

        for fid, fname in enumerate(fnames):
            intents = {str(x).upper() for x in msmd.intentsforfield(fid)}
            if any(i.startswith("OBSERVE_TARGET") for i in intents):
                targets.append(fname)
            elif any(i.startswith("CALIBRATE_PHASE") for i in intents) and not any(
                i.startswith("CALIBRATE_FLUX") for i in intents
            ):
                phase_cal = fname
            if any(i.startswith("CALIBRATE_FLUX") for i in intents):
                pdir = msmd.phasecenter(fid)
                scans = msmd.scansforfield(fid)
                times = np.concatenate([msmd.timesforscan(s) for s in scans])
                els = []
                for t in times:
                    me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
                    els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))
                el_min = min(els)
                is_pol_angle = any(i.startswith("CALIBRATE_POL_ANGLE") for i in intents)
                m = cross & (fids_col == fid)
                med_amp = float(np.median(amp[[0, 3], :, :][:, :, m])) if np.any(m) else 0.0
                flux_cands.append((el_min >= 15.0, not is_pol_angle, el_min, med_amp, fname))

        flux_cands.sort(reverse=True)
        _, _, bp_el_min, _, flux_cal = flux_cands[0]

        # Check for zero-amplitude dead antennas in raw cross-corr data
        tb.open(str(ms_path / "ANTENNA"), nomodify=True)
        ant_names = [str(x) for x in tb.getcol("NAME")]
        tb.close()
        ant_meds: dict[str, float] = {}
        for idx, aname in enumerate(ant_names):
            m = cross & ((a1 == idx) | (a2 == idx))
            if np.any(m):
                ant_meds[aname] = float(np.median(amp[:, :, m]))
        arr_med = float(np.median(list(ant_meds.values()))) if ant_meds else 1.0
        dead_ants = sorted(n for n, v in ant_meds.items() if v < 0.05 * arr_med)

        out: dict[str, Any] = {
            "flux_calibrator": flux_cal,
            "bandpass_calibrator": flux_cal,
            "phase_calibrator": phase_cal,
            "target_fields": targets,
            "dead_antennas": dead_ants,
            "bandpass_min_elevation_deg": round(bp_el_min, 4),
        }
        (workspace / "inspect_summary.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )
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
            tb.done()


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
