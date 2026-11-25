"""Standalone CASA 6.7.6.14 oracle for g1_3c286pol_inspect (field roles from scan intents)."""

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


def _has(intents: set[str], prefix: str) -> bool:
    return any(i.startswith(prefix) for i in intents)


def run_solver(workspace: Path) -> None:
    from casatools import measures, msmetadata, quanta

    ms_path = workspace / "input.ms"
    msmd = msmetadata()
    me = measures()
    qa = quanta()
    try:
        msmd.open(str(ms_path))
        me.doframe(me.observatory("ALMA"))
        fnames = [str(x) for x in msmd.fieldnames()]
        roles: dict[str, str] = {}
        targets: list[str] = []
        for fid, fname in enumerate(fnames):
            intents = {str(x).upper() for x in msmd.intentsforfield(fid)}
            if _has(intents, "OBSERVE_TARGET"):
                targets.append(fname)
            if _has(intents, "CALIBRATE_FLUX"):
                roles["flux"] = fname
            if _has(intents, "CALIBRATE_BANDPASS"):
                roles["bandpass"] = fname
            if _has(intents, "CALIBRATE_PHASE"):
                roles["phase"] = fname
        bp_fid = fnames.index(roles["bandpass"])
        pdir = msmd.phasecenter(bp_fid)
        times = np.concatenate([msmd.timesforscan(int(s)) for s in msmd.scansforfield(bp_fid)])
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))
        out = {
            "flux_calibrator": roles["flux"],
            "bandpass_calibrator": roles["bandpass"],
            "phase_calibrator": roles["phase"],
            "target_fields": targets,
            # The reference walkthrough flags several antennas but describes none as dead.
            "dead_antennas": [],
            "bandpass_min_elevation_deg": round(min(els), 4),
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
