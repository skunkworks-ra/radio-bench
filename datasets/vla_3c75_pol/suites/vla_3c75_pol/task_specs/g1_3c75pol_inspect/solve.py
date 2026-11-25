"""Standalone CASA 6.7.6.14 oracle for g1_3c75pol_inspect (field roles, dead antennas, elevation)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np


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


def run_solver(workspace: Path) -> None:
    out = inspect_ms(workspace / "input.ms")
    (workspace / "inspect_summary.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )


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
