"""Checkpoint and ground-truth builder for the vla_3c129_pband reference suite.

Needs only casatools, casatasks and numpy (runs inside the casa-base container). Steps follow
the CASA Guide "VLA Radio galaxy 3C 129: P-band continuum tutorial-CASA6.7.2" on a
deterministic slice of the guide's ASDM:

  slice: source spws 2~9 (8 x 16 MHz, 256-384 MHz), Hanning smoothed (guide 7c), then
         2-channel average (64 x 250 kHz per spw) and 4 s time average (native 2 s),
         all 5 scans, all antennas, all 4 correlations.

Outputs under --out-root (<data_dir>/casaguides/vla_3c129_pband):
  raw/input.ms, post_precal/{input.ms,calibrators.ms,precal_masks.npz},
  post_calibrate/{input.ms,...caltables}, reference_image/, <task_id>/ground_truth.json,
  *_summary.json. HISTORY is cleared on every delivered MS.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ASDM_NAME = "imag-test-copy.57080.956837025464"
SLICE_SPW = "2~9"
CHANBIN = 2
TIMEBIN = "4s"
FLUX_CAL = "3C147"
TARGET = "3C129"
REFANT = "ea09"
DEAD_ANTENNAS = ["ea19"]
SUITE_ID = "vla_3c129_pband"
MISSING_STATIONS = ["ea04"]
GUIDE_DUMMY_SCANS = "1~2"
EDGE_CHANS = 2

# Guide section "Antenna Position Corrections": gencal logger offsets (m) for 17 antennas.
# Applied with explicit parameters because gencal(caltype='antpos') needs network access.
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

TASK_IDS = {
    "inspect": "g1_3c129p_inspect",
    "precal": "g1_3c129p_precal",
    "calibrate": "g1_3c129p_calibrate",
    "image": "g1_3c129p_image",
    "e2e": "g2_3c129p_e2e",
}

# Tolerances copied from the 24A-376 builder (CONVENTION) or quoted from the guide (GUIDE).
# REFERENCE.json carries the provenance of every entry.
TOL_INSPECT = {"elevation_abs_deg": 1.0}
TOL_PRECAL = {"model_amp_rel_tol": 0.02, "flag_recall_min": 0.85, "clean_false_positive_max": 0.05}
TOL_CALIBRATE = {
    "target_unflagged_rows_min": 0.40,
    "target_unflagged_rows_max": 0.98,
    "bp_channel_phase_rms_max_deg": 5.0,
    "verifier_tclean_dr_min": None,  # 0.70 x verifier DR on post_calibrate, set at build
}
TOL_IMAGE = {
    "pixels_per_beam_min": 2.5,
    "pixels_per_beam_max": 6.0,
    "off_source_residual_rms_min_jy_per_beam": 0.00004,
    "dynamic_range_min": None,  # 0.55 x reference-image DR, set at build
    "negative_model_bowl_ratio_max": 0.12,
}
GUIDE_TRUTH = {
    "guide_flagged_fraction_after_autoflag": 0.37,
    "guide_expected_hpbw_arcsec": 18.5,
    "guide_cell_arcsec": 5.0,
    "guide_delay_abs_max_ns": 30.0,
}


def _setup_env(out_root: Path) -> None:
    tmp_dir = out_root / "_tmp"
    mpl_dir = tmp_dir / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    for k, v in (
        ("OMP_NUM_THREADS", "8"),
        ("OMP_WAIT_POLICY", "PASSIVE"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(k, v)


@contextlib.contextmanager
def open_tb(path: Path | str, nomodify: bool = True) -> Iterator[Any]:
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


def clear_history(ms_path: Path) -> None:
    hist = ms_path / "HISTORY"
    if (hist / "table.dat").is_file():
        with open_tb(hist, nomodify=False) as tb:
            n = int(tb.nrows())
            if n > 0:
                tb.removerows(list(range(n)))


def assert_unlocked(ms_path: Path) -> None:
    import casatasks

    casatasks.clearstat()
    with open_tb(ms_path) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active table lock on {ms_path}")


def dir_size_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1048576.0, 2)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def raw_data_blake2b(ms_path: Path, chunk_size: int = 50_000) -> str:
    """Same digest as radio_bench.grading.extractors.common.compute_raw_data_blake2b."""
    cols = ("TIME", "ANTENNA1", "ANTENNA2", "FIELD_ID", "DATA_DESC_ID", "DATA")
    hasher = hashlib.blake2b(digest_size=32)
    with open_tb(ms_path) as tb:
        hasher.update(f"total_nrows={int(tb.nrows())};".encode())
        ddids = sorted(int(x) for x in np.unique(np.asarray(tb.getcol("DATA_DESC_ID"))))
        for ddid in ddids:
            sub = tb.query(f"DATA_DESC_ID == {ddid}")
            try:
                n = int(sub.nrows())
                for sr in range(0, n, chunk_size):
                    nr = min(chunk_size, n - sr)
                    hasher.update(f"ddid={ddid}:sr={sr}:nr={nr};".encode())
                    for col in cols:
                        arr = np.ascontiguousarray(
                            np.asarray(sub.getcol(col, startrow=sr, nrow=nr))
                        )
                        hasher.update(f"{col}:{arr.dtype.str}:{arr.shape};".encode())
                        hasher.update(memoryview(arr))
            finally:
                with contextlib.suppress(Exception):
                    sub.close()
                with contextlib.suppress(Exception):
                    sub.done()
    digest = hasher.hexdigest()
    try:  # cross-check against the grader when the repo package is importable (host build)
        from radio_bench.grading.extractors.common import compute_raw_data_blake2b

        if compute_raw_data_blake2b(ms_path) != digest:
            raise RuntimeError(f"blake2b mismatch vs grader for {ms_path}")
    except ImportError:
        pass
    return digest


def inspect_ms_meta(ms_path: Path) -> dict[str, Any]:
    from casatools import msmetadata

    with open_tb(ms_path) as tb:
        nrows = int(tb.nrows())
        cols = sorted(tb.colnames())
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        active = sorted({int(x) for x in np.unique(np.concatenate([ant1, ant2]))})
    with open_tb(ms_path / "FIELD") as tb:
        field_names = [str(x) for x in tb.getcol("NAME")]
    with open_tb(ms_path / "ANTENNA") as tb:
        ant_names = [str(x) for x in tb.getcol("NAME")]
        stations = [str(x) for x in tb.getcol("STATION")]
    with open_tb(ms_path / "SPECTRAL_WINDOW") as tb:
        nchan = [int(x) for x in tb.getcol("NUM_CHAN")]
        ref_mhz = [round(float(x) / 1e6, 4) for x in tb.getcol("REF_FREQUENCY")]
        bw_mhz = [round(float(x) / 1e6, 4) for x in tb.getcol("TOTAL_BANDWIDTH")]
    with open_tb(ms_path / "POLARIZATION") as tb:
        corr = [list(map(int, tb.getcell("CORR_TYPE", i))) for i in range(int(tb.nrows()))]
    with open_tb(ms_path / "HISTORY") as tb:
        hist_n = int(tb.nrows())
    sub_rows: dict[str, int] = {}
    for sub in ("SYSPOWER", "FLAG_CMD", "STATE"):
        if (ms_path / sub / "table.dat").is_file():
            with open_tb(ms_path / sub) as tb:
                sub_rows[sub] = int(tb.nrows())
        else:
            sub_rows[sub] = -1
    msmd = msmetadata()
    try:
        msmd.open(str(ms_path))
        intents = {
            n: sorted(str(x) for x in msmd.intentsforfield(i)) for i, n in enumerate(field_names)
        }
        scans = [int(s) for s in sorted(msmd.scannumbers())]
        scan_fields = {str(s): [field_names[int(f)] for f in msmd.fieldsforscan(s)] for s in scans}
        scan_secs = {}
        for s in scans:
            t = np.asarray(msmd.timesforscan(s), dtype=float)
            scan_secs[str(s)] = round(float(t.max() - t.min()), 1) if t.size else 0.0
        intg = (
            # default spwid=0 may be absent from the scan (slice keeps spw 2-9)
            float(
                np.median(
                    np.asarray(
                        msmd.exposuretime(
                            scan=scans[0], spwid=int(min(msmd.spwsforscan(scans[0])))
                        )["value"]
                    )
                )
            )
            if scans
            else None
        )
    finally:
        with contextlib.suppress(Exception):
            msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()
    return {
        "ms_path": str(ms_path),
        "size_mb": dir_size_mb(ms_path),
        "nrows": nrows,
        "columns": cols,
        "field_names": field_names,
        "field_intents": intents,
        "scans": scans,
        "scan_fields": scan_fields,
        "scan_span_s": scan_secs,
        "integration_s": intg,
        "antenna_names": ant_names,
        "antenna_stations": stations,
        "n_active_antennas": len(active),
        "spw_nchan": nchan,
        "spw_ref_freq_mhz": ref_mhz,
        "spw_bw_mhz": bw_mhz,
        "corr_types": corr,
        "history_nrows": hist_n,
        "subtable_rows": sub_rows,
    }


def min_elevation_deg(ms_path: Path, field_name: str) -> float:
    from casatools import measures, msmetadata, quanta

    msmd = msmetadata()
    me = measures()
    qa = quanta()
    try:
        msmd.open(str(ms_path))
        me.doframe(me.observatory("VLA"))
        fid = [str(x) for x in msmd.fieldnames()].index(field_name)
        pdir = msmd.phasecenter(fid)
        times = np.concatenate([msmd.timesforscan(s) for s in msmd.scansforfield(fid)])
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pdir, "AZEL")["m1"], "deg")["value"]))
        return float(min(els))
    finally:
        with contextlib.suppress(Exception):
            msmd.close()
        for tool in (msmd, me, qa):
            with contextlib.suppress(Exception):
                tool.done()


def raw_antenna_median_amps(ms_path: Path) -> dict[str, float]:
    """Median raw cross-correlation amplitude per antenna on the flux calibrator (diagnostic)."""
    with open_tb(ms_path / "ANTENNA") as tb:
        names = [str(x) for x in tb.getcol("NAME")]
    with open_tb(ms_path / "FIELD") as tb:
        fid = [str(x) for x in tb.getcol("NAME")].index(FLUX_CAL)
    with open_tb(ms_path) as tb:
        # slice keeps spw 2-9, so DDID 0 has no rows; use the lowest DDID present
        dd0 = int(np.min(tb.getcol("DATA_DESC_ID")))
        sub = tb.query(f"FIELD_ID == {fid} && ANTENNA1 != ANTENNA2 && DATA_DESC_ID == {dd0}")
        try:
            a1 = sub.getcol("ANTENNA1")
            a2 = sub.getcol("ANTENNA2")
            amp = np.abs(sub.getcol("DATA")[[0, 3]])
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()
    out: dict[str, float] = {}
    for i, n in enumerate(names):
        m = (a1 == i) | (a2 == i)
        if np.any(m):
            out[n] = round(float(np.median(amp[:, :, m])), 6)
    return out


def antpos_table(vis: Path, caltable: Path) -> None:
    import casatasks

    ants = sorted(ANTPOS_OFFSETS_M)
    params = [v for a in ants for v in ANTPOS_OFFSETS_M[a]]
    casatasks.gencal(
        vis=str(vis),
        caltable=str(caltable),
        caltype="antpos",
        antenna=",".join(ants),
        parameter=params,
    )


def set_intents(ms_path: Path) -> None:
    """Write CALIBRATE_* / OBSERVE_TARGET intents per field (post_precal checkpoint only)."""
    with open_tb(ms_path / "FIELD") as tb:
        fnames = [str(x) for x in tb.getcol("NAME")]
    modes = {
        FLUX_CAL: "CALIBRATE_FLUX#UNSPECIFIED,CALIBRATE_BANDPASS#UNSPECIFIED,CALIBRATE_PHASE#UNSPECIFIED",
        TARGET: "OBSERVE_TARGET#UNSPECIFIED",
    }
    with open_tb(ms_path / "STATE", nomodify=False) as tb:
        n0 = int(tb.nrows())
        tb.addrows(len(fnames))
        for i, fn in enumerate(fnames):
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
    with open_tb(ms_path, nomodify=False) as tb:
        fid = tb.getcol("FIELD_ID")
        tb.putcol("STATE_ID", (n0 + fid).astype(np.int32))


def parallel_amp_mad_classes(
    corr: np.ndarray, flag: np.ndarray, field_id: np.ndarray, ddid: np.ndarray, cross: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Strong (>= 8 MAD) / mild (>= 3.5 MAD) parallel-hand amplitude outliers (24A-376 rule)."""
    par_amp = 0.5 * (np.abs(corr[0]) + np.abs(corr[3]))
    valid = (~np.any(flag, axis=0)) & cross[np.newaxis, :]
    strong = np.zeros_like(par_amp, dtype=bool)
    mild = np.zeros_like(par_amp, dtype=bool)
    for fid in np.unique(field_id):
        for d in np.unique(ddid):
            rows = np.flatnonzero((field_id == fid) & (ddid == d))
            if rows.size < 16:
                continue
            f_amp = par_amp[:, rows]
            f_val = valid[:, rows]
            for ch in range(par_amp.shape[0]):
                v = f_amp[ch, f_val[ch]]
                if v.size < 16:
                    continue
                med = float(np.median(v))
                mad = float(1.4826 * np.median(np.abs(v - med)))
                if mad <= 0.0:
                    continue
                dev = np.abs(f_amp[ch] - med)
                strong[ch, rows] = dev >= 8.0 * mad
                mild[ch, rows] = dev >= 3.5 * mad
    return strong, mild


def getcol_all(ms_path: Path, col: str) -> np.ndarray:
    """Full-column read; safe here because every spw in the slice has the same shape."""
    with open_tb(ms_path) as tb:
        return np.asarray(tb.getcol(col))


def step_slice(out_root: Path, asdm: Path, summary: dict[str, Any]) -> Path:
    import casatasks

    work = out_root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    full_ms = work / "full.ms"
    onl = work / "importflags.txt"
    if not (full_ms / "table.dat").is_file():
        casatasks.importasdm(
            asdm=str(asdm),
            vis=str(full_ms),
            process_flags=True,
            applyflags=False,
            savecmds=True,
            outfile=str(onl),
            overwrite=True,
        )
    casatasks.listobs(vis=str(full_ms), listfile=str(work / "full.listobs"), overwrite=True)
    summary["full_ms"] = {"size_mb": dir_size_mb(full_ms)}

    han_ms = work / "hanning.ms"
    if han_ms.exists():
        shutil.rmtree(han_ms)
    casatasks.hanningsmooth(
        vis=str(full_ms), outputvis=str(han_ms), spw=SLICE_SPW, datacolumn="data"
    )

    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_ms = raw_dir / "input.ms"
    if raw_ms.exists():
        shutil.rmtree(raw_ms)
    casatasks.mstransform(
        vis=str(han_ms),
        outputvis=str(raw_ms),
        datacolumn="data",
        chanaverage=True,
        chanbin=CHANBIN,
        timeaverage=True,
        timebin=TIMEBIN,
        keepflags=True,
        reindex=True,
    )
    shutil.rmtree(han_ms)
    shutil.copy2(onl, raw_dir / "input.flagonline.txt")
    lines = [ln for ln in onl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if any("spw=" in ln for ln in lines):
        raise RuntimeError("online flag commands select spw; reindexed slice would mis-apply them")
    with open_tb(raw_ms / "FLAG_CMD") as tb:
        n_cmd = int(tb.nrows())
    if n_cmd == 0 and lines:
        casatasks.flagcmd(
            vis=str(raw_ms),
            inpmode="list",
            inpfile=str(onl),
            action="list",
            savepars=True,
            flagbackup=False,
        )
    clear_history(raw_ms)
    assert_unlocked(raw_ms)
    summary["online_flag_commands"] = len(lines)
    return raw_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--extracted", type=Path, required=True)
    ap.add_argument("--skip-reference-image", action="store_true")
    args = ap.parse_args()
    out_root = args.out_root.resolve()
    asdm = (args.extracted / ASDM_NAME).resolve()
    if not (asdm / "ASDM.xml").is_file():
        raise SystemExit(f"error: ASDM not found at {asdm} (run `run.py fetch` first)")
    _setup_env(out_root)
    os.chdir(out_root)

    import casatasks

    timings: dict[str, float] = {}
    summary: dict[str, Any] = {
        "suite_id": "vla_3c129_pband",
        "slice": {
            "spw": SLICE_SPW,
            "hanning": True,
            "chanbin": CHANBIN,
            "timebin": TIMEBIN,
            "scans": "all",
            "correlations": "all",
        },
        "asdm": str(asdm),
    }

    # 1. raw slice
    t0 = time.perf_counter()
    raw_ms = step_slice(out_root, asdm, summary)
    raw_hash = raw_data_blake2b(raw_ms)
    raw_meta = inspect_ms_meta(raw_ms)
    bp_el_min = round(min_elevation_deg(raw_ms, FLUX_CAL), 4)
    raw_meta.update(
        {
            "raw_data_blake2b": raw_hash,
            "bandpass_min_elevation_deg": bp_el_min,
            "flux_cal_raw_median_amp_per_antenna": raw_antenna_median_amps(raw_ms),
        }
    )
    timings["slice_s"] = round(time.perf_counter() - t0, 1)
    write_json(out_root / "raw_summary.json", raw_meta)
    summary["raw"] = raw_meta
    print(
        f"[1/5] raw slice {raw_meta['size_mb']} MB rows={raw_meta['nrows']} el_min={bp_el_min}",
        flush=True,
    )

    # 2. post_precal: guide sections 7a-8 (online, ea19, scans 1~2, clip, shadow, tfcrop, setjy,
    #    initial G0/K0/B0 + rflag on the bandpass calibrator)
    t0 = time.perf_counter()
    pre_dir = out_root / "post_precal"
    if pre_dir.exists():
        shutil.rmtree(pre_dir)
    pre_dir.mkdir(parents=True)
    pre_ms = pre_dir / "input.ms"
    shutil.copytree(raw_ms, pre_ms)

    with open_tb(pre_ms) as tb:
        field_id = tb.getcol("FIELD_ID")
        ddid = tb.getcol("DATA_DESC_ID")
        cross = tb.getcol("ANTENNA1") != tb.getcol("ANTENNA2")
    raw_flag = getcol_all(pre_ms, "FLAG").astype(bool)
    with open_tb(pre_ms / "FIELD") as tb:
        fnames = [str(x) for x in tb.getcol("NAME")]
    flux_fid = fnames.index(FLUX_CAL)

    fd = casatasks.flagdata
    casatasks.flagcmd(vis=str(pre_ms), inpmode="table", action="apply", flagbackup=False)
    fd(vis=str(pre_ms), mode="manual", antenna=",".join(DEAD_ANTENNAS), flagbackup=False)
    fd(vis=str(pre_ms), mode="manual", scan=GUIDE_DUMMY_SCANS, flagbackup=False)
    fd(vis=str(pre_ms), mode="clip", clipzeros=True, flagbackup=False)
    fd(vis=str(pre_ms), mode="shadow", tolerance=0.0, flagbackup=False)
    must_det = getcol_all(pre_ms, "FLAG").astype(bool)
    det_all = must_det

    fd(
        vis=str(pre_ms),
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
        vis=str(pre_ms),
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
    fd(vis=str(pre_ms), mode="extend", flagbackup=False)
    tfcrop_flag = getcol_all(pre_ms, "FLAG").astype(bool)

    casatasks.setjy(
        vis=str(pre_ms),
        field=FLUX_CAL,
        model="3C147_P.im",
        standard="Perley-Butler 2017",
        usescratch=True,
        scalebychan=True,
        spw="",
    )

    rq_pre = pre_dir / "requantizer.rq"
    antpos_pre = pre_dir / "antpos.ant"
    casatasks.gencal(vis=str(pre_ms), caltable=str(rq_pre), caltype="rq")
    antpos_table(pre_ms, antpos_pre)
    prior = [str(antpos_pre), str(rq_pre)]

    rfi_dir = out_root / "_tmp" / "precal_rfi"
    if rfi_dir.exists():
        shutil.rmtree(rfi_dir)
    rfi_dir.mkdir(parents=True)
    rfi_ms = rfi_dir / "work.ms"
    shutil.copytree(pre_ms, rfi_ms)
    g0, k0, b0 = (rfi_dir / n for n in ("G0.cal", "K0.cal", "B0.cal"))
    casatasks.gaincal(
        vis=str(rfi_ms),
        caltable=str(g0),
        gaintype="G",
        calmode="p",
        solint="int",
        field=FLUX_CAL,
        refant=REFANT,
        gaintable=prior,
    )
    casatasks.gaincal(
        vis=str(rfi_ms),
        caltable=str(k0),
        gaintype="K",
        solint="inf",
        field=FLUX_CAL,
        refant=REFANT,
        gaintable=[*prior, str(g0)],
    )
    casatasks.bandpass(
        vis=str(rfi_ms),
        caltable=str(b0),
        solint="inf",
        field=FLUX_CAL,
        refant=REFANT,
        minsnr=2.0,
        gaintable=[*prior, str(g0), str(k0)],
    )
    casatasks.applycal(
        vis=str(rfi_ms),
        field=FLUX_CAL,
        applymode="calflagstrict",
        gaintable=[*prior, str(g0), str(k0), str(b0)],
        flagbackup=False,
    )
    # Classification-only pass: K0/B0 on the target too, so target RFI amplitudes can be ranked.
    casatasks.applycal(
        vis=str(rfi_ms),
        field=TARGET,
        applymode="calonly",
        gaintable=[*prior, str(k0), str(b0)],
        gainfield=["", "", FLUX_CAL, FLUX_CAL],
        flagbackup=False,
    )
    binit_corr = getcol_all(rfi_ms, "CORRECTED_DATA")
    binit_flag = getcol_all(rfi_ms, "FLAG").astype(bool)
    strong, mild = parallel_amp_mad_classes(binit_corr, binit_flag, field_id, ddid, cross)
    del binit_corr, binit_flag
    fd(
        vis=str(rfi_ms),
        field=FLUX_CAL,
        mode="rflag",
        datacolumn="corrected",
        timedevscale=4.0,
        freqdevscale=3.0,
        combinescans=True,
        ntime="3600s",
        flagbackup=False,
    )
    golden = getcol_all(rfi_ms, "FLAG").astype(bool)
    # calflagstrict may flag bandpass-cal rows lacking solutions; keep those as RFI-pass flags.
    shutil.rmtree(rfi_dir)

    raw_2d = np.any(raw_flag, axis=0)
    must_det_2d = np.any(must_det, axis=0)
    det_all_2d = np.any(det_all, axis=0)
    tfcrop_2d = np.any(tfcrop_flag, axis=0)
    golden_2d = np.any(golden, axis=0)
    rfi_only_2d = golden_2d & (~det_all_2d)
    edge_2d = np.zeros_like(raw_2d)
    nch = raw_2d.shape[0]
    edge_2d[list(range(EDGE_CHANS)) + list(range(nch - EDGE_CHANS, nch)), :] = True
    must_2d = (must_det_2d & (~raw_2d)) | (rfi_only_2d & strong)
    neutral_2d = (
        raw_2d | (det_all_2d & ~must_det_2d) | tfcrop_2d | golden_2d | mild | edge_2d
    ) & ~must_2d
    clean_2d = ~(must_2d | neutral_2d)
    assert not np.any(must_2d & neutral_2d)
    assert np.all(must_2d | neutral_2d | clean_2d)

    with open_tb(pre_ms, nomodify=False) as tb:
        tb.putcol("FLAG", golden)
        tb.putcol("FLAG_ROW", np.all(golden, axis=(0, 1)))
        if "CORRECTED_DATA" in tb.colnames():
            tb.removecols(["CORRECTED_DATA"])
    model = getcol_all(pre_ms, "MODEL_DATA")
    par_mod = 0.5 * (np.abs(model[0]) + np.abs(model[3]))
    flux_rows = field_id == flux_fid
    par_good = (~np.any(golden[[0, 3]], axis=0)) & flux_rows[np.newaxis, :]
    flux_mean_jy = float(np.mean(par_mod[par_good]))
    spw0_rows = flux_rows & (ddid == int(np.min(ddid)))  # lowest DDID present (slice has no DDID 0)
    spw0_spec = [round(float(np.mean(par_mod[ch, spw0_rows])), 6) for ch in range(nch)]
    del model, par_mod
    set_intents(pre_ms)

    cal_ms = pre_dir / "calibrators.ms"
    casatasks.mstransform(
        vis=str(pre_ms), outputvis=str(cal_ms), field=FLUX_CAL, datacolumn="all", keepflags=False
    )
    for ms in (pre_ms, cal_ms):
        clear_history(ms)
        assert_unlocked(ms)
    masks_npz = pre_dir / "precal_masks.npz"
    np.savez_compressed(
        masks_npz,
        shape_2d=np.array(must_2d.shape, dtype=np.int64),
        must_flag_2d=np.packbits(must_2d, axis=1),
        neutral_2d=np.packbits(neutral_2d, axis=1),
        clean_2d=np.packbits(clean_2d, axis=1),
        ref_flux_cal_spectrum_jy=np.array(spw0_spec, dtype=np.float64),
    )
    pre_hash = raw_data_blake2b(pre_ms)
    timings["precal_s"] = round(time.perf_counter() - t0, 1)
    total_cells = int(golden.size)
    pre_summary = {
        "input_ms": inspect_ms_meta(pre_ms),
        "calibrators_ms": inspect_ms_meta(cal_ms),
        "raw_data_blake2b": pre_hash,
        "flagged_fraction_golden": round(float(np.sum(golden)) / total_cells, 6),
        "flagged_fraction_deterministic": round(float(np.sum(must_det)) / total_cells, 6),
        "flux_cal_model_mean_jy": round(flux_mean_jy, 6),
        "flux_cal_spw0_spectrum_jy": spw0_spec,
        "mask_counts": {
            "must_flag": int(must_2d.sum()),
            "neutral": int(neutral_2d.sum()),
            "clean": int(clean_2d.sum()),
            "total": int(must_2d.size),
        },
    }
    del golden, must_det, tfcrop_flag, raw_flag
    write_json(out_root / "precal_summary.json", pre_summary)
    summary["precal"] = pre_summary
    print(
        f"[2/5] post_precal model={flux_mean_jy:.4f} Jy flagged={pre_summary['flagged_fraction_golden']}",
        flush=True,
    )

    # 3. post_calibrate: guide sections 10-13 (K1, B1, G1, smoothcal, applycal, target rflag, statwt)
    t0 = time.perf_counter()
    cal_dir = out_root / "post_calibrate"
    if cal_dir.exists():
        shutil.rmtree(cal_dir)
    cal_dir.mkdir(parents=True)
    w_ms = cal_dir / "input.ms"
    w_cal = cal_dir / "calibrators.ms"
    shutil.copytree(pre_ms, w_ms)
    shutil.copytree(cal_ms, w_cal)
    rq = cal_dir / "requantizer.rq"
    antpos = cal_dir / "antpos.ant"
    casatasks.gencal(vis=str(w_cal), caltable=str(rq), caltype="rq")
    antpos_table(w_cal, antpos)
    prior = [str(antpos), str(rq)]
    k1, b1, g1, gs1 = (cal_dir / n for n in ("K1.cal", "B1.cal", "G1.cal", "Gs1.cal"))
    casatasks.gaincal(
        vis=str(w_cal),
        caltable=str(k1),
        field=FLUX_CAL,
        solint="inf",
        refant=REFANT,
        gaintype="K",
        gaintable=prior,
        parang=True,
    )
    casatasks.bandpass(
        vis=str(w_cal),
        caltable=str(b1),
        field=FLUX_CAL,
        solint="inf",
        refant=REFANT,
        minsnr=3.0,
        parang=True,
        gaintable=[*prior, str(k1)],
        interp=["", "", "nearest,nearestflag"],
    )
    casatasks.gaincal(
        vis=str(w_cal),
        caltable=str(g1),
        field=FLUX_CAL,
        solint="int",
        refant=REFANT,
        minsnr=3.0,
        gaintype="G",
        calmode="ap",
        gaintable=[*prior, str(k1), str(b1)],
        interp=["", "", "nearest,nearestflag", "nearest,nearestflag"],
        parang=True,
    )
    casatasks.smoothcal(
        vis=str(w_cal), tablein=str(g1), caltable=str(gs1), smoothtype="median", smoothtime=3600.0
    )
    full = [*prior, str(k1), str(b1), str(gs1)]
    interp = ["", "", "nearest,nearestflag", "nearest,nearestflag", "nearest,nearestflag"]
    for vis in (w_ms, w_cal):
        casatasks.applycal(
            vis=str(vis),
            parang=True,
            applymode="calflagstrict",
            gaintable=full,
            interp=interp,
            flagbackup=False,
        )
    fd(vis=str(w_ms), field=TARGET, mode="rflag", datacolumn="corrected", flagbackup=False)
    fd(vis=str(w_ms), field=TARGET, mode="extend", flagbackup=False)
    casatasks.statwt(vis=str(w_ms), field=TARGET, datacolumn="corrected", timebin=30)
    with open_tb(w_ms, nomodify=False) as tb:  # imaging input needs DATA + CORRECTED_DATA only
        if "MODEL_DATA" in tb.colnames():
            tb.removecols(["MODEL_DATA"])
    for ms in (w_ms, w_cal):
        clear_history(ms)
        assert_unlocked(ms)
    cal_hash = raw_data_blake2b(w_ms)
    with open_tb(w_ms) as tb:
        tf = tb.getcol("FIELD_ID")
        tc = tb.getcol("ANTENNA1") != tb.getcol("ANTENNA2")
    flg = getcol_all(w_ms, "FLAG").astype(bool)
    tsel = (tf == fnames.index(TARGET)) & tc
    tgt_ret = float(np.mean(~flg[[0, flg.shape[0] - 1]][:, :, tsel]))
    del flg
    with open_tb(k1) as tb:
        delays = np.asarray(tb.getcol("FPARAM"))
        dflag = np.asarray(tb.getcol("FLAG")).astype(bool)
    good_d = delays[~dflag]
    timings["calibrate_s"] = round(time.perf_counter() - t0, 1)
    cal_summary = {
        "input_ms": inspect_ms_meta(w_ms),
        "raw_data_blake2b": cal_hash,
        "target_parallel_retention": round(tgt_ret, 6),
        "k1_delay_abs_max_ns": round(float(np.max(np.abs(good_d))), 4) if good_d.size else None,
        "k1_delay_abs_median_ns": round(float(np.median(np.abs(good_d))), 4)
        if good_d.size
        else None,
        "caltables": sorted(
            p.name for p in cal_dir.iterdir() if p.suffix in (".cal", ".rq", ".ant")
        ),
    }
    write_json(out_root / "cal_summary.json", cal_summary)
    summary["calibrate"] = cal_summary
    print(
        f"[3/5] post_calibrate retention={tgt_ret:.4f} k1_max={cal_summary['k1_delay_abs_max_ns']} ns",
        flush=True,
    )

    # 4. reference image: guide section 14 tclean on the slice (non-interactive). Reference only;
    #    no rubric target is taken from it.
    img_summary: dict[str, Any] = {"skipped": bool(args.skip_reference_image)}
    if not args.skip_reference_image:
        t0 = time.perf_counter()
        img_dir = out_root / "reference_image"
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True)
        prefix = img_dir / "3C129_initial_clean"
        casatasks.tclean(
            vis=str(w_ms),
            field=TARGET,
            imagename=str(prefix),
            datacolumn="corrected",
            cell=["5.0arcsec", "5.0arcsec"],
            imsize=[4860, 4860],
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
        casatasks.clearstat()
        stats = casatasks.imstat(imagename=f"{prefix}.image.tt0")
        rstats = casatasks.imstat(imagename=f"{prefix}.residual.tt0")
        hdr = casatasks.imhead(imagename=f"{prefix}.image.tt0", mode="summary")
        beam = hdr.get("restoringbeam", {}) if isinstance(hdr, dict) else {}
        img_summary = {
            "imagename_prefix": str(prefix),
            "tclean": "guide section 14 parameters, interactive=False, niter=10000, usemask=auto-multithresh",
            "peak_jy_per_beam": round(float(stats["max"][0]), 6),
            "residual_rms_jy_per_beam": round(float(rstats["rms"][0]), 8),
            "beam_major_arcsec": round(float(beam["major"]["value"]), 3)
            if "major" in beam
            else None,
            "beam_minor_arcsec": round(float(beam["minor"]["value"]), 3)
            if "minor" in beam
            else None,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    write_json(out_root / "image_summary.json", img_summary)
    summary["image"] = img_summary
    print(f"[4/5] reference image {img_summary}", flush=True)

    # 4b. DR floors measured with the grader's own extractors (24A-376 factors 0.55 / 0.70)
    if args.skip_reference_image:
        raise SystemExit("DR floors need the reference image; rerun without --skip-reference-image")
    import tomllib

    from radio_bench.grading.extractors.calibrate import extract_calibrate
    from radio_bench.grading.extractors.image import extract_image
    from radio_bench.grading.schema import SuiteManifest

    manifest = SuiteManifest.model_validate(
        tomllib.loads(
            (Path(__file__).parent / "suites" / SUITE_ID / "suite.toml").read_text(encoding="utf-8")
        )
    )
    img_obs, _ = extract_image(
        out_root / "reference_image", manifest.to_extract_config(TASK_IDS["image"])
    )
    ref_dr = img_obs["image.dynamic_range"].value
    cal_obs, _ = extract_calibrate(cal_dir, manifest.to_extract_config(TASK_IDS["calibrate"]))
    ver_dr = cal_obs["cal.verifier_tclean_dr"].value
    if not ref_dr or not ver_dr:
        raise SystemExit(f"DR floor measurement failed: ref_dr={ref_dr} ver_dr={ver_dr}")
    TOL_IMAGE["dynamic_range_min"] = round(0.55 * float(ref_dr), 1)
    TOL_CALIBRATE["verifier_tclean_dr_min"] = round(0.70 * float(ver_dr), 1)
    summary["dr_floors"] = {"reference_dr": ref_dr, "verifier_dr": ver_dr}
    print(f"[4b/5] DR floors: ref_dr={ref_dr} ver_dr={ver_dr}", flush=True)

    # 5. per-task ground truth (targets: GUIDE constants, PRODUCT measurements, CONVENTION tolerances)
    inspect_gt = {
        "raw_data_blake2b": raw_hash,
        "flux_calibrator": FLUX_CAL,
        "bandpass_calibrator": FLUX_CAL,
        "phase_calibrator": FLUX_CAL,
        "target_fields": [TARGET],
        "mosaic_target_fields": [TARGET],
        "dead_antennas": DEAD_ANTENNAS,
        "missing_vla_pads": MISSING_STATIONS,
        "bandpass_min_elevation_deg": bp_el_min,
        "tolerances": TOL_INSPECT,
    }
    precal_gt = {
        "raw_data_blake2b": raw_hash,
        "flux_cal_model_parallel_mean_jy": round(flux_mean_jy, 6),
        "flux_cal_model_parallel_per_chan_jy": spw0_spec,
        "tolerances": TOL_PRECAL,
    }
    calibrate_gt = {
        "raw_data_blake2b": pre_hash,
        "guide": GUIDE_TRUTH,
        "tolerances": TOL_CALIBRATE,
    }
    image_gt = {"raw_data_blake2b": cal_hash, "guide": GUIDE_TRUTH, "tolerances": TOL_IMAGE}
    e2e_gt = {
        "raw_data_blake2b": raw_hash,
        "calibrate_ground_truth": {**calibrate_gt, "raw_data_blake2b": raw_hash},
        "image_ground_truth": {**image_gt, "raw_data_blake2b": raw_hash},
    }
    for stage, gt in (
        ("inspect", inspect_gt),
        ("precal", precal_gt),
        ("calibrate", calibrate_gt),
        ("image", image_gt),
        ("e2e", e2e_gt),
    ):
        tdir = out_root / TASK_IDS[stage]
        tdir.mkdir(parents=True, exist_ok=True)
        write_json(tdir / "ground_truth.json", gt)
        if stage in ("precal", "e2e"):
            shutil.copy2(masks_npz, tdir / "precal_masks.npz")

    summary["timings_s"] = timings
    summary["work_full_ms_kept"] = str(out_root / "_work" / "full.ms")
    write_json(out_root / "summary.json", summary)
    print(f"[5/5] done; delivered sizes: raw={raw_meta['size_mb']} MB", flush=True)


if __name__ == "__main__":
    main()
