"""Deterministic checkpoint + ground-truth builder for the vla_snr_g55 reference suite.

Needs only casatools, casatasks and numpy (runs inside the casa-base container).
Reads the extracted guide archives under <suite_root>/extracted/ and writes:

  raw/input.ms                     uncalibrated MS, HISTORY cleared
  post_precal/input.ms             guide flagging recipe applied, 3C147 MODEL_DATA set
  post_precal/calibrators.ms       calibrator fields split from post_precal/input.ms
  post_precal/precal_masks.npz     must / neutral / clean reference masks
  guide_calibrated/input.ms        guide-shipped calibrated MS, HISTORY cleared
  <task_id>/ground_truth.json      one per task
  build_summary.json

Every target and tolerance is read from REFERENCE.json; this script only adds
checksums, guide-product measurements, and the flag masks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RAW_SRC = "SNR_G55_10s.ms"
CAL_SRC = "SNR_G55_10s.calib.ms"
FLUX_CAL = "0542+498=3C147"
PHASE_CAL = "J1925+2106"
TARGET = "G55.7+3.4"
CAL_FIELDS = f"{PHASE_CAL},{FLUX_CAL}"
REFANT = "ea24"
INIT_PHASE_SPW = "0:20~24,1:49~52,2:38~41,3:41~44"
TASK_IDS = ("g1_g55_inspect", "g1_g55_precal", "g1_g55_calibrate", "g1_g55_image", "g2_g55_e2e")
HASH_COLS = ("TIME", "ANTENNA1", "ANTENNA2", "FIELD_ID", "DATA_DESC_ID", "DATA")
CHUNK = 50_000


# ---------------------------------------------------------------------------
# Table helpers (every handle closed in finally)
# ---------------------------------------------------------------------------


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


def drop_lock_files(ms: Path) -> None:
    for lock in ms.rglob("table.lock"):
        lock.unlink()


def copy_ms(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    drop_lock_files(dst)


def clear_history(ms: Path) -> None:
    if (ms / "HISTORY" / "table.dat").is_file():
        with open_tb(ms / "HISTORY", nomodify=False) as tb:
            n = int(tb.nrows())
            if n > 0:
                tb.removerows(list(range(n)))


def assert_unlocked(ms: Path) -> None:
    import casatasks

    casatasks.clearstat()
    with open_tb(ms) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active table lock on {ms}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def raw_data_blake2b(ms: Path) -> str:
    """Byte-identical copy of radio_bench.grading.extractors.common.compute_raw_data_blake2b."""
    hasher = hashlib.blake2b(digest_size=32)
    with open_tb(ms) as tb:
        hasher.update(f"total_nrows={int(tb.nrows())};".encode())
        ddids = sorted(int(x) for x in np.unique(np.asarray(tb.getcol("DATA_DESC_ID"))))
        for ddid in ddids:
            sub = tb.query(f"DATA_DESC_ID == {ddid}")
            try:
                n = int(sub.nrows())
                for sr in range(0, n, CHUNK):
                    nr = min(CHUNK, n - sr)
                    hasher.update(f"ddid={ddid}:sr={sr}:nr={nr};".encode())
                    for col in HASH_COLS:
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
    return hasher.hexdigest()


def cross_check_hash(ms: Path, digest: str) -> str:
    """If radio_bench is importable, confirm the local hash matches the verifier's."""
    try:
        from radio_bench.grading.extractors.common import compute_raw_data_blake2b
    except Exception:
        return "radio_bench not importable; not cross-checked"
    other = compute_raw_data_blake2b(ms)
    if other != digest:
        raise RuntimeError(f"blake2b mismatch on {ms}: local={digest} radio_bench={other}")
    return "matches radio_bench.compute_raw_data_blake2b"


def field_names(ms: Path) -> list[str]:
    with open_tb(ms / "FIELD") as tb:
        return [str(x) for x in tb.getcol("NAME")]


def ant_names(ms: Path) -> list[str]:
    with open_tb(ms / "ANTENNA") as tb:
        return [str(x) for x in tb.getcol("NAME")]


def ddid_spw(ms: Path) -> list[int]:
    with open_tb(ms / "DATA_DESCRIPTION") as tb:
        return [int(x) for x in tb.getcol("SPECTRAL_WINDOW_ID")]


def read_flags_2d(ms: Path) -> np.ndarray:
    """(nchan, nrow) = any over correlations of FLAG, OR FLAG_ROW."""
    with open_tb(ms) as tb:
        n = int(tb.nrows())
        out: np.ndarray | None = None
        for sr in range(0, n, CHUNK * 4):
            nr = min(CHUNK * 4, n - sr)
            f = np.asarray(tb.getcol("FLAG", startrow=sr, nrow=nr), dtype=bool)
            fr = np.asarray(tb.getcol("FLAG_ROW", startrow=sr, nrow=nr), dtype=bool)
            if out is None:
                out = np.zeros((f.shape[1], n), dtype=bool)
            out[:, sr : sr + nr] = np.any(f, axis=0) | fr[np.newaxis, :]
    assert out is not None
    return out


def row_keys(ms: Path, ant_map: list[int] | None = None) -> np.ndarray:
    """int64 key per row from (TIME, ANTENNA1, ANTENNA2, SPW) with optional antenna remap."""
    spw_of = np.asarray(ddid_spw(ms), dtype=np.int64)
    with open_tb(ms) as tb:
        t = np.rint(np.asarray(tb.getcol("TIME"), dtype=np.float64) * 100.0).astype(np.int64)
        a1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int64)
        a2 = np.asarray(tb.getcol("ANTENNA2"), dtype=np.int64)
        dd = np.asarray(tb.getcol("DATA_DESC_ID"), dtype=np.int64)
    if ant_map is not None:
        amap = np.asarray(ant_map, dtype=np.int64)
        a1, a2 = amap[a1], amap[a2]
    return ((t * 64 + a1) * 64 + a2) * 16 + spw_of[dd]


# ---------------------------------------------------------------------------
# Measurements on guide-shipped products
# ---------------------------------------------------------------------------


def min_elevation_deg(ms: Path, fname: str) -> float:
    from casatools import measures, msmetadata, quanta

    md = msmetadata()
    me = measures()
    qa = quanta()
    try:
        md.open(str(ms))
        me.doframe(me.observatory("VLA"))
        fid = [str(x) for x in md.fieldnames()].index(fname)
        pc = md.phasecenter(fid)
        times = np.unique(np.concatenate([md.timesforscan(s) for s in md.scansforfield(fid)]))
        els = []
        for t in times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t), "s")))
            els.append(float(qa.convert(me.measure(pc, "AZEL")["m1"], "deg")["value"]))
        return float(min(els))
    finally:
        with contextlib.suppress(Exception):
            md.close()
        with contextlib.suppress(Exception):
            md.done()
        with contextlib.suppress(Exception):
            me.done()
        with contextlib.suppress(Exception):
            qa.done()


def product_target_retention(raw_ms: Path, cal_ms: Path) -> float:
    """Unflagged parallel-hand cells on target cross rows in the guide MS / raw target cells."""
    tgt_fid = field_names(raw_ms).index(TARGET)
    with open_tb(raw_ms) as tb:
        fid = np.asarray(tb.getcol("FIELD_ID"))
        a1 = np.asarray(tb.getcol("ANTENNA1"))
        a2 = np.asarray(tb.getcol("ANTENNA2"))
    raw_rows = int(np.sum((fid == tgt_fid) & (a1 != a2)))
    unflagged = 0
    nchan = 0
    with open_tb(cal_ms) as tb:
        c1 = np.asarray(tb.getcol("ANTENNA1"))
        c2 = np.asarray(tb.getcol("ANTENNA2"))
        n = int(tb.nrows())
        for sr in range(0, n, CHUNK * 4):
            nr = min(CHUNK * 4, n - sr)
            f = np.asarray(tb.getcol("FLAG", startrow=sr, nrow=nr), dtype=bool)
            fr = np.asarray(tb.getcol("FLAG_ROW", startrow=sr, nrow=nr), dtype=bool)
            cross = c1[sr : sr + nr] != c2[sr : sr + nr]
            eff = f | fr[np.newaxis, np.newaxis, :]
            par = [0, eff.shape[0] - 1]
            unflagged += int(np.sum(~eff[par][:, :, cross]))
            nchan = int(f.shape[1])
    return unflagged / float(raw_rows * nchan * 2)


def guide_target_flags_2d(pre_ms: Path, cal_ms: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Map the guide-shipped target flags onto pre_ms rows.

    Returns (flags_2d for target rows, target row numbers, n target rows absent from guide MS).
    Target rows absent from the guide MS were dropped by the guide's split and count as flagged.
    """
    raw_ants = ant_names(pre_ms)
    cal_ants = ant_names(cal_ms)
    ant_map = [raw_ants.index(a) if a in raw_ants else 63 for a in cal_ants]
    if ddid_spw(pre_ms) != ddid_spw(cal_ms):
        raise RuntimeError("DATA_DESCRIPTION spw maps differ between raw and guide MS")
    tgt_fid = field_names(pre_ms).index(TARGET)
    with open_tb(pre_ms) as tb:
        fid = np.asarray(tb.getcol("FIELD_ID"))
    tgt_rows = np.flatnonzero(fid == tgt_fid)
    raw_keys = row_keys(pre_ms)[tgt_rows]
    cal_keys = row_keys(cal_ms, ant_map)
    order = np.argsort(raw_keys, kind="stable")
    sorted_keys = raw_keys[order]
    if np.any(np.diff(sorted_keys) == 0):
        raise RuntimeError("Duplicate (TIME, ANT1, ANT2, SPW) keys on raw target rows")
    pos = np.searchsorted(sorted_keys, cal_keys)
    pos = np.clip(pos, 0, sorted_keys.size - 1)
    if not np.all(sorted_keys[pos] == cal_keys):
        raise RuntimeError("Guide MS rows not all found in the raw MS target rows")
    cal_flags = read_flags_2d(cal_ms)
    out = np.ones((cal_flags.shape[0], tgt_rows.size), dtype=bool)
    out[:, order[pos]] = cal_flags
    n_absent = int(tgt_rows.size - cal_keys.size)
    return out, tgt_rows, n_absent


# ---------------------------------------------------------------------------
# Guide recipe steps (same calls as the oracle solve.py files)
# ---------------------------------------------------------------------------


def repair_flux_intent(ms: Path) -> None:
    """Add CALIBRATE_FLUX to the flux calibrator's STATE rows (the MS carries CALIBRATE_AMPLI)."""
    flux_fid = field_names(ms).index(FLUX_CAL)
    with open_tb(ms) as tb:
        fid = np.asarray(tb.getcol("FIELD_ID"))
        sid = np.asarray(tb.getcol("STATE_ID"))
    flux_sids = sorted({int(s) for s in np.unique(sid[fid == flux_fid]) if s >= 0})
    other_sids = {int(s) for s in np.unique(sid[fid != flux_fid]) if s >= 0}
    with open_tb(ms / "STATE", nomodify=False) as st:
        for s in flux_sids:
            if s in other_sids:
                continue
            mode = str(st.getcell("OBS_MODE", s))
            if "CALIBRATE_FLUX" not in mode:
                st.putcell(
                    "OBS_MODE",
                    s,
                    f"{mode},CALIBRATE_FLUX#UNSPECIFIED" if mode else "CALIBRATE_FLUX#UNSPECIFIED",
                )


def apply_deterministic_flags(ms: Path) -> None:
    import casatasks

    casatasks.flagcmd(vis=str(ms), inpmode="table", reason="any", action="apply", flagbackup=False)
    casatasks.flagdata(vis=str(ms), mode="shadow", tolerance=0.0, flagbackup=False)
    casatasks.flagdata(vis=str(ms), mode="clip", clipzeros=True, flagbackup=False)
    casatasks.flagdata(
        vis=str(ms), mode="quack", quackinterval=5.0, quackmode="beg", flagbackup=False
    )


def apply_guide_rfi_flags(ms: Path, scratch: Path) -> None:
    import casatasks

    for spw in ("0", "1~3"):
        casatasks.flagdata(
            vis=str(ms),
            mode="tfcrop",
            spw=spw,
            datacolumn="data",
            action="apply",
            display="",
            flagbackup=False,
        )
    scratch.mkdir(parents=True, exist_ok=True)
    init_ph = scratch / "initPh.cal"
    init_bp = scratch / "initBP.cal"
    for tab in (init_ph, init_bp):
        if tab.exists():
            shutil.rmtree(tab)
    casatasks.gaincal(
        vis=str(ms),
        caltable=str(init_ph),
        field=PHASE_CAL,
        solint="int",
        spw=INIT_PHASE_SPW,
        refant=REFANT,
        minblperant=3,
        minsnr=3.0,
        calmode="p",
    )
    casatasks.bandpass(
        vis=str(ms),
        caltable=str(init_bp),
        field=PHASE_CAL,
        solint="inf",
        combine="scan",
        refant=REFANT,
        minblperant=3,
        minsnr=10.0,
        gaintable=[str(init_ph)],
        interp=["nearest"],
        solnorm=False,
    )
    casatasks.applycal(vis=str(ms), gaintable=[str(init_bp)], calwt=False, flagbackup=False)
    for spw, fdev, tdev in (("0", 2.5, 3.5), ("1,3", 4.0, 4.0), ("2", 2.5, 2.5)):
        casatasks.flagdata(
            vis=str(ms),
            mode="rflag",
            spw=spw,
            datacolumn="corrected",
            freqdevscale=fdev,
            timedevscale=tdev,
            action="apply",
            display="",
            flagbackup=False,
        )
        casatasks.flagdata(
            vis=str(ms),
            mode="extend",
            spw=spw,
            extendpols=True,
            action="apply",
            display="",
            flagbackup=False,
        )
        casatasks.flagdata(
            vis=str(ms),
            mode="extend",
            spw=spw,
            growtime=50.0,
            growfreq=90.0,
            action="apply",
            display="",
            flagbackup=False,
        )
    casatasks.flagdata(vis=str(ms), scan="87,88", antenna="ea16", flagbackup=False)
    for bl in ("ea04&ea16", "ea02&ea27", "ea02&ea08"):
        casatasks.flagdata(vis=str(ms), antenna=bl, spw="1", flagbackup=False)
    casatasks.clearstat()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def ref_value(ref: dict[str, Any], name: str) -> Any:
    entry = ref["entries"].get(name)
    if entry is None:
        raise KeyError(f"REFERENCE.json has no entry {name!r}")
    return entry["value"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suite-root", type=Path, required=True, help="<data_dir>/casaguides/vla_snr_g55"
    )
    parser.add_argument("--reference", type=Path, default=HERE / "REFERENCE.json")
    args = parser.parse_args()
    root = args.suite_root.resolve()
    ext = root / "extracted"
    src_raw = ext / RAW_SRC
    src_cal = ext / CAL_SRC
    for p in (src_raw, src_cal):
        if not (p / "table.dat").is_file():
            raise SystemExit(f"error: missing {p} (run `run.py fetch` first)")
    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    scratch = root / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(scratch))
    orig_cwd = Path.cwd()
    os.chdir(scratch)

    import casatasks

    summary: dict[str, Any] = {"suite_root": str(root), "reference": str(args.reference)}

    # 1. raw checkpoint
    raw_ms = root / "raw" / "input.ms"
    copy_ms(src_raw, raw_ms)
    clear_history(raw_ms)
    assert_unlocked(raw_ms)
    raw_hash = raw_data_blake2b(raw_ms)
    summary["raw_hash_check"] = cross_check_hash(raw_ms, raw_hash)
    bp_el = round(min_elevation_deg(raw_ms, FLUX_CAL), 4)
    print(f"[1/5] raw: blake2b={raw_hash[:16]} 3C147 min el={bp_el} deg", flush=True)

    # 2. guide-calibrated checkpoint (image task input)
    gcal_ms = root / "guide_calibrated" / "input.ms"
    copy_ms(src_cal, gcal_ms)
    clear_history(gcal_ms)
    assert_unlocked(gcal_ms)
    gcal_hash = raw_data_blake2b(gcal_ms)
    summary["guide_calibrated_hash_check"] = cross_check_hash(gcal_ms, gcal_hash)
    retention = round(product_target_retention(raw_ms, gcal_ms), 4)
    print(
        f"[2/5] guide_calibrated: blake2b={gcal_hash[:16]} target retention={retention}", flush=True
    )

    for name, measured in (
        ("inspect.bandpass_min_elevation_deg", bp_el),
        ("calibrate.guide_target_row_retention", retention),
    ):
        expected = float(ref_value(ref, name))
        if abs(expected - measured) > 1e-3:
            raise RuntimeError(f"{name}: REFERENCE.json={expected} but measured {measured}")

    # 3. post_precal checkpoint + masks
    pre_dir = root / "post_precal"
    pre_ms = pre_dir / "input.ms"
    copy_ms(raw_ms, pre_ms)
    repair_flux_intent(pre_ms)
    apply_deterministic_flags(pre_ms)
    must_2d = read_flags_2d(pre_ms)
    apply_guide_rfi_flags(pre_ms, scratch / "precal")
    recipe_2d = read_flags_2d(pre_ms)

    guide_tgt_2d, tgt_rows, n_absent = guide_target_flags_2d(pre_ms, gcal_ms)
    golden_2d = recipe_2d.copy()
    golden_2d[:, tgt_rows] = guide_tgt_2d | must_2d[:, tgt_rows]
    must_not_in_guide = int(np.sum(must_2d[:, tgt_rows] & ~guide_tgt_2d))

    with open_tb(pre_ms, nomodify=False) as tb:
        n = int(tb.nrows())
        for sr in range(0, n, CHUNK * 4):
            nr = min(CHUNK * 4, n - sr)
            f = np.asarray(tb.getcol("FLAG", startrow=sr, nrow=nr), dtype=bool)
            g = golden_2d[:, sr : sr + nr]
            f[:] = g[np.newaxis, :, :]
            tb.putcol("FLAG", f, startrow=sr, nrow=nr)
            tb.putcol("FLAG_ROW", np.all(g, axis=0), startrow=sr, nrow=nr)
        if "CORRECTED_DATA" in tb.colnames():
            tb.removecols(["CORRECTED_DATA"])
    casatasks.setjy(vis=str(pre_ms), field=FLUX_CAL, standard="Perley-Butler 2017", usescratch=True)

    cal_ms = pre_dir / "calibrators.ms"
    if cal_ms.exists():
        shutil.rmtree(cal_ms)
    casatasks.mstransform(
        vis=str(pre_ms), outputvis=str(cal_ms), field=CAL_FIELDS, datacolumn="all", keepflags=False
    )
    for ms in (pre_ms, cal_ms):
        clear_history(ms)
        assert_unlocked(ms)
    pre_hash = raw_data_blake2b(pre_ms)

    neutral_2d = golden_2d & ~must_2d
    clean_2d = ~golden_2d
    assert not np.any(must_2d & neutral_2d)
    assert not np.any(must_2d & clean_2d)
    assert np.all(must_2d | neutral_2d | clean_2d)
    masks_npz = pre_dir / "precal_masks.npz"
    np.savez_compressed(
        masks_npz,
        shape_2d=np.array(must_2d.shape, dtype=np.int64),
        must_flag_2d=np.packbits(must_2d, axis=1),
        neutral_2d=np.packbits(neutral_2d, axis=1),
        clean_2d=np.packbits(clean_2d, axis=1),
    )
    summary["precal"] = {
        "mask_counts": {
            "must_flag": int(must_2d.sum()),
            "neutral": int(neutral_2d.sum()),
            "clean": int(clean_2d.sum()),
            "total": int(must_2d.size),
        },
        "golden_flag_fraction": round(float(golden_2d.mean()), 6),
        "target_rows_absent_from_guide_ms": n_absent,
        "target_must_cells_unflagged_in_guide_ms": must_not_in_guide,
        "post_precal_blake2b": pre_hash,
    }
    print(f"[3/5] post_precal: masks={summary['precal']['mask_counts']}", flush=True)
    shutil.rmtree(scratch / "precal", ignore_errors=True)

    # 4. ground truth (values from REFERENCE.json, hashes and masks from this build)
    tol_inspect = {"elevation_abs_deg": ref_value(ref, "inspect.elevation_abs_deg")}
    inspect_gt = {
        "raw_data_blake2b": raw_hash,
        "flux_calibrator": ref_value(ref, "inspect.flux_calibrator"),
        "bandpass_calibrator": ref_value(ref, "inspect.bandpass_calibrator"),
        "phase_calibrator": ref_value(ref, "inspect.phase_calibrator"),
        "target_fields": ref_value(ref, "inspect.target_fields"),
        "dead_antennas": ref_value(ref, "inspect.dead_antennas"),
        "bandpass_min_elevation_deg": bp_el,
        "tolerances": tol_inspect,
    }
    precal_gt = {
        "raw_data_blake2b": raw_hash,
        "tolerances": {
            "flag_recall_min": ref_value(ref, "precal.flag_recall_min"),
            "clean_false_positive_max": ref_value(ref, "precal.clean_false_positive_max"),
        },
    }
    guide_rms = float(ref_value(ref, "image.guide_rms_jy_per_beam"))
    rms_factor = float(ref_value(ref, "image.rms_max_factor"))
    calibrate_gt = {
        "raw_data_blake2b": pre_hash,
        "guide_reference": {
            "target_row_retention": retention,
            "rms_jy_per_beam": guide_rms,
        },
        "tolerances": {
            "target_unflagged_rows_min": ref_value(ref, "calibrate.target_unflagged_rows_min"),
            "target_unflagged_rows_max": ref_value(ref, "calibrate.target_unflagged_rows_max"),
            "bp_channel_phase_rms_max_deg": ref_value(
                ref, "calibrate.bp_channel_phase_rms_max_deg"
            ),
            "verifier_tclean_rms_max_jy_per_beam": round(guide_rms * rms_factor, 8),
        },
    }
    image_gt = {
        "raw_data_blake2b": gcal_hash,
        "guide_reference": {
            "beam_major_arcsec": ref_value(ref, "image.guide_beam_major_arcsec"),
            "beam_minor_arcsec": ref_value(ref, "image.guide_beam_minor_arcsec"),
            "rms_jy_per_beam": guide_rms,
        },
        "tolerances": {
            "pixels_per_beam_min": ref_value(ref, "image.pixels_per_beam_min"),
            "pixels_per_beam_max": ref_value(ref, "image.pixels_per_beam_max"),
            "off_source_residual_rms_max_jy_per_beam": round(guide_rms * rms_factor, 8),
            "off_source_residual_rms_min_jy_per_beam": ref_value(ref, "image.rms_min_jy_per_beam"),
            "negative_model_bowl_ratio_max": ref_value(ref, "image.negative_model_bowl_ratio_max"),
        },
    }
    e2e_gt = {
        "raw_data_blake2b": raw_hash,
        "calibrate_ground_truth": {**calibrate_gt, "raw_data_blake2b": raw_hash},
        "image_ground_truth": {**image_gt, "raw_data_blake2b": raw_hash},
    }
    gts = dict(zip(TASK_IDS, (inspect_gt, precal_gt, calibrate_gt, image_gt, e2e_gt), strict=True))
    for tid, gt in gts.items():
        write_json(root / tid / "ground_truth.json", gt)
        if tid in ("g1_g55_precal", "g2_g55_e2e"):
            shutil.copy2(masks_npz, root / tid / "precal_masks.npz")
    print("[4/5] ground_truth.json written for " + ", ".join(TASK_IDS), flush=True)

    summary["hashes"] = {"raw": raw_hash, "post_precal": pre_hash, "guide_calibrated": gcal_hash}
    summary["product_measurements"] = {
        "bandpass_min_elevation_deg": bp_el,
        "target_row_retention": retention,
    }
    write_json(root / "build_summary.json", summary)
    os.chdir(orig_cwd)  # leave _scratch before deleting it (CASA aborts on a deleted cwd)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[5/5] done: {root / 'build_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
