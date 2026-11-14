"""Deterministic scientific verifier for precal-stage tasks (s1_precal, r1_3c391_precal)."""

from __future__ import annotations

import contextlib
import fcntl
import math
import os
import re
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

PASS_REWARD_MIN_DEFAULT = 0.85


def has_active_table_lock(table_dir: Path) -> bool:
    """Check whether `table_dir` or any immediate subtable holds an active POSIX fcntl lock."""
    if not table_dir.is_dir():
        return False
    lock_files = [table_dir / "table.lock"]
    for child in table_dir.iterdir():
        if child.is_dir() and (child / "table.lock").is_file():
            lock_files.append(child / "table.lock")

    for lf in lock_files:
        if not lf.is_file():
            continue
        try:
            fd = os.open(str(lf), os.O_RDONLY)
        except OSError:
            continue
        try:
            # Linux x86_64 struct flock: l_type(h), l_whence(h), l_start(l), l_len(l), l_pid(i)
            probe = struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0)
            res = fcntl.fcntl(fd, fcntl.F_GETLK, probe)
            l_type = struct.unpack("hhqqi", res)[0]
            if l_type != fcntl.F_UNLCK:
                return True
        except OSError:
            continue
        finally:
            os.close(fd)
    return False


@contextlib.contextmanager
def _open_ro_table(table_path: Path) -> Iterator[Any]:
    import casatools

    tb = casatools.table()
    opened = False
    try:
        opened = bool(
            tb.open(
                str(table_path),
                nomodify=True,
                lockoptions={"option": "usernoread"},
            )
        )
        if not opened:
            raise RuntimeError(f"casatools.table.open returned False for {table_path}")
        yield tb
    finally:
        if opened:
            with contextlib.suppress(Exception):
                tb.close()
        with contextlib.suppress(Exception):
            tb.done()


def _split_intents(raw_intents: list[str] | tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for entry in raw_intents:
        for part in re.split(r"[,;]", str(entry)):
            tok = part.strip()
            if tok:
                tokens.add(tok)
    return tokens


def _read_field_intents(ms_path: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Read field names and split intents on both `,` and `;` safely."""
    import casatools

    with _open_ro_table(ms_path / "FIELD") as tb_f:
        field_names = [str(x).strip() for x in tb_f.getcol("NAME")]

    intents_by_field: dict[str, set[str]] = {fn: set() for fn in field_names}

    # Try msmetadata first; fall back to direct STATE + MAIN lookup if STATE_ID has -1
    msmd = casatools.msmetadata()
    msmd_opened = False
    try:
        msmd_opened = bool(msmd.open(str(ms_path)))
        if msmd_opened:
            for fid, fname in enumerate(field_names):
                raw_list = [str(x) for x in msmd.intentsforfield(fid)]
                intents_by_field[fname] = _split_intents(raw_list)
            return field_names, intents_by_field
    except Exception:
        pass
    finally:
        if msmd_opened:
            with contextlib.suppress(Exception):
                msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()

    state_path = ms_path / "STATE"
    if not (state_path / "table.dat").is_file():
        return field_names, intents_by_field

    with _open_ro_table(state_path) as tb_s:
        if int(tb_s.nrows()) == 0 or "OBS_MODE" not in tb_s.colnames():
            return field_names, intents_by_field
        obs_modes = [str(x) for x in tb_s.getcol("OBS_MODE")]

    with _open_ro_table(ms_path) as tb_m:
        if int(tb_m.nrows()) == 0:
            return field_names, intents_by_field
        fids = tb_m.getcol("FIELD_ID")
        sids = tb_m.getcol("STATE_ID")

    for fid, fname in enumerate(field_names):
        u_sids = np.unique(sids[fids == fid])
        modes = [obs_modes[int(s)] for s in u_sids if 0 <= int(s) < len(obs_modes)]
        intents_by_field[fname] = _split_intents(modes)
    return field_names, intents_by_field


def _has_intent_prefix(intents: set[str], prefix: str) -> bool:
    return any(tok.upper().startswith(prefix.upper()) for tok in intents)


def _extract_precal_truth(ground_truth: dict[str, Any]) -> dict[str, Any]:
    if "precal_ground_truth" in ground_truth and isinstance(
        ground_truth["precal_ground_truth"], dict
    ):
        ground_truth = ground_truth["precal_ground_truth"]

    fields_sub = ground_truth.get("fields") if isinstance(ground_truth.get("fields"), dict) else {}
    flux_cal = str(
        ground_truth.get("flux_calibrator") or fields_sub.get("flux_calibrator") or "J1331+3030"
    ).strip()
    phase_cal = str(
        ground_truth.get("phase_calibrator") or fields_sub.get("phase_calibrator") or "J1822-0938"
    ).strip()

    if "target_fields" in fields_sub and isinstance(fields_sub["target_fields"], list):
        target_fields = [str(x).strip() for x in fields_sub["target_fields"]]
    elif "mosaic_target_fields" in ground_truth and isinstance(
        ground_truth["mosaic_target_fields"], list
    ):
        target_fields = [str(x).strip() for x in ground_truth["mosaic_target_fields"]]
    elif str(ground_truth.get("suite", "")).upper() == "R" or "3c391" in str(
        ground_truth.get("task_id", "")
    ):
        target_fields = [f"3C391 C{i}" for i in range(1, 8)]
    else:
        target_fields = ["SYNTH_MOSAIC_P1", "SYNTH_MOSAIC_P2"]

    pb2017 = (
        ground_truth.get("j1331_pb2017")
        if isinstance(ground_truth.get("j1331_pb2017"), dict)
        else {}
    )
    expected_model_amp = float(
        ground_truth.get("j1331_model_parallel_mean_jy")
        or pb2017.get("mean_parallel_amp_jy")
        or 7.601253
    )
    per_chan_amp = ground_truth.get("j1331_model_parallel_per_chan_jy") or pb2017.get(
        "per_chan_parallel_amp_jy"
    )

    tols = (
        ground_truth.get("tolerances") if isinstance(ground_truth.get("tolerances"), dict) else {}
    )
    model_rel_tol = float(
        tols.get("j1331_model_rel_tol")
        or tols.get("model_amp_rel_tol")
        or pb2017.get("rel_tolerance")
        or 0.01
    )
    recall_min = float(tols.get("min_corrupted_recall") or tols.get("flag_recall_min") or 0.95)
    fpr_max = float(
        tols.get("max_clean_false_positive_rate") or tols.get("clean_false_positive_max") or 0.05
    )
    pass_min = float(
        ground_truth.get("pass_reward_min", tols.get("pass_reward_min", PASS_REWARD_MIN_DEFAULT))
    )

    weights_raw = (
        ground_truth.get("weights") if isinstance(ground_truth.get("weights"), dict) else {}
    )
    weights = {
        "calibrators_ms_valid": float(weights_raw.get("calibrators_ms_valid", 0.20)),
        "model_data_accuracy": float(weights_raw.get("model_data_accuracy", 0.25)),
        "rfi_preflag_recall": float(weights_raw.get("rfi_preflag_recall", 0.30)),
        "clean_false_positive_guard": float(weights_raw.get("clean_false_positive_guard", 0.25)),
    }

    return {
        "flux_calibrator": flux_cal,
        "phase_calibrator": phase_cal,
        "target_fields": target_fields,
        "expected_j1331_model_amp_jy": expected_model_amp,
        "expected_j1331_per_chan_amp_jy": per_chan_amp,
        "model_rel_tol": model_rel_tol,
        "flag_recall_min": recall_min,
        "clean_fpr_max": fpr_max,
        "pass_reward_min": pass_min,
        "weights": weights,
    }


def _load_precal_masks(masks_npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load `(must_flag_2d, clean_2d)` boolean arrays of shape `(nchan, nrow)`."""
    with np.load(masks_npz_path, allow_pickle=False) as z:
        if "corrupted_chan_row" in z.files:
            corrupted = z["corrupted_chan_row"].astype(bool).copy()
            return corrupted, ~corrupted
        if "must_flag_2d" in z.files and "clean_2d" in z.files:
            raw_must = z["must_flag_2d"]
            raw_clean = z["clean_2d"]
            if "shape_2d" in z.files and raw_must.dtype == np.uint8:
                shape_2d = z["shape_2d"]
                nrow = int(shape_2d[1])
                must = np.unpackbits(raw_must, axis=1)[:, :nrow].astype(bool)
                clean = np.unpackbits(raw_clean, axis=1)[:, :nrow].astype(bool)
                return must, clean
            return raw_must.astype(bool).copy(), raw_clean.astype(bool).copy()
    raise ValueError(f"Unrecognized precal mask keys in {masks_npz_path}")


def _measure_j1331_model_amp(ms_path: Path, flux_cal_name: str) -> tuple[float | None, str | None]:
    """Return unflagged parallel-hand (RR, LL) mean MODEL_DATA amplitude on `flux_cal_name`."""
    with _open_ro_table(ms_path / "FIELD") as tb_f:
        fnames = [str(x).strip() for x in tb_f.getcol("NAME")]
    matching_fids = [i for i, n in enumerate(fnames) if n.upper() == flux_cal_name.upper()]
    if not matching_fids:
        return None, f"Field {flux_cal_name} not present in {ms_path.name}"
    fid = matching_fids[0]

    with _open_ro_table(ms_path) as tb:
        if "MODEL_DATA" not in tb.colnames():
            return None, f"MODEL_DATA absent in {ms_path.name}"
        sub = tb.query(f"FIELD_ID == {fid}")
        try:
            if int(sub.nrows()) == 0:
                return None, f"Zero rows for field {flux_cal_name} in {ms_path.name}"
            model = sub.getcol("MODEL_DATA")
            flag = sub.getcol("FLAG")
            flag_row = sub.getcol("FLAG_ROW")
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()

    eff_flag = flag | flag_row[np.newaxis, np.newaxis, :]
    npol = model.shape[0]
    p_indices = (0, 3) if npol >= 4 else (0, npol - 1)
    vals: list[np.ndarray] = []
    for p_idx in p_indices:
        good = ~eff_flag[p_idx]
        if np.any(good):
            vals.append(np.abs(model[p_idx][good]))
    if not vals:
        # Fall back to all rows of flux_cal_name if 100% flagged
        par = 0.5 * (np.abs(model[p_indices[0]]) + np.abs(model[p_indices[1]]))
        amp = float(np.mean(par))
    else:
        amp = float(np.mean(np.concatenate(vals)))

    if not math.isfinite(amp) or amp <= 0.0:
        return None, f"Non-positive/non-finite MODEL_DATA amplitude in {ms_path.name}"
    return amp, None


def _merge_calibrator_flags_into_input(
    in_chan_row_flag: np.ndarray,
    in_fnames: list[str],
    in_fids: np.ndarray,
    in_scan: np.ndarray,
    in_time: np.ndarray,
    in_ant_names: list[str],
    in_a1: np.ndarray,
    in_a2: np.ndarray,
    in_ddid: np.ndarray,
    cal_ms_path: Path,
) -> int:
    """OR `(nchan, nrow_cal)` flags from `calibrators.ms` onto matching `input.ms` calibrator rows."""
    with _open_ro_table(cal_ms_path / "FIELD") as tb_cf:
        cal_fnames = [str(x).strip() for x in tb_cf.getcol("NAME")]
    with _open_ro_table(cal_ms_path / "ANTENNA") as tb_ca:
        cal_ant_names = [str(x).strip() for x in tb_ca.getcol("NAME")]

    name_to_in_fid = {n.upper(): idx for idx, n in enumerate(in_fnames)}
    name_to_in_aid = {n.upper(): idx for idx, n in enumerate(in_ant_names)}

    with _open_ro_table(cal_ms_path) as tb_c:
        if int(tb_c.nrows()) == 0:
            return 0
        cal_flag = tb_c.getcol("FLAG")
        cal_flag_row = tb_c.getcol("FLAG_ROW")
        cal_fids = tb_c.getcol("FIELD_ID")
        cal_scan = tb_c.getcol("SCAN_NUMBER")
        cal_time = tb_c.getcol("TIME")
        cal_a1 = tb_c.getcol("ANTENNA1")
        cal_a2 = tb_c.getcol("ANTENNA2")
        cal_ddid = tb_c.getcol("DATA_DESC_ID")

    if cal_flag.shape[1] != in_chan_row_flag.shape[0]:
        return 0

    cal_eff_2d = np.any(cal_flag, axis=0) | cal_flag_row[np.newaxis, :]

    mapped_fids = np.array(
        [
            name_to_in_fid.get(cal_fnames[int(f)].upper(), -1)
            if 0 <= int(f) < len(cal_fnames)
            else -1
            for f in cal_fids
        ],
        dtype=np.int32,
    )
    mapped_a1 = np.array(
        [
            name_to_in_aid.get(cal_ant_names[int(a)].upper(), int(a))
            if 0 <= int(a) < len(cal_ant_names)
            else int(a)
            for a in cal_a1
        ],
        dtype=np.int32,
    )
    mapped_a2 = np.array(
        [
            name_to_in_aid.get(cal_ant_names[int(a)].upper(), int(a))
            if 0 <= int(a) < len(cal_ant_names)
            else int(a)
            for a in cal_a2
        ],
        dtype=np.int32,
    )

    valid_cal_fids = sorted({int(x) for x in mapped_fids if int(x) >= 0})
    if not valid_cal_fids:
        return 0

    in_cal_rows = np.flatnonzero(np.isin(in_fids, valid_cal_fids))
    t_in_q = np.rint(in_time[in_cal_rows] * 100.0).astype(np.int64)

    lut_scan: dict[tuple[int, int, int, int, int, int], int] = {}
    lut_noscan: dict[tuple[int, int, int, int, int], int] = {}
    for pos, r_idx in enumerate(in_cal_rows):
        r = int(r_idx)
        fid_v = int(in_fids[r])
        sc_v = int(in_scan[r])
        tq_v = int(t_in_q[pos])
        a1_v = int(in_a1[r])
        a2_v = int(in_a2[r])
        dd_v = int(in_ddid[r])
        lut_scan[(fid_v, sc_v, tq_v, a1_v, a2_v, dd_v)] = r
        lut_noscan.setdefault((fid_v, tq_v, a1_v, a2_v, dd_v), r)

    t_cal_q = np.rint(cal_time * 100.0).astype(np.int64)
    matched_in: list[int] = []
    matched_cal: list[int] = []
    for j in range(len(cal_fids)):
        mf = int(mapped_fids[j])
        if mf < 0:
            continue
        k_sc = (
            mf,
            int(cal_scan[j]),
            int(t_cal_q[j]),
            int(mapped_a1[j]),
            int(mapped_a2[j]),
            int(cal_ddid[j]),
        )
        r_match = lut_scan.get(k_sc)
        if r_match is None:
            r_match = lut_noscan.get(
                (mf, int(t_cal_q[j]), int(mapped_a1[j]), int(mapped_a2[j]), int(cal_ddid[j]))
            )
        if r_match is not None:
            matched_in.append(r_match)
            matched_cal.append(j)

    if matched_in:
        in_idx_arr = np.array(matched_in, dtype=np.int64)
        cal_idx_arr = np.array(matched_cal, dtype=np.int64)
        in_chan_row_flag[:, in_idx_arr] |= cal_eff_2d[:, cal_idx_arr]
    return len(matched_in)


def grade_precal(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path,
    task_id: str,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade `/workspace/input.ms` + `/workspace/calibrators.ms` after pre-calibration."""
    from casatasks import clearstat

    with contextlib.suppress(Exception):
        clearstat()

    ws = Path(workspace_dir)
    input_ms = ws / "input.ms"
    cal_ms = ws / "calibrators.ms"
    truth = _extract_precal_truth(ground_truth)
    weights = truth["weights"]
    pass_min = truth["pass_reward_min"]

    errors: list[str] = []
    table_locked = False

    input_exists = (input_ms / "table.dat").is_file()
    cal_exists = (cal_ms / "table.dat").is_file()

    if input_exists and has_active_table_lock(input_ms):
        table_locked = True
        errors.append(f"Active casacore write lock on {input_ms}")
    if cal_exists and has_active_table_lock(cal_ms):
        table_locked = True
        errors.append(f"Active casacore write lock on {cal_ms}")

    # 1. calibrators.ms valid + scan intents fixed (0.20 weight)
    cal_nrows = 0
    cal_fields_ok = False
    cal_intents_ok = False
    target_intents_ok = False
    cal_fnames: list[str] = []
    cal_intents: dict[str, list[str]] = {}
    in_intents: dict[str, list[str]] = {}

    flux_cal = truth["flux_calibrator"]
    phase_cal = truth["phase_calibrator"]
    target_set_upper = {t.upper() for t in truth["target_fields"]}

    in_intents_sets: dict[str, set[str]] = {}
    cal_intents_sets: dict[str, set[str]] = {}

    if input_exists and not has_active_table_lock(input_ms):
        try:
            _in_fn, in_intents_sets = _read_field_intents(input_ms)
            in_intents = {k: sorted(v) for k, v in in_intents_sets.items()}
            if truth["target_fields"]:
                t_checks: list[bool] = []
                for tf in truth["target_fields"]:
                    matched_key = next(
                        (k for k in in_intents_sets if k.upper() == tf.upper()),
                        None,
                    )
                    if matched_key is None:
                        t_checks.append(False)
                    else:
                        t_set = in_intents_sets[matched_key]
                        has_obs = _has_intent_prefix(t_set, "OBSERVE_TARGET")
                        has_pcal = _has_intent_prefix(t_set, "CALIBRATE_PHASE")
                        t_checks.append(has_obs and not has_pcal)
                target_intents_ok = all(t_checks)
        except Exception as exc:
            errors.append(f"Failed reading intents from input.ms: {exc}")

    if cal_exists and not has_active_table_lock(cal_ms):
        try:
            with _open_ro_table(cal_ms) as tb_c:
                if tb_c.ismultiused(checksubtables=True):
                    table_locked = True
                    errors.append("calibrators.ms reported ismultiused=True")
                else:
                    cal_nrows = int(tb_c.nrows())
            cal_fnames, cal_intents_sets = _read_field_intents(cal_ms)
            cal_intents = {k: sorted(v) for k, v in cal_intents_sets.items()}
            cal_upper = {f.upper() for f in cal_fnames}
            has_req_cals = (flux_cal.upper() in cal_upper) and (phase_cal.upper() in cal_upper)
            has_leaked_targets = bool(cal_upper & target_set_upper)
            cal_fields_ok = cal_nrows > 0 and has_req_cals and not has_leaked_targets

            flux_key_cal = next(
                (k for k in cal_intents_sets if k.upper() == flux_cal.upper()), None
            )
            phase_key_cal = next(
                (k for k in cal_intents_sets if k.upper() == phase_cal.upper()), None
            )
            flux_key_in = next((k for k in in_intents_sets if k.upper() == flux_cal.upper()), None)
            phase_key_in = next(
                (k for k in in_intents_sets if k.upper() == phase_cal.upper()), None
            )

            flux_set = set(cal_intents_sets.get(flux_key_cal or "", set())) | set(
                in_intents_sets.get(flux_key_in or "", set())
            )
            phase_set = set(cal_intents_sets.get(phase_key_cal or "", set())) | set(
                in_intents_sets.get(phase_key_in or "", set())
            )

            cal_intents_ok = (
                _has_intent_prefix(flux_set, "CALIBRATE_BANDPASS")
                and _has_intent_prefix(flux_set, "CALIBRATE_FLUX")
                and _has_intent_prefix(phase_set, "CALIBRATE_PHASE")
            )
        except Exception as exc:
            errors.append(f"Failed inspecting calibrators.ms: {exc}")
    elif not cal_exists:
        errors.append("Missing /workspace/calibrators.ms")

    cal_struct_sub = 1.0 if (cal_exists and cal_fields_ok and not table_locked) else 0.0
    intents_sub = 0.5 * (1.0 if cal_intents_ok else 0.0) + 0.5 * (1.0 if target_intents_ok else 0.0)
    calibrators_ms_score = (
        (0.5 * cal_struct_sub + 0.5 * intents_sub) if cal_struct_sub > 0.0 else 0.0
    )

    # 2. J1331+3030 MODEL_DATA parallel-hand amplitude within +-1.0% of PB2017 (0.25 weight)
    model_amp_jy: float | None = None
    model_source_ms: str | None = None
    model_rel_err: float | None = None
    expected_amp = truth["expected_j1331_model_amp_jy"]

    if cal_exists and not has_active_table_lock(cal_ms):
        try:
            cand_amp, _msg = _measure_j1331_model_amp(cal_ms, flux_cal)
            if cand_amp is not None:
                model_amp_jy = cand_amp
                model_source_ms = "calibrators.ms"
        except Exception as exc:
            errors.append(f"Error reading MODEL_DATA on calibrators.ms: {exc}")

    # Fall back to input.ms if MODEL_DATA was absent on calibrators.ms (or left at unity 1.0 Jy
    # while setjy was applied to input.ms).
    if input_exists and not has_active_table_lock(input_ms):
        need_fallback = model_amp_jy is None or abs(model_amp_jy - 1.0) < 1e-3
        if need_fallback:
            try:
                in_amp, _in_msg = _measure_j1331_model_amp(input_ms, flux_cal)
                if in_amp is not None and (model_amp_jy is None or abs(in_amp - 1.0) >= 1e-3):
                    model_amp_jy = in_amp
                    model_source_ms = "input.ms"
            except Exception as exc:
                errors.append(f"Error reading MODEL_DATA on input.ms: {exc}")

    if model_amp_jy is None:
        model_data_score = 0.0
        errors.append(f"MODEL_DATA not populated for {flux_cal}")
    else:
        model_rel_err = abs(model_amp_jy - expected_amp) / max(expected_amp, 1e-12)
        tol = truth["model_rel_tol"]
        if model_rel_err <= tol:
            model_data_score = 1.0
        else:
            # Smooth linear decay from 1.0 at 1.0% error to 0.0 at 5.0% error
            model_data_score = max(0.0, 1.0 - (model_rel_err - tol) / max(0.05 - tol, 1e-6))

    # 3 & 4. Must-flag recall >= 0.95 (0.30) and clean false-positive rate < 0.05 (0.25)
    must_flag_recall: float | None = None
    clean_fpr: float | None = None
    matched_cal_rows = 0
    flag_recall_score = 0.0
    clean_fpr_score = 0.0

    if not input_exists:
        errors.append("Missing /workspace/input.ms")
    elif masks_npz_path is None or not Path(masks_npz_path).is_file():
        errors.append(f"Missing precal masks NPZ: {masks_npz_path}")
    elif not has_active_table_lock(input_ms):
        try:
            must_mask_2d, clean_mask_2d = _load_precal_masks(Path(masks_npz_path))
            with _open_ro_table(input_ms / "FIELD") as tb_if:
                in_fnames = [str(x).strip() for x in tb_if.getcol("NAME")]
            with _open_ro_table(input_ms / "ANTENNA") as tb_ia:
                in_ant_names = [str(x).strip() for x in tb_ia.getcol("NAME")]

            with _open_ro_table(input_ms) as tb_in:
                in_flag = tb_in.getcol("FLAG")
                in_flag_row = tb_in.getcol("FLAG_ROW")
                in_fids = tb_in.getcol("FIELD_ID")
                in_scan = tb_in.getcol("SCAN_NUMBER")
                in_time = tb_in.getcol("TIME")
                in_a1 = tb_in.getcol("ANTENNA1")
                in_a2 = tb_in.getcol("ANTENNA2")
                in_ddid = tb_in.getcol("DATA_DESC_ID")

            eff_2d = np.any(in_flag, axis=0) | in_flag_row[np.newaxis, :]
            if eff_2d.shape != must_mask_2d.shape:
                errors.append(
                    f"input.ms (nchan, nrow) shape {eff_2d.shape} != mask shape {must_mask_2d.shape}"
                )
            else:
                if cal_exists and not has_active_table_lock(cal_ms):
                    with contextlib.suppress(Exception):
                        matched_cal_rows = _merge_calibrator_flags_into_input(
                            eff_2d,
                            in_fnames,
                            in_fids,
                            in_scan,
                            in_time,
                            in_ant_names,
                            in_a1,
                            in_a2,
                            in_ddid,
                            cal_ms,
                        )

                must_flag_recall = (
                    float(np.mean(eff_2d[must_mask_2d])) if np.any(must_mask_2d) else 1.0
                )
                clean_fpr = float(np.mean(eff_2d[clean_mask_2d])) if np.any(clean_mask_2d) else 0.0

                rec_min = truth["flag_recall_min"]
                if must_flag_recall >= rec_min:
                    raw_rec_score = 1.0
                else:
                    raw_rec_score = max(0.0, (must_flag_recall - 0.50) / max(rec_min - 0.50, 1e-6))

                # Guard against blanket over-flagging (e.g., flagging 100% of rows to game recall)
                if clean_fpr > 0.25:
                    overflag_cap = max(0.0, 1.0 - (clean_fpr - 0.25) / 0.50)
                    raw_rec_score *= overflag_cap
                flag_recall_score = raw_rec_score

                fpr_max = truth["clean_fpr_max"]
                if clean_fpr < fpr_max:
                    clean_fpr_score = 1.0
                else:
                    clean_fpr_score = max(0.0, 1.0 - (clean_fpr - fpr_max) / 0.15)
        except Exception as exc:
            errors.append(f"Failed evaluating precal flag masks: {exc}")

    composite = (
        weights["calibrators_ms_valid"] * calibrators_ms_score
        + weights["model_data_accuracy"] * model_data_score
        + weights["rfi_preflag_recall"] * flag_recall_score
        + weights["clean_false_positive_guard"] * clean_fpr_score
    )
    if table_locked:
        composite = 0.0

    reward = round(min(1.0, max(0.0, composite)), 6)
    passed = 1 if reward >= pass_min else 0

    reward_dict: dict[str, float | int] = {
        "reward": reward,
        "passed": passed,
        "calibrators_ms_score": round(calibrators_ms_score, 6),
        "model_data_score": round(model_data_score, 6),
        "flag_recall_score": round(flag_recall_score, 6),
        "clean_fpr_score": round(clean_fpr_score, 6),
    }
    metrics_dict: dict[str, Any] = {
        "task_id": task_id,
        "stage": "precal",
        "pass_reward_min": pass_min,
        "table_locked": table_locked,
        "truth": {
            "flux_calibrator": flux_cal,
            "phase_calibrator": phase_cal,
            "target_fields": truth["target_fields"],
            "expected_j1331_model_amp_jy": expected_amp,
            "model_rel_tol": truth["model_rel_tol"],
            "flag_recall_min": truth["flag_recall_min"],
            "clean_fpr_max": truth["clean_fpr_max"],
        },
        "measured": {
            "calibrators_ms_exists": cal_exists,
            "calibrators_ms_nrows": cal_nrows,
            "calibrators_ms_fields": cal_fnames,
            "calibrators_ms_fields_ok": cal_fields_ok,
            "calibrator_intents_ok": cal_intents_ok,
            "target_intents_ok": target_intents_ok,
            "calibrators_ms_intents": cal_intents,
            "input_ms_intents": in_intents,
            "j1331_model_amp_jy": round(model_amp_jy, 6) if model_amp_jy is not None else None,
            "j1331_model_source_ms": model_source_ms,
            "j1331_model_rel_err": round(model_rel_err, 6) if model_rel_err is not None else None,
            "must_flag_recall": round(must_flag_recall, 6)
            if must_flag_recall is not None
            else None,
            "clean_false_positive_rate": round(clean_fpr, 6) if clean_fpr is not None else None,
            "merged_calibrator_rows": matched_cal_rows,
        },
        "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
        "weights": weights,
        "errors": errors,
    }
    return reward_dict, metrics_dict
