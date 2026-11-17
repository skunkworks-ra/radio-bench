"""Pure metric extractor for `precal` stage (zero dataset branches, zero ground-truth reads)."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np

from radio_bench.grading.extractors.common import (
    check_table_multiused,
    has_active_table_lock,
    iter_ddid_chunks,
    open_ro_table,
    read_field_intents,
)
from radio_bench.grading.schema import ExtractConfig, ObservableRecord, finite_observable


def _has_intent_prefix(intents: set[str], prefix: str) -> bool:
    return any(tok.upper().startswith(prefix.upper()) for tok in intents)


def load_precal_masks(masks_npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load `(must_flag_2d, clean_2d, ref_spectrum_jy)` from a reference flags `.npz` bundle."""
    with np.load(masks_npz_path, allow_pickle=False) as z:
        ref_spec: np.ndarray | None = None
        if "ref_flux_cal_spectrum_jy" in z.files:
            ref_spec = np.asarray(z["ref_flux_cal_spectrum_jy"], dtype=np.float64).ravel().copy()

        if "corrupted_chan_row" in z.files:
            corrupted = z["corrupted_chan_row"].astype(bool).copy()
            return corrupted, ~corrupted, ref_spec
        if "must_flag_2d" in z.files and "clean_2d" in z.files:
            raw_must = z["must_flag_2d"]
            raw_clean = z["clean_2d"]
            if "shape_2d" in z.files and raw_must.dtype == np.uint8:
                shape_2d = z["shape_2d"]
                nrow = int(shape_2d[1])
                must = np.unpackbits(raw_must, axis=1)[:, :nrow].astype(bool)
                clean = np.unpackbits(raw_clean, axis=1)[:, :nrow].astype(bool)
                return must, clean, ref_spec
            return raw_must.astype(bool).copy(), raw_clean.astype(bool).copy(), ref_spec
    raise ValueError(f"Unrecognized precal mask keys in {masks_npz_path}")


def _extract_virtual_source_model(
    ms_path: Path,
    flux_cal_name: str,
) -> tuple[float | None, list[float] | None, str | None]:
    """Read virtual `setjy(..., usescratch=False)` model from `SOURCE::SOURCE_MODEL`."""
    source_dir = ms_path / "SOURCE"
    if not (source_dir / "table.dat").is_file():
        return None, None, None

    target_upper = flux_cal_name.strip().upper()
    with open_ro_table(source_dir) as tb_src:
        cols = set(tb_src.colnames())
        if "NAME" not in cols or "SOURCE_MODEL" not in cols:
            return None, None, None
        nrows = int(tb_src.nrows())
        for r in range(nrows):
            name_val = str(tb_src.getcell("NAME", r)).strip()
            if name_val.upper() != target_upper:
                continue
            if not tb_src.iscelldefined("SOURCE_MODEL", r):
                continue
            rec = tb_src.getcell("SOURCE_MODEL", r)
            if not isinstance(rec, dict) or not rec:
                continue
            for cl_val in rec.values():
                if not isinstance(cl_val, dict):
                    continue
                cont = cl_val.get("container")
                if not isinstance(cont, dict):
                    continue
                for comp in cont.values():
                    if not isinstance(comp, dict):
                        continue
                    spec = comp.get("spectrum")
                    if isinstance(spec, dict) and "ival" in spec:
                        ival = np.asarray(spec["ival"], dtype=np.float64).ravel()
                        if ival.size > 0 and np.all(np.isfinite(ival)):
                            mean_v = float(np.mean(ival))
                            if mean_v > 0.0:
                                return (
                                    mean_v,
                                    [round(float(x), 6) for x in ival.tolist()],
                                    f"{ms_path.name}::SOURCE::SOURCE_MODEL",
                                )
                    flux_blk = comp.get("flux")
                    if isinstance(flux_blk, dict) and "value" in flux_blk:
                        f_arr = np.asarray(flux_blk["value"], dtype=np.float64).ravel()
                        if (
                            f_arr.size > 0
                            and math.isfinite(float(f_arr[0]))
                            and float(f_arr[0]) > 0.0
                        ):
                            return (
                                float(f_arr[0]),
                                None,
                                f"{ms_path.name}::SOURCE::SOURCE_MODEL",
                            )
    return None, None, None


def _measure_flux_cal_model_single_ms(
    ms_path: Path,
    flux_cal_name: str,
    *,
    allow_virtual_model: bool = False,
) -> tuple[float | None, list[float] | None, str | None]:
    """Measure unflagged parallel-hand `MODEL_DATA` amplitude & channel spectrum safely per DDID."""
    with open_ro_table(ms_path / "FIELD") as tb_f:
        fnames = [str(x).strip() for x in tb_f.getcol("NAME")]
    matching_fids = [i for i, n in enumerate(fnames) if n.upper() == flux_cal_name.strip().upper()]
    if not matching_fids:
        return None, None, None
    fid = matching_fids[0]

    has_model_col = False
    unflg_sum = 0.0
    unflg_cnt = 0
    all_sum = 0.0
    all_cnt = 0
    chan_sum: np.ndarray | None = None
    chan_cnt: np.ndarray | None = None
    chan_all_sum: np.ndarray | None = None
    chan_all_cnt: int = 0
    primary_ddid: int | None = None

    with open_ro_table(ms_path) as tb:
        has_model_col = "MODEL_DATA" in set(tb.colnames())
        if has_model_col:
            for chunk in iter_ddid_chunks(
                tb,
                ("MODEL_DATA", "FLAG", "FLAG_ROW"),
                query_expr=f"FIELD_ID == {fid}",
            ):
                model = chunk.columns["MODEL_DATA"]
                flag = chunk.columns["FLAG"].astype(bool)
                flag_row = chunk.columns["FLAG_ROW"].astype(bool)
                eff_flag = flag | flag_row[np.newaxis, np.newaxis, :]
                npol, nchan, nrow = model.shape
                p_indices = (0, 3) if npol >= 4 else ((0, npol - 1) if npol >= 2 else (0,))

                if primary_ddid is None:
                    primary_ddid = chunk.ddid
                    chan_sum = np.zeros(nchan, dtype=np.float64)
                    chan_cnt = np.zeros(nchan, dtype=np.int64)
                    chan_all_sum = np.zeros(nchan, dtype=np.float64)

                par_all = np.mean([np.abs(model[p]) for p in p_indices], axis=0)
                finite_all = np.isfinite(par_all)
                if np.any(finite_all):
                    all_sum += float(np.sum(par_all[finite_all]))
                    all_cnt += int(np.sum(finite_all))
                if chunk.ddid == primary_ddid and chan_all_sum is not None:
                    chan_all_sum += np.sum(np.where(finite_all, par_all, 0.0), axis=1)
                    chan_all_cnt += nrow

                for p_idx in p_indices:
                    abs_m = np.abs(model[p_idx])
                    good = (~eff_flag[p_idx]) & np.isfinite(abs_m)
                    if np.any(good):
                        unflg_sum += float(np.sum(abs_m[good]))
                        unflg_cnt += int(np.sum(good))
                        if (
                            chunk.ddid == primary_ddid
                            and chan_sum is not None
                            and chan_cnt is not None
                        ):
                            chan_sum += np.sum(np.where(good, abs_m, 0.0), axis=1)
                            chan_cnt += np.sum(good, axis=1)

    if has_model_col and (unflg_cnt > 0 or all_cnt > 0):
        amp = (unflg_sum / unflg_cnt) if unflg_cnt > 0 else (all_sum / all_cnt)
        if math.isfinite(amp) and amp > 0.0:
            spec_list: list[float] | None = None
            if (
                chan_sum is not None
                and chan_cnt is not None
                and chan_all_sum is not None
                and chan_all_cnt > 0
            ):
                fallback_chan = chan_all_sum / float(chan_all_cnt)
                per_ch = np.where(chan_cnt > 0, chan_sum / np.maximum(chan_cnt, 1), fallback_chan)
                if np.all(np.isfinite(per_ch)):
                    spec_list = [round(float(x), 6) for x in per_ch.tolist()]
            return amp, spec_list, f"{ms_path.name}::MODEL_DATA"

    if allow_virtual_model:
        return _extract_virtual_source_model(ms_path, flux_cal_name)

    return None, None, None


def _read_effective_2d_flags_and_meta(
    ms_path: Path,
    expected_nchan: int | None = None,
) -> tuple[
    np.ndarray | None,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Read `(nchan, nrow)` effective flags and scalar row keys safely via per-DDID chunks."""
    with open_ro_table(ms_path / "FIELD") as tb_f:
        fnames = [str(x).strip() for x in tb_f.getcol("NAME")]
    with open_ro_table(ms_path / "ANTENNA") as tb_a:
        ant_names = [str(x).strip() for x in tb_a.getcol("NAME")]

    with open_ro_table(ms_path) as tb:
        nrows = int(tb.nrows())
        if nrows == 0:
            empty_i = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=np.float64)
            return None, fnames, ant_names, empty_i, empty_i, empty_f, empty_i, empty_i, empty_i

        fids = np.asarray(tb.getcol("FIELD_ID"), dtype=np.int64)
        scan = np.asarray(tb.getcol("SCAN_NUMBER"), dtype=np.int64)
        time_col = np.asarray(tb.getcol("TIME"), dtype=np.float64)
        a1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int64)
        a2 = np.asarray(tb.getcol("ANTENNA2"), dtype=np.int64)
        ddid = np.asarray(tb.getcol("DATA_DESC_ID"), dtype=np.int64)

        eff_2d: np.ndarray | None = None
        shape_mismatch = False
        for chunk in iter_ddid_chunks(tb, ("FLAG", "FLAG_ROW"), include_rownumbers=True):
            c_flg = chunk.columns["FLAG"].astype(bool)
            c_flgr = chunk.columns["FLAG_ROW"].astype(bool)
            c_nchan = int(c_flg.shape[1])
            if expected_nchan is not None and c_nchan != expected_nchan:
                shape_mismatch = True
                break
            if eff_2d is None:
                eff_2d = np.zeros((c_nchan, nrows), dtype=bool)
            elif eff_2d.shape[0] != c_nchan:
                shape_mismatch = True
                break
            eff_2d[:, chunk.rownumbers] = np.any(c_flg, axis=0) | c_flgr[np.newaxis, :]

        if shape_mismatch:
            eff_2d = None

    return eff_2d, fnames, ant_names, fids, scan, time_col, a1, a2, ddid


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
    (
        cal_eff_2d,
        cal_fnames,
        cal_ant_names,
        cal_fids,
        cal_scan,
        cal_time,
        cal_a1,
        cal_a2,
        cal_ddid,
    ) = _read_effective_2d_flags_and_meta(cal_ms_path, expected_nchan=in_chan_row_flag.shape[0])

    if cal_eff_2d is None or cal_fids.size == 0:
        return 0

    name_to_in_fid = {n.upper(): idx for idx, n in enumerate(in_fnames)}
    name_to_in_aid = {n.upper(): idx for idx, n in enumerate(in_ant_names)}

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


def _discover_ref_flags_npz(ref_arrays_dir: Path | None) -> Path | None:
    if ref_arrays_dir is None:
        return None
    if ref_arrays_dir.is_file() and ref_arrays_dir.suffix == ".npz":
        return ref_arrays_dir
    if not ref_arrays_dir.is_dir():
        return None
    for name in ("ref_flags.npz", "corruption_masks.npz", "precal_masks.npz"):
        cand = ref_arrays_dir / name
        if cand.is_file():
            return cand
    return None


def extract_precal(
    workspace_dir: Path,
    extract_config: ExtractConfig,
    ref_arrays_dir: Path | None = None,
    *,
    table_locked: bool = False,
) -> tuple[dict[str, ObservableRecord], list[str]]:
    """Extract raw physical observables from `/workspace/{input.ms,calibrators.ms}`."""
    from casatasks import clearstat

    with contextlib.suppress(Exception):
        clearstat()

    ws = Path(workspace_dir)
    input_ms = ws / "input.ms"
    cal_ms = ws / "calibrators.ms"
    roles = extract_config.roles
    errors: list[str] = []
    obs: dict[str, ObservableRecord] = {}

    flux_cal = roles.flux_cal[0].strip() if roles.flux_cal else ""
    bp_cal = roles.bandpass_cal[0].strip() if roles.bandpass_cal else flux_cal
    phase_cal = roles.phase_cal[0].strip() if roles.phase_cal else ""
    target_fields = [t.strip() for t in roles.targets if t.strip()]
    target_set_upper = {t.upper() for t in target_fields}

    input_exists = (input_ms / "table.dat").is_file()
    cal_exists = (cal_ms / "table.dat").is_file()

    in_locked = input_exists and check_table_multiused(input_ms)
    cal_locked = cal_exists and check_table_multiused(cal_ms)
    any_locked = table_locked or in_locked or cal_locked

    if any_locked:
        for key, unit in (
            ("precal.calibrators_ms_struct_ok", None),
            ("precal.calibrators_ms_nrows", None),
            ("precal.calibrator_intents_ok", None),
            ("precal.target_intents_ok", None),
            ("precal.flux_cal_model_amp_jy", "Jy"),
            ("precal.flux_cal_model_spectrum_jy", "Jy"),
            ("precal.flux_cal_model_spec_nrmse", None),
            ("precal.must_flag_recall", None),
            ("precal.clean_false_positive_rate", None),
            ("precal.merged_calibrator_rows", None),
        ):
            obs[key] = ObservableRecord(
                value=None,
                status="locked_or_unreadable",
                unit=unit,
                provenance="workspace",
            )
        return obs, errors

    # 1. Structure & scan-intent observables
    cal_nrows = 0
    cal_struct_ok = False
    cal_intents_ok = False
    target_intents_ok = False
    in_intents_sets: dict[str, set[str]] = {}
    cal_intents_sets: dict[str, set[str]] = {}

    if input_exists:
        try:
            _in_fn, in_intents_sets = read_field_intents(input_ms)
            if target_fields:
                t_checks: list[bool] = []
                for tf in target_fields:
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

    if cal_exists:
        try:
            with open_ro_table(cal_ms) as tb_c:
                cal_nrows = int(tb_c.nrows())
            cal_fnames, cal_intents_sets = read_field_intents(cal_ms)
            cal_upper = {f.upper() for f in cal_fnames}
            req_cals_ok = True
            if flux_cal and flux_cal.upper() not in cal_upper:
                req_cals_ok = False
            if bp_cal and bp_cal.upper() not in cal_upper:
                req_cals_ok = False
            if phase_cal and phase_cal.upper() not in cal_upper:
                req_cals_ok = False
            has_leaked_targets = bool(cal_upper & target_set_upper)
            cal_struct_ok = cal_nrows > 0 and req_cals_ok and not has_leaked_targets
        except Exception as exc:
            errors.append(f"Failed inspecting calibrators.ms: {exc}")
    else:
        errors.append("Missing /workspace/calibrators.ms")

    # Intents across calibrators.ms unioned with input.ms
    def _merged_intents(fname: str) -> set[str]:
        if not fname:
            return set()
        k_c = next((k for k in cal_intents_sets if k.upper() == fname.upper()), None)
        k_i = next((k for k in in_intents_sets if k.upper() == fname.upper()), None)
        return set(cal_intents_sets.get(k_c or "", set())) | set(
            in_intents_sets.get(k_i or "", set())
        )

    flux_set = _merged_intents(flux_cal)
    bp_set = _merged_intents(bp_cal)
    phase_set = _merged_intents(phase_cal)
    cal_intents_ok = (
        (not bp_cal or _has_intent_prefix(bp_set, "CALIBRATE_BANDPASS"))
        and (not flux_cal or _has_intent_prefix(flux_set, "CALIBRATE_FLUX"))
        and (not phase_cal or _has_intent_prefix(phase_set, "CALIBRATE_PHASE"))
    )

    obs["precal.calibrators_ms_struct_ok"] = ObservableRecord(
        value=bool(cal_struct_ok),
        status="measured",
        provenance="calibrators.ms" if cal_exists else "workspace",
    )
    obs["precal.calibrators_ms_nrows"] = ObservableRecord(
        value=int(cal_nrows),
        status="measured" if cal_exists else "missing_artifact",
        provenance="calibrators.ms",
    )
    obs["precal.calibrator_intents_ok"] = ObservableRecord(
        value=bool(cal_intents_ok),
        status="measured" if (cal_exists or input_exists) else "missing_artifact",
        provenance="calibrators.ms+input.ms",
    )
    obs["precal.target_intents_ok"] = ObservableRecord(
        value=bool(target_intents_ok),
        status="measured" if input_exists else "missing_artifact",
        provenance="input.ms::STATE",
    )

    # 2. Flux-standard MODEL_DATA amplitude (with optional virtual SOURCE_MODEL fallback)
    model_amp_jy: float | None = None
    model_spec_jy: list[float] | None = None
    model_prov: str | None = None

    if flux_cal and cal_exists and not has_active_table_lock(cal_ms):
        try:
            c_amp, c_spec, c_prov = _measure_flux_cal_model_single_ms(
                cal_ms,
                flux_cal,
                allow_virtual_model=extract_config.allow_virtual_model,
            )
            if c_amp is not None:
                model_amp_jy, model_spec_jy, model_prov = c_amp, c_spec, c_prov
        except Exception as exc:
            errors.append(f"Error reading flux calibrator model on calibrators.ms: {exc}")

    if flux_cal and input_exists and not has_active_table_lock(input_ms):
        need_fallback = model_amp_jy is None or abs(model_amp_jy - 1.0) < 1e-3
        if need_fallback:
            try:
                in_amp, in_spec, in_prov = _measure_flux_cal_model_single_ms(
                    input_ms,
                    flux_cal,
                    allow_virtual_model=extract_config.allow_virtual_model,
                )
                if in_amp is not None and (model_amp_jy is None or abs(in_amp - 1.0) >= 1e-3):
                    model_amp_jy, model_spec_jy, model_prov = in_amp, in_spec, in_prov
            except Exception as exc:
                errors.append(f"Error reading flux calibrator model on input.ms: {exc}")

    if model_amp_jy is None and flux_cal:
        errors.append(f"MODEL_DATA not populated for {flux_cal}")

    if not flux_cal:
        obs["precal.flux_cal_model_amp_jy"] = ObservableRecord(
            value=None, status="not_applicable", unit="Jy", provenance=model_prov
        )
        obs["precal.flux_cal_model_spectrum_jy"] = ObservableRecord(
            value=None, status="not_applicable", unit="Jy", provenance=model_prov
        )
    else:
        obs["precal.flux_cal_model_amp_jy"] = finite_observable(
            model_amp_jy,
            unit="Jy",
            provenance=model_prov,
            missing_status="missing_artifact",
            ndigits=6,
        )
        spec_valid = model_spec_jy is not None and all(
            math.isfinite(float(x)) for x in model_spec_jy
        )
        obs["precal.flux_cal_model_spectrum_jy"] = ObservableRecord(
            value=[float(x) for x in model_spec_jy]
            if (model_spec_jy is not None and spec_valid)
            else None,
            status=(
                "measured"
                if (model_spec_jy is not None and spec_valid)
                else ("missing_artifact" if model_spec_jy is None else "extractor_error")
            ),
            unit="Jy",
            provenance=model_prov,
        )

    # 3. Flag masks (`must_flag_recall` & `clean_false_positive_rate` & optional `spec_nrmse`)
    npz_path = _discover_ref_flags_npz(ref_arrays_dir)
    must_flag_recall: float | None = None
    clean_fpr: float | None = None
    spec_nrmse: float | None = None
    matched_cal_rows = 0

    if not input_exists:
        errors.append("Missing /workspace/input.ms")
    elif npz_path is None or not npz_path.is_file():
        if "precal" in extract_config.stages and len(extract_config.stages) == 1:
            errors.append(f"Missing reference precal flags NPZ in {ref_arrays_dir}")
    else:
        try:
            must_mask_2d, clean_mask_2d, ref_spec = load_precal_masks(npz_path)
            if (
                ref_spec is not None
                and model_spec_jy is not None
                and len(model_spec_jy) == ref_spec.size
            ):
                meas_arr = np.asarray(model_spec_jy, dtype=np.float64)
                denom = float(np.mean(np.abs(ref_spec)))
                if denom > 1e-12:
                    spec_nrmse = float(np.sqrt(np.mean((meas_arr - ref_spec) ** 2)) / denom)

            (
                eff_2d,
                in_fnames,
                in_ant_names,
                in_fids,
                in_scan,
                in_time,
                in_a1,
                in_a2,
                in_ddid,
            ) = _read_effective_2d_flags_and_meta(input_ms, expected_nchan=must_mask_2d.shape[0])

            if eff_2d is None or eff_2d.shape != must_mask_2d.shape:
                errors.append(
                    f"input.ms (nchan, nrow) shape {None if eff_2d is None else eff_2d.shape} "
                    f"!= mask shape {must_mask_2d.shape}"
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
        except Exception as exc:
            errors.append(f"Failed evaluating precal flag masks: {exc}")

    obs["precal.flux_cal_model_spec_nrmse"] = finite_observable(
        spec_nrmse,
        unit=None,
        provenance=model_prov,
        missing_status="not_applicable",
        ndigits=6,
    )
    obs["precal.must_flag_recall"] = finite_observable(
        must_flag_recall,
        unit=None,
        provenance="input.ms+calibrators.ms::FLAG" if cal_exists else "input.ms::FLAG",
        missing_status="not_applicable" if npz_path is None else "extractor_error",
        ndigits=6,
    )
    obs["precal.clean_false_positive_rate"] = finite_observable(
        clean_fpr,
        unit=None,
        provenance="input.ms+calibrators.ms::FLAG" if cal_exists else "input.ms::FLAG",
        missing_status="not_applicable" if npz_path is None else "extractor_error",
        ndigits=6,
    )
    obs["precal.merged_calibrator_rows"] = ObservableRecord(
        value=int(matched_cal_rows),
        status="measured" if input_exists else "missing_artifact",
        unit=None,
        provenance="calibrators.ms",
    )
    return obs, errors
