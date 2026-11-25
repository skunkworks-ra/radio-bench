"""Pure metric extractor for `calibrate` stage (zero dataset branches, zero ground-truth reads)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.grading.extractors.common import (
    check_table_multiused,
    discover_fluxscale_from_caltables,
    has_active_table_lock,
    iter_ddid_chunks,
    natural_thermal_noise_jy,
    open_ro_table,
    parse_sky_coord_rad,
    read_ddid_to_spw_map,
    read_spw_chan_freqs,
    resolve_ant_ids_by_names,
    resolve_field_id_by_name,
    resolve_field_ids_by_names,
    resolve_scratch_root,
    thermal_floor_ratio,
    verify_pb_taper,
)
from radio_bench.grading.schema import ExtractConfig, ObservableRecord, finite_observable


def _empty_flatness_dict(
    *,
    field_calibrated: bool = False,
    unflagged_fraction: float | None = None,
    source_ms: str | None = None,
) -> dict[str, Any]:
    return {
        "coh_slope_deg_per_100mhz": None,
        "coh_chan_phase_rms_deg": None,
        "med_bl_slope_deg_per_100mhz": None,
        "med_bl_chan_phase_rms_deg": None,
        "median_row_chan_phase_rms_deg": None,
        "field_calibrated": field_calibrated,
        "unflagged_fraction": round(unflagged_fraction, 6)
        if unflagged_fraction is not None
        else None,
        "source_ms": source_ms,
    }


def _unwrap_centered_deg(spec_arr: np.ndarray) -> np.ndarray:
    unit_spec = spec_arr / np.maximum(np.abs(spec_arr), 1e-12)
    mean_phasor = complex(np.mean(unit_spec))
    if abs(mean_phasor) > 1e-9:
        ref_rot = np.conj(mean_phasor / abs(mean_phasor))
        ang_rad = np.unwrap(np.angle(spec_arr * ref_rot))
    else:
        ang_rad = np.unwrap(np.angle(spec_arr))
    deg = np.rad2deg(ang_rad)
    return deg - float(np.mean(deg))


def _measure_bp_flatness_single_ddid(
    corr: np.ndarray,
    raw: np.ndarray | None,
    flag: np.ndarray,
    flag_row: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    freqs_hz: np.ndarray,
) -> dict[str, Any]:
    """Compute unit-phasor & outlier-clipped bandpass phase flatness on one homogeneous DDID."""
    eff_flag = flag | flag_row[np.newaxis, np.newaxis, :]
    npol, nchan, _nrow = corr.shape
    pol_indices = (0, 3) if npol >= 4 else ((0, npol - 1) if npol >= 2 else (0,))

    par_eff_flag = eff_flag[list(pol_indices), :, :]
    par_total_cells = int(par_eff_flag.size)
    par_unflg_cells = int(np.sum(~par_eff_flag))
    unflagged_frac = float(par_unflg_cells / max(par_total_cells, 1))

    par_corr = corr[list(pol_indices), :, :]
    good_par = (~par_eff_flag) & np.isfinite(par_corr)
    if not np.any(good_par):
        return {
            "field_calibrated": False,
            "par_unflg_cells": par_unflg_cells,
            "par_total_cells": par_total_cells,
            "unflagged_fraction": unflagged_frac,
            "coh_slope": None,
            "coh_rms": None,
            "med_bl_slope": None,
            "med_bl_rms": None,
            "row_rms": None,
        }

    max_abs_c = float(np.max(np.abs(par_corr[good_par])))
    if raw is not None:
        par_raw = raw[list(pol_indices), :, :]
        max_diff = float(np.max(np.abs(par_corr[good_par] - par_raw[good_par])))
        field_calibrated = max_abs_c > 1e-9 and max_diff > 1e-6
    else:
        field_calibrated = max_abs_c > 1e-9

    if not field_calibrated or nchan < 8 or freqs_hz.size != nchan:
        return {
            "field_calibrated": field_calibrated,
            "par_unflg_cells": par_unflg_cells,
            "par_total_cells": par_total_cells,
            "unflagged_fraction": unflagged_frac,
            "coh_slope": None,
            "coh_rms": None,
            "med_bl_slope": None,
            "med_bl_rms": None,
            "row_rms": None,
        }

    clean_flag = eff_flag.copy()
    min_row_chans = min(8, max(4, nchan // 4))
    for p_idx in pol_indices:
        good_p = (~eff_flag[p_idx]) & np.isfinite(corr[p_idx]) & (np.abs(corr[p_idx]) > 1e-12)
        clean_flag[p_idx] = ~good_p
        valid_r = np.flatnonzero(np.sum(good_p, axis=0) >= min_row_chans)
        if valid_r.size >= 4:
            row_amps = np.array(
                [float(np.median(np.abs(corr[p_idx, good_p[:, r], r]))) for r in valid_r],
                dtype=float,
            )
            med_row_amp = float(np.median(row_amps))
            if med_row_amp > 1e-12:
                outlier_mask = row_amps > (5.0 * med_row_amp)
                if np.any(outlier_mask) and int(np.sum(~outlier_mask)) >= max(2, valid_r.size // 2):
                    clean_flag[p_idx, :, valid_r[outlier_mask]] = True

        rem_good = ~clean_flag[p_idx]
        n_rem = int(np.sum(rem_good))
        if n_rem >= 16:
            med_vis_amp = float(np.median(np.abs(corr[p_idx][rem_good])))
            if med_vis_amp > 1e-12:
                vis_outlier = rem_good & (np.abs(corr[p_idx]) > (5.0 * med_vis_amp))
                n_vis_out = int(np.sum(vis_outlier))
                if 0 < n_vis_out <= (n_rem // 4):
                    clean_flag[p_idx][vis_outlier] = True

    unit_corr = corr / np.maximum(np.abs(corr), 1e-12)

    chan_counts = np.sum(~clean_flag[list(pol_indices), :, :], axis=(0, 2))
    active_chan_counts = chan_counts[chan_counts > 0]
    med_chan_count = float(np.median(active_chan_counts)) if active_chan_counts.size > 0 else 0.0
    min_chan_support = max(2, int(0.10 * med_chan_count)) if med_chan_count >= 20.0 else 1

    ch_indices: list[int] = []
    vec_spec: list[complex] = []
    for ch in range(nchan):
        if int(chan_counts[ch]) < min_chan_support:
            continue
        vals: list[complex] = []
        for p_idx in pol_indices:
            good = ~clean_flag[p_idx, ch]
            if np.any(good):
                u_ch = unit_corr[p_idx, ch, good]
                finite = np.isfinite(u_ch.real) & np.isfinite(u_ch.imag)
                if np.any(finite):
                    vals.append(complex(np.mean(u_ch[finite])))
        if vals:
            ch_indices.append(ch)
            vec_spec.append(complex(np.mean(vals)))

    if len(ch_indices) < 8:
        return {
            "field_calibrated": field_calibrated,
            "par_unflg_cells": par_unflg_cells,
            "par_total_cells": par_total_cells,
            "unflagged_fraction": unflagged_frac,
            "coh_slope": None,
            "coh_rms": None,
            "med_bl_slope": None,
            "med_bl_rms": None,
            "row_rms": None,
        }

    f_valid_hz = freqs_hz[ch_indices]
    phs_coh_deg = _unwrap_centered_deg(np.array(vec_spec, dtype=np.complex128))
    slope_hz, _ = np.polyfit(f_valid_hz - float(np.mean(f_valid_hz)), phs_coh_deg, 1)
    coh_slope = abs(float(slope_hz) * 1.0e8)
    coh_rms = float(np.sqrt(np.mean(phs_coh_deg**2)))

    min_good_ch = min(16, max(8, nchan // 2))
    bl_slopes: list[float] = []
    bl_rms: list[float] = []
    pairs = sorted(set(zip(a1.tolist(), a2.tolist(), strict=False)))
    for p1, p2 in pairs:
        bm = (a1 == p1) & (a2 == p2)
        for pol in pol_indices:
            up = unit_corr[pol, :, :][:, bm]
            fp = clean_flag[pol, :, :][:, bm]
            ch_unflg_bl = np.sum(~fp, axis=1)
            active_bl = ch_unflg_bl[ch_unflg_bl > 0]
            med_bl_cnt = float(np.median(active_bl)) if active_bl.size > 0 else 0.0
            min_bl_sup = max(2, int(0.10 * med_bl_cnt)) if med_bl_cnt >= 10.0 else 1
            good_ch = ch_unflg_bl >= min_bl_sup
            if int(np.sum(good_ch)) < min_good_ch:
                continue
            ch_sel = np.flatnonzero(good_ch)
            spec_bl = np.array([np.mean(up[c, ~fp[c]]) for c in ch_sel], dtype=np.complex128)
            if not np.all(np.isfinite(spec_bl)) or np.any(np.abs(spec_bl) <= 1e-12):
                continue
            ph_bl = _unwrap_centered_deg(spec_bl)
            fc = freqs_hz[ch_sel]
            sl_bl, _ = np.polyfit(fc - float(np.mean(fc)), ph_bl, 1)
            bl_slopes.append(abs(float(sl_bl) * 1.0e8))
            bl_rms.append(float(np.sqrt(np.mean(ph_bl**2))))

    row_rms_list: list[float] = []
    for pol in pol_indices:
        good_mat = ~clean_flag[pol]
        valid_rows = np.flatnonzero(np.sum(good_mat, axis=0) >= min_good_ch)
        if valid_rows.size > 2000:
            valid_rows = valid_rows[:: max(1, valid_rows.size // 2000)]
        for r_idx in valid_rows:
            u_r = unit_corr[pol, good_mat[:, r_idx], r_idx]
            if not np.all(np.isfinite(u_r)):
                continue
            ang = np.angle(u_r)
            mean_ang = float(np.angle(np.mean(np.exp(1j * ang))))
            wrapped_deg = np.rad2deg(np.angle(np.exp(1j * (ang - mean_ang))))
            row_rms_list.append(float(np.sqrt(np.mean(wrapped_deg**2))))

    return {
        "field_calibrated": True,
        "par_unflg_cells": par_unflg_cells,
        "par_total_cells": par_total_cells,
        "unflagged_fraction": unflagged_frac,
        "coh_slope": coh_slope,
        "coh_rms": coh_rms,
        "med_bl_slope": float(np.median(bl_slopes)) if bl_slopes else coh_slope,
        "med_bl_rms": float(np.median(bl_rms)) if bl_rms else coh_rms,
        "row_rms": float(np.median(row_rms_list)) if row_rms_list else coh_rms,
    }


def _measure_bp_flatness_ms(
    ms_path: Path,
    bp_fid: int,
    dead_ant_ids: list[int],
) -> dict[str, Any]:
    """Measure bandpass calibrator phase flatness safely across all `DATA_DESC_ID`s."""
    ddid_to_spw = read_ddid_to_spw_map(ms_path)
    dead_clauses = "".join(f" && ANTENNA1 != {d} && ANTENNA2 != {d}" for d in dead_ant_ids)
    query_str = f"FIELD_ID == {bp_fid} && ANTENNA1 != ANTENNA2{dead_clauses}"

    with open_ro_table(ms_path) as tb:
        cols = set(tb.colnames())
        if "CORRECTED_DATA" not in cols:
            return _empty_flatness_dict(field_calibrated=False, source_ms=ms_path.name)
        has_raw = "DATA" in cols
        read_cols = (
            ("CORRECTED_DATA", "DATA", "FLAG", "FLAG_ROW", "ANTENNA1", "ANTENNA2")
            if has_raw
            else ("CORRECTED_DATA", "FLAG", "FLAG_ROW", "ANTENNA1", "ANTENNA2")
        )

        by_ddid: dict[int, list[dict[str, np.ndarray]]] = {}
        for chunk in iter_ddid_chunks(tb, read_cols, query_expr=query_str):
            by_ddid.setdefault(chunk.ddid, []).append(chunk.columns)

    if not by_ddid:
        return _empty_flatness_dict(field_calibrated=False, source_ms=ms_path.name)

    total_unflg = 0
    total_cells = 0
    any_calibrated = False
    coh_slopes: list[float] = []
    coh_rmss: list[float] = []
    med_bl_slopes: list[float] = []
    med_bl_rmss: list[float] = []
    row_rmss: list[float] = []

    for ddid, chunk_list in sorted(by_ddid.items()):
        corr_d = np.concatenate([c["CORRECTED_DATA"] for c in chunk_list], axis=2)
        raw_d = np.concatenate([c["DATA"] for c in chunk_list], axis=2) if has_raw else None
        flg_d = np.concatenate([c["FLAG"] for c in chunk_list], axis=2).astype(bool)
        flgr_d = np.concatenate([c["FLAG_ROW"] for c in chunk_list], axis=0).astype(bool)
        a1_d = np.concatenate([c["ANTENNA1"] for c in chunk_list], axis=0)
        a2_d = np.concatenate([c["ANTENNA2"] for c in chunk_list], axis=0)
        spw_id = ddid_to_spw.get(ddid, ddid)
        freqs_hz = read_spw_chan_freqs(ms_path, spw_id)

        d_res = _measure_bp_flatness_single_ddid(corr_d, raw_d, flg_d, flgr_d, a1_d, a2_d, freqs_hz)
        total_unflg += int(d_res["par_unflg_cells"])
        total_cells += int(d_res["par_total_cells"])
        if d_res["field_calibrated"]:
            any_calibrated = True
            if d_res["coh_slope"] is not None:
                coh_slopes.append(float(d_res["coh_slope"]))
                coh_rmss.append(float(d_res["coh_rms"]))
                med_bl_slopes.append(float(d_res["med_bl_slope"]))
                med_bl_rmss.append(float(d_res["med_bl_rms"]))
                row_rmss.append(float(d_res["row_rms"]))

    overall_unflg = total_unflg / max(total_cells, 1)
    if not any_calibrated or not coh_slopes:
        return _empty_flatness_dict(
            field_calibrated=any_calibrated,
            unflagged_fraction=overall_unflg,
            source_ms=ms_path.name,
        )

    return {
        "coh_slope_deg_per_100mhz": round(max(coh_slopes), 4),
        "coh_chan_phase_rms_deg": round(max(coh_rmss), 4),
        "med_bl_slope_deg_per_100mhz": round(max(med_bl_slopes), 4),
        "med_bl_chan_phase_rms_deg": round(max(med_bl_rmss), 4),
        "median_row_chan_phase_rms_deg": round(max(row_rmss), 4),
        "field_calibrated": True,
        "unflagged_fraction": round(overall_unflg, 6),
        "source_ms": ms_path.name,
    }


def _measure_bp_flatness_across_workspace(
    workspace_dir: Path,
    bp_cal_name: str,
    dead_ant_names: list[str],
) -> dict[str, Any]:
    """Select calibrated MS (`calibrators.ms` vs `input.ms`) deterministically without truth."""
    candidates_ms = [workspace_dir / "calibrators.ms", workspace_dir / "input.ms"]
    evaluated: list[tuple[int, float, float, int, dict[str, Any]]] = []
    fallback_info = _empty_flatness_dict()

    for order_idx, cand_ms in enumerate(candidates_ms):
        if not (cand_ms / "table.dat").is_file() or has_active_table_lock(cand_ms):
            continue
        fid = resolve_field_id_by_name(cand_ms, bp_cal_name)
        if fid is None:
            continue
        dead_ids = resolve_ant_ids_by_names(cand_ms, dead_ant_names)
        info = _measure_bp_flatness_ms(cand_ms, fid, dead_ids)
        fallback_info = info
        if info.get("field_calibrated") and info.get("coh_slope_deg_per_100mhz") is not None:
            u_frac = float(info.get("unflagged_fraction") or 0.0)
            has_support = 1 if u_frac >= 0.50 else 0
            eff_rms = max(
                float(info["coh_chan_phase_rms_deg"]),
                float(info["med_bl_chan_phase_rms_deg"]),
                float(info["median_row_chan_phase_rms_deg"]),
            )
            eff_slope = max(
                float(info["coh_slope_deg_per_100mhz"]),
                float(info["med_bl_slope_deg_per_100mhz"]),
            )
            evaluated.append((has_support, -eff_rms, -eff_slope, -order_idx, info))
        elif info.get("field_calibrated"):
            evaluated.append((0, -1e9, -1e9, -order_idx, info))

    if not evaluated:
        return fallback_info
    evaluated.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return evaluated[0][4]


def _measure_pcal_flux_single_ms(
    ms_path: Path,
    phase_fid: int,
    dead_ant_ids: list[int],
) -> tuple[float | None, float | None, bool]:
    """Return `(corr_amp_jy, unflagged_par_frac, field_calibrated)` safely per `DATA_DESC_ID`."""
    dead_clauses = "".join(f" && ANTENNA1 != {d} && ANTENNA2 != {d}" for d in dead_ant_ids)
    query_str = f"FIELD_ID == {phase_fid} && ANTENNA1 != ANTENNA2{dead_clauses}"

    with open_ro_table(ms_path) as tb:
        cols = set(tb.colnames())
        has_corr = "CORRECTED_DATA" in cols
        has_raw = "DATA" in cols
        read_cols: list[str] = ["FLAG", "FLAG_ROW"]
        if has_corr:
            read_cols.append("CORRECTED_DATA")
        if has_raw:
            read_cols.append("DATA")

        par_unflg = 0
        par_total = 0
        max_abs_c = 0.0
        max_diff = 0.0
        pol_sums: dict[int, complex] = {}
        pol_cnts: dict[int, int] = {}
        saw_rows = False

        for chunk in iter_ddid_chunks(tb, read_cols, query_expr=query_str):
            saw_rows = True
            flg = chunk.columns["FLAG"].astype(bool)
            flgr = chunk.columns["FLAG_ROW"].astype(bool)
            eff_flg = flg | flgr[np.newaxis, np.newaxis, :]
            npol = flg.shape[0]
            p_idx = (0, 3) if npol >= 4 else ((0, npol - 1) if npol >= 2 else (0,))
            par_eff = eff_flg[list(p_idx), :, :]
            par_unflg += int(np.sum(~par_eff))
            par_total += int(par_eff.size)

            if has_corr:
                corr = chunk.columns["CORRECTED_DATA"]
                par_corr = corr[list(p_idx), :, :]
                good_par = (~par_eff) & np.isfinite(par_corr)
                if np.any(good_par):
                    max_abs_c = max(max_abs_c, float(np.max(np.abs(par_corr[good_par]))))
                    if has_raw:
                        raw = chunk.columns["DATA"]
                        par_raw = raw[list(p_idx), :, :]
                        max_diff = max(
                            max_diff,
                            float(np.max(np.abs(par_corr[good_par] - par_raw[good_par]))),
                        )
                for p_pos, p in enumerate(p_idx):
                    good = (~eff_flg[p]) & np.isfinite(corr[p])
                    if np.any(good):
                        v = corr[p][good].astype(np.complex128)
                        pol_sums[p_pos] = pol_sums.get(p_pos, 0.0 + 0.0j) + complex(np.sum(v))
                        pol_cnts[p_pos] = pol_cnts.get(p_pos, 0) + int(v.size)

    if not saw_rows or par_total == 0:
        return None, None, False

    unflg_frac = float(par_unflg / par_total)
    if not has_corr:
        return None, unflg_frac, False

    field_calibrated = (max_abs_c > 1e-9 and max_diff > 1e-6) if has_raw else (max_abs_c > 1e-9)
    if not field_calibrated:
        return None, unflg_frac, False

    amps = [abs(pol_sums[k] / pol_cnts[k]) for k in sorted(pol_cnts) if pol_cnts[k] > 0]
    corr_amp_jy = float(np.mean(amps)) if amps else None
    return corr_amp_jy, unflg_frac, True


def _check_antenna_gain_consistency_ddid_safe(
    tb: Any,
    tgt_denom_mask: np.ndarray,
    flgr: np.ndarray,
    time_col: np.ndarray,
    ddid_col: np.ndarray,
) -> dict[str, Any]:
    """Verify `CORRECTED_DATA` variance & rank-1 antenna gain closure safely per `DATA_DESC_ID`."""
    cand_rows = np.flatnonzero(tgt_denom_mask & (~flgr))
    if cand_rows.size == 0:
        return {
            "corrected_distinct": False,
            "target_corrected_std_jy": None,
            "rank1_logamp_rms_median": None,
        }

    c_vals_parts: list[np.ndarray] = []
    max_abs_c = 0.0
    max_diff = 0.0
    slice_resids: list[float] = []

    for ddid in sorted(int(x) for x in np.unique(ddid_col[cand_rows])):
        dd_cand = cand_rows[ddid_col[cand_rows] == ddid]
        if dd_cand.size == 0:
            continue
        uniq_t = np.unique(time_col[dd_cand])
        chosen_t = (
            uniq_t[np.linspace(0, uniq_t.size - 1, 8, dtype=int)] if uniq_t.size > 8 else uniq_t
        )
        sel_rows = dd_cand[np.isin(time_col[dd_cand], chosen_t)]
        if sel_rows.size > 3000:
            sel_rows = sel_rows[:3000]

        sub_s = tb.selectrows(sel_rows.tolist())
        try:
            c_samp = np.asarray(sub_s.getcol("CORRECTED_DATA"))
            d_samp = np.asarray(sub_s.getcol("DATA"))
            f_samp = np.asarray(sub_s.getcol("FLAG"), dtype=bool)
            fr_samp = np.asarray(sub_s.getcol("FLAG_ROW"), dtype=bool)
            a1_s = np.asarray(sub_s.getcol("ANTENNA1"), dtype=int)
            a2_s = np.asarray(sub_s.getcol("ANTENNA2"), dtype=int)
            t_s = np.asarray(sub_s.getcol("TIME"), dtype=float)
        finally:
            with contextlib.suppress(Exception):
                sub_s.close()
            with contextlib.suppress(Exception):
                sub_s.done()

        eff_s = f_samp | fr_samp[np.newaxis, np.newaxis, :]
        npol, nchan, _nr = c_samp.shape
        p_indices = [0, npol - 1] if npol >= 2 else [0]

        par_c = c_samp[p_indices, :, :]
        par_d = d_samp[p_indices, :, :]
        par_good = (~eff_s[p_indices, :, :]) & np.isfinite(par_c) & np.isfinite(par_d)
        if not np.any(par_good):
            continue

        c_good = par_c[par_good]
        d_good = par_d[par_good]
        c_vals_parts.append(c_good)
        max_abs_c = max(max_abs_c, float(np.max(np.abs(c_good))))
        max_diff = max(max_diff, float(np.max(np.abs(c_good - d_good))))

        ch_probes = sorted(
            {
                max(0, min(nchan - 1, ch))
                for ch in (nchan // 5, (2 * nchan) // 5, (3 * nchan) // 5, (4 * nchan) // 5)
            }
        )
        for t_val in chosen_t:
            t_mask = t_s == t_val
            if int(np.sum(t_mask)) < 12:
                continue
            t_idx = np.flatnonzero(t_mask)
            for ch in ch_probes:
                for pol in p_indices:
                    good_slice = (
                        (~eff_s[pol, ch, t_idx])
                        & np.isfinite(c_samp[pol, ch, t_idx])
                        & np.isfinite(d_samp[pol, ch, t_idx])
                        & (np.abs(d_samp[pol, ch, t_idx]) > 1e-7)
                        & (np.abs(c_samp[pol, ch, t_idx]) > 1e-7)
                    )
                    if int(np.sum(good_slice)) < 12:
                        continue
                    rs = t_idx[good_slice]
                    ants = np.unique(np.concatenate([a1_s[rs], a2_s[rs]]))
                    if rs.size <= ants.size + 2:
                        continue
                    ant_map = {int(a): idx for idx, a in enumerate(ants)}
                    design = np.zeros((rs.size, ants.size), dtype=float)
                    for k, r_i in enumerate(rs):
                        design[k, ant_map[int(a1_s[r_i])]] += 1.0
                        design[k, ant_map[int(a2_s[r_i])]] += 1.0
                    y = np.log(np.abs(c_samp[pol, ch, rs]) / np.abs(d_samp[pol, ch, rs]))
                    if not np.all(np.isfinite(y)):
                        continue
                    u_fit, *_ = np.linalg.lstsq(design, y, rcond=None)
                    resid = y - design @ u_fit
                    slice_resids.append(float(np.sqrt(np.mean(resid**2))))

    if not c_vals_parts:
        return {
            "corrected_distinct": False,
            "target_corrected_std_jy": None,
            "rank1_logamp_rms_median": None,
        }

    all_c_vals = np.concatenate(c_vals_parts)
    corrected_distinct = max_abs_c > 1e-9 and max_diff > 1e-7
    target_corr_std = float(np.std(all_c_vals))
    med_rank1_rms = float(np.median(slice_resids)) if slice_resids else None
    return {
        "corrected_distinct": corrected_distinct,
        "target_corrected_std_jy": round(target_corr_std, 8),
        "rank1_logamp_rms_median": round(med_rank1_rms, 8) if med_rank1_rms is not None else None,
    }


def _discover_true_target_vis_npz(ref_arrays_dir: Path | None) -> Path | None:
    if ref_arrays_dir is None:
        return None
    candidates: list[Path] = []
    if ref_arrays_dir.is_file():
        candidates.append(ref_arrays_dir)
        if ref_arrays_dir.parent.is_dir():
            candidates.append(ref_arrays_dir.parent / "true_target_vis.npz")
    elif ref_arrays_dir.is_dir():
        candidates.append(ref_arrays_dir / "true_target_vis.npz")

    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            with np.load(cand, allow_pickle=False) as z:
                if {
                    "target_row_indices",
                    "clean_target_row_mask",
                    "clean_chan_mask",
                    "true_rr",
                    "true_ll",
                }.issubset(set(z.files)):
                    return cand
        except Exception:
            continue
    return None


def _measure_coherent_target_vis_fidelity(
    ms_path: Path,
    npz_path: Path,
) -> dict[str, float | None]:
    """Compare target `CORRECTED_DATA` against `true_target_vis.npz` safely per `DATA_DESC_ID`."""
    with np.load(npz_path, allow_pickle=False) as z:
        tgt_row_idx = z["target_row_indices"].astype(np.int64).copy()
        clean_tgt_mask_sub = z["clean_target_row_mask"].astype(bool).copy()
        clean_chan_mask = z["clean_chan_mask"].astype(bool).copy()
        true_rr = z["true_rr"].astype(np.complex128).copy()
        true_ll = z["true_ll"].astype(np.complex128).copy()

    nchan_true, n_tgt = true_rr.shape
    corr_rr = np.zeros((nchan_true, n_tgt), dtype=np.complex128)
    corr_ll = np.zeros((nchan_true, n_tgt), dtype=np.complex128)
    flg_rr = np.ones((nchan_true, n_tgt), dtype=bool)
    flg_ll = np.ones((nchan_true, n_tgt), dtype=bool)
    cross_sub = np.zeros(n_tgt, dtype=bool)

    global_to_tgt_pos = {int(g_row): pos for pos, g_row in enumerate(tgt_row_idx.tolist())}

    with open_ro_table(ms_path) as tb:
        for chunk in iter_ddid_chunks(
            tb,
            ("CORRECTED_DATA", "FLAG", "FLAG_ROW", "ANTENNA1", "ANTENNA2"),
            include_rownumbers=True,
        ):
            c_arr = chunk.columns["CORRECTED_DATA"]
            f_arr = chunk.columns["FLAG"].astype(bool)
            fr_arr = chunk.columns["FLAG_ROW"].astype(bool)
            a1 = chunk.columns["ANTENNA1"]
            a2 = chunk.columns["ANTENNA2"]
            if c_arr.shape[1] != nchan_true:
                continue

            eff = f_arr | fr_arr[np.newaxis, np.newaxis, :]
            npol = c_arr.shape[0]
            p_hi = 3 if npol >= 4 else (npol - 1)

            matched_chunk_pos: list[int] = []
            matched_tgt_pos: list[int] = []
            for local_idx, g_row in enumerate(chunk.rownumbers.tolist()):
                t_pos = global_to_tgt_pos.get(int(g_row))
                if t_pos is not None:
                    matched_chunk_pos.append(local_idx)
                    matched_tgt_pos.append(t_pos)

            if matched_chunk_pos:
                loc_arr = np.array(matched_chunk_pos, dtype=np.int64)
                tp_arr = np.array(matched_tgt_pos, dtype=np.int64)
                corr_rr[:, tp_arr] = c_arr[0, :, :][:, loc_arr]
                corr_ll[:, tp_arr] = c_arr[p_hi, :, :][:, loc_arr]
                flg_rr[:, tp_arr] = eff[0, :, :][:, loc_arr]
                flg_ll[:, tp_arr] = eff[p_hi, :, :][:, loc_arr]
                cross_sub[tp_arr] = a1[loc_arr] != a2[loc_arr]

    row_gate = clean_tgt_mask_sub & cross_sub
    denom_clean_par = 2 * int(np.sum(clean_chan_mask)) * int(np.sum(row_gate))
    valid_rr = (
        (~flg_rr) & clean_chan_mask[:, np.newaxis] & row_gate[np.newaxis, :] & np.isfinite(corr_rr)
    )
    valid_ll = (
        (~flg_ll) & clean_chan_mask[:, np.newaxis] & row_gate[np.newaxis, :] & np.isfinite(corr_ll)
    )
    clean_unflg_frac = (int(np.sum(valid_rr)) + int(np.sum(valid_ll))) / max(denom_clean_par, 1)

    if not np.any(valid_rr) and not np.any(valid_ll):
        return {
            "coherent_amp_ratio": None,
            "coherent_amp_rel_err": None,
            "coherent_phase_deg": None,
            "coherent_pearson_r": None,
            "clean_target_unflagged_fraction": round(clean_unflg_frac, 6),
        }

    vc_parts: list[np.ndarray] = []
    vt_parts: list[np.ndarray] = []
    num = 0.0 + 0.0j
    den = 0.0
    if np.any(valid_rr):
        vc_rr = corr_rr[valid_rr].astype(np.complex128)
        vt_rr = true_rr[valid_rr].astype(np.complex128)
        vc_parts.append(vc_rr)
        vt_parts.append(vt_rr)
        num += complex(np.sum(vc_rr * np.conj(vt_rr)))
        den += float(np.sum(np.abs(vt_rr) ** 2))
    if np.any(valid_ll):
        vc_ll = corr_ll[valid_ll].astype(np.complex128)
        vt_ll = true_ll[valid_ll].astype(np.complex128)
        vc_parts.append(vc_ll)
        vt_parts.append(vt_ll)
        num += complex(np.sum(vc_ll * np.conj(vt_ll)))
        den += float(np.sum(np.abs(vt_ll) ** 2))

    if den <= 1e-18:
        return {
            "coherent_amp_ratio": None,
            "coherent_amp_rel_err": None,
            "coherent_phase_deg": None,
            "coherent_pearson_r": None,
            "clean_target_unflagged_fraction": round(clean_unflg_frac, 6),
        }

    vc_all = np.concatenate(vc_parts)
    vt_all = np.concatenate(vt_parts)
    vc_cent = vc_all - np.mean(vc_all)
    vt_cent = vt_all - np.mean(vt_all)
    var_vc = float(np.sum(np.abs(vc_cent) ** 2))
    var_vt = float(np.sum(np.abs(vt_cent) ** 2))
    if var_vc <= 1e-24 or var_vt <= 1e-24:
        pearson_r = 0.0
    else:
        pearson_r = float(np.abs(np.sum(vc_cent * np.conj(vt_cent))) / math.sqrt(var_vc * var_vt))

    gamma = complex(num / den)
    amp_ratio = float(abs(gamma))
    phs_deg = float(math.degrees(math.atan2(gamma.imag, gamma.real)))
    return {
        "coherent_amp_ratio": round(amp_ratio, 6),
        "coherent_amp_rel_err": round(abs(amp_ratio - 1.0), 6),
        "coherent_phase_deg": round(phs_deg, 4),
        "coherent_pearson_r": round(pearson_r, 6),
        "clean_target_unflagged_fraction": round(clean_unflg_frac, 6),
    }


def _verifier_tclean_child_entry(payload_json_path: str) -> None:
    """Child-process entrypoint executing `mstransform` + `tclean` and measuring raw observables."""
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    import casatasks
    import casatools

    req_path = Path(payload_json_path)
    req = json.loads(req_path.read_text(encoding="utf-8"))
    ms_path = Path(req["ms_path"])
    recipe: dict[str, Any] = req["recipe"]
    out_path = Path(req["out_json"])
    dish_diameter_m = float(req.get("dish_diameter_m") or 25.0)
    offsource_inner_exclusion_beams = float(req.get("offsource_inner_exclusion_beams") or 5.0)
    work_dir = req_path.parent
    os.chdir(work_dir)

    try:
        field_sel = str(recipe.get("field") or "").strip()
        if not field_sel:
            raise ValueError("verifier_tclean_recipe requires non-empty 'field'")
        sub_ms = work_dir / "verify_target.ms"
        casatasks.mstransform(
            vis=str(ms_path),
            outputvis=str(sub_ms),
            field=field_sel,
            spw=str(recipe.get("spw", "")),
            datacolumn="corrected",
        )

        img_prefix = work_dir / "verifier_mosaic"
        imsize_raw = recipe.get("imsize", [640, 640])
        imsize = (
            [int(imsize_raw[0]), int(imsize_raw[1])]
            if isinstance(imsize_raw, list) and len(imsize_raw) >= 2
            else (
                [int(imsize_raw[0]), int(imsize_raw[0])]
                if isinstance(imsize_raw, list)
                else [int(imsize_raw), int(imsize_raw)]
            )
        )
        if recipe.get("cell_arcsec") is not None:
            cell_arcsec = float(recipe["cell_arcsec"])
            cell_str = f"{cell_arcsec}arcsec"
        else:
            raw_cell = str(recipe.get("cell", "2.5arcsec")).strip()
            cell_str = (
                raw_cell
                if raw_cell.lower().endswith(("arcsec", "arcmin", "deg"))
                else f"{raw_cell}arcsec"
            )
            if cell_str.lower().endswith("arcsec"):
                cell_arcsec = float(cell_str[: -len("arcsec")].strip())
            elif cell_str.lower().endswith("arcmin"):
                cell_arcsec = float(cell_str[: -len("arcmin")].strip()) * 60.0
            else:
                cell_arcsec = float(cell_str[: -len("deg")].strip()) * 3600.0

        usemask = str(recipe.get("usemask", "auto-multithresh"))
        do_pbcor = bool(recipe.get("pbcor", True))
        deconvolver = str(recipe.get("deconvolver", "multiscale"))

        tclean_kwargs: dict[str, Any] = {
            "vis": str(sub_ms),
            "imagename": str(img_prefix),
            "spw": "",
            "datacolumn": "data",
            "specmode": str(recipe.get("specmode", "mfs")),
            "gridder": str(recipe.get("gridder", "mosaic")),
            "deconvolver": deconvolver,
            "imsize": imsize,
            "cell": [cell_str, cell_str],
            "weighting": str(recipe.get("weighting", "briggs")),
            "robust": float(recipe.get("robust", 0.5)),
            "pblimit": float(recipe.get("pblimit", 0.2)),
            "usemask": usemask,
            "pbmask": float(recipe.get("pbmask", 0.2)),
            "niter": int(recipe.get("niter", 18000)),
            "threshold": str(recipe.get("threshold", "0.30mJy")),
            "pbcor": do_pbcor,
            "savemodel": "none",
            "interactive": False,
        }
        if recipe.get("phasecenter"):
            tclean_kwargs["phasecenter"] = str(recipe["phasecenter"])
        if deconvolver == "multiscale":
            scales_raw = recipe.get("scales")
            tclean_kwargs["scales"] = (
                [int(s) for s in scales_raw] if scales_raw is not None else [0, 5, 15, 30]
            )
        if deconvolver == "mtmfs" and recipe.get("nterms") is not None:
            tclean_kwargs["nterms"] = int(recipe["nterms"])
        if recipe.get("smallscalebias") is not None:
            tclean_kwargs["smallscalebias"] = float(recipe["smallscalebias"])
        if recipe.get("cycleniter") is not None:
            tclean_kwargs["cycleniter"] = int(recipe["cycleniter"])
        if recipe.get("nmajor") is not None:
            tclean_kwargs["nmajor"] = int(recipe["nmajor"])
        if usemask == "auto-multithresh":
            for k in (
                "sidelobethreshold",
                "noisethreshold",
                "lownoisethreshold",
                "negativethreshold",
                "minbeamfrac",
                "growiterations",
                "dogrowprune",
            ):
                if recipe.get(k) is not None:
                    tclean_kwargs[k] = recipe[k]

        casatasks.tclean(**tclean_kwargs)

        sfx_tt0 = ".tt0" if deconvolver == "mtmfs" and int(recipe.get("nterms") or 1) > 1 else ""

        def _read_plane(suffix: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
            ia = casatools.image()
            qa = casatools.quanta() if suffix.startswith("image") else None
            opened = False
            try:
                opened = bool(ia.open(f"{img_prefix}.{suffix}"))
                if not opened:
                    raise RuntimeError(f"Failed opening verifier product {img_prefix}.{suffix}")
                arr = ia.getchunk()
                msk = ia.getchunk(getmask=True)
                while arr.ndim > 2:
                    arr = arr[..., 0]
                    msk = msk[..., 0]
                wcs_info: dict[str, Any] | None = None
                if suffix in ("image", "image.tt0") and qa is not None:
                    cs = ia.coordsys()
                    try:
                        units = [str(u).strip() for u in cs.units()]
                        refpix = [float(x) for x in cs.referencepixel()["numeric"]]
                        raw_rv = [float(x) for x in cs.referencevalue()["numeric"]]
                        raw_inc = [float(x) for x in cs.increment()["numeric"]]
                    finally:
                        with contextlib.suppress(Exception):
                            cs.done()
                    u0 = units[0] if len(units) > 0 and units[0] else "rad"
                    u1 = units[1] if len(units) > 1 and units[1] else "rad"
                    rv0 = float(qa.convert({"value": raw_rv[0], "unit": u0}, "rad")["value"])
                    rv1 = float(qa.convert({"value": raw_rv[1], "unit": u1}, "rad")["value"])
                    ic0 = float(qa.convert({"value": raw_inc[0], "unit": u0}, "rad")["value"])
                    ic1 = float(qa.convert({"value": raw_inc[1], "unit": u1}, "rad")["value"])
                    if abs(ic0) < 1e-18 or abs(ic1) < 1e-18:
                        raise RuntimeError("Degenerate zero WCS increment in verifier tclean image")
                    freq_hz: float | None = None
                    for ax_idx in range(2, min(len(units), len(raw_rv))):
                        u_ax = units[ax_idx]
                        if "hz" in u_ax.lower():
                            freq_hz = float(
                                qa.convert({"value": raw_rv[ax_idx], "unit": u_ax}, "Hz")["value"]
                            )
                            break
                    b_major_arcsec: float | None = None
                    b_minor_arcsec: float | None = None
                    with contextlib.suppress(Exception):
                        rb = ia.restoringbeam()
                        if isinstance(rb, dict) and "major" in rb:
                            b_major_arcsec = float(qa.convert(rb["major"], "arcsec")["value"])
                            b_minor_arcsec = float(qa.convert(rb["minor"], "arcsec")["value"])
                    wcs_info = {
                        "refpix": refpix,
                        "refval": [rv0, rv1],
                        "inc": [ic0, ic1],
                        "freq_hz": freq_hz,
                        "b_major_arcsec": b_major_arcsec,
                        "b_minor_arcsec": b_minor_arcsec,
                    }
                return arr.astype(np.float64).copy(), msk.astype(bool).copy(), wcs_info
            finally:
                if opened:
                    with contextlib.suppress(Exception):
                        ia.close()
                with contextlib.suppress(Exception):
                    ia.done()
                if qa is not None:
                    with contextlib.suppress(Exception):
                        qa.done()

        img_arr, img_msk, wcs = _read_plane(f"image{sfx_tt0}")
        pb_arr, pb_msk, _ = _read_plane(f"pb{sfx_tt0}")
        res_arr, res_msk, _ = _read_plane(f"residual{sfx_tt0}")
        mod_arr, _mod_msk, _ = _read_plane(f"model{sfx_tt0}")
        pbcor_arr: np.ndarray | None = None
        pbcor_msk: np.ndarray | None = None
        for pbcor_cand in (f"image{sfx_tt0}.pbcor", "image.pbcor"):
            if do_pbcor and (Path(f"{img_prefix}.{pbcor_cand}") / "table.dat").is_file():
                pbcor_arr, pbcor_msk, _ = _read_plane(pbcor_cand)
                break

        pblim = float(recipe.get("pblimit", 0.2))
        pb_gate = np.isfinite(pb_arr) & pb_msk & (pb_arr >= pblim)
        pb_valid = pb_gate & np.isfinite(res_arr) & res_msk & np.isfinite(img_arr) & img_msk
        if int(np.sum(pb_valid)) < 50:
            out_path.write_text(
                json.dumps(
                    {
                        "verifier_tclean_peak_jy": None,
                        "verifier_tclean_offsource_rms_jy": None,
                        "verifier_tclean_offsource_mad_std_jy": None,
                        "verifier_tclean_offsource_mad_over_rms": None,
                        "verifier_tclean_dynamic_range": None,
                        "verifier_tclean_pb_valid": False,
                        "verifier_tclean_error": "Insufficient valid pixels inside primary beam",
                    }
                ),
                encoding="utf-8",
            )
            return

        pb_taper_ok, _pb_stats = verify_pb_taper(
            pb_arr,
            pb_gate,
            inc_x_rad=wcs["inc"][0] if wcs else None,
            inc_y_rad=wcs["inc"][1] if wcs else None,
            freq_hz=wcs.get("freq_hz") if wcs else None,
            dish_diameter_m=dish_diameter_m,
        )

        peak_jy = float(np.nanmax(img_arr[pb_valid]))
        peak_abs_jy = float(np.nanmax(np.abs(img_arr[pb_valid])))

        pbcor_ok = True
        if do_pbcor:
            if pbcor_arr is None or pbcor_msk is None:
                pbcor_ok = False
            else:
                pb25 = pb_valid & pbcor_msk & np.isfinite(pbcor_arr) & (pb_arr >= 0.25)
                if int(np.sum(pb25)) < 25:
                    pbcor_ok = False
                else:
                    pbc_diff = np.abs(pbcor_arr[pb25] * pb_arr[pb25] - img_arr[pb25])
                    med_err = float(np.nanmedian(pbc_diff))
                    p999_err = float(np.nanpercentile(pbc_diff, 99.9))
                    pbc_tol = 0.03 * max(abs(peak_abs_jy), 1e-6)
                    pbcor_ok = (
                        math.isfinite(med_err)
                        and math.isfinite(p999_err)
                        and med_err <= pbc_tol
                        and p999_err <= pbc_tol
                    )

        raw_min_rad = recipe.get("off_source_min_radius_arcsec")
        if raw_min_rad is not None:
            min_rad_arcsec = float(raw_min_rad)
        else:
            b_maj = (
                float(wcs["b_major_arcsec"])
                if (wcs is not None and wcs.get("b_major_arcsec"))
                else (3.0 * cell_arcsec)
            )
            min_rad_arcsec = offsource_inner_exclusion_beams * max(b_maj, cell_arcsec)
        nx, ny = img_arr.shape
        if wcs is not None and recipe.get("phasecenter"):
            ra_c, dec_c = parse_sky_coord_rad(str(recipe["phasecenter"]))
            ra0, dec0 = float(wcs["refval"][0]), float(wcs["refval"][1])
            inc_x, inc_y = float(wcs["inc"][0]), float(wcs["inc"][1])
            l_rad = math.cos(dec_c) * math.sin(ra_c - ra0)
            m_rad = math.sin(dec_c) * math.cos(dec0) - math.cos(dec_c) * math.sin(dec0) * math.cos(
                ra_c - ra0
            )
            px_c = float(wcs["refpix"][0] + l_rad / inc_x)
            py_c = float(wcs["refpix"][1] + m_rad / inc_y)
            cx_arcsec = abs(math.degrees(inc_x) * 3600.0)
            cy_arcsec = abs(math.degrees(inc_y) * 3600.0)
            ix, iy = np.meshgrid(
                np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij"
            )
            radius_arcsec = np.hypot((ix - px_c) * cx_arcsec, (iy - py_c) * cy_arcsec)
        else:
            masked_img = np.where(pb_valid, img_arr, -np.inf)
            pk_flat = int(np.argmax(masked_img))
            px_pk, py_pk = np.unravel_index(pk_flat, img_arr.shape)
            ix, iy = np.meshgrid(
                np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij"
            )
            radius_arcsec = np.hypot(ix - float(px_pk), iy - float(py_pk)) * cell_arcsec

        off_mask = pb_valid & (radius_arcsec >= min_rad_arcsec) & (np.abs(mod_arr) == 0.0)
        if int(np.sum(off_mask)) < 500:
            off_mask = pb_valid & (radius_arcsec >= min_rad_arcsec)
        if int(np.sum(off_mask)) < 200:
            off_mask = pb_valid & (np.abs(mod_arr) == 0.0)
        if int(np.sum(off_mask)) < 50:
            off_mask = pb_valid

        off_vals = res_arr[off_mask]
        n_off_beams: float | None = None
        if wcs is not None and wcs.get("b_major_arcsec") and wcs.get("b_minor_arcsec"):
            beam_pix = (math.pi * float(wcs["b_major_arcsec"]) * float(wcs["b_minor_arcsec"])) / (
                4.0 * math.log(2.0) * cell_arcsec * cell_arcsec
            )
            if beam_pix > 0.0:
                n_off_beams = float(np.sum(off_mask)) / beam_pix
        off_rms_jy = float(np.sqrt(np.mean(off_vals**2)))
        off_med = float(np.median(off_vals))
        off_mad_std_jy = float(1.4826 * np.median(np.abs(off_vals - off_med)))

        mad_over_rms = (
            (off_mad_std_jy / off_rms_jy)
            if (math.isfinite(off_rms_jy) and off_rms_jy > 0.0 and math.isfinite(off_mad_std_jy))
            else None
        )
        dr = (
            (peak_jy / off_rms_jy)
            if (math.isfinite(peak_jy) and math.isfinite(off_rms_jy) and off_rms_jy > 0.0)
            else None
        )
        pb_both_ok = bool(pb_taper_ok and pbcor_ok)

        out_path.write_text(
            json.dumps(
                {
                    "verifier_tclean_peak_jy": round(peak_jy, 6)
                    if math.isfinite(peak_jy)
                    else None,
                    "verifier_tclean_offsource_rms_jy": round(off_rms_jy, 8)
                    if math.isfinite(off_rms_jy)
                    else None,
                    "verifier_tclean_offsource_mad_std_jy": round(off_mad_std_jy, 8)
                    if math.isfinite(off_mad_std_jy)
                    else None,
                    "verifier_tclean_offsource_mad_over_rms": round(mad_over_rms, 6)
                    if mad_over_rms is not None
                    else None,
                    "verifier_tclean_dynamic_range": round(dr, 3) if dr is not None else None,
                    "verifier_tclean_pb_valid": pb_both_ok,
                    "verifier_tclean_offsource_n_beams": n_off_beams,
                }
            ),
            encoding="utf-8",
        )
    finally:
        with contextlib.suppress(Exception):
            casatasks.clearstat()


def _run_verifier_tclean_recipe(
    ms_path: Path,
    recipe: dict[str, Any],
    scratch_root: Path,
    *,
    dish_diameter_m: float = 25.0,
    offsource_inner_exclusion_beams: float = 5.0,
) -> dict[str, Any]:
    """Run fixed-recipe verifier `tclean` in a child subprocess (survives CASA C++ `SIGABRT`)."""
    resolved_scratch = resolve_scratch_root(scratch_root)
    tclean_dir = Path(tempfile.mkdtemp(prefix="verifier_tclean_", dir=str(resolved_scratch)))
    try:
        mpl_dir = tclean_dir / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        req_json = tclean_dir / "tclean_req.json"
        out_json = tclean_dir / "tclean_out.json"

        req_json.write_text(
            json.dumps(
                {
                    "ms_path": str(Path(ms_path).resolve()),
                    "recipe": recipe,
                    "out_json": str(out_json),
                    "dish_diameter_m": float(dish_diameter_m),
                    "offsource_inner_exclusion_beams": float(offsource_inner_exclusion_beams),
                }
            ),
            encoding="utf-8",
        )

        child_env = os.environ.copy()
        child_env["TMPDIR"] = str(tclean_dir)
        child_env["MPLCONFIGDIR"] = str(mpl_dir)
        child_env["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8").strip() or "8"
        child_env["OMP_WAIT_POLICY"] = "PASSIVE"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        child_env["NUMEXPR_NUM_THREADS"] = "1"
        if not child_env.get("CASASITECONFIG", "").strip():
            from radio_bench.config import load_bench_config

            for cand_cfg in (
                Path("/etc/casa/config.py"),
                load_bench_config().casaconfig_path,
            ):
                if cand_cfg.is_file():
                    child_env["CASASITECONFIG"] = str(cand_cfg)
                    break

        src_root = str(Path(__file__).resolve().parents[3])
        existing_py = child_env.get("PYTHONPATH", "").strip()
        child_env["PYTHONPATH"] = f"{src_root}:{existing_py}" if existing_py else src_root

        timeout_sec = float(recipe.get("timeout_sec", 900.0))
        cmd = [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE'); "
                "from radio_bench.grading.extractors.calibrate import _verifier_tclean_child_entry; "
                "_verifier_tclean_child_entry(sys.argv[1])"
            ),
            str(req_json),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tclean_dir),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "verifier_tclean_peak_jy": None,
                "verifier_tclean_offsource_rms_jy": None,
                "verifier_tclean_offsource_mad_over_rms": None,
                "verifier_tclean_dynamic_range": None,
                "verifier_tclean_pb_valid": False,
                "verifier_tclean_returncode": -9,
                "verifier_tclean_error": f"Verifier tclean subprocess timed out after {timeout_sec}s",
            }

        if proc.returncode != 0 or not out_json.is_file():
            stderr_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            return {
                "verifier_tclean_peak_jy": None,
                "verifier_tclean_offsource_rms_jy": None,
                "verifier_tclean_offsource_mad_over_rms": None,
                "verifier_tclean_dynamic_range": None,
                "verifier_tclean_pb_valid": False,
                "verifier_tclean_returncode": int(proc.returncode),
                "verifier_tclean_error": (
                    f"Verifier tclean subprocess exited with code {proc.returncode}: {stderr_tail}"
                ),
            }

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        payload["verifier_tclean_returncode"] = 0
        return payload
    except Exception as exc:
        return {
            "verifier_tclean_peak_jy": None,
            "verifier_tclean_offsource_rms_jy": None,
            "verifier_tclean_offsource_mad_over_rms": None,
            "verifier_tclean_dynamic_range": None,
            "verifier_tclean_pb_valid": False,
            "verifier_tclean_error": f"Verifier tclean runner error: {type(exc).__name__}: {exc}",
        }
    finally:
        shutil.rmtree(tclean_dir, ignore_errors=True)


def extract_calibrate(
    workspace_dir: Path,
    extract_config: ExtractConfig,
    ref_arrays_dir: Path | None = None,
    *,
    scratch_root: Path | None = None,
    table_locked: bool = False,
) -> tuple[dict[str, ObservableRecord], list[str]]:
    """Extract raw calibration physical observables from `/workspace` (zero dataset branches)."""
    from casatasks import clearstat

    with contextlib.suppress(Exception):
        clearstat()

    ws = Path(workspace_dir)
    input_ms = ws / "input.ms"
    cal_ms = ws / "calibrators.ms"
    roles = extract_config.roles
    resolved_scratch = resolve_scratch_root(scratch_root)
    errors: list[str] = []
    obs: dict[str, ObservableRecord] = {}

    all_cal_keys = (
        ("cal.corrected_distinct_from_data", None),
        ("cal.target_corrected_std_jy", "Jy"),
        ("cal.rank1_logamp_rms_median", None),
        ("cal.dead_antenna_flagged_fraction", None),
        ("cal.target_row_retention", None),
        ("cal.unflagged_target_rows_count", None),
        ("cal.bandpass_cal_calibrated", None),
        ("cal.bandpass_cal_unflagged_fraction", None),
        ("cal.bandpass_coh_phase_slope_deg_per_100mhz", "deg/100MHz"),
        ("cal.bandpass_med_bl_phase_slope_deg_per_100mhz", "deg/100MHz"),
        ("cal.bandpass_coh_chan_phase_rms_deg", "deg"),
        ("cal.bandpass_med_bl_chan_phase_rms_deg", "deg"),
        ("cal.bandpass_row_chan_phase_rms_deg", "deg"),
        ("cal.phase_cal_unflagged_fraction", None),
        ("cal.phase_cal_corrected_amp_jy", "Jy"),
        ("cal.fluxscale_table_flux_jy", "Jy"),
        ("cal.clean_target_unflagged_fraction", None),
        ("cal.coherent_vis_amp_rel_err", None),
        ("cal.coherent_vis_phase_err_deg", "deg"),
        ("cal.coherent_vis_pearson_r", None),
        ("cal.verifier_tclean_dr", None),
        ("cal.verifier_tclean_peak_jy", "Jy/beam"),
        ("cal.verifier_tclean_offsource_rms_jy", "Jy/beam"),
        ("cal.verifier_tclean_offsource_mad_over_rms", None),
        ("cal.verifier_tclean_rms_over_thermal_floor", None),
        ("cal.verifier_tclean_pb_valid", None),
    )

    if not (input_ms / "table.dat").is_file():
        errors.append("Missing /workspace/input.ms")
        for key, unit in all_cal_keys:
            obs[key] = ObservableRecord(
                value=False if key == "cal.corrected_distinct_from_data" else None,
                status="missing_artifact",
                unit=unit,
                provenance="input.ms",
            )
        return obs, errors

    in_locked = check_table_multiused(input_ms)
    cal_locked = (cal_ms / "table.dat").is_file() and check_table_multiused(cal_ms)
    if table_locked or in_locked or cal_locked:
        for key, unit in all_cal_keys:
            obs[key] = ObservableRecord(
                value=None,
                status="locked_or_unreadable",
                unit=unit,
                provenance="workspace",
            )
        return obs, errors

    flux_cal = roles.flux_cal[0].strip() if roles.flux_cal else ""
    bp_cal = roles.bandpass_cal[0].strip() if roles.bandpass_cal else flux_cal
    phase_cal = roles.phase_cal[0].strip() if roles.phase_cal else ""
    dead_ant_names = [a.strip() for a in roles.dead_antennas if a.strip()]
    target_names = [t.strip() for t in roles.targets if t.strip()]

    dead_ant_ids = resolve_ant_ids_by_names(input_ms, dead_ant_names)
    target_fids = resolve_field_ids_by_names(input_ms, target_names)
    in_flux_fid = resolve_field_id_by_name(input_ms, flux_cal)
    in_phase_fid = resolve_field_id_by_name(input_ms, phase_cal)

    has_corrected = False
    corrected_distinct = False
    target_corr_std: float | None = None
    rank1_logamp_rms: float | None = None
    dead_ant_flagged_frac: float | None = None
    target_row_retention: float | None = None
    unflagged_target_rows_count = 0

    try:
        with open_ro_table(input_ms) as tb:
            cols = set(tb.colnames())
            has_corrected = "CORRECTED_DATA" in cols
            fids_1d = np.asarray(tb.getcol("FIELD_ID"), dtype=np.int64)
            a1_1d = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int64)
            a2_1d = np.asarray(tb.getcol("ANTENNA2"), dtype=np.int64)
            time_1d = np.asarray(tb.getcol("TIME"), dtype=np.float64)
            flgr_1d = np.asarray(tb.getcol("FLAG_ROW"), dtype=bool)
            ddid_1d = np.asarray(tb.getcol("DATA_DESC_ID"), dtype=np.int64)

            cross_1d = a1_1d != a2_1d
            if dead_ant_ids:
                dead_mask_1d = (
                    np.isin(a1_1d, dead_ant_ids) | np.isin(a2_1d, dead_ant_ids)
                ) & cross_1d
            else:
                dead_mask_1d = np.zeros_like(cross_1d)

            tgt_base_1d = np.isin(fids_1d, target_fids) & cross_1d
            if extract_config.target_retention_excludes_dead_antenna and dead_ant_ids:
                tgt_denom_1d = (
                    tgt_base_1d & (~np.isin(a1_1d, dead_ant_ids)) & (~np.isin(a2_1d, dead_ant_ids))
                )
            else:
                tgt_denom_1d = tgt_base_1d

            dead_flg_cells = 0
            dead_tot_cells = 0
            tgt_unflg_par_cells = 0
            tgt_tot_par_cells = 0
            unflagged_target_rows_count = 0

            for chunk in iter_ddid_chunks(tb, ("FLAG", "FLAG_ROW"), include_rownumbers=True):
                c_flg = chunk.columns["FLAG"].astype(bool)
                c_flgr = chunk.columns["FLAG_ROW"].astype(bool)
                rnums = chunk.rownumbers

                c_dead = dead_mask_1d[rnums]
                if np.any(c_dead):
                    d_eff = c_flg[:, :, c_dead] | c_flgr[np.newaxis, np.newaxis, c_dead]
                    dead_flg_cells += int(np.sum(d_eff))
                    dead_tot_cells += int(d_eff.size)

                c_tgt = tgt_denom_1d[rnums]
                if np.any(c_tgt):
                    p_idx_ret = [0, c_flg.shape[0] - 1] if c_flg.shape[0] >= 2 else [0]
                    eff_par = (
                        c_flg[p_idx_ret, :, :][:, :, c_tgt] | c_flgr[np.newaxis, np.newaxis, c_tgt]
                    )
                    tgt_unflg_par_cells += int(np.sum(~eff_par))
                    tgt_tot_par_cells += int(eff_par.size)
                    row_all_flg = np.all(eff_par, axis=(0, 1))
                    unflagged_target_rows_count += int(np.sum(~row_all_flg))

            if dead_ant_ids:
                dead_ant_flagged_frac = (
                    float(dead_flg_cells / dead_tot_cells) if dead_tot_cells > 0 else 1.0
                )
            else:
                dead_ant_flagged_frac = 1.0

            target_row_retention = (
                float(tgt_unflg_par_cells / tgt_tot_par_cells) if tgt_tot_par_cells > 0 else 0.0
            )

            if has_corrected and unflagged_target_rows_count > 0:
                gain_info = _check_antenna_gain_consistency_ddid_safe(
                    tb, tgt_denom_1d, flgr_1d, time_1d, ddid_1d
                )
                corrected_distinct = bool(gain_info["corrected_distinct"])
                target_corr_std = gain_info["target_corrected_std_jy"]
                rank1_logamp_rms = gain_info["rank1_logamp_rms_median"]
    except Exception as exc:
        errors.append(f"Failed inspecting input.ms retention/gain consistency: {exc}")

    if not has_corrected:
        errors.append("CORRECTED_DATA column missing from /workspace/input.ms")
    elif not corrected_distinct:
        errors.append("CORRECTED_DATA is empty, non-finite, or identical to uncalibrated DATA")

    obs["cal.corrected_distinct_from_data"] = ObservableRecord(
        value=bool(has_corrected and corrected_distinct),
        status="measured",
        provenance="input.ms::CORRECTED_DATA",
    )
    obs["cal.target_corrected_std_jy"] = finite_observable(
        target_corr_std,
        unit="Jy",
        provenance="input.ms::CORRECTED_DATA",
        missing_status="missing_artifact",
    )
    obs["cal.rank1_logamp_rms_median"] = finite_observable(
        rank1_logamp_rms,
        provenance="input.ms::CORRECTED_DATA",
        missing_status="missing_artifact",
    )
    obs["cal.dead_antenna_flagged_fraction"] = finite_observable(
        dead_ant_flagged_frac,
        provenance="input.ms::FLAG",
        missing_status="extractor_error",
        ndigits=6,
    )
    obs["cal.target_row_retention"] = finite_observable(
        target_row_retention,
        provenance="input.ms::FLAG",
        missing_status="extractor_error",
        ndigits=6,
    )
    obs["cal.unflagged_target_rows_count"] = ObservableRecord(
        value=int(unflagged_target_rows_count),
        status="measured",
        provenance="input.ms::FLAG",
    )

    # 2. Bandpass flatness observables
    flat_info = _empty_flatness_dict()
    if has_corrected and corrected_distinct and bp_cal:
        try:
            flat_info = _measure_bp_flatness_across_workspace(ws, bp_cal, dead_ant_names)
        except Exception as exc:
            errors.append(f"Failed measuring bandpass flatness on {bp_cal}: {exc}")

    bp_src_prov = f"{flat_info.get('source_ms') or 'input.ms'}::CORRECTED_DATA"
    obs["cal.bandpass_cal_calibrated"] = ObservableRecord(
        value=bool(flat_info.get("field_calibrated")),
        status="measured" if bp_cal else "not_applicable",
        provenance=bp_src_prov,
    )
    obs["cal.bandpass_cal_unflagged_fraction"] = (
        ObservableRecord(value=None, status="not_applicable", provenance=bp_src_prov)
        if not bp_cal
        else finite_observable(
            flat_info.get("unflagged_fraction"),
            provenance=bp_src_prov,
            missing_status="missing_artifact",
        )
    )
    for obs_key, info_key, unit in (
        ("cal.bandpass_coh_phase_slope_deg_per_100mhz", "coh_slope_deg_per_100mhz", "deg/100MHz"),
        (
            "cal.bandpass_med_bl_phase_slope_deg_per_100mhz",
            "med_bl_slope_deg_per_100mhz",
            "deg/100MHz",
        ),
        ("cal.bandpass_coh_chan_phase_rms_deg", "coh_chan_phase_rms_deg", "deg"),
        ("cal.bandpass_med_bl_chan_phase_rms_deg", "med_bl_chan_phase_rms_deg", "deg"),
        ("cal.bandpass_row_chan_phase_rms_deg", "median_row_chan_phase_rms_deg", "deg"),
    ):
        if not bp_cal:
            obs[obs_key] = ObservableRecord(
                value=None, status="not_applicable", unit=unit, provenance=bp_src_prov
            )
        else:
            obs[obs_key] = finite_observable(
                flat_info.get(info_key),
                unit=unit,
                provenance=bp_src_prov,
                missing_status="missing_artifact",
            )

    # 3. Phase calibrator fluxscale (`fluxscale` VisCal table + MS `CORRECTED_DATA`, zero truth sorting)
    fluxscale_table_jy: float | None = None
    fluxscale_prov: str | None = None
    pcal_corr_amp_jy: float | None = None
    pcal_unflg_frac: float | None = None
    pcal_source_ms: str | None = None

    if has_corrected and corrected_distinct and phase_cal:
        try:
            if flux_cal:
                fluxscale_table_jy, fluxscale_prov = discover_fluxscale_from_caltables(
                    ws,
                    flux_cal_name=flux_cal,
                    phase_cal_name=phase_cal,
                    fallback_flux_fid=in_flux_fid,
                    fallback_phase_fid=in_phase_fid,
                )

            cand_mss: list[Path] = []
            if (cal_ms / "table.dat").is_file() and not has_active_table_lock(cal_ms):
                cand_mss.append(cal_ms)
            cand_mss.append(input_ms)

            corr_candidates: list[tuple[float, int, str]] = []
            unflg_fractions: list[float] = []
            for ord_idx, cms in enumerate(cand_mss):
                pfid = resolve_field_id_by_name(
                    cms, phase_cal, fallback_fid=in_phase_fid if cms == input_ms else None
                )
                if pfid is None:
                    continue
                d_ids = resolve_ant_ids_by_names(cms, dead_ant_names)
                c_amp, u_frac, is_cal = _measure_pcal_flux_single_ms(cms, pfid, d_ids)
                if u_frac is not None:
                    unflg_fractions.append(u_frac)
                if is_cal and c_amp is not None and math.isfinite(c_amp):
                    corr_candidates.append((c_amp, float(u_frac or 0.0), ord_idx, cms.name))

            pcal_unflg_frac = max(unflg_fractions) if unflg_fractions else None
            if corr_candidates:
                # Prefer MS candidates with substantial unflagged support before low-row remnants
                max_u = max(item[1] for item in corr_candidates)
                viable = (
                    [item for item in corr_candidates if item[1] >= 0.5 * max_u]
                    if max_u > 0.0
                    else corr_candidates
                )
                if fluxscale_table_jy is not None:
                    viable.sort(key=lambda item: (abs(item[0] - fluxscale_table_jy), item[2]))
                else:
                    viable.sort(key=lambda item: (0 if abs(item[0] - 1.0) >= 1e-2 else 1, item[2]))
                pcal_corr_amp_jy = viable[0][0]
                pcal_source_ms = viable[0][3]
        except Exception as exc:
            errors.append(f"Failed measuring phase calibrator fluxscale: {exc}")

    obs["cal.phase_cal_unflagged_fraction"] = (
        ObservableRecord(
            value=None, status="not_applicable", provenance=f"{pcal_source_ms or 'input.ms'}::FLAG"
        )
        if not phase_cal
        else finite_observable(
            pcal_unflg_frac,
            provenance=f"{pcal_source_ms or 'input.ms'}::FLAG",
            missing_status="missing_artifact",
            ndigits=6,
        )
    )
    obs["cal.phase_cal_corrected_amp_jy"] = (
        ObservableRecord(
            value=None,
            status="not_applicable",
            unit="Jy",
            provenance=f"{pcal_source_ms}::CORRECTED_DATA" if pcal_source_ms else None,
        )
        if not phase_cal
        else finite_observable(
            pcal_corr_amp_jy,
            unit="Jy",
            provenance=f"{pcal_source_ms}::CORRECTED_DATA" if pcal_source_ms else None,
            missing_status="missing_artifact",
            ndigits=6,
        )
    )
    obs["cal.fluxscale_table_flux_jy"] = (
        ObservableRecord(value=None, status="not_applicable", unit="Jy", provenance=fluxscale_prov)
        if not (flux_cal and phase_cal)
        else finite_observable(
            fluxscale_table_jy,
            unit="Jy",
            provenance=fluxscale_prov,
            missing_status="missing_artifact",
            ndigits=6,
        )
    )

    # 4a. Coherent target visibility fidelity (when `true_target_vis.npz` is staged)
    tvis_npz = _discover_true_target_vis_npz(ref_arrays_dir)
    coh_info: dict[str, float | None] = {}
    if (
        tvis_npz is not None
        and has_corrected
        and corrected_distinct
        and unflagged_target_rows_count > 0
    ):
        try:
            coh_info = _measure_coherent_target_vis_fidelity(input_ms, tvis_npz)
        except Exception as exc:
            errors.append(f"Failed measuring coherent target visibility fidelity: {exc}")

    for obs_key, info_key, unit in (
        ("cal.clean_target_unflagged_fraction", "clean_target_unflagged_fraction", None),
        ("cal.coherent_vis_amp_rel_err", "coherent_amp_rel_err", None),
        ("cal.coherent_vis_phase_err_deg", "coherent_phase_deg", "deg"),
        ("cal.coherent_vis_pearson_r", "coherent_pearson_r", None),
    ):
        if tvis_npz is None:
            obs[obs_key] = ObservableRecord(value=None, status="not_applicable", unit=unit)
        else:
            obs[obs_key] = finite_observable(
                coh_info.get(info_key),
                unit=unit,
                provenance="input.ms::CORRECTED_DATA vs true_target_vis.npz",
                missing_status="missing_artifact",
            )

    # 4b. Fixed-recipe verifier `tclean` (when `verifier_tclean_recipe` is configured)
    recipe_model = extract_config.geometry.verifier_tclean_recipe
    tclean_info: dict[str, Any] = {}
    ran_tclean = False
    if (
        recipe_model is not None
        and has_corrected
        and corrected_distinct
        and unflagged_target_rows_count > 0
    ):
        ran_tclean = True
        recipe_dict = recipe_model.model_dump(exclude_none=True)
        tclean_info = _run_verifier_tclean_recipe(
            input_ms,
            recipe_dict,
            resolved_scratch,
            dish_diameter_m=extract_config.geometry.dish_diameter_m,
            offsource_inner_exclusion_beams=extract_config.geometry.offsource_inner_exclusion_beams,
        )
        if tclean_info.get("verifier_tclean_error"):
            errors.append(str(tclean_info["verifier_tclean_error"]))

    for obs_key, info_key, unit in (
        ("cal.verifier_tclean_dr", "verifier_tclean_dynamic_range", None),
        ("cal.verifier_tclean_peak_jy", "verifier_tclean_peak_jy", "Jy/beam"),
        ("cal.verifier_tclean_offsource_rms_jy", "verifier_tclean_offsource_rms_jy", "Jy/beam"),
        (
            "cal.verifier_tclean_offsource_mad_over_rms",
            "verifier_tclean_offsource_mad_over_rms",
            None,
        ),
        ("cal.verifier_tclean_pb_valid", "verifier_tclean_pb_valid", None),
    ):
        if recipe_model is None:
            obs[obs_key] = ObservableRecord(value=None, status="not_applicable", unit=unit)
        elif not ran_tclean:
            obs[obs_key] = ObservableRecord(
                value=None,
                status="missing_artifact",
                unit=unit,
                provenance="verifier_tclean",
            )
        elif obs_key == "cal.verifier_tclean_pb_valid":
            val = tclean_info.get(info_key)
            obs[obs_key] = ObservableRecord(
                value=bool(val) if val is not None else None,
                status="measured" if val is not None else "extractor_error",
                unit=unit,
                provenance="verifier_tclean",
            )
        else:
            obs[obs_key] = finite_observable(
                tclean_info.get(info_key),
                unit=unit,
                provenance="verifier_tclean",
                missing_status="extractor_error",
            )

    floor_ratio: float | None = None
    if ran_tclean and target_fids:
        try:
            sigma_nat_jy, _ = natural_thermal_noise_jy(input_ms, target_fids)
            floor_ratio = thermal_floor_ratio(
                tclean_info.get("verifier_tclean_offsource_rms_jy"),
                sigma_nat_jy,
                tclean_info.get("verifier_tclean_offsource_n_beams"),
            )
        except Exception as exc:
            errors.append(f"Thermal noise estimate failed: {type(exc).__name__}: {exc}")
    obs["cal.verifier_tclean_rms_over_thermal_floor"] = (
        finite_observable(floor_ratio, provenance="verifier_tclean vs input.ms")
        if floor_ratio is not None
        else ObservableRecord(value=None, status="not_applicable")
    )

    return obs, errors
