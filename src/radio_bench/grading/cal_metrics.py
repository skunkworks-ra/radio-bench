"""Deterministic scientific verifier for calibrate-stage tasks (s1_calibrate, r1_3c391_calibrate)."""

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

from radio_bench.grading.precal_metrics import _open_ro_table, has_active_table_lock

PASS_REWARD_MIN_DEFAULT = 0.85
MIN_CALIBRATOR_UNFLAGGED_FRAC = 0.50
MIN_CLEAN_TARGET_UNFLAGGED_FRAC = 0.75
MIN_SYNTH_PEARSON_R = 0.25
MAX_RANK1_LOGAMP_RMS = 0.12


def resolve_scratch_root(scratch_root: Path | None = None) -> Path:
    """Resolve a writable scratch root strictly outside `/tmp`."""
    if scratch_root is not None:
        resolved = Path(scratch_root).resolve()
        if str(resolved) == "/tmp" or str(resolved).startswith("/tmp/"):
            raise ValueError(f"scratch_root must never use /tmp: {resolved}")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    env_tmp = os.environ.get("TMPDIR", "").strip()
    if env_tmp and env_tmp != "/tmp" and not env_tmp.startswith("/tmp/"):
        cand = Path(env_tmp).resolve()
        if str(cand) != "/tmp" and not str(cand).startswith("/tmp/"):
            cand.mkdir(parents=True, exist_ok=True)
            return cand

    var_tmp = Path("/var/tmp/radio-bench-verifier")
    if Path("/var/tmp").exists() and os.access("/var/tmp", os.W_OK):
        var_tmp.mkdir(parents=True, exist_ok=True)
        return var_tmp

    from radio_bench.config import load_bench_config

    cfg_tmp = load_bench_config().tmp_dir
    cfg_tmp.mkdir(parents=True, exist_ok=True)
    return cfg_tmp


def _parse_sky_coord_rad(
    coord_str_ra: str, coord_str_dec: str | None = None
) -> tuple[float, float]:
    from astropy.coordinates import SkyCoord

    if coord_str_dec is None:
        cleaned = str(coord_str_ra).strip()
        for prefix in ("J2000", "ICRS"):
            if cleaned.upper().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        parts = cleaned.split()
        if len(parts) < 2:
            raise ValueError(f"Cannot parse sky coordinate string: {coord_str_ra!r}")
        ra_tok, dec_tok = parts[0], parts[1]
    else:
        ra_tok, dec_tok = str(coord_str_ra).strip(), str(coord_str_dec).strip()

    sc = SkyCoord(ra_tok, dec_tok, frame="icrs")
    return float(sc.ra.rad), float(sc.dec.rad)


def _extract_cal_truth(ground_truth: dict[str, Any]) -> dict[str, Any]:
    top_gt = ground_truth
    if "calibrate_ground_truth" in ground_truth and isinstance(
        ground_truth["calibrate_ground_truth"], dict
    ):
        gt = ground_truth["calibrate_ground_truth"]
    else:
        gt = ground_truth

    suite_raw = str(gt.get("suite") or top_gt.get("suite") or "").strip().lower()
    is_suite_r = suite_raw == "r" or "3c391" in str(gt.get("task_id", top_gt.get("task_id", "")))

    fields_sub = gt.get("fields") if isinstance(gt.get("fields"), dict) else {}
    flux_cal = str(gt.get("flux_calibrator") or fields_sub.get("flux_calibrator") or "J1331+3030")
    flux_fid = int(
        gt.get("flux_calibrator_field_id")
        if gt.get("flux_calibrator_field_id") is not None
        else fields_sub.get("flux_calibrator_id", 0)
    )
    phase_cal = str(
        gt.get("phase_calibrator") or fields_sub.get("phase_calibrator") or "J1822-0938"
    )
    phase_fid = int(
        gt.get("phase_calibrator_field_id")
        if gt.get("phase_calibrator_field_id") is not None
        else fields_sub.get("phase_calibrator_id", 1)
    )

    if "target_field_ids" in gt and isinstance(gt["target_field_ids"], list):
        target_fids = [int(x) for x in gt["target_field_ids"]]
    elif "target_field_ids" in fields_sub and isinstance(fields_sub["target_field_ids"], list):
        target_fids = [int(x) for x in fields_sub["target_field_ids"]]
    else:
        target_fids = [2, 3, 4, 5, 6, 7, 8] if is_suite_r else [2, 3]

    dead_ant_name = (
        str(gt.get("dead_antenna") or ("ea13" if is_suite_r else "ea07")).strip().lower()
    )
    dead_ant_id = int(
        gt.get("dead_antenna_id")
        if gt.get("dead_antenna_id") is not None
        else (11 if is_suite_r else 6)
    )

    tols = gt.get("tolerances") if isinstance(gt.get("tolerances"), dict) else {}
    top_tols = top_gt.get("tolerances") if isinstance(top_gt.get("tolerances"), dict) else {}

    dead_min = float(tols.get("dead_antenna_flagged_min", 0.99))
    excl_dead_raw = None
    for mapping in (tols, gt, top_gt):
        if "target_retention_excludes_dead_antenna" in mapping:
            excl_dead_raw = mapping["target_retention_excludes_dead_antenna"]
            break
    excludes_dead = bool((not is_suite_r) if excl_dead_raw is None else excl_dead_raw)

    if is_suite_r:
        ret_min = float(tols.get("target_unflagged_rows_min", 0.70))
        ret_max: float | None = float(tols.get("target_unflagged_rows_max", 0.90))
    else:
        ret_min = float(tols.get("min_clean_target_row_retention", 0.85))
        ret_max = None

    pcal_truth_raw = gt.get("phase_calibrator_truth") or top_gt.get("phase_calibrator_truth")
    pcal_truth = pcal_truth_raw if isinstance(pcal_truth_raw, dict) else {}
    pcal_flux_from_truth = pcal_truth.get(
        "spw_mean_flux_jy",
        pcal_truth.get("flux_at_ref_freq_jy"),
    )
    pcal_flux_candidate = gt.get(
        "j1822_expected_flux_jy",
        gt.get(
            "expected_phase_calibrator_flux_jy",
            pcal_flux_from_truth
            if pcal_flux_from_truth is not None
            else (2.296 if is_suite_r else 2.150366),
        ),
    )
    expected_pcal_flux = float(pcal_flux_candidate)

    pcal_rel_tol_candidate = tols.get(
        "j1822_flux_rel_tol",
        tols.get(
            "max_fluxscale_rel_err",
            pcal_truth.get("fluxscale_rel_tolerance", top_tols.get("max_fluxscale_rel_err")),
        ),
    )
    pcal_rel_tol = float(
        pcal_rel_tol_candidate
        if pcal_rel_tol_candidate is not None
        else (0.08 if is_suite_r else 0.05)
    )

    max_slope_deg_100mhz = float(tols.get("max_j1331_phase_slope_deg_per_100mhz", 5.0))
    max_chan_rms_deg = float(tols.get("j1331_channel_phase_rms_max_deg", 8.0))

    max_coh_amp_err = float(tols.get("max_target_coherent_amp_rel_err", 0.08))
    max_coh_phs_deg = float(tols.get("max_target_coherent_phase_deg", 10.0))
    min_pearson_r = float(tols.get("min_target_coherent_pearson_r", MIN_SYNTH_PEARSON_R))
    tclean_dr_min = float(tols.get("verifier_tclean_dr_min", 180.0))

    noise = gt.get("noise") if isinstance(gt.get("noise"), dict) else top_gt.get("noise", {})
    if is_suite_r:
        min_tclean_rms_jy = float(
            tols.get("verifier_tclean_offsource_rms_min_jy_per_beam", 0.08e-3)
        )
    else:
        sigma_th = float(noise.get("sigma_th_jy_per_beam", 0.00014895))
        min_tclean_rms_jy = 0.35 * sigma_th

    pass_min = float(
        gt.get(
            "pass_reward_min",
            tols.get("pass_reward_min", top_tols.get("pass_reward_min", PASS_REWARD_MIN_DEFAULT)),
        )
    )

    cal_weights_src: dict[str, Any] = {}
    for cand_weights in (
        gt.get("calibration_weights"),
        gt.get("calibrate_weights"),
        top_gt.get("calibration_weights"),
        top_gt.get("calibrate_weights"),
        gt.get("weights"),
    ):
        if isinstance(cand_weights, dict) and cand_weights:
            cal_weights_src = cand_weights
            break
    weights = {
        "retention_and_dead_ant": float(
            cal_weights_src.get("corrected_dead_ant_target_retention", 0.20)
        ),
        "bandpass_flatness": float(cal_weights_src.get("bandpass_delay_flatness", 0.25)),
        "fluxscale_accuracy": float(cal_weights_src.get("fluxscale_accuracy", 0.25)),
        "target_fidelity": float(
            cal_weights_src.get(
                "target_coherent_visibility",
                cal_weights_src.get("target_verifier_tclean_dr", 0.30),
            )
        ),
    }

    return {
        "is_suite_r": is_suite_r,
        "flux_calibrator": flux_cal,
        "flux_calibrator_id": flux_fid,
        "phase_calibrator": phase_cal,
        "phase_calibrator_id": phase_fid,
        "target_field_ids": target_fids,
        "dead_antenna": dead_ant_name,
        "dead_antenna_id": dead_ant_id,
        "excludes_dead_in_retention": excludes_dead,
        "dead_antenna_flagged_min": dead_min,
        "retention_min": ret_min,
        "retention_max": ret_max,
        "expected_pcal_flux_jy": expected_pcal_flux,
        "pcal_flux_rel_tol": pcal_rel_tol,
        "max_slope_deg_100mhz": max_slope_deg_100mhz,
        "max_chan_rms_deg": max_chan_rms_deg,
        "max_coh_amp_err": max_coh_amp_err,
        "max_coh_phs_deg": max_coh_phs_deg,
        "min_pearson_r": min_pearson_r,
        "verifier_tclean_dr_min": tclean_dr_min,
        "min_verifier_tclean_rms_jy": min_tclean_rms_jy,
        "verifier_tclean_recipe": gt.get("verifier_tclean_recipe"),
        "true_target_vis_file": gt.get("true_target_vis_file")
        or top_gt.get("true_target_vis_file"),
        "sky_model": gt.get("sky_model") or top_gt.get("sky_model"),
        "corruptions_summary": gt.get("corruptions_summary") or top_gt.get("corruptions_summary"),
        "pass_reward_min": pass_min,
        "weights": weights,
    }


def _resolve_field_id_by_name(
    ms_path: Path,
    field_name: str,
    *,
    fallback_fid: int | None = None,
) -> int | None:
    """Resolve a field's numeric `FIELD_ID` inside `ms_path/FIELD` strictly by name."""
    if not (ms_path / "FIELD" / "table.dat").is_file():
        return fallback_fid
    with _open_ro_table(ms_path / "FIELD") as tb_f:
        fnames = [str(x).strip() for x in tb_f.getcol("NAME")]
    target_upper = field_name.strip().upper()
    for idx, fn in enumerate(fnames):
        if fn.upper() == target_upper:
            return idx
    if fallback_fid is not None and 0 <= fallback_fid < len(fnames):
        return fallback_fid
    return None


def _resolve_dead_ant_id(ms_path: Path, dead_ant_name: str, fallback_aid: int) -> int:
    if not (ms_path / "ANTENNA" / "table.dat").is_file():
        return fallback_aid
    with _open_ro_table(ms_path / "ANTENNA") as tb_a:
        ant_names = [str(x).strip().lower() for x in tb_a.getcol("NAME")]
    key = dead_ant_name.strip().lower()
    if key in ant_names:
        return ant_names.index(key)
    return fallback_aid


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


def _measure_j1331_flatness(
    ms_path: Path,
    flux_fid: int,
    dead_ant_id: int,
) -> dict[str, Any]:
    """Measure J1331+3030 phase flatness using unit-phasor & outlier-clipped averaging."""
    with _open_ro_table(ms_path / "SPECTRAL_WINDOW") as tb_spw:
        freqs_hz = tb_spw.getcol("CHAN_FREQ")[:, 0].astype(float)

    with _open_ro_table(ms_path) as tb:
        cols = set(tb.colnames())
        if "CORRECTED_DATA" not in cols:
            return _empty_flatness_dict(field_calibrated=False, source_ms=ms_path.name)
        has_raw = "DATA" in cols
        sub = tb.query(
            f"FIELD_ID == {flux_fid} && ANTENNA1 != ANTENNA2 "
            f"&& ANTENNA1 != {dead_ant_id} && ANTENNA2 != {dead_ant_id}"
        )
        try:
            if int(sub.nrows()) == 0:
                return _empty_flatness_dict(field_calibrated=False, source_ms=ms_path.name)
            corr = sub.getcol("CORRECTED_DATA")
            raw = sub.getcol("DATA") if has_raw else None
            flag = sub.getcol("FLAG")
            flag_row = sub.getcol("FLAG_ROW")
            a1 = sub.getcol("ANTENNA1")
            a2 = sub.getcol("ANTENNA2")
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()

    eff_flag = flag | flag_row[np.newaxis, np.newaxis, :]
    npol, nchan, nrow = corr.shape
    pol_indices = (0, 3) if npol >= 4 else ((0, npol - 1) if npol >= 2 else (0,))

    par_eff_flag = eff_flag[list(pol_indices), :, :]
    unflagged_frac = float(1.0 - np.mean(par_eff_flag))

    par_corr = corr[list(pol_indices), :, :]
    good_par = (~par_eff_flag) & np.isfinite(par_corr)
    if not np.any(good_par):
        return _empty_flatness_dict(
            field_calibrated=False,
            unflagged_fraction=unflagged_frac,
            source_ms=ms_path.name,
        )

    max_abs_c = float(np.max(np.abs(par_corr[good_par])))
    if raw is not None:
        par_raw = raw[list(pol_indices), :, :]
        max_diff = float(np.max(np.abs(par_corr[good_par] - par_raw[good_par])))
        field_calibrated = max_abs_c > 1e-9 and max_diff > 1e-6
    else:
        field_calibrated = max_abs_c > 1e-9

    if not field_calibrated or unflagged_frac < MIN_CALIBRATOR_UNFLAGGED_FRAC:
        return _empty_flatness_dict(
            field_calibrated=field_calibrated,
            unflagged_fraction=unflagged_frac,
            source_ms=ms_path.name,
        )

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

    # Mask extreme amplitude-outlier rows and narrowband RFI cells (> 5x median amplitude)
    # and normalize each valid visibility to a unit phasor c / max(|c|, 1e-12) before summing over time.
    clean_flag = eff_flag.copy()
    for p_idx in pol_indices:
        good_p = (~eff_flag[p_idx]) & np.isfinite(corr[p_idx]) & (np.abs(corr[p_idx]) > 1e-12)
        clean_flag[p_idx] = ~good_p
        valid_r = np.flatnonzero(np.sum(good_p, axis=0) >= 8)
        if valid_r.size >= 4:
            row_amps = np.array(
                [float(np.median(np.abs(corr[p_idx, good_p[:, r], r]))) for r in valid_r],
                dtype=float,
            )
            med_row_amp = float(np.median(row_amps))
            if med_row_amp > 1e-12:
                outlier_mask = row_amps > (5.0 * med_row_amp)
                # Only drop outlier rows if at least half of valid rows remain
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

    # Exclude persistent RFI channels that are >90% flagged relative to typical active channels
    chan_counts = np.sum(~clean_flag[list(pol_indices), :, :], axis=(0, 2))
    active_chan_counts = chan_counts[chan_counts > 0]
    med_chan_count = float(np.median(active_chan_counts)) if active_chan_counts.size > 0 else 0.0
    min_chan_support = max(2, int(0.10 * med_chan_count)) if med_chan_count >= 20.0 else 1

    # 1. Coherent unit-phasor channel spectrum across unflagged cross-hand rows
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
        return _empty_flatness_dict(
            field_calibrated=field_calibrated,
            unflagged_fraction=unflagged_frac,
            source_ms=ms_path.name,
        )

    f_valid_hz = freqs_hz[ch_indices]
    phs_coh_deg = _unwrap_centered_deg(np.array(vec_spec, dtype=np.complex128))
    slope_hz, _ = np.polyfit(f_valid_hz - float(np.mean(f_valid_hz)), phs_coh_deg, 1)
    coh_slope = abs(float(slope_hz) * 1.0e8)
    coh_rms = float(np.sqrt(np.mean(phs_coh_deg**2)))

    # 2. Per-baseline unit-phasor time-averaged spectrum slope & RMS (prevents multi-baseline cancellation)
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
            if int(np.sum(good_ch)) < 16:
                continue
            ch_sel = np.flatnonzero(good_ch)
            spec_bl = np.array(
                [np.mean(up[c, ~fp[c]]) for c in ch_sel],
                dtype=np.complex128,
            )
            if not np.all(np.isfinite(spec_bl)) or np.any(np.abs(spec_bl) <= 1e-12):
                continue
            ph_bl = _unwrap_centered_deg(spec_bl)
            fc = freqs_hz[ch_sel]
            sl_bl, _ = np.polyfit(fc - float(np.mean(fc)), ph_bl, 1)
            bl_slopes.append(abs(float(sl_bl) * 1.0e8))
            bl_rms.append(float(np.sqrt(np.mean(ph_bl**2))))

    # 3. Per-integration channel phase RMS (sampled up to 2000 rows per pol for speed)
    row_rms_list: list[float] = []
    for pol in pol_indices:
        good_mat = ~clean_flag[pol]
        valid_rows = np.flatnonzero(np.sum(good_mat, axis=0) >= 16)
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
        "coh_slope_deg_per_100mhz": round(coh_slope, 4),
        "coh_chan_phase_rms_deg": round(coh_rms, 4),
        "med_bl_slope_deg_per_100mhz": round(float(np.median(bl_slopes)), 4)
        if bl_slopes
        else round(coh_slope, 4),
        "med_bl_chan_phase_rms_deg": round(float(np.median(bl_rms)), 4)
        if bl_rms
        else round(coh_rms, 4),
        "median_row_chan_phase_rms_deg": round(float(np.median(row_rms_list)), 4)
        if row_rms_list
        else round(coh_rms, 4),
        "field_calibrated": True,
        "unflagged_fraction": round(unflagged_frac, 6),
        "source_ms": ms_path.name,
    }


def _score_flatness_info(
    flatness_info: dict[str, Any], truth: dict[str, Any]
) -> tuple[float, float]:
    """Return `(bandpass_flatness_score, eff_metric_for_tiebreak)`."""
    if not flatness_info.get("field_calibrated"):
        return 0.0, float("inf")
    unflg = flatness_info.get("unflagged_fraction")
    if unflg is None or float(unflg) < MIN_CALIBRATOR_UNFLAGGED_FRAC:
        return 0.0, float("inf")

    if truth["is_suite_r"]:
        c_rms = flatness_info.get("coh_chan_phase_rms_deg")
        b_rms = flatness_info.get("med_bl_chan_phase_rms_deg")
        r_rms = flatness_info.get("median_row_chan_phase_rms_deg")
        if c_rms is None or b_rms is None or r_rms is None:
            return 0.0, float("inf")
        eff_rms = max(float(c_rms), float(b_rms), float(r_rms))
        rms_max = truth["max_chan_rms_deg"]
        score = 1.0 if eff_rms < rms_max else max(0.0, 1.0 - (eff_rms - rms_max) / 12.0)
        return score, eff_rms

    c_sl = flatness_info.get("coh_slope_deg_per_100mhz")
    b_sl = flatness_info.get("med_bl_slope_deg_per_100mhz")
    if c_sl is None or b_sl is None:
        return 0.0, float("inf")
    eff_slope = max(float(c_sl), float(b_sl))
    sl_max = truth["max_slope_deg_100mhz"]
    score = 1.0 if eff_slope < sl_max else max(0.0, 1.0 - (eff_slope - sl_max) / 20.0)
    return score, eff_slope


def _measure_j1331_flatness_across_workspace(
    workspace_dir: Path,
    truth: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Evaluate `calibrators.ms` and `input.ms`, selecting the calibrated MS with best flatness."""
    candidates_ms = [workspace_dir / "calibrators.ms", workspace_dir / "input.ms"]
    evaluated: list[tuple[float, float, int, dict[str, Any]]] = []
    fallback_info = _empty_flatness_dict()

    for order_idx, cand_ms in enumerate(candidates_ms):
        if not (cand_ms / "table.dat").is_file() or has_active_table_lock(cand_ms):
            continue
        fb_fid = truth["flux_calibrator_id"] if cand_ms.name == "input.ms" else None
        fid = _resolve_field_id_by_name(cand_ms, truth["flux_calibrator"], fallback_fid=fb_fid)
        if fid is None:
            continue
        dead_id = _resolve_dead_ant_id(cand_ms, truth["dead_antenna"], truth["dead_antenna_id"])
        info = _measure_j1331_flatness(cand_ms, fid, dead_id)
        fallback_info = info
        sc, metric = _score_flatness_info(info, truth)
        if info.get("field_calibrated"):
            evaluated.append((sc, -metric, -order_idx, info))

    if not evaluated:
        return fallback_info, 0.0
    evaluated.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_sc, _neg_metric, _neg_ord, best_info = evaluated[0]
    return best_info, best_sc


def _discover_fluxscale_from_caltables(
    workspace_dir: Path,
    flux_fid: int,
    phase_fid: int,
    *,
    flux_cal_name: str = "J1331+3030",
    phase_cal_name: str = "J1822-0938",
    max_depth: int = 3,
) -> float | None:
    """Recover bootstrapped phase-cal flux from `(G_table, fluxscale_table)` pair if present."""
    skip_suffixes = (
        ".ms",
        ".image",
        ".pb",
        ".residual",
        ".model",
        ".pbcor",
        ".psf",
        ".sumwt",
        ".weight",
        ".mask",
    )
    visited_inodes: set[tuple[int, int]] = set()
    table_dirs: list[Path] = []

    def _scan_dir(cur_dir: Path, depth: int) -> None:
        if depth > max_depth or not cur_dir.is_dir():
            return
        try:
            st = os.stat(cur_dir)
            inode_key = (int(st.st_dev), int(st.st_ino))
        except OSError:
            return
        if inode_key in visited_inodes:
            return
        visited_inodes.add(inode_key)

        if cur_dir != workspace_dir and (cur_dir / "table.dat").is_file():
            table_dirs.append(cur_dir)
            return

        try:
            entries = sorted(cur_dir.iterdir(), key=lambda x: x.name)
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if name.startswith(".") or name.startswith("TempLattice"):
                continue
            low = name.lower()
            if any(low.endswith(sfx) for sfx in skip_suffixes):
                continue
            if entry.is_dir():
                _scan_dir(entry, depth + 1)

    _scan_dir(workspace_dir, 0)

    flux_upper = flux_cal_name.strip().upper()
    phase_upper = phase_cal_name.strip().upper()

    candidates: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for p in table_dirs:
        if has_active_table_lock(p):
            continue
        try:
            ct_flux_id: int | None = None
            ct_phase_id: int | None = None
            if (p / "FIELD" / "table.dat").is_file():
                with _open_ro_table(p / "FIELD") as tb_cf:
                    if int(tb_cf.nrows()) > 0 and "NAME" in tb_cf.colnames():
                        cf_names = [str(x).strip().upper() for x in tb_cf.getcol("NAME")]
                        if any(cf_names):
                            for idx, nm in enumerate(cf_names):
                                if nm == flux_upper and ct_flux_id is None:
                                    ct_flux_id = idx
                                if nm == phase_upper and ct_phase_id is None:
                                    ct_phase_id = idx
                            if ct_flux_id is None or ct_phase_id is None:
                                continue
            if ct_flux_id is None:
                ct_flux_id = flux_fid
            if ct_phase_id is None:
                ct_phase_id = phase_fid

            with _open_ro_table(p) as tb:
                cols = set(tb.colnames())
                if not {"CPARAM", "FIELD_ID", "FLAG"}.issubset(cols):
                    continue
                if int(tb.nrows()) == 0:
                    continue
                cparam = tb.getcol("CPARAM")
                if cparam.ndim != 3 or cparam.shape[1] != 1:
                    continue
                fids = tb.getcol("FIELD_ID")
                flg = tb.getcol("FLAG")
                if ct_flux_id in fids and ct_phase_id in fids:
                    candidates.append((p, cparam, fids, flg, ct_flux_id, ct_phase_id))
        except Exception:
            continue

    # Match unscaled G table and fluxscaled G table: identical nrows & identical reference-field CPARAM,
    # scaled transfer-field CPARAM where S_pcal = median((|G_pcal| / |F_pcal|)^2).
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j:
                continue
            _pi, cpi, fidi, flgi, ref_i, tr_i = candidates[i]
            _pj, cpj, fidj, flgj, ref_j, tr_j = candidates[j]
            if (
                cpi.shape != cpj.shape
                or ref_i != ref_j
                or tr_i != tr_j
                or not np.array_equal(fidi, fidj)
            ):
                continue
            ref_m = (fidi == ref_i)[np.newaxis, np.newaxis, :] & (~flgi) & (~flgj)
            tr_m = (fidi == tr_i)[np.newaxis, np.newaxis, :] & (~flgi) & (~flgj)
            if not np.any(ref_m) or not np.any(tr_m):
                continue
            ref_diff = float(np.median(np.abs(np.abs(cpi[ref_m]) - np.abs(cpj[ref_m]))))
            if ref_diff > 1e-4:
                continue
            denom = np.abs(cpj[tr_m])
            valid_d = denom > 1e-12
            if not np.any(valid_d):
                continue
            ratios_sq = (np.abs(cpi[tr_m][valid_d]) / denom[valid_d]) ** 2
            s_est = float(np.median(ratios_sq))
            if math.isfinite(s_est) and abs(s_est - 1.0) > 0.05 and 0.05 <= s_est <= 100.0:
                return s_est
    return None


def _measure_pcal_flux_single_ms(
    ms_path: Path,
    phase_fid: int,
    dead_ant_id: int,
) -> tuple[float | None, float | None, bool]:
    """Return `(corr_amp_jy, unflagged_par_frac, field_calibrated)` for `phase_fid` in `ms_path`."""
    with _open_ro_table(ms_path) as tb:
        cols = set(tb.colnames())
        has_corr = "CORRECTED_DATA" in cols
        has_raw = "DATA" in cols
        sub = tb.query(
            f"FIELD_ID == {phase_fid} && ANTENNA1 != ANTENNA2 "
            f"&& ANTENNA1 != {dead_ant_id} && ANTENNA2 != {dead_ant_id}"
        )
        try:
            if int(sub.nrows()) == 0:
                return None, None, False
            flg = sub.getcol("FLAG")
            flgr = sub.getcol("FLAG_ROW")
            corr = sub.getcol("CORRECTED_DATA") if has_corr else None
            raw = sub.getcol("DATA") if has_raw else None
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()

    eff_flg = flg | flgr[np.newaxis, np.newaxis, :]
    npol = flg.shape[0]
    p_idx = (0, 3) if npol >= 4 else ((0, npol - 1) if npol >= 2 else (0,))
    unflg_frac = float(1.0 - np.mean(eff_flg[list(p_idx), :, :]))

    if corr is None:
        return None, unflg_frac, False

    par_corr = corr[list(p_idx), :, :]
    good_par = (~eff_flg[list(p_idx), :, :]) & np.isfinite(par_corr)
    if not np.any(good_par):
        return None, unflg_frac, False

    max_abs_c = float(np.max(np.abs(par_corr[good_par])))
    if raw is not None:
        par_raw = raw[list(p_idx), :, :]
        max_diff = float(np.max(np.abs(par_corr[good_par] - par_raw[good_par])))
        field_calibrated = max_abs_c > 1e-9 and max_diff > 1e-6
    else:
        field_calibrated = max_abs_c > 1e-9

    if not field_calibrated or unflg_frac < MIN_CALIBRATOR_UNFLAGGED_FRAC:
        return None, unflg_frac, field_calibrated

    amps: list[float] = []
    for p in p_idx:
        good = ~eff_flg[p]
        if np.any(good):
            v = corr[p][good]
            fin = np.isfinite(v.real) & np.isfinite(v.imag)
            if np.any(fin):
                amps.append(float(np.abs(np.mean(v[fin]))))
    corr_amp_jy = float(np.mean(amps)) if amps else None
    return corr_amp_jy, unflg_frac, True


def _measure_pcal_flux(
    workspace_dir: Path,
    ms_path: Path,
    flux_fid: int,
    phase_fid: int,
    dead_ant_id: int,
    *,
    flux_cal_name: str = "J1331+3030",
    phase_cal_name: str = "J1822-0938",
    expected_pcal_flux_jy: float | None = None,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Return `(pcal_corrected_amp_jy, caltable_fluxscale_jy, max_pcal_unflagged_frac, source_ms)`."""
    candidates_ms: list[Path] = []
    cal_ms = workspace_dir / "calibrators.ms"
    if (cal_ms / "table.dat").is_file() and not has_active_table_lock(cal_ms):
        candidates_ms.append(cal_ms)
    if (
        (ms_path / "table.dat").is_file()
        and not has_active_table_lock(ms_path)
        and ms_path not in candidates_ms
    ):
        candidates_ms.append(ms_path)

    corr_candidates: list[tuple[float, int, float, str]] = []
    unflg_fractions: list[float] = []

    for order_idx, cand_ms in enumerate(candidates_ms):
        fb_pfid = phase_fid if cand_ms == ms_path else None
        pfid = _resolve_field_id_by_name(cand_ms, phase_cal_name, fallback_fid=fb_pfid)
        if pfid is None:
            continue
        d_aid = (
            _resolve_dead_ant_id(cand_ms, "", dead_ant_id) if cand_ms == ms_path else dead_ant_id
        )
        if (cand_ms / "ANTENNA" / "table.dat").is_file() and (
            ms_path / "ANTENNA" / "table.dat"
        ).is_file():
            with _open_ro_table(ms_path / "ANTENNA") as tb_ma:
                m_ants = [str(x).strip().lower() for x in tb_ma.getcol("NAME")]
            if 0 <= dead_ant_id < len(m_ants):
                d_aid = _resolve_dead_ant_id(cand_ms, m_ants[dead_ant_id], dead_ant_id)

        c_amp, u_frac, is_cal = _measure_pcal_flux_single_ms(cand_ms, pfid, d_aid)
        if u_frac is not None:
            unflg_fractions.append(u_frac)
        if is_cal and c_amp is not None and math.isfinite(c_amp):
            err_key = (
                abs(c_amp - expected_pcal_flux_jy) if expected_pcal_flux_jy is not None else 0.0
            )
            corr_candidates.append((err_key, order_idx, c_amp, cand_ms.name))

    max_unflg_frac = max(unflg_fractions) if unflg_fractions else None
    if max_unflg_frac is not None and max_unflg_frac < MIN_CALIBRATOR_UNFLAGGED_FRAC:
        return None, None, max_unflg_frac, None

    corr_amp_jy: float | None = None
    source_ms: str | None = None
    if corr_candidates:
        corr_candidates.sort(key=lambda item: (item[0], item[1]))
        corr_amp_jy = corr_candidates[0][2]
        source_ms = corr_candidates[0][3]

    caltable_flux_jy = _discover_fluxscale_from_caltables(
        workspace_dir,
        flux_fid,
        phase_fid,
        flux_cal_name=flux_cal_name,
        phase_cal_name=phase_cal_name,
    )
    return corr_amp_jy, caltable_flux_jy, max_unflg_frac, source_ms


def _check_antenna_gain_consistency(
    tb: Any,
    tgt_denom_mask: np.ndarray,
    flgr: np.ndarray,
    time_col: np.ndarray,
) -> dict[str, Any]:
    """Verify `CORRECTED_DATA` has real variance and satisfies antenna-based rank-1 gain factorization."""
    cand_rows = np.flatnonzero(tgt_denom_mask & (~flgr))
    if cand_rows.size == 0:
        return {
            "corrected_distinct": False,
            "gain_consistent": False,
            "target_corrected_std_jy": None,
            "rank1_logamp_rms_median": None,
        }

    uniq_t = np.unique(time_col[cand_rows])
    chosen_t = uniq_t[np.linspace(0, uniq_t.size - 1, 8, dtype=int)] if uniq_t.size > 8 else uniq_t

    sel_rows = cand_rows[np.isin(time_col[cand_rows], chosen_t)]
    if sel_rows.size > 3000:
        sel_rows = sel_rows[:3000]

    sub_s = tb.selectrows(sel_rows.tolist())
    try:
        c_samp = sub_s.getcol("CORRECTED_DATA")
        d_samp = sub_s.getcol("DATA")
        f_samp = sub_s.getcol("FLAG")
        fr_samp = sub_s.getcol("FLAG_ROW")
        a1_s = sub_s.getcol("ANTENNA1")
        a2_s = sub_s.getcol("ANTENNA2")
        t_s = sub_s.getcol("TIME")
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
        return {
            "corrected_distinct": False,
            "gain_consistent": False,
            "target_corrected_std_jy": None,
            "rank1_logamp_rms_median": None,
        }

    c_vals = par_c[par_good]
    d_vals = par_d[par_good]
    max_abs_c = float(np.max(np.abs(c_vals)))
    max_diff = float(np.max(np.abs(c_vals - d_vals)))
    corrected_distinct = max_abs_c > 1e-9 and max_diff > 1e-7
    target_corr_std = float(np.std(c_vals))

    # Sample clean interior channels across the band
    ch_probes = sorted(
        {
            max(0, min(nchan - 1, ch))
            for ch in (nchan // 5, (2 * nchan) // 5, (3 * nchan) // 5, (4 * nchan) // 5)
        }
    )
    slice_resids: list[float] = []
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

    med_rank1_rms = float(np.median(slice_resids)) if slice_resids else None
    gain_consistent = (
        corrected_distinct
        and target_corr_std > 1e-5
        and med_rank1_rms is not None
        and math.isfinite(med_rank1_rms)
        and med_rank1_rms <= MAX_RANK1_LOGAMP_RMS
    )
    return {
        "corrected_distinct": corrected_distinct,
        "gain_consistent": gain_consistent,
        "target_corrected_std_jy": round(target_corr_std, 8),
        "rank1_logamp_rms_median": round(med_rank1_rms, 8) if med_rank1_rms is not None else None,
    }


def _predict_synth_target_vis_fallback(
    ms_path: Path,
    sky_model: dict[str, Any],
    target_fids: list[int],
    dead_ant_id: int,
    rfi_channels: list[int],
    scratch_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict noiseless target `(true_rr, true_ll)` in a scratch copy when `.npz` is absent."""
    from casatools import componentlist, measures, simulator

    tmp_dir = Path(tempfile.mkdtemp(prefix="synth_pred_", dir=str(scratch_root)))
    prev_cwd = Path.cwd()
    try:
        os.chdir(tmp_dir)
        tmp_ms = tmp_dir / "pred.ms"
        shutil.copytree(ms_path, tmp_ms)

        ext = sky_model["extended_gaussian"]
        pt = sky_model["point_source"]
        cl_path = tmp_dir / "target.cl"

        me = measures()
        try:
            cl = componentlist()
            try:
                cl.done()
                cl.addcomponent(
                    dir=me.direction("J2000", str(ext["ra"]), str(ext["dec"])),
                    flux=float(ext["flux_jy"]),
                    fluxunit="Jy",
                    freq="4.600GHz",
                    shape="Gaussian",
                    majoraxis=f"{float(ext['major_arcsec'])}arcsec",
                    minoraxis=f"{float(ext['minor_arcsec'])}arcsec",
                    positionangle=f"{float(ext['pa_deg'])}deg",
                    spectrumtype="constant",
                )
                cl.addcomponent(
                    dir=me.direction("J2000", str(pt["ra"]), str(pt["dec"])),
                    flux=float(pt["flux_jy"]),
                    fluxunit="Jy",
                    freq="4.600GHz",
                    shape="point",
                    spectrumtype="constant",
                )
                cl.rename(str(cl_path))
            finally:
                with contextlib.suppress(Exception):
                    cl.close()
                with contextlib.suppress(Exception):
                    cl.done()
        finally:
            with contextlib.suppress(Exception):
                me.done()

        sm = simulator()
        sm_opened = False
        try:
            sm.openfromms(str(tmp_ms))
            sm_opened = True
            sm.setvp(dovp=True, usedefaultvp=True, dosquint=False)
            sm.predict(complist=str(cl_path), incremental=False)
        finally:
            if sm_opened:
                with contextlib.suppress(Exception):
                    sm.close()
            with contextlib.suppress(Exception):
                sm.done()

        with _open_ro_table(tmp_ms) as tb_p:
            pred_data = tb_p.getcol("DATA")
            fids = tb_p.getcol("FIELD_ID")
            a1 = tb_p.getcol("ANTENNA1")
            a2 = tb_p.getcol("ANTENNA2")

        tgt_mask = np.isin(fids, target_fids)
        tgt_row_idx = np.flatnonzero(tgt_mask).astype(np.int32)
        dead_sub = (a1[tgt_mask] == dead_ant_id) | (a2[tgt_mask] == dead_ant_id)
        clean_tgt_mask_sub = (~dead_sub) & (a1[tgt_mask] != a2[tgt_mask])
        nchan = pred_data.shape[1]
        clean_chan_mask = np.ones(nchan, dtype=bool)
        for ch in rfi_channels:
            if 0 <= int(ch) < nchan:
                clean_chan_mask[int(ch)] = False
        true_rr = pred_data[0, :, :][:, tgt_mask].astype(np.complex128).copy()
        true_ll = pred_data[3, :, :][:, tgt_mask].astype(np.complex128).copy()
        return tgt_row_idx, clean_tgt_mask_sub, clean_chan_mask, true_rr, true_ll
    finally:
        with contextlib.suppress(Exception):
            os.chdir(prev_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _measure_synth_coherent_fidelity(
    ms_path: Path,
    truth: dict[str, Any],
    masks_npz_path: Path | None,
    scratch_root: Path,
) -> dict[str, float | None]:
    tvis_candidates: list[Path] = []
    tvis_name = str(truth.get("true_target_vis_file") or "true_target_vis.npz")
    if masks_npz_path is not None:
        mp = Path(masks_npz_path)
        if mp.is_file():
            tvis_candidates.append(mp)
        if mp.parent.is_dir():
            tvis_candidates.append(mp.parent / tvis_name)
            tvis_candidates.append(mp.parent / "true_target_vis.npz")

    loaded_tvis = False
    for cand in tvis_candidates:
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
                    tgt_row_idx = z["target_row_indices"].astype(int).copy()
                    clean_tgt_mask_sub = z["clean_target_row_mask"].astype(bool).copy()
                    clean_chan_mask = z["clean_chan_mask"].astype(bool).copy()
                    true_rr = z["true_rr"].astype(np.complex128).copy()
                    true_ll = z["true_ll"].astype(np.complex128).copy()
                    loaded_tvis = True
                    break
        except Exception:
            continue

    if not loaded_tvis:
        sky_model = truth.get("sky_model")
        if not isinstance(sky_model, dict):
            return {
                "coherent_amp_ratio": None,
                "coherent_amp_rel_err": None,
                "coherent_phase_deg": None,
                "coherent_pearson_r": None,
                "clean_target_unflagged_fraction": None,
            }
        csum = (
            truth.get("corruptions_summary")
            if isinstance(truth.get("corruptions_summary"), dict)
            else {}
        )
        rfi_ch = list(csum.get("rfi_channels", [29, 30, 31]))
        tgt_row_idx, clean_tgt_mask_sub, clean_chan_mask, true_rr, true_ll = (
            _predict_synth_target_vis_fallback(
                ms_path=ms_path,
                sky_model=sky_model,
                target_fids=truth["target_field_ids"],
                dead_ant_id=truth["dead_antenna_id"],
                rfi_channels=rfi_ch,
                scratch_root=scratch_root,
            )
        )

    with _open_ro_table(ms_path) as tb:
        corr = tb.getcol("CORRECTED_DATA")
        flag = tb.getcol("FLAG")
        flag_row = tb.getcol("FLAG_ROW")
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")

    eff_flag = flag | flag_row[np.newaxis, np.newaxis, :]
    cross_sub = a1[tgt_row_idx] != a2[tgt_row_idx]

    corr_rr = corr[0, :, :][:, tgt_row_idx]
    corr_ll = corr[3, :, :][:, tgt_row_idx]
    flg_rr = eff_flag[0, :, :][:, tgt_row_idx]
    flg_ll = eff_flag[3, :, :][:, tgt_row_idx]

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
    """Child-process entrypoint executing `mstransform` + `tclean` safely outside parent process."""
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    import casatasks
    import casatools

    req_path = Path(payload_json_path)
    req = json.loads(req_path.read_text(encoding="utf-8"))
    ms_path = Path(req["ms_path"])
    recipe: dict[str, Any] = req["recipe"]
    out_path = Path(req["out_json"])
    min_off_rms_jy = float(req.get("min_offsource_rms_jy", 0.08e-3))
    work_dir = req_path.parent
    os.chdir(work_dir)

    try:
        sub_ms = work_dir / "verify_target.ms"
        casatasks.mstransform(
            vis=str(ms_path),
            outputvis=str(sub_ms),
            field=str(recipe.get("field", "2~8")),
            spw=str(recipe.get("spw", "0:4~59")),
            datacolumn="corrected",
        )

        img_prefix = work_dir / "verifier_mosaic"
        imsize_raw = recipe.get("imsize", [640, 640])
        imsize = (
            [int(imsize_raw[0]), int(imsize_raw[1])]
            if isinstance(imsize_raw, list)
            else [int(imsize_raw), int(imsize_raw)]
        )
        cell_arcsec = float(recipe.get("cell_arcsec", 2.5))
        cell_str = f"{cell_arcsec}arcsec"
        usemask = str(recipe.get("usemask", "auto-multithresh"))
        do_pbcor = bool(recipe.get("pbcor", True))

        tclean_kwargs: dict[str, Any] = {
            "vis": str(sub_ms),
            "imagename": str(img_prefix),
            "spw": "0",
            "datacolumn": "data",
            "specmode": str(recipe.get("specmode", "mfs")),
            "gridder": str(recipe.get("gridder", "mosaic")),
            "deconvolver": str(recipe.get("deconvolver", "multiscale")),
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
        if "phasecenter" in recipe:
            tclean_kwargs["phasecenter"] = str(recipe["phasecenter"])
        if tclean_kwargs["deconvolver"] == "multiscale":
            tclean_kwargs["scales"] = [int(s) for s in recipe.get("scales", [0, 5, 15, 30])]
        if "smallscalebias" in recipe:
            tclean_kwargs["smallscalebias"] = float(recipe["smallscalebias"])
        if "cycleniter" in recipe:
            tclean_kwargs["cycleniter"] = int(recipe["cycleniter"])
        if "nmajor" in recipe:
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
                if k in recipe:
                    tclean_kwargs[k] = recipe[k]

        casatasks.tclean(**tclean_kwargs)

        def _read_plane(suffix: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
            ia = casatools.image()
            qa = casatools.quanta() if suffix == "image" else None
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
                if suffix == "image" and qa is not None:
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
                    wcs_info = {
                        "refpix": refpix,
                        "refval": [rv0, rv1],
                        "inc": [ic0, ic1],
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

        img_arr, img_msk, wcs = _read_plane("image")
        pb_arr, pb_msk, _ = _read_plane("pb")
        res_arr, res_msk, _ = _read_plane("residual")
        mod_arr, _mod_msk, _ = _read_plane("model")
        pbcor_arr: np.ndarray | None = None
        pbcor_msk: np.ndarray | None = None
        if do_pbcor and (Path(f"{img_prefix}.image.pbcor") / "table.dat").is_file():
            pbcor_arr, pbcor_msk, _ = _read_plane("image.pbcor")

        pblim = float(recipe.get("pblimit", 0.2))
        pb_gate = np.isfinite(pb_arr) & pb_msk & (pb_arr >= pblim)
        pb_valid = pb_gate & np.isfinite(res_arr) & res_msk & np.isfinite(img_arr) & img_msk
        if int(np.sum(pb_valid)) < 50:
            out_path.write_text(
                json.dumps(
                    {
                        "verifier_tclean_peak_jy": None,
                        "verifier_tclean_offsource_rms_jy": None,
                        "verifier_tclean_dynamic_range": None,
                        "verifier_tclean_physical_valid": False,
                        "verifier_tclean_error": "Insufficient valid pixels inside primary beam",
                    }
                ),
                encoding="utf-8",
            )
            return

        # Physical PB taper check (MF-4)
        pb_in = pb_arr[pb_gate]
        pb_max = float(np.max(pb_in))
        pb_min_20 = float(np.min(pb_in))
        pb_fin_all = pb_arr[np.isfinite(pb_arr)]
        pb_min_full = float(np.min(pb_fin_all)) if pb_fin_all.size > 0 else 1.0
        pb_std = float(np.std(pb_in))
        pb_taper_ok = (
            (0.92 <= pb_max <= 1.05)
            and (pb_min_20 <= 0.65 or pb_min_full < 0.85)
            and (pb_std >= 0.05)
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

        min_rad_arcsec = float(recipe.get("off_source_min_radius_arcsec", 220.0))
        nx, ny = img_arr.shape
        if wcs is not None and "phasecenter" in recipe:
            ra_c, dec_c = _parse_sky_coord_rad(str(recipe["phasecenter"]))
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
            cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
            ix, iy = np.meshgrid(
                np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij"
            )
            radius_arcsec = np.hypot(ix - cx, iy - cy) * cell_arcsec

        off_mask = pb_valid & (radius_arcsec >= min_rad_arcsec) & (np.abs(mod_arr) == 0.0)
        if int(np.sum(off_mask)) < 500:
            off_mask = pb_valid & (radius_arcsec >= min_rad_arcsec)
        if int(np.sum(off_mask)) < 200:
            off_mask = pb_valid & (np.abs(mod_arr) == 0.0)
        if int(np.sum(off_mask)) < 50:
            off_mask = pb_valid

        off_vals = res_arr[off_mask]
        off_rms_jy = float(np.sqrt(np.mean(off_vals**2)))
        off_med = float(np.median(off_vals))
        off_mad_std_jy = float(1.4826 * np.median(np.abs(off_vals - off_med)))

        rms_phys_ok = (
            math.isfinite(off_rms_jy)
            and math.isfinite(off_mad_std_jy)
            and off_rms_jy >= min_off_rms_jy
            and off_mad_std_jy >= 0.50 * off_rms_jy
        )
        phys_ok = pb_taper_ok and pbcor_ok and rms_phys_ok and math.isfinite(peak_jy)

        if not phys_ok or off_rms_jy <= 0.0:
            reasons: list[str] = []
            if not pb_taper_ok:
                reasons.append(f"unphysical .pb taper (std={pb_std:.4f})")
            if not pbcor_ok:
                reasons.append("inconsistent .image.pbcor * .pb vs .image")
            if not rms_phys_ok:
                reasons.append(
                    f"unphysical residual (rms={off_rms_jy:.6e}, min={min_off_rms_jy:.6e}, "
                    f"mad_std={off_mad_std_jy:.6e})"
                )
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
                        "verifier_tclean_dynamic_range": None,
                        "verifier_tclean_physical_valid": False,
                        "verifier_tclean_error": "; ".join(reasons)
                        or "Verifier tclean image failed physical checks",
                    }
                ),
                encoding="utf-8",
            )
            return

        dr = peak_jy / off_rms_jy
        out_path.write_text(
            json.dumps(
                {
                    "verifier_tclean_peak_jy": round(peak_jy, 6),
                    "verifier_tclean_offsource_rms_jy": round(off_rms_jy, 8),
                    "verifier_tclean_offsource_mad_std_jy": round(off_mad_std_jy, 8),
                    "verifier_tclean_dynamic_range": round(dr, 3),
                    "verifier_tclean_physical_valid": True,
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
    min_offsource_rms_jy: float = 0.08e-3,
) -> dict[str, Any]:
    """Run fixed-recipe multiscale mosaic tclean in a child subprocess (survives CASA SIGABRT)."""
    resolved_scratch = resolve_scratch_root(scratch_root)
    tclean_dir = Path(tempfile.mkdtemp(prefix="verifier_tclean_", dir=str(resolved_scratch)))
    try:
        mpl_dir = tclean_dir / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        req_json = tclean_dir / "tclean_req.json"
        out_json = tclean_dir / "tclean_out.json"
        eff_min_rms = float(recipe.get("min_offsource_rms_jy", min_offsource_rms_jy))

        req_json.write_text(
            json.dumps(
                {
                    "ms_path": str(Path(ms_path).resolve()),
                    "recipe": recipe,
                    "out_json": str(out_json),
                    "min_offsource_rms_jy": eff_min_rms,
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
        src_root = str(Path(__file__).resolve().parents[2])
        existing_py = child_env.get("PYTHONPATH", "").strip()
        child_env["PYTHONPATH"] = f"{src_root}:{existing_py}" if existing_py else src_root

        timeout_sec = float(recipe.get("timeout_sec", 900.0))
        cmd = [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE'); "
                "from radio_bench.grading.cal_metrics import _verifier_tclean_child_entry; "
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
                "verifier_tclean_dynamic_range": None,
                "verifier_tclean_physical_valid": False,
                "verifier_tclean_returncode": -9,
                "verifier_tclean_error": f"Verifier tclean subprocess timed out after {timeout_sec}s",
            }

        if proc.returncode != 0 or not out_json.is_file():
            stderr_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            return {
                "verifier_tclean_peak_jy": None,
                "verifier_tclean_offsource_rms_jy": None,
                "verifier_tclean_dynamic_range": None,
                "verifier_tclean_physical_valid": False,
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
            "verifier_tclean_dynamic_range": None,
            "verifier_tclean_physical_valid": False,
            "verifier_tclean_error": f"Verifier tclean runner error: {type(exc).__name__}: {exc}",
        }
    finally:
        shutil.rmtree(tclean_dir, ignore_errors=True)


def grade_calibrate(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path | None = None,
    task_id: str,
    scratch_root: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade `/workspace/input.ms` (+ optional `/workspace/calibrators.ms` & caltables) after calibration."""
    from casatasks import clearstat

    with contextlib.suppress(Exception):
        clearstat()

    ws = Path(workspace_dir)
    input_ms = ws / "input.ms"
    cal_ms = ws / "calibrators.ms"
    truth = _extract_cal_truth(ground_truth)
    weights = truth["weights"]
    pass_min = truth["pass_reward_min"]
    resolved_scratch = resolve_scratch_root(scratch_root)

    errors: list[str] = []
    table_locked = False

    if not (input_ms / "table.dat").is_file():
        errors.append("Missing /workspace/input.ms")
        reward_dict: dict[str, float | int] = {
            "reward": 0.0,
            "passed": 0,
            "retention_and_dead_ant_score": 0.0,
            "bandpass_flatness_score": 0.0,
            "fluxscale_score": 0.0,
            "target_fidelity_score": 0.0,
        }
        metrics_dict: dict[str, Any] = {
            "task_id": task_id,
            "stage": "calibrate",
            "pass_reward_min": pass_min,
            "table_locked": False,
            "has_corrected_data": False,
            "truth": {
                k: v for k, v in truth.items() if k not in ("sky_model", "corruptions_summary")
            },
            "measured": {},
            "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
            "weights": weights,
            "errors": errors,
        }
        return reward_dict, metrics_dict

    if has_active_table_lock(input_ms):
        table_locked = True
        errors.append(f"Active casacore write lock on {input_ms}")
    if (cal_ms / "table.dat").is_file() and has_active_table_lock(cal_ms):
        table_locked = True
        errors.append(f"Active casacore write lock on {cal_ms}")

    has_corrected = False
    corrected_distinct = False
    gain_consistent = False
    target_corr_std: float | None = None
    rank1_logamp_rms: float | None = None
    dead_ant_flagged_frac: float | None = None
    target_row_retention: float | None = None
    unflagged_target_rows_count = 0

    retention_and_dead_ant_score = 0.0
    bandpass_flatness_score = 0.0
    fluxscale_score = 0.0
    target_fidelity_score = 0.0

    dead_ant_id = truth["dead_antenna_id"]

    if not table_locked:
        try:
            dead_ant_id = _resolve_dead_ant_id(
                input_ms, truth["dead_antenna"], truth["dead_antenna_id"]
            )

            with _open_ro_table(input_ms) as tb:
                if tb.ismultiused(checksubtables=True):
                    table_locked = True
                    errors.append("input.ms reported ismultiused=True")
                else:
                    cols = set(tb.colnames())
                    has_corrected = "CORRECTED_DATA" in cols
                    fids = tb.getcol("FIELD_ID")
                    a1 = tb.getcol("ANTENNA1")
                    a2 = tb.getcol("ANTENNA2")
                    time_col = tb.getcol("TIME")
                    flg = tb.getcol("FLAG")
                    flgr = tb.getcol("FLAG_ROW")

                    cross = a1 != a2
                    dead_rows = ((a1 == dead_ant_id) | (a2 == dead_ant_id)) & cross
                    if np.any(dead_rows):
                        dead_eff = flg[:, :, dead_rows] | flgr[np.newaxis, np.newaxis, dead_rows]
                        dead_ant_flagged_frac = float(np.mean(dead_eff))
                    else:
                        dead_ant_flagged_frac = 1.0

                    tgt_base = np.isin(fids, truth["target_field_ids"]) & cross
                    if truth["excludes_dead_in_retention"]:
                        tgt_denom_mask = tgt_base & (a1 != dead_ant_id) & (a2 != dead_ant_id)
                    else:
                        tgt_denom_mask = tgt_base

                    if np.any(tgt_denom_mask):
                        p_idx_ret = [0, flg.shape[0] - 1] if flg.shape[0] >= 2 else [0]
                        eff_par_flg = (
                            flg[p_idx_ret, :, :][:, :, tgt_denom_mask]
                            | flgr[np.newaxis, np.newaxis, tgt_denom_mask]
                        )
                        target_row_retention = float(1.0 - np.mean(eff_par_flg))
                        tgt_par_row_all_flg = np.all(eff_par_flg, axis=(0, 1))
                        unflagged_target_rows_count = int(np.sum(~tgt_par_row_all_flg))
                    else:
                        target_row_retention = 0.0

                    if has_corrected and unflagged_target_rows_count > 0:
                        gain_check = _check_antenna_gain_consistency(
                            tb, tgt_denom_mask, flgr, time_col
                        )
                        corrected_distinct = bool(gain_check["corrected_distinct"])
                        gain_consistent = bool(gain_check["gain_consistent"])
                        target_corr_std = gain_check["target_corrected_std_jy"]
                        rank1_logamp_rms = gain_check["rank1_logamp_rms_median"]
                        if corrected_distinct and not gain_consistent:
                            errors.append(
                                "Target CORRECTED_DATA fails antenna-based gain consistency "
                                f"(std={target_corr_std}, rank1_logamp_rms={rank1_logamp_rms})"
                            )
        except Exception as exc:
            errors.append(f"Failed inspecting input.ms retention/dead-antenna state: {exc}")

    # Score Subscore 1 (0.20): CORRECTED_DATA + dead antenna + parallel-hand retention + gain consistency
    if has_corrected and corrected_distinct and gain_consistent and not table_locked:
        dead_min = truth["dead_antenna_flagged_min"]
        if dead_ant_flagged_frac is None:
            dead_sub = 0.0
        elif dead_ant_flagged_frac >= dead_min:
            dead_sub = 1.0
        else:
            dead_sub = max(0.0, (dead_ant_flagged_frac - 0.50) / max(dead_min - 0.50, 1e-6))

        ret_min = truth["retention_min"]
        ret_max = truth["retention_max"]
        if target_row_retention is None or target_row_retention <= 0.0:
            ret_sub = 0.0
        elif ret_max is None:
            if target_row_retention >= ret_min:
                ret_sub = 1.0
            else:
                ret_sub = max(0.0, (target_row_retention - 0.40) / max(ret_min - 0.40, 1e-6))
        else:
            if ret_min <= target_row_retention <= ret_max:
                ret_sub = 1.0
            elif target_row_retention < ret_min:
                ret_sub = max(0.0, (target_row_retention - 0.40) / max(ret_min - 0.40, 1e-6))
            else:
                ret_sub = max(0.0, 1.0 - (target_row_retention - ret_max) / 0.08)

        retention_and_dead_ant_score = (
            0.0 if (ret_sub == 0.0 or dead_sub == 0.0) else 0.5 * dead_sub + 0.5 * ret_sub
        )
    elif not has_corrected:
        errors.append("CORRECTED_DATA column missing from /workspace/input.ms")
    elif not corrected_distinct:
        errors.append("CORRECTED_DATA is empty, non-finite, or identical to uncalibrated DATA")

    # Score Subscore 2 (0.25): J1331+3030 bandpass + delay flatness (checks calibrators.ms & input.ms)
    flatness_info: dict[str, Any] = _empty_flatness_dict()
    if has_corrected and corrected_distinct and not table_locked:
        try:
            flatness_info, bandpass_flatness_score = _measure_j1331_flatness_across_workspace(
                ws, truth
            )
            j_unflg = flatness_info.get("unflagged_fraction")
            if j_unflg is not None and float(j_unflg) < MIN_CALIBRATOR_UNFLAGGED_FRAC:
                errors.append(
                    f"J1331+3030 parallel-hand unflagged fraction {j_unflg:.4f} < {MIN_CALIBRATOR_UNFLAGGED_FRAC}"
                )
        except Exception as exc:
            errors.append(f"Failed measuring J1331 flatness: {exc}")

    # Score Subscore 3 (0.25): Secondary phase calibrator fluxscale accuracy
    pcal_corr_amp_jy: float | None = None
    pcal_caltable_flux_jy: float | None = None
    pcal_best_flux_jy: float | None = None
    pcal_rel_err: float | None = None
    pcal_unflg_frac: float | None = None
    pcal_source_ms: str | None = None

    if has_corrected and corrected_distinct and not table_locked:
        try:
            (
                pcal_corr_amp_jy,
                pcal_caltable_flux_jy,
                pcal_unflg_frac,
                pcal_source_ms,
            ) = _measure_pcal_flux(
                ws,
                input_ms,
                truth["flux_calibrator_id"],
                truth["phase_calibrator_id"],
                dead_ant_id,
                flux_cal_name=truth["flux_calibrator"],
                phase_cal_name=truth["phase_calibrator"],
                expected_pcal_flux_jy=truth["expected_pcal_flux_jy"],
            )
            if pcal_unflg_frac is not None and pcal_unflg_frac < MIN_CALIBRATOR_UNFLAGGED_FRAC:
                errors.append(
                    f"{truth['phase_calibrator']} parallel-hand unflagged fraction "
                    f"{pcal_unflg_frac:.4f} < {MIN_CALIBRATOR_UNFLAGGED_FRAC}"
                )
            else:
                exp_flux = truth["expected_pcal_flux_jy"]
                cand_errs: list[tuple[float, float]] = []
                if pcal_corr_amp_jy is not None and math.isfinite(pcal_corr_amp_jy):
                    cand_errs.append(
                        (abs(pcal_corr_amp_jy - exp_flux) / exp_flux, pcal_corr_amp_jy)
                    )
                if pcal_caltable_flux_jy is not None and math.isfinite(pcal_caltable_flux_jy):
                    cand_errs.append(
                        (abs(pcal_caltable_flux_jy - exp_flux) / exp_flux, pcal_caltable_flux_jy)
                    )
                if cand_errs:
                    cand_errs.sort(key=lambda pair: pair[0])
                    pcal_rel_err, pcal_best_flux_jy = cand_errs[0]
                    tol = truth["pcal_flux_rel_tol"]
                    if pcal_rel_err <= tol:
                        fluxscale_score = 1.0
                    else:
                        fluxscale_score = max(0.0, 1.0 - (pcal_rel_err - tol) / (2.0 * tol))
        except Exception as exc:
            errors.append(f"Failed measuring phase calibrator fluxscale: {exc}")

    # Score Subscore 4 (0.30): Target fidelity (coherent visibility on S; verifier tclean DR on R)
    fidelity_measured: dict[str, Any] = {}
    if (
        has_corrected
        and corrected_distinct
        and unflagged_target_rows_count > 0
        and not table_locked
    ):
        recipe = truth.get("verifier_tclean_recipe")
        if isinstance(recipe, dict):
            # Run verifier tclean if gain_consistent OR zero-variance (fast crash-isolation check)
            if gain_consistent or (target_corr_std is not None and target_corr_std <= 1e-5):
                try:
                    tclean_kw: dict[str, Any] = {}
                    if "min_verifier_tclean_rms_jy" in truth:
                        recipe_with_min = {
                            "min_offsource_rms_jy": truth["min_verifier_tclean_rms_jy"],
                            **recipe,
                        }
                    else:
                        recipe_with_min = recipe
                    fidelity_measured = _run_verifier_tclean_recipe(
                        input_ms, recipe_with_min, resolved_scratch, **tclean_kw
                    )
                    if fidelity_measured.get("verifier_tclean_error"):
                        errors.append(str(fidelity_measured["verifier_tclean_error"]))
                    dr = fidelity_measured.get("verifier_tclean_dynamic_range")
                    dr_min = truth["verifier_tclean_dr_min"]
                    if gain_consistent and dr is not None and math.isfinite(float(dr)):
                        dr_f = float(dr)
                        if dr_f >= dr_min:
                            target_fidelity_score = 1.0
                        else:
                            target_fidelity_score = max(
                                0.0, (dr_f - 0.4 * dr_min) / max(0.6 * dr_min, 1.0)
                            )
                except Exception as exc:
                    errors.append(f"Failed running verifier tclean recipe: {exc}")
        elif not truth["is_suite_r"]:
            try:
                fidelity_measured = _measure_synth_coherent_fidelity(
                    input_ms, truth, masks_npz_path, resolved_scratch
                )
                amp_err = fidelity_measured.get("coherent_amp_rel_err")
                phs_deg = fidelity_measured.get("coherent_phase_deg")
                pearson_r = fidelity_measured.get("coherent_pearson_r")
                clean_u_frac = fidelity_measured.get("clean_target_unflagged_fraction")
                if (
                    clean_u_frac is not None
                    and float(clean_u_frac) < MIN_CLEAN_TARGET_UNFLAGGED_FRAC
                ):
                    errors.append(
                        f"Clean target parallel-hand unflagged fraction {clean_u_frac:.4f} "
                        f"< {MIN_CLEAN_TARGET_UNFLAGGED_FRAC}"
                    )
                    target_fidelity_score = 0.0
                elif pearson_r is None or float(pearson_r) < truth["min_pearson_r"]:
                    errors.append(
                        f"Target complex Pearson correlation {pearson_r} < {truth['min_pearson_r']}"
                    )
                    target_fidelity_score = 0.0
                elif not gain_consistent:
                    target_fidelity_score = 0.0
                elif amp_err is not None and phs_deg is not None:
                    phs_abs = abs(float(phs_deg))
                    a_tol = truth["max_coh_amp_err"]
                    p_tol = truth["max_coh_phs_deg"]
                    a_sub = (
                        1.0
                        if float(amp_err) <= a_tol
                        else max(0.0, 1.0 - (float(amp_err) - a_tol) / 0.22)
                    )
                    p_sub = 1.0 if phs_abs <= p_tol else max(0.0, 1.0 - (phs_abs - p_tol) / 35.0)
                    target_fidelity_score = (
                        0.0 if (a_sub == 0.0 or p_sub == 0.0) else 0.5 * a_sub + 0.5 * p_sub
                    )
            except Exception as exc:
                errors.append(f"Failed measuring synthetic target coherent fidelity: {exc}")

    composite = (
        weights["retention_and_dead_ant"] * retention_and_dead_ant_score
        + weights["bandpass_flatness"] * bandpass_flatness_score
        + weights["fluxscale_accuracy"] * fluxscale_score
        + weights["target_fidelity"] * target_fidelity_score
    )
    if table_locked:
        composite = 0.0

    reward = round(min(1.0, max(0.0, composite)), 6)
    passed = 1 if reward >= pass_min else 0

    reward_dict = {
        "reward": reward,
        "passed": passed,
        "retention_and_dead_ant_score": round(retention_and_dead_ant_score, 6),
        "bandpass_flatness_score": round(bandpass_flatness_score, 6),
        "fluxscale_score": round(fluxscale_score, 6),
        "target_fidelity_score": round(target_fidelity_score, 6),
    }
    metrics_dict = {
        "task_id": task_id,
        "stage": "calibrate",
        "pass_reward_min": pass_min,
        "table_locked": table_locked,
        "has_corrected_data": has_corrected,
        "corrected_distinct_from_data": corrected_distinct,
        "antenna_gain_consistent": gain_consistent,
        "truth": {
            "flux_calibrator": truth["flux_calibrator"],
            "phase_calibrator": truth["phase_calibrator"],
            "dead_antenna": truth["dead_antenna"],
            "dead_antenna_flagged_min": truth["dead_antenna_flagged_min"],
            "retention_min": truth["retention_min"],
            "retention_max": truth["retention_max"],
            "expected_pcal_flux_jy": truth["expected_pcal_flux_jy"],
            "pcal_flux_rel_tol": truth["pcal_flux_rel_tol"],
            "max_slope_deg_100mhz": truth["max_slope_deg_100mhz"],
            "max_chan_rms_deg": truth["max_chan_rms_deg"],
            "max_coh_amp_err": None if truth["is_suite_r"] else truth["max_coh_amp_err"],
            "max_coh_phs_deg": None if truth["is_suite_r"] else truth["max_coh_phs_deg"],
            "min_pearson_r": None if truth["is_suite_r"] else truth["min_pearson_r"],
            "verifier_tclean_dr_min": (
                truth["verifier_tclean_dr_min"] if truth["is_suite_r"] else None
            ),
        },
        "measured": {
            "dead_antenna_flagged_fraction": round(dead_ant_flagged_frac, 6)
            if dead_ant_flagged_frac is not None
            else None,
            "target_row_retention": round(target_row_retention, 6)
            if target_row_retention is not None
            else None,
            "unflagged_target_rows_count": unflagged_target_rows_count,
            "target_corrected_std_jy": target_corr_std,
            "rank1_logamp_rms_median": rank1_logamp_rms,
            "j1331_source_ms": flatness_info.get("source_ms"),
            "j1331_unflagged_fraction": flatness_info.get("unflagged_fraction"),
            "j1331_phase_slope_deg_per_100mhz": flatness_info["coh_slope_deg_per_100mhz"],
            "j1331_med_baseline_slope_deg_per_100mhz": flatness_info["med_bl_slope_deg_per_100mhz"],
            "j1331_chan_phase_rms_deg": flatness_info["coh_chan_phase_rms_deg"],
            "j1331_med_baseline_chan_phase_rms_deg": flatness_info["med_bl_chan_phase_rms_deg"],
            "j1331_median_row_chan_phase_rms_deg": flatness_info["median_row_chan_phase_rms_deg"],
            "pcal_source_ms": pcal_source_ms,
            "pcal_unflagged_fraction": round(pcal_unflg_frac, 6)
            if pcal_unflg_frac is not None
            else None,
            "pcal_corrected_amp_jy": round(pcal_corr_amp_jy, 6)
            if pcal_corr_amp_jy is not None
            else None,
            "pcal_caltable_flux_jy": round(pcal_caltable_flux_jy, 6)
            if pcal_caltable_flux_jy is not None
            else None,
            "pcal_flux_jy": round(pcal_best_flux_jy, 6) if pcal_best_flux_jy is not None else None,
            "pcal_flux_rel_err": round(pcal_rel_err, 6) if pcal_rel_err is not None else None,
            **fidelity_measured,
        },
        "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
        "weights": weights,
        "errors": errors,
    }
    return reward_dict, metrics_dict
