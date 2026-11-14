"""Deterministic scientific verifier for inspect-stage tasks (s1_inspect, r1_3c391_inspect)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from radio_bench.grading.precal_metrics import has_active_table_lock

PASS_REWARD_MIN_DEFAULT = 0.85

FLUX_CAL_ALIASES = (
    "flux_calibrator",
    "flux_cal",
    "flux_calibrators",
    "primary_calibrator",
    "flux_density_calibrator",
)
BP_CAL_ALIASES = (
    "bandpass_calibrator",
    "bandpass_cal",
    "bp_calibrator",
    "bp_cal",
    "bandpass_calibrators",
)
COMBINED_FLUX_BP_ALIASES = (
    "flux_bandpass_calibrator",
    "flux_and_bandpass_calibrator",
)
PHASE_CAL_ALIASES = (
    "phase_calibrator",
    "phase_cal",
    "phase_calibrators",
    "gain_calibrator",
    "complex_gain_calibrator",
)
TARGET_ALIASES = (
    "target_fields",
    "mosaic_target_fields",
    "targets",
    "science_targets",
    "mosaic_fields",
    "target_pointings",
    "mosaic_pointings",
    "target_field_names",
)
DEAD_ANT_ALIASES = (
    "dead_antennas",
    "dead_antenna",
    "bad_antennas",
    "bad_antenna",
    "flagged_antennas",
    "corrupted_antennas",
    "zero_amplitude_antennas",
)
MISSING_ANT_ALIASES = (
    "missing_antennas",
    "missing_vla_pads",
    "missing_pads",
)
ELEV_ALIASES = (
    "bandpass_min_elevation_deg",
    "bp_min_elevation_deg",
    "min_bandpass_elevation_deg",
    "bandpass_calibrator_min_elevation_deg",
    "j1331_min_elevation_deg",
    "bandpass_min_el_deg",
)


def _norm_field(val: Any) -> str:
    if isinstance(val, dict):
        for k in ("name", "field", "field_name", "source"):
            if k in val and val[k] is not None:
                val = val[k]
                break
    s = str(val).strip().strip("'\"")
    return " ".join(s.split())


def _norm_ant(val: Any) -> str:
    if isinstance(val, dict):
        for k in ("name", "antenna", "antenna_name", "station"):
            if k in val and val[k] is not None:
                val = val[k]
                break
    return str(val).strip().strip("'\"").lower()


def _extract_value(payload: dict[str, Any], aliases: tuple[str, ...]) -> tuple[bool, Any]:
    for key in aliases:
        if key in payload and payload[key] is not None:
            return True, payload[key]
    for section_key in ("fields", "calibrators", "antennas", "elevation", "summary"):
        sub = payload.get(section_key)
        if isinstance(sub, dict):
            for key in aliases:
                if key in sub and sub[key] is not None:
                    return True, sub[key]
    return False, None


def _to_str_list(raw: Any, normalizer: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [normalizer(p) for p in parts if normalizer(p)]
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            normed = normalizer(item)
            if normed:
                out.append(normed)
        return out
    if isinstance(raw, dict):
        normed = normalizer(raw)
        return [normed] if normed else []
    return []


def _to_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    if isinstance(raw, str):
        cleaned = raw.strip().lower().replace("deg", "").replace("degrees", "").strip()
        try:
            val = float(cleaned)
            return val if math.isfinite(val) else None
        except ValueError:
            return None
    if isinstance(raw, dict):
        for k in ("value", "min_deg", "min_elevation_deg", "deg"):
            if k in raw:
                return _to_float(raw[k])
    return None


def _extract_truth(ground_truth: dict[str, Any]) -> dict[str, Any]:
    fields_sub = ground_truth.get("fields") if isinstance(ground_truth.get("fields"), dict) else {}
    flux_cal = (
        ground_truth.get("flux_calibrator") or fields_sub.get("flux_calibrator") or "J1331+3030"
    )
    bp_cal = (
        ground_truth.get("bandpass_calibrator") or fields_sub.get("bandpass_calibrator") or flux_cal
    )
    phase_cal = (
        ground_truth.get("phase_calibrator") or fields_sub.get("phase_calibrator") or "J1822-0938"
    )
    pol_cal = ground_truth.get("pol_leakage_calibrator") or fields_sub.get("pol_leakage_calibrator")

    raw_targets = (
        ground_truth.get("mosaic_target_fields")
        or ground_truth.get("target_fields")
        or fields_sub.get("target_fields")
        or []
    )
    target_fields = [_norm_field(t) for t in raw_targets]

    if "dead_antennas" in ground_truth and isinstance(ground_truth["dead_antennas"], list):
        dead_ants = [_norm_ant(a) for a in ground_truth["dead_antennas"]]
    elif "dead_antenna" in ground_truth and ground_truth["dead_antenna"]:
        dead_ants = [_norm_ant(ground_truth["dead_antenna"])]
    else:
        dead_ants = []

    obs = (
        ground_truth.get("observation") if isinstance(ground_truth.get("observation"), dict) else {}
    )
    missing_pads_raw = ground_truth.get("missing_vla_pads") or obs.get("missing_vla_pads") or []
    missing_pads = [_norm_ant(a) for a in missing_pads_raw]

    bp_el = ground_truth.get("bandpass_min_elevation_deg")
    if bp_el is None:
        bp_el = ground_truth.get("bp_min_elevation_deg")
    bp_el_deg = float(bp_el) if bp_el is not None else 0.0

    tols = (
        ground_truth.get("tolerances") if isinstance(ground_truth.get("tolerances"), dict) else {}
    )
    el_tol = (
        tols.get("elevation_tol_deg")
        or tols.get("elevation_abs_deg")
        or ground_truth.get("bp_min_elevation_tolerance_deg")
        or 0.25
    )
    pass_min = float(
        ground_truth.get("pass_reward_min", tols.get("pass_reward_min", PASS_REWARD_MIN_DEFAULT))
    )

    weights_raw = (
        ground_truth.get("weights") if isinstance(ground_truth.get("weights"), dict) else {}
    )
    weights = {
        "valid_json": float(weights_raw.get("valid_json", 0.20)),
        "flux_bandpass_calibrator": float(weights_raw.get("flux_bandpass_calibrator", 0.20)),
        "phase_vs_target_separation": float(weights_raw.get("phase_vs_target_separation", 0.25)),
        "dead_antenna": float(weights_raw.get("dead_antenna", 0.15)),
        "bandpass_min_elevation": float(weights_raw.get("bandpass_min_elevation", 0.20)),
    }

    return {
        "flux_calibrator": _norm_field(flux_cal),
        "bandpass_calibrator": _norm_field(bp_cal),
        "phase_calibrator": _norm_field(phase_cal),
        "pol_leakage_calibrator": _norm_field(pol_cal) if pol_cal else None,
        "target_fields": target_fields,
        "dead_antennas": dead_ants,
        "missing_vla_pads": missing_pads,
        "bandpass_min_elevation_deg": bp_el_deg,
        "elevation_tol_deg": float(el_tol),
        "pass_reward_min": pass_min,
        "weights": weights,
    }


def grade_inspect(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    task_id: str,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade `/workspace/inspect_summary.json` against ground-truth observation triage."""
    truth = _extract_truth(ground_truth)
    weights = truth["weights"]
    pass_min = truth["pass_reward_min"]
    ws = Path(workspace_dir)
    input_ms = ws / "input.ms"
    summary_path = ws / "inspect_summary.json"

    errors: list[str] = []
    if input_ms.is_dir() and has_active_table_lock(input_ms):
        errors.append(f"Active casacore write lock on {input_ms}")
        reward_dict_locked: dict[str, float | int] = {
            "reward": 0.0,
            "passed": 0,
            "valid_json_score": 0.0,
            "flux_bandpass_score": 0.0,
            "phase_vs_target_score": 0.0,
            "dead_antenna_score": 0.0,
            "elevation_score": 0.0,
        }
        metrics_dict_locked: dict[str, Any] = {
            "task_id": task_id,
            "stage": "inspect",
            "pass_reward_min": pass_min,
            "table_locked": True,
            "summary_exists": summary_path.is_file(),
            "valid_json": False,
            "truth": truth,
            "measured": {},
            "subscores": {
                k: v for k, v in reward_dict_locked.items() if k not in ("reward", "passed")
            },
            "weights": weights,
            "errors": errors,
        }
        return reward_dict_locked, metrics_dict_locked

    if not summary_path.is_file():
        errors.append(f"Missing summary file: {summary_path}")
        reward_dict: dict[str, float | int] = {
            "reward": 0.0,
            "passed": 0,
            "valid_json_score": 0.0,
            "flux_bandpass_score": 0.0,
            "phase_vs_target_score": 0.0,
            "dead_antenna_score": 0.0,
            "elevation_score": 0.0,
        }
        metrics_dict: dict[str, Any] = {
            "task_id": task_id,
            "stage": "inspect",
            "pass_reward_min": pass_min,
            "table_locked": False,
            "summary_exists": False,
            "valid_json": False,
            "truth": truth,
            "measured": {},
            "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
            "weights": weights,
            "errors": errors,
        }
        return reward_dict, metrics_dict

    try:
        raw_text = summary_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except Exception as exc:
        errors.append(f"Malformed JSON in {summary_path}: {exc}")
        reward_dict = {
            "reward": 0.0,
            "passed": 0,
            "valid_json_score": 0.0,
            "flux_bandpass_score": 0.0,
            "phase_vs_target_score": 0.0,
            "dead_antenna_score": 0.0,
            "elevation_score": 0.0,
        }
        metrics_dict = {
            "task_id": task_id,
            "stage": "inspect",
            "pass_reward_min": pass_min,
            "table_locked": False,
            "summary_exists": True,
            "valid_json": False,
            "truth": truth,
            "measured": {},
            "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
            "weights": weights,
            "errors": errors,
        }
        return reward_dict, metrics_dict

    if not isinstance(payload, dict) or len(payload) == 0:
        errors.append("inspect_summary.json must be a non-empty JSON object")
        reward_dict = {
            "reward": 0.0,
            "passed": 0,
            "valid_json_score": 0.0,
            "flux_bandpass_score": 0.0,
            "phase_vs_target_score": 0.0,
            "dead_antenna_score": 0.0,
            "elevation_score": 0.0,
        }
        metrics_dict = {
            "task_id": task_id,
            "stage": "inspect",
            "pass_reward_min": pass_min,
            "table_locked": False,
            "summary_exists": True,
            "valid_json": False,
            "truth": truth,
            "measured": {},
            "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
            "weights": weights,
            "errors": errors,
        }
        return reward_dict, metrics_dict

    has_flux, raw_flux = _extract_value(payload, FLUX_CAL_ALIASES)
    has_bp, raw_bp = _extract_value(payload, BP_CAL_ALIASES)
    if not has_flux or not has_bp:
        has_comb, raw_comb = _extract_value(payload, COMBINED_FLUX_BP_ALIASES)
        if has_comb:
            if not has_flux:
                has_flux, raw_flux = True, raw_comb
            if not has_bp:
                has_bp, raw_bp = True, raw_comb

    has_phase, raw_phase = _extract_value(payload, PHASE_CAL_ALIASES)
    has_targets, raw_targets = _extract_value(payload, TARGET_ALIASES)
    has_dead, raw_dead = _extract_value(payload, DEAD_ANT_ALIASES)
    _has_missing, raw_missing = _extract_value(payload, MISSING_ANT_ALIASES)
    has_elev, raw_elev = _extract_value(payload, ELEV_ALIASES)

    pred_flux = _to_str_list(raw_flux, _norm_field)
    pred_bp = _to_str_list(raw_bp, _norm_field)
    pred_phase = _to_str_list(raw_phase, _norm_field)
    pred_targets = _to_str_list(raw_targets, _norm_field)
    pred_dead = _to_str_list(raw_dead, _norm_ant)
    pred_missing = _to_str_list(raw_missing, _norm_ant)
    pred_elev = _to_float(raw_elev)

    # 1. Valid JSON & required logical fields present (0.20 weight)
    sections_present = sum(
        [
            1 if (pred_flux or pred_bp) else 0,
            1 if (pred_phase and pred_targets) else 0,
            1 if has_dead else 0,
            1 if (has_elev and pred_elev is not None) else 0,
        ]
    )
    valid_json_score = 0.20 + 0.80 * (sections_present / 4.0)

    # 2. Flux + bandpass calibrator identification (0.20 weight)
    true_flux_upper = truth["flux_calibrator"].upper()
    true_bp_upper = truth["bandpass_calibrator"].upper()
    flux_set_upper = {x.upper() for x in pred_flux}
    bp_set_upper = {x.upper() for x in pred_bp}

    flux_ok = 1.0 if flux_set_upper == {true_flux_upper} else 0.0
    bp_ok = 1.0 if bp_set_upper == {true_bp_upper} else 0.0
    flux_bandpass_score = 0.5 * flux_ok + 0.5 * bp_ok

    # 3. Phase calibrator vs target mosaic pointing separation (0.25 weight)
    true_phase_upper = truth["phase_calibrator"].upper()
    true_pol_upper = (
        truth["pol_leakage_calibrator"].upper() if truth["pol_leakage_calibrator"] else None
    )
    true_targets_upper = {t.upper() for t in truth["target_fields"]}
    pred_phase_upper = {x.upper() for x in pred_phase}
    pred_targets_upper = {x.upper() for x in pred_targets}

    if pred_phase_upper == {true_phase_upper}:
        phase_part = 1.0
    elif true_pol_upper is not None and pred_phase_upper == {true_phase_upper, true_pol_upper}:
        # ms_set_intents assigns CALIBRATE_PHASE;CALIBRATE_POL_LEAKAGE to J0319+4130
        phase_part = 0.85
    else:
        phase_part = 0.0

    calibrator_names_upper = {true_flux_upper, true_bp_upper, true_phase_upper}
    if true_pol_upper:
        calibrator_names_upper.add(true_pol_upper)

    cal_confused_as_target = bool(pred_targets_upper & calibrator_names_upper)
    target_confused_as_phase = bool(pred_phase_upper & true_targets_upper)

    if not pred_targets_upper or not true_targets_upper:
        target_part = 0.0
    else:
        inter = len(pred_targets_upper & true_targets_upper)
        union = len(pred_targets_upper | true_targets_upper)
        target_part = inter / max(union, 1)
        if cal_confused_as_target:
            # Penalize confusing J0319+4130 or J1822-0938 as a science mosaic pointing
            target_part *= 0.25

    if target_confused_as_phase:
        phase_part = 0.0

    phase_vs_target_score = 0.5 * phase_part + 0.5 * target_part

    # 4. Dead antenna identification (0.15 weight)
    true_dead_set = set(truth["dead_antennas"])
    missing_pads_set = set(truth["missing_vla_pads"])
    pred_dead_set = set(pred_dead)
    # Do not penalize if missing VLA pads (ea06, ea10 on 3c391) are included alongside ea13
    effective_pred_dead = pred_dead_set - (missing_pads_set - true_dead_set)

    if not true_dead_set:
        dead_antenna_score = 1.0 if not effective_pred_dead else 0.0
    elif not effective_pred_dead:
        dead_antenna_score = 0.0
    else:
        tp = len(effective_pred_dead & true_dead_set)
        fp = len(effective_pred_dead - true_dead_set)
        fn = len(true_dead_set - effective_pred_dead)
        dead_antenna_score = tp / max(tp + fp + fn, 1)

    # 5. Bandpass calibrator minimum elevation within +-0.25 deg (0.20 weight)
    elev_err_deg: float | None = None
    if pred_elev is None:
        elevation_score = 0.0
    else:
        elev_err_deg = abs(pred_elev - truth["bandpass_min_elevation_deg"])
        tol = truth["elevation_tol_deg"]
        if elev_err_deg <= tol:
            elevation_score = 1.0
        else:
            # Smooth linear ramp from 1.0 at tol (0.25 deg) to 0.0 at tol + 1.25 (1.50 deg)
            elevation_score = max(0.0, 1.0 - (elev_err_deg - tol) / 1.25)

    composite = (
        weights["valid_json"] * valid_json_score
        + weights["flux_bandpass_calibrator"] * flux_bandpass_score
        + weights["phase_vs_target_separation"] * phase_vs_target_score
        + weights["dead_antenna"] * dead_antenna_score
        + weights["bandpass_min_elevation"] * elevation_score
    )
    reward = round(min(1.0, max(0.0, composite)), 6)
    core_subscores_ok = (
        flux_bandpass_score >= 0.5
        and phase_part >= 0.5
        and target_part >= 0.5
        and dead_antenna_score >= 0.5
        and elevation_score >= 0.5
    )
    passed = 1 if (reward >= pass_min and core_subscores_ok) else 0

    reward_dict = {
        "reward": reward,
        "passed": passed,
        "valid_json_score": round(valid_json_score, 6),
        "flux_bandpass_score": round(flux_bandpass_score, 6),
        "phase_vs_target_score": round(phase_vs_target_score, 6),
        "dead_antenna_score": round(dead_antenna_score, 6),
        "elevation_score": round(elevation_score, 6),
    }
    metrics_dict = {
        "task_id": task_id,
        "stage": "inspect",
        "pass_reward_min": pass_min,
        "table_locked": False,
        "summary_exists": True,
        "valid_json": True,
        "truth": truth,
        "measured": {
            "flux_calibrator": pred_flux,
            "bandpass_calibrator": pred_bp,
            "phase_calibrator": pred_phase,
            "target_fields": pred_targets,
            "dead_antennas": pred_dead,
            "missing_antennas": pred_missing,
            "bandpass_min_elevation_deg": pred_elev,
            "elevation_abs_error_deg": round(elev_err_deg, 6) if elev_err_deg is not None else None,
            "phase_part": round(phase_part, 6),
            "target_part": round(target_part, 6),
            "core_subscores_ok": core_subscores_ok,
            "calibrator_confused_as_target": cal_confused_as_target,
            "target_confused_as_phase_cal": target_confused_as_phase,
        },
        "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
        "weights": weights,
        "errors": errors,
    }
    return reward_dict, metrics_dict
