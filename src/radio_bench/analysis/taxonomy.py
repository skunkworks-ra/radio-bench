"""Deterministic non-LLM failure-mode taxonomy for radio-bench trials."""

from __future__ import annotations

import math
from typing import Any

FAILURE_NONE = "none"

# Exact SPEC.md Section 5.3 categories in strict deterministic evaluation priority order.
FAILURE_CATEGORIES: tuple[str, ...] = (
    "lock_or_timeout",
    "tool_protocol_error",
    "script_or_casa_crash",
    "flux_model_or_fluxscale_error",
    "skipped_delay_or_bandpass",
    "flagging_recall_or_overflag",
    "beam_sampling_or_clean_bowl",
)

ALL_OUTCOME_LABELS: tuple[str, ...] = (FAILURE_NONE, *FAILURE_CATEGORIES)

NON_TOOL_ARMS: frozenset[str] = frozenset({"00_none", "01_skills_raw", "01_skills_casa"})

TIMEOUT_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "EnvironmentStartTimeoutError",
        "AgentSetupTimeoutError",
        "Timeout",
        "TimeoutError",
    }
)

PROTOCOL_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "APIResponseValidationError",
        "ToolProtocolError",
        "MCPConnectionError",
    }
)

_MCP_EXC_PHRASES: tuple[str, ...] = (
    "mcp server",
    "mcp tool",
    "mcpconnection",
    "mcp error",
    "mcp_",
)

# Deterministic fallback map from stage subscore keys -> SPEC.md category.
_SUBSCORE_TO_CATEGORY: dict[str, str] = {
    "valid_json_score": "script_or_casa_crash",
    "calibrators_ms_score": "script_or_casa_crash",
    "products_score": "script_or_casa_crash",
    "img_products_score": "script_or_casa_crash",
    "model_data_score": "flux_model_or_fluxscale_error",
    "fluxscale_score": "flux_model_or_fluxscale_error",
    "cal_fluxscale_score": "flux_model_or_fluxscale_error",
    "flux_bandpass_score": "flux_model_or_fluxscale_error",
    "bandpass_flatness_score": "skipped_delay_or_bandpass",
    "cal_bandpass_flatness_score": "skipped_delay_or_bandpass",
    "target_fidelity_score": "skipped_delay_or_bandpass",
    "cal_target_fidelity_score": "skipped_delay_or_bandpass",
    "elevation_score": "skipped_delay_or_bandpass",
    "calibration_score": "skipped_delay_or_bandpass",
    "flag_recall_score": "flagging_recall_or_overflag",
    "recall_score": "flagging_recall_or_overflag",
    "clean_fpr_score": "flagging_recall_or_overflag",
    "dead_antenna_score": "flagging_recall_or_overflag",
    "retention_and_dead_ant_score": "flagging_recall_or_overflag",
    "cal_retention_and_dead_ant_score": "flagging_recall_or_overflag",
    "phase_vs_target_score": "flagging_recall_or_overflag",
    "beam_sampling_score": "beam_sampling_or_clean_bowl",
    "img_beam_sampling_score": "beam_sampling_or_clean_bowl",
    "residual_rms_score": "beam_sampling_or_clean_bowl",
    "img_residual_rms_score": "beam_sampling_or_clean_bowl",
    "deconv_and_bowl_score": "beam_sampling_or_clean_bowl",
    "img_deconv_and_bowl_score": "beam_sampling_or_clean_bowl",
    "delivered_image_score": "beam_sampling_or_clean_bowl",
}

_CATEGORY_PRIORITY: dict[str, int] = {cat: idx for idx, cat in enumerate(FAILURE_CATEGORIES)}


def _as_float(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        if math.isfinite(fval):
            return fval
    return None


def _collect_subscores(
    reward_dict: dict[str, Any] | None,
    metrics_dict: dict[str, Any] | None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(metrics_dict, dict):
        for prefix, block_key in (("cal_", "calibration"), ("img_", "image")):
            block = metrics_dict.get(block_key)
            if isinstance(block, dict) and isinstance(block.get("subscores"), dict):
                for k, v in block["subscores"].items():
                    fv = _as_float(v)
                    if fv is not None:
                        scores[f"{prefix}{k}"] = fv
                        scores.setdefault(str(k), fv)
        if isinstance(metrics_dict.get("subscores"), dict):
            for k, v in metrics_dict["subscores"].items():
                fv = _as_float(v)
                if fv is not None:
                    scores[str(k)] = fv
    if isinstance(reward_dict, dict):
        for k, v in reward_dict.items():
            if k in ("reward", "passed"):
                continue
            fv = _as_float(v)
            if fv is not None:
                scores[str(k)] = fv
    return scores


def _get_first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        if k in mapping:
            fv = _as_float(mapping.get(k))
            if fv is not None:
                return fv
    return None


def _has_lock_phrase(text: str) -> bool:
    low = text.lower()
    return "casacore write lock" in low or "table lock" in low or "table.lock" in low


def _is_table_locked(metrics_dict: dict[str, Any] | None, exc_msg_low: str = "") -> bool:
    if exc_msg_low and _has_lock_phrase(exc_msg_low):
        return True
    if not isinstance(metrics_dict, dict):
        return False
    if metrics_dict.get("table_locked") is True:
        return True
    errors = metrics_dict.get("errors")
    if isinstance(errors, list):
        for err in errors:
            if _has_lock_phrase(str(err)):
                return True
    for nested_key in ("calibration", "image"):
        nested = metrics_dict.get(nested_key)
        if isinstance(nested, dict):
            if nested.get("table_locked") is True:
                return True
            n_errs = nested.get("errors")
            if isinstance(n_errs, list):
                for err in n_errs:
                    if _has_lock_phrase(str(err)):
                        return True
    return False


def _image_set_missing(img_measured: dict[str, Any]) -> bool:
    chosen_prefix = img_measured.get("chosen_prefix")
    if isinstance(chosen_prefix, str) and chosen_prefix.strip():
        return False
    sets_found = img_measured.get("image_sets_found")
    if isinstance(sets_found, (int, float)) and not isinstance(sets_found, bool):
        return int(sets_found) == 0
    return False


def _deliverable_missing(
    stage: str,
    reward_present: bool,
    subscores: dict[str, float],
    metrics_dict: dict[str, Any] | None,
) -> bool:
    if not reward_present and not isinstance(metrics_dict, dict):
        return True

    m = metrics_dict if isinstance(metrics_dict, dict) else {}
    measured = m.get("measured") if isinstance(m.get("measured"), dict) else {}

    if stage == "inspect":
        if m.get("summary_exists") is False or m.get("valid_json") is False:
            return True
        vj = subscores.get("valid_json_score")
        return bool(vj is not None and vj <= 0.0)

    if stage == "precal":
        if measured.get("calibrators_ms_exists") is False:
            return True
        cms = subscores.get("calibrators_ms_score")
        return bool(cms is not None and cms <= 0.0 and measured.get("calibrators_ms_nrows", 0) == 0)

    if stage == "calibrate":
        has_corr = (
            m.get("has_corrected_data")
            if "has_corrected_data" in m
            else measured.get("has_corrected_data")
        )
        corr_dist = (
            m.get("corrected_distinct_from_data")
            if "corrected_distinct_from_data" in m
            else measured.get("corrected_distinct_from_data")
        )
        return bool(has_corr is False or corr_dist is False)

    if stage == "image":
        # Only treat deliverable as missing when NO CASA image set was produced,
        # NOT when products_score == 0.0 due to gridder='standard' or missing pbcor.
        return _image_set_missing(measured)

    if stage == "e2e":
        # In Rule 3, only treat e2e deliverable as immediately missing when calibration
        # deliverable is missing; if calibration exists, evaluate calibration physics
        # rules (Rules 4-6) before falling back to missing image deliverable.
        cal_m = m.get("calibration") if isinstance(m.get("calibration"), dict) else {}
        meas_cal = (
            measured.get("calibration") if isinstance(measured.get("calibration"), dict) else {}
        )
        has_corr = next(
            (d["has_corrected_data"] for d in (cal_m, meas_cal, m) if "has_corrected_data" in d),
            None,
        )
        corr_dist = next(
            (
                d["corrected_distinct_from_data"]
                for d in (cal_m, meas_cal, m)
                if "corrected_distinct_from_data" in d
            ),
            None,
        )
        return bool(has_corr is False or corr_dist is False)

    return False


def _e2e_image_deliverable_missing(metrics_dict: dict[str, Any] | None) -> bool:
    if not isinstance(metrics_dict, dict):
        return False
    img_m = metrics_dict.get("image")
    if isinstance(img_m, dict) and isinstance(img_m.get("measured"), dict):
        return _image_set_missing(img_m["measured"])
    meas = metrics_dict.get("measured")
    if isinstance(meas, dict) and isinstance(meas.get("image"), dict):
        return _image_set_missing(meas["image"])
    return False


def classify_failure(
    *,
    passed: int | bool,
    stage: str,
    arm: str = "00_none",
    reward: float = 0.0,
    reward_dict: dict[str, Any] | None = None,
    metrics_dict: dict[str, Any] | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
    tool_errors: int = 0,
    protocol_errors: int = 0,
    mcp_calls_total: int = 0,
    bash_nonzero_exits: int = 0,
    severe_casa_errors: int = 0,
) -> str:
    """Deterministically classify a trial into `'none'` or one of the 7 SPEC.md failure modes."""
    if bool(passed):
        return FAILURE_NONE

    exc_type = (exception_type or "").strip()
    exc_msg_low = (exception_message or "").lower()
    subscores = _collect_subscores(reward_dict, metrics_dict)
    reward_present = bool(reward_dict)
    missing_deliverable = _deliverable_missing(stage, reward_present, subscores, metrics_dict)

    # 1. lock_or_timeout
    if (
        exc_type in TIMEOUT_EXCEPTIONS
        or "timeout" in exc_type.lower()
        or _is_table_locked(metrics_dict, exc_msg_low=exc_msg_low)
    ):
        return "lock_or_timeout"

    # 2. tool_protocol_error
    attempted_mcp_without_tools = arm in NON_TOOL_ARMS and mcp_calls_total > 0
    if (
        exc_type in PROTOCOL_EXCEPTIONS
        or any(phrase in exc_msg_low for phrase in _MCP_EXC_PHRASES)
        or attempted_mcp_without_tools
        or protocol_errors >= 2
        or (protocol_errors >= 1 and (missing_deliverable or reward < 0.25 or arm in NON_TOOL_ARMS))
        or (tool_errors >= 3 and (missing_deliverable or reward < 0.25))
    ):
        return "tool_protocol_error"

    # 3. script_or_casa_crash
    if (
        bool(exc_type)
        or missing_deliverable
        or (bash_nonzero_exits >= 3 and reward < 0.25)
        or (severe_casa_errors > 0 and reward < 0.25)
    ):
        return "script_or_casa_crash"

    # 4. flux_model_or_fluxscale_error
    model_data_sc = subscores.get("model_data_score")
    flux_bp_sc = subscores.get("flux_bandpass_score")
    fluxscale_sc = _get_first_float(subscores, "cal_fluxscale_score", "fluxscale_score")
    if (
        (stage == "inspect" and flux_bp_sc is not None and flux_bp_sc < 0.5)
        or (stage == "precal" and model_data_sc is not None and model_data_sc < 0.5)
        or (stage in ("calibrate", "e2e") and fluxscale_sc is not None and fluxscale_sc < 0.5)
    ):
        return "flux_model_or_fluxscale_error"

    # 5. skipped_delay_or_bandpass
    elev_sc = subscores.get("elevation_score")
    bp_flat_sc = _get_first_float(
        subscores, "cal_bandpass_flatness_score", "bandpass_flatness_score"
    )
    tgt_fid_sc = _get_first_float(subscores, "cal_target_fidelity_score", "target_fidelity_score")
    ret_sc = _get_first_float(
        subscores, "cal_retention_and_dead_ant_score", "retention_and_dead_ant_score"
    )
    if stage == "inspect" and elev_sc is not None and elev_sc < 0.5:
        return "skipped_delay_or_bandpass"
    if stage in ("calibrate", "e2e"):
        if bp_flat_sc is not None and bp_flat_sc < 0.5:
            return "skipped_delay_or_bandpass"
        if (
            tgt_fid_sc is not None
            and tgt_fid_sc < 0.5
            and (ret_sc is None or ret_sc >= 0.5)
            and (fluxscale_sc is None or fluxscale_sc >= 0.5)
        ):
            return "skipped_delay_or_bandpass"

    # 6. flagging_recall_or_overflag
    recall_sc = _get_first_float(subscores, "flag_recall_score", "recall_score")
    fpr_sc = subscores.get("clean_fpr_score")
    dead_ant_sc = subscores.get("dead_antenna_score")
    if stage == "precal" and (
        (recall_sc is not None and recall_sc < 0.5) or (fpr_sc is not None and fpr_sc < 0.5)
    ):
        return "flagging_recall_or_overflag"
    if stage == "inspect" and dead_ant_sc is not None and dead_ant_sc < 0.5:
        return "flagging_recall_or_overflag"
    if stage in ("calibrate", "e2e") and ret_sc is not None and ret_sc < 0.5:
        return "flagging_recall_or_overflag"

    # On e2e, if calibration physics passed Rules 4-6 but no CASA image set was produced,
    # classify as script_or_casa_crash before checking image quality rules.
    if stage == "e2e" and _e2e_image_deliverable_missing(metrics_dict):
        return "script_or_casa_crash"

    # 7. beam_sampling_or_clean_bowl
    beam_sc = _get_first_float(subscores, "img_beam_sampling_score", "beam_sampling_score")
    rms_sc = _get_first_float(subscores, "img_residual_rms_score", "residual_rms_score")
    deconv_sc = _get_first_float(subscores, "img_deconv_and_bowl_score", "deconv_and_bowl_score")
    m = metrics_dict if isinstance(metrics_dict, dict) else {}
    img_m = m.get("image") if stage == "e2e" and isinstance(m.get("image"), dict) else m
    img_meas = img_m.get("measured") if isinstance(img_m.get("measured"), dict) else {}
    img_truth = img_m.get("truth") if isinstance(img_m.get("truth"), dict) else {}
    if not img_truth and stage == "e2e" and isinstance(m.get("truth"), dict):
        nested_t = m["truth"].get("image")
        if isinstance(nested_t, dict):
            img_truth = nested_t
    bowl_max = _get_first_float(img_truth, "bowl_max", "max_negative_bowl_ratio")
    if bowl_max is None:
        bowl_max = 0.12
    bowl_ratio = _get_first_float(img_meas, "negative_bowl_ratio", "negative_model_bowl_ratio")

    if stage in ("image", "e2e") and (
        (beam_sc is not None and beam_sc < 0.5)
        or (rms_sc is not None and rms_sc < 0.5)
        or (deconv_sc is not None and deconv_sc < 0.5)
        or (bowl_ratio is not None and bowl_ratio >= bowl_max)
    ):
        return "beam_sampling_or_clean_bowl"

    # Deterministic fallback: map lowest-scoring leaf stage subscore to its SPEC.md category.
    has_cal_leaves = any(k.startswith("cal_") for k in subscores)
    has_img_leaves = any(k.startswith("img_") for k in subscores)
    candidates: list[tuple[float, int, str]] = []
    for sub_key, sub_val in subscores.items():
        if sub_key == "calibration_score" and has_cal_leaves:
            continue
        if sub_key == "delivered_image_score" and has_img_leaves:
            continue
        cat = _SUBSCORE_TO_CATEGORY.get(sub_key)
        if cat is not None:
            candidates.append((sub_val, _CATEGORY_PRIORITY[cat], cat))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    return "script_or_casa_crash"


def summarize_failure_taxonomy(
    records: list[dict[str, Any]],
    arms_order: tuple[str, ...] = (
        "00_none",
        "10_tools",
        "01_skills_raw",
        "01_skills_casa",
        "11_full",
        "11_orchestrated",
    ),
) -> dict[str, Any]:
    """Aggregate failure-mode counts and fractions overall, by arm, and by (model, arm)."""
    overall_counts = {label: 0 for label in ALL_OUTCOME_LABELS}
    by_arm: dict[str, dict[str, Any]] = {}
    by_model_arm: dict[str, dict[str, Any]] = {}

    present_arms = [arm for arm in arms_order if any(r.get("arm") == arm for r in records)]
    extra_arms = sorted(
        {
            str(r.get("arm"))
            for r in records
            if r.get("arm") and str(r.get("arm")) not in present_arms
        }
    )
    ordered_arms = [*present_arms, *extra_arms]

    for arm in ordered_arms:
        by_arm[arm] = {
            "n_trials": 0,
            "n_failed": 0,
            "counts": {label: 0 for label in ALL_OUTCOME_LABELS},
            "rates": {label: 0.0 for label in ALL_OUTCOME_LABELS},
        }

    for rec in records:
        mode = str(rec.get("failure_mode") or "script_or_casa_crash")
        if mode not in overall_counts:
            mode = "script_or_casa_crash"
        overall_counts[mode] += 1

        arm = str(rec.get("arm") or "00_none")
        model = str(rec.get("model_name") or "unknown")
        if arm not in by_arm:
            by_arm[arm] = {
                "n_trials": 0,
                "n_failed": 0,
                "counts": {label: 0 for label in ALL_OUTCOME_LABELS},
                "rates": {label: 0.0 for label in ALL_OUTCOME_LABELS},
            }
        arm_entry = by_arm[arm]
        arm_entry["n_trials"] += 1
        if mode != FAILURE_NONE:
            arm_entry["n_failed"] += 1
        arm_entry["counts"][mode] += 1

        ma_key = f"{model}__{arm}"
        if ma_key not in by_model_arm:
            by_model_arm[ma_key] = {
                "model_name": model,
                "arm": arm,
                "n_trials": 0,
                "n_failed": 0,
                "counts": {label: 0 for label in ALL_OUTCOME_LABELS},
                "rates": {label: 0.0 for label in ALL_OUTCOME_LABELS},
            }
        ma_entry = by_model_arm[ma_key]
        ma_entry["n_trials"] += 1
        if mode != FAILURE_NONE:
            ma_entry["n_failed"] += 1
        ma_entry["counts"][mode] += 1

    for entry in [*by_arm.values(), *by_model_arm.values()]:
        n = int(entry["n_trials"])
        if n > 0:
            entry["rates"] = {
                label: round(entry["counts"][label] / n, 6) for label in ALL_OUTCOME_LABELS
            }

    n_total = len(records)
    overall_rates = {
        label: round(overall_counts[label] / n_total, 6) if n_total > 0 else 0.0
        for label in ALL_OUTCOME_LABELS
    }
    return {
        "n_trials": n_total,
        "n_failed": sum(v for k, v in overall_counts.items() if k != FAILURE_NONE),
        "overall_counts": overall_counts,
        "overall_rates": overall_rates,
        "by_arm": by_arm,
        "by_model_arm": dict(sorted(by_model_arm.items())),
    }
