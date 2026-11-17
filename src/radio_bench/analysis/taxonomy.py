"""Deterministic non-LLM failure-mode taxonomy for radio-bench trials."""

from __future__ import annotations

import math
from typing import Any

FAILURE_NONE = "none"

# Exact v0.2 SPEC-output-and-metrics Section 3.1 categories in strict evaluation priority order.
FAILURE_CATEGORIES: tuple[str, ...] = (
    "lock_or_timeout",
    "tool_protocol_error",
    "invalid_abstention_or_halt",
    "raw_data_or_calflag_mutation",
    "unsupported_or_missing_claim",
    "script_or_casa_crash",
    "flux_model_or_fluxscale_error",
    "skipped_delay_or_bandpass",
    "flagging_recall_or_overflag",
    "beam_sampling_or_clean_bowl",
    "postcal_clip_or_uv_bias",
    "polcal_feed_or_parang_error",
    "cube_baseline_or_frame_error",
    "selfcal_divergence_or_shift",
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

# Deterministic v0.1 fallback map from stage subscore keys -> failure category.
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


def _get_measurement_record(
    metrics_dict: dict[str, Any] | None, obs_id: str
) -> dict[str, Any] | None:
    if not isinstance(metrics_dict, dict):
        return None
    measurements = metrics_dict.get("measurements")
    if isinstance(measurements, dict):
        rec = measurements.get(obs_id)
        if isinstance(rec, dict):
            return rec
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
    lk_rec = _get_measurement_record(metrics_dict, "proc.table_locked")
    if lk_rec is not None and lk_rec.get("value") is True:
        return True
    tags = metrics_dict.get("triggered_failure_tags")
    if isinstance(tags, list) and "lock_or_timeout" in tags:
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


def _is_invalid_abstention_or_halt(metrics_dict: dict[str, Any] | None) -> bool:
    if not isinstance(metrics_dict, dict):
        return False
    tags = metrics_dict.get("triggered_failure_tags")
    if isinstance(tags, list) and "invalid_abstention_or_halt" in tags:
        return True

    rubric_meta = metrics_dict.get("rubric_meta")
    expect_halt = bool(rubric_meta.get("expect_halt")) if isinstance(rubric_meta, dict) else False
    st_rec = _get_measurement_record(metrics_dict, "proc.manifest_status")
    st_val = str(st_rec.get("value") or "").strip().lower() if st_rec is not None else ""
    ab_rec = _get_measurement_record(metrics_dict, "proc.abstention_correct")

    if expect_halt:
        if (
            ab_rec is not None
            and ab_rec.get("status") == "measured"
            and ab_rec.get("value") is False
        ):
            return True
        if st_val and st_val != "halted":
            return True
    elif st_val == "halted":
        return True
    return False


def _is_raw_data_or_calflag_mutation(metrics_dict: dict[str, Any] | None) -> bool:
    if not isinstance(metrics_dict, dict):
        return False
    tags = metrics_dict.get("triggered_failure_tags")
    if isinstance(tags, list) and "raw_data_or_calflag_mutation" in tags:
        return True

    raw_rec = _get_measurement_record(metrics_dict, "proc.raw_data_unmodified")
    if (
        raw_rec is not None
        and raw_rec.get("status") == "measured"
        and raw_rec.get("value") is False
    ):
        return True

    cal_flg_rec = _get_measurement_record(metrics_dict, "postcal.calibrator_flag_mutation")
    if (
        cal_flg_rec is not None
        and cal_flg_rec.get("status") == "measured"
        and (_as_float(cal_flg_rec.get("value")) or 0.0) > 0.0
    ):
        return True

    vis_unmod_rec = _get_measurement_record(metrics_dict, "postcal.vis_data_unmodified")
    return bool(
        vis_unmod_rec is not None
        and vis_unmod_rec.get("status") == "measured"
        and vis_unmod_rec.get("value") is False
    )


def _is_unsupported_or_missing_claim(metrics_dict: dict[str, Any] | None) -> bool:
    """Return True strictly on Layer B Claim Auditor (`reported_metrics` / `report.html`) failures."""
    if not isinstance(metrics_dict, dict):
        return False
    tags = metrics_dict.get("triggered_failure_tags")
    if isinstance(tags, list) and "unsupported_or_missing_claim" in tags:
        return True

    unsup_rec = _get_measurement_record(metrics_dict, "proc.unsupported_claim_count")
    if (
        unsup_rec is not None
        and unsup_rec.get("status") == "measured"
        and int(_as_float(unsup_rec.get("value")) or 0) > 0
    ):
        return True

    rubric_meta = metrics_dict.get("rubric_meta")
    req_claims = (
        rubric_meta.get("required_reported_metrics") if isinstance(rubric_meta, dict) else None
    )
    miss_rec = _get_measurement_record(metrics_dict, "proc.missing_required_claim_count")
    return bool(
        isinstance(req_claims, list)
        and len(req_claims) > 0
        and miss_rec is not None
        and miss_rec.get("status") == "measured"
        and int(_as_float(miss_rec.get("value")) or 0) > 0
    )


def _image_set_missing(
    img_measured: dict[str, Any], metrics_dict: dict[str, Any] | None = None
) -> bool:
    vp_rec = _get_measurement_record(metrics_dict, "image.valid_product_count")
    if vp_rec is not None and vp_rec.get("status") == "measured":
        return int(_as_float(vp_rec.get("value")) or 0) == 0
    chosen_prefix = img_measured.get("chosen_prefix")
    if isinstance(chosen_prefix, str) and chosen_prefix.strip():
        return False
    sets_found = img_measured.get("image_sets_found")
    if isinstance(sets_found, (int, float)) and not isinstance(sets_found, bool):
        return int(sets_found) == 0
    return False


def _stage_deliverable_missing_v02(stage: str, metrics_dict: dict[str, Any]) -> bool | None:
    """Check whether primary stage deliverable is missing using `v0.2` `measurements`."""
    measurements = metrics_dict.get("measurements")
    if not isinstance(measurements, dict):
        return None

    st = stage.strip().lower()
    if st == "inspect":
        sv = measurements.get("inspect.summary_valid")
        if isinstance(sv, dict) and sv.get("status") == "measured":
            return bool(sv.get("value") is False)
        return None

    if st == "precal":
        rm = metrics_dict.get("rubric_meta")
        grades_cal_ms = True
        if isinstance(rm, dict) and isinstance(rm.get("display_rows"), list):
            obs_set = {
                str(r.get("observable") or "") for r in rm["display_rows"] if isinstance(r, dict)
            }
            if obs_set:
                grades_cal_ms = any(k.startswith("precal.calibrators_ms_") for k in obs_set)
        if not grades_cal_ms:
            in_rec = measurements.get("precal.merged_calibrator_rows")
            return bool(isinstance(in_rec, dict) and in_rec.get("status") == "missing_artifact")
        nr = measurements.get("precal.calibrators_ms_nrows")
        if isinstance(nr, dict):
            if nr.get("status") == "missing_artifact":
                return True
            return bool(
                nr.get("status") == "measured" and int(_as_float(nr.get("value")) or 0) == 0
            )
        return None

    if st == "calibrate":
        cd = measurements.get("cal.corrected_distinct_from_data")
        if isinstance(cd, dict):
            return bool(cd.get("status") == "missing_artifact" or cd.get("value") is False)
        return None

    if st in ("image", "image_cont"):
        vp = measurements.get("image.valid_product_count")
        if isinstance(vp, dict) and vp.get("status") == "measured":
            return int(_as_float(vp.get("value")) or 0) == 0
        return None

    if st == "report":
        rh = measurements.get("report.html_valid")
        if isinstance(rh, dict) and rh.get("status") == "measured":
            return bool(rh.get("value") is False)
        return None

    return None


def _deliverable_missing(
    stage: str,
    reward_present: bool,
    subscores: dict[str, float],
    metrics_dict: dict[str, Any] | None,
) -> bool:
    if not reward_present and not isinstance(metrics_dict, dict):
        return True

    m = metrics_dict if isinstance(metrics_dict, dict) else {}
    st_rec = _get_measurement_record(m, "proc.manifest_status")
    if st_rec is not None and str(st_rec.get("value") or "").strip().lower() == "failed":
        return True

    if stage == "e2e":
        # Only treat e2e deliverable as immediately missing at Priority 6 when Stage 1 (calibrate)
        # deliverable is missing; if calibration exists, evaluate Stage 1 physics tags before
        # checking whether Stage 2 (image) deliverable is missing.
        v02_cal_miss = _stage_deliverable_missing_v02("calibrate", m)
        if v02_cal_miss is not None:
            return v02_cal_miss
    else:
        v02_miss = _stage_deliverable_missing_v02(stage, m)
        if v02_miss is not None:
            return v02_miss

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
        return _image_set_missing(measured, m)

    if stage == "e2e":
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
    v02_img_miss = _stage_deliverable_missing_v02("image", metrics_dict)
    if v02_img_miss is not None:
        return v02_img_miss
    img_m = metrics_dict.get("image")
    if isinstance(img_m, dict) and isinstance(img_m.get("measured"), dict):
        return _image_set_missing(img_m["measured"], metrics_dict)
    meas = metrics_dict.get("measured")
    if isinstance(meas, dict) and isinstance(meas.get("image"), dict):
        return _image_set_missing(meas["image"], metrics_dict)
    return False


def _classify_physics_from_rubric_meta(
    metrics_dict: dict[str, Any],
    subscores: dict[str, float],
) -> str | None:
    """Self-describing physics failure classification from `metrics_dict["rubric_meta"]`."""
    rubric_meta = metrics_dict.get("rubric_meta")
    if not isinstance(rubric_meta, dict):
        return None
    meta_subs = rubric_meta.get("subscores")
    if not isinstance(meta_subs, list) or not meta_subs:
        return None

    stages_in_order: list[str] = []
    for entry in meta_subs:
        if isinstance(entry, dict) and isinstance(entry.get("stage"), str):
            st = entry["stage"]
            if st not in stages_in_order:
                stages_in_order.append(st)

    for stage_idx, st in enumerate(stages_in_order):
        # On multi-stage pipelines (e.g. e2e), if downstream Stage > 0 deliverable is completely
        # absent after upstream Stage 0 passed physics checks, classify as script_or_casa_crash.
        if stage_idx > 0:
            st_missing = _stage_deliverable_missing_v02(st, metrics_dict)
            if st_missing is True:
                return "script_or_casa_crash"

        st_entries: list[dict[str, Any]] = []
        for e in meta_subs:
            if not isinstance(e, dict) or e.get("stage") != st:
                continue
            w = _as_float(e.get("weight")) or 0.0
            if w <= 0.0:
                continue
            sid = str(e.get("id") or "")
            sc = subscores.get(sid, _as_float(e.get("score")))
            if sc is None:
                continue
            tag = str(e.get("failure_tag") or "")
            if tag not in _CATEGORY_PRIORITY:
                continue
            st_entries.append(
                {
                    "id": sid,
                    "score": sc,
                    "weight": w,
                    "tag": tag,
                    "symptom_only": bool(e.get("symptom_only")),
                }
            )

        # Evaluate severe (< 0.5) root-cause (`symptom_only == False`) subscores first in category
        # priority order, before falling back to `symptom_only == True` subscores.
        severe_roots = [x for x in st_entries if not x["symptom_only"] and x["score"] < 0.5]
        if severe_roots:
            severe_roots.sort(key=lambda x: (_CATEGORY_PRIORITY[x["tag"]], x["score"]))
            return str(severe_roots[0]["tag"])

        severe_symp = [x for x in st_entries if x["symptom_only"] and x["score"] < 0.5]
        if severe_symp:
            severe_symp.sort(key=lambda x: (_CATEGORY_PRIORITY[x["tag"]], x["score"]))
            return str(severe_symp[0]["tag"])

    # Clean-bowl override when rubric display trace marks `image.negative_bowl_ratio` failed
    rm_disp = rubric_meta.get("display_rows")
    if isinstance(rm_disp, list):
        for r_row in rm_disp:
            if (
                isinstance(r_row, dict)
                and r_row.get("observable") == "image.negative_bowl_ratio"
                and r_row.get("status") == "measured"
            ):
                r_sc = _as_float(r_row.get("score"))
                if r_sc is not None and r_sc <= 0.0:
                    return "beam_sampling_or_clean_bowl"

    # Moderate failure fallback (all subscores >= 0.5, composite < pass_reward_min or floor failed)
    all_pos: list[tuple[int, float, int, str]] = []
    for e in meta_subs:
        if not isinstance(e, dict):
            continue
        w = _as_float(e.get("weight")) or 0.0
        if w <= 0.0:
            continue
        sid = str(e.get("id") or "")
        sc = subscores.get(sid, _as_float(e.get("score")))
        tag = str(e.get("failure_tag") or "")
        if sc is None or sc >= 1.0 - 1e-9 or tag not in _CATEGORY_PRIORITY:
            continue
        symp_rank = 1 if bool(e.get("symptom_only")) else 0
        all_pos.append((symp_rank, sc, _CATEGORY_PRIORITY[tag], tag))

    if all_pos:
        all_pos.sort(key=lambda item: (item[1], item[0], item[2]))
        return all_pos[0][3]

    return None


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
    """Deterministically classify a trial into `'none'` or one of the 14 `FAILURE_CATEGORIES`."""
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

    # 3. invalid_abstention_or_halt (evaluated BEFORE missing_deliverable -> script_or_casa_crash)
    if _is_invalid_abstention_or_halt(metrics_dict):
        return "invalid_abstention_or_halt"

    # 4. raw_data_or_calflag_mutation (evaluated BEFORE missing_deliverable / bash_nonzero_exits)
    if _is_raw_data_or_calflag_mutation(metrics_dict):
        return "raw_data_or_calflag_mutation"

    # 5. unsupported_or_missing_claim (strictly Layer B Claim Auditor failures, never inspect triage)
    if _is_unsupported_or_missing_claim(metrics_dict):
        return "unsupported_or_missing_claim"

    # 6. script_or_casa_crash
    if (
        bool(exc_type)
        or missing_deliverable
        or (bash_nonzero_exits >= 3 and reward < 0.25)
        or (severe_casa_errors > 0 and reward < 0.25)
    ):
        return "script_or_casa_crash"

    # 7+. Self-describing v0.2 `rubric_meta` physics tags (zero dataset branches)
    if isinstance(metrics_dict, dict):
        v02_cat = _classify_physics_from_rubric_meta(metrics_dict, subscores)
        if v02_cat is not None:
            return v02_cat

    # Legacy v0.1 read-compat fallback when `rubric_meta` is absent
    model_data_sc = subscores.get("model_data_score")
    flux_bp_sc = subscores.get("flux_bandpass_score")
    fluxscale_sc = _get_first_float(subscores, "cal_fluxscale_score", "fluxscale_score")
    if (
        (stage == "inspect" and flux_bp_sc is not None and flux_bp_sc < 0.5)
        or (stage == "precal" and model_data_sc is not None and model_data_sc < 0.5)
        or (stage in ("calibrate", "e2e") and fluxscale_sc is not None and fluxscale_sc < 0.5)
    ):
        return "flux_model_or_fluxscale_error"

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

    if stage == "e2e" and _e2e_image_deliverable_missing(metrics_dict):
        return "script_or_casa_crash"

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
    bowl_ratio = _get_first_float(
        img_meas,
        "image.negative_bowl_ratio",
        "negative_bowl_ratio",
        "negative_model_bowl_ratio",
    )

    if stage in ("image", "e2e") and (
        (beam_sc is not None and beam_sc < 0.5)
        or (rms_sc is not None and rms_sc < 0.5)
        or (deconv_sc is not None and deconv_sc < 0.5)
        or (bowl_ratio is not None and bowl_ratio >= bowl_max)
    ):
        return "beam_sampling_or_clean_bowl"

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
