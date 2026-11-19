"""Truth-vs-measured physical comparison tables, summary report writer, and optional plots."""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.analysis.collect import (
    CANONICAL_ARMS,
    TrialRecord,
    _is_trial_directory,
    sanitize_json_tree,
    summarize_run_health,
)
from radio_bench.analysis.taxonomy import ALL_OUTCOME_LABELS
from radio_bench.config import reject_slash_tmp
from radio_bench.grading import PASS_REWARD_MIN
from radio_bench.grading.extractors.common import has_active_table_lock

_INSPECT_SCALAR_CAL_FIELDS: frozenset[str] = frozenset(
    {
        "flux_calibrator",
        "bandpass_calibrator",
        "phase_calibrator",
        "pol_leakage_calibrator",
        "inspect.flux_cal_list",
        "inspect.bandpass_cal_list",
        "inspect.phase_cal_list",
    }
)

_NON_CALTABLE_SUFFIXES: tuple[str, ...] = (
    ".ms",
    ".flagversions",
    ".image",
    ".residual",
    ".pb",
    ".model",
    ".pbcor",
    ".psf",
    ".sumwt",
    ".weight",
    ".mask",
    ".tt0",
    ".tt1",
    ".tt2",
    ".alpha",
)


def _atomic_write_text(target_path: Path, text: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".{target_path.name}.{os.getpid()}.tmp"
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_savefig(fig: Any, target_path: Path, plt_mod: Any) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".{target_path.name}.{os.getpid()}.tmp"
    try:
        fig.savefig(tmp_path, format="png")
        os.replace(tmp_path, target_path)
    finally:
        plt_mod.close(fig)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _md_cell(val: Any) -> str:
    s = str(val)
    s = s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    s = s.replace("\u2014", "--").replace("\u2013", "--")
    return s.replace("|", "\\|")


def _fmt_num(val: Any, digits: int = 4) -> str:
    if val is None or isinstance(val, bool):
        return "n/a"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            return "n/a"
        return f"{val:.{digits}f}"
    return str(val)


def _fmt_signed(val: Any, digits: int = 4) -> str:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        return "n/a"
    fval = float(val)
    if not math.isfinite(fval):
        return "n/a"
    return f"{fval:+.{digits}f}"


def _ordered_arms(records: list[dict[str, Any]]) -> list[str]:
    present = {str(r.get("arm")) for r in records if r.get("arm")}
    ordered = [a for a in CANONICAL_ARMS if a in present]
    extras = sorted(present - set(ordered))
    return [*ordered, *extras]


def _extract_stage_metric_specs(
    stage: str,
    truth: dict[str, Any],
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Return `(metric_label, truth_target_str, measured_lookup_path)` specs for a stage."""
    specs: list[tuple[str, str, tuple[str, ...]]] = []

    if stage == "inspect":
        if "flux_calibrator" in truth:
            specs.append(("flux_calibrator", str(truth["flux_calibrator"]), ("flux_calibrator",)))
        if "bandpass_calibrator" in truth:
            specs.append(
                ("bandpass_calibrator", str(truth["bandpass_calibrator"]), ("bandpass_calibrator",))
            )
        if "phase_calibrator" in truth:
            specs.append(
                ("phase_calibrator", str(truth["phase_calibrator"]), ("phase_calibrator",))
            )
        if "target_fields" in truth:
            specs.append(("target_fields", json.dumps(truth["target_fields"]), ("target_fields",)))
        if "dead_antennas" in truth:
            specs.append(("dead_antennas", json.dumps(truth["dead_antennas"]), ("dead_antennas",)))
        if "bandpass_min_elevation_deg" in truth:
            tol = truth.get("elevation_tol_deg", 0.25)
            specs.append(
                (
                    "bandpass_min_elevation_deg",
                    f"{_fmt_num(truth['bandpass_min_elevation_deg'], 2)} (+-{_fmt_num(tol, 2)})",
                    ("bandpass_min_elevation_deg",),
                )
            )
        return specs

    if stage == "precal":
        if "expected_j1331_model_amp_jy" in truth:
            specs.append(
                (
                    "j1331_model_amp_jy",
                    f"{_fmt_num(truth['expected_j1331_model_amp_jy'], 4)} Jy",
                    ("j1331_model_amp_jy",),
                )
            )
        if "flag_recall_min" in truth:
            specs.append(
                (
                    "must_flag_recall",
                    f">= {_fmt_num(truth['flag_recall_min'], 2)}",
                    ("must_flag_recall",),
                )
            )
        if "clean_fpr_max" in truth:
            specs.append(
                (
                    "clean_false_positive_rate",
                    f"< {_fmt_num(truth['clean_fpr_max'], 2)}",
                    ("clean_false_positive_rate",),
                )
            )
        return specs

    if stage == "calibrate":
        if "dead_antenna_flagged_min" in truth and truth["dead_antenna_flagged_min"] is not None:
            specs.append(
                (
                    "dead_antenna_flagged_fraction",
                    f">= {_fmt_num(truth['dead_antenna_flagged_min'], 2)}",
                    ("dead_antenna_flagged_fraction",),
                )
            )
        elif (
            "dead_antenna_flagged_fraction" in truth
            and truth["dead_antenna_flagged_fraction"] is not None
        ):
            specs.append(
                (
                    "dead_antenna_flagged_fraction",
                    f">= {_fmt_num(truth['dead_antenna_flagged_fraction'], 2)}",
                    ("dead_antenna_flagged_fraction",),
                )
            )
        if "expected_pcal_flux_jy" in truth:
            specs.append(
                (
                    "pcal_flux_jy",
                    f"{_fmt_num(truth['expected_pcal_flux_jy'], 4)} Jy",
                    ("pcal_flux_jy",),
                )
            )
        if "max_slope_deg_100mhz" in truth and truth["max_slope_deg_100mhz"] is not None:
            specs.append(
                (
                    "j1331_phase_slope_deg_per_100mhz",
                    f"<= {_fmt_num(truth['max_slope_deg_100mhz'], 2)}",
                    ("j1331_phase_slope_deg_per_100mhz",),
                )
            )
        if "max_chan_rms_deg" in truth and truth["max_chan_rms_deg"] is not None:
            specs.append(
                (
                    "j1331_chan_phase_rms_deg",
                    f"<= {_fmt_num(truth['max_chan_rms_deg'], 2)} deg",
                    ("j1331_chan_phase_rms_deg",),
                )
            )
        if truth.get("retention_min") is not None:
            rmin = truth["retention_min"]
            rmax = truth.get("retention_max")
            ret_target = (
                f">= {_fmt_num(rmin, 2)}"
                if rmax is None
                else f"[{_fmt_num(rmin, 2)}, {_fmt_num(rmax, 2)}]"
            )
            specs.append(
                (
                    "target_row_retention",
                    ret_target,
                    ("target_row_retention",),
                )
            )
        if truth.get("verifier_tclean_dr_min") is not None:
            specs.append(
                (
                    "verifier_tclean_dynamic_range",
                    f">= {_fmt_num(truth['verifier_tclean_dr_min'], 1)}",
                    ("verifier_tclean_dynamic_range",),
                )
            )
        if truth.get("max_coh_amp_err") is not None:
            specs.append(
                (
                    "coherent_amp_rel_err",
                    f"<= {_fmt_num(truth['max_coh_amp_err'], 3)}",
                    ("coherent_amp_rel_err",),
                )
            )
        coh_phs_key = (
            "max_coh_phs_deg"
            if truth.get("max_coh_phs_deg") is not None
            else ("coherent_phase_deg" if truth.get("coherent_phase_deg") is not None else None)
        )
        if coh_phs_key is not None:
            specs.append(
                (
                    "coherent_phase_deg",
                    f"<= {_fmt_num(truth[coh_phs_key], 2)} deg",
                    ("coherent_phase_deg",),
                )
            )
        return specs

    if stage == "image":
        ppb_lo = truth.get("ppb_min", truth.get("pixels_per_beam_min", 3.0))
        ppb_hi = truth.get("ppb_max", truth.get("pixels_per_beam_max", 7.0))
        if truth:
            specs.append(
                (
                    "pixels_per_beam",
                    f"[{_fmt_num(ppb_lo, 1)}, {_fmt_num(ppb_hi, 1)}]",
                    ("pixels_per_beam",),
                )
            )
        if "max_offsource_rms_jy" in truth and truth["max_offsource_rms_jy"] is not None:
            specs.append(
                (
                    "offsource_residual_rms_mjy_per_beam",
                    f"<= {_fmt_num(float(truth['max_offsource_rms_jy']) * 1e3, 4)} mJy/bm",
                    ("offsource_residual_rms_mjy_per_beam",),
                )
            )
        if "dynamic_range_min" in truth and truth["dynamic_range_min"] is not None:
            specs.append(
                (
                    "dynamic_range",
                    f">= {_fmt_num(truth['dynamic_range_min'], 1)}",
                    ("dynamic_range",),
                )
            )
        if "ext_flux_jy" in truth and truth["ext_flux_jy"] is not None:
            specs.append(
                (
                    "ext_integrated_flux_jy",
                    f"{_fmt_num(truth['ext_flux_jy'], 4)} Jy",
                    ("ext_integrated_flux_jy",),
                )
            )
        if "min_point_source_snr" in truth and truth["min_point_source_snr"] is not None:
            specs.append(
                (
                    "pt_snr_image",
                    f">= {_fmt_num(truth['min_point_source_snr'], 1)}",
                    ("pt_snr_image",),
                )
            )
        if "bowl_max" in truth or "max_negative_bowl_ratio" in truth or truth:
            bmax = truth.get("bowl_max", truth.get("max_negative_bowl_ratio", 0.12))
            specs.append(
                (
                    "negative_bowl_ratio",
                    f"< {_fmt_num(bmax, 2)}",
                    ("negative_bowl_ratio",),
                )
            )
        return specs

    if stage == "e2e":
        has_nested = isinstance(truth.get("calibration"), dict) or isinstance(
            truth.get("image"), dict
        )
        cal_truth = (
            truth.get("calibration")
            if isinstance(truth.get("calibration"), dict)
            else ({} if has_nested else truth)
        )
        img_truth = (
            truth.get("image")
            if isinstance(truth.get("image"), dict)
            else ({} if has_nested else truth)
        )
        for lbl, tgt, path_tup in _extract_stage_metric_specs("calibrate", cal_truth):
            specs.append((f"cal.{lbl}", tgt, ("calibration", *path_tup)))
        for lbl, tgt, path_tup in _extract_stage_metric_specs("image", img_truth):
            specs.append((f"img.{lbl}", tgt, ("image", *path_tup)))
        return specs

    return specs


def _lookup_nested(mapping: Any, path_tuple: tuple[str, ...]) -> Any:
    cur = mapping
    for key in path_tuple:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _normalize_measured_entry(stage: str, metric_label: str, val: Any) -> Any:
    if (
        stage == "inspect"
        and metric_label in _INSPECT_SCALAR_CAL_FIELDS
        and isinstance(val, list)
        and len(val) == 1
        and isinstance(val[0], str)
    ):
        return val[0]
    return val


def _summarize_measured_values(values: list[Any]) -> Any:
    present = [v for v in values if v is not None]
    if not present:
        return None
    nums = [
        float(v)
        for v in present
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    ]
    if len(nums) == len(present):
        return round(float(np.median(nums)), 6)

    # Categorical / list quantities (e.g. calibrator field name or dead antenna list): modal string.
    counts: dict[str, int] = {}
    for v in present:
        s = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
        counts[s] = counts.get(s, 0) + 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best


def _lookup_v02_observable_value(rec: dict[str, Any], obs_id: str) -> Any:
    measurements = rec.get("measurements")
    if isinstance(measurements, dict):
        obs_rec = measurements.get(obs_id)
        if isinstance(obs_rec, dict):
            if obs_rec.get("status") == "measured":
                return obs_rec.get("value")
            return None
    measured = rec.get("measured")
    if isinstance(measured, dict) and obs_id in measured:
        return measured[obs_id]
    return None


def _format_v02_truth_str(drow: dict[str, Any]) -> str:
    t_disp = str(drow.get("truth_display") or "").strip()
    unit = str(drow.get("unit") or "").strip()
    if unit and t_disp and unit.lower() not in t_disp.lower():
        return f"{t_disp} {unit}"
    return t_disp


def build_truth_vs_measured_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Truth-vs-Measured comparison rows per task across all present arms.

    Reads self-describing `v0.2` `rubric_meta["display_rows"]` + `measurements` directly
    without dataset branches, falling back to `_extract_stage_metric_specs` for `v0.1` records.
    """
    arms = _ordered_arms(records)
    tasks = sorted({str(r.get("base_task")) for r in records if r.get("base_task")})
    rows: list[dict[str, Any]] = []

    for task_id in tasks:
        t_recs = [r for r in records if str(r.get("base_task")) == task_id]
        if not t_recs:
            continue
        stage = str(t_recs[0].get("stage") or "unknown")
        suite = str(t_recs[0].get("suite") or "unknown")

        # Check for self-describing v0.2 `rubric_meta.display_rows` first (zero dataset branches).
        v02_meta: dict[str, Any] | None = None
        for r in t_recs:
            rm = r.get("rubric_meta")
            if (
                isinstance(rm, dict)
                and isinstance(rm.get("display_rows"), list)
                and rm["display_rows"]
            ):
                v02_meta = rm
                break

        pass_min = (
            float(v02_meta["pass_reward_min"])
            if (
                v02_meta is not None
                and isinstance(v02_meta.get("pass_reward_min"), (int, float))
                and not isinstance(v02_meta.get("pass_reward_min"), bool)
            )
            else PASS_REWARD_MIN
        )

        # Always include composite reward vs pass threshold so every task has >= 1 row.
        reward_by_arm: dict[str, Any] = {}
        for arm in arms:
            a_vals = [float(r.get("reward") or 0.0) for r in t_recs if str(r.get("arm")) == arm]
            reward_by_arm[arm] = round(float(np.mean(a_vals)), 4) if a_vals else None
        rows.append(
            {
                "base_task": task_id,
                "suite": suite,
                "stage": stage,
                "metric": "composite_reward",
                "truth": f">= {pass_min:.2f}",
                "by_arm": reward_by_arm,
            }
        )

        if v02_meta is not None:
            for drow in v02_meta["display_rows"]:
                if not isinstance(drow, dict):
                    continue
                obs_id = str(drow.get("observable") or drow.get("id") or "")
                if not obs_id:
                    continue
                d_stage = str(drow.get("stage") or stage)
                truth_str = _format_v02_truth_str(drow)
                by_arm_val: dict[str, Any] = {}
                any_present = False
                for arm in arms:
                    arm_vals = [
                        _normalize_measured_entry(
                            d_stage,
                            obs_id,
                            _lookup_v02_observable_value(r, obs_id),
                        )
                        for r in t_recs
                        if str(r.get("arm")) == arm
                    ]
                    summ = _summarize_measured_values(arm_vals)
                    if summ is not None:
                        any_present = True
                    by_arm_val[arm] = summ
                if any_present or truth_str:
                    rows.append(
                        {
                            "base_task": task_id,
                            "suite": suite,
                            "stage": stage,
                            "metric": obs_id,
                            "truth": truth_str,
                            "by_arm": by_arm_val,
                        }
                    )
            continue

        # Legacy v0.1 read-compat fallback when `rubric_meta` is absent.
        ref_truth: dict[str, Any] = {}
        for r in t_recs:
            tb = r.get("truth")
            if isinstance(tb, dict) and tb:
                ref_truth = tb
                break

        specs = _extract_stage_metric_specs(stage, ref_truth)
        if not specs and ref_truth:
            for k in sorted(ref_truth):
                val = ref_truth[k]
                if isinstance(val, (int, float, str)):
                    specs.append((k, _fmt_num(val, 4), (k,)))

        for metric_label, truth_str, lookup_path in specs:
            by_arm_val = {}
            any_present = False
            for arm in arms:
                arm_vals = [
                    _normalize_measured_entry(
                        stage,
                        metric_label,
                        _lookup_nested(r.get("measured"), lookup_path),
                    )
                    for r in t_recs
                    if str(r.get("arm")) == arm
                ]
                summ = _summarize_measured_values(arm_vals)
                if summ is not None:
                    any_present = True
                by_arm_val[arm] = summ
            if any_present or ref_truth:
                rows.append(
                    {
                        "base_task": task_id,
                        "suite": suite,
                        "stage": stage,
                        "metric": metric_label,
                        "truth": truth_str,
                        "by_arm": by_arm_val,
                    }
                )

    return sanitize_json_tree(rows)


def _arm_display_label(arm: str, cells: dict[str, Any], *, markdown: bool = True) -> str:
    c = cells.get(arm, {})
    has_non_e2e = any(not bool(v.get("e2e_only")) for v in cells.values())
    is_e2e_arm = arm == "11_orchestrated" or (bool(c.get("e2e_only")) and has_non_e2e)
    if markdown:
        return f"`{arm}` (e2e-only)" if is_e2e_arm else f"`{arm}`"
    return f"{arm} (e2e-only)" if is_e2e_arm else arm


def _fmt_parity_pass_k(pobj: dict[str, Any], k: int) -> str:
    s_v = pobj.get(f"small_pass_at_{k}")
    l_v = pobj.get(f"large_pass_at_{k}", pobj.get(f"frontier_pass_at_{k}"))
    gap = pobj.get(f"pass_at_{k}_gap")
    if s_v is None and l_v is None:
        return "n/a"
    s_s = _fmt_num(s_v, 3)
    l_s = _fmt_num(l_v, 3)
    if gap is not None:
        return f"{s_s} / {l_s} ({_fmt_signed(gap, 3)})"
    return f"{s_s} / {l_s}"


def render_summary_markdown(
    *,
    job_dirs: list[str],
    records: list[dict[str, Any]],
    truth_vs_measured: list[dict[str, Any]],
    factorial_summary: dict[str, Any],
    taxonomy_summary: dict[str, Any],
) -> str:
    """Render concise Markdown report with Truth-vs-Measured tables FIRST and zero em/en dashes."""
    arms = _ordered_arms(records)
    models = sorted({str(r.get("model_name")) for r in records if r.get("model_name")})
    suites = sorted({str(r.get("suite")) for r in records if r.get("suite")})

    lines: list[str] = [
        "# radio-bench Analysis Summary",
        "",
        f"- Total trials: `{len(records)}`",
        f"- Job directories: `{_md_cell(', '.join(job_dirs))}`",
        f"- Suites: `{_md_cell(', '.join(suites) if suites else 'none')}`",
        f"- Models: `{_md_cell(', '.join(models) if models else 'none')}`",
        f"- Capability arms: `{_md_cell(', '.join(arms) if arms else 'none')}`",
        "",
        "## 1. Truth vs Measured Physical Metrics",
        "",
    ]

    if truth_vs_measured and arms:
        header = ["Task", "Stage", "Metric", "Truth / Target", *arms]
        lines.append("| " + " | ".join(_md_cell(h) for h in header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in truth_vs_measured:
            by_arm = row.get("by_arm") or {}
            arm_cells = [_fmt_num(by_arm.get(a), 4) for a in arms]
            row_vals = [
                f"`{row['base_task']}`",
                str(row["stage"]),
                f"`{row['metric']}`",
                str(row["truth"]),
                *arm_cells,
            ]
            lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
    else:
        lines.append("No trials found.")
    lines.append("")

    # 2. Capability Arm Cell Summary
    lines.extend(["## 2. Capability Arm Cell Summary", ""])
    cells = factorial_summary.get("overall", {}).get("cells", {})
    if cells:
        c_hdr = [
            "Arm",
            "Trials",
            "Pass@1",
            "Pass@1 (Task-Eq)",
            "Wilson 95% CI",
            "Reward Mean",
            "Reward (Task-Eq)",
            "Reward Median",
            "Reward SD",
            "Mean Tokens",
            "Mean USD",
            "Mean USD (Task-Eq)",
            "Mean Wall (s)",
        ]
        lines.append("| " + " | ".join(_md_cell(h) for h in c_hdr) + " |")
        lines.append("| " + " | ".join(["---"] * len(c_hdr)) + " |")
        for arm in arms:
            c = cells.get(arm)
            if not c:
                continue
            w_ci = c.get("pass_at_1_wilson_ci") or [0.0, 0.0]
            p1_te = c.get("pass_at_1_task_equal", c.get("pass_at_1"))
            rew_te = c.get("reward_task_equal_mean", c.get("reward_mean"))
            usd_te = c.get("mean_usd_task_equal", c.get("mean_usd"))
            row_vals = [
                _arm_display_label(arm, cells, markdown=True),
                str(c["n_trials"]),
                _fmt_num(c["pass_at_1"], 4),
                _fmt_num(p1_te, 4),
                f"[{_fmt_num(w_ci[0], 4)}, {_fmt_num(w_ci[1], 4)}]",
                _fmt_num(c["reward_mean"], 4),
                _fmt_num(rew_te, 4),
                _fmt_num(c["reward_median"], 4),
                _fmt_num(c["reward_sd"], 4),
                _fmt_num(c["mean_total_tokens"], 1),
                f"${_fmt_num(c['mean_usd'], 4)}",
                f"${_fmt_num(usd_te, 4)}",
                _fmt_num(c["mean_wall_sec"], 2),
            ]
            lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
    else:
        lines.append("No arm cell data.")
    lines.append("")

    # 3. 2x2 Factorial Contrasts & Synergy
    lines.extend(["## 3. 2x2 Factorial Contrasts and Synergy", ""])
    contrasts = factorial_summary.get("overall", {}).get("contrasts", {})
    f_hdr = ["Contrast", "Formula", "Paired Units", "Estimate", "95% Bootstrap CI", "Status"]
    lines.append("| " + " | ".join(_md_cell(h) for h in f_hdr) + " |")
    lines.append("| " + " | ".join(["---"] * len(f_hdr)) + " |")
    for cname, cobj in contrasts.items():
        if cobj.get("status") == "ok":
            ci_str = f"[{_fmt_signed(cobj['ci_low'], 4)}, {_fmt_signed(cobj['ci_high'], 4)}]"
            est_str = _fmt_signed(cobj["estimate"], 4)
        else:
            ci_str = "n/a"
            est_str = "n/a"
        row_vals = [
            f"`{cname}`",
            f"`{cobj['formula']}`",
            str(cobj["n_paired_units"]),
            est_str,
            ci_str,
            str(cobj["status"]),
        ]
        lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
    lines.append("")

    # 4. Small + Full Stack vs Frontier Bare Parity
    lines.extend(["## 4. Small + Full Stack vs Frontier Bare Parity", ""])
    parity = factorial_summary.get("overall", {}).get("parity", {})
    p_hdr = [
        "Comparison",
        "Small Cell",
        "Frontier Cell",
        "Delta Reward",
        "95% Bootstrap CI",
        "Pass@1 Gap",
        "Pass@3 (S/L/Gap)",
        "Pass@5 (S/L/Gap)",
        "Cost Ratio",
        "Status",
    ]
    lines.append("| " + " | ".join(_md_cell(h) for h in p_hdr) + " |")
    lines.append("| " + " | ".join(["---"] * len(p_hdr)) + " |")
    for pkey, pobj in parity.items():
        if pobj.get("status") == "ok":
            est_str = _fmt_signed(pobj["estimate"], 4)
            ci_str = f"[{_fmt_signed(pobj['ci_low'], 4)}, {_fmt_signed(pobj['ci_high'], 4)}]"
            p1_str = _fmt_signed(pobj.get("pass_at_1_diff"), 4)
            p3_str = _fmt_parity_pass_k(pobj, 3)
            p5_str = _fmt_parity_pass_k(pobj, 5)
            cr_str = (
                f"{_fmt_num(pobj['cost_ratio'], 3)}x" if pobj["cost_ratio"] is not None else "n/a"
            )
        else:
            est_str = "n/a"
            ci_str = "n/a"
            p1_str = "n/a"
            p3_str = "n/a"
            p5_str = "n/a"
            cr_str = "n/a"
        s_tl = str(pobj.get("small_thinking_level") or "default").strip() or "default"
        f_tl = str(pobj.get("frontier_thinking_level") or "default").strip() or "default"
        s_disp = (
            str(pobj["small_model"]) if s_tl == "default" else f"{pobj['small_model']} ({s_tl})"
        )
        f_disp = (
            str(pobj["frontier_model"])
            if f_tl == "default"
            else f"{pobj['frontier_model']} ({f_tl})"
        )
        row_vals = [
            f"`{pkey}`",
            f"`{s_disp} @ {pobj['small_arm']}`",
            f"`{f_disp} @ {pobj['frontier_arm']}`",
            est_str,
            ci_str,
            p1_str,
            p3_str,
            p5_str,
            cr_str,
            str(pobj["status"]),
        ]
        lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
    lines.append("")

    # 5. Stratified (Suite x Model) Summary
    lines.extend(["## 5. Stratified (Suite x Model) Summary", ""])
    by_sm = factorial_summary.get("by_suite_model", {})
    if by_sm:
        sm_hdr = [
            "Suite",
            "Model",
            "Trials",
            "I_casa",
            "I_raw",
            "Delta_Tools_casa",
            "Delta_Skills_casa",
        ]
        lines.append("| " + " | ".join(_md_cell(h) for h in sm_hdr) + " |")
        lines.append("| " + " | ".join(["---"] * len(sm_hdr)) + " |")
        for _sm_key, sm_val in by_sm.items():
            sm_c = sm_val.get("contrasts", {})
            row_vals = [
                f"`{sm_val['suite']}`",
                f"`{sm_val['model_name']}`",
                str(sm_val["n_trials"]),
                _fmt_signed(sm_c.get("I_casa", {}).get("estimate"), 4),
                _fmt_signed(sm_c.get("I_raw", {}).get("estimate"), 4),
                _fmt_signed(sm_c.get("Delta_Tools_casa", {}).get("estimate"), 4),
                _fmt_signed(sm_c.get("Delta_Skills_casa", {}).get("estimate"), 4),
            ]
            lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
    else:
        lines.append("No stratified cells.")
    lines.append("")

    # 6. Deterministic Failure-Mode Taxonomy by Arm
    lines.extend(["## 6. Deterministic Failure-Mode Taxonomy by Arm", ""])
    tax_by_arm = taxonomy_summary.get("by_arm", {})
    if tax_by_arm and arms:
        t_hdr = ["Outcome / Failure Mode", *arms, "Total"]
        lines.append("| " + " | ".join(_md_cell(h) for h in t_hdr) + " |")
        lines.append("| " + " | ".join(["---"] * len(t_hdr)) + " |")
        overall_c = taxonomy_summary.get("overall_counts", {})
        for label in ALL_OUTCOME_LABELS:
            arm_counts = [str(tax_by_arm.get(a, {}).get("counts", {}).get(label, 0)) for a in arms]
            row_vals = [f"`{label}`", *arm_counts, str(overall_c.get(label, 0))]
            lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
    else:
        lines.append("No taxonomy records.")
    lines.append("")

    # 7. Cohort-Stratified Cost-Quality Pareto Frontier
    lines.extend(["## 7. Cohort-Stratified Cost-Quality Pareto Frontier", ""])
    pareto_block = factorial_summary.get("pareto", {})
    if isinstance(pareto_block, dict) and pareto_block:
        par_hdr = [
            "Cohort",
            "Common Tasks",
            "Cell",
            "Mean USD (Task-Eq)",
            "Reward (Task-Eq)",
            "Pareto (Reward)",
            "Boot Prob (Reward)",
            "Cost/Pass USD",
            "Pass@1 (Task-Eq)",
            "Pareto (Pass)",
            "Boot Prob (Pass)",
        ]
        any_pareto_rows = False
        table_lines: list[str] = [
            "| " + " | ".join(_md_cell(h) for h in par_hdr) + " |",
            "| " + " | ".join(["---"] * len(par_hdr)) + " |",
        ]
        for cohort_key in ("all_5_arms", "e2e_6_arms"):
            c_info = pareto_block.get(cohort_key)
            if not isinstance(c_info, dict) or c_info.get("status") != "ok":
                continue
            n_common = str(c_info.get("n_common_tasks", 0))
            for c_row in c_info.get("cells") or []:
                if not isinstance(c_row, dict):
                    continue
                any_pareto_rows = True
                cpp_v = c_row.get("cost_per_passing_run_usd_task_equal")
                cpp_str = f"${_fmt_num(cpp_v, 4)}" if cpp_v is not None else "n/a"
                row_vals = [
                    f"`{cohort_key}`",
                    n_common,
                    f"`{c_row.get('cell_id')}`",
                    f"${_fmt_num(c_row.get('mean_usd_task_equal'), 4)}",
                    _fmt_num(c_row.get("reward_task_equal_mean"), 4),
                    str(c_row.get("pareto_reward_status")),
                    _fmt_num(c_row.get("pareto_reward_boot_prob"), 4),
                    cpp_str,
                    _fmt_num(c_row.get("pass_at_1_task_equal"), 4),
                    str(c_row.get("pareto_pass_status")),
                    _fmt_num(c_row.get("pareto_pass_boot_prob"), 4),
                ]
                table_lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
        if any_pareto_rows:
            lines.extend(table_lines)
        else:
            lines.append("No eligible LLM Pareto frontier cells.")
    else:
        lines.append("No eligible LLM Pareto frontier cells.")
    lines.append("")

    md_text = "\n".join(lines)
    return md_text.replace("\u2014", "--").replace("\u2013", "--")


def _table_has_active_lock(table_dir: Path) -> bool:
    return bool(has_active_table_lock(table_dir))


def _discover_workspace_casa_tables(workspace_dir: Path, max_depth: int = 2) -> list[Path]:
    found: list[Path] = []

    def _walk(cur: Path, depth: int) -> None:
        if depth > max_depth or not cur.is_dir():
            return
        if (cur / "table.dat").is_file():
            found.append(cur)
            return
        try:
            children = sorted(cur.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            _walk(child, depth + 1)

    _walk(workspace_dir, 0)
    return found


def _import_casatools() -> Any | None:
    from radio_bench.config import load_bench_config

    cfg = load_bench_config().casaconfig_path
    if cfg.is_file():
        os.environ.setdefault("CASASITECONFIG", str(cfg))
    try:
        import casatools
    except ImportError:
        return None
    return casatools


def _collect_caltable_profiles(
    records: list[dict[str, Any]],
) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract `(arm, label, ant_ids, med_amp, med_phase_deg)` from `<trial_dir>/artifacts/workspace/`."""
    casatools = _import_casatools()
    if casatools is None:
        return []

    profiles: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for rec in records:
        t_dir_raw = rec.get("trial_dir")
        if not isinstance(t_dir_raw, str) or not t_dir_raw:
            continue
        ws_dir = Path(t_dir_raw) / "artifacts" / "workspace"
        if not ws_dir.is_dir():
            continue
        arm = str(rec.get("arm") or "00_none")
        for tbl_path in _discover_workspace_casa_tables(ws_dir):
            low_name = tbl_path.name.lower()
            if low_name.endswith(_NON_CALTABLE_SUFFIXES):
                continue
            if _table_has_active_lock(tbl_path):
                continue
            tb = casatools.table()
            opened = False
            try:
                opened = bool(
                    tb.open(
                        str(tbl_path),
                        nomodify=True,
                        lockoptions={"option": "usernoread"},
                    )
                )
                if not opened:
                    continue
                cols = set(tb.colnames())
                kw = tb.getkeywords()
                viscal = str(kw.get("VisCal") or "").strip() if isinstance(kw, dict) else ""
                # Prefer VisCal keyword inspection; fall back to caltable column signature
                # when synthetic test tables omit VisCal.
                has_cal_cols = ("CPARAM" in cols or "FPARAM" in cols) and "ANTENNA1" in cols
                if not viscal and not (has_cal_cols and "FLAG" in cols):
                    continue
                if not has_cal_cols or "FLAG" not in cols:
                    continue
                if int(tb.nrows()) <= 0:
                    continue
                is_complex = "CPARAM" in cols
                raw_p = np.asarray(tb.getcol("CPARAM" if is_complex else "FPARAM"))
                raw_f = np.asarray(tb.getcol("FLAG"), dtype=bool)
                ant1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int64)
            except Exception:
                continue
            finally:
                if opened:
                    with contextlib.suppress(Exception):
                        tb.close()
                with contextlib.suppress(Exception):
                    tb.done()

            if raw_p.shape != raw_f.shape or raw_p.ndim < 2 or raw_p.shape[-1] != ant1.size:
                continue

            unique_ants = np.unique(ant1)
            ant_out: list[int] = []
            amp_out: list[float] = []
            phs_out: list[float] = []
            for aid in unique_ants:
                row_sel = ant1 == aid
                sub_p = raw_p[..., row_sel]
                sub_f = raw_f[..., row_sel]
                valid = sub_p[~sub_f]
                if valid.size == 0:
                    continue
                if is_complex:
                    amps = np.abs(valid)
                    phs = np.degrees(np.angle(valid))
                else:
                    amps = np.abs(valid.astype(np.float64))
                    phs = valid.astype(np.float64)
                finite_m = np.isfinite(amps) & np.isfinite(phs)
                if not np.any(finite_m):
                    continue
                ant_out.append(int(aid))
                amp_out.append(float(np.median(amps[finite_m])))
                phs_out.append(float(np.median(phs[finite_m])))

            if ant_out:
                profiles.append(
                    (
                        arm,
                        tbl_path.name,
                        np.array(ant_out, dtype=np.int64),
                        np.array(amp_out, dtype=np.float64),
                        np.array(phs_out, dtype=np.float64),
                    )
                )
    return profiles


def _read_casa_image_plane(
    img_path: Path, casatools_mod: Any
) -> tuple[np.ndarray, np.ndarray] | None:
    if _table_has_active_lock(img_path):
        return None
    ia = casatools_mod.image()
    opened = False
    try:
        opened = bool(ia.open(str(img_path)))
        if not opened:
            return None
        arr = np.asarray(ia.getchunk(), dtype=np.float64)
        msk = np.asarray(ia.getchunk(getmask=True), dtype=bool)
        while arr.ndim > 2:
            arr = arr[..., 0]
            msk = msk[..., 0]
        return arr.copy(), msk.copy()
    except Exception:
        return None
    finally:
        if opened:
            with contextlib.suppress(Exception):
                ia.close()
        with contextlib.suppress(Exception):
            ia.done()


def _collect_residual_pixel_samples_mjy(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Read off-source `.residual` (+ `.pb`) pixels (mJy/beam) by arm from `<trial_dir>/artifacts/workspace/`."""
    casatools = _import_casatools()
    if casatools is None:
        return {}

    samples_by_arm: dict[str, list[np.ndarray]] = {}
    rng = np.random.default_rng(20260413)

    for rec in records:
        t_dir_raw = rec.get("trial_dir")
        if not isinstance(t_dir_raw, str) or not t_dir_raw:
            continue
        ws_dir = Path(t_dir_raw) / "artifacts" / "workspace"
        if not ws_dir.is_dir():
            continue
        arm = str(rec.get("arm") or "00_none")
        for tbl_path in _discover_workspace_casa_tables(ws_dir):
            name = tbl_path.name
            if name.endswith(".residual.tt0"):
                pb_candidates = [
                    tbl_path.parent / (name[: -len(".residual.tt0")] + ".pb.tt0"),
                    tbl_path.parent / (name[: -len(".residual.tt0")] + ".pb"),
                ]
            elif name.endswith(".residual"):
                pb_candidates = [
                    tbl_path.parent / (name[: -len(".residual")] + ".pb"),
                    tbl_path.parent / (name[: -len(".residual")] + ".pb.tt0"),
                ]
            else:
                continue

            res_plane = _read_casa_image_plane(tbl_path, casatools)
            if res_plane is None:
                continue
            res_arr, res_msk = res_plane
            valid_mask = res_msk & np.isfinite(res_arr)

            for pb_cand in pb_candidates:
                if pb_cand.is_dir() and (pb_cand / "table.dat").is_file():
                    pb_plane = _read_casa_image_plane(pb_cand, casatools)
                    if pb_plane is not None and pb_plane[0].shape == res_arr.shape:
                        pb_arr, pb_msk = pb_plane
                        valid_mask &= pb_msk & np.isfinite(pb_arr) & (pb_arr >= 0.20)
                    break

            pix_mjy = res_arr[valid_mask] * 1e3
            if pix_mjy.size == 0:
                continue
            # Exclude bright uncleaned on-source outliers (> 5 MAD) to isolate off-source noise/bowl.
            med = float(np.median(pix_mjy))
            mad_std = 1.4826 * float(np.median(np.abs(pix_mjy - med)))
            if mad_std > 0.0 and pix_mjy.size >= 32:
                inlier = pix_mjy[np.abs(pix_mjy - med) <= 5.0 * mad_std]
                if inlier.size >= 16:
                    pix_mjy = inlier
            if pix_mjy.size > 20_000:
                idx = rng.choice(pix_mjy.size, size=20_000, replace=False)
                pix_mjy = pix_mjy[idx]
            samples_by_arm.setdefault(arm, []).append(pix_mjy)

    return {arm: np.concatenate(arrs) for arm, arrs in samples_by_arm.items() if arrs}


def _collect_trial_residual_rms_mjy(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Fallback off-source residual RMS (mJy/beam) per arm from `measured` trial dicts."""
    rms_by_arm: dict[str, list[float]] = {}
    for rec in records:
        meas = rec.get("measured")
        if not isinstance(meas, dict):
            continue
        stage = str(rec.get("stage") or "")
        img_meas = (
            meas.get("image")
            if (stage == "e2e" or bool(rec.get("is_e2e"))) and isinstance(meas.get("image"), dict)
            else meas
        )
        val_mjy: float | None = None
        v02_rms = _lookup_v02_observable_value(rec, "image.offsource_residual_rms_jy")
        if (
            isinstance(v02_rms, (int, float))
            and not isinstance(v02_rms, bool)
            and math.isfinite(float(v02_rms))
        ):
            val_mjy = float(v02_rms) * 1e3
        else:
            for key, scale in (
                ("image.offsource_residual_rms_jy", 1e3),
                ("offsource_residual_rms_mjy_per_beam", 1.0),
                ("off_source_residual_rms_mjy_per_beam", 1.0),
                ("offsource_residual_rms_jy_per_beam", 1e3),
                ("off_source_residual_rms", 1e3),
            ):
                raw = img_meas.get(key)
                if (
                    isinstance(raw, (int, float))
                    and not isinstance(raw, bool)
                    and math.isfinite(float(raw))
                ):
                    val_mjy = float(raw) * scale
                    break
        if val_mjy is not None:
            arm = str(rec.get("arm") or "00_none")
            rms_by_arm.setdefault(arm, []).append(val_mjy)

    return {arm: np.array(vals, dtype=np.float64) for arm, vals in rms_by_arm.items() if vals}


def generate_diagnostic_plots(
    *,
    records: list[dict[str, Any]],
    factorial_summary: dict[str, Any],
    taxonomy_summary: dict[str, Any],
    plots_dir: Path,
) -> list[str]:
    """Generate optional diagnostic PNGs strictly inside `plots_dir` when `--plots` is requested."""
    reject_slash_tmp(Path(plots_dir), label="plots_dir")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    arms = _ordered_arms(records)
    cells = factorial_summary.get("overall", {}).get("cells", {})
    written: list[str] = []

    if not arms or not cells:
        return written

    x_labels = [_arm_display_label(a, cells, markdown=False) for a in arms]

    # Plot 1: Task-equal Reward mean and Pass@1 by capability arm.
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=140)
    x = np.arange(len(arms))
    width = 0.36
    r_means = [float(cells[a].get("reward_task_equal_mean", cells[a]["reward_mean"])) for a in arms]
    p1_vals = [float(cells[a].get("pass_at_1_task_equal", cells[a]["pass_at_1"])) for a in arms]
    ax.bar(x - width / 2, r_means, width=width, label="Task-Equal Mean Reward")
    ax.bar(x + width / 2, p1_vals, width=width, label="Task-Equal Pass@1")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score / Rate")
    ax.set_title("radio-bench Task-Equal Reward and Pass@1 by Capability Arm")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1_path = plots_dir / "reward_by_arm.png"
    _atomic_savefig(fig, p1_path, plt)
    written.append(str(p1_path))

    # Plot 2: Mean USD cost vs Mean Reward across paired 5-arm cells + Pareto staircase.
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    by_mt = factorial_summary.get("by_model_thinking", {})
    by_model = factorial_summary.get("by_model", {})
    pareto_5 = (factorial_summary.get("pareto") or {}).get("all_5_arms")
    has_legend_items = False
    p2_title = "Cost vs Reward Across Model x Arm Cells"

    p5_cells = (
        [c for c in (pareto_5.get("cells") or []) if isinstance(c, dict)]
        if isinstance(pareto_5, dict) and pareto_5.get("status") == "ok"
        else []
    )
    if p5_cells:
        n_common_t = int(pareto_5.get("n_common_tasks") or 0)
        p2_title = f"Cost vs Reward Across 5-Arm Paired Cells (n={n_common_t} common tasks)"
        p5_packs = {str(c.get("pack_id") or "ra_00a11ea") for c in p5_cells}
        multi_p5_packs = len(p5_packs) > 1
        series_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for c in p5_cells:
            gk = (
                str(c.get("pack_id") or "ra_00a11ea"),
                str(c.get("model_name") or ""),
                str(c.get("thinking_level") or "default"),
            )
            series_groups.setdefault(gk, []).append(c)

        arm_rank = {a: idx for idx, a in enumerate(arms)}
        for (pk, m_name, t_lvl), c_list in sorted(series_groups.items()):
            c_ordered = sorted(c_list, key=lambda d: arm_rank.get(str(d.get("arm")), 999))
            base_lbl = f"{pk}:{m_name}" if multi_p5_packs else m_name
            series_lbl = base_lbl if t_lvl == "default" else f"{base_lbl} ({t_lvl})"
            xs = [float(d["mean_usd_task_equal"]) for d in c_ordered]
            ys = [float(d["reward_task_equal_mean"]) for d in c_ordered]
            a_lbls = [str(d["arm"]) for d in c_ordered]
            if xs:
                has_legend_items = True
                ax2.scatter(xs, ys, s=45, label=series_lbl)
                for xv, yv, al in zip(xs, ys, a_lbls, strict=True):
                    ax2.annotate(
                        al, (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points"
                    )
    elif isinstance(by_mt, dict) and by_mt:
        mt_packs = {
            str(v.get("pack_id") or "ra_00a11ea") for v in by_mt.values() if isinstance(v, dict)
        }
        multi_mt_packs = len(mt_packs) > 1
        for _mt_key, mt_data in by_mt.items():
            if not isinstance(mt_data, dict):
                continue
            m_name = str(mt_data.get("model_name") or "")
            if m_name.strip().lower() == "oracle":
                continue
            pk = str(mt_data.get("pack_id") or "ra_00a11ea")
            t_lvl = str(mt_data.get("thinking_level") or "default")
            base_lbl = f"{pk}:{m_name}" if multi_mt_packs else m_name
            series_lbl = base_lbl if t_lvl == "default" else f"{base_lbl} ({t_lvl})"
            m_cells = mt_data.get("cells", {})
            xs = [
                float(m_cells[a].get("mean_usd_task_equal", m_cells[a]["mean_usd"]))
                for a in arms
                if a in m_cells and a != "oracle"
            ]
            ys = [
                float(m_cells[a].get("reward_task_equal_mean", m_cells[a]["reward_mean"]))
                for a in arms
                if a in m_cells and a != "oracle"
            ]
            a_lbls = [a for a in arms if a in m_cells and a != "oracle"]
            if xs:
                has_legend_items = True
                ax2.scatter(xs, ys, s=45, label=series_lbl)
                for xv, yv, al in zip(xs, ys, a_lbls, strict=True):
                    ax2.annotate(
                        al, (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points"
                    )
    else:
        for model_name, m_data in by_model.items():
            if str(model_name).strip().lower() == "oracle":
                continue
            m_cells = m_data.get("cells", {})
            xs = [
                float(m_cells[a].get("mean_usd_task_equal", m_cells[a]["mean_usd"]))
                for a in arms
                if a in m_cells and a != "oracle"
            ]
            ys = [
                float(m_cells[a].get("reward_task_equal_mean", m_cells[a]["reward_mean"]))
                for a in arms
                if a in m_cells and a != "oracle"
            ]
            a_lbls = [a for a in arms if a in m_cells and a != "oracle"]
            if xs:
                has_legend_items = True
                ax2.scatter(xs, ys, s=45, label=model_name)
                for xv, yv, al in zip(xs, ys, a_lbls, strict=True):
                    ax2.annotate(
                        al, (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points"
                    )

    if p5_cells:
        frontier_pts = [
            (
                float(c["mean_usd_task_equal"]),
                float(c["reward_task_equal_mean"]),
            )
            for c in p5_cells
            if bool(c.get("is_pareto_reward"))
        ]
        frontier_pts.sort(key=lambda pt: (pt[0], pt[1]))
        if frontier_pts:
            fx = [pt[0] for pt in frontier_pts]
            fy = [pt[1] for pt in frontier_pts]
            has_legend_items = True
            if len(frontier_pts) >= 2:
                ax2.step(
                    fx,
                    fy,
                    where="post",
                    linestyle="--",
                    linewidth=1.5,
                    color="black",
                    alpha=0.75,
                    label=f"Pareto Frontier (5-arm, n={pareto_5.get('n_common_tasks', 0)})",
                )
                ax2.scatter(
                    fx,
                    fy,
                    s=80,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.2,
                )
            else:
                ax2.scatter(
                    fx,
                    fy,
                    s=80,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.2,
                    label=f"Pareto Frontier (5-arm, n={pareto_5.get('n_common_tasks', 0)})",
                )

    ax2.set_xlabel("Task-Equal Mean List-Price Cost per Trial (USD)")
    ax2.set_ylabel("Task-Equal Mean Composite Reward")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title(p2_title)
    if has_legend_items:
        ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    p2_path = plots_dir / "cost_vs_reward.png"
    _atomic_savefig(fig2, p2_path, plt)
    written.append(str(p2_path))

    # Plot 3: Stacked failure taxonomy by arm.
    fig3, ax3 = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    tax_by_arm = taxonomy_summary.get("by_arm", {})
    bottom = np.zeros(len(arms), dtype=np.float64)
    for label in ALL_OUTCOME_LABELS:
        vals = np.array(
            [float(tax_by_arm.get(a, {}).get("counts", {}).get(label, 0)) for a in arms],
            dtype=np.float64,
        )
        if np.any(vals > 0):
            ax3.bar(x, vals, bottom=bottom, label=label)
            bottom += vals
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, rotation=20, ha="right")
    ax3.set_ylabel("Trial Count")
    ax3.set_title("Deterministic Outcome and Failure-Mode Counts by Arm")
    ax3.legend(loc="best", fontsize=7)
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    p3_path = plots_dir / "failure_taxonomy_by_arm.png"
    _atomic_savefig(fig3, p3_path, plt)
    written.append(str(p3_path))

    # Plot 4 (SPEC.md Section 5.4a): Per-antenna bandpass/gain amplitude and phase overlay.
    cal_profiles = _collect_caltable_profiles(records)
    if cal_profiles:
        fig4, (ax4_amp, ax4_phs) = plt.subplots(2, 1, figsize=(8.5, 6.0), dpi=140, sharex=True)
        seen_labels: set[str] = set()
        for arm, tbl_name, ant_ids, med_amp, med_phs in cal_profiles:
            lbl = f"{arm} ({tbl_name})"
            plot_lbl = lbl if lbl not in seen_labels else None
            seen_labels.add(lbl)
            ax4_amp.plot(ant_ids, med_amp, marker="o", markersize=3.5, alpha=0.8, label=plot_lbl)
            ax4_phs.plot(ant_ids, med_phs, marker="o", markersize=3.5, alpha=0.8, label=plot_lbl)
        ax4_amp.set_ylabel("Median |Gain|")
        ax4_amp.set_title("Per-Antenna Calibration Amplitude and Phase Overlay")
        ax4_amp.grid(alpha=0.3)
        if len(seen_labels) <= 12:
            ax4_amp.legend(loc="best", fontsize=7)
        ax4_phs.set_xlabel("Antenna ID")
        ax4_phs.set_ylabel("Median Phase / Param (deg)")
        ax4_phs.grid(alpha=0.3)
        fig4.tight_layout()
        p4_path = plots_dir / "caltable_amp_phase_overlay.png"
        _atomic_savefig(fig4, p4_path, plt)
        written.append(str(p4_path))

    # Plot 5 (SPEC.md Section 5.4b): Off-source residual histogram from .residual images or RMS metrics.
    pix_samples = _collect_residual_pixel_samples_mjy(records)
    if pix_samples:
        fig5, ax5 = plt.subplots(figsize=(8.0, 4.5), dpi=140)
        for arm in arms:
            if arm in pix_samples and pix_samples[arm].size > 0:
                ax5.hist(
                    pix_samples[arm],
                    bins=40,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=f"{arm} (N={pix_samples[arm].size})",
                )
        ax5.set_xlabel("Off-Source Residual Pixel Flux Density (mJy/beam)")
        ax5.set_ylabel("Probability Density")
        ax5.set_title("Off-Source Residual Distribution by Capability Arm")
        ax5.legend(loc="best", fontsize=8)
        ax5.grid(alpha=0.3)
        fig5.tight_layout()
        p5_path = plots_dir / "offsource_residual_hist.png"
        _atomic_savefig(fig5, p5_path, plt)
        written.append(str(p5_path))
    else:
        rms_samples = _collect_trial_residual_rms_mjy(records)
        if rms_samples:
            fig5, ax5 = plt.subplots(figsize=(8.0, 4.5), dpi=140)
            all_vals = np.concatenate(list(rms_samples.values()))
            bins = min(20, max(5, int(all_vals.size)))
            for arm in arms:
                if arm in rms_samples and rms_samples[arm].size > 0:
                    ax5.hist(
                        rms_samples[arm],
                        bins=bins,
                        alpha=0.6,
                        label=f"{arm} (n={rms_samples[arm].size})",
                    )
            ax5.set_xlabel("Off-Source Residual RMS (mJy/beam)")
            ax5.set_ylabel("Trial Count")
            ax5.set_title("Off-Source Residual RMS Distribution Across Trials")
            ax5.legend(loc="best", fontsize=8)
            ax5.grid(alpha=0.3)
            fig5.tight_layout()
            p5_path = plots_dir / "offsource_residual_hist.png"
            _atomic_savefig(fig5, p5_path, plt)
            written.append(str(p5_path))

    return written


_HUMAN_ARM_ORDER: tuple[str, ...] = (
    "00_none",
    "01_skills_casa",
    "01_skills_raw",
    "10_tools",
    "11_full",
    "11_orchestrated",
)

_HUMAN_ARM_TITLES: dict[str, str] = {
    "00_none": "Bare CASA (No Guides, No MCP Tools)",
    "01_skills_casa": "Workflow Guides Only (CASA)",
    "01_skills_raw": "Workflow Guides Only (Raw)",
    "10_tools": "MCP Tools Only (No Guides)",
    "11_full": "Guides + MCP Tools (Single Agent)",
    "11_orchestrated": "Guides + MCP Tools (Orchestrated Subagents)",
}

_HUMAN_SUITE_TITLES: dict[str, str] = {
    "3c391": "VLA 3C391 C-band 7-Pointing Continuum Mosaic (4.6 GHz)",
    "r3c391": "VLA 3C391 C-band 7-Pointing Continuum Mosaic (4.6 GHz)",
    "suite_r": "VLA 3C391 C-band 7-Pointing Continuum Mosaic (4.6 GHz)",
    "vla_3c391_cband": "VLA 3C391 C-band 7-Pointing Continuum Mosaic (4.6 GHz)",
    "synth": "Synthetic VLA D-Configuration C-band Continuum",
    "suite_s": "Synthetic VLA D-Configuration C-band Continuum",
    "synth_vla_d": "Synthetic VLA D-Configuration C-band Continuum",
}

_HUMAN_STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("inspect", "1. Observation Inspection"),
    ("precal", "2. Pre-Calibration"),
    ("calibrate", "3. Calibration"),
    ("postcal", "3b. Post-Calibration Flagging"),
    ("polcal", "3c. Polarization Calibration"),
    ("image", "4. Continuum Imaging"),
    ("image_cube", "4b. Spectral-Line Cube Imaging"),
    ("selfcal", "4c. Self-Calibration"),
    ("report", "4d. Reduction Report"),
    ("e2e", "5. Full Start-to-Finish Reduction"),
)

_HUMAN_STAGE_CHECKS: dict[str, str] = {
    "inspect": (
        "Checks `/workspace/inspect_summary.json` (or `stage_output.json`) against MS metadata "
        "for primary flux calibrator, bandpass calibrator, secondary phase calibrator, science "
        "target pointings, dead antennas, and minimum bandpass-calibrator elevation (deg)."
    ),
    "precal": (
        "Checks `/workspace/input.ms` and `/workspace/calibrators.ms` for restored scan intents, "
        "calibrator split integrity, Perley-Butler flux-standard `MODEL_DATA` amplitude (Jy), "
        "and calibrator vs target RFI/quack flag recall (%) without over-flagging clean data (%)."
    ),
    "calibrate": (
        "Checks calibrated `/workspace/input.ms` (`CORRECTED_DATA`) and caltables for dead-antenna "
        "flagging, target row retention (%), bandpass phase slope (deg/100MHz) and channel RMS (deg), "
        "bootstrapped phase-calibrator flux density (Jy), and calibrated target dynamic range."
    ),
    "image": (
        "Checks delivered Stokes I continuum image products (`.image`, `.image.pbcor`, `.pb`, "
        "`.residual`, `.model`) for synthesized beam sampling (pixels/beam), primary-beam-masked "
        "off-source residual RMS (mJy/beam), continuum dynamic range, integrated flux (Jy), and "
        "absence of negative clean-bowl artifacts."
    ),
    "e2e": (
        "Checks the complete start-to-finish reduction from raw `input.ms` through calibrated "
        "`CORRECTED_DATA`, calibration tables, and final primary-beam-corrected Stokes I continuum "
        "mosaic on disk."
    ),
}

_FAILURE_MODE_HUMAN: dict[str, str] = {
    "lock_or_timeout": "table lock or timeout",
    "tool_protocol_error": "malformed tool call",
    "invalid_abstention_or_halt": "invalid halt on solvable data",
    "raw_data_or_calflag_mutation": "mutated raw DATA column",
    "unsupported_or_missing_claim": "unsupported metric claim",
    "script_or_casa_crash": "missing deliverable or CASA error",
    "flux_model_or_fluxscale_error": "flux scale error",
    "skipped_delay_or_bandpass": "residual delay/bandpass error",
    "flagging_recall_or_overflag": "RFI recall or overflagging",
    "beam_sampling_or_clean_bowl": "beam sampling or clean bowl",
    "postcal_clip_or_uv_bias": "post-cal clip or UV bias",
    "polcal_feed_or_parang_error": "polarization feed/parang error",
    "cube_baseline_or_frame_error": "cube contsub or frame error",
    "selfcal_divergence_or_shift": "self-cal divergence or shift",
}

_ROLE_DISPLAY_ORDER: tuple[str, ...] = (
    "orchestrator",
    "main",
    "inspect-stage",
    "precal-stage",
    "calibrate-stage",
    "postcal-stage",
    "polcal-stage",
    "image-stage",
    "selfcal-stage",
)


def _human_suite_title(suite_raw: str) -> str:
    key = suite_raw.strip()
    low = key.lower()
    if low in _HUMAN_SUITE_TITLES:
        return _HUMAN_SUITE_TITLES[low]
    pretty = key.replace("_", " ").strip()
    return f"{pretty} ({key})" if pretty.lower() != low else key


def _human_arm_title(arm: str) -> str:
    return _HUMAN_ARM_TITLES.get(arm, arm)


def _human_stage_title(stage: str) -> str:
    norm = "image" if stage == "image_cont" else stage
    for st_key, st_lbl in _HUMAN_STAGE_ORDER:
        if st_key == norm:
            return st_lbl
    return stage.replace("_", " ").title()


def _ordered_human_arms(records: Sequence[dict[str, Any]]) -> list[str]:
    present = {str(r.get("arm")) for r in records if r.get("arm")}
    ordered = [a for a in _HUMAN_ARM_ORDER if a in present]
    extras = sorted(present - set(ordered))
    return [*ordered, *extras]


def _ordered_human_stages(records: Sequence[dict[str, Any]]) -> list[str]:
    present = {
        ("image" if str(r.get("stage")) == "image_cont" else str(r.get("stage")))
        for r in records
        if r.get("stage")
    }
    ordered = [st for st, _lbl in _HUMAN_STAGE_ORDER if st in present]
    extras = sorted(present - set(ordered))
    return [*ordered, *extras]


def _get_obs_num(rec: dict[str, Any], *keys: str) -> float | None:
    measurements = rec.get("measurements")
    if isinstance(measurements, dict):
        for k in keys:
            m_rec = measurements.get(k)
            if isinstance(m_rec, dict) and m_rec.get("status") == "measured":
                val = m_rec.get("value")
                if (
                    isinstance(val, (int, float))
                    and not isinstance(val, bool)
                    and math.isfinite(float(val))
                ):
                    return float(val)
    measured = rec.get("measured")
    if isinstance(measured, dict):
        for k in keys:
            if k in measured:
                val = measured.get(k)
                if (
                    isinstance(val, (int, float))
                    and not isinstance(val, bool)
                    and math.isfinite(float(val))
                ):
                    return float(val)
        for blk_k in ("calibration", "image"):
            blk = measured.get(blk_k)
            if isinstance(blk, dict):
                for k in keys:
                    if k in blk:
                        val = blk.get(k)
                        if (
                            isinstance(val, (int, float))
                            and not isinstance(val, bool)
                            and math.isfinite(float(val))
                        ):
                            return float(val)
    return None


def _get_obs_any(rec: dict[str, Any], *keys: str) -> Any:
    measurements = rec.get("measurements")
    if isinstance(measurements, dict):
        for k in keys:
            m_rec = measurements.get(k)
            if isinstance(m_rec, dict) and m_rec.get("status") == "measured":
                return m_rec.get("value")
    measured = rec.get("measured")
    if isinstance(measured, dict):
        for k in keys:
            if k in measured and measured[k] is not None:
                return measured[k]
        for blk_k in ("calibration", "image"):
            blk = measured.get(blk_k)
            if isinstance(blk, dict):
                for k in keys:
                    if k in blk and blk[k] is not None:
                        return blk[k]
    return None


def _get_obs_bool(rec: dict[str, Any], *keys: str) -> bool | None:
    val = _get_obs_any(rec, *keys)
    return val if isinstance(val, bool) else None


def _fmt_str_or_list(val: Any) -> str:
    if isinstance(val, list):
        return ", ".join(str(x) for x in val)
    return str(val)


def _provider_drop_label(rec: dict[str, Any], *, include_turn: bool = True) -> str:
    p_kind = str(rec.get("provider_error_kind") or "")
    p_code = rec.get("provider_status_code")
    turns = int(rec.get("n_turns") or 0)
    turn_sfx = f" at turn {turns}" if (include_turn and turns > 0) else ""

    if p_kind == "tool_call_parse_400":
        return f"malformed tool call HTTP 400{turn_sfx}"
    if p_kind == "rate_limit_429":
        return f"upstream API {p_code or 429} rate limit{turn_sfx}"
    if p_kind == "upstream_timeout_504":
        return f"upstream API {p_code or 504} timeout{turn_sfx}"
    if p_kind == "upstream_connection_or_5xx":
        return f"upstream API {p_code or 500} drop{turn_sfx}"
    if p_code is not None:
        return f"upstream API {p_code} error{turn_sfx}"
    return f"upstream API drop{turn_sfx}"


def _output_token_limit_label(rec: dict[str, Any], *, include_turn: bool = True) -> str:
    turns = int(rec.get("n_turns") or 0)
    turn_sfx = f" at turn {turns}" if (include_turn and turns > 0) else ""
    return f"hit output-token limit during reasoning{turn_sfx}"


def _diagnose_trial_cell_reason(rec: dict[str, Any]) -> tuple[str, str]:
    """Return `('PASS' | 'FAIL' | 'INCOMPLETE', plain_english_reason)` for one trial."""
    if int(rec.get("passed") or 0) == 1:
        return "PASS", ""

    rew = float(rec.get("reward") or 0.0)
    exit_st = str(rec.get("agent_exit_status") or "ok")
    stage = str(rec.get("stage") or "")
    fmode = str(rec.get("failure_mode") or "script_or_casa_crash")
    subs = rec.get("subscores") if isinstance(rec.get("subscores"), dict) else {}

    if exit_st == "provider_api_error" and rew < 0.25:
        return "INCOMPLETE", _provider_drop_label(rec, include_turn=True)

    if exit_st == "output_token_limit":
        corr_dist_chk = _get_obs_bool(
            rec, "cal.corrected_distinct_from_data", "corrected_distinct_from_data"
        )
        valid_prod_chk = _get_obs_num(rec, "image.valid_product_count", "image_sets_found")
        missing_deliv = (
            rew == 0.0
            or rew < 0.25
            or (stage in ("calibrate", "e2e") and corr_dist_chk is False)
            or (
                stage in ("image", "image_cont", "e2e")
                and valid_prod_chk is not None
                and valid_prod_chk <= 0.0
            )
        )
        if missing_deliv:
            return "INCOMPLETE", _output_token_limit_label(rec, include_turn=True)

    reason = ""
    if stage == "inspect":
        fb_sc = float(subs.get("flux_bandpass_score", 1.0))
        el_sc = float(subs.get("elevation_score", 1.0))
        if fb_sc < 1.0 and el_sc < 0.5:
            reason = "wrong bandpass calibrator and elevation"
        elif fb_sc < 1.0:
            reason = "wrong flux/bandpass calibrator"
        elif el_sc < 0.5:
            reason = "wrong bandpass elevation"
        elif subs.get("dead_antenna_score", 1.0) < 0.5:
            reason = "missed dead antenna"
        elif subs.get("phase_vs_target_score", 1.0) < 0.5:
            reason = "calibrator/target role confusion"
    elif stage == "precal":
        if subs.get("calibrators_ms_score", 1.0) < 0.5 and rew < 0.25:
            reason = "missing calibrators.ms"
        elif subs.get("model_data_score", 1.0) < 0.5 and fmode == "flux_model_or_fluxscale_error":
            reason = "wrong flux calibrator model"
        elif subs.get("clean_fpr_score", 1.0) < 0.5:
            reason = "clean data overflagged"
        elif subs.get("flag_recall_score", 1.0) < 0.5:
            reason = "incomplete target/calibrator RFI flagging"
    elif stage in ("calibrate", "e2e") and fmode in (
        "flux_model_or_fluxscale_error",
        "skipped_delay_or_bandpass",
        "flagging_recall_or_overflag",
    ):
        corr_dist = _get_obs_bool(
            rec, "cal.corrected_distinct_from_data", "corrected_distinct_from_data"
        )
        bp_cal_ok = _get_obs_bool(rec, "cal.bandpass_cal_calibrated", "bandpass_cal_calibrated")
        if corr_dist is False:
            reason = "CORRECTED_DATA not populated"
        elif fmode == "flux_model_or_fluxscale_error":
            reason = "fluxscale bootstrap error"
        elif fmode == "skipped_delay_or_bandpass":
            if bp_cal_ok is False:
                reason = "bandpass calibrator left uncalibrated"
            else:
                bp_sc = subs.get(
                    "cal_bandpass_flatness_score", subs.get("bandpass_flatness_score", 1.0)
                )
                reason = (
                    "residual delay/bandpass phase slope"
                    if float(bp_sc) < 0.5
                    else "degraded target visibility fidelity"
                )
        else:
            reason = "dead antenna or target overflagging"
    elif stage == "calibrate":
        corr_dist = _get_obs_bool(
            rec, "cal.corrected_distinct_from_data", "corrected_distinct_from_data"
        )
        if corr_dist is False and exit_st != "provider_api_error":
            reason = "CORRECTED_DATA not populated"
    elif stage in ("image", "e2e") and fmode == "beam_sampling_or_clean_bowl":
        beam_sc = float(subs.get("img_beam_sampling_score", subs.get("beam_sampling_score", 1.0)))
        deconv_sc = float(
            subs.get("img_deconv_and_bowl_score", subs.get("deconv_and_bowl_score", 1.0))
        )
        rms_sc = float(subs.get("img_residual_rms_score", subs.get("residual_rms_score", 1.0)))
        bowl_v = _get_obs_num(rec, "image.negative_bowl_ratio", "negative_bowl_ratio")
        if deconv_sc < 0.85 or (bowl_v is not None and bowl_v >= 0.12):
            reason = "clean bowl"
        elif beam_sc < 0.5:
            reason = "bad beam sampling"
        elif rms_sc < 0.5:
            reason = "elevated residual RMS"
        else:
            reason = "clean bowl or beam sampling"

    if not reason:
        if exit_st == "provider_api_error":
            reason = _provider_drop_label(rec, include_turn=True)
        elif exit_st == "output_token_limit":
            reason = _output_token_limit_label(rec, include_turn=True)
        else:
            reason = _FAILURE_MODE_HUMAN.get(fmode, fmode.replace("_", " "))

    if exit_st == "agent_timeout" and "timeout" not in reason.lower():
        reason = f"{reason} + late timeout"
    elif exit_st == "output_token_limit" and "output-token limit" not in reason.lower():
        reason = f"{reason} + {_output_token_limit_label(rec, include_turn=False)}"
    elif (
        exit_st == "provider_api_error"
        and "upstream api" not in reason.lower()
        and "400" not in reason
    ):
        reason = f"{reason} + {_provider_drop_label(rec, include_turn=False)}"

    return "FAIL", reason


def _format_scorecard_cell(cell_recs: list[dict[str, Any]], *, arm: str, stage: str) -> str:
    if not cell_recs:
        if arm == "11_orchestrated" and stage != "e2e":
            return "n/a (full reduction only)"
        return "not run"

    if len(cell_recs) == 1:
        r = cell_recs[0]
        rew = float(r.get("reward") or 0.0)
        mins = float(r.get("agent_wall_sec") or r.get("total_wall_sec") or 0.0) / 60.0
        tag, reason = _diagnose_trial_cell_reason(r)
        if tag == "PASS":
            return f"PASS {rew:.2f} ({mins:.1f}m)"
        return f"{tag} {rew:.2f} -- {reason} ({mins:.1f}m)"

    n = len(cell_recs)
    k = sum(int(r.get("passed") or 0) for r in cell_recs)
    mean_rew = float(np.mean([float(r.get("reward") or 0.0) for r in cell_recs]))
    mean_min = float(
        np.mean(
            [
                float(r.get("agent_wall_sec") or r.get("total_wall_sec") or 0.0) / 60.0
                for r in cell_recs
            ]
        )
    )
    if k == n:
        return f"{k}/{n} PASS, mean {mean_rew:.2f} ({mean_min:.1f}m)"

    reason_counts: dict[str, int] = {}
    n_inc = 0
    for r in cell_recs:
        tag, rsn = _diagnose_trial_cell_reason(r)
        if tag == "INCOMPLETE":
            n_inc += 1
        if tag != "PASS" and rsn:
            reason_counts[rsn] = reason_counts.get(rsn, 0) + 1

    top_rsn = (
        sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if reason_counts
        else "failed"
    )
    inc_note = (
        f", {n_inc} incomplete" if (n_inc > 0 and "upstream api" not in top_rsn.lower()) else ""
    )
    return f"{k}/{n} PASS, mean {mean_rew:.2f} -- {top_rsn}{inc_note} ({mean_min:.1f}m)"


def _load_diffs_index(diffs_dir: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if diffs_dir is None or not Path(diffs_dir).is_dir():
        return out
    try:
        json_files = sorted(Path(diffs_dir).glob("*.json"))
    except OSError:
        return out
    for jf in json_files:
        try:
            parsed = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        tname = str(parsed.get("trial_name") or jf.stem).strip()
        if tname:
            out[tname] = parsed
            out.setdefault(jf.stem, parsed)
    return out


def _format_diff_snippet(diff_doc: dict[str, Any] | None) -> str:
    if not isinstance(diff_doc, dict) or not diff_doc:
        return ""
    parts: list[str] = []
    vis = diff_doc.get("visibilities")
    if isinstance(vis, dict) and vis.get("status") == "ok":
        ov = vis.get("overall") if isinstance(vis.get("overall"), dict) else vis
        jac = ov.get("flag_jaccard")
        vnrmse = ov.get("parallel_hand_nrmse")
        vphs = ov.get("parallel_hand_phase_rms_deg")
        v_bits: list[str] = []
        if isinstance(jac, (int, float)) and math.isfinite(float(jac)):
            v_bits.append(f"flag Jaccard {float(jac) * 100.0:.1f}%")
        if isinstance(vnrmse, (int, float)) and math.isfinite(float(vnrmse)):
            v_bits.append(f"vis NRMSE {float(vnrmse):.4f}")
        if isinstance(vphs, (int, float)) and math.isfinite(float(vphs)):
            v_bits.append(f"vis phase RMS {float(vphs):.2f} deg")
        if v_bits:
            parts.append("ref-MS diff: " + ", ".join(v_bits))

    cal = diff_doc.get("caltables")
    if isinstance(cal, dict) and cal.get("status") == "ok":
        c_bits: list[str] = []
        b_blk = cal.get("B")
        if isinstance(b_blk, dict) and b_blk.get("status") == "ok":
            bp_phs = b_blk.get("bp_phase_rms_deg")
            if isinstance(bp_phs, (int, float)) and math.isfinite(float(bp_phs)):
                c_bits.append(f"B phase RMS {float(bp_phs):.2f} deg")
        g_blk = cal.get("G")
        if isinstance(g_blk, dict):
            frat = g_blk.get("fluxscale_ratio")
            if isinstance(frat, (int, float)) and math.isfinite(float(frat)):
                c_bits.append(f"fluxscale ratio {float(frat):.4f}")
            for gk in ("F_fluxscale", "G_ap", "G0_phase_only"):
                gp = g_blk.get(gk)
                if isinstance(gp, dict) and gp.get("status") == "ok":
                    g_phs = gp.get("gain_phase_rms_deg")
                    if isinstance(g_phs, (int, float)) and math.isfinite(float(g_phs)):
                        c_bits.append(f"{gk} phase RMS {float(g_phs):.2f} deg")
                        break
        if c_bits:
            parts.append("caltable diff: " + ", ".join(c_bits))

    img = diff_doc.get("images")
    if isinstance(img, dict) and img.get("status") == "ok":
        i_bits: list[str] = []
        ncc = img.get("ncc")
        afr = img.get("aperture_flux_ratio")
        rr = img.get("offsource_rms_ratio")
        if isinstance(ncc, (int, float)) and math.isfinite(float(ncc)):
            i_bits.append(f"common-beam NCC {float(ncc):.4f}")
        if isinstance(afr, (int, float)) and math.isfinite(float(afr)):
            i_bits.append(f"aperture flux ratio {float(afr):.3f}")
        if isinstance(rr, (int, float)) and math.isfinite(float(rr)):
            i_bits.append(f"RMS ratio {float(rr):.2f}x")
        if i_bits:
            parts.append("image diff: " + ", ".join(i_bits))

    return "; ".join(parts)


def _format_stage_physical_summary(
    rec: dict[str, Any],
    *,
    diff_doc: dict[str, Any] | None = None,
) -> str:
    """Format concise physical observables with astronomical units for one trial."""
    stage = str(rec.get("stage") or "")
    bits: list[str] = []

    if stage == "inspect":
        fcal = _get_obs_any(rec, "inspect.flux_cal_list", "flux_calibrator")
        bpcal = _get_obs_any(rec, "inspect.bandpass_cal_list", "bandpass_calibrator")
        pcal = _get_obs_any(rec, "inspect.phase_cal_list", "phase_calibrator")
        dead = _get_obs_any(rec, "inspect.dead_antennas_list", "dead_antennas")
        elev = _get_obs_num(
            rec,
            "inspect.bandpass_min_el_deg",
            "inspect.bandpass_min_elevation_deg",
            "bandpass_min_elevation_deg",
        )
        f_s = _fmt_str_or_list(fcal) if fcal is not None else None
        bp_s = _fmt_str_or_list(bpcal) if bpcal is not None else None
        if f_s is not None and bp_s is not None:
            if f_s == bp_s:
                bits.append(f"flux and bandpass calibrator `{f_s}`")
            else:
                bits.append(f"flux calibrator `{f_s}`")
                bits.append(f"bandpass calibrator `{bp_s}`")
        elif f_s is not None:
            bits.append(f"flux calibrator `{f_s}`")
        elif bp_s is not None:
            bits.append(f"bandpass calibrator `{bp_s}`")
        if pcal is not None:
            p_s = _fmt_str_or_list(pcal)
            bits.append(f"phase calibrator `{p_s}`")
        if dead is not None:
            d_s = _fmt_str_or_list(dead)
            bits.append(f"dead antennas `[{d_s}]`")
        if elev is not None:
            bits.append(f"minimum bandpass elevation `{elev:.2f} deg`")

    elif stage == "precal":
        m_jy = _get_obs_num(rec, "precal.flux_cal_model_amp_jy", "j1331_model_amp_jy")
        cal_rec = _get_obs_num(rec, "precal.calibrator_must_flag_recall")
        tgt_rec = _get_obs_num(rec, "precal.target_must_flag_recall")
        tot_rec = _get_obs_num(rec, "precal.must_flag_recall", "must_flag_recall")
        cal_fpr = _get_obs_num(rec, "precal.calibrator_clean_false_positive_rate")
        tot_fpr = _get_obs_num(rec, "precal.clean_false_positive_rate", "clean_false_positive_rate")
        if m_jy is not None:
            bits.append(f"flux calibrator model `{m_jy:.3f} Jy`")
        if cal_rec is not None and tgt_rec is not None:
            bits.append(
                f"calibrator RFI recall `{cal_rec * 100.0:.1f}%`, target preflag recall `{tgt_rec * 100.0:.1f}%`"
            )
        elif tot_rec is not None:
            bits.append(f"flag recall `{tot_rec * 100.0:.1f}%`")
        if cal_fpr is not None:
            bits.append(
                f"clean overflagging `{tot_fpr * 100.0:.2f}%` (calibrator `{cal_fpr * 100.0:.2f}%`)"
                if tot_fpr is not None
                else f"calibrator clean overflagging `{cal_fpr * 100.0:.2f}%`"
            )
        elif tot_fpr is not None:
            bits.append(f"clean overflagging `{tot_fpr * 100.0:.2f}%`")

    if stage in ("calibrate", "e2e"):
        corr_dist = _get_obs_bool(
            rec, "cal.corrected_distinct_from_data", "corrected_distinct_from_data"
        )
        bp_cal_ok = _get_obs_bool(rec, "cal.bandpass_cal_calibrated", "bandpass_cal_calibrated")
        if corr_dist is False:
            bits.append("CORRECTED_DATA not populated")
        else:
            p_flux = _get_obs_num(
                rec,
                "cal.fluxscale_table_flux_jy",
                "cal.phase_cal_fluxscale_jy",
                "cal.phase_cal_corrected_amp_jy",
                "pcal_flux_jy",
            )
            bp_slp = _get_obs_num(
                rec,
                "cal.bandpass_coh_phase_slope_deg_per_100mhz",
                "cal.bandpass_med_bl_phase_slope_deg_per_100mhz",
                "cal.bp_phase_slope_deg_per_100mhz",
                "j1331_phase_slope_deg_per_100mhz",
            )
            bp_rms = _get_obs_num(
                rec,
                "cal.bandpass_coh_chan_phase_rms_deg",
                "cal.bandpass_med_bl_chan_phase_rms_deg",
                "cal.bandpass_row_chan_phase_rms_deg",
                "cal.bp_chan_phase_rms_deg",
                "j1331_chan_phase_rms_deg",
            )
            ret = _get_obs_num(rec, "cal.target_row_retention", "target_row_retention")
            dead_f = _get_obs_num(
                rec, "cal.dead_antenna_flagged_fraction", "dead_antenna_flagged_fraction"
            )
            v_dr = _get_obs_num(rec, "cal.verifier_tclean_dr", "verifier_tclean_dynamic_range")
            coh_phs = _get_obs_num(rec, "cal.coherent_vis_phase_err_deg", "coherent_phase_deg")
            if p_flux is not None:
                bits.append(f"bootstrapped phase calibrator flux `{p_flux:.3f} Jy`")
            if bp_cal_ok is False:
                bits.append("bandpass calibrator left uncalibrated")
            else:
                if bp_slp is not None:
                    bits.append(f"bandpass phase slope `{bp_slp:.2f} deg/100MHz`")
                if bp_rms is not None:
                    bits.append(f"bandpass channel phase RMS `{bp_rms:.2f} deg`")
            if ret is not None:
                bits.append(f"target retention `{ret * 100.0:.1f}%`")
            if dead_f is not None:
                bits.append(f"dead antennas flagged `{dead_f * 100.0:.1f}%`")
            if v_dr is not None:
                bits.append(f"calibration check dynamic range `{v_dr:.1f}`")
            elif coh_phs is not None:
                bits.append(f"target coherent phase error `{coh_phs:.2f} deg`")

    if stage in ("image", "image_cont", "e2e"):
        ppb = _get_obs_num(rec, "image.pixels_per_beam", "pixels_per_beam")
        rms_jy = _get_obs_num(rec, "image.offsource_residual_rms_jy", "off_source_residual_rms")
        rms_mjy = (
            rms_jy * 1e3
            if rms_jy is not None
            else _get_obs_num(rec, "offsource_residual_rms_mjy_per_beam")
        )
        dr = _get_obs_num(rec, "image.dynamic_range", "dynamic_range")
        iflux = _get_obs_num(
            rec,
            "image.source_integrated_flux_jy",
            "image.ext_integrated_flux_jy",
            "ext_integrated_flux_jy",
        )
        bowl = _get_obs_num(
            rec,
            "image.negative_bowl_ratio",
            "negative_bowl_ratio",
            "negative_model_bowl_ratio",
        )
        if ppb is not None:
            bits.append(f"beam sampling `{ppb:.2f} pix/beam`")
        if rms_mjy is not None:
            bits.append(f"off-source residual RMS `{rms_mjy:.3f} mJy/beam`")
        if dr is not None:
            bits.append(f"dynamic range `{dr:.1f}`")
        if iflux is not None:
            bits.append(f"integrated flux `{iflux:.3f} Jy`")
        if bowl is not None:
            bits.append(f"negative clean-bowl ratio `{bowl:.3f}`")

    # Generic fallback for unseen future stages (`polcal`, `postcal`, `selfcal`, `image_cube`, etc.)
    if not bits and isinstance(rec.get("measurements"), dict):
        for obs_k, obs_rec in sorted(rec["measurements"].items()):
            if obs_k.startswith("proc.") or not isinstance(obs_rec, dict):
                continue
            if obs_rec.get("status") != "measured" or obs_rec.get("value") is None:
                continue
            val = obs_rec["value"]
            u_str = f" {obs_rec['unit']}" if obs_rec.get("unit") else ""
            if isinstance(val, float) and math.isfinite(val):
                bits.append(f"`{obs_k}` = `{val:.4g}{u_str}`")
            else:
                bits.append(f"`{obs_k}` = `{val}{u_str}`")
            if len(bits) >= 5:
                break

    diff_txt = _format_diff_snippet(
        diff_doc or (rec.get("diff_summary") if isinstance(rec.get("diff_summary"), dict) else None)
    )
    if diff_txt:
        bits.append(diff_txt)

    if not bits:
        return "no physical deliverables written before exit"
    return "; ".join(bits)


def _record_model_display(rec: dict[str, Any], *, multi_tl: bool, multi_pack: bool) -> str:
    m_raw = str(rec.get("model_name") or "unknown")
    tl = str(rec.get("thinking_level") or "default")
    pid = str(rec.get("pack_id") or "ra_00a11ea")
    tags: list[str] = []
    if multi_tl and tl != "default":
        tags.append(tl)
    if multi_pack:
        tags.append(f"pack={pid}")
    return f"{m_raw} ({', '.join(tags)})" if tags else m_raw


def _missing_factorial_arms_for_group(group_recs: list[dict[str, Any]]) -> list[str]:
    present = {str(r.get("arm")) for r in group_recs if r.get("arm")}
    missing: list[str] = []
    if "00_none" not in present:
        missing.append("`Bare CASA` (`00_none`)")
    if not (present & {"01_skills_casa", "01_skills_raw"}):
        missing.append("`Workflow Guides Only` (`01_skills_casa` / `01_skills_raw`)")
    if "10_tools" not in present:
        missing.append("`MCP Tools Only` (`10_tools`)")
    if "11_full" not in present:
        missing.append("`Guides + MCP Tools` (`11_full`)")
    return missing


def render_human_report_markdown(
    records: Sequence[TrialRecord | dict[str, Any]],
    summary: dict[str, Any],
    *,
    diffs_dir: Path | None = None,
) -> str:
    """Render an astronomer-readable Markdown benchmark report (`report.md`) with zero em/en dashes."""
    rec_list: list[dict[str, Any]] = [dict(r) for r in records]
    diffs_index = _load_diffs_index(diffs_dir)
    run_health = (
        summary["run_health"]
        if isinstance(summary.get("run_health"), dict)
        else summarize_run_health(rec_list)
    )

    lines: list[str] = [
        "# Radio Astronomy Reduction Benchmark Report",
        "",
        (
            "> **What This Benchmark Measures:** `radio-bench` evaluates **scientific "
            "radio-interferometry data reduction outcomes produced by LLM agents**, graded "
            "directly on arm-neutral CASA Measurement Sets (`input.ms`, `calibrators.ms`), "
            "calibration tables, continuum images, and stage metadata against physical reference "
            f"truth (`PASS` requires composite score >= `{PASS_REWARD_MIN:.2f}` with all hard "
            "data-integrity gates satisfied). It compares end-to-end reduction quality across "
            "capability setups -- it does **not** unit-test prompt guides or MCP tools in isolation."
        ),
        "",
    ]

    if not rec_list:
        lines.extend(
            [
                "## 1. At-a-Glance Stage x Capability Scorecard",
                "",
                "No trials were found in the supplied job directory.",
                "",
            ]
        )
        return "\n".join(lines).replace("\u2014", "--").replace("\u2013", "--")

    # Group strictly by (model_display, suite, pack_key) across BOTH Section 1 and Section 2
    multi_tl = len({str(r.get("thinking_level") or "default") for r in rec_list}) > 1
    multi_pack = len({str(r.get("pack_id") or "ra_00a11ea") for r in rec_list}) > 1
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rec_list:
        m_disp = _record_model_display(r, multi_tl=multi_tl, multi_pack=multi_pack)
        s_raw = str(r.get("suite") or "unknown")
        p_key = str(r.get("pack_id") or "ra_00a11ea") if multi_pack else ""
        groups.setdefault((m_disp, s_raw, p_key), []).append(r)

    sorted_groups = sorted(groups.items())
    lines.extend(["## 1. At-a-Glance Stage x Capability Scorecard", ""])

    for (m_disp, s_raw, _p_key), g_recs in sorted_groups:
        g_arms = _ordered_human_arms(g_recs)
        g_stages = _ordered_human_stages(g_recs)
        lines.append(
            f"### Model: `{_md_cell(m_disp)}` -- Dataset: {_md_cell(_human_suite_title(s_raw))}"
        )
        lines.append("")
        hdr = ["Pipeline Stage", *(_human_arm_title(a) for a in g_arms)]
        lines.append("| " + " | ".join(_md_cell(h) for h in hdr) + " |")
        lines.append("| " + " | ".join(["---"] * len(hdr)) + " |")
        for st in g_stages:
            row_cells = [_human_stage_title(st)]
            for arm in g_arms:
                c_recs = [
                    r
                    for r in g_recs
                    if str(r.get("arm")) == arm
                    and ("image" if str(r.get("stage")) == "image_cont" else str(r.get("stage")))
                    == st
                ]
                row_cells.append(_format_scorecard_cell(c_recs, arm=arm, stage=st))
            lines.append("| " + " | ".join(_md_cell(c) for c in row_cells) + " |")
        lines.append("")

    # Run Health & Missing Arms Note (mutually exclusive partition of n_trials)
    n_total = int(run_health.get("n_trials", len(rec_list)))
    n_clean = int(run_health.get("clean_completions", 0))
    n_epipe = int(run_health.get("benign_epipe_exits", 0))
    n_api = int(run_health.get("provider_api_errors", 0))
    n_recov = int(run_health.get("recovered_provider_errors", 0))
    n_tok_lim = int(run_health.get("output_token_limits", 0))
    n_to = int(run_health.get("agent_timeouts", 0))
    n_err = int(run_health.get("agent_errors", 0))
    api_trials = [
        it
        for it in (run_health.get("provider_error_trials") or [])
        if isinstance(it, dict)
        and str(it.get("agent_exit_status") or "provider_api_error") == "provider_api_error"
    ]
    recov_trials = [
        it
        for it in (run_health.get("recovered_provider_error_trials") or [])
        if isinstance(it, dict)
    ]
    tok_lim_trials = [
        it for it in (run_health.get("output_token_limit_trials") or []) if isinstance(it, dict)
    ]
    to_trials = [it for it in (run_health.get("timeout_trials") or []) if isinstance(it, dict)]
    err_trials = [it for it in (run_health.get("agent_error_trials") or []) if isinstance(it, dict)]

    lines.append("### Run Health and Experimental Coverage Note")
    lines.append("")
    health_notes: list[str] = []
    if n_epipe > 0:
        health_notes.append(
            f"`{n_epipe}` OpenCode runs had a benign post-completion stdout `EPIPE` after work finished"
        )
    if n_recov > 0:
        recov_names = ", ".join(
            f"`{it.get('trial_name') or it.get('base_task') or 'trial'}`" for it in recov_trials
        )
        recov_detail = f": {recov_names}" if recov_names else ""
        health_notes.append(
            f"`{n_recov}` runs recovered from transient upstream provider errors{recov_detail}"
        )
    note_clause = f" ({'; '.join(health_notes)})" if health_notes else ""
    lines.append(
        f"- **Completed agent runs:** `{n_clean + n_epipe}/{n_total}` trials finished agent execution cleanly{note_clause}."
    )
    if n_api > 0 or api_trials:
        api_descs: list[str] = []
        for it in api_trials:
            tn = str(it.get("trial_name") or it.get("base_task") or "trial")
            sc = it.get("provider_status_code")
            kd = str(it.get("provider_error_kind") or "provider_api_error")
            trn = int(it.get("n_turns") or 0)
            sc_s = f"HTTP {sc}" if sc is not None else kd
            api_descs.append(f"`{tn}` ({sc_s}, {kd}, turn {trn})")
        joined_api = ", ".join(api_descs) if api_descs else f"`{n_api}` trials"
        lines.append(f"- **Upstream LLM/API interruptions (`{n_api}`):** {joined_api}.")
    else:
        lines.append(
            "- **Upstream LLM/API interruptions:** `0` trials interrupted by provider errors."
        )

    if n_tok_lim > 0 or tok_lim_trials:
        tl_descs: list[str] = []
        for it in tok_lim_trials:
            tn = str(it.get("trial_name") or it.get("base_task") or "trial")
            trn = int(it.get("n_turns") or 0)
            lst_tok = it.get("last_step_tokens")
            tok_str = (
                f"{int(lst_tok):,} tokens, "
                if isinstance(lst_tok, (int, float))
                and not isinstance(lst_tok, bool)
                and int(lst_tok) > 0
                else ""
            )
            tl_descs.append(f"`{tn}` ({tok_str}turn {trn})")
        joined_tl = ", ".join(tl_descs) if tl_descs else f"`{n_tok_lim}` trials"
        lines.append(f"- **Output/reasoning token limit cutoffs (`{n_tok_lim}`):** {joined_tl}.")

    if n_to > 0 or to_trials:
        to_descs: list[str] = []
        for it in to_trials:
            tn = str(it.get("trial_name") or it.get("base_task") or "trial")
            w_min = float(it.get("agent_wall_sec") or 0.0) / 60.0
            rw = float(it.get("reward") or 0.0)
            to_descs.append(f"`{tn}` ({w_min:.1f}m, score {rw:.2f})")
        lines.append(
            f"- **Wall-clock timeouts (`{n_to}`):** {', '.join(to_descs) if to_descs else f'`{n_to}` trials'}."
        )
    else:
        lines.append("- **Wall-clock timeouts:** `0` trials timed out.")

    if n_err > 0 or err_trials:
        err_descs: list[str] = []
        for it in err_trials:
            tn = str(it.get("trial_name") or it.get("base_task") or "trial")
            et = str(it.get("exception_type") or "agent_error")
            rw = float(it.get("reward") or 0.0)
            err_descs.append(f"`{tn}` ({et}, score {rw:.2f})")
        lines.append(
            f"- **Local non-zero agent exits (`{n_err}`):** {', '.join(err_descs) if err_descs else f'`{n_err}` trials'}."
        )

    group_missing: list[tuple[str, str, list[str]]] = []
    for (m_disp, s_raw, _p_key), g_recs in sorted_groups:
        m_arms = _missing_factorial_arms_for_group(g_recs)
        if m_arms:
            group_missing.append((m_disp, s_raw, m_arms))

    if group_missing:
        if len(sorted_groups) == 1:
            _m0, _s0, m_arms0 = group_missing[0]
            lines.append(
                "- **Capability arms not included in this run:** "
                + ", ".join(m_arms0)
                + " were not part of this job, so full 2x2 factorial main effects and interaction synergy are not estimated here."
            )
        else:
            per_grp_strs = [
                f"`{m_d}` on {_human_suite_title(s_r)} ({', '.join(m_a)})"
                for m_d, s_r, m_a in group_missing
            ]
            lines.append(
                "- **Capability arms not included in this run:** "
                + "; ".join(per_grp_strs)
                + " -- full 2x2 factorial main effects and interaction synergy require all four factorial arms per model and dataset."
            )
    lines.append("")

    # Section 2: Stage-by-Stage Plain-English What Worked & What Missed (grouped identically to Section 1)
    lines.extend(["## 2. Stage-by-Stage Plain-English What Worked and What Missed", ""])
    multi_group = len(sorted_groups) > 1
    stage_heading_prefix = "####" if multi_group else "###"

    for (m_disp, s_raw, _p_key), g_recs in sorted_groups:
        if multi_group:
            lines.append(
                f"### Model: `{_md_cell(m_disp)}` -- Dataset: {_md_cell(_human_suite_title(s_raw))}"
            )
            lines.append("")
        g_stages = _ordered_human_stages(g_recs)
        g_arms = _ordered_human_arms(g_recs)

        for st in g_stages:
            lines.append(f"{stage_heading_prefix} {_human_stage_title(st)}")
            lines.append("")
            check_desc = _HUMAN_STAGE_CHECKS.get(
                st,
                "Checks physical CASA deliverables, calibration/image fidelity, and data-integrity gates against reference truth.",
            )
            lines.append(f"**What was checked on disk:** {check_desc}")
            lines.append("")

            st_recs = [
                r
                for r in g_recs
                if ("image" if str(r.get("stage")) == "image_cont" else str(r.get("stage"))) == st
            ]
            for arm in g_arms:
                arm_st_recs = [r for r in st_recs if str(r.get("arm")) == arm]
                if not arm_st_recs:
                    continue
                arm_lbl = _human_arm_title(arm)
                if len(arm_st_recs) == 1:
                    r0 = arm_st_recs[0]
                    tname = str(r0.get("trial_name") or r0.get("base_task") or "")
                    diff_doc = diffs_index.get(tname)
                    cell_str = _format_scorecard_cell([r0], arm=arm, stage=st)
                    phys_str = _format_stage_physical_summary(r0, diff_doc=diff_doc)
                    lines.append(f"- **{arm_lbl}** (`{cell_str}`): {phys_str}.")
                else:
                    cell_str = _format_scorecard_cell(arm_st_recs, arm=arm, stage=st)
                    # Representative median-reward trial for physical telemetry readout
                    ordered_by_rew = sorted(
                        arm_st_recs, key=lambda x: float(x.get("reward") or 0.0)
                    )
                    rep_rec = ordered_by_rew[len(ordered_by_rew) // 2]
                    tname = str(rep_rec.get("trial_name") or rep_rec.get("base_task") or "")
                    diff_doc = diffs_index.get(tname)
                    phys_str = _format_stage_physical_summary(rep_rec, diff_doc=diff_doc)
                    lines.append(
                        f"- **{arm_lbl}** (`{cell_str}` across `n={len(arm_st_recs)}` seeds; median run `{tname}`): {phys_str}."
                    )
            lines.append("")

    # Section 3: Runtime, Cost, & Multi-Agent Breakdown
    lines.extend(["## 3. Runtime, Cost, and Multi-Agent Breakdown", ""])
    rt_hdr = [
        "Model",
        "Stage",
        "Capability Setup",
        "Trial",
        "Outcome",
        "Score",
        "Runtime (min)",
        "Turns",
        "MCP Calls",
        "Bash Calls",
        "Cost (USD)",
        "Exit Status",
    ]
    lines.append("| " + " | ".join(_md_cell(h) for h in rt_hdr) + " |")
    lines.append("| " + " | ".join(["---"] * len(rt_hdr)) + " |")

    stage_rank = {st: idx for idx, (st, _lbl) in enumerate(_HUMAN_STAGE_ORDER)}
    arm_rank = {a: idx for idx, a in enumerate(_HUMAN_ARM_ORDER)}
    sorted_trials = sorted(
        rec_list,
        key=lambda r: (
            _record_model_display(r, multi_tl=multi_tl, multi_pack=multi_pack),
            str(r.get("suite") or ""),
            str(r.get("pack_id") or "ra_00a11ea"),
            stage_rank.get(
                "image" if str(r.get("stage")) == "image_cont" else str(r.get("stage")), 99
            ),
            arm_rank.get(str(r.get("arm")), 99),
            int(r.get("attempt_idx") or 0),
        ),
    )

    for r in sorted_trials:
        st_norm = "image" if str(r.get("stage")) == "image_cont" else str(r.get("stage") or "")
        m_disp_row = _record_model_display(r, multi_tl=multi_tl, multi_pack=multi_pack)
        tag, _rsn = _diagnose_trial_cell_reason(r)
        rew = float(r.get("reward") or 0.0)
        mins = float(r.get("agent_wall_sec") or r.get("total_wall_sec") or 0.0) / 60.0
        usd = r.get("total_usd")
        usd_s = (
            f"${float(usd):.4f}"
            if isinstance(usd, (int, float))
            and not isinstance(usd, bool)
            and math.isfinite(float(usd))
            else "n/a"
        )
        ex_st = str(r.get("agent_exit_status") or "ok")
        if ex_st == "provider_api_error" and r.get("provider_status_code") is not None:
            ex_st = f"provider_api_error (HTTP {r['provider_status_code']})"
        elif ex_st == "output_token_limit" and r.get("last_step_tokens") is not None:
            ex_st = f"output_token_limit ({int(r['last_step_tokens']):,} tok)"
        row_vals = [
            f"`{m_disp_row}`",
            _human_stage_title(st_norm),
            _human_arm_title(str(r.get("arm") or "")),
            f"`{r.get('trial_name', '')}`",
            tag,
            f"{rew:.2f}",
            f"{mins:.1f}",
            str(int(r.get("n_turns") or 0)),
            str(int(r.get("mcp_calls_total") or 0)),
            str(int(r.get("bash_calls") or 0)),
            usd_s,
            ex_st,
        ]
        lines.append("| " + " | ".join(_md_cell(v) for v in row_vals) + " |")
    lines.append("")

    # Multi-agent subagent breakdown when 11_orchestrated or multi-role trials exist
    orch_trials = [
        r
        for r in sorted_trials
        if isinstance(r.get("role_breakdown"), dict)
        and (
            str(r.get("arm")) == "11_orchestrated"
            or len(set(r["role_breakdown"].keys()) - {"main"}) > 0
        )
    ]
    if orch_trials:
        lines.extend(["### Orchestrated Subagent Breakdown", ""])
        ob_hdr = [
            "Trial",
            "Role / Stage Agent",
            "Turns",
            "MCP Calls",
            "Bash Calls",
            "Input Tokens",
            "Output Tokens",
            "Cost (USD)",
        ]
        lines.append("| " + " | ".join(_md_cell(h) for h in ob_hdr) + " |")
        lines.append("| " + " | ".join(["---"] * len(ob_hdr)) + " |")
        role_rank = {rl: idx for idx, rl in enumerate(_ROLE_DISPLAY_ORDER)}
        for r in orch_trials:
            rb_map = r["role_breakdown"]
            ordered_roles = sorted(
                rb_map.keys(),
                key=lambda rk: (role_rank.get(str(rk), 99), str(rk)),
            )
            for rk in ordered_roles:
                rinfo = rb_map.get(rk)
                if not isinstance(rinfo, dict):
                    continue
                r_usd = rinfo.get("total_usd")
                r_usd_s = (
                    f"${float(r_usd):.4f}"
                    if isinstance(r_usd, (int, float))
                    and not isinstance(r_usd, bool)
                    and math.isfinite(float(r_usd))
                    else "n/a"
                )
                lines.append(
                    "| "
                    + " | ".join(
                        _md_cell(v)
                        for v in (
                            f"`{r.get('trial_name', '')}`",
                            f"`{rk}`",
                            str(int(rinfo.get("turns") or 0)),
                            str(int(rinfo.get("mcp_calls_total") or 0)),
                            str(int(rinfo.get("bash_calls") or 0)),
                            f"{int(rinfo.get('input_tokens') or 0):,}",
                            f"{int(rinfo.get('output_tokens') or 0):,}",
                            r_usd_s,
                        )
                    )
                    + " |"
                )
        lines.append("")

    report_text = "\n".join(lines) + "\n"
    return report_text.replace("\u2014", "--").replace("\u2013", "--")


def _resolve_default_diffs_dir(output_dir: Path, job_dirs: Sequence[Path]) -> Path | None:
    cand_direct = output_dir / "diffs"
    if cand_direct.is_dir():
        return cand_direct
    for jp in job_dirs:
        p = Path(jp)
        cand_job = p / "analysis" / "diffs"
        if cand_job.is_dir():
            return cand_job
        if _is_trial_directory(p):
            cand_trial_parent = p.parent / "analysis" / "diffs"
            if cand_trial_parent.is_dir():
                return cand_trial_parent
    return None


def write_diagnostic_reports(
    *,
    job_dirs: list[Path],
    output_dir: Path,
    records: list[dict[str, Any]],
    factorial_summary: dict[str, Any],
    taxonomy_summary: dict[str, Any],
    plots: bool = False,
    diffs_dir: Path | None = None,
) -> dict[str, Any]:
    """Write `summary.json`, `summary.md`, `report.md`, and (only if `plots=True`) PNGs."""
    out_p = Path(output_dir)
    reject_slash_tmp(out_p, label="output_dir")
    for jp in job_dirs:
        reject_slash_tmp(Path(jp), label="job_dir")
    if diffs_dir is not None:
        reject_slash_tmp(Path(diffs_dir), label="diffs_dir")
    out_p.mkdir(parents=True, exist_ok=True)
    job_dir_strs = [str(p) for p in job_dirs]
    truth_vs_measured = build_truth_vs_measured_table(records)
    run_health = summarize_run_health(records)

    plot_files: list[str] = []
    if plots:
        plot_files = generate_diagnostic_plots(
            records=records,
            factorial_summary=factorial_summary,
            taxonomy_summary=taxonomy_summary,
            plots_dir=out_p / "plots",
        )

    summary_payload = sanitize_json_tree(
        {
            "job_dirs": job_dir_strs,
            "output_dir": str(out_p),
            "n_trials": len(records),
            "run_health": run_health,
            "truth_vs_measured": truth_vs_measured,
            "factorial": factorial_summary,
            "taxonomy": taxonomy_summary,
            "plot_files": plot_files,
        }
    )

    summary_json_path = out_p / "summary.json"
    summary_md_path = out_p / "summary.md"
    report_md_path = out_p / "report.md"

    json_text = json.dumps(summary_payload, indent=2, allow_nan=False) + "\n"
    json_text = json_text.replace("\u2014", "--").replace("\u2013", "--")
    _atomic_write_text(summary_json_path, json_text)

    md_text = render_summary_markdown(
        job_dirs=job_dir_strs,
        records=records,
        truth_vs_measured=truth_vs_measured,
        factorial_summary=factorial_summary,
        taxonomy_summary=taxonomy_summary,
    )
    _atomic_write_text(summary_md_path, md_text)

    if diffs_dir is not None:
        eff_diffs_dir = Path(diffs_dir) if Path(diffs_dir).is_dir() else None
    else:
        eff_diffs_dir = _resolve_default_diffs_dir(out_p, job_dirs)

    human_report_md = render_human_report_markdown(
        records,
        summary_payload,
        diffs_dir=eff_diffs_dir,
    )
    _atomic_write_text(report_md_path, human_report_md)

    return {
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
        "report_md_path": str(report_md_path),
        "plot_files": plot_files,
        "summary": summary_payload,
        "markdown": md_text,
        "report_markdown": human_report_md,
    }
