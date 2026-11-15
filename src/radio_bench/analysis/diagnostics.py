"""Truth-vs-measured physical comparison tables, summary report writer, and optional plots."""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.analysis.collect import CANONICAL_ARMS, sanitize_json_tree
from radio_bench.analysis.taxonomy import ALL_OUTCOME_LABELS
from radio_bench.grading import PASS_REWARD_MIN

_INSPECT_SCALAR_CAL_FIELDS: frozenset[str] = frozenset(
    {
        "flux_calibrator",
        "bandpass_calibrator",
        "phase_calibrator",
        "pol_leakage_calibrator",
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


def build_truth_vs_measured_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Truth-vs-Measured comparison rows per task across all present arms."""
    arms = _ordered_arms(records)
    tasks = sorted({str(r.get("base_task")) for r in records if r.get("base_task")})
    rows: list[dict[str, Any]] = []

    for task_id in tasks:
        t_recs = [r for r in records if str(r.get("base_task")) == task_id]
        if not t_recs:
            continue
        stage = str(t_recs[0].get("stage") or "unknown")
        suite = str(t_recs[0].get("suite") or "unknown")

        # Find first non-empty truth dict across trials of this task.
        ref_truth: dict[str, Any] = {}
        for r in t_recs:
            tb = r.get("truth")
            if isinstance(tb, dict) and tb:
                ref_truth = tb
                break

        specs = _extract_stage_metric_specs(stage, ref_truth)
        if not specs and ref_truth:
            # Generic fallback for custom truth keys in synthetic fixtures.
            for k in sorted(ref_truth):
                val = ref_truth[k]
                if isinstance(val, (int, float, str)):
                    specs.append((k, _fmt_num(val, 4), (k,)))

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
                "truth": f">= {PASS_REWARD_MIN:.2f}",
                "by_arm": reward_by_arm,
            }
        )

        for metric_label, truth_str, lookup_path in specs:
            by_arm_val: dict[str, Any] = {}
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
        row_vals = [
            f"`{pkey}`",
            f"`{pobj['small_model']} @ {pobj['small_arm']}`",
            f"`{pobj['frontier_model']} @ {pobj['frontier_arm']}`",
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

    md_text = "\n".join(lines)
    return md_text.replace("\u2014", "--").replace("\u2013", "--")


def _table_has_active_lock(table_dir: Path) -> bool:
    lock_p = table_dir / "table.lock"
    if not lock_p.is_file():
        return False
    try:
        from radio_bench.grading.precal_metrics import has_active_table_lock

        return bool(has_active_table_lock(table_dir))
    except Exception:
        return True


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
    cfg = Path("/var/mnt/fast/radio-bench/casaconfig_local.py")
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
                opened = bool(tb.open(str(tbl_path), nomodify=True))
                if not opened:
                    continue
                cols = set(tb.colnames())
                if not (
                    ("CPARAM" in cols or "FPARAM" in cols) and "FLAG" in cols and "ANTENNA1" in cols
                ):
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
        for key, scale in (
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

    # Plot 2: Mean USD cost vs Mean Reward across (model, arm) cells.
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    by_model = factorial_summary.get("by_model", {})
    for model_name, m_data in by_model.items():
        m_cells = m_data.get("cells", {})
        xs = [
            float(m_cells[a].get("mean_usd_task_equal", m_cells[a]["mean_usd"]))
            for a in arms
            if a in m_cells
        ]
        ys = [
            float(m_cells[a].get("reward_task_equal_mean", m_cells[a]["reward_mean"]))
            for a in arms
            if a in m_cells
        ]
        a_lbls = [a for a in arms if a in m_cells]
        if xs:
            ax2.scatter(xs, ys, s=45, label=model_name)
            for xv, yv, al in zip(xs, ys, a_lbls, strict=True):
                ax2.annotate(al, (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel("Task-Equal Mean List-Price Cost per Trial (USD)")
    ax2.set_ylabel("Task-Equal Mean Composite Reward")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Cost vs Reward Across Model x Arm Cells")
    if by_model:
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


def write_diagnostic_reports(
    *,
    job_dirs: list[Path],
    output_dir: Path,
    records: list[dict[str, Any]],
    factorial_summary: dict[str, Any],
    taxonomy_summary: dict[str, Any],
    plots: bool = False,
) -> dict[str, Any]:
    """Write `summary.json`, `summary.md`, and (only if `plots=True`) PNGs in `output_dir/plots`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir_strs = [str(p) for p in job_dirs]
    truth_vs_measured = build_truth_vs_measured_table(records)

    plot_files: list[str] = []
    if plots:
        plot_files = generate_diagnostic_plots(
            records=records,
            factorial_summary=factorial_summary,
            taxonomy_summary=taxonomy_summary,
            plots_dir=output_dir / "plots",
        )

    summary_payload = sanitize_json_tree(
        {
            "job_dirs": job_dir_strs,
            "output_dir": str(output_dir),
            "n_trials": len(records),
            "truth_vs_measured": truth_vs_measured,
            "factorial": factorial_summary,
            "taxonomy": taxonomy_summary,
            "plot_files": plot_files,
        }
    )

    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

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

    return {
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
        "plot_files": plot_files,
        "summary": summary_payload,
        "markdown": md_text,
    }
