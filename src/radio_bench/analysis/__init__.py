"""Post-run analysis, list-price token accounting, 2x2 factorial stats, Pareto, and reference diff."""

from typing import TYPE_CHECKING, Any

from radio_bench.analysis._trace import extract_agent_runtime_diagnostics
from radio_bench.analysis.collect import (
    CANONICAL_ARMS,
    TrialRecord,
    collect_single_trial,
    collect_trial,
    collect_trials,
    discover_trial_dirs,
    extract_thinking_level,
    extract_trajectory_telemetry,
    parse_task_variant,
    parse_task_variant_with_pack,
    reconcile_trace_completeness,
    sanitize_json_tree,
    summarize_run_health,
    write_trials_outputs,
)
from radio_bench.analysis.diagnostics import (
    build_truth_vs_measured_table,
    generate_diagnostic_plots,
    render_human_report_markdown,
    render_summary_markdown,
    write_diagnostic_reports,
)
from radio_bench.analysis.factorial import (
    DEFAULT_BOOT_SEED,
    DEFAULT_N_BOOT,
    PARETO_ALL_5_ARMS,
    PARETO_E2E_6_ARMS,
    analyze_factorial,
    compute_pareto_frontiers,
    compute_parity_summary,
    compute_stratum_contrasts,
    pass_at_k_unbiased,
    summarize_arm_cells,
    wilson_ci,
)
from radio_bench.analysis.pricing import (
    HAIKU_CLIFF_MULTIPLIER,
    HAIKU_CLIFF_TOKEN_THRESHOLD,
    ModelRate,
    TrialExecutionAnalysis,
    TrialPricing,
    TurnPricing,
    analyze_trial_execution,
    compute_trial_pricing,
    get_model_rate,
    normalize_model_key,
    price_from_agent_context,
    price_turn,
)
from radio_bench.analysis.taxonomy import (
    ALL_OUTCOME_LABELS,
    FAILURE_CATEGORIES,
    FAILURE_NONE,
    classify_failure,
    summarize_failure_taxonomy,
)

_DIFF_EXPORTS = frozenset(
    {
        "diff_caltables",
        "diff_images",
        "diff_job_directory",
        "diff_visibilities",
        "diff_workspaces",
    }
)

__all__ = [
    "ALL_OUTCOME_LABELS",
    "CANONICAL_ARMS",
    "DEFAULT_BOOT_SEED",
    "DEFAULT_N_BOOT",
    "FAILURE_CATEGORIES",
    "FAILURE_NONE",
    "HAIKU_CLIFF_MULTIPLIER",
    "HAIKU_CLIFF_TOKEN_THRESHOLD",
    "MODEL_RATES",
    "ModelRate",
    "PARETO_ALL_5_ARMS",
    "PARETO_E2E_6_ARMS",
    "TrialExecutionAnalysis",
    "TrialPricing",
    "TrialRecord",
    "TurnPricing",
    "analyze_factorial",
    "analyze_trial_execution",
    "build_truth_vs_measured_table",
    "classify_failure",
    "collect_single_trial",
    "collect_trial",
    "collect_trials",
    "compute_pareto_frontiers",
    "compute_parity_summary",
    "compute_stratum_contrasts",
    "compute_trial_pricing",
    "diff_caltables",
    "diff_images",
    "diff_job_directory",
    "diff_visibilities",
    "diff_workspaces",
    "discover_trial_dirs",
    "extract_agent_runtime_diagnostics",
    "extract_thinking_level",
    "extract_trajectory_telemetry",
    "generate_diagnostic_plots",
    "get_model_rate",
    "main",
    "normalize_model_key",
    "parse_task_variant",
    "parse_task_variant_with_pack",
    "pass_at_k_unbiased",
    "price_from_agent_context",
    "price_turn",
    "reconcile_trace_completeness",
    "render_human_report_markdown",
    "render_summary_markdown",
    "run_analysis_pipeline",
    "sanitize_json_tree",
    "summarize_arm_cells",
    "summarize_failure_taxonomy",
    "summarize_run_health",
    "wilson_ci",
    "write_diagnostic_reports",
    "write_trials_outputs",
]


if TYPE_CHECKING:
    from radio_bench.analysis.cli import main, run_analysis_pipeline
    from radio_bench.analysis.diff import (
        diff_caltables,
        diff_images,
        diff_job_directory,
        diff_visibilities,
        diff_workspaces,
    )
    from radio_bench.analysis.pricing import MODEL_RATES


def __getattr__(name: str) -> Any:
    if name == "MODEL_RATES":
        from radio_bench.analysis import pricing as _pricing

        return _pricing.MODEL_RATES
    if name in ("main", "run_analysis_pipeline"):
        from radio_bench.analysis import cli as _cli

        return getattr(_cli, name)
    if name in _DIFF_EXPORTS:
        from radio_bench.analysis import diff as _diff

        return getattr(_diff, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
