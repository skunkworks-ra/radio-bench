"""CLI entry point for `pixi run analyze <job_dir> [extra_job_dirs...]`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from radio_bench.analysis.collect import _is_trial_directory, collect_trials
from radio_bench.analysis.diagnostics import write_diagnostic_reports
from radio_bench.analysis.factorial import DEFAULT_BOOT_SEED, DEFAULT_N_BOOT, analyze_factorial
from radio_bench.analysis.taxonomy import summarize_failure_taxonomy
from radio_bench.config import (
    add_config_cli_args,
    config_from_cli_args,
    prepare_runtime_env,
    reject_slash_tmp,
)


def run_analysis_pipeline(
    job_dirs: Sequence[Path | str],
    *,
    output_dir: Path | None = None,
    diffs_dir: Path | None = None,
    plots: bool = False,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
) -> dict[str, Any]:
    """Execute end-to-end collection, factorial stats, failure taxonomy, and report writing."""
    resolved_jobs = [Path(p) for p in job_dirs]
    if not resolved_jobs:
        raise ValueError("At least one Harbor job directory is required.")
    for jp in resolved_jobs:
        reject_slash_tmp(jp, label="job_dir")

    if output_dir is not None:
        target_out = Path(output_dir)
    elif _is_trial_directory(resolved_jobs[0]):
        target_out = resolved_jobs[0].parent / "analysis"
    else:
        target_out = resolved_jobs[0] / "analysis"

    reject_slash_tmp(target_out, label="output_dir")
    if diffs_dir is not None:
        reject_slash_tmp(Path(diffs_dir), label="diffs_dir")

    records = collect_trials(resolved_jobs, output_dir=target_out)
    factorial_summary = analyze_factorial(records, n_boot=n_boot, seed=seed)
    taxonomy_summary = summarize_failure_taxonomy(records)
    report_bundle = write_diagnostic_reports(
        job_dirs=resolved_jobs,
        output_dir=target_out,
        records=records,
        factorial_summary=factorial_summary,
        taxonomy_summary=taxonomy_summary,
        plots=plots,
        diffs_dir=diffs_dir,
    )

    return {
        "output_dir": str(target_out),
        "trials_csv_path": str(target_out / "trials.csv"),
        "trials_jsonl_path": str(target_out / "trials.jsonl"),
        **report_bundle,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radio-bench-analyze",
        description=(
            "Analyze one or more Harbor 0.25.0 job directories: pricing, 2x2 factorial "
            "contrasts, paired bootstrap CIs, failure taxonomy, and truth-vs-measured tables."
        ),
    )
    add_config_cli_args(parser)
    parser.add_argument(
        "job_dirs",
        nargs="+",
        type=Path,
        help="One or more Harbor job directories (or trial directories) to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory (default: <first_job_dir>/analysis).",
    )
    parser.add_argument(
        "--diffs-dir",
        type=Path,
        default=None,
        help="Optional directory of diff-run *.json files (default: <output_dir>/diffs if present).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate optional PNG plots inside <output_dir>/plots/.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help=f"Number of paired bootstrap resamples (default: {DEFAULT_N_BOOT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOT_SEED,
        help=f"Deterministic RNG seed for paired bootstrap (default: {DEFAULT_BOOT_SEED}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = config_from_cli_args(args)
        prepare_runtime_env(cfg, config_path=args.config, mutate_os_environ=True)
        result = run_analysis_pipeline(
            args.job_dirs,
            output_dir=args.output_dir,
            diffs_dir=args.diffs_dir,
            plots=bool(args.plots),
            n_boot=int(args.n_boot),
            seed=int(args.seed),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(result["markdown"])
    if not result["markdown"].endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
