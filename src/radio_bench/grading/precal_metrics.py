"""Backward-compatible `grade_precal` wrapper delegating to `extractors` + `rubric`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radio_bench.grading.extractors.common import has_active_table_lock
from radio_bench.grading.inspect_metrics import resolve_task_configs
from radio_bench.grading.runner import run_verification

PASS_REWARD_MIN_DEFAULT = 0.85


def grade_precal(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path | None,
    task_id: str,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade `/workspace/{input.ms,calibrators.ms}` via pure `precal` extractor + `rubric`."""
    ec, rb = resolve_task_configs(task_id, "precal", ground_truth)
    ws = Path(workspace_dir)
    ref_dir = Path(masks_npz_path) if masks_npz_path is not None else ws
    return run_verification(
        workspace_dir=ws,
        tests_dir=ref_dir if ref_dir.is_dir() else ref_dir.parent,
        log_dir=None,
        extract_config=ec,
        rubric=rb,
        ground_truth=ground_truth,
        ref_arrays_dir=ref_dir,
        compute_raw_hash=(ws / "input.ms" / "table.dat").is_file(),
    )


__all__ = ["PASS_REWARD_MIN_DEFAULT", "grade_precal", "has_active_table_lock"]
