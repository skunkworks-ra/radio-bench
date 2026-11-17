"""Backward-compatible `grade_image` wrapper delegating to `extractors` + `rubric`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radio_bench.grading.extractors.common import has_active_table_lock
from radio_bench.grading.inspect_metrics import resolve_task_configs
from radio_bench.grading.runner import run_verification

PASS_REWARD_MIN_DEFAULT = 0.85


def grade_image(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path | None = None,
    task_id: str,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade delivered CASA continuum image products inside `/workspace`."""
    ws = Path(workspace_dir)
    ec, rb = resolve_task_configs(task_id, "image", ground_truth)
    if rb.stage_blocks:
        img_blk = next(b for b in rb.stage_blocks if b.stage == "image")
        rb = rb.model_copy(
            update={
                "stages": ["image"],
                "stage_blocks": [],
                "subscores": img_blk.subscores,
                "core_subscore_floors": {
                    **rb.core_subscore_floors,
                    **img_blk.core_subscore_floors,
                },
            },
            deep=True,
        )
        ec.stages = ["image"]
        if img_blk.truth_prefix and isinstance(ground_truth.get(img_blk.truth_prefix), dict):
            ground_truth = ground_truth[img_blk.truth_prefix]

    ref_dir = Path(masks_npz_path) if masks_npz_path is not None else None
    return run_verification(
        workspace_dir=ws,
        tests_dir=ws,
        log_dir=None,
        extract_config=ec,
        rubric=rb,
        ground_truth=ground_truth,
        ref_arrays_dir=ref_dir,
        compute_raw_hash=(ws / "input.ms" / "table.dat").is_file(),
    )


__all__ = ["PASS_REWARD_MIN_DEFAULT", "grade_image", "has_active_table_lock"]
