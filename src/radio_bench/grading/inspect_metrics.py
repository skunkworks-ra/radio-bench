"""Backward-compatible `grade_inspect` wrapper delegating to `extractors` + `rubric`."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from radio_bench.config import resolve_repo_root
from radio_bench.grading.extractors.common import has_active_table_lock
from radio_bench.grading.runner import run_verification
from radio_bench.grading.schema import (
    ExtractConfig,
    RubricSpec,
    SuiteManifest,
    VerifierTcleanRecipe,
    loads_strict_json,
)

PASS_REWARD_MIN_DEFAULT = 0.85


@lru_cache(maxsize=4)
def load_suite_task_registry(
    suites_root_str: str | None = None,
) -> dict[str, tuple[ExtractConfig, RubricSpec]]:
    """Load `(ExtractConfig, RubricSpec)` keyed by `task_id` from `suites/*/suite.toml`."""
    candidates: list[Path] = []
    if suites_root_str:
        candidates.append(Path(suites_root_str))
    repo_root = resolve_repo_root()
    candidates.append(repo_root / "suites")
    candidates.append(Path("/opt/radio-bench/suites"))

    registry: dict[str, tuple[ExtractConfig, RubricSpec]] = {}
    for root in candidates:
        if not root.is_dir():
            continue
        for toml_path in sorted(root.glob("*/suite.toml")):
            sdir = toml_path.parent
            sm = SuiteManifest.model_validate(tomllib.loads(toml_path.read_text(encoding="utf-8")))
            for tid, tspec in sm.tasks.items():
                ec = sm.to_extract_config(tid)
                rb_path = sdir / tspec.rubric
                if not rb_path.is_file():
                    rb_path = repo_root / tspec.rubric
                rb = RubricSpec.model_validate(
                    loads_strict_json(rb_path.read_text(encoding="utf-8"))
                )
                registry[tid] = (ec, rb)
        if registry:
            break
    return registry


def resolve_task_configs(
    task_id: str,
    stage: str,
    ground_truth: dict[str, Any],
) -> tuple[ExtractConfig, RubricSpec]:
    """Resolve `(ExtractConfig, RubricSpec)` from `suites/` registry with structural overrides."""
    reg = load_suite_task_registry()
    if task_id in reg:
        ec_base, rb_base = reg[task_id]
        ec = ec_base.model_copy(deep=True)
        rb = rb_base.model_copy(deep=True)
    else:
        # Fallback to matching canonical stage template in `suites/` without dataset branches
        norm_stage = "image" if stage == "image_cont" else stage
        matched = next(
            (
                pair
                for _tid, pair in reg.items()
                if (
                    (norm_stage == "e2e" and len(pair[1].stage_blocks) > 0)
                    or (norm_stage != "e2e" and pair[1].stages == [norm_stage])
                )
            ),
            None,
        )
        if matched is None:
            raise KeyError(
                f"Unable to resolve rubric template for task_id={task_id!r}, stage={stage!r}"
            )
        ec = matched[0].model_copy(deep=True)
        rb = matched[1].model_copy(deep=True)
        rb.task_id = task_id

    if isinstance(ground_truth.get("pass_reward_min"), (int, float)) and not isinstance(
        ground_truth.get("pass_reward_min"), bool
    ):
        rb.pass_reward_min = float(ground_truth["pass_reward_min"])

    if isinstance(ground_truth.get("verifier_tclean_recipe"), dict):
        ec.geometry.verifier_tclean_recipe = VerifierTcleanRecipe.model_validate(
            ground_truth["verifier_tclean_recipe"]
        )

    return ec, rb


def grade_inspect(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    task_id: str,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade `/workspace/inspect_summary.json` or `/workspace/stage_output.json`."""
    ec, rb = resolve_task_configs(task_id, "inspect", ground_truth)
    return run_verification(
        workspace_dir=Path(workspace_dir),
        tests_dir=Path(workspace_dir),
        log_dir=None,
        extract_config=ec,
        rubric=rb,
        ground_truth=ground_truth,
        compute_raw_hash=(Path(workspace_dir) / "input.ms" / "table.dat").is_file(),
    )


__all__ = [
    "PASS_REWARD_MIN_DEFAULT",
    "grade_inspect",
    "has_active_table_lock",
    "resolve_task_configs",
]
