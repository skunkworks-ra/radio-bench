"""Public scientific grading and Harbor `reward.json` / `metrics.json` contract.

Exports `runner` and stage-wrapper functions lazily via `__getattr__` so invoking
`python -m radio_bench.grading.runner` never pre-populates `radio_bench.grading.runner`
in `sys.modules` (`<frozen runpy>:128: RuntimeWarning`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PASS_REWARD_MIN = 0.85

__all__ = [
    "PASS_REWARD_MIN",
    "grade_calibrate",
    "grade_e2e",
    "grade_image",
    "grade_inspect",
    "grade_precal",
    "grade_task",
    "write_verifier_outputs",
]


def grade_e2e(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path | None = None,
    task_id: str,
    scratch_root: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade end-to-end tasks (`s2_e2e_synth`, `r2_3c391_e2e`) via `extractors` + `rubric`."""
    import radio_bench.grading.cal_metrics as _cm
    import radio_bench.grading.extractors.calibrate as _cal_ext
    from radio_bench.grading.cal_metrics import resolve_scratch_root
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.runner import run_verification

    ws = Path(workspace_dir)
    ec, rb = resolve_task_configs(task_id, "e2e", ground_truth)
    ref_dir = Path(masks_npz_path) if masks_npz_path is not None else None
    orig_ext_runner = _cal_ext._run_verifier_tclean_recipe
    try:
        if _cm._run_verifier_tclean_recipe is not orig_ext_runner:
            _cal_ext._run_verifier_tclean_recipe = _cm._run_verifier_tclean_recipe
        return run_verification(
            workspace_dir=ws,
            tests_dir=(ref_dir if (ref_dir is not None and ref_dir.is_dir()) else ws),
            log_dir=None,
            extract_config=ec,
            rubric=rb,
            ground_truth=ground_truth,
            ref_arrays_dir=ref_dir,
            scratch_root=resolve_scratch_root(scratch_root),
            compute_raw_hash=(ws / "input.ms" / "table.dat").is_file(),
        )
    finally:
        _cal_ext._run_verifier_tclean_recipe = orig_ext_runner


def grade_task(
    task_id: str,
    stage: str,
    workspace_dir: Path,
    tests_dir: Path,
    log_dir: Path,
    *,
    scratch_root: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Unified stage dispatcher evaluating `extract_config.json` + `rubric.json` + `ground_truth.json`."""
    import radio_bench.grading.cal_metrics as _cm
    import radio_bench.grading.extractors.calibrate as _cal_ext
    from radio_bench.grading.cal_metrics import resolve_scratch_root
    from radio_bench.grading.inspect_metrics import resolve_task_configs
    from radio_bench.grading.runner import run_verification, write_verifier_outputs
    from radio_bench.grading.schema import ExtractConfig, RubricSpec, loads_strict_json

    ws = Path(workspace_dir)
    td = Path(tests_dir)
    ld = Path(log_dir)
    stage_norm = str(stage).strip().lower()
    if stage_norm == "cal":
        stage_norm = "calibrate"
    elif stage_norm == "end_to_end":
        stage_norm = "e2e"

    if stage_norm not in {
        "inspect",
        "precal",
        "calibrate",
        "image",
        "image_cont",
        "e2e",
        "postcal",
        "polcal",
        "selfcal",
        "image_cube",
        "simulate",
        "report",
    }:
        err_reward: dict[str, float | int] = {"reward": 0.0, "passed": 0}
        err_metrics: dict[str, Any] = {
            "schema_version": "0.2",
            "task_id": task_id,
            "stage": stage_norm,
            "pass_reward_min": PASS_REWARD_MIN,
            "errors": [f"Unknown grading stage: {stage!r}"],
        }
        write_verifier_outputs(ld, err_reward, err_metrics)
        return err_reward, err_metrics

    gt_path = td / "ground_truth.json"
    if not gt_path.is_file():
        err_reward = {"reward": 0.0, "passed": 0}
        err_metrics = {
            "schema_version": "0.2",
            "task_id": task_id,
            "stage": stage_norm,
            "pass_reward_min": PASS_REWARD_MIN,
            "errors": [f"Missing ground_truth.json at {gt_path}"],
        }
        write_verifier_outputs(ld, err_reward, err_metrics)
        return err_reward, err_metrics

    try:
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        if not isinstance(ground_truth, dict):
            raise ValueError("ground_truth.json root must be a JSON object")
    except Exception as exc:
        err_reward = {"reward": 0.0, "passed": 0}
        err_metrics = {
            "schema_version": "0.2",
            "task_id": task_id,
            "stage": stage_norm,
            "pass_reward_min": PASS_REWARD_MIN,
            "errors": [f"Invalid ground_truth.json at {gt_path}: {exc}"],
        }
        write_verifier_outputs(ld, err_reward, err_metrics)
        return err_reward, err_metrics

    orig_ext_runner = _cal_ext._run_verifier_tclean_recipe
    try:
        if _cm._run_verifier_tclean_recipe is not orig_ext_runner:
            _cal_ext._run_verifier_tclean_recipe = _cm._run_verifier_tclean_recipe

        ec_path = td / "extract_config.json"
        rb_path = td / "rubric.json"
        if ec_path.is_file() and rb_path.is_file():
            ec = ExtractConfig.model_validate(
                loads_strict_json(ec_path.read_text(encoding="utf-8"))
            )
            rb = RubricSpec.model_validate(loads_strict_json(rb_path.read_text(encoding="utf-8")))
        else:
            ec, rb = resolve_task_configs(task_id, stage_norm, ground_truth)

        reward_dict, metrics_dict = run_verification(
            workspace_dir=ws,
            tests_dir=td,
            log_dir=ld,
            extract_config=ec,
            rubric=rb,
            ground_truth=ground_truth,
            ref_arrays_dir=td,
            scratch_root=resolve_scratch_root(scratch_root),
            compute_raw_hash=(ws / "input.ms" / "table.dat").is_file(),
        )
        return reward_dict, metrics_dict
    except Exception as exc:
        err_reward = {"reward": 0.0, "passed": 0}
        err_metrics = {
            "schema_version": "0.2",
            "task_id": task_id,
            "stage": stage_norm,
            "pass_reward_min": PASS_REWARD_MIN,
            "errors": [f"Unhandled verifier exception ({type(exc).__name__}): {exc}"],
        }
        write_verifier_outputs(ld, err_reward, err_metrics)
        return err_reward, err_metrics
    finally:
        _cal_ext._run_verifier_tclean_recipe = orig_ext_runner


def __getattr__(name: str) -> Any:
    if name == "grade_calibrate":
        from radio_bench.grading.cal_metrics import grade_calibrate

        return grade_calibrate
    if name == "grade_image":
        from radio_bench.grading.image_metrics import grade_image

        return grade_image
    if name == "grade_inspect":
        from radio_bench.grading.inspect_metrics import grade_inspect

        return grade_inspect
    if name == "grade_precal":
        from radio_bench.grading.precal_metrics import grade_precal

        return grade_precal
    if name == "write_verifier_outputs":
        from radio_bench.grading.runner import write_verifier_outputs

        return write_verifier_outputs
    if name == "run_verification":
        from radio_bench.grading.runner import run_verification

        return run_verification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
