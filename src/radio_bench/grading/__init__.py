"""Public scientific grading and Harbor `reward.json` / `metrics.json` contract."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.grading.cal_metrics import grade_calibrate, resolve_scratch_root
from radio_bench.grading.image_metrics import grade_image
from radio_bench.grading.inspect_metrics import grade_inspect
from radio_bench.grading.precal_metrics import grade_precal

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


def _sanitize_metrics_value(
    val: Any,
    path_prefix: str,
    non_finite_fields: list[str],
) -> Any:
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        fv = float(val)
        if not math.isfinite(fv):
            non_finite_fields.append(path_prefix or "<root>")
            return None
        return fv
    if isinstance(val, (complex, np.complexfloating)):
        rv, iv = float(val.real), float(val.imag)
        if not (math.isfinite(rv) and math.isfinite(iv)):
            non_finite_fields.append(path_prefix or "<root>")
            return None
        return {"real": rv, "imag": iv}
    if isinstance(val, str):
        return val
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, np.ndarray):
        return _sanitize_metrics_value(val.tolist(), path_prefix, non_finite_fields)
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        for k, sub_v in val.items():
            ks = str(k)
            child_path = f"{path_prefix}.{ks}" if path_prefix else ks
            out[ks] = _sanitize_metrics_value(sub_v, child_path, non_finite_fields)
        return out
    if isinstance(val, (list, tuple, set)):
        out_list: list[Any] = []
        for idx, item in enumerate(val):
            child_path = f"{path_prefix}[{idx}]" if path_prefix else f"[{idx}]"
            out_list.append(_sanitize_metrics_value(item, child_path, non_finite_fields))
        return out_list
    return str(val)


def _atomic_write_json(target_path: Path, payload: dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp_path = target_path.parent / f".{target_path.name}.{os.getpid()}.tmp"
    try:
        tmp_path.write_text(serialized, encoding="utf-8")
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_verifier_outputs(
    log_dir: Path,
    reward_dict: dict[str, float | int],
    metrics_dict: dict[str, Any],
) -> None:
    """Validate flat finite numeric `reward_dict` and write `reward.json` + `metrics.json`."""
    if not isinstance(reward_dict, dict):
        raise ValueError(f"reward_dict must be a dict, got {type(reward_dict).__name__}")
    if "reward" not in reward_dict:
        raise ValueError("reward_dict must include 'reward'")

    validated_subscores: dict[str, float | int] = {}
    for key, val in reward_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"reward_dict keys must be str, got {key!r}")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"Non-numeric or boolean reward value for key {key!r}: {val!r}")
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite reward value for key {key!r}: {val!r}")
        if isinstance(val, int):
            validated_subscores[key] = int(val)
        else:
            validated_subscores[key] = round(fval, 6)

    raw_pass_min = (
        metrics_dict.get("pass_reward_min", PASS_REWARD_MIN)
        if isinstance(metrics_dict, dict)
        else PASS_REWARD_MIN
    )
    pass_min = (
        float(raw_pass_min)
        if isinstance(raw_pass_min, (int, float))
        and not isinstance(raw_pass_min, bool)
        and math.isfinite(float(raw_pass_min))
        else PASS_REWARD_MIN
    )

    clamped_reward = round(min(1.0, max(0.0, float(validated_subscores["reward"]))), 6)
    stage_passed_ok = True
    if "passed" in validated_subscores:
        stage_passed_ok = int(round(float(validated_subscores["passed"]))) != 0
    passed_int = 1 if (clamped_reward >= pass_min and stage_passed_ok) else 0

    clean_reward: dict[str, float | int] = {
        "reward": clamped_reward,
        "passed": int(passed_int),
    }
    for key, val in validated_subscores.items():
        if key in ("reward", "passed"):
            continue
        clean_reward[key] = val

    non_finite_fields: list[str] = []
    sanitized_metrics = _sanitize_metrics_value(
        metrics_dict if isinstance(metrics_dict, dict) else {"raw_metrics": metrics_dict},
        "",
        non_finite_fields,
    )
    if not isinstance(sanitized_metrics, dict):
        sanitized_metrics = {"data": sanitized_metrics}
    sanitized_metrics["reward"] = clamped_reward
    sanitized_metrics["passed"] = bool(passed_int == 1)
    existing_nf = sanitized_metrics.get("non_finite_fields")
    if isinstance(existing_nf, list):
        merged_nf = list(dict.fromkeys([*(str(x) for x in existing_nf), *non_finite_fields]))
    else:
        merged_nf = non_finite_fields
    sanitized_metrics["non_finite_fields"] = merged_nf

    out_dir = Path(log_dir)
    _atomic_write_json(out_dir / "reward.json", clean_reward)
    _atomic_write_json(out_dir / "metrics.json", sanitized_metrics)


def grade_e2e(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path | None = None,
    task_id: str,
    scratch_root: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade end-to-end tasks (`s2_e2e_synth`, `r2_3c391_e2e`): 0.60 calibration + 0.40 image."""
    weights_raw = (
        ground_truth.get("weights") if isinstance(ground_truth.get("weights"), dict) else {}
    )
    tols = (
        ground_truth.get("tolerances") if isinstance(ground_truth.get("tolerances"), dict) else {}
    )

    cal_weight = float(weights_raw.get("calibration_score", tols.get("calibration_weight", 0.60)))
    img_weight = float(weights_raw.get("delivered_image_score", tols.get("imaging_weight", 0.40)))
    pass_min = float(
        ground_truth.get("pass_reward_min", tols.get("pass_reward_min", PASS_REWARD_MIN))
    )

    cal_reward_dict, cal_metrics = grade_calibrate(
        workspace_dir,
        ground_truth,
        masks_npz_path=masks_npz_path,
        task_id=task_id,
        scratch_root=scratch_root,
    )
    img_reward_dict, img_metrics = grade_image(
        workspace_dir,
        ground_truth,
        masks_npz_path=masks_npz_path,
        task_id=task_id,
    )

    table_locked = bool(cal_metrics.get("table_locked") or img_metrics.get("table_locked"))
    cal_score = float(cal_reward_dict["reward"])
    img_score = float(img_reward_dict["reward"])
    composite = 0.0 if table_locked else (cal_weight * cal_score + img_weight * img_score)
    reward = round(min(1.0, max(0.0, composite)), 6)
    passed = 1 if (reward >= pass_min and not table_locked) else 0

    reward_dict: dict[str, float | int] = {
        "reward": reward,
        "passed": passed,
        "calibration_score": round(cal_score, 6),
        "delivered_image_score": round(img_score, 6),
    }
    for k, v in cal_reward_dict.items():
        if k not in ("reward", "passed"):
            reward_dict[f"cal_{k}"] = v
    for k, v in img_reward_dict.items():
        if k not in ("reward", "passed"):
            reward_dict[f"img_{k}"] = v

    errors = [
        *(str(e) for e in cal_metrics.get("errors", [])),
        *(str(e) for e in img_metrics.get("errors", [])),
    ]
    metrics_dict: dict[str, Any] = {
        "task_id": task_id,
        "stage": "e2e",
        "pass_reward_min": pass_min,
        "table_locked": table_locked,
        "weights": {
            "calibration_score": cal_weight,
            "delivered_image_score": img_weight,
        },
        "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
        "calibration": cal_metrics,
        "image": img_metrics,
        "errors": errors,
    }
    return reward_dict, metrics_dict


def _discover_masks_npz(tests_dir: Path, ground_truth: dict[str, Any], stage: str) -> Path | None:
    candidates: list[str] = []
    if stage == "precal":
        for k in ("corruption_masks_file", "precal_masks_file"):
            if isinstance(ground_truth.get(k), str):
                candidates.append(str(ground_truth[k]))
        candidates.extend(["corruption_masks.npz", "precal_masks.npz"])
    elif stage in ("calibrate", "e2e"):
        for k in ("true_target_vis_file", "corruption_masks_file", "precal_masks_file"):
            if isinstance(ground_truth.get(k), str):
                candidates.append(str(ground_truth[k]))
        candidates.extend(["true_target_vis.npz", "corruption_masks.npz", "precal_masks.npz"])
    else:
        candidates.extend(["corruption_masks.npz", "precal_masks.npz", "true_target_vis.npz"])

    for fname in candidates:
        cand = tests_dir / fname
        if cand.is_file():
            return cand
    return None


def grade_task(
    task_id: str,
    stage: str,
    workspace_dir: Path,
    tests_dir: Path,
    log_dir: Path,
    *,
    scratch_root: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Unified stage dispatcher loading `tests_dir/ground_truth.json` and writing `/logs/verifier/*`."""
    ws = Path(workspace_dir)
    td = Path(tests_dir)
    ld = Path(log_dir)
    stage_norm = str(stage).strip().lower()

    gt_path = td / "ground_truth.json"
    if not gt_path.is_file():
        err_reward: dict[str, float | int] = {"reward": 0.0, "passed": 0}
        err_metrics: dict[str, Any] = {
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
            "task_id": task_id,
            "stage": stage_norm,
            "pass_reward_min": PASS_REWARD_MIN,
            "errors": [f"Invalid ground_truth.json at {gt_path}: {exc}"],
        }
        write_verifier_outputs(ld, err_reward, err_metrics)
        return err_reward, err_metrics

    try:
        resolved_scratch = resolve_scratch_root(scratch_root)
        masks_npz = _discover_masks_npz(td, ground_truth, stage_norm)

        if stage_norm == "inspect":
            reward_dict, metrics_dict = grade_inspect(ws, ground_truth, task_id=task_id)
        elif stage_norm == "precal":
            fallback_npz = masks_npz if masks_npz is not None else (td / "corruption_masks.npz")
            reward_dict, metrics_dict = grade_precal(
                ws,
                ground_truth,
                masks_npz_path=fallback_npz,
                task_id=task_id,
            )
        elif stage_norm in ("calibrate", "cal"):
            reward_dict, metrics_dict = grade_calibrate(
                ws,
                ground_truth,
                masks_npz_path=masks_npz,
                task_id=task_id,
                scratch_root=resolved_scratch,
            )
        elif stage_norm == "image":
            reward_dict, metrics_dict = grade_image(
                ws,
                ground_truth,
                masks_npz_path=masks_npz,
                task_id=task_id,
            )
        elif stage_norm in ("e2e", "end_to_end"):
            reward_dict, metrics_dict = grade_e2e(
                ws,
                ground_truth,
                masks_npz_path=masks_npz,
                task_id=task_id,
                scratch_root=resolved_scratch,
            )
        else:
            reward_dict = {"reward": 0.0, "passed": 0}
            metrics_dict = {
                "task_id": task_id,
                "stage": stage_norm,
                "pass_reward_min": PASS_REWARD_MIN,
                "errors": [f"Unknown grading stage: {stage!r}"],
            }
    except Exception as exc:
        reward_dict = {"reward": 0.0, "passed": 0}
        metrics_dict = {
            "task_id": task_id,
            "stage": stage_norm,
            "pass_reward_min": PASS_REWARD_MIN,
            "errors": [f"Unhandled verifier exception ({type(exc).__name__}): {exc}"],
        }

    write_verifier_outputs(ld, reward_dict, metrics_dict)
    return reward_dict, metrics_dict
