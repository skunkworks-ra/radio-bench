"""Generic isolated Harbor verifier runner (`python -m radio_bench.grading.runner`)."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.config import is_under_slash_tmp, reject_slash_tmp
from radio_bench.grading.extractors import extract_workspace, resolve_scratch_root
from radio_bench.grading.rubric import RubricEvaluationResult, evaluate_rubric
from radio_bench.grading.schema import (
    ExtractConfig,
    RubricSpec,
    loads_strict_json,
)

PASS_REWARD_MIN_DEFAULT = 0.85


def _sanitize_json_payload(
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
        return _sanitize_json_payload(val.tolist(), path_prefix, non_finite_fields)
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        for k, sub_v in val.items():
            ks = str(k)
            c_path = f"{path_prefix}.{ks}" if path_prefix else ks
            out[ks] = _sanitize_json_payload(sub_v, c_path, non_finite_fields)
        return out
    if isinstance(val, (list, tuple, set)):
        return [
            _sanitize_json_payload(
                item,
                f"{path_prefix}[{idx}]" if path_prefix else f"[{idx}]",
                non_finite_fields,
            )
            for idx, item in enumerate(val)
        ]
    return str(val)


def atomic_write_json(target_path: Path, payload: dict[str, Any]) -> None:
    """Atomically serialize strict JSON (`allow_nan=False`) to `target_path`."""
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
    """Validate Harbor-safe flat finite `reward_dict` and write `reward.json` + `metrics.json`."""
    if not isinstance(reward_dict, dict):
        raise ValueError(f"reward_dict must be a dict, got {type(reward_dict).__name__}")
    if "reward" not in reward_dict:
        raise ValueError("reward_dict must include 'reward'")

    validated: dict[str, float | int] = {}
    for key, val in reward_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"reward_dict keys must be str, got {key!r}")
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"Non-numeric or boolean reward value for key {key!r}: {val!r}")
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite reward value for key {key!r}: {val!r}")
        if isinstance(val, (int, np.integer)):
            validated[key] = int(val)
        else:
            validated[key] = round(fval, 6)

    raw_pass_min = (
        metrics_dict.get("pass_reward_min", PASS_REWARD_MIN_DEFAULT)
        if isinstance(metrics_dict, dict)
        else PASS_REWARD_MIN_DEFAULT
    )
    pass_min = (
        float(raw_pass_min)
        if isinstance(raw_pass_min, (int, float))
        and not isinstance(raw_pass_min, bool)
        and math.isfinite(float(raw_pass_min))
        else PASS_REWARD_MIN_DEFAULT
    )

    clamped_reward = round(min(1.0, max(0.0, float(validated["reward"]))), 6)
    stage_passed_ok = True
    if "passed" in validated:
        stage_passed_ok = int(round(float(validated["passed"]))) != 0
    passed_int = 1 if (clamped_reward >= pass_min and stage_passed_ok) else 0

    clean_reward: dict[str, float | int] = {
        "reward": clamped_reward,
        "passed": int(passed_int),
    }
    for key, val in validated.items():
        if key in ("reward", "passed"):
            continue
        clean_reward[key] = val

    non_finite_fields: list[str] = []
    sanitized_metrics = _sanitize_json_payload(
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
    atomic_write_json(out_dir / "reward.json", clean_reward)
    atomic_write_json(out_dir / "metrics.json", sanitized_metrics)


def build_v02_metrics_dict(
    *,
    extract_config: ExtractConfig,
    rubric: RubricSpec,
    eval_result: RubricEvaluationResult,
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the self-describing `v0.2` `/logs/verifier/metrics.json` payload."""
    canonical_stage = (
        "e2e"
        if (rubric.stage_blocks or len(rubric.stages) > 1)
        else ("image" if rubric.stages[0] == "image_cont" else rubric.stages[0])
    )
    measurements = {k: rec.model_dump() for k, rec in eval_result.observables.items()}
    measured_flat = {
        k: rec.value
        for k, rec in eval_result.observables.items()
        if rec.status == "measured" and rec.value is not None
    }
    table_locked = bool(
        eval_result.observables.get("proc.table_locked") is not None
        and eval_result.observables["proc.table_locked"].value is True
    )

    return {
        "schema_version": "0.2",
        "task_id": rubric.task_id,
        "suite_id": extract_config.suite_id,
        "split": extract_config.split,
        "stage": canonical_stage,
        "stages": list(rubric.stages),
        "pass_reward_min": float(rubric.pass_reward_min),
        "table_locked": table_locked,
        "reward": eval_result.reward,
        "passed": bool(eval_result.passed == 1),
        "measurements": measurements,
        "measured": measured_flat,
        "truth": ground_truth,
        "subscores": dict(eval_result.subscores),
        "rubric_meta": eval_result.rubric_meta,
        "triggered_failure_tags": list(eval_result.triggered_failure_tags),
        "diff_summary": {},
        "errors": list(eval_result.errors),
    }


def run_verification(
    workspace_dir: Path,
    tests_dir: Path,
    log_dir: Path | None = None,
    *,
    extract_config: ExtractConfig | None = None,
    rubric: RubricSpec | None = None,
    ground_truth: dict[str, Any] | None = None,
    ref_arrays_dir: Path | None = None,
    ref_checkpoint_dir: Path | None = None,
    scratch_root: Path | None = None,
    compute_raw_hash: bool = True,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Execute `extract_workspace` + `evaluate_rubric` and optionally write `/logs/verifier/*`."""
    ws = Path(workspace_dir).resolve()
    td = Path(tests_dir).resolve()
    reject_slash_tmp(ws, label="workspace")
    reject_slash_tmp(td, label="tests_dir")
    if log_dir is not None:
        reject_slash_tmp(Path(log_dir), label="log_dir")

    if extract_config is None:
        ec_path = td / "extract_config.json"
        if not ec_path.is_file():
            raise FileNotFoundError(f"Missing extract_config.json at {ec_path}")
        extract_config = ExtractConfig.model_validate(
            loads_strict_json(ec_path.read_text(encoding="utf-8"))
        )

    if rubric is None:
        rb_path = td / "rubric.json"
        if not rb_path.is_file():
            raise FileNotFoundError(f"Missing rubric.json at {rb_path}")
        rubric = RubricSpec.model_validate(loads_strict_json(rb_path.read_text(encoding="utf-8")))

    if ground_truth is None:
        gt_path = td / "ground_truth.json"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing ground_truth.json at {gt_path}")
        parsed_gt = loads_strict_json(gt_path.read_text(encoding="utf-8"))
        if not isinstance(parsed_gt, dict):
            raise ValueError(f"ground_truth.json at {gt_path} must be a JSON object")
        ground_truth = parsed_gt

    eff_ref_dir = ref_arrays_dir if ref_arrays_dir is not None else td
    resolved_scratch = resolve_scratch_root(scratch_root)

    extraction = extract_workspace(
        ws,
        extract_config,
        eff_ref_dir,
        scratch_root=resolved_scratch,
        compute_raw_hash=compute_raw_hash,
    )
    eval_res = evaluate_rubric(ws, extraction, rubric, ground_truth)
    metrics_dict = build_v02_metrics_dict(
        extract_config=extract_config,
        rubric=rubric,
        eval_result=eval_res,
        ground_truth=ground_truth,
    )

    eff_ref_ckpt = (
        Path(ref_checkpoint_dir).resolve()
        if ref_checkpoint_dir is not None
        else (td / "ref_checkpoint")
    )
    if eff_ref_ckpt.is_dir():
        has_ckpt_entries = False
        with contextlib.suppress(OSError):
            has_ckpt_entries = any(True for _ in eff_ref_ckpt.iterdir())
        if has_ckpt_entries:
            try:
                from radio_bench.analysis.diff import diff_workspaces

                metrics_dict["diff_summary"] = diff_workspaces(
                    ws,
                    eff_ref_ckpt,
                    stage=str(metrics_dict.get("stage") or ""),
                    base_task=rubric.task_id,
                    extract_config=extract_config,
                    scratch_root=resolved_scratch,
                )
            except Exception as diff_exc:
                metrics_dict["diff_summary"] = {
                    "status": "diff_error",
                    "error": f"{type(diff_exc).__name__}: {diff_exc}",
                }

    if log_dir is not None:
        write_verifier_outputs(Path(log_dir), eval_res.reward_dict, metrics_dict)

    return eval_res.reward_dict, metrics_dict


def configure_container_scratch_env(task_hint: str = "runner") -> Path:
    """Configure non-`/tmp` scratch environment (`OMP_WAIT_POLICY=PASSIVE`, `TMPDIR`, `MPLCONFIGDIR`)."""
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur_tmp = os.environ.get("TMPDIR", "").strip()
    in_container = Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
    if not cur_tmp or is_under_slash_tmp(cur_tmp):
        var_tmp = Path("/var/tmp")
        if in_container and var_tmp.exists() and os.access(var_tmp, os.W_OK):
            scratch = var_tmp / f"verifier_{task_hint}_{os.getpid()}"
        else:
            from radio_bench.config import load_bench_config

            scratch = load_bench_config().tmp_dir / f"verifier_{task_hint}_{os.getpid()}"
    else:
        scratch = Path(cur_tmp).resolve() / f"verifier_{task_hint}_{os.getpid()}"

    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)

    cur_mpl = os.environ.get("MPLCONFIGDIR", "").strip()
    if not cur_mpl or is_under_slash_tmp(cur_mpl):
        mpl_dir = scratch / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    if not os.access(os.environ.get("HOME", "/"), os.W_OK):
        os.environ["HOME"] = str(scratch)

    if not os.environ.get("CASASITECONFIG", "").strip():
        etc_cfg = Path("/etc/casa/config.py")
        if etc_cfg.is_file():
            os.environ["CASASITECONFIG"] = str(etc_cfg)
        else:
            with contextlib.suppress(Exception):
                from radio_bench.config import load_bench_config

                host_cfg = load_bench_config().casaconfig_path
                if host_cfg.is_file():
                    os.environ["CASASITECONFIG"] = str(host_cfg)

    return scratch


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `/tests/test.sh` inside the isolated `verifier` container."""
    parser = argparse.ArgumentParser(description="Run radio-bench declarative rubric verifier.")
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--tests-dir", type=Path, default=Path("/tests"))
    parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args(argv)

    workspace_dir = args.workspace.resolve()
    tests_dir = args.tests_dir.resolve()
    log_dir = args.log_dir.resolve()
    reject_slash_tmp(workspace_dir, label="workspace")
    reject_slash_tmp(tests_dir, label="tests_dir")
    reject_slash_tmp(log_dir, label="log_dir")

    scratch_dir: Path | None = None
    prev_cwd: Path | None = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir = configure_container_scratch_env()
        with contextlib.suppress(Exception):
            prev_cwd = Path.cwd()
        os.chdir(scratch_dir)
        reward_dict, metrics_dict = run_verification(
            workspace_dir=workspace_dir,
            tests_dir=tests_dir,
            log_dir=log_dir,
            scratch_root=scratch_dir,
        )
        print(
            json.dumps(
                {
                    "task_id": metrics_dict.get("task_id"),
                    "stage": metrics_dict.get("stage"),
                    "reward": reward_dict["reward"],
                    "passed": reward_dict["passed"],
                }
            )
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        err_reward: dict[str, float | int] = {"reward": 0.0, "passed": 0}
        err_metrics: dict[str, Any] = {
            "schema_version": "0.2",
            "task_id": "unknown",
            "stage": "unknown",
            "pass_reward_min": PASS_REWARD_MIN_DEFAULT,
            "reward": 0.0,
            "passed": False,
            "errors": [f"Unhandled verifier exception ({type(exc).__name__}): {exc}"],
        }
        with contextlib.suppress(Exception):
            write_verifier_outputs(log_dir, err_reward, err_metrics)
        if not (log_dir / "reward.json").is_file():
            with contextlib.suppress(Exception):
                log_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(log_dir / "reward.json", {"reward": 0.0, "passed": 0})
        print(json.dumps({"reward": 0.0, "passed": 0, "error": f"{type(exc).__name__}: {exc}"}))
    finally:
        if prev_cwd is not None:
            with contextlib.suppress(Exception):
                os.chdir(prev_cwd)
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
