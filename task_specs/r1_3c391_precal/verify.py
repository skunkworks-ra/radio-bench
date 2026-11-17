"""Isolated Harbor scientific verifier entrypoint for r1_3c391_precal (precal)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

TASK_ID = "r1_3c391_precal"
STAGE = "precal"


def _configure_temp_env() -> Path:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur_tmp = os.environ.get("TMPDIR", "").strip()
    if not cur_tmp or cur_tmp == "/tmp" or cur_tmp.startswith("/tmp/"):
        var_tmp = Path("/var/tmp")
        scratch = var_tmp / f"verifier_{TASK_ID}_{os.getpid()}"
    else:
        scratch = Path(cur_tmp) / f"verifier_{TASK_ID}_{os.getpid()}"

    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)

    cur_mpl = os.environ.get("MPLCONFIGDIR", "").strip()
    if not cur_mpl or cur_mpl == "/tmp" or cur_mpl.startswith("/tmp/"):
        mpl_dir = scratch / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    if not os.access(os.environ.get("HOME", "/"), os.W_OK):
        os.environ["HOME"] = str(scratch)

    return scratch


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Verify {TASK_ID} ({STAGE}).")
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--tests-dir", type=Path, default=Path("/tests"))
    parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()

    workspace_dir = args.workspace.resolve()
    tests_dir = args.tests_dir.resolve()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    scratch_dir = _configure_temp_env()
    prev_cwd = Path.cwd()
    try:
        os.chdir(scratch_dir)
        from radio_bench.grading import grade_task

        reward_dict, _ = grade_task(
            task_id=TASK_ID,
            stage=STAGE,
            workspace_dir=workspace_dir,
            tests_dir=tests_dir,
            log_dir=log_dir,
            scratch_root=scratch_dir,
        )
        reward_path = log_dir / "reward.json"
        if reward_path.exists():
            reward_payload = json.loads(reward_path.read_text(encoding="utf-8"))
            reward_val = float(reward_payload.get("reward", 0.0))
            passed_val = int(reward_payload.get("passed", 0))
        else:
            reward_val = float(reward_dict.get("reward", 0.0))
            passed_val = int(reward_dict.get("passed", int(reward_val >= 0.85)))
            reward_path.write_text(
                json.dumps({"reward": reward_val, "passed": passed_val}, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "stage": STAGE,
                    "reward": round(reward_val, 6),
                    "passed": passed_val,
                }
            )
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        reward_path = log_dir / "reward.json"
        if not reward_path.exists():
            reward_path.write_text(
                json.dumps({"reward": 0.0, "passed": 0}, indent=2) + "\n",
                encoding="utf-8",
            )
        metrics_path = log_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path.write_text(
                json.dumps(
                    {
                        "task_id": TASK_ID,
                        "stage": STAGE,
                        "reward": 0.0,
                        "passed": False,
                        "verifier_exception": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps({"task_id": TASK_ID, "stage": STAGE, "reward": 0.0, "passed": 0}))
    finally:
        with contextlib.suppress(Exception):
            os.chdir(prev_cwd)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
