"""Unified portable CLI dispatcher for radio-bench tasks."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from radio_bench.config import (
    BenchConfig,
    add_config_cli_args,
    config_from_cli_args,
    prepare_runtime_env,
)
from radio_bench.podman_env import ensure_docker_compose_plugin


def _prepare_env_with_compose(
    cfg: BenchConfig,
    *,
    config_path: Path | None = None,
) -> dict[str, str]:
    env = prepare_runtime_env(cfg, config_path=config_path)
    compose_cfg = ensure_docker_compose_plugin(cfg.docker_config_dir)
    if compose_cfg is not None:
        env["DOCKER_CONFIG"] = str(compose_cfg)
    pixi_bin = str(Path(sys.executable).resolve().parent)
    cur_path = env.get("PATH", "")
    parts = cur_path.split(os.pathsep) if cur_path else []
    if pixi_bin not in parts:
        env["PATH"] = os.pathsep.join([pixi_bin, *parts])
    src_dir = str(cfg.repo_root / "src")
    cur_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{cur_pypath}" if cur_pypath else src_dir
    return env


def _run_child(cmd: list[str], *, cfg: BenchConfig, env: dict[str, str]) -> int:
    proc = subprocess.run(cmd, cwd=str(cfg.work_dir), env=env, check=False)
    return int(proc.returncode)


def _resolve_relative_caller_paths(
    extra_args: Sequence[str],
    caller_cwd: Path,
) -> list[str]:
    """Resolve relative existing directory/file arguments against `caller_cwd` before chdir to `work_dir`."""
    resolved: list[str] = []
    for arg in extra_args:
        if arg.startswith("-"):
            resolved.append(arg)
            continue
        cand = Path(arg)
        if not cand.is_absolute() and (caller_cwd / cand).exists():
            resolved.append(str((caller_cwd / cand).resolve()))
        else:
            resolved.append(arg)
    return resolved


def _normalize_pytest_args(extra_args: Sequence[str], cfg: BenchConfig) -> list[str]:
    normalized: list[str] = []
    has_target = False
    has_verbosity = False

    skip_next = False
    opts_taking_value = {
        "-k",
        "-m",
        "-o",
        "-c",
        "--basetemp",
        "--rootdir",
        "--maxfail",
        "--tb",
        "--durations",
        "-n",
    }

    for arg in extra_args:
        if arg == "--":
            continue
        if skip_next:
            normalized.append(arg)
            skip_next = False
            continue
        if arg in opts_taking_value:
            normalized.append(arg)
            skip_next = True
            continue
        if arg in {"-v", "-vv", "-q", "--quiet", "--verbose"}:
            has_verbosity = True
            normalized.append(arg)
            continue
        if arg.startswith("-"):
            normalized.append(arg)
            continue

        has_target = True
        path_part, sep, node_part = arg.partition("::")
        cand = Path(path_part)
        if not cand.is_absolute() and (cfg.repo_root / cand).exists():
            resolved_target = str(cfg.repo_root / cand)
            normalized.append(f"{resolved_target}{sep}{node_part}" if sep else resolved_target)
        else:
            normalized.append(arg)

    if not has_verbosity:
        normalized.append("-v")
    if not has_target:
        normalized.append(str(cfg.repo_root / "tests"))
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    caller_cwd = Path.cwd()
    parser = argparse.ArgumentParser(
        prog="radio-bench",
        description="Portable task runner for radio-bench.",
    )
    add_config_cli_args(parser)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    subparsers.add_parser("podman-check", help="Verify rootless Podman, Compose v2, and storage.")
    subparsers.add_parser("podman-clean", help="Remove leaked Harbor containers/volumes/networks.")
    subparsers.add_parser(
        "podman-smoke-compose",
        help="Smoke-test offline verifier and pasta sidecar Compose lifecycle.",
    )
    subparsers.add_parser(
        "build-images", help="Build container images via radio_bench.build_images."
    )
    subparsers.add_parser("build-synth", help="Build Suite S synthetic Measurement Sets.")
    subparsers.add_parser("build-checkpoints", help="Build Suite R 3C391 golden checkpoints.")
    subparsers.add_parser("export-skills", help="Export skill trees from configured pack.")
    subparsers.add_parser("build-tasks", help="Compile Harbor 0.25.0 task directories.")
    subparsers.add_parser("render-jobs", help="Render Harbor 0.25.0 job YAML files.")
    subparsers.add_parser("harbor-lint", help="Run harbor lint across compiled task directories.")
    subparsers.add_parser("analyze", help="Analyze one or more Harbor job or trial directories.")

    oracle_parser = subparsers.add_parser(
        "run-oracle",
        help="Render canonical oracle job config(s) and execute non-LLM Harbor oracle verification.",
    )
    oracle_parser.add_argument(
        "--suite",
        choices=("synth", "3c391", "all"),
        default="all",
        help="Benchmark suite to run with Harbor oracle agent (default: all).",
    )

    subparsers.add_parser("test-unit", help="Run pytest unit suite with work_dir scratch/cache.")

    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    # Also allow `--config` / `--work-dir` / `--data-dir` / `--runs-dir` placed AFTER the subcommand!
    post_cfg_parser = argparse.ArgumentParser(add_help=False)
    add_config_cli_args(post_cfg_parser)
    post_args, extra = post_cfg_parser.parse_known_args(extra)
    for attr in ("config", "work_dir", "data_dir", "runs_dir"):
        if getattr(post_args, attr, None) is not None and getattr(args, attr, None) is None:
            setattr(args, attr, getattr(post_args, attr))

    cfg = config_from_cli_args(args)
    env = _prepare_env_with_compose(cfg, config_path=args.config)

    cmd = args.subcommand
    if cmd == "podman-check":
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.podman_env", *extra], cfg=cfg, env=env
        )
    if cmd == "podman-clean":
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.podman_env", "--cleanup", *extra],
            cfg=cfg,
            env=env,
        )
    if cmd == "podman-smoke-compose":
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.podman_env", "--smoke-compose", *extra],
            cfg=cfg,
            env=env,
        )
    if cmd == "build-images":
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.build_images", *extra], cfg=cfg, env=env
        )
    if cmd == "build-synth":
        script = cfg.repo_root / "data_builders" / "build_synthetic_suite.py"
        return _run_child([sys.executable, "-B", str(script), *extra], cfg=cfg, env=env)
    if cmd == "build-checkpoints":
        script = cfg.repo_root / "data_builders" / "build_3c391_checkpoints.py"
        return _run_child([sys.executable, "-B", str(script), *extra], cfg=cfg, env=env)
    if cmd == "export-skills":
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.export_skills", *extra], cfg=cfg, env=env
        )
    if cmd == "build-tasks":
        bt_args = list(extra) if extra else ["--skills-mode", "both", "--include-orchestrated"]
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.build_tasks", *bt_args], cfg=cfg, env=env
        )
    if cmd == "render-jobs":
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.render_jobs", *extra], cfg=cfg, env=env
        )
    if cmd == "analyze":
        norm_extra = _resolve_relative_caller_paths(extra, caller_cwd)
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.analysis.cli", *norm_extra],
            cfg=cfg,
            env=env,
        )
    if cmd == "harbor-lint":
        if not cfg.tasks_dir.is_dir():
            print(
                f"error: compiled tasks directory does not exist: {cfg.tasks_dir}", file=sys.stderr
            )
            return 1
        task_dirs = sorted(str(p) for p in cfg.tasks_dir.iterdir() if p.is_dir())
        if not task_dirs:
            print(f"error: zero compiled task directories in {cfg.tasks_dir}", file=sys.stderr)
            return 1
        harbor_bin = shutil.which("harbor", path=env.get("PATH")) or "harbor"
        return _run_child([harbor_bin, "lint", *extra, *task_dirs], cfg=cfg, env=env)
    if cmd == "run-oracle":
        from radio_bench.render_jobs import render_preset_job

        rc_chk = _run_child(
            [sys.executable, "-B", "-m", "radio_bench.podman_env"], cfg=cfg, env=env
        )
        if rc_chk != 0:
            return rc_chk
        presets = (
            ["oracle_synth", "oracle_3c391"] if args.suite == "all" else [f"oracle_{args.suite}"]
        )
        harbor_bin = shutil.which("harbor", path=env.get("PATH")) or "harbor"
        for preset_name in presets:
            job_yaml = render_preset_job(preset_name, cfg=cfg, env=env)
            rc_job = _run_child(
                [harbor_bin, "run", "--config", str(job_yaml), *extra], cfg=cfg, env=env
            )
            if rc_job != 0:
                return rc_job
        return 0
    if cmd == "test-unit":
        pytest_args = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-c",
            str(cfg.repo_root / "pyproject.toml"),
            f"--rootdir={cfg.repo_root}",
            "-o",
            f"cache_dir={cfg.pytest_cache_dir}",
            f"--basetemp={cfg.pytest_dir}",
            *_normalize_pytest_args(extra, cfg),
        ]
        return _run_child(pytest_args, cfg=cfg, env=env)

    parser.error(f"Unhandled subcommand: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
