"""Unified portable CLI dispatcher for radio-bench tasks."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

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


@contextlib.contextmanager
def _podman_exec_safe_socket_env(
    cfg: BenchConfig, base_env: dict[str, str]
) -> Iterator[dict[str, str]]:
    """Provide a rootless Podman API socket safe against systemd `OOMScoreAdjust > 0` `crun` exec errors."""
    env = dict(base_env)
    env.setdefault("PODMAN_COMPOSE_WARNING_LOGS", "false")
    needs_local_svc = False
    try:
        self_oom = int(Path("/proc/self/oom_score_adj").read_text(encoding="utf-8").strip())
        proc_oom = subprocess.run(
            ["systemctl", "--user", "show", "podman.service", "-p", "OOMScoreAdjust", "--value"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if proc_oom.returncode == 0 and proc_oom.stdout.strip():
            svc_oom = int(proc_oom.stdout.strip())
            if svc_oom > self_oom and "/podman/podman.sock" in env.get("DOCKER_HOST", ""):
                needs_local_svc = True
    except Exception:
        needs_local_svc = False

    if not needs_local_svc:
        yield env
        return

    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    sock = cfg.tmp_dir / f"podman_api_{os.getpid()}.sock"
    if len(str(sock)) > 100:
        run_user = Path(f"/run/user/{os.getuid()}")
        if run_user.is_dir() and os.access(run_user, os.W_OK):
            sock = run_user / f"rb_podman_{os.getpid()}.sock"
    sock.unlink(missing_ok=True)

    svc = subprocess.Popen(
        ["podman", "system", "service", "--time=0", f"unix://{sock}"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    prev_sigterm: Any = None
    installed_sigterm = False

    def _handle_sigterm(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    try:
        with contextlib.suppress(Exception):
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _handle_sigterm)
            installed_sigterm = True

        for _ in range(40):
            if sock.exists():
                break
            if svc.poll() is not None:
                break
            time.sleep(0.1)
        if not sock.exists():
            rc_poll = svc.poll()
            raise RuntimeError(
                f"podman system service failed to create socket {sock} (poll={rc_poll})"
            )
        env["DOCKER_HOST"] = f"unix://{sock}"
        env["CONTAINER_HOST"] = f"unix://{sock}"
        yield env
    finally:
        if installed_sigterm:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGTERM, prev_sigterm)
        if svc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(svc.pid, signal.SIGTERM)
            svc.terminate()
        try:
            svc.wait(timeout=10.0)
        except Exception:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(svc.pid, signal.SIGKILL)
            svc.kill()
            with contextlib.suppress(Exception):
                svc.wait(timeout=3.0)
        sock.unlink(missing_ok=True)


def _run_child(cmd: list[str], *, cfg: BenchConfig, env: dict[str, str]) -> int:
    proc = subprocess.run(cmd, cwd=str(cfg.work_dir), env=env, check=False)
    return int(proc.returncode)


def _resolve_relative_caller_paths(
    extra_args: Sequence[str],
    caller_cwd: Path,
) -> list[str]:
    """Resolve relative directory/file arguments and `--output-dir` / `--diffs-dir` against `caller_cwd`."""
    path_flags = {"--output-dir", "--diffs-dir"}
    resolved: list[str] = []
    expect_path = False
    for arg in extra_args:
        if expect_path:
            expect_path = False
            if not arg.startswith("-"):
                cand_opt = Path(os.path.expanduser(arg))
                resolved.append(
                    str(cand_opt if cand_opt.is_absolute() else (caller_cwd / cand_opt).resolve())
                )
            else:
                resolved.append(arg)
            continue
        if arg in path_flags:
            resolved.append(arg)
            expect_path = True
            continue
        matched_eq = False
        for pf in path_flags:
            if arg.startswith(f"{pf}="):
                val = arg[len(pf) + 1 :]
                cand_eq = Path(os.path.expanduser(val))
                abs_s = str(cand_eq if cand_eq.is_absolute() else (caller_cwd / cand_eq).resolve())
                resolved.append(f"{pf}={abs_s}")
                matched_eq = True
                break
        if matched_eq:
            continue
        if arg.startswith("-"):
            resolved.append(arg)
            continue
        cand = Path(arg)
        if not cand.is_absolute() and (caller_cwd / cand).exists():
            resolved.append(str((caller_cwd / cand).resolve()))
        else:
            resolved.append(arg)
    return resolved


def _resolve_diff_caller_paths(
    extra_args: Sequence[str],
    caller_cwd: Path,
) -> list[str]:
    """Resolve relative `--agent-dir`, `--ref-dir`, `--job-dir`, and `--output` paths against `caller_cwd`."""
    path_flags = {"--agent-dir", "--ref-dir", "--job-dir", "--output"}
    resolved: list[str] = []
    expect_path = False
    for arg in extra_args:
        if expect_path:
            expect_path = False
            if not arg.startswith("-"):
                cand = Path(os.path.expanduser(arg))
                resolved.append(str(cand if cand.is_absolute() else (caller_cwd / cand).resolve()))
            else:
                resolved.append(arg)
            continue
        if arg in path_flags:
            resolved.append(arg)
            expect_path = True
            continue
        matched_eq = False
        for pf in path_flags:
            if arg.startswith(f"{pf}="):
                val = arg[len(pf) + 1 :]
                cand = Path(os.path.expanduser(val))
                abs_s = str(cand if cand.is_absolute() else (caller_cwd / cand).resolve())
                resolved.append(f"{pf}={abs_s}")
                matched_eq = True
                break
        if not matched_eq:
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
    subparsers.add_parser(
        "diff-run",
        help="Compare an agent workspace, trial, or Harbor job against a reference checkpoint.",
    )

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

    job_parser = subparsers.add_parser(
        "run-job",
        help="Execute a Harbor job YAML config with rootless Podman socket preflight.",
    )
    job_parser.add_argument(
        "--job-config",
        type=Path,
        default=None,
        help="Path to rendered Harbor job YAML config.",
    )
    job_parser.add_argument(
        "job_yaml",
        nargs="?",
        type=Path,
        default=None,
        help="Positional path to rendered Harbor job YAML config.",
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
    if cmd == "diff-run":
        norm_extra = _resolve_diff_caller_paths(extra, caller_cwd)
        return _run_child(
            [sys.executable, "-B", "-m", "radio_bench.analysis.diff", *norm_extra],
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
        with _podman_exec_safe_socket_env(cfg, env) as oracle_env:
            for preset_name in presets:
                job_yaml = render_preset_job(preset_name, cfg=cfg, env=oracle_env)
                rc_job = _run_child(
                    [harbor_bin, "run", "--config", str(job_yaml), *extra], cfg=cfg, env=oracle_env
                )
                if rc_job != 0:
                    return rc_job
        return 0
    if cmd == "run-job":
        raw_job = args.job_config or args.job_yaml
        if raw_job is None:
            print(
                "error: run-job requires a job YAML path (--job-config or positional)",
                file=sys.stderr,
            )
            return 2
        resolved_job = (
            raw_job.resolve() if raw_job.is_absolute() else (caller_cwd / raw_job).resolve()
        )
        if not resolved_job.is_file():
            print(f"error: job YAML not found: {resolved_job}", file=sys.stderr)
            return 1
        rc_chk = _run_child(
            [sys.executable, "-B", "-m", "radio_bench.podman_env"], cfg=cfg, env=env
        )
        if rc_chk != 0:
            return rc_chk
        harbor_bin = shutil.which("harbor", path=env.get("PATH")) or "harbor"
        with _podman_exec_safe_socket_env(cfg, env) as job_env:
            return _run_child(
                [harbor_bin, "run", "--config", str(resolved_job), *extra],
                cfg=cfg,
                env=job_env,
            )
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
