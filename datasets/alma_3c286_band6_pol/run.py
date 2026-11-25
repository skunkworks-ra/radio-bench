"""Single entrypoint for the alma_3c286_band6_pol reference dataset package.

Subcommands: fetch, build, tasks, oracle, eval, all. Every subcommand takes --dry-run.
`fetch` uses only the Python standard library so it can run on a bare OSG node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

SUITE_ID = "alma_3c286_band6_pol"
DATASET_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATASET_DIR.parents[1]
SUITES_DIR = DATASET_DIR / "suites"
SOURCES_JSON = DATASET_DIR / "sources.json"
BUILDER = DATASET_DIR / f"build_{SUITE_ID}.py"
MIN_FREE_GB = 150.0
STAGE_TASKS = ("g1_3c286pol_inspect", "g1_3c286pol_calibrate", "g1_3c286pol_image")
E2E_TASKS = ("g2_3c286pol_e2e",)


# ---------------------------------------------------------------------------
# Path resolution (stdlib fallback when radio_bench is not importable)
# ---------------------------------------------------------------------------


def _bench_cfg() -> Any | None:
    try:
        from radio_bench.config import load_bench_config
    except ImportError:
        return None
    return load_bench_config()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    cfg = None if (args.data_dir and args.mirror_dir and args.work_dir) else _bench_cfg()

    def pick(cli_val: str | None, cfg_attr: str, env_key: str) -> Path | None:
        if cli_val:
            return Path(cli_val).expanduser().resolve()
        if cfg is not None:
            return Path(getattr(cfg, cfg_attr)).resolve()
        if os.environ.get(env_key):
            return Path(os.environ[env_key]).expanduser().resolve()
        return None

    data_dir = pick(args.data_dir, "data_dir", "RADIO_BENCH_DATA_DIR")
    if data_dir is None:
        sys.exit("error: data_dir unknown (pass --data-dir or set RADIO_BENCH_DATA_DIR)")
    mirror = pick(args.mirror_dir, "casaguides_dir", "RADIO_BENCH_CASAGUIDES_DIR")
    work = pick(args.work_dir, "work_dir", "RADIO_BENCH_WORK_DIR")
    root = data_dir / "casaguides" / SUITE_ID
    out = {
        "data_dir": data_dir,
        "root": root,
        "downloads": root / "downloads",
        "extracted": root / "extracted",
    }
    if mirror is not None:
        out["mirror"] = mirror / SUITE_ID
    if work is not None:
        out["work_dir"] = work
        out["tasks_dir"] = work / f"tasks_{SUITE_ID}"
        out["jobs_dir"] = work / "jobs"
    for key, val in out.items():
        if str(val) == "/tmp" or str(val).startswith("/tmp/"):
            sys.exit(f"error: {key}={val} is under /tmp, which is not allowed")
    return out


def _need(paths: dict[str, Path], key: str) -> Path:
    if key not in paths:
        sys.exit(f"error: {key} unknown (pass --work-dir/--mirror-dir or install radio_bench)")
    return paths[key]


def _free_gb(path: Path) -> float:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False).returncode


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, total_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have >= total_bytes:
        part.replace(dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "radio-bench-fetch/1.0"})
    if have > 0:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as resp:
        if have > 0 and resp.status != 206:
            print(f"  server ignored Range; restarting {dest.name} from byte 0", flush=True)
            have = 0
        mode = "ab" if have > 0 else "wb"
        with part.open(mode) as fh:
            done = have
            next_report = done + 256 * 1024 * 1024
            while True:
                buf = resp.read(8 * 1024 * 1024)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if done >= next_report:
                    print(
                        f"  {dest.name}: {done / 1e9:.2f} / {total_bytes / 1e9:.2f} GB", flush=True
                    )
                    next_report = done + 256 * 1024 * 1024
    part.replace(dest)


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dest, filter="data")


def cmd_fetch(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    sources = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    free_gb = _free_gb(paths["root"])
    print(f"target root: {paths['root']} (free {free_gb:.1f} GB)")
    if free_gb < MIN_FREE_GB:
        print(f"error: need >= {MIN_FREE_GB} GB free on target filesystem", file=sys.stderr)
        return 1
    for arc in sources["archives"]:
        name = arc["name"]
        sha = arc["sha256"]
        ext_dir = (
            paths["extracted"] / arc["extract_subdir"]
            if arc["extract_subdir"]
            else paths["extracted"]
        )
        marker = paths["extracted"] / f".{name}.extracted"
        top = paths["extracted"] / arc["top_dir"]
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha and top.exists():
            print(f"[skip] {name}: already extracted at {top}")
            continue
        mirror_file = paths["mirror"] / name if "mirror" in paths else None
        src: Path | None = None
        if mirror_file is not None and mirror_file.is_file():
            print(f"[mirror] {name}: {mirror_file}")
            if args.dry_run or sha256_file(mirror_file) == sha:
                src = mirror_file
            else:
                print(f"  mirror sha256 mismatch for {mirror_file}; downloading instead")
        if src is None:
            dl = paths["downloads"] / name
            print(f"[download] {arc['url']} -> {dl} (resume enabled)")
            if not args.dry_run:
                if not (dl.is_file() and dl.stat().st_size == int(arc["bytes"])):
                    download_with_resume(arc["url"], dl, int(arc["bytes"]))
                got = sha256_file(dl)
                if got != sha:
                    print(f"error: sha256 mismatch for {dl}: {got} != {sha}", file=sys.stderr)
                    return 1
            src = dl
        print(f"[extract] {src} -> {ext_dir}")
        if not args.dry_run:
            safe_extract(src, ext_dir)
            marker.write_text(sha + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# build / tasks / oracle / eval
# ---------------------------------------------------------------------------


def _casa_env(paths: dict[str, Path]) -> dict[str, str]:
    env = dict(os.environ)
    work = paths.get("work_dir")
    if work is not None:
        env["TMPDIR"] = str(work / "tmp")
        env["MPLCONFIGDIR"] = str(work / "mplconfig")
        site = work / "casaconfig_local.py"
        if site.is_file():
            env.setdefault("CASASITECONFIG", str(site))
    return env


def cmd_build(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    cmd = [sys.executable, "-B", str(BUILDER), "--root", str(paths["root"])]
    if args.dry_run:
        cmd.append("--dry-run")
        return _run(cmd, dry_run=False, env=_casa_env(paths))
    env = _casa_env(paths)
    for key in ("TMPDIR", "MPLCONFIGDIR"):
        if key in env:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    return _run(cmd, dry_run=False, env=env)


def cmd_tasks(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    tasks_dir = _need(paths, "tasks_dir")
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "radio_bench.build_tasks",
        "--suites-dir",
        str(SUITES_DIR),
        "--output-dir",
        str(tasks_dir),
    ]
    if args.pack:
        cmd += ["--pack", args.pack]
    rc = _run(cmd, dry_run=args.dry_run)
    if rc != 0:
        return rc
    harbor = shutil.which("harbor") or "harbor"
    task_dirs = (
        sorted(str(p) for p in tasks_dir.iterdir() if p.is_dir())
        if tasks_dir.is_dir() and not args.dry_run
        else [f"{tasks_dir}/<compiled task dir>"]
    )
    for tdir in task_dirs:
        rc = _run([harbor, "lint", tdir], dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def _register_suite_with_render_jobs() -> Any:
    """Register this suite's task ids with radio_bench.render_jobs (in this process only)."""
    from radio_bench import render_jobs

    render_jobs.SUITE_STAGE_TASKS.setdefault(SUITE_ID, STAGE_TASKS)
    render_jobs.SUITE_E2E_TASKS.setdefault(SUITE_ID, E2E_TASKS)
    render_jobs.SUITE_DEFAULT_CONCURRENCY.setdefault(SUITE_ID, 1)
    return render_jobs


def _render_job(
    paths: dict[str, Path],
    *,
    job_name: str,
    oracle: bool,
    arms: list[str],
    scope: str,
    model_keys: list[str],
    thinking_levels: list[str] | None,
    n_attempts: int,
    pack: str | None,
    dry_run: bool,
) -> Path:
    import yaml

    from radio_bench.config import load_bench_config

    render_jobs = _register_suite_with_render_jobs()
    cfg = load_bench_config()
    jobs_dir = _need(paths, "jobs_dir")
    out_path = jobs_dir / f"{job_name}.yaml"
    job = render_jobs.compile_job_dict(
        job_name=job_name,
        suites=[SUITE_ID],
        arms=arms,
        task_scope=scope,
        oracle=oracle,
        model_keys=model_keys,
        thinking_levels=thinking_levels,
        pack_ids=[pack] if pack else None,
        n_attempts=n_attempts,
        suite_concurrency=1,
        max_retries=0 if oracle else None,
        tasks_dir=_need(paths, "tasks_dir"),
        jobs_out_dir=jobs_dir,
        cfg=cfg,
    )
    print(f"job yaml: {out_path}")
    print(yaml.safe_dump(job, sort_keys=False))
    if not dry_run:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    return out_path


def cmd_oracle(args: argparse.Namespace) -> int:
    from radio_bench.build_tasks import ARM_NONE

    paths = resolve_paths(args)
    job_yaml = _render_job(
        paths,
        job_name=f"oracle_{SUITE_ID}",
        oracle=True,
        arms=[ARM_NONE],
        scope="all",
        model_keys=[],
        thinking_levels=None,
        n_attempts=1,
        pack=args.pack,
        dry_run=args.dry_run,
    )
    return _run(
        [sys.executable, "-B", "-m", "radio_bench.cli", "run-job", str(job_yaml)],
        dry_run=args.dry_run,
    )


def cmd_eval(args: argparse.Namespace) -> int:
    from radio_bench.build_tasks import (
        ARM_FULL,
        ARM_NONE,
        ARM_ORCHESTRATED,
        ARM_SKILLS_CASA,
        ARM_SKILLS_RAW,
        ARM_TOOLS,
    )

    known = {ARM_NONE, ARM_SKILLS_RAW, ARM_SKILLS_CASA, ARM_TOOLS, ARM_FULL, ARM_ORCHESTRATED}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = sorted(set(arms) - known)
    if bad or not arms:
        print(f"error: unknown arms {bad}; known: {sorted(known)}", file=sys.stderr)
        return 2
    if args.n_attempts < 1:
        print("error: --n-attempts must be >= 1", file=sys.stderr)
        return 2
    paths = resolve_paths(args)
    levels = [x.strip() for x in args.thinking_levels.split(",")] if args.thinking_levels else None
    job_yaml = _render_job(
        paths,
        job_name=f"eval_{SUITE_ID}_{args.model}_{args.scope}",
        oracle=False,
        arms=arms,
        scope=args.scope,
        model_keys=[args.model],
        thinking_levels=levels,
        n_attempts=args.n_attempts,
        pack=args.pack,
        dry_run=args.dry_run,
    )
    return _run(
        [sys.executable, "-B", "-m", "radio_bench.cli", "run-job", str(job_yaml)],
        dry_run=args.dry_run,
    )


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_fetch, cmd_build, cmd_tasks, cmd_oracle):
        print(f"=== {step.__name__[4:]} ===", flush=True)
        rc = step(args)
        if rc != 0:
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{SUITE_ID} dataset package entrypoint.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="Print commands/paths only.")
    common.add_argument("--data-dir", default=None, help="Override bench data_dir.")
    common.add_argument("--mirror-dir", default=None, help="Override casaguides mirror dir.")
    common.add_argument("--work-dir", default=None, help="Override bench work_dir.")
    common.add_argument("--pack", default=None, help="radio-analyst pack id (default pack).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", parents=[common], help="Fetch, verify, and extract archives.")
    sub.add_parser("build", parents=[common], help="Build checkpoints and ground truth.")
    sub.add_parser("tasks", parents=[common], help="Compile Harbor tasks and harbor lint them.")
    sub.add_parser("oracle", parents=[common], help="Render and run the oracle job.")
    p_eval = sub.add_parser("eval", parents=[common], help="Render and run an LLM eval job.")
    p_eval.add_argument("--model", required=True, help="Model key from configs/models.toml.")
    p_eval.add_argument("--arms", required=True, help="Comma-separated arms, e.g. 00_none,11_full.")
    p_eval.add_argument("--n-attempts", type=int, required=True, help="Attempts per cell.")
    p_eval.add_argument("--scope", choices=("all", "stage", "e2e"), default="all")
    p_eval.add_argument("--thinking-levels", default=None, help="Comma-separated levels.")
    sub.add_parser("all", parents=[common], help="fetch -> build -> tasks -> oracle (never eval).")
    args = parser.parse_args(argv)
    handlers = {
        "fetch": cmd_fetch,
        "build": cmd_build,
        "tasks": cmd_tasks,
        "oracle": cmd_oracle,
        "eval": cmd_eval,
        "all": cmd_all,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
