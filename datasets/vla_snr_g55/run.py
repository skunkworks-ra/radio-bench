"""Single entrypoint for the vla_snr_g55 reference dataset package.

Subcommands: fetch, build, tasks, oracle, eval, all. Every subcommand accepts --dry-run.
`fetch` uses only the standard library so it can run on a bare OSG node.
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

SUITE_ID = "vla_snr_g55"
HERE = Path(__file__).resolve().parent
SOURCES_JSON = HERE / "sources.json"
SUITES_DIR = HERE / "suites"
BUILDER = HERE / f"build_{SUITE_ID}.py"
MIN_FREE_BYTES = 150 * 1024**3
STAGE_TASKS = ("g1_g55_inspect", "g1_g55_precal", "g1_g55_calibrate", "g1_g55_image")
E2E_TASKS = ("g2_g55_e2e",)
CHUNK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Path resolution (stdlib fallback when radio_bench is not importable)
# ---------------------------------------------------------------------------


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Return data_dir, work_dir, runs_dir, casaguides_dir from CLI > env > bench config."""
    cfg_vals: dict[str, Path] = {}
    try:
        from radio_bench.config import load_bench_config

        cfg = load_bench_config()
        cfg_vals = {
            "data_dir": cfg.data_dir,
            "work_dir": cfg.work_dir,
            "runs_dir": cfg.runs_dir,
            "casaguides_dir": cfg.casaguides_dir,
        }
    except Exception:
        cfg_vals = {}
    env_keys = {
        "data_dir": "RADIO_BENCH_DATA_DIR",
        "work_dir": "RADIO_BENCH_WORK_DIR",
        "runs_dir": "RADIO_BENCH_RUNS_DIR",
        "casaguides_dir": "RADIO_BENCH_CASAGUIDES_DIR",
    }
    out: dict[str, Path] = {}
    for key, env_key in env_keys.items():
        cli_val = getattr(args, key, None)
        if cli_val:
            out[key] = Path(cli_val).expanduser().resolve()
        elif os.environ.get(env_key, "").strip():
            out[key] = Path(os.environ[env_key]).expanduser().resolve()
        elif key in cfg_vals:
            out[key] = Path(cfg_vals[key])
    if "data_dir" not in out:
        raise SystemExit("error: data_dir unknown (pass --data-dir or set RADIO_BENCH_DATA_DIR)")
    out.setdefault("casaguides_dir", out["data_dir"] / "casaguides")
    for key in ("data_dir", "work_dir", "runs_dir"):
        if key in out and (str(out[key]) == "/tmp" or str(out[key]).startswith("/tmp/")):
            raise SystemExit(f"error: {key}={out[key]} is under /tmp (forbidden)")
    return out


def suite_root(paths: dict[str, Path]) -> Path:
    return paths["data_dir"] / "casaguides" / SUITE_ID


def tasks_dir(paths: dict[str, Path]) -> Path:
    if "work_dir" not in paths:
        raise SystemExit("error: work_dir unknown (pass --work-dir or set RADIO_BENCH_WORK_DIR)")
    return paths["work_dir"] / f"tasks_{SUITE_ID}"


def jobs_dir(paths: dict[str, Path]) -> Path:
    return paths["work_dir"] / "jobs"


def show(cmd: list[str], *, env_note: str = "") -> None:
    prefix = f"{env_note} " if env_note else ""
    print("+ " + prefix + " ".join(cmd), flush=True)


def run_cmd(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> int:
    show(cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False, env=env).returncode


def casa_env(paths: dict[str, Path]) -> dict[str, str]:
    """Scratch env for CASA children: never /tmp, always under work_dir when known."""
    env = dict(os.environ)
    base = paths.get("work_dir", suite_root(paths))
    env.setdefault("TMPDIR", str(base / "tmp"))
    env.setdefault("MPLCONFIGDIR", str(base / "mplconfig"))
    site_cfg = base / "casaconfig_local.py"
    if "CASASITECONFIG" not in env and site_cfg.is_file():
        env["CASASITECONFIG"] = str(site_cfg)
    return env


# ---------------------------------------------------------------------------
# fetch (stdlib only)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def nearest_existing(path: Path) -> Path:
    cur = path
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def download_resume(url: str, dest: Path, total_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > total_bytes:
        part.unlink()
        have = 0
    if have < total_bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "radio-bench-fetch/1"})
        if have > 0:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = getattr(resp, "status", 200)
            mode = "ab" if (have > 0 and status == 206) else "wb"
            if mode == "wb":
                have = 0
            with part.open(mode) as fh:
                while True:
                    buf = resp.read(CHUNK)
                    if not buf:
                        break
                    fh.write(buf)
                    have += len(buf)
    part.replace(dest)


def extract_archive(archive: Path, out_dir: Path, member_root: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / member_root
    if target.exists():
        shutil.rmtree(target)
    with tarfile.open(archive, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(out_dir, filter="data")
        else:
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise SystemExit(f"error: unsafe tar member {m.name!r} in {archive}")
            tf.extractall(out_dir)
    # The published tarballs ship stale table.lock files; drop them so CASA opens cleanly.
    for lock in target.rglob("table.lock"):
        lock.unlink()


def cmd_fetch(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    sources = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    root = suite_root(paths)
    downloads = root / "downloads"
    extracted = root / "extracted"
    mirror = paths["casaguides_dir"] / SUITE_ID

    free = shutil.disk_usage(nearest_existing(root)).free
    print(f"target={root} free={free / 1024**3:.1f} GiB (min 150 GiB)")
    if free < MIN_FREE_BYTES:
        print("error: refusing to fetch, target filesystem has < 150 GiB free", file=sys.stderr)
        return 1

    for arc in sources["archives"]:
        name, sha, nbytes = arc["name"], arc["sha256"], int(arc["bytes"])
        member = arc["contains"]
        marker = extracted / f".{name}.sha256"
        mirror_path = mirror / name
        dl_path = downloads / name
        if marker.is_file() and marker.read_text().strip() == sha and (extracted / member).is_dir():
            print(f"[skip] {name}: already extracted to {extracted / member}")
            continue
        src: Path | None = None
        if mirror_path.is_file() and mirror_path.stat().st_size == nbytes:
            print(f"[mirror] {name}: checking sha256 of {mirror_path}")
            if args.dry_run or sha256_file(mirror_path) == sha:
                src = mirror_path
            else:
                print(f"[mirror] {name}: sha256 mismatch, falling back to download")
        if src is None:
            if dl_path.is_file() and (args.dry_run or sha256_file(dl_path) == sha):
                src = dl_path
            else:
                print(f"[download] {arc['url']} -> {dl_path} (resume)")
                if not args.dry_run:
                    download_resume(arc["url"], dl_path, nbytes)
                    got = sha256_file(dl_path)
                    if got != sha:
                        print(f"error: sha256 mismatch for {dl_path}: {got}", file=sys.stderr)
                        return 1
                src = dl_path
        print(f"[extract] {src} -> {extracted / member}")
        if not args.dry_run:
            extract_archive(src, extracted, member)
            marker.write_text(sha + "\n")
    return 0


# ---------------------------------------------------------------------------
# build / tasks / oracle / eval
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    root = suite_root(paths)
    cmd = [args.python or sys.executable, "-B", str(BUILDER), "--suite-root", str(root)]
    env = casa_env(paths)
    for key in ("TMPDIR", "MPLCONFIGDIR"):
        if not args.dry_run:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    print(f"# env TMPDIR={env['TMPDIR']} MPLCONFIGDIR={env['MPLCONFIGDIR']}")
    return run_cmd(cmd, dry_run=args.dry_run, env=env)


def config_args(paths: dict[str, Path]) -> list[str]:
    out = ["--data-dir", str(paths["data_dir"])]
    if "work_dir" in paths:
        out += ["--work-dir", str(paths["work_dir"])]
    if "runs_dir" in paths:
        out += ["--runs-dir", str(paths["runs_dir"])]
    return out


def cmd_tasks(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    out_dir = tasks_dir(paths)
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "radio_bench.build_tasks",
        *config_args(paths),
        "--suites-dir",
        str(SUITES_DIR),
        "--output-dir",
        str(out_dir),
    ]
    if args.pack:
        cmd += ["--pack", args.pack]
    rc = run_cmd(cmd, dry_run=args.dry_run)
    if rc != 0:
        return rc
    harbor_bin = shutil.which("harbor") or "harbor"
    prefixes = (*STAGE_TASKS, *E2E_TASKS)
    if args.dry_run:
        show([harbor_bin, "lint", f"{out_dir}/<task>__<arm>/ (one call per compiled task)"])
        return 0
    task_dirs = sorted(
        p
        for p in out_dir.iterdir()
        if p.is_dir() and p.name.startswith(tuple(f"{t}__" for t in prefixes))
    )
    if not task_dirs:
        print(f"error: zero compiled task dirs in {out_dir}", file=sys.stderr)
        return 1
    from radio_bench.podman_env import build_podman_env  # harbor lint needs the Podman/Compose env

    lint_env = build_podman_env()
    failed = 0
    for tdir in task_dirs:
        if run_cmd([harbor_bin, "lint", f"{tdir}/"], dry_run=False, env=lint_env) != 0:
            failed += 1
    print(f"harbor lint: {len(task_dirs) - failed} passed, {failed} failed")
    return 1 if failed else 0


def _register_suite_in_render_jobs() -> Any:
    """Register this suite's task ids with radio_bench.render_jobs (in-process only)."""
    from radio_bench import render_jobs

    render_jobs.SUITE_STAGE_TASKS.setdefault(SUITE_ID, STAGE_TASKS)
    render_jobs.SUITE_E2E_TASKS.setdefault(SUITE_ID, E2E_TASKS)
    render_jobs.SUITE_DEFAULT_CONCURRENCY.setdefault(SUITE_ID, 1)
    return render_jobs


def render_job(
    paths: dict[str, Path],
    *,
    job_name: str,
    oracle: bool,
    arms: list[str],
    model_keys: list[str],
    n_attempts: int,
    pack: str | None,
    dry_run: bool,
) -> Path:
    out_yaml = jobs_dir(paths) / f"{job_name}.yaml"
    print(
        f"# render job {job_name}: oracle={oracle} models={model_keys} arms={arms} "
        f"n_attempts={n_attempts} tasks_dir={tasks_dir(paths)} -> {out_yaml}"
    )
    if dry_run:
        return out_yaml
    import yaml

    from radio_bench.config import load_bench_config

    render_jobs = _register_suite_in_render_jobs()
    cfg = load_bench_config(
        data_dir=paths["data_dir"], work_dir=paths.get("work_dir"), runs_dir=paths.get("runs_dir")
    )
    job = render_jobs.compile_job_dict(
        job_name=job_name,
        suites=[SUITE_ID],
        arms=arms,
        task_scope="all",
        oracle=oracle,
        model_keys=model_keys,
        thinking_levels=None,
        pack_ids=[pack] if pack else None,
        n_attempts=n_attempts,
        suite_concurrency=1,
        tasks_dir=tasks_dir(paths),
        cfg=cfg,
    )
    missing = [t["path"] for t in job["tasks"] if not Path(t["path"]).is_dir()]
    if missing:
        raise SystemExit(
            "error: compiled task dirs missing (run `tasks` first):\n" + "\n".join(missing)
        )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    print(f"wrote {out_yaml}")
    return out_yaml


def run_job(paths: dict[str, Path], job_yaml: Path, *, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "radio_bench.cli",
        *config_args(paths),
        "run-job",
        str(job_yaml),
    ]
    return run_cmd(cmd, dry_run=dry_run)


def cmd_oracle(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    job_yaml = render_job(
        paths,
        job_name=f"oracle_{SUITE_ID}",
        oracle=True,
        arms=["00_none"],
        model_keys=[],
        n_attempts=1,
        pack=args.pack,
        dry_run=args.dry_run,
    )
    return run_job(paths, job_yaml, dry_run=args.dry_run)


def cmd_eval(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("error: --arms is empty")
    if args.n_attempts < 1:
        raise SystemExit("error: --n-attempts must be >= 1")
    job_yaml = render_job(
        paths,
        job_name=f"{SUITE_ID}_{args.model}_{'-'.join(arms)}_n{args.n_attempts}",
        oracle=False,
        arms=arms,
        model_keys=[args.model],
        n_attempts=args.n_attempts,
        pack=args.pack,
        dry_run=args.dry_run,
    )
    return run_job(paths, job_yaml, dry_run=args.dry_run)


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_fetch, cmd_build, cmd_tasks, cmd_oracle):
        rc = step(args)
        if rc != 0:
            print(f"error: {step.__name__} failed with rc={rc}", file=sys.stderr)
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{SUITE_ID} dataset package entrypoint")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="Print commands and paths only.")
    common.add_argument("--data-dir", default=None, help="Override data_dir.")
    common.add_argument("--work-dir", default=None, help="Override work_dir.")
    common.add_argument("--runs-dir", default=None, help="Override runs_dir.")
    common.add_argument("--casaguides-dir", default=None, help="Override local mirror root.")
    common.add_argument("--pack", default=None, help="radio-analyst pack id (tasks/oracle/eval).")
    common.add_argument("--python", default=None, help="Interpreter for the CASA builder.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", parents=[common], help="Fetch + verify + extract archives.")
    sub.add_parser("build", parents=[common], help="Build checkpoints and ground truth (CASA).")
    sub.add_parser("tasks", parents=[common], help="Compile Harbor tasks and harbor lint them.")
    sub.add_parser("oracle", parents=[common], help="Render and run the oracle job.")
    p_eval = sub.add_parser("eval", parents=[common], help="Render and run a model eval job.")
    p_eval.add_argument("--model", required=True, help="Model key from configs/models.toml.")
    p_eval.add_argument("--arms", required=True, help="Comma-separated arms, e.g. 00_none,11_full.")
    p_eval.add_argument("--n-attempts", type=int, required=True, help="Attempts per cell.")
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
