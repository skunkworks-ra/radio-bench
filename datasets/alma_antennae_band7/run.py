"""Single entrypoint for the alma_antennae_band7 reference dataset package.

Subcommands: fetch, build, tasks, oracle, eval, all (fetch -> build -> tasks -> oracle).
`fetch` needs only the Python standard library so it can run on a bare OSG node.
Every subcommand accepts `--dry-run` (print commands and paths, change nothing).
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

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
SUITE_ID = "alma_antennae_band7"
SUITES_DIR = PKG_DIR / "suites"
SUITE_TOML = SUITES_DIR / SUITE_ID / "suite.toml"
SOURCES_JSON = PKG_DIR / "sources.json"
BUILDER = PKG_DIR / f"build_{SUITE_ID}.py"
MIN_FREE_BYTES = 150 * 1024**3
CHUNK = 8 * 1024 * 1024
KNOWN_ARMS = (
    "00_none",
    "01_skills_raw",
    "01_skills_casa",
    "10_tools",
    "11_full",
    "11_orchestrated",
)


# ---------------------------------------------------------------------------
# Paths (radio_bench.config is stdlib-only; explicit CLI overrides win)
# ---------------------------------------------------------------------------


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    src = REPO_ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from radio_bench.config import load_bench_config

        cfg = load_bench_config(
            config_path=args.config,
            repo_root=REPO_ROOT,
            work_dir=args.work_dir,
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
            casaguides_dir=args.casaguides_dir,
        )
        paths = {
            "data_dir": cfg.data_dir,
            "work_dir": cfg.work_dir,
            "runs_dir": cfg.runs_dir,
            "casaguides_dir": cfg.casaguides_dir,
        }
    except ImportError:
        if not args.data_dir:
            sys.exit("radio_bench.config not importable: pass --data-dir (and --work-dir)")
        data_dir = Path(args.data_dir)
        paths = {
            "data_dir": data_dir,
            "work_dir": Path(args.work_dir) if args.work_dir else data_dir / "work",
            "runs_dir": Path(args.runs_dir) if args.runs_dir else data_dir / "runs",
            "casaguides_dir": Path(args.casaguides_dir)
            if args.casaguides_dir
            else data_dir / "casaguides_mirror",
        }
    for key, val in paths.items():
        p = Path(val).expanduser().resolve()
        if p == Path("/tmp") or str(p).startswith("/tmp/"):
            sys.exit(f"{key}={p} is under /tmp; refusing")
        paths[key] = p
    paths["suite_root"] = paths["data_dir"] / "casaguides" / SUITE_ID
    paths["downloads"] = paths["suite_root"] / "downloads"
    paths["extracted"] = paths["suite_root"] / "extracted"
    paths["tasks_out"] = paths["work_dir"] / f"tasks_{SUITE_ID}"
    paths["jobs_out"] = paths["work_dir"] / "jobs"
    return paths


def load_sources() -> dict[str, Any]:
    return json.loads(SOURCES_JSON.read_text(encoding="utf-8"))


def load_task_ids() -> list[str]:
    import tomllib

    doc = tomllib.loads(SUITE_TOML.read_text(encoding="utf-8"))
    return sorted(doc["tasks"].keys())


def run_cmd(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    proc = subprocess.run([str(c) for c in cmd], env=env, check=False)
    if proc.returncode != 0:
        sys.exit(f"command failed with exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# fetch
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
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def download_with_resume(url: str, dest: Path, total_bytes: int) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > total_bytes:
        part.unlink()
        have = 0
    req = urllib.request.Request(url, headers={"User-Agent": "radio-bench-fetch"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as resp:
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0
        with part.open(mode) as out:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                out.write(buf)
                have += len(buf)
    if have != total_bytes:
        sys.exit(f"{dest.name}: downloaded {have} bytes, expected {total_bytes}; rerun to resume")
    part.replace(dest)


def safe_extract(archive: Path, out_dir: Path) -> None:
    out_root = out_dir.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for m in tf.getmembers():
            target = (out_root / m.name).resolve()
            if target != out_root and out_root not in target.parents:
                sys.exit(f"{archive.name}: unsafe member path {m.name!r}")
            if m.issym() or m.islnk():
                link = (target.parent / m.linkname).resolve()
                if out_root not in link.parents:
                    sys.exit(f"{archive.name}: link escapes extract dir: {m.name!r}")
        if sys.version_info >= (3, 12):  # noqa: UP036 -- OSG nodes may run older Python
            tf.extractall(out_root, filter="data")
        else:
            tf.extractall(out_root)


def cmd_fetch(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    sources = load_sources()
    target_fs = nearest_existing(paths["suite_root"])
    free = shutil.disk_usage(target_fs).free
    print(f"target filesystem {target_fs}: {free / 1024**3:.1f} GB free (minimum 150 GB)")
    if free < MIN_FREE_BYTES:
        sys.exit("refusing: less than 150 GB free on the target filesystem")
    mirror_dir = paths["casaguides_dir"] / SUITE_ID
    for arc in sources["archives"]:
        name, url, sha, nbytes = arc["name"], arc["url"], arc["sha256"], int(arc["bytes"])
        mirror = mirror_dir / name
        local = paths["downloads"] / name
        stamp = paths["extracted"] / f".{name}.sha256"
        if args.dry_run:
            print(f"[{name}] mirror candidate: {mirror} (used if sha256 == {sha})")
            print(f"[{name}] else download {url} -> {local} (resume via {local}.part)")
            print(f"[{name}] extract -> {paths['extracted']} (stamp {stamp})")
            continue
        if stamp.is_file() and stamp.read_text().strip() == sha:
            print(f"[{name}] already extracted (stamp matches); skipping")
            continue
        src: Path | None = None
        if mirror.is_file() and mirror.stat().st_size == nbytes:
            print(f"[{name}] verifying mirror {mirror}")
            if sha256_file(mirror) == sha:
                src = mirror
            else:
                print(f"[{name}] mirror sha256 mismatch; falling back to download")
        if src is None:
            paths["downloads"].mkdir(parents=True, exist_ok=True)
            if not (local.is_file() and local.stat().st_size == nbytes):
                print(f"[{name}] downloading {url}")
                download_with_resume(url, local, nbytes)
            got = sha256_file(local)
            if got != sha:
                sys.exit(f"[{name}] sha256 mismatch: got {got}, expected {sha}")
            src = local
        paths["extracted"].mkdir(parents=True, exist_ok=True)
        print(f"[{name}] extracting {src} -> {paths['extracted']}")
        safe_extract(src, paths["extracted"])
        stamp.write_text(sha + "\n")
    print("fetch done")


# ---------------------------------------------------------------------------
# build / tasks / oracle / eval
# ---------------------------------------------------------------------------


def casa_env(paths: dict[str, Path]) -> dict[str, str]:
    env = dict(os.environ)
    work = paths["work_dir"]
    defaults = {
        "CASASITECONFIG": work / "casaconfig_local.py",
        "TMPDIR": work / "tmp",
        "MPLCONFIGDIR": work / "mplconfig",
    }
    for key, val in defaults.items():
        if not env.get(key, "").strip() and (val.exists() or key != "CASASITECONFIG"):
            env[key] = str(val)
    return env


def cmd_build(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    cmd = [
        args.python or sys.executable,
        str(BUILDER),
        "--suite-root",
        str(paths["suite_root"]),
    ]
    env = casa_env(paths)
    for key in ("CASASITECONFIG", "TMPDIR", "MPLCONFIGDIR"):
        print(f"  {key}={env.get(key, '')}")
    if not args.dry_run:
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    run_cmd(cmd, dry_run=args.dry_run, env=env)


def compiled_task_dirs(paths: dict[str, Path]) -> list[Path]:
    ids = load_task_ids()
    out = paths["tasks_out"]
    if not out.is_dir():
        return []
    return sorted(d for d in out.iterdir() if d.is_dir() and d.name.split("__", 1)[0] in ids)


def cmd_tasks(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "radio_bench.build_tasks",
        "--data-dir",
        str(paths["data_dir"]),
        "--work-dir",
        str(paths["work_dir"]),
        "--suites-dir",
        str(SUITES_DIR),
        "--output-dir",
        str(paths["tasks_out"]),
    ]
    if args.pack:
        cmd += ["--pack", args.pack]
    run_cmd(cmd, dry_run=args.dry_run)
    if args.dry_run:
        print(f"+ harbor lint <each dir in {paths['tasks_out']} named <task_id>__<arm>>/")
        return
    dirs = compiled_task_dirs(paths)
    if not dirs:
        sys.exit(f"no compiled task dirs found in {paths['tasks_out']}")
    for d in dirs:
        run_cmd(["harbor", "lint", f"{d}/"], dry_run=False)


def _register_suite_for_render() -> Any:
    """Register this suite's task ids with radio_bench.render_jobs in this process only."""
    from radio_bench import render_jobs

    ids = load_task_ids()
    render_jobs.SUITE_STAGE_TASKS[SUITE_ID] = tuple(t for t in ids if not t.startswith("g2_"))
    render_jobs.SUITE_E2E_TASKS[SUITE_ID] = tuple(t for t in ids if t.startswith("g2_"))
    render_jobs.SUITE_DEFAULT_CONCURRENCY[SUITE_ID] = 1
    return render_jobs


def render_and_run(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    job_name: str,
    oracle: bool,
    arms: list[str],
    model_keys: list[str],
    n_attempts: int,
) -> None:
    out_yaml = paths["jobs_out"] / f"{job_name}.yaml"
    print(f"job yaml: {out_yaml}")
    print(f"tasks dir: {paths['tasks_out']}  runs dir: {paths['runs_dir']}")
    print(f"agents: {'oracle' if oracle else model_keys}  arms: {arms}  n_attempts: {n_attempts}")
    run_cmd_list = [sys.executable, "-m", "radio_bench.cli", "run-job", str(out_yaml)]
    if args.dry_run:
        print("+ (render via radio_bench.render_jobs.compile_job_dict)")
        print("+ " + " ".join(run_cmd_list))
        return
    import yaml

    render_jobs = _register_suite_for_render()
    job = render_jobs.compile_job_dict(
        job_name=job_name,
        suites=[SUITE_ID],
        arms=arms,
        task_scope="all",
        oracle=oracle,
        model_keys=model_keys,
        pack_ids=[args.pack] if args.pack else None,
        n_attempts=n_attempts,
        suite_concurrency=1,
        tasks_dir=paths["tasks_out"],
        runs_dir=paths["runs_dir"],
        jobs_out_dir=paths["jobs_out"],
    )
    missing = [t["path"] for t in job["tasks"] if not Path(t["path"]).is_dir()]
    if missing:
        sys.exit("compiled task dirs missing (run `tasks` first):\n" + "\n".join(missing))
    paths["jobs_out"].mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    run_cmd(run_cmd_list, dry_run=False)


def cmd_oracle(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    render_and_run(
        args,
        paths,
        job_name=f"oracle_{SUITE_ID}",
        oracle=True,
        arms=["00_none"],
        model_keys=[],
        n_attempts=1,
    )


def cmd_eval(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in KNOWN_ARMS]
    if not arms or bad:
        sys.exit(f"--arms must be a csv of {list(KNOWN_ARMS)}; bad: {bad}")
    if args.n_attempts < 1:
        sys.exit("--n-attempts must be >= 1")
    render_and_run(
        args,
        paths,
        job_name=f"{SUITE_ID}_{args.model}",
        oracle=False,
        arms=arms,
        model_keys=[args.model],
        n_attempts=int(args.n_attempts),
    )


def cmd_all(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    cmd_fetch(args, paths)
    cmd_build(args, paths)
    cmd_tasks(args, paths)
    cmd_oracle(args, paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dry-run", action="store_true", help="Print commands/paths only.")
        p.add_argument("--config", default=None, help="Explicit bench.toml path.")
        p.add_argument("--data-dir", default=None)
        p.add_argument("--work-dir", default=None)
        p.add_argument("--runs-dir", default=None)
        p.add_argument("--casaguides-dir", default=None, help="Local mirror root.")

    p_fetch = sub.add_parser("fetch", help="Fetch (mirror first), verify sha256, extract.")
    p_build = sub.add_parser("build", help="Run the CASA checkpoint + ground-truth builder.")
    p_build.add_argument("--python", default=None, help="Python with casatools (default: this).")
    p_tasks = sub.add_parser("tasks", help="Compile Harbor tasks and harbor lint each one.")
    p_oracle = sub.add_parser("oracle", help="Render and run the oracle job.")
    p_eval = sub.add_parser("eval", help="Render and run a model evaluation job.")
    p_eval.add_argument("--model", required=True, help="Key in configs/models.toml.")
    p_eval.add_argument("--arms", required=True, help="Comma-separated arms.")
    p_eval.add_argument("--n-attempts", required=True, type=int)
    p_all = sub.add_parser("all", help="fetch -> build -> tasks -> oracle (never eval).")
    p_all.add_argument("--python", default=None, help="Python with casatools (default: this).")
    for p in (p_fetch, p_build, p_tasks, p_oracle, p_eval, p_all):
        common(p)
    for p in (p_tasks, p_oracle, p_eval, p_all):
        p.add_argument("--pack", default=None, help="Pack id (default: bench default_pack).")

    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    if args.dry_run:
        print(f"[dry-run] {args.cmd}: " + json.dumps({k: str(v) for k, v in paths.items()}))
    handlers = {
        "fetch": cmd_fetch,
        "build": cmd_build,
        "tasks": cmd_tasks,
        "oracle": cmd_oracle,
        "eval": cmd_eval,
        "all": cmd_all,
    }
    handlers[args.cmd](args, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
