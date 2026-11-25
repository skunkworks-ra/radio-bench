"""One entrypoint for the alma_twhya_band7 reference dataset: fetch, build, tasks, oracle, eval.

`fetch` is stdlib-only so it can run on an OSG node. `build` runs the CASA builder. `tasks`,
`oracle`, and `eval` shell out to the existing radio_bench tools (`build_tasks`, `render_jobs`
compile_job_dict, `cli run-job`) and `harbor lint`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

SUITE_ID = "alma_twhya_band7"
TASKS_PREFIX = ("g1_twhya_", "g2_twhya_")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITES_DIR = HERE / "suites"
SUITE_TOML = SUITES_DIR / SUITE_ID / "suite.toml"
SOURCES_JSON = HERE / "sources.json"
BUILDER = HERE / f"build_{SUITE_ID}.py"
MIN_FREE_BYTES = 150 * 1024**3
STAGE_ARMS = ("00_none", "01_skills_raw", "01_skills_casa", "10_tools", "11_full")
CHUNK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    """CLI flags > radio_bench.config (bench.toml / RADIO_BENCH_* env) > RADIO_BENCH_* env alone."""
    vals: dict[str, Path | None] = {
        "data_dir": None,
        "work_dir": None,
        "runs_dir": None,
        "casaguides_dir": None,
    }
    try:
        from radio_bench.config import load_bench_config

        cfg = load_bench_config()
        for key in vals:
            vals[key] = Path(getattr(cfg, key))
    except ImportError:
        env_keys = {
            "data_dir": "RADIO_BENCH_DATA_DIR",
            "work_dir": "RADIO_BENCH_WORK_DIR",
            "runs_dir": "RADIO_BENCH_RUNS_DIR",
            "casaguides_dir": "RADIO_BENCH_CASAGUIDES_DIR",
        }
        for key, env in env_keys.items():
            if os.environ.get(env):
                vals[key] = Path(os.environ[env]).expanduser()
    for key in vals:
        cli_val = getattr(args, key, None)
        if cli_val:
            vals[key] = Path(cli_val).expanduser()
    for key, val in vals.items():
        if val is not None and (str(val) == "/tmp" or str(val).startswith("/tmp/")):
            raise SystemExit(f"{key}={val} is under /tmp; use a durable work area")
    return vals


def need(paths: dict[str, Path | None], key: str) -> Path:
    val = paths.get(key)
    if val is None:
        flag = "--" + key.replace("_", "-")
        raise SystemExit(f"{key} unresolved: pass {flag} or set RADIO_BENCH_{key.upper()}")
    return val


def suite_root(paths: dict[str, Path | None]) -> Path:
    return need(paths, "data_dir") / "casaguides" / SUITE_ID


def tasks_out_dir(paths: dict[str, Path | None]) -> Path:
    return need(paths, "work_dir") / f"tasks_{SUITE_ID}"


def casa_env(paths: dict[str, Path | None]) -> dict[str, str]:
    env = dict(os.environ)
    work = paths.get("work_dir")
    if work is not None:
        env.setdefault("CASASITECONFIG", str(work / "casaconfig_local.py"))
        env.setdefault("TMPDIR", str(work / "tmp"))
        env.setdefault("MPLCONFIGDIR", str(work / "mplconfig"))
    return env


def config_flags(paths: dict[str, Path | None]) -> list[str]:
    out: list[str] = []
    for key in ("work_dir", "data_dir", "runs_dir"):
        if paths.get(key) is not None:
            out += ["--" + key.replace("_", "-"), str(paths[key])]
    return out


def run_cmd(cmd: list[str], *, dry: bool, env: dict[str, str] | None = None) -> None:
    print(f"$ (cd {shlex.quote(str(REPO))} && {shlex.join(cmd)})", flush=True)
    if dry:
        return
    proc = subprocess.run(cmd, cwd=REPO, env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"command failed (exit {proc.returncode}): {shlex.join(cmd)}")


def py_cmd(args: argparse.Namespace) -> list[str]:
    return shlex.split(args.python)


def suite_task_ids() -> list[str]:
    raw = tomllib.loads(SUITE_TOML.read_text(encoding="utf-8"))
    return sorted(raw.get("tasks", {}))


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


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def download_resume(url: str, dest: Path, total: int) -> None:
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        part.unlink()
        have = 0
    req = urllib.request.Request(url, headers={"User-Agent": "radio-bench-fetch"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as resp:
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0
        with part.open(mode) as fh:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                fh.write(buf)
                have += len(buf)
    if have != total:
        raise SystemExit(f"{dest.name}: downloaded {have} bytes, expected {total}; rerun to resume")
    part.rename(dest)


def safe_extract(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(out_dir, filter="data")
            return
        root = out_dir.resolve()
        for m in tf.getmembers():
            target = (out_dir / m.name).resolve()
            if not str(target).startswith(str(root) + os.sep) or m.issym() or m.islnk():
                raise SystemExit(f"{tar_path.name}: unsafe member {m.name!r}")
        tf.extractall(out_dir)


def cmd_fetch(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    root = suite_root(paths)
    dl_dir = root / "downloads"
    ex_dir = root / "extracted"
    sources = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    mirror_root = paths.get("casaguides_dir")

    free = free_bytes(root)
    print(f"target filesystem free: {free / 1024**3:.1f} GB (minimum 150 GB) at {root}")
    if free < MIN_FREE_BYTES:
        raise SystemExit("refusing: target filesystem has < 150 GB free")

    for arc in sources["archives"]:
        name, url, sha, size = arc["name"], arc["url"], arc["sha256"], int(arc["bytes"])
        mirror = (mirror_root / SUITE_ID / name) if mirror_root else None
        marker = ex_dir / f".{name}.extracted"
        if mirror is not None and mirror.is_file() and mirror.stat().st_size == size:
            src = mirror
            print(f"[{name}] source: mirror {mirror} (read in place, not copied)")
        else:
            src = dl_dir / name
            print(f"[{name}] source: download {url} -> {src} (resume via {src.name}.part)")
        if args.dry_run:
            print(f"[{name}] would verify sha256 == {sha}")
            print(f"[{name}] would extract -> {ex_dir} (marker {marker})")
            continue
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha:
            print(f"[{name}] already extracted (marker matches sha256); skipping")
            continue
        if src != mirror and not (src.is_file() and src.stat().st_size == size):
            download_resume(url, src, size)
        got = sha256_file(src)
        if got != sha:
            if src == mirror:
                print(f"[{name}] mirror sha256 mismatch ({got}); downloading instead")
                src = dl_dir / name
                download_resume(url, src, size)
                got = sha256_file(src)
            if got != sha:
                raise SystemExit(f"[{name}] sha256 mismatch: got {got}, expected {sha}")
        print(f"[{name}] sha256 OK")
        safe_extract(src, ex_dir)
        marker.write_text(sha + "\n", encoding="utf-8")
        print(f"[{name}] extracted -> {ex_dir}")


# ---------------------------------------------------------------------------
# build / tasks / oracle / eval
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    data_dir = need(paths, "data_dir")
    env = casa_env(paths)
    casa_py = shlex.split(args.casa_python) if args.casa_python else py_cmd(args)
    run_cmd(
        [*casa_py, str(BUILDER), "checkpoints", "--data-dir", str(data_dir)],
        dry=args.dry_run,
        env=env,
    )
    run_cmd(
        [*py_cmd(args), str(BUILDER), "truth", "--data-dir", str(data_dir)],
        dry=args.dry_run,
        env=env,
    )


def cmd_tasks(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    out_dir = tasks_out_dir(paths)
    cmd = [
        *py_cmd(args),
        "-m",
        "radio_bench.build_tasks",
        *config_flags(paths),
        "--suites-dir",
        str(SUITES_DIR),
        "--output-dir",
        str(out_dir),
    ]
    if args.pack:
        cmd += ["--pack", args.pack]
    run_cmd(cmd, dry=args.dry_run, env=casa_env(paths))
    if args.dry_run:
        task_dirs = [out_dir / f"{t}__{arm}" for t in suite_task_ids() for arm in STAGE_ARMS]
    else:
        task_dirs = sorted(
            p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(TASKS_PREFIX)
        )
        if not task_dirs:
            raise SystemExit(f"no compiled task directories under {out_dir}")
    harbor = shlex.split(args.harbor)
    for tdir in task_dirs:
        run_cmd([*harbor, "lint", f"{tdir}/"], dry=args.dry_run)


def render_cmd(args: argparse.Namespace, paths: dict[str, Path | None], **kw: Any) -> list[str]:
    cmd = [*py_cmd(args), str(Path(__file__).resolve()), "_render-job", *config_flags(paths)]
    for key, val in kw.items():
        flag = "--" + key.replace("_", "-")
        if val is True:
            cmd.append(flag)
        elif val not in (None, False):
            cmd += [flag, str(val)]
    return cmd


def job_yaml_path(paths: dict[str, Path | None], job_name: str) -> Path:
    return need(paths, "work_dir") / "jobs" / f"{job_name}.yaml"


def cmd_oracle(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    job_name = f"oracle_{SUITE_ID}"
    run_cmd(
        render_cmd(
            args,
            paths,
            job_name=job_name,
            oracle=True,
            arms="00_none",
            n_attempts=1,
            pack=args.pack,
        ),
        dry=args.dry_run,
    )
    run_cmd(
        [*py_cmd(args), "-m", "radio_bench.cli", "run-job", str(job_yaml_path(paths, job_name))],
        dry=args.dry_run,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    job_name = args.job_name or f"{SUITE_ID}_{args.model}_{args.n_attempts}x"
    run_cmd(
        render_cmd(
            args,
            paths,
            job_name=job_name,
            model=args.model,
            arms=args.arms,
            n_attempts=args.n_attempts,
            thinking_levels=args.thinking_levels,
            pack=args.pack,
        ),
        dry=args.dry_run,
    )
    run_cmd(
        [*py_cmd(args), "-m", "radio_bench.cli", "run-job", str(job_yaml_path(paths, job_name))],
        dry=args.dry_run,
    )


def cmd_all(args: argparse.Namespace) -> None:
    for step in (cmd_fetch, cmd_build, cmd_tasks, cmd_oracle):
        print(f"=== {step.__name__.removeprefix('cmd_')} ===", flush=True)
        step(args)


def cmd_render_job(args: argparse.Namespace) -> None:
    """Internal: render one job YAML via render_jobs.compile_job_dict (needs radio_bench)."""
    import yaml

    from radio_bench import render_jobs
    from radio_bench.config import load_bench_config
    from radio_bench.grading.schema import SuiteManifest

    manifest = SuiteManifest.model_validate(tomllib.loads(SUITE_TOML.read_text(encoding="utf-8")))
    stage_ids = tuple(t for t, s in sorted(manifest.tasks.items()) if not s.is_e2e)
    e2e_ids = tuple(t for t, s in sorted(manifest.tasks.items()) if s.is_e2e)
    # render_jobs keeps its suite registry in module dicts; register this suite in-process only.
    render_jobs.SUITE_STAGE_TASKS[SUITE_ID] = stage_ids
    render_jobs.SUITE_E2E_TASKS[SUITE_ID] = e2e_ids
    render_jobs.SUITE_DEFAULT_CONCURRENCY[SUITE_ID] = manifest.resources.n_concurrent_trials

    paths = resolve_paths(args)
    cfg = load_bench_config(
        work_dir=paths["work_dir"], data_dir=paths["data_dir"], runs_dir=paths["runs_dir"]
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    levels = [x.strip() for x in args.thinking_levels.split(",")] if args.thinking_levels else None
    job = render_jobs.compile_job_dict(
        job_name=args.job_name,
        suites=[SUITE_ID],
        arms=arms,
        task_scope="all",
        oracle=bool(args.oracle),
        model_keys=[args.model] if args.model else (),
        thinking_levels=levels,
        pack_ids=[args.pack] if args.pack else None,
        n_attempts=int(args.n_attempts),
        suite_concurrency=manifest.resources.n_concurrent_trials,
        tasks_dir=tasks_out_dir(paths),
        cfg=cfg,
    )
    out = job_yaml_path(paths, args.job_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    print(json.dumps({"rendered_job": str(out), "n_tasks": len(job["tasks"])}, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="Print commands/paths only.")
    common.add_argument("--data-dir", default=None, help="Override data_dir.")
    common.add_argument("--work-dir", default=None, help="Override work_dir.")
    common.add_argument("--runs-dir", default=None, help="Override runs_dir.")
    common.add_argument("--casaguides-dir", default=None, help="Override local mirror root.")
    common.add_argument(
        "--python",
        default="pixi run python",
        help="Interpreter command with radio_bench importable (default: 'pixi run python').",
    )
    common.add_argument("--pack", default=None, help="Pack ID (default: bench.toml default_pack).")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", parents=[common], help="Fetch, verify, and extract archives.")
    p_build = sub.add_parser("build", parents=[common], help="Build checkpoints + ground truth.")
    p_build.add_argument(
        "--casa-python",
        default=None,
        help="Interpreter for the CASA-only checkpoint phase (default: --python), e.g. python3 "
        "inside the casa-base container.",
    )
    p_tasks = sub.add_parser("tasks", parents=[common], help="Compile tasks + harbor lint.")
    p_tasks.add_argument("--harbor", default="pixi run harbor", help="Harbor CLI command.")
    sub.add_parser("oracle", parents=[common], help="Render + run the oracle job.")
    p_eval = sub.add_parser("eval", parents=[common], help="Render + run a model eval job.")
    p_eval.add_argument("--model", required=True, help="Model key from configs/models.toml.")
    p_eval.add_argument("--arms", required=True, help="Comma-separated arms, e.g. 00_none,11_full.")
    p_eval.add_argument("--n-attempts", required=True, type=int, help="Attempts per cell.")
    p_eval.add_argument("--thinking-levels", default=None, help="Comma-separated thinking levels.")
    p_eval.add_argument("--job-name", default=None, help="Job name override.")
    p_all = sub.add_parser("all", parents=[common], help="fetch -> build -> tasks -> oracle.")
    p_all.add_argument("--casa-python", default=None, help="See build --casa-python.")
    p_all.add_argument("--harbor", default="pixi run harbor", help="Harbor CLI command.")

    p_r = sub.add_parser("_render-job", parents=[common])
    p_r.add_argument("--job-name", required=True)
    p_r.add_argument("--arms", required=True)
    p_r.add_argument("--n-attempts", required=True, type=int)
    p_r.add_argument("--oracle", action="store_true")
    p_r.add_argument("--model", default=None)
    p_r.add_argument("--thinking-levels", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "fetch": cmd_fetch,
        "build": cmd_build,
        "tasks": cmd_tasks,
        "oracle": cmd_oracle,
        "eval": cmd_eval,
        "all": cmd_all,
        "_render-job": cmd_render_job,
    }
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
