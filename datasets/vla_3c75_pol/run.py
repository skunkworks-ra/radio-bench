"""One entrypoint for the vla_3c75_pol reference dataset (fetch, build, tasks, oracle, eval).

`fetch` and `build --stage casa` need only the Python standard library plus (for the builder)
casatools/casatasks/numpy, so they can run on an OSG node or inside the casa-base container.
`build --stage truth`, `tasks`, `oracle` and `eval` need the radio-bench pixi environment.
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
import time
import urllib.request
from pathlib import Path
from typing import Any

SUITE_ID = "vla_3c75_pol"
HERE = Path(__file__).resolve().parent
SUITES_DIR = HERE / "suites"
SUITE_DIR = SUITES_DIR / SUITE_ID
BUILDER = HERE / f"build_{SUITE_ID}.py"
SOURCES_JSON = HERE / "sources.json"
MIN_FREE_BYTES = 150 * 1024**3
CHUNK = 8 * 1024 * 1024

IMAGE_TASK = "g1_3c75pol_image"
INSPECT_TASK = "g1_3c75pol_inspect"
E2E_TASK = "g2_3c75pol_e2e"
STAGE_TASKS = (INSPECT_TASK, IMAGE_TASK)
E2E_TASKS = (E2E_TASK,)


# ---------------------------------------------------------------------------
# Path resolution (radio_bench config when importable, else CLI / env only)
# ---------------------------------------------------------------------------


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    """Return data_dir, casaguides_dir, work_dir, runs_dir (CLI > env/bench config)."""
    out: dict[str, Path | None] = {
        "data_dir": None,
        "casaguides_dir": None,
        "work_dir": None,
        "runs_dir": None,
    }
    try:
        from radio_bench.config import load_bench_config

        cfg = load_bench_config(
            config_path=args.config,
            data_dir=args.data_dir,
            casaguides_dir=args.casaguides_dir,
            work_dir=args.work_dir,
            runs_dir=args.runs_dir,
        )
        out.update(
            data_dir=cfg.data_dir,
            casaguides_dir=cfg.casaguides_dir,
            work_dir=cfg.work_dir,
            runs_dir=cfg.runs_dir,
        )
    except ImportError:
        env_map = {
            "data_dir": "RADIO_BENCH_DATA_DIR",
            "casaguides_dir": "RADIO_BENCH_CASAGUIDES_DIR",
            "work_dir": "RADIO_BENCH_WORK_DIR",
            "runs_dir": "RADIO_BENCH_RUNS_DIR",
        }
        for key, env_name in env_map.items():
            cli_val = getattr(args, key, None)
            raw = cli_val if cli_val is not None else os.environ.get(env_name)
            out[key] = Path(raw).expanduser().resolve() if raw else None
    for key, val in out.items():
        if val is not None and (str(val) == "/tmp" or str(val).startswith("/tmp/")):
            raise SystemExit(f"Refusing {key}={val}: /tmp is never a work area")
    if out["data_dir"] is None:
        raise SystemExit("data_dir unresolved: pass --data-dir or set RADIO_BENCH_DATA_DIR")
    return out


def suite_root(paths: dict[str, Path | None]) -> Path:
    assert paths["data_dir"] is not None
    return paths["data_dir"] / "casaguides" / SUITE_ID


def load_sources() -> dict[str, Any]:
    return json.loads(SOURCES_JSON.read_text(encoding="utf-8"))


def show(cmd: list[str], env_extra: dict[str, str] | None = None) -> str:
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env_extra or {}).items())
    return (prefix + " " if prefix else "") + " ".join(shlex.quote(c) for c in cmd)


def run_cmd(cmd: list[str], *, dry_run: bool, env_extra: dict[str, str] | None = None) -> None:
    print(f"$ {show(cmd, env_extra)}", flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env.update(env_extra or {})
    subprocess.run(cmd, check=True, env=env)


def pixi_tool(name: str) -> str:
    """Prefer the tool next to the running interpreter (pixi env), else PATH."""
    cand = Path(sys.executable).parent / name
    if cand.is_file():
        return str(cand)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name!r} not found next to {sys.executable} or on PATH")
    return found


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


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def download_with_resume(url: str, dest: Path, expected_bytes: int | None) -> None:
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "radio-bench-fetch/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = getattr(resp, "status", 200)
        mode = "ab" if (have and status == 206) else "wb"
        if mode == "wb":
            have = 0
        with part.open(mode) as fh:
            last = time.monotonic()
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                fh.write(buf)
                have += len(buf)
                if time.monotonic() - last > 30:
                    print(f"  {dest.name}: {have / 1e9:.2f} GB", flush=True)
                    last = time.monotonic()
    if expected_bytes is not None and part.stat().st_size != expected_bytes:
        raise SystemExit(
            f"{dest.name}: size {part.stat().st_size} != expected {expected_bytes}; rerun to resume"
        )
    part.rename(dest)


def safe_extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with tarfile.open(archive, "r:*") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(out_dir, filter="data")
            return
        members = tf.getmembers()
        for m in members:
            target = (out_dir / m.name).resolve()
            if not (target == root or str(target).startswith(str(root) + os.sep)):
                raise SystemExit(f"Unsafe path in {archive.name}: {m.name}")
            if m.issym() or m.islnk() or m.isdev():
                raise SystemExit(f"Refusing link/device member in {archive.name}: {m.name}")
        tf.extractall(out_dir, members=members)


def cmd_fetch(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    root = suite_root(paths)
    downloads = root / "downloads"
    extracted = root / "extracted"
    mirror_dir = (paths["casaguides_dir"] / SUITE_ID) if paths["casaguides_dir"] else None
    free = free_bytes(root)
    print(f"suite_root={root}  free={free / 1024**3:.1f} GiB  mirror={mirror_dir}")
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"Refusing fetch: {free / 1024**3:.1f} GiB free on target filesystem (< 150 GiB)"
        )

    for arc in load_sources()["archives"]:
        name, sha, nbytes = arc["name"], arc["sha256"], int(arc["bytes"])
        stamp = extracted / f".{name}.extracted.json"
        mirror = mirror_dir / name if mirror_dir else None
        use_mirror = bool(mirror and mirror.is_file() and mirror.stat().st_size == nbytes)
        src = mirror if (use_mirror and mirror is not None) else downloads / name
        print(f"[{name}] source={'mirror' if use_mirror else 'download'} {src}")
        if stamp.is_file() and not args.force:
            rec = json.loads(stamp.read_text(encoding="utf-8"))
            if rec.get("sha256") == sha:
                print(f"[{name}] already extracted (stamp {stamp}); skip")
                continue
        if args.dry_run:
            if not use_mirror and not src.is_file():
                print(f"[{name}] would download {arc['url']} -> {src} (resume via .part)")
            print(f"[{name}] would verify sha256={sha} and extract to {extracted}/")
            continue
        if not use_mirror and not src.is_file():
            download_with_resume(arc["url"], src, nbytes)
        print(f"[{name}] hashing {src} ...", flush=True)
        got = sha256_file(src)
        if got != sha:
            if use_mirror:
                raise SystemExit(f"[{name}] mirror sha256 mismatch: {got} != {sha}")
            src.unlink()
            raise SystemExit(f"[{name}] download sha256 mismatch ({got}); deleted, rerun")
        print(f"[{name}] sha256 ok; extracting to {extracted}/ ...", flush=True)
        safe_extract(src, extracted)
        extracted.mkdir(parents=True, exist_ok=True)
        stamp.write_text(
            json.dumps(
                {"sha256": sha, "source": str(src), "extracted_to": arc["extracts_to"]}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[{name}] done -> {extracted / arc['extracts_to']}")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def casa_env(paths: dict[str, Path | None]) -> dict[str, str]:
    """CASA env for the builder; keep any CASASITECONFIG already set (casa-base container)."""
    env: dict[str, str] = {}
    work = paths["work_dir"]
    if "CASASITECONFIG" not in os.environ and work is not None:
        site = work / "casaconfig_local.py"
        if site.is_file():
            env["CASASITECONFIG"] = str(site)
    if work is not None:
        env["TMPDIR"] = str(work / "tmp")
        env["MPLCONFIGDIR"] = str(work / "mplconfig")
    return env


def finalize_truth(root: Path, *, dry_run: bool) -> None:
    """Add raw-data integrity hashes (radio_bench.compute_raw_data_blake2b) to ground truth.

    No target numbers are computed here: targets come from REFERENCE.json via the builder.
    """
    print("truth: blake2b of raw/input.ms (inspect, e2e) and post_polcal/input.ms (image)")
    print(f"truth: update {root}/<task_id>/ground_truth.json for {[*STAGE_TASKS, *E2E_TASKS]}")
    if dry_run:
        return
    from radio_bench.grading.extractors.common import compute_raw_data_blake2b

    hashes = {
        "raw": compute_raw_data_blake2b(root / "raw" / "input.ms"),
        "post_polcal": compute_raw_data_blake2b(root / "post_polcal" / "input.ms"),
    }
    which = {INSPECT_TASK: "raw", IMAGE_TASK: "post_polcal", E2E_TASK: "raw"}
    for tid, ckpt in which.items():
        gt_path = root / tid / "ground_truth.json"
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        gt["raw_data_blake2b"] = hashes[ckpt]
        for block in ("inspect_ground_truth", "image_ground_truth"):
            if isinstance(gt.get(block), dict):
                gt[block]["raw_data_blake2b"] = hashes[ckpt]
        gt_path.write_text(json.dumps(gt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(hashes, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    root = suite_root(paths)
    extracted = root / "extracted"
    if args.stage in ("casa", "all"):
        if not args.dry_run and not (extracted / "TDRW0001_calibrated.ms" / "table.dat").is_file():
            raise SystemExit(f"Missing {extracted}/TDRW0001_calibrated.ms; run `fetch` first")
        env = casa_env(paths)
        if not args.dry_run:
            for key in ("TMPDIR", "MPLCONFIGDIR"):
                if key in env:
                    Path(env[key]).mkdir(parents=True, exist_ok=True)
        run_cmd(
            [sys.executable, str(BUILDER), "--root", str(root), "--extracted", str(extracted)],
            dry_run=args.dry_run,
            env_extra=env,
        )
    if args.stage in ("truth", "all"):
        finalize_truth(root, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# tasks / oracle / eval
# ---------------------------------------------------------------------------


def tasks_dir(paths: dict[str, Path | None]) -> Path:
    if paths["work_dir"] is None:
        raise SystemExit("work_dir unresolved (needs the radio-bench pixi environment)")
    return paths["work_dir"] / f"tasks_{SUITE_ID}"


def config_flags(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for flag, val in (
        ("--config", args.config),
        ("--work-dir", args.work_dir),
        ("--data-dir", args.data_dir),
        ("--runs-dir", args.runs_dir),
    ):
        if val is not None:
            out += [flag, str(val)]
    return out


def cmd_tasks(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    out_dir = tasks_dir(paths)
    cmd = [
        sys.executable,
        "-m",
        "radio_bench.build_tasks",
        *config_flags(args),
        "--suites-dir",
        str(SUITES_DIR),
        "--output-dir",
        str(out_dir),
    ]
    if args.pack:
        cmd += ["--pack", args.pack]
    run_cmd(cmd, dry_run=args.dry_run)
    harbor = pixi_tool("harbor") if not args.dry_run else "harbor"
    prefixes = tuple(f"{t}__" for t in (*STAGE_TASKS, *E2E_TASKS))
    if args.dry_run:
        run_cmd([harbor, "lint", f"{out_dir}/<each {'|'.join(prefixes)}* dir>/"], dry_run=True)
        return
    task_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(prefixes))
    if not task_dirs:
        raise SystemExit(f"No compiled task dirs under {out_dir}")
    from radio_bench.podman_env import build_podman_env  # harbor lint needs the Podman/Compose env

    lint_env = {k: v for k, v in build_podman_env().items() if os.environ.get(k) != v}
    for tdir in task_dirs:
        run_cmd([harbor, "lint", f"{tdir}/"], dry_run=False, env_extra=lint_env)


def render_job(
    args: argparse.Namespace,
    *,
    job_name: str,
    oracle: bool,
    arms: list[str],
    model_keys: list[str],
    n_attempts: int,
) -> Path:
    """Render a Harbor job with radio_bench.render_jobs.compile_job_dict for this suite.

    render_jobs only knows the built-in suite keys, so this registers SUITE_ID in its
    in-process task tables (no source edits) and points tasks_dir at tasks_<suite_id>.
    """
    paths = resolve_paths(args)
    t_dir = tasks_dir(paths)
    assert paths["work_dir"] is not None
    jobs_dir = paths["work_dir"] / "jobs"
    out_path = jobs_dir / f"{job_name}.yaml"
    print(
        f"render: job={job_name} oracle={oracle} arms={arms} models={model_keys} "
        f"n_attempts={n_attempts} tasks_dir={t_dir} -> {out_path}"
    )
    if args.dry_run:
        return out_path
    import yaml

    from radio_bench import render_jobs

    render_jobs.SUITE_STAGE_TASKS[SUITE_ID] = STAGE_TASKS
    render_jobs.SUITE_E2E_TASKS[SUITE_ID] = E2E_TASKS
    render_jobs.SUITE_DEFAULT_CONCURRENCY[SUITE_ID] = 1
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job = render_jobs.compile_job_dict(
        job_name=job_name,
        suites=[SUITE_ID],
        arms=arms,
        task_scope="all",
        oracle=oracle,
        model_keys=tuple(model_keys),
        pack_ids=[args.pack] if args.pack else None,
        n_attempts=n_attempts,
        suite_concurrency=1,
        max_retries=0 if oracle else None,
        tasks_dir=t_dir,
        runs_dir=paths["runs_dir"],
        jobs_out_dir=jobs_dir,
    )
    out_path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    return out_path


def run_job(args: argparse.Namespace, job_yaml: Path) -> None:
    run_cmd(
        [sys.executable, "-m", "radio_bench.cli", "run-job", str(job_yaml)], dry_run=args.dry_run
    )


def cmd_oracle(args: argparse.Namespace) -> None:
    job = render_job(
        args,
        job_name=f"oracle_{SUITE_ID}",
        oracle=True,
        arms=["00_none"],
        model_keys=[],
        n_attempts=1,
    )
    run_job(args, job)


def cmd_eval(args: argparse.Namespace) -> None:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("--arms must list at least one arm")
    if args.n_attempts < 1:
        raise SystemExit("--n-attempts must be >= 1")
    job = render_job(
        args,
        job_name=f"eval_{SUITE_ID}_{args.model}",
        oracle=False,
        arms=arms,
        model_keys=[args.model],
        n_attempts=args.n_attempts,
    )
    run_job(args, job)


def cmd_all(args: argparse.Namespace) -> None:
    args.stage = "all"
    cmd_fetch(args)
    cmd_build(args)
    cmd_tasks(args)
    cmd_oracle(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="Print commands and paths only.")
    common.add_argument("--config", type=Path, default=None, help="Explicit bench.toml path.")
    common.add_argument("--data-dir", type=Path, default=None, help="Override data_dir.")
    common.add_argument("--casaguides-dir", type=Path, default=None, help="Override mirror dir.")
    common.add_argument("--work-dir", type=Path, default=None, help="Override work_dir.")
    common.add_argument("--runs-dir", type=Path, default=None, help="Override runs_dir.")
    common.add_argument("--pack", default=None, help="Pack id for tasks/oracle/eval.")

    parser = argparse.ArgumentParser(description=f"radio-bench dataset package {SUITE_ID}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", parents=[common], help="Verify/download and extract archives.")
    p.add_argument("--force", action="store_true", help="Re-verify and re-extract.")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", parents=[common], help="Build checkpoints and ground truth.")
    p.add_argument("--stage", choices=("casa", "truth", "all"), default="all")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("tasks", parents=[common], help="Compile Harbor tasks and lint them.")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("oracle", parents=[common], help="Render and run the oracle job.")
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser("eval", parents=[common], help="Render and run a model evaluation job.")
    p.add_argument("--model", required=True, help="Model key from configs/models.toml.")
    p.add_argument("--arms", required=True, help="Comma-separated arms, e.g. 00_none,10_tools.")
    p.add_argument("--n-attempts", type=int, required=True, help="Attempts per cell.")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("all", parents=[common], help="fetch -> build -> tasks -> oracle (no eval).")
    p.add_argument("--force", action="store_true", help="Re-verify and re-extract on fetch.")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
