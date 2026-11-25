"""One entrypoint for the vla_3c129_pband dataset package (fetch, build, tasks, oracle, eval).

`fetch` uses only the Python standard library so it can run on a bare OSG node.
`build` runs the CASA builder (casatools/casatasks/numpy only).
`tasks`, `oracle`, and `eval` reuse radio_bench.build_tasks, radio_bench.render_jobs and
radio_bench.cli run-job from the repository environment (`pixi run python ...`).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SUITE_ID = "vla_3c129_pband"
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
SUITES_DIR = PKG_DIR / "suites"
BUILDER = PKG_DIR / f"build_{SUITE_ID}.py"
SOURCES_JSON = PKG_DIR / "sources.json"
MIN_FREE_BYTES = 150 * 1000**3
STAGE_TASKS = ("g1_3c129p_inspect", "g1_3c129p_precal", "g1_3c129p_calibrate", "g1_3c129p_image")
E2E_TASKS = ("g2_3c129p_e2e",)
CHUNK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Config (stdlib only: radio_bench.config is itself stdlib-only and is loaded by file path)
# ---------------------------------------------------------------------------


def _load_config_module() -> Any | None:
    try:
        import radio_bench.config as mod  # type: ignore[import-not-found]

        return mod
    except ImportError:
        pass
    cfg_py = REPO_ROOT / "src" / "radio_bench" / "config.py"
    if not cfg_py.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_rb_config", cfg_py)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rb_config"] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_paths(args: argparse.Namespace) -> SimpleNamespace:
    mod = _load_config_module()
    if mod is not None:
        cfg = mod.load_bench_config(
            repo_root=REPO_ROOT,
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            runs_dir=args.runs_dir,
            casaguides_dir=args.casaguides_dir,
        )
        out = SimpleNamespace(
            data_dir=cfg.data_dir,
            work_dir=cfg.work_dir,
            runs_dir=cfg.runs_dir,
            casaguides_dir=cfg.casaguides_dir,
        )
    else:
        env = os.environ

        def _pick(cli: str | None, var: str) -> Path | None:
            raw = cli or env.get(var, "").strip()
            return Path(raw).expanduser().resolve() if raw else None

        out = SimpleNamespace(
            data_dir=_pick(args.data_dir, "RADIO_BENCH_DATA_DIR"),
            work_dir=_pick(args.work_dir, "RADIO_BENCH_WORK_DIR"),
            runs_dir=_pick(args.runs_dir, "RADIO_BENCH_RUNS_DIR"),
            casaguides_dir=_pick(args.casaguides_dir, "RADIO_BENCH_CASAGUIDES_DIR"),
        )
        if out.data_dir is None:
            raise SystemExit(
                "error: no bench config found; pass --data-dir or RADIO_BENCH_DATA_DIR"
            )
    for label in ("data_dir", "work_dir", "runs_dir", "casaguides_dir"):
        val = getattr(out, label)
        if val is not None and (str(val) == "/tmp" or str(val).startswith("/tmp/")):
            raise SystemExit(f"error: {label} must not be under /tmp: {val}")
    out.suite_root = out.data_dir / "casaguides" / SUITE_ID
    out.downloads = out.suite_root / "downloads"
    out.extracted = out.suite_root / "extracted"
    return out


def _say(msg: str) -> None:
    print(msg, flush=True)


def _check_free(path: Path, dry: bool) -> None:
    probe = path
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    _say(f"free space on {probe}: {free / 1e9:.1f} GB (minimum {MIN_FREE_BYTES / 1e9:.0f} GB)")
    if free < MIN_FREE_BYTES:
        msg = (
            f"refusing: {probe} has {free / 1e9:.1f} GB free, need >= {MIN_FREE_BYTES / 1e9:.0f} GB"
        )
        if dry:
            _say(f"[dry-run] would refuse: {msg}")
        else:
            raise SystemExit(f"error: {msg}")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(CHUNK)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _verified(path: Path, sha: str, stamp_dir: Path) -> bool:
    """Hash once and remember (path, size, mtime) so reruns skip the 24 GB hash."""
    st = path.stat()
    key = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
    stamp = stamp_dir / f".verified_{path.name}_{key}.json"
    want = {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": sha,
    }
    if stamp.is_file():
        try:
            if json.loads(stamp.read_text(encoding="utf-8")) == want:
                _say(f"sha256 ok (cached stamp {stamp.name}): {path}")
                return True
        except (OSError, ValueError):
            pass
    _say(f"hashing {path} ({st.st_size / 1e9:.2f} GB) ...")
    t0 = time.time()
    got = _sha256(path)
    _say(f"sha256 {got} in {time.time() - t0:.0f}s")
    if got != sha:
        return False
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(want, indent=2) + "\n", encoding="utf-8")
    return True


def _download(url: str, dest: Path, total: int) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "radio-bench-fetch/1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    _say(f"downloading {url} -> {part} (resume at {have} of {total} bytes)")
    with urllib.request.urlopen(req, timeout=120) as resp:
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0
        with part.open(mode) as fh:
            last = time.time()
            while True:
                blk = resp.read(CHUNK)
                if not blk:
                    break
                fh.write(blk)
                have += len(blk)
                if time.time() - last > 30:
                    _say(f"  {have / 1e9:.2f} / {total / 1e9:.2f} GB")
                    last = time.time()
    part.rename(dest)


def _extract(archive: Path, dest: Path, sha: str, top: str, dry: bool) -> None:
    marker = dest / f".extracted_{archive.name}.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("sha256") == sha:
                _say(f"already extracted (marker {marker})")
                return
        except (OSError, ValueError):
            pass
    target_top = dest / top
    if dry:
        _say(f"[dry-run] would extract {archive} -> {dest} (top dir {top})")
        return
    if target_top.exists():
        raise SystemExit(f"error: {target_top} exists without a valid marker; remove it first")
    dest.mkdir(parents=True, exist_ok=True)
    _say(f"extracting {archive} -> {dest} ...")
    t0 = time.time()
    with tarfile.open(archive, "r:*") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:
            for m in tf.getmembers():
                p = (dest / m.name).resolve()
                if dest.resolve() not in p.parents and p != dest.resolve():
                    raise SystemExit(f"error: unsafe member path {m.name}")
                if m.issym() or m.islnk():
                    raise SystemExit(f"error: link member refused {m.name}")
            tf.extractall(dest)
    n_bytes = sum(p.stat().st_size for p in target_top.rglob("*") if p.is_file())
    info = {
        "archive": str(archive),
        "sha256": sha,
        "extracted_top": str(target_top),
        "extracted_bytes": n_bytes,
        "elapsed_s": round(time.time() - t0, 1),
    }
    marker.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    _say(f"extracted {n_bytes} bytes in {info['elapsed_s']}s")


def cmd_fetch(args: argparse.Namespace, paths: SimpleNamespace) -> int:
    sources = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    _check_free(paths.suite_root, args.dry_run)
    for arc in sources["archives"]:
        name, sha, nbytes = arc["name"], arc["sha256"], int(arc["bytes"])
        mirror = (paths.casaguides_dir / SUITE_ID / name) if paths.casaguides_dir else None
        local = paths.downloads / name
        src: Path | None = None
        if mirror is not None and mirror.is_file() and mirror.stat().st_size == nbytes:
            if args.dry_run:
                _say(f"[dry-run] would verify sha256 of mirror {mirror} and read it in place")
                src = mirror
            elif _verified(mirror, sha, paths.downloads):
                src = mirror
            else:
                _say(f"mirror sha256 mismatch, ignoring {mirror}")
        if src is None and local.is_file():
            if args.dry_run:
                _say(f"[dry-run] would verify sha256 of {local}")
                src = local
            elif _verified(local, sha, paths.downloads):
                src = local
            else:
                raise SystemExit(f"error: sha256 mismatch for {local}; delete it and rerun")
        if src is None:
            if args.dry_run:
                _say(f"[dry-run] would download {arc['url']} -> {local} (resume) and verify sha256")
                src = local
            else:
                paths.downloads.mkdir(parents=True, exist_ok=True)
                _download(arc["url"], local, nbytes)
                if not _verified(local, sha, paths.downloads):
                    raise SystemExit(f"error: sha256 mismatch after download: {local}")
                src = local
        if arc.get("extract", False):
            _extract(src, paths.extracted, sha, arc["extracted_top"], args.dry_run)
    return 0


# ---------------------------------------------------------------------------
# build / tasks / oracle / eval
# ---------------------------------------------------------------------------


def _casa_env(paths: SimpleNamespace) -> dict[str, str]:
    env = dict(os.environ)
    scratch = paths.suite_root / "_tmp"
    env.setdefault("TMPDIR", str(scratch))
    env.setdefault("MPLCONFIGDIR", str(scratch / "mplconfig"))
    if paths.work_dir is not None:
        site = paths.work_dir / "casaconfig_local.py"
        if site.is_file():
            env.setdefault("CASASITECONFIG", str(site))
    return env


def _run(cmd: list[str], *, dry: bool, env: dict[str, str] | None = None) -> int:
    _say(("[dry-run] " if dry else "+ ") + " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=False).returncode


def cmd_build(args: argparse.Namespace, paths: SimpleNamespace) -> int:
    _check_free(paths.suite_root, args.dry_run)
    env = _casa_env(paths)
    if not args.dry_run:
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    _say(
        f"TMPDIR={env['TMPDIR']} MPLCONFIGDIR={env['MPLCONFIGDIR']} "
        f"CASASITECONFIG={env.get('CASASITECONFIG', '')}"
    )
    cmd = [
        sys.executable,
        str(BUILDER),
        "--out-root",
        str(paths.suite_root),
        "--extracted",
        str(paths.extracted),
    ]
    return _run(cmd, dry=args.dry_run, env=env)


def _tasks_dir(paths: SimpleNamespace) -> Path:
    return paths.work_dir / f"tasks_{SUITE_ID}"


def cmd_tasks(args: argparse.Namespace, paths: SimpleNamespace) -> int:
    out_dir = _tasks_dir(paths)
    cmd = [
        sys.executable,
        "-m",
        "radio_bench.build_tasks",
        "--suites-dir",
        str(SUITES_DIR),
        "--output-dir",
        str(out_dir),
    ]
    if args.pack:
        cmd += ["--pack", args.pack]
    rc = _run(cmd, dry=args.dry_run)
    if rc != 0:
        return rc
    ids = (*STAGE_TASKS, *E2E_TASKS)
    if args.dry_run:
        dirs = [f"{out_dir}/{t}__<arm>/" for t in ids]
    else:
        dirs = sorted(
            f"{p}/" for p in out_dir.iterdir() if p.is_dir() and p.name.split("__")[0] in ids
        )
        if not dirs:
            raise SystemExit(f"error: no compiled task dirs for {SUITE_ID} in {out_dir}")
    harbor = shutil.which("harbor") or "harbor"
    return _run([harbor, "lint", *dirs], dry=args.dry_run)


def _render_job(
    args: argparse.Namespace,
    paths: SimpleNamespace,
    *,
    job_name: str,
    oracle: bool,
    arms: list[str],
    models: list[str],
    n_attempts: int,
    scope: str,
    thinking_levels: list[str] | None,
) -> Path:
    jobs_dir = paths.work_dir / "jobs"
    out = jobs_dir / f"{job_name}.yaml"
    desc = (
        f"radio_bench.render_jobs.compile_job_dict(suites=['{SUITE_ID}'], arms={arms}, "
        f"oracle={oracle}, models={models}, n_attempts={n_attempts}, scope='{scope}', "
        f"thinking_levels={thinking_levels}, tasks_dir='{_tasks_dir(paths)}') -> {out}"
    )
    if args.dry_run:
        _say(f"[dry-run] render {desc}")
        return out
    import yaml

    from radio_bench import render_jobs as rj

    # render_jobs keys suites by name in module dicts; register this suite for this process only.
    rj.SUITE_STAGE_TASKS[SUITE_ID] = STAGE_TASKS
    rj.SUITE_E2E_TASKS[SUITE_ID] = E2E_TASKS
    rj.SUITE_DEFAULT_CONCURRENCY[SUITE_ID] = 1
    job = rj.compile_job_dict(
        job_name=job_name,
        suites=[SUITE_ID],
        arms=arms,
        task_scope=scope,
        oracle=oracle,
        model_keys=tuple(models),
        thinking_levels=thinking_levels,
        pack_ids=[args.pack] if args.pack else None,
        n_attempts=n_attempts,
        suite_concurrency=1,
        tasks_dir=_tasks_dir(paths),
        jobs_out_dir=jobs_dir,
    )
    missing = [t["path"] for t in job["tasks"] if not Path(t["path"]).is_dir()]
    if missing:
        raise SystemExit(
            "error: compiled task dirs missing (run `tasks` first):\n" + "\n".join(missing)
        )
    jobs_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    _say(f"rendered {out}")
    return out


def cmd_oracle(args: argparse.Namespace, paths: SimpleNamespace) -> int:
    job = _render_job(
        args,
        paths,
        job_name=f"oracle_{SUITE_ID}",
        oracle=True,
        arms=["00_none"],
        models=[],
        n_attempts=1,
        scope="all",
        thinking_levels=None,
    )
    return _run([sys.executable, "-m", "radio_bench.cli", "run-job", str(job)], dry=args.dry_run)


def cmd_eval(args: argparse.Namespace, paths: SimpleNamespace) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    levels = [x.strip() for x in args.thinking_levels.split(",")] if args.thinking_levels else None
    name = f"{SUITE_ID}_{args.model}_{'-'.join(arms)}_n{args.n_attempts}"
    job = _render_job(
        args,
        paths,
        job_name=name,
        oracle=False,
        arms=arms,
        models=[args.model],
        n_attempts=args.n_attempts,
        scope=args.scope,
        thinking_levels=levels,
    )
    return _run([sys.executable, "-m", "radio_bench.cli", "run-job", str(job)], dry=args.dry_run)


def cmd_all(args: argparse.Namespace, paths: SimpleNamespace) -> int:
    for fn in (cmd_fetch, cmd_build, cmd_tasks, cmd_oracle):
        rc = fn(args, paths)
        if rc != 0:
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dry-run", action="store_true", help="Print commands and paths only.")
        p.add_argument("--data-dir", default=None, help="Override bench data_dir.")
        p.add_argument("--work-dir", default=None, help="Override bench work_dir.")
        p.add_argument("--runs-dir", default=None, help="Override bench runs_dir.")
        p.add_argument("--casaguides-dir", default=None, help="Override local mirror dir.")
        p.add_argument("--pack", default=None, help="Skill pack id (default: bench default).")

    for name, hlp in (
        ("fetch", "Fetch (mirror or URL), verify sha256, extract (stdlib only)."),
        ("build", "Run the CASA builder: slice, checkpoints, ground truth."),
        ("tasks", "Compile Harbor tasks and run harbor lint on each."),
        ("oracle", "Render and run the oracle job for this suite."),
        ("all", "fetch -> build -> tasks -> oracle (never eval)."),
    ):
        _common(sub.add_parser(name, help=hlp))
    ev = sub.add_parser("eval", help="Render and run an LLM eval job (maintainer picks all args).")
    _common(ev)
    ev.add_argument("--model", required=True, help="Model key from configs/models.toml.")
    ev.add_argument("--arms", required=True, help="Comma-separated arms, e.g. 00_none,11_full.")
    ev.add_argument("--n-attempts", required=True, type=int, help="Attempts per cell.")
    ev.add_argument("--scope", default="all", choices=("all", "stage", "e2e"))
    ev.add_argument("--thinking-levels", default=None, help="Comma-separated thinking levels.")

    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    _say(
        f"suite={SUITE_ID} data_dir={paths.data_dir} suite_root={paths.suite_root} "
        f"work_dir={paths.work_dir} casaguides_dir={paths.casaguides_dir}"
    )
    handlers = {
        "fetch": cmd_fetch,
        "build": cmd_build,
        "tasks": cmd_tasks,
        "oracle": cmd_oracle,
        "eval": cmd_eval,
        "all": cmd_all,
    }
    return handlers[args.cmd](args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
