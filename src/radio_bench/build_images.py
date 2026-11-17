"""CLI wrapper for rootless Podman image builds (`python -m radio_bench.build_images`)."""

from __future__ import annotations

import argparse
import os
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


def run_build_images(
    cfg: BenchConfig,
    *,
    config_path: Path | None = None,
    pack_id: str | None = None,
    verifier_only: bool = False,
    sidecar_only: bool = False,
    skip_smoke: bool = False,
) -> int:
    """Invoke `images/build_images.sh` with prepared non-`/tmp` runtime environment."""
    if verifier_only and sidecar_only:
        raise ValueError("Cannot combine --verifier-only and --sidecar-only")

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

    script = cfg.repo_root / "images" / "build_images.sh"
    cmd = ["bash", str(script)]
    if config_path is not None:
        cmd.extend(["--config", str(Path(os.path.expanduser(str(config_path))).resolve())])
    if pack_id:
        cmd.extend(["--pack", pack_id])
    if verifier_only:
        cmd.append("--verifier-only")
    if sidecar_only:
        cmd.append("--sidecar-only")
    if skip_smoke:
        cmd.append("--skip-smoke")

    proc = subprocess.run(cmd, cwd=str(cfg.work_dir), env=env, check=False)
    return int(proc.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build rootless Podman images (`casa-base`, `agent-base`, `mcp-sidecar`, `verifier`)."
    )
    add_config_cli_args(parser)
    parser.add_argument("--pack", type=str, default=None, help="Pack ID to build.")
    parser.add_argument(
        "--verifier-only",
        action="store_true",
        help="Build only localhost/radio-bench-verifier:latest.",
    )
    parser.add_argument(
        "--sidecar-only",
        action="store_true",
        help="Build only the configured pack's mcp-sidecar image.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip automated container isolation and MCP HTTP smoke checks.",
    )
    args = parser.parse_args(argv)
    cfg = config_from_cli_args(args)
    return run_build_images(
        cfg,
        config_path=args.config,
        pack_id=args.pack,
        verifier_only=bool(args.verifier_only),
        sidecar_only=bool(args.sidecar_only),
        skip_smoke=bool(args.skip_smoke),
    )


if __name__ == "__main__":
    raise SystemExit(main())
