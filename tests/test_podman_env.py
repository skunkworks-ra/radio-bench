"""Unit tests for rootless Podman and storage hygiene checks."""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

from radio_bench.podman_env import (
    build_podman_env,
    check_podman_env,
    ensure_docker_compose_plugin,
)


def _fake_runner_ok(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    if cmd[:2] == ["docker", "version"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="5.8.8\n", stderr="")
    if cmd[:2] == ["podman", "info"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="5.8.8|true\n", stderr="")
    if cmd[:2] == ["docker", "compose"]:
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Docker Compose version v2.29.0\n", stderr=""
        )
    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected command")


def test_build_podman_env_replaces_slash_tmp(tmp_path: Path) -> None:
    assert not str(tmp_path).startswith("/tmp"), f"pytest basetemp must not be /tmp: {tmp_path}"
    test_root = tmp_path / "test_root"
    data_root = tmp_path / "data_root"
    env = build_podman_env(
        {"TMPDIR": "/tmp/bad", "PATH": "/usr/bin"},
        uid=1000,
        test_root=test_root,
        data_root=data_root,
    )
    assert env["DOCKER_HOST"] == "unix:///run/user/1000/podman/podman.sock"
    assert env["TMPDIR"] == str(test_root / "tmp")
    assert not env["TMPDIR"].startswith("/tmp")
    assert env["CONTAINER_TMPDIR"] == str(data_root / "podman_tmp")


def test_ensure_docker_compose_plugin_creates_symlink(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin" / "docker-compose"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.write_text("#!/bin/sh\necho ok\n")
    fake_bin.chmod(0o755)

    cfg_dir = tmp_path / "docker_cfg"
    result = ensure_docker_compose_plugin(docker_config_dir=cfg_dir, compose_binary=fake_bin)
    assert result == cfg_dir
    plugin = cfg_dir / "cli-plugins" / "docker-compose"
    assert plugin.is_symlink()
    assert plugin.resolve() == fake_bin.resolve()


def test_check_podman_env_passes_with_valid_unix_socket(tmp_path: Path) -> None:
    sock_path = tmp_path / "podman.sock"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    try:
        report = check_podman_env(
            uid=1000,
            socket_path=sock_path,
            test_root=tmp_path / "test_root",
            data_root=tmp_path / "data_root",
            runs_root=tmp_path / "runs_root",
            base_env={"PATH": "/usr/bin"},
            runner=_fake_runner_ok,
        )
        assert report.ok is True, report.errors
        assert report.socket_is_unix_socket is True
        assert report.podman_version == "5.8.8"
        assert report.podman_rootless is True
        assert report.compose_cmd == ["docker", "compose"]
        assert report.harbor_version is not None
        assert report.tmpdir_not_slash_tmp is True
    finally:
        srv.close()
        sock_path.unlink(missing_ok=True)


def test_check_podman_env_rejects_slash_tmp_roots(tmp_path: Path) -> None:
    sock_path = tmp_path / "podman.sock"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    try:
        report = check_podman_env(
            uid=1000,
            socket_path=sock_path,
            test_root=tmp_path / "test_root",
            data_root=Path("/tmp/forbidden_data"),
            runs_root=tmp_path / "runs_root",
            base_env={"PATH": "/usr/bin"},
            runner=_fake_runner_ok,
        )
        assert report.ok is False
        assert report.storage_paths_ok is False
        assert any("/tmp" in err for err in report.errors)
    finally:
        srv.close()
        sock_path.unlink(missing_ok=True)
