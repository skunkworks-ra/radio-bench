"""Unit tests for rootless Podman and storage hygiene checks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

import pytest
import yaml

from radio_bench.podman_env import (
    build_podman_env,
    check_podman_env,
    cleanup_stale_harbor_resources,
    ensure_docker_compose_plugin,
    smoke_compose_teardown,
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
    assert env["CONTAINER_TMPDIR"] == str(test_root / "podman_tmp")


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


def test_cleanup_stale_harbor_resources_skips_running_and_non_harbor(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    ps_payload = json.dumps(
        [
            {
                "Id": "cid_running_harbor",
                "Names": ["s1_inspect__10_tools__env-main-1"],
                "State": "running",
                "Networks": ["s1_inspect__10_tools__env_default"],
            },
            {
                "Id": "cid_exited_harbor",
                "Names": ["s1_image__00_none__verifier__trial-main-1"],
                "State": "exited",
                "Networks": ["s1_image__00_none__verifier__trial_default"],
            },
            {
                "Id": "cid_exited_other",
                "Names": ["unrelated-dev-container"],
                "State": "exited",
                "Networks": ["bridge"],
            },
        ]
    )
    vol_payload = "\n".join(
        [
            "s1_image__00_none__verifier__trial_verifier_ws",
            "s1_inspect__10_tools__env_workspace_vol",
            "unrelated_postgres_data",
        ]
    )
    net_payload = "\n".join(
        [
            "podman",
            "s1_image__00_none__verifier__trial_default",
            "custom_unrelated_net",
        ]
    )

    aardvark_dir = tmp_path / "aardvark-dns"
    aardvark_dir.mkdir(parents=True)
    alive_pid_file = aardvark_dir / "alive.pid"
    alive_pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    dead_pid_file = aardvark_dir / "dead.pid"
    dead_pid_file.write_text("99999999\n", encoding="utf-8")

    def _runner(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] == ["podman", "ps", "-a"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=ps_payload, stderr="")
        if cmd[:3] == ["podman", "volume", "ls"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=vol_payload, stderr="")
        if cmd[:3] == ["podman", "network", "ls"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=net_payload, stderr="")
        if cmd[:3] in (
            ["podman", "network", "disconnect"],
            ["podman", "rm", "-f"],
            ["podman", "volume", "rm"],
            ["podman", "network", "rm"],
        ):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    summary = cleanup_stale_harbor_resources(
        include_running=False,
        base_env={"TMPDIR": str(tmp_path / "tmp"), "PATH": "/usr/bin"},
        runner=_runner,
        aardvark_dir=aardvark_dir,
    )
    assert summary["removed_containers"] == ["s1_image__00_none__verifier__trial-main-1"]
    assert summary["removed_volumes"] == [
        "s1_image__00_none__verifier__trial_verifier_ws",
        "s1_inspect__10_tools__env_workspace_vol",
    ]
    assert summary["removed_networks"] == ["s1_image__00_none__verifier__trial_default"]
    assert summary["removed_stale_pids"] == [str(dead_pid_file)]
    assert alive_pid_file.is_file()
    assert not dead_pid_file.exists()
    assert not any("cid_running_harbor" in c for c in calls)
    assert not any("cid_exited_other" in c for c in calls)
    assert not any("unrelated_postgres_data" in c for c in calls)
    assert not any("custom_unrelated_net" in c for c in calls)


def test_cleanup_stale_harbor_resources_include_running_removes_orphans(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    ps_payload = json.dumps(
        [
            {
                "Id": "cid_running_harbor",
                "Names": ["s1_inspect__10_tools__env-main-1"],
                "State": "running",
                "Networks": ["s1_inspect__10_tools__env_default"],
            },
            {
                "Id": "cid_running_other",
                "Names": ["keep-me-running"],
                "State": "running",
                "Networks": [],
            },
        ]
    )

    def _runner(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] == ["podman", "ps", "-a"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=ps_payload, stderr="")
        if cmd[:3] in (["podman", "volume", "ls"], ["podman", "network", "ls"]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    summary = cleanup_stale_harbor_resources(
        include_running=True,
        base_env={"TMPDIR": str(tmp_path / "tmp"), "PATH": "/usr/bin"},
        runner=_runner,
        aardvark_dir=tmp_path / "missing-aardvark",
    )
    assert summary["removed_containers"] == ["s1_inspect__10_tools__env-main-1"]
    assert not any("cid_running_other" in c for c in calls)


def test_smoke_compose_teardown_validates_up_tcp_probe_and_down_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_cmds: list[list[str]] = []
    captured_specs: dict[str, dict[str, object]] = {}

    def _ok_runner(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(cmd)
        if cmd == ["docker", "compose", "version"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker Compose v2.29.0\n", stderr="")
        if "-f" in cmd:
            f_idx = cmd.index("-f")
            compose_path = Path(cmd[f_idx + 1])
            p_idx = cmd.index("--project-name")
            proj = cmd[p_idx + 1]
            if compose_path.is_file() and proj not in captured_specs:
                captured_specs[proj] = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
            if cmd[-2:] == ["ps", "main"] or cmd[-3:] == ["ps", "-q", "main"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="cid_smoke_main\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["podman", "exec", "cid_smoke_main"]:
            assert "8000" in cmd[-1] and "8001" in cmd[-1] and "8002" in cmd[-1]
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["podman", "volume", "ls"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="unrelated_vol\n", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    res_ok = smoke_compose_teardown(
        base_env={"TMPDIR": str(tmp_path / "tmp"), "PATH": "/usr/bin"},
        runner=_ok_runner,
        scratch_root=tmp_path / "smoke_tmp",
    )
    assert res_ok["ok"] is True, res_ok
    assert res_ok["leftover_volumes"] == []
    ver_proj = str(res_ok["verifier"]["project"])
    tools_proj = str(res_ok["tools"]["project"])

    ver_spec = captured_specs[ver_proj]
    assert ver_spec["services"]["main"]["network_mode"] == "none"
    assert "verifier_ws:/workspace" in ver_spec["services"]["main"]["volumes"]

    tools_spec = captured_specs[tools_proj]
    assert tools_spec["services"]["main"]["network_mode"] == "pasta"
    assert "mcp-servers:127.0.0.1" in tools_spec["services"]["main"]["extra_hosts"]
    assert tools_spec["services"]["mcp-servers"]["network_mode"] == "service:main"
    assert tools_spec["services"]["mcp-servers"]["environment"]["RADIO_MCP_HOST"] == "127.0.0.1"
    assert any(
        c[-5:] == ["down", "--rmi", "local", "--volumes", "--remove-orphans"] for c in seen_cmds
    )

    def _fail_probe_runner(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["podman", "exec"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Timed out waiting")
        return _ok_runner(cmd, env)

    res_fail = smoke_compose_teardown(
        base_env={"TMPDIR": str(tmp_path / "tmp"), "PATH": "/usr/bin"},
        runner=_fail_probe_runner,
        scratch_root=tmp_path / "smoke_tmp_fail",
    )
    assert res_fail["ok"] is False
    assert res_fail["tools"]["tcp_probe_rc"] == 1
    assert any("TCP probe failed" in e for e in res_fail["errors"])
