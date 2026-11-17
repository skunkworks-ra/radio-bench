"""Rootless Podman socket, Compose v2, Harbor CLI, and work-directory readiness checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from radio_bench.config import (
    BenchConfig,
    add_config_cli_args,
    config_from_cli_args,
    is_under_slash_tmp,
    load_bench_config,
    reject_slash_tmp,
)

CommandRunner = Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class PodmanEnvReport:
    """Structured readiness report for rootless Podman + Compose + Harbor."""

    uid: int
    socket_path: str
    socket_exists: bool
    socket_is_unix_socket: bool
    socket_api_ok: bool
    docker_host: str
    podman_version: str | None
    podman_rootless: bool | None
    compose_cmd: list[str] | None
    compose_version: str | None
    harbor_version: str | None
    storage_paths_ok: bool
    tmpdir: str
    tmpdir_not_slash_tmp: bool
    ok: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_podman_socket(uid: int | None = None) -> Path:
    resolved_uid = os.getuid() if uid is None else uid
    return Path(f"/run/user/{resolved_uid}/podman/podman.sock")


def _is_under_slash_tmp(path: Path) -> bool:
    return is_under_slash_tmp(path)


def ensure_docker_compose_plugin(
    docker_config_dir: Path | None = None,
    compose_binary: Path | None = None,
) -> Path | None:
    """Ensure DOCKER_CONFIG/cli-plugins/docker-compose points at an executable binary."""
    cfg_dir = (
        docker_config_dir
        if docker_config_dir is not None
        else load_bench_config().docker_config_dir
    )
    reject_slash_tmp(cfg_dir, label="docker_config_dir")
    binary = compose_binary
    if binary is None:
        which_compose = shutil.which("docker-compose")
        if which_compose:
            binary = Path(which_compose)
        else:
            candidate = Path(sys.executable).resolve().parent / "docker-compose"
            if candidate.is_file():
                binary = candidate

    if binary is None or not binary.is_file():
        return None

    plugin_dir = cfg_dir / "cli-plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_link = plugin_dir / "docker-compose"
    if plugin_link.is_symlink() or plugin_link.exists():
        try:
            if plugin_link.resolve() == binary.resolve():
                return cfg_dir
        except OSError:
            pass
        plugin_link.unlink(missing_ok=True)
    plugin_link.symlink_to(binary)
    return cfg_dir


def ensure_podman_socket(
    socket_path: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    runner: CommandRunner | None = None,
) -> bool:
    """Ensure rootless podman.sock exists and is listening."""
    sock = socket_path if socket_path is not None else default_podman_socket()
    try:
        if stat.S_ISSOCK(sock.stat().st_mode):
            return True
    except OSError:
        pass

    exec_env = dict(os.environ if env is None else env)
    run_fn = _default_runner if runner is None else runner

    # Try systemd user socket activation first.
    run_fn(["systemctl", "--user", "start", "podman.socket"], exec_env)
    try:
        if stat.S_ISSOCK(sock.stat().st_mode):
            return True
    except OSError:
        pass

    # Fallback: start detached rootless podman system service with no idle timeout.
    if runner is None:
        sock.parent.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            ["podman", "system", "service", "--time=0", f"unix://{sock}"],
            env=exec_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(25):
            try:
                if stat.S_ISSOCK(sock.stat().st_mode):
                    return True
            except OSError:
                pass
            time.sleep(0.2)

    return False


def build_podman_env(
    base_env: Mapping[str, str] | None = None,
    *,
    uid: int | None = None,
    test_root: Path | None = None,
    data_root: Path | None = None,
    container_tmpdir: Path | None = None,
    cfg: BenchConfig | None = None,
) -> dict[str, str]:
    """Construct environment dictionary wired to rootless podman.sock and configured work_dir."""
    env = dict(os.environ if base_env is None else base_env)
    resolved_cfg = cfg
    if resolved_cfg is None and test_root is None:
        resolved_cfg = load_bench_config(uid=uid)

    eff_work = test_root if test_root is not None else resolved_cfg.work_dir  # type: ignore[union-attr]
    _ = data_root  # retained for call-site backward compatibility

    if uid is not None or resolved_cfg is None:
        sock = default_podman_socket(uid=uid)
        env.setdefault("DOCKER_HOST", f"unix://{sock}")
    else:
        env.setdefault("DOCKER_HOST", resolved_cfg.docker_host)

    tmpdir = eff_work / "tmp"
    mpl_dir = eff_work / "mplconfig"
    if container_tmpdir is not None:
        podman_tmp = container_tmpdir
    elif test_root is not None or resolved_cfg is None:
        podman_tmp = eff_work / "podman_tmp"
    else:
        podman_tmp = resolved_cfg.container_tmpdir

    for directory in (tmpdir, mpl_dir, podman_tmp):
        if not _is_under_slash_tmp(directory):
            directory.mkdir(parents=True, exist_ok=True)

    if _is_under_slash_tmp(Path(env.get("TMPDIR", "/tmp"))):
        env["TMPDIR"] = str(tmpdir)
    env.setdefault("CONTAINER_TMPDIR", str(podman_tmp))
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))

    casa_cfg = eff_work / "casaconfig_local.py"
    if casa_cfg.is_file():
        env.setdefault("CASASITECONFIG", str(casa_cfg))

    if not _is_under_slash_tmp(eff_work):
        cfg_dir = ensure_docker_compose_plugin(eff_work / "docker_config")
        if cfg_dir is not None:
            env.setdefault("DOCKER_CONFIG", str(cfg_dir))

    pixi_bin = str(Path(sys.executable).resolve().parent)
    current_path = env.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if pixi_bin not in path_entries:
        env["PATH"] = os.pathsep.join([pixi_bin, *path_entries])

    return env


def _default_runner(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))


def check_podman_env(
    *,
    uid: int | None = None,
    socket_path: Path | None = None,
    ensure_socket: bool = True,
    test_root: Path | None = None,
    data_root: Path | None = None,
    runs_root: Path | None = None,
    cfg: BenchConfig | None = None,
    base_env: Mapping[str, str] | None = None,
    runner: CommandRunner = _default_runner,
) -> PodmanEnvReport:
    """Verify rootless podman.sock, Compose v2, Harbor CLI, and configured non-/tmp storage."""
    resolved_uid = os.getuid() if uid is None else uid
    resolved_cfg = cfg
    if resolved_cfg is None and (test_root is None or data_root is None or runs_root is None):
        resolved_cfg = load_bench_config(uid=resolved_uid)

    eff_work = test_root if test_root is not None else resolved_cfg.work_dir  # type: ignore[union-attr]
    eff_data = data_root if data_root is not None else resolved_cfg.data_dir  # type: ignore[union-attr]
    eff_runs = runs_root if runs_root is not None else resolved_cfg.runs_dir  # type: ignore[union-attr]

    if socket_path is not None:
        sock = socket_path
    elif resolved_cfg is not None and resolved_cfg.docker_host.startswith("unix://"):
        sock = Path(resolved_cfg.docker_host[len("unix://") :])
    else:
        sock = default_podman_socket(resolved_uid)

    env = build_podman_env(
        base_env,
        uid=resolved_uid,
        test_root=eff_work,
        data_root=eff_data,
        cfg=resolved_cfg if test_root is None else None,
    )
    if socket_path is not None:
        env["DOCKER_HOST"] = f"unix://{sock}"

    errors: list[str] = []

    storage_paths_ok = True
    for label, root in (
        ("work_dir", eff_work),
        ("data_dir", eff_data),
        ("runs_dir", eff_runs),
    ):
        if _is_under_slash_tmp(root):
            storage_paths_ok = False
            errors.append(f"{label} resolves under /tmp: {root}")
            continue
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            storage_paths_ok = False
            errors.append(f"{label} not writable ({root}): {exc}")

    tmpdir = env.get("TMPDIR", "")
    tmpdir_not_slash_tmp = bool(tmpdir) and not _is_under_slash_tmp(Path(tmpdir))
    if not tmpdir_not_slash_tmp:
        errors.append(f"TMPDIR must not use /tmp, got: {tmpdir!r}")

    if ensure_socket:
        ensure_podman_socket(
            sock,
            env=env,
            runner=None if runner is _default_runner else runner,
        )

    socket_exists = False
    socket_is_unix_socket = False
    try:
        st = sock.stat()
        socket_exists = True
        socket_is_unix_socket = stat.S_ISSOCK(st.st_mode)
        if not socket_is_unix_socket:
            errors.append(f"Podman socket path is not a Unix socket: {sock}")
    except OSError as exc:
        errors.append(f"Podman socket unavailable at {sock}: {exc}")

    docker_host = env.get("DOCKER_HOST", "")
    expected_prefix = "unix://"
    if not docker_host.startswith(expected_prefix):
        errors.append(f"DOCKER_HOST must start with {expected_prefix!r}, got {docker_host!r}")

    socket_api_ok = False
    if socket_is_unix_socket:
        proc_api = runner(["docker", "version", "--format", "{{.Server.Version}}"], env)
        if proc_api.returncode == 0 and proc_api.stdout.strip():
            socket_api_ok = True
        else:
            errors.append(
                f"Docker API check against {docker_host} failed (rc={proc_api.returncode}): "
                f"{proc_api.stderr.strip()}"
            )

    podman_version: str | None = None
    podman_rootless: bool | None = None
    proc_info = runner(
        ["podman", "info", "--format", "{{.Version.Version}}|{{.Host.Security.Rootless}}"],
        env,
    )
    if proc_info.returncode == 0:
        parts = proc_info.stdout.strip().split("|")
        podman_version = parts[0].strip() if parts else None
        if len(parts) > 1:
            podman_rootless = parts[1].strip().lower() == "true"
            if not podman_rootless:
                errors.append("Podman is not running in rootless mode.")
    else:
        errors.append(f"podman info failed (rc={proc_info.returncode}): {proc_info.stderr.strip()}")

    compose_cmd: list[str] | None = None
    compose_version: str | None = None
    for candidate in (["docker", "compose"], ["podman", "compose"]):
        proc_comp = runner([*candidate, "version"], env)
        if proc_comp.returncode == 0:
            compose_cmd = candidate
            compose_version = (
                proc_comp.stdout.strip().splitlines()[0] if proc_comp.stdout.strip() else "ok"
            )
            break
    if compose_cmd is None:
        errors.append("Neither 'docker compose version' nor 'podman compose version' succeeded.")

    harbor_version: str | None = None
    try:
        harbor_version = importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError:
        errors.append("Python package 'harbor' is not installed.")

    ok = (
        storage_paths_ok
        and tmpdir_not_slash_tmp
        and socket_exists
        and socket_is_unix_socket
        and socket_api_ok
        and podman_version is not None
        and podman_rootless is True
        and compose_cmd is not None
        and harbor_version is not None
        and not errors
    )

    return PodmanEnvReport(
        uid=resolved_uid,
        socket_path=str(sock),
        socket_exists=socket_exists,
        socket_is_unix_socket=socket_is_unix_socket,
        socket_api_ok=socket_api_ok,
        docker_host=docker_host,
        podman_version=podman_version,
        podman_rootless=podman_rootless,
        compose_cmd=compose_cmd,
        compose_version=compose_version,
        harbor_version=harbor_version,
        storage_paths_ok=storage_paths_ok,
        tmpdir=tmpdir,
        tmpdir_not_slash_tmp=tmpdir_not_slash_tmp,
        ok=ok,
        errors=errors,
    )


HARBOR_CONTAINER_TOKENS = ("__env-", "__verifier__trial-")
HARBOR_VOLUME_SUFFIXES = ("_workspace_vol", "_agent_tmp", "_verifier_ws", "_verifier_tmp")
HARBOR_NETWORK_TOKENS = ("__env_", "__verifier__trial_")


def default_aardvark_dns_dir(uid: int | None = None) -> Path:
    resolved_uid = os.getuid() if uid is None else uid
    return Path(f"/run/user/{resolved_uid}/containers/networks/aardvark-dns")


def cleanup_stale_harbor_resources(
    *,
    include_running: bool = False,
    base_env: Mapping[str, str] | None = None,
    runner: CommandRunner = _default_runner,
    aardvark_dir: Path | None = None,
    cfg: BenchConfig | None = None,
) -> dict[str, object]:
    """Remove leaked Harbor trial containers, networks, named volumes, and dead aardvark pids."""
    env = build_podman_env(base_env, cfg=cfg)
    removed_containers: list[str] = []
    removed_volumes: list[str] = []
    removed_networks: list[str] = []
    removed_stale_pids: list[str] = []

    proc_ps = runner(["podman", "ps", "-a", "--format", "json"], env)
    if proc_ps.returncode == 0 and proc_ps.stdout.strip():
        try:
            items = json.loads(proc_ps.stdout)
        except json.JSONDecodeError:
            items = []
        for item in items:
            names = item.get("Names") or []
            if isinstance(names, str):
                names = [names]
            state = str(item.get("State", "")).lower()
            if not any(tok in n for n in names for tok in HARBOR_CONTAINER_TOKENS):
                continue
            if not include_running and state == "running":
                continue
            cid = str(item.get("Id", ""))
            nets = item.get("Networks") or []
            for net in nets:
                runner(["podman", "network", "disconnect", "-f", str(net), cid], env)
            rm_proc = runner(["podman", "rm", "-f", "--ignore", cid], env)
            if rm_proc.returncode == 0:
                removed_containers.append(names[0] if names else cid[:12])

    proc_vol = runner(["podman", "volume", "ls", "--format", "{{.Name}}"], env)
    if proc_vol.returncode == 0:
        for vname in proc_vol.stdout.splitlines():
            vname = vname.strip()
            if vname.endswith(HARBOR_VOLUME_SUFFIXES):
                rv = runner(["podman", "volume", "rm", "-f", vname], env)
                if rv.returncode == 0:
                    removed_volumes.append(vname)

    proc_net = runner(["podman", "network", "ls", "--format", "{{.Name}}"], env)
    if proc_net.returncode == 0:
        for nname in proc_net.stdout.splitlines():
            nname = nname.strip()
            if any(tok in nname for tok in HARBOR_NETWORK_TOKENS):
                rn = runner(["podman", "network", "rm", "-f", nname], env)
                if rn.returncode == 0:
                    removed_networks.append(nname)

    dns_dir = aardvark_dir if aardvark_dir is not None else default_aardvark_dns_dir()
    try:
        dns_exists = dns_dir.is_dir()
    except OSError:
        dns_exists = False
    if dns_exists:
        try:
            pid_files = sorted(dns_dir.glob("*.pid"))
        except OSError:
            pid_files = []
        for pid_file in pid_files:
            try:
                raw_pid = pid_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid <= 0:
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                try:
                    pid_file.unlink()
                    removed_stale_pids.append(str(pid_file))
                except (PermissionError, OSError):
                    pass
            except (PermissionError, OSError):
                pass

    return {
        "removed_containers": removed_containers,
        "removed_volumes": removed_volumes,
        "removed_networks": removed_networks,
        "removed_stale_pids": removed_stale_pids,
    }


def smoke_compose_teardown(
    *,
    base_env: Mapping[str, str] | None = None,
    runner: CommandRunner = _default_runner,
    scratch_root: Path | None = None,
    pack_id: str | None = None,
    cfg: BenchConfig | None = None,
) -> dict[str, object]:
    """Smoke-test Compose up/down lifecycle for offline verifier and loopback MCP sidecar."""
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    tmp_base = scratch_root if scratch_root is not None else resolved_cfg.tmp_dir
    reject_slash_tmp(tmp_base, label="scratch_root")
    pack = resolved_cfg.get_pack(pack_id)
    env = build_podman_env(base_env, cfg=resolved_cfg)
    errors: list[str] = []

    compose_cmd: list[str] | None = None
    for candidate in (["docker", "compose"], ["podman", "compose"]):
        proc_comp = runner([*candidate, "version"], env)
        if proc_comp.returncode == 0:
            compose_cmd = candidate
            break
    if compose_cmd is None:
        return {
            "ok": False,
            "verifier": {"ok": False},
            "tools": {"ok": False},
            "leftover_volumes": [],
            "errors": ["Neither 'docker compose version' nor 'podman compose version' succeeded."],
        }

    tmp_base.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    proj_ver = f"rb_smoke_ver_{token}"
    proj_tools = f"rb_smoke_tools_{token}"

    ver_dir = Path(tempfile.mkdtemp(prefix=f"{proj_ver}_", dir=str(tmp_base)))
    tools_dir = Path(tempfile.mkdtemp(prefix=f"{proj_tools}_", dir=str(tmp_base)))

    ver_result: dict[str, object] = {"project": proj_ver, "ok": False}
    tools_result: dict[str, object] = {"project": proj_tools, "ok": False}
    leftover_volumes: list[str] = []

    ports_tuple = tuple(int(p) for p in pack.mcp_ports.values())

    try:
        # (a) Verifier compose project (network_mode: none, verifier_ws:/workspace)
        ver_file = ver_dir / "docker-compose.yaml"
        ver_spec = {
            "services": {
                "main": {
                    "image": "localhost/radio-bench-verifier:latest",
                    "network_mode": "none",
                    "command": ["sleep", "infinity"],
                    "volumes": ["verifier_ws:/workspace"],
                }
            },
            "volumes": {"verifier_ws": {}},
        }
        ver_file.write_text(yaml.safe_dump(ver_spec, sort_keys=False), encoding="utf-8")

        ver_up = runner(
            [*compose_cmd, "--project-name", proj_ver, "-f", str(ver_file), "up", "-d"],
            env,
        )
        ver_down = runner(
            [
                *compose_cmd,
                "--project-name",
                proj_ver,
                "-f",
                str(ver_file),
                "down",
                "--rmi",
                "local",
                "--volumes",
                "--remove-orphans",
            ],
            env,
        )
        ver_ok = ver_up.returncode == 0 and ver_down.returncode == 0
        ver_result = {
            "project": proj_ver,
            "up_rc": ver_up.returncode,
            "down_rc": ver_down.returncode,
            "ok": ver_ok,
        }
        if ver_up.returncode != 0:
            errors.append(
                f"verifier compose up failed (rc={ver_up.returncode}): {ver_up.stderr.strip()}"
            )
        if ver_down.returncode != 0:
            errors.append(
                f"verifier compose down failed (rc={ver_down.returncode}): {ver_down.stderr.strip()}"
            )

        # (b) Tools compose project (pasta + extra_hosts + service:main + RADIO_MCP_HOST=127.0.0.1)
        tools_file = tools_dir / "docker-compose.yaml"
        main_cmd = "\n".join(
            [
                "set -euo pipefail",
                "mkdir -p /var/tmp/mplconfig",
                "chmod 1777 /var/tmp /var/tmp/mplconfig",
                "chmod -R u+rwX,go+rX /workspace",
                "chown -R 1000:1000 /workspace",
                "touch /workspace/.ws_owned",
                "chown 1000:1000 /workspace/.ws_owned",
                "exec sleep infinity",
            ]
        )
        sidecar_cmd = "\n".join(
            [
                "set -euo pipefail",
                "until [ -w /workspace ] && [ -f /workspace/.ws_owned ]; do",
                "  sleep 0.2",
                "done",
                "exec /opt/casa-venv/bin/python /usr/local/bin/serve-mcp-sidecar.py",
            ]
        )
        sidecar_env = {
            "TMPDIR": "/var/tmp",
            "RADIO_MCP_HOST": "127.0.0.1",
            "RADIO_MCP_PORT_MS_INSPECT": str(pack.mcp_ports.get("ms-inspect", 8000)),
            "RADIO_MCP_PORT_MS_MODIFY": str(pack.mcp_ports.get("ms-modify", 8001)),
            "RADIO_MCP_PORT_MS_CREATE": str(pack.mcp_ports.get("ms-create", 8002)),
        }
        tools_spec = {
            "services": {
                "main": {
                    "image": "localhost/radio-bench-agent-base:latest",
                    "network_mode": "pasta",
                    "extra_hosts": ["mcp-servers:127.0.0.1"],
                    "environment": {"TMPDIR": "/var/tmp"},
                    "command": ["bash", "-lc", main_cmd],
                    "volumes": ["workspace_vol:/workspace", "agent_tmp:/var/tmp"],
                },
                "mcp-servers": {
                    "image": pack.sidecar_image,
                    "network_mode": "service:main",
                    "environment": sidecar_env,
                    "command": ["bash", "-lc", sidecar_cmd],
                    "volumes": ["workspace_vol:/workspace", "agent_tmp:/var/tmp"],
                },
            },
            "volumes": {
                "workspace_vol": {},
                "agent_tmp": {},
            },
        }
        tools_file.write_text(yaml.safe_dump(tools_spec, sort_keys=False), encoding="utf-8")

        tools_up = runner(
            [*compose_cmd, "--project-name", proj_tools, "-f", str(tools_file), "up", "-d"],
            env,
        )
        tcp_rc = 1
        main_cid = ""
        if tools_up.returncode == 0:
            ps_proc = runner(
                [
                    *compose_cmd,
                    "--project-name",
                    proj_tools,
                    "-f",
                    str(tools_file),
                    "ps",
                    "-q",
                    "main",
                ],
                env,
            )
            if ps_proc.returncode == 0 and ps_proc.stdout.strip():
                main_cid = ps_proc.stdout.strip().splitlines()[0].strip()
            else:
                main_cid = f"{proj_tools}-main-1"

            probe_snippet = (
                "import socket, sys, time\n"
                f"for port in {ports_tuple!r}:\n"
                "    for _ in range(150):\n"
                "        try:\n"
                "            with socket.create_connection(('mcp-servers', port), timeout=1.0):\n"
                "                break\n"
                "        except OSError:\n"
                "            time.sleep(0.2)\n"
                "    else:\n"
                "        sys.exit(f'Timed out waiting for mcp-servers:{port}')\n"
            )
            exec_proc = runner(
                [
                    "podman",
                    "exec",
                    main_cid,
                    "/opt/casa-venv/bin/python",
                    "-c",
                    probe_snippet,
                ],
                env,
            )
            tcp_rc = exec_proc.returncode
            if tcp_rc != 0:
                errors.append(f"tools TCP probe failed (rc={tcp_rc}): {exec_proc.stderr.strip()}")
        else:
            errors.append(
                f"tools compose up failed (rc={tools_up.returncode}): {tools_up.stderr.strip()}"
            )

        tools_down = runner(
            [
                *compose_cmd,
                "--project-name",
                proj_tools,
                "-f",
                str(tools_file),
                "down",
                "--rmi",
                "local",
                "--volumes",
                "--remove-orphans",
            ],
            env,
        )
        if tools_down.returncode != 0:
            errors.append(
                f"tools compose down failed (rc={tools_down.returncode}): {tools_down.stderr.strip()}"
            )

        vol_ls = runner(["podman", "volume", "ls", "--format", "{{.Name}}"], env)
        if vol_ls.returncode == 0:
            prefixes = (f"{proj_ver}_", f"{proj_tools}_")
            leftover_volumes = [
                v.strip() for v in vol_ls.stdout.splitlines() if v.strip().startswith(prefixes)
            ]
            if leftover_volumes:
                errors.append(f"leftover smoke compose volumes: {leftover_volumes}")
        else:
            errors.append(
                f"podman volume ls failed (rc={vol_ls.returncode}): {vol_ls.stderr.strip()}"
            )

        tools_ok = (
            tools_up.returncode == 0
            and tcp_rc == 0
            and tools_down.returncode == 0
            and vol_ls.returncode == 0
            and not leftover_volumes
        )
        tools_result = {
            "project": proj_tools,
            "pack_id": pack.pack_id,
            "sidecar_image": pack.sidecar_image,
            "up_rc": tools_up.returncode,
            "tcp_probe_rc": tcp_rc,
            "down_rc": tools_down.returncode,
            "ok": tools_ok,
        }
    finally:
        shutil.rmtree(ver_dir, ignore_errors=True)
        shutil.rmtree(tools_dir, ignore_errors=True)

    overall_ok = bool(ver_result.get("ok")) and bool(tools_result.get("ok")) and not errors
    return {
        "ok": overall_ok,
        "verifier": ver_result,
        "tools": tools_result,
        "leftover_volumes": leftover_volumes,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify rootless Podman + Compose + Harbor environment."
    )
    add_config_cli_args(parser)
    parser.add_argument("--pack", type=str, default=None, help="Pack ID for smoke-compose.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report only.")
    parser.add_argument(
        "--no-ensure-socket",
        action="store_true",
        help="Do not attempt to start podman.socket if absent.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove leaked Harbor containers, networks, and named volumes (including orphaned running trials).",
    )
    parser.add_argument(
        "--smoke-compose",
        action="store_true",
        help="Run Compose up/down smoke tests for verifier (network_mode=none) and tools (pasta + service:main).",
    )
    args = parser.parse_args(argv)

    cfg = config_from_cli_args(args)

    if args.cleanup:
        cleanup_summary = cleanup_stale_harbor_resources(include_running=True, cfg=cfg)
    else:
        cleanup_summary = cleanup_stale_harbor_resources(include_running=False, cfg=cfg)

    report = check_podman_env(ensure_socket=not args.no_ensure_socket, cfg=cfg)
    out_dict = report.to_dict()
    if cleanup_summary is not None:
        out_dict["cleanup"] = cleanup_summary

    smoke_summary: dict[str, object] | None = None
    if args.smoke_compose:
        smoke_summary = smoke_compose_teardown(pack_id=args.pack, cfg=cfg)
        out_dict["smoke_compose"] = smoke_summary

    overall_ok = report.ok and (smoke_summary is None or bool(smoke_summary.get("ok")))
    payload = json.dumps(out_dict, indent=2)
    if args.json:
        print(payload)
    else:
        status = "PASS" if overall_ok else "FAIL"
        print(f"[podman_env] {status}")
        print(payload)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
