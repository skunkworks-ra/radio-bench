"""Rootless Podman socket, Compose v2, Harbor CLI, and fast-storage readiness checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from radio_bench import DATA_ROOT, RUNS_ROOT, TEST_ROOT

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
    raw = str(path)
    if raw == "/tmp" or raw.startswith("/tmp/"):
        return True
    try:
        resolved = str(path.resolve())
    except OSError:
        return False
    return resolved == "/tmp" or resolved.startswith("/tmp/")


def ensure_docker_compose_plugin(
    docker_config_dir: Path | None = None,
    compose_binary: Path | None = None,
) -> Path | None:
    """Ensure DOCKER_CONFIG/cli-plugins/docker-compose points at an executable binary."""
    cfg_dir = docker_config_dir or (TEST_ROOT / "docker_config")
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
    test_root: Path = TEST_ROOT,
    data_root: Path = DATA_ROOT,
) -> dict[str, str]:
    """Construct environment dictionary wired to rootless podman.sock and /var/mnt/fast."""
    env = dict(os.environ if base_env is None else base_env)
    sock = default_podman_socket(uid=uid)
    env.setdefault("DOCKER_HOST", f"unix://{sock}")

    tmpdir = test_root / "tmp"
    mpl_dir = test_root / "mplconfig"
    podman_tmp = data_root / "podman_tmp"
    for directory in (tmpdir, mpl_dir, podman_tmp):
        directory.mkdir(parents=True, exist_ok=True)

    if _is_under_slash_tmp(Path(env.get("TMPDIR", "/tmp"))):
        env["TMPDIR"] = str(tmpdir)
    env.setdefault("CONTAINER_TMPDIR", str(podman_tmp))
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))

    casa_cfg = test_root / "casaconfig_local.py"
    if casa_cfg.is_file():
        env.setdefault("CASASITECONFIG", str(casa_cfg))

    cfg_dir = ensure_docker_compose_plugin(test_root / "docker_config")
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
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))


def check_podman_env(
    *,
    uid: int | None = None,
    socket_path: Path | None = None,
    ensure_socket: bool = True,
    test_root: Path = TEST_ROOT,
    data_root: Path = DATA_ROOT,
    runs_root: Path = RUNS_ROOT,
    base_env: Mapping[str, str] | None = None,
    runner: CommandRunner = _default_runner,
) -> PodmanEnvReport:
    """Verify rootless podman.sock, Compose v2, Harbor CLI, and /var/mnt/fast storage."""
    resolved_uid = os.getuid() if uid is None else uid
    sock = socket_path if socket_path is not None else default_podman_socket(resolved_uid)
    env = build_podman_env(base_env, uid=resolved_uid, test_root=test_root, data_root=data_root)
    if socket_path is not None:
        env["DOCKER_HOST"] = f"unix://{sock}"

    errors: list[str] = []

    storage_paths_ok = True
    for label, root in (
        ("test_root", test_root),
        ("data_root", data_root),
        ("runs_root", runs_root),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify rootless Podman + Compose + Harbor environment."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report only.")
    parser.add_argument(
        "--no-ensure-socket",
        action="store_true",
        help="Do not attempt to start podman.socket if absent.",
    )
    args = parser.parse_args(argv)

    report = check_podman_env(ensure_socket=not args.no_ensure_socket)
    payload = json.dumps(report.to_dict(), indent=2)
    if args.json:
        print(payload)
    else:
        status = "PASS" if report.ok else "FAIL"
        print(f"[podman_env] {status}")
        print(payload)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
