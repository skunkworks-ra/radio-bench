#!/usr/bin/env python3
"""Supervisor launching ms-inspect (8000), ms-modify (8001), and ms-create (8002) over HTTP."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

SERVERS: list[tuple[str, str]] = [
    ("ms-inspect", "8000"),
    ("ms-modify", "8001"),
    ("ms-create", "8002"),
]


def main() -> int:
    mcp_dir = os.environ.get("RADIO_MCP_DIR", "/workspace/.radio_mcp")
    os.makedirs(mcp_dir, exist_ok=True)
    with contextlib.suppress(PermissionError):
        os.chmod(mcp_dir, 0o775)
    host = os.environ.get("RADIO_MCP_HOST", "0.0.0.0")

    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    stopping = False

    def _terminate_all(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _terminate_all)
    signal.signal(signal.SIGINT, _terminate_all)

    try:
        for binary, port in SERVERS:
            env = dict(os.environ)
            env["RADIO_MCP_TRANSPORT"] = "http"
            env["RADIO_MCP_HOST"] = host
            env["RADIO_MCP_PORT"] = port
            env["RADIO_MCP_DIR"] = mcp_dir
            proc = subprocess.Popen(
                ["/opt/casa-venv/bin/" + binary],
                env=env,
                start_new_session=True,
            )
            procs.append((binary, proc))
            print(f"[mcp-sidecar] started {binary} on {host}:{port} (pid={proc.pid})", flush=True)

        while not stopping:
            for binary, proc in procs:
                rc = proc.poll()
                if rc is not None:
                    print(f"[mcp-sidecar] {binary} exited with rc={rc}; shutting down", flush=True)
                    return rc if rc > 0 else (128 + abs(rc) if rc < 0 else 1)
            time.sleep(0.5)
    finally:
        for _, proc in procs:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        for _, proc in procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
