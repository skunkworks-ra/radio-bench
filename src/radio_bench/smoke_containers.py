"""Automated hard-isolation and streamable-http MCP tool smoke checks for container images."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from radio_bench.config import (
    BenchConfig,
    add_config_cli_args,
    config_from_cli_args,
    load_bench_config,
    materialize_pack_source,
    reject_slash_tmp,
    write_casaconfig_local,
)
from radio_bench.podman_env import build_podman_env

CASA_BASE_IMAGE = "localhost/radio-bench-casa-base:latest"
AGENT_IMAGE = "localhost/radio-bench-agent-base:latest"
SIDECAR_IMAGE = "localhost/radio-bench-mcp-sidecar:latest"
VERIFIER_IMAGE = "localhost/radio-bench-verifier:latest"

_HEX_SHORT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

OFFLINE_CASA_SNIPPET = (
    "import importlib.metadata as m, os, casatools, casatasks; "
    "assert os.environ.get('CASASITECONFIG') == '/etc/casa/config.py'; "
    "me = casatools.measures(); "
    "vla = me.observatory('VLA'); "
    "assert me.doframe(vla) and me.doframe(me.epoch('UTC', '2026-01-01/00:00:00')); "
    "azel = me.measure(me.direction('J2000', '13h31m08.3s', '+30d30m33s'), 'AZELGEO'); "
    "me.done(); "
    "cfg = casatools.ctsys.resolve('alma/simmos/vla.d.cfg'); "
    "eph_jpl = casatools.ctsys.resolve('ephemerides/JPL-Horizons'); "
    "eph_de = casatools.ctsys.resolve('ephemerides/DE405'); "
    "assert os.path.isfile(cfg), cfg; "
    "assert os.path.isdir(eph_jpl), eph_jpl; "
    "assert os.path.isdir(eph_de), eph_de; "
    "assert os.path.isdir(os.path.expanduser('~/.casa/data')); "
    "print(casatools.version_string(), m.version('casatasks'), bool(vla), bool(azel))"
)

# Baseline expected counts for default frozen pack ra_00a11ea -> (container_port, expected_tool_count).
EXPECTED_SERVER_COUNTS: dict[str, tuple[int, int]] = {
    "ms-inspect": (8000, 38),
    "ms-modify": (8001, 19),
    "ms-create": (8002, 4),
}

CORE_SUCCESS_TOOLS: frozenset[str] = frozenset(
    {
        "ms_observation_info",
        "ms_field_list",
        "ms_spectral_window_list",
        "ms_antenna_list",
        "ms_flag_summary",
    }
)


@dataclass(frozen=True)
class ContainerRunResult:
    """Result from a bounded-timeout offline rootless Podman invocation."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def classify_tool_probe_response(tool_name: str, res: dict[str, Any]) -> tuple[bool, str]:
    """Validate an MCP tools/call response from the smoke manifest."""
    is_error = bool(res.get("isError", False))
    structured = res.get("structuredContent") or {}
    content_items = res.get("content") or []
    first_text = (
        content_items[0].get("text", "")
        if content_items and isinstance(content_items[0], dict)
        else ""
    )
    error_type = str(structured.get("error_type") or "")
    if not error_type and '"error_type"' in first_text:
        with contextlib.suppress(json.JSONDecodeError):
            parsed_text = json.loads(first_text)
            if isinstance(parsed_text, dict):
                error_type = str(parsed_text.get("error_type") or "")

    if error_type == "UNEXPECTED_ERROR" or '"UNEXPECTED_ERROR"' in first_text:
        return False, f"unexpected unhandled exception inside tool: {first_text[:300]}"
    if "ImportError" in first_text or "ModuleNotFoundError" in first_text:
        return False, f"import failure inside tool: {first_text[:300]}"
    if tool_name in CORE_SUCCESS_TOOLS and is_error:
        return False, f"core smoke.ms tool failed ({error_type}): {first_text[:300]}"
    if is_error and not error_type and '"error_type"' not in first_text:
        return False, f"MCP error without tool envelope: {first_text[:300]}"
    return True, ""


def parse_mcp_http_body(content_type: str, text: str) -> dict[str, Any]:
    """Parse an MCP streamable-http response body (JSON or text/event-stream SSE)."""
    ctype = content_type.lower()
    if "text/event-stream" in ctype:
        events: list[dict[str, Any]] = []
        data_lines: list[str] = []

        def _flush(lines: list[str]) -> None:
            joined = "\n".join(lines).strip()
            if not joined:
                return
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(joined)
                if isinstance(parsed, dict):
                    events.append(parsed)

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\r")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line == "" and data_lines:
                _flush(data_lines)
                data_lines = []
        if data_lines:
            _flush(data_lines)

        for ev in reversed(events):
            if "result" in ev or "error" in ev:
                return ev
        if events:
            return events[-1]
        raise ValueError(f"No JSON-RPC event found in SSE payload: {text[:300]!r}")

    parsed_direct = json.loads(text)
    if not isinstance(parsed_direct, dict):
        raise ValueError(f"Expected JSON object from MCP server, got: {type(parsed_direct)}")
    return parsed_direct


class StreamableHTTPMCPClient:
    """Minimal synchronous JSON-RPC 2.0 client for FastMCP streamable-http servers."""

    def __init__(
        self,
        url: str,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url
        self._client = httpx.Client(timeout=timeout_s, follow_redirects=True, transport=transport)
        self.session_id: str | None = None
        self._next_id = 1

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        resp = self._client.post(self.url, headers=self._headers(), json=payload)
        resp.raise_for_status()
        sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        body = parse_mcp_http_body(resp.headers.get("content-type", "application/json"), resp.text)
        if "error" in body:
            raise RuntimeError(f"JSON-RPC error on {method}: {body['error']}")
        result = body.get("result", {})
        return result if isinstance(result, dict) else {"value": result}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        resp = self._client.post(self.url, headers=self._headers(), json=payload)
        if resp.status_code >= 400:
            resp.raise_for_status()

    def initialize(self) -> dict[str, Any]:
        res = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "radio-bench-smoke", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})
        return res

    def list_tools(self) -> list[str]:
        res = self.request("tools/list", {})
        tools = res.get("tools", [])
        return [str(t["name"]) for t in tools if isinstance(t, dict) and "name" in t]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})


def generate_minimal_smoke_ms(ms_path: Path, *, cfg: BenchConfig | None = None) -> None:
    """Create a 3-antenna VLA-D smoke MS in configured work_dir for MCP HTTP probing."""
    reject_slash_tmp(ms_path, label="ms_path")
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    casa_cfg = write_casaconfig_local(resolved_cfg)
    os.environ.setdefault("CASASITECONFIG", str(casa_cfg))
    from casatools import componentlist, ctsys, measures, simulator, table

    if ms_path.exists():
        shutil.rmtree(ms_path)
    cl_path = ms_path.parent / (ms_path.name + ".cl")
    if cl_path.exists():
        shutil.rmtree(cl_path)

    ms_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_file = Path(ctsys.resolve("alma/simmos")) / "vla.d.cfg"
    x, y, z, diam, antnames = [], [], [], [], []
    for raw in cfg_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        x.append(float(parts[0]))
        y.append(float(parts[1]))
        z.append(float(parts[2]))
        diam.append(float(parts[3]))
        antnames.append(parts[4] if len(parts) > 4 else f"ea{len(x):02d}")
        if len(x) == 3:
            break

    try:
        me = measures()
        try:
            pos = me.observatory("VLA")
            direction = me.direction("J2000", "13h31m08.3s", "+30d30m33s")
            ref_epoch = me.epoch("UTC", "2026-01-01/00:00:00")
        finally:
            me.done()

        sm = simulator()
        try:
            sm.open(ms=str(ms_path))
            sm.setconfig(
                telescopename="VLA",
                x=x,
                y=y,
                z=z,
                dishdiameter=diam,
                mount=["ALT-AZ"] * len(x),
                antname=antnames,
                coordsystem="global",
                referencelocation=pos,
            )
            sm.setspwindow(
                spwname="spw0",
                freq="1.4GHz",
                deltafreq="1MHz",
                freqresolution="1MHz",
                nchannels=8,
                stokes="RR RL LR LL",
            )
            sm.setfeed(mode="perfect R L", pol=[""])
            sm.setfield(sourcename="smoke_target", sourcedirection=direction)
            sm.setlimits(shadowlimit=0.001, elevationlimit="8deg")
            sm.setauto(autocorrwt=0.0)
            sm.settimes(
                integrationtime="10s",
                usehourangle=True,
                referencetime=ref_epoch,
            )
            sm.observe("smoke_target", "spw0", starttime="-1min", stoptime="+1min")
        finally:
            sm.close()
            sm.done()

        cl = componentlist()
        try:
            cl.done()
            cl.addcomponent(
                dir=direction,
                flux=1.0,
                fluxunit="Jy",
                freq="1.4GHz",
                shape="point",
            )
            cl.rename(str(cl_path))
        finally:
            cl.close()
            cl.done()

        sm2 = simulator()
        try:
            sm2.openfromms(str(ms_path))
            sm2.setvp()
            sm2.predict(complist=str(cl_path))
        finally:
            sm2.close()
            sm2.done()

        tb = table()
        try:
            tb.open(str(ms_path), nomodify=False)
            tb.putcol("DATA", tb.getcol("MODEL_DATA"))
        finally:
            tb.close()
            tb.done()
    finally:
        shutil.rmtree(cl_path, ignore_errors=True)


def load_pinned_tool_manifest(
    sha: str | None = None,
    *,
    pack_id: str | None = None,
    cfg: BenchConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Read tool_manifest.json from configured pack source without touching its working tree."""
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    pack = resolved_cfg.get_pack(pack_id)
    repo_path, full_sha, _ = materialize_pack_source(
        pack, resolved_cfg.work_dir, repo_root=resolved_cfg.repo_root
    )
    target_ref = sha if sha is not None else full_sha
    proc = subprocess.run(
        ["git", "-C", str(repo_path), "show", f"{target_ref}:scripts/ci/tool_manifest.json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    return payload["tools"]


def reap_stale_smoke_containers(
    env: dict[str, str],
    patterns: tuple[str, ...] = ("^radio-bench-sidecar-smoke-", "^radio-bench-offline-smoke-"),
) -> None:
    """Force-remove any leftover smoke containers from prior interrupted runs."""
    for pattern in patterns:
        proc_ps = subprocess.run(
            ["podman", "ps", "-aq", "--filter", f"name={pattern}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        for cid in proc_ps.stdout.split():
            subprocess.run(
                ["podman", "rm", "-f", "-t", "0", cid],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )


def run_offline_container(
    image: str,
    cmd_args: list[str],
    env: dict[str, str],
    *,
    timeout_s: float = 60.0,
    extra_podman_args: list[str] | None = None,
    label: str = "check",
) -> ContainerRunResult:
    """Run a named `--network=none` container with timeout handling and guaranteed `rm -f -t 0`."""
    cname = f"radio-bench-offline-smoke-{label}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    podman_cmd = [
        "podman",
        "run",
        "--rm",
        "--name",
        cname,
        "--network=none",
        *(extra_podman_args or []),
        image,
        *cmd_args,
    ]
    try:
        proc = subprocess.run(
            podman_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return ContainerRunResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        err = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return ContainerRunResult(
            returncode=124,
            stdout=out,
            stderr=f"{err}\nTimed out after {timeout_s}s".strip(),
            timed_out=True,
        )
    finally:
        subprocess.run(
            ["podman", "rm", "-f", "-t", "0", cname],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


def check_casa_base_offline(env: dict[str, str]) -> dict[str, Any]:
    """Verify CASA 6.7.6.14 + measures/ephemeris tables run offline in casa-base."""
    proc = run_offline_container(
        CASA_BASE_IMAGE,
        ["python", "-c", OFFLINE_CASA_SNIPPET],
        env,
        timeout_s=60.0,
        label="casa-base",
    )
    ok = proc.returncode == 0 and "6.7.6-14 6.7.6.14 True True" in proc.stdout
    return {
        "ok": ok,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[-500:],
    }


def check_agent_base_offline_and_isolation(env: dict[str, str]) -> dict[str, Any]:
    """Verify CASA 6.7.6.14 runs offline as analyst (1000:1000) in agent-base and forbidden imports fail."""
    proc_casa = run_offline_container(
        AGENT_IMAGE,
        ["python", "-c", OFFLINE_CASA_SNIPPET],
        env,
        timeout_s=60.0,
        extra_podman_args=["--user", "1000:1000"],
        label="agent-casa",
    )
    casa_ok = proc_casa.returncode == 0 and "6.7.6-14 6.7.6.14 True True" in proc_casa.stdout

    proc_bins = run_offline_container(
        AGENT_IMAGE,
        [
            "bash",
            "-lc",
            (
                'test "$(id -u)" = "1000" '
                '&& test "$(id -un)" = "analyst" '
                "&& test -x /usr/local/bin/run-step.sh "
                "&& test -x /opt/radio-tools/scripts/plugin/run-step.sh "
                "&& test ! -e /opt/radio-bench "
                "&& test ! -e /opt/radio-analyst "
                "&& test ! -e /tmp/opencode "
                "&& test ! -e /tmp/cc-socks "
                "&& test ! -e /opt/agent-tools/.claude.json "
                "&& python -c 'import casatools, casatasks' "
                "&& node --version "
                "&& npm --version "
                "&& /usr/local/bin/claude --version "
                "&& /usr/local/bin/opencode --version"
            ),
        ],
        env,
        timeout_s=45.0,
        extra_podman_args=["--user", "1000:1000"],
        label="agent-bins",
    )
    binaries_ok = proc_bins.returncode == 0

    forbidden = ["ms_inspect", "ms_modify", "ms_create", "radio_bench"]
    isolation_results: dict[str, dict[str, Any]] = {}
    all_isolated = True
    for mod in forbidden:
        proc_imp = run_offline_container(
            AGENT_IMAGE,
            ["python", "-c", f"import {mod}"],
            env,
            timeout_s=30.0,
            extra_podman_args=["--user", "1000:1000"],
            label=f"agent-iso-{mod.replace('_', '-')}",
        )
        blocked = proc_imp.returncode != 0 and "ModuleNotFoundError" in proc_imp.stderr
        isolation_results[mod] = {
            "returncode": proc_imp.returncode,
            "blocked_module_not_found": blocked,
        }
        if not blocked:
            all_isolated = False

    return {
        "casa_offline_ok": casa_ok,
        "casa_stdout": proc_casa.stdout.strip(),
        "casa_stderr": proc_casa.stderr.strip()[-500:],
        "user_1000_ok": casa_ok and binaries_ok,
        "binaries_ok": binaries_ok,
        "binaries_stdout": proc_bins.stdout.strip(),
        "binaries_stderr": proc_bins.stderr.strip()[-500:],
        "isolation": isolation_results,
        "all_forbidden_blocked": all_isolated,
        "ok": casa_ok and binaries_ok and all_isolated,
    }


def check_verifier_offline(env: dict[str, str]) -> dict[str, Any]:
    """Verify isolated verifier image runs CASA + measures/ephemeris + radio_bench offline."""
    code = (
        f"{OFFLINE_CASA_SNIPPET}; import radio_bench; print('verifier_ok', radio_bench.__version__)"
    )
    proc = run_offline_container(
        VERIFIER_IMAGE,
        ["python", "-c", code],
        env,
        timeout_s=60.0,
        label="verifier",
    )
    ok = (
        proc.returncode == 0
        and "6.7.6-14 6.7.6.14 True True" in proc.stdout
        and "verifier_ok" in proc.stdout
    )
    return {"ok": ok, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()[-500:]}


def _make_tree_world_writable(root: Path) -> None:
    """Ensure root and all children are writable before/without userns chown."""
    with contextlib.suppress(OSError):
        root.chmod(0o777)
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        with contextlib.suppress(OSError):
            dp.chmod(0o777)
        for dname in dirnames:
            with contextlib.suppress(OSError):
                (dp / dname).chmod(0o777)
        for fname in filenames:
            with contextlib.suppress(OSError):
                (dp / fname).chmod(0o666)


def _extract_tool_data(res: dict[str, Any]) -> dict[str, Any]:
    """Extract the unwrapped `data` payload dict from an MCP tool response."""
    content_items = res.get("content") or []
    if content_items and isinstance(content_items[0], dict):
        text = content_items[0].get("text", "")
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict):
                return parsed["data"]
    structured = res.get("structuredContent")
    if isinstance(structured, dict) and isinstance(structured.get("data"), dict):
        return structured["data"]
    return {}


def _resolve_published_port(cname: str, container_port: int, env: dict[str, str]) -> int:
    """Query the ephemeral 127.0.0.1 host port bound to `container_port` in `cname`."""
    proc = subprocess.run(
        ["podman", "port", cname, f"{container_port}/tcp"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"Failed to resolve published port {container_port}/tcp for {cname}: {proc.stderr.strip()}"
        )
    first_line = proc.stdout.strip().splitlines()[0].strip()
    return int(first_line.rsplit(":", 1)[1])


def _resolve_server_expectations(
    pack_id: str,
    mcp_ports: dict[str, int],
    manifest: dict[str, dict[str, Any]],
) -> tuple[dict[str, tuple[int, int]], int]:
    if pack_id == "ra_00a11ea":
        counts = {
            "ms-inspect": (int(mcp_ports.get("ms-inspect", 8000)), 38),
            "ms-modify": (int(mcp_ports.get("ms-modify", 8001)), 19),
            "ms-create": (int(mcp_ports.get("ms-create", 8002)), 4),
        }
        return counts, 61

    dynamic_counts: dict[str, tuple[int, int]] = {}
    for srv_name, port in mcp_ports.items():
        dynamic_counts[srv_name] = (int(port), len(manifest.get(srv_name, {})))
    total_expected = sum(cnt for _, cnt in dynamic_counts.values())
    return dynamic_counts, total_expected


def check_mcp_sidecar_http(
    env: dict[str, str],
    smoke_dir: Path,
    *,
    pack_id: str | None = None,
    cfg: BenchConfig | None = None,
) -> dict[str, Any]:
    """Start mcp-sidecar and verify uid 1000, sidecar offload, run-step handoff, and all tools."""
    reject_slash_tmp(smoke_dir, label="smoke_dir")
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    pack = resolved_cfg.get_pack(pack_id)

    reap_stale_smoke_containers(env, patterns=("^radio-bench-sidecar-smoke-",))

    ws_dir = smoke_dir / "workspace"
    if ws_dir.exists():
        subprocess.run(
            ["podman", "unshare", "rm", "-rf", str(ws_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        shutil.rmtree(ws_dir, ignore_errors=True)
    ws_dir.mkdir(parents=True, exist_ok=True)
    smoke_ms = ws_dir / "smoke.ms"
    if cfg is None:
        generate_minimal_smoke_ms(smoke_ms)
    else:
        generate_minimal_smoke_ms(smoke_ms, cfg=cfg)
    _make_tree_world_writable(ws_dir)
    subprocess.run(
        ["podman", "unshare", "chown", "-R", "1000:1000", str(ws_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    if pack_id is None and cfg is None:
        manifest = load_pinned_tool_manifest()
    else:
        manifest = load_pinned_tool_manifest(pack_id=pack.pack_id, cfg=resolved_cfg)

    if _HEX_SHORT_RE.fullmatch(pack.analyst_ref) and pack.analyst_ref.upper() != "HEAD":
        expected_sha = pack.analyst_ref.lower()
    else:
        _, _, expected_sha = materialize_pack_source(
            pack, resolved_cfg.work_dir, repo_root=resolved_cfg.repo_root
        )

    server_counts, expected_total = _resolve_server_expectations(
        pack.pack_id, pack.mcp_ports, manifest
    )
    sidecar_image = pack.sidecar_image
    cname = f"radio-bench-sidecar-smoke-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    server_summaries: dict[str, Any] = {}
    total_advertised = 0
    total_probed = 0
    total_passed = 0
    failures: list[dict[str, Any]] = []
    sidecar_uid_ok = False
    sidecar_detail_ok = False
    sidecar_detail_path = ""
    handoff_script_ready = False
    run_step_handoff_ok = False

    port_publish_args: list[str] = []
    for srv_port in (
        pack.mcp_ports.get("ms-inspect", 8000),
        pack.mcp_ports.get("ms-modify", 8001),
        pack.mcp_ports.get("ms-create", 8002),
    ):
        port_publish_args.extend(["-p", f"127.0.0.1::{int(srv_port)}"])

    try:
        try:
            proc_start = subprocess.run(
                [
                    "podman",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    cname,
                    *port_publish_args,
                    "-e",
                    f"RADIO_MCP_PORT_MS_INSPECT={int(pack.mcp_ports.get('ms-inspect', 8000))}",
                    "-e",
                    f"RADIO_MCP_PORT_MS_MODIFY={int(pack.mcp_ports.get('ms-modify', 8001))}",
                    "-e",
                    f"RADIO_MCP_PORT_MS_CREATE={int(pack.mcp_ports.get('ms-create', 8002))}",
                    "-v",
                    f"{ws_dir}:/workspace:rw,z",
                    sidecar_image,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if proc_start.returncode != 0:
                return {
                    "ok": False,
                    "error": f"Failed to start {sidecar_image}: {proc_start.stderr.strip()}",
                }

            proc_uid = subprocess.run(
                [
                    "podman",
                    "exec",
                    cname,
                    "sh",
                    "-c",
                    'printf "%s|%s|%s|%s" "$(id -u)" "$RADIO_MCP_DIR" "$ANALYST_REV" "$RADIO_ANALYST_SHA"',
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            expected_ident = f"1000|/workspace/.radio_mcp|{expected_sha}|{expected_sha}"
            sidecar_uid_ok = proc_uid.returncode == 0 and proc_uid.stdout.strip() == expected_ident
            if not sidecar_uid_ok:
                failures.append(
                    {
                        "server": "mcp-sidecar",
                        "tool": "<uid-env-check>",
                        "reason": f"expected {expected_ident!r}, got {proc_uid.stdout.strip()!r} ({proc_uid.stderr.strip()})",
                    }
                )

            for server_name, (container_port, expected_count) in server_counts.items():
                try:
                    host_port = _resolve_published_port(cname, container_port, env)
                except Exception as exc:
                    failures.append(
                        {
                            "server": server_name,
                            "tool": "<port-resolve>",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                url = f"http://127.0.0.1:{host_port}/mcp"
                client = StreamableHTTPMCPClient(url, timeout_s=90.0)
                try:
                    ready = False
                    last_exc = ""
                    for _ in range(60):
                        try:
                            client.initialize()
                            ready = True
                            break
                        except Exception as exc:
                            last_exc = f"{type(exc).__name__}: {exc}"
                            time.sleep(0.5)
                    if not ready:
                        proc_logs = subprocess.run(
                            ["podman", "logs", cname],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        log_tail = (proc_logs.stdout + "\n" + proc_logs.stderr).strip()[-600:]
                        failures.append(
                            {
                                "server": server_name,
                                "tool": "<initialize>",
                                "reason": f"{last_exc} | logs: {log_tail}",
                            }
                        )
                        continue

                    try:
                        advertised = client.list_tools()
                    except Exception as exc:
                        failures.append(
                            {
                                "server": server_name,
                                "tool": "<tools/list>",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue

                    total_advertised += len(advertised)
                    server_tools = manifest.get(server_name, {})
                    server_passed = 0

                    if len(advertised) != expected_count:
                        failures.append(
                            {
                                "server": server_name,
                                "tool": "<tools/list>",
                                "reason": f"expected {expected_count} tools, got {len(advertised)}",
                            }
                        )

                    for tool_name, raw_args in server_tools.items():
                        total_probed += 1
                        if tool_name not in advertised:
                            failures.append(
                                {
                                    "server": server_name,
                                    "tool": tool_name,
                                    "reason": "not in tools/list",
                                }
                            )
                            continue
                        filled_args = json.loads(
                            json.dumps(raw_args).replace("{SMOKE_MS}", "/workspace/smoke.ms")
                        )
                        try:
                            res = client.call_tool(tool_name, {"params": filled_args})
                        except Exception as exc:
                            failures.append(
                                {
                                    "server": server_name,
                                    "tool": tool_name,
                                    "reason": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            continue

                        probe_ok, fail_reason = classify_tool_probe_response(tool_name, res)
                        if not probe_ok:
                            failures.append(
                                {
                                    "server": server_name,
                                    "tool": tool_name,
                                    "reason": fail_reason,
                                }
                            )
                        else:
                            total_passed += 1
                            server_passed += 1

                        if server_name == "ms-inspect" and tool_name == "ms_antenna_list":
                            data_dict = _extract_tool_data(res)
                            dpath = str(data_dict.get("detail_path") or "")
                            sidecar_detail_path = dpath
                            if dpath.startswith("/workspace/.radio_mcp/sidecars/"):
                                rel = Path(dpath).relative_to("/workspace")
                                host_sidecar = ws_dir / rel
                                if host_sidecar.is_file() and host_sidecar.stat().st_size > 0:
                                    sidecar_detail_ok = True
                            if not sidecar_detail_ok:
                                failures.append(
                                    {
                                        "server": "ms-inspect",
                                        "tool": "ms_antenna_list<detail_path>",
                                        "reason": f"expected sidecar file under /workspace/.radio_mcp/sidecars/, got {dpath!r}",
                                    }
                                )

                    if server_name == "ms-modify":
                        try:
                            handoff_res = client.call_tool(
                                "ms_flag_data",
                                {
                                    "params": {
                                        "ms_path": "/workspace/smoke.ms",
                                        "workdir": "/workspace",
                                        "reason": "wave2_cross_container_smoke",
                                        "antenna": "0",
                                        "execute": False,
                                    }
                                },
                            )
                            handoff_ok, handoff_reason = classify_tool_probe_response(
                                "ms_flag_data", handoff_res
                            )
                            if handoff_ok and (ws_dir / "flag_data.py").is_file():
                                handoff_script_ready = True
                            else:
                                failures.append(
                                    {
                                        "server": "ms-modify",
                                        "tool": "ms_flag_data<handoff_script>",
                                        "reason": handoff_reason
                                        or "flag_data.py missing on ws_dir",
                                    }
                                )
                        except Exception as exc:
                            failures.append(
                                {
                                    "server": "ms-modify",
                                    "tool": "ms_flag_data<handoff_script>",
                                    "reason": f"{type(exc).__name__}: {exc}",
                                }
                            )

                    server_summaries[server_name] = {
                        "container_port": container_port,
                        "host_port": host_port,
                        "expected_tools": expected_count,
                        "advertised_tools": len(advertised),
                        "probed_tools": len(server_tools),
                        "passed_tools": server_passed,
                    }
                finally:
                    client.close()
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "error": f"Timed out starting or querying {sidecar_image}: {exc}",
            }
        finally:
            subprocess.run(
                ["podman", "rm", "-f", "-t", "0", cname],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        if handoff_script_ready:
            proc_step = run_offline_container(
                AGENT_IMAGE,
                [
                    "/usr/local/bin/run-step.sh",
                    "/opt/casa-venv/bin/python",
                    "/workspace/flag_data.py",
                    "--wait",
                    "60",
                ],
                env,
                timeout_s=90.0,
                extra_podman_args=["--user", "1000:1000", "-v", f"{ws_dir}:/workspace:rw,z"],
                label="agent-run-step",
            )
            rc_file = ws_dir / "flag_data.py.rc"
            log_file = ws_dir / "flag_data.py.log"
            stage_log = ws_dir / "stage_log.jsonl"
            stage_text = stage_log.read_text() if stage_log.is_file() else ""
            run_step_handoff_ok = (
                proc_step.returncode == 0
                and "status=done rc=0" in proc_step.stdout
                and rc_file.is_file()
                and rc_file.read_text().strip() == "0"
                and log_file.is_file()
                and '"tool": "ms_flag_data"' in stage_text
                and "/workspace/.radio_mcp/logs/" in stage_text
            )
            if not run_step_handoff_ok:
                failures.append(
                    {
                        "server": "agent-base",
                        "tool": "run-step.sh<flag_data.py>",
                        "reason": f"rc={proc_step.returncode} stdout={proc_step.stdout[-300:]!r} stderr={proc_step.stderr[-300:]!r}",
                    }
                )
    finally:
        subprocess.run(
            ["podman", "unshare", "chown", "-R", "0:0", str(ws_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    ok = (
        total_advertised == expected_total
        and total_probed == expected_total
        and total_passed == expected_total
        and sidecar_uid_ok
        and sidecar_detail_ok
        and run_step_handoff_ok
        and not failures
    )
    return {
        "ok": ok,
        "pack_id": pack.pack_id,
        "sidecar_image": sidecar_image,
        "analyst_sha": expected_sha,
        "sidecar_uid_1000_ok": sidecar_uid_ok,
        "sidecar_detail_ok": sidecar_detail_ok,
        "sidecar_detail_path": sidecar_detail_path,
        "run_step_handoff_ok": run_step_handoff_ok,
        "expected_total": expected_total,
        "total_advertised": total_advertised,
        "total_probed": total_probed,
        "total_passed": total_passed,
        "servers": server_summaries,
        "failures": failures,
    }


def run_all_smoke_checks(
    smoke_dir: Path | None = None,
    *,
    pack_id: str | None = None,
    cfg: BenchConfig | None = None,
) -> dict[str, Any]:
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    out_dir = smoke_dir or (resolved_cfg.work_dir / "smoke")
    reject_slash_tmp(out_dir, label="smoke_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_podman_env(cfg=resolved_cfg)
    reap_stale_smoke_containers(env)
    t0 = time.monotonic()

    casa_base_res = check_casa_base_offline(env)
    agent_res = check_agent_base_offline_and_isolation(env)
    verifier_res = check_verifier_offline(env)
    sidecar_res = check_mcp_sidecar_http(env, out_dir, pack_id=pack_id, cfg=resolved_cfg)

    elapsed_s = round(time.monotonic() - t0, 2)
    report = {
        "ok": bool(
            casa_base_res["ok"] and agent_res["ok"] and verifier_res["ok"] and sidecar_res["ok"]
        ),
        "pack_id": sidecar_res.get("pack_id"),
        "elapsed_s": elapsed_s,
        "casa_base": casa_base_res,
        "agent_base": agent_res,
        "verifier": verifier_res,
        "mcp_sidecar": sidecar_res,
    }
    report_path = out_dir / "wave2_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run container isolation and MCP HTTP smoke checks."
    )
    add_config_cli_args(parser)
    parser.add_argument("--pack", type=str, default=None, help="Pack ID to probe.")
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=None,
        help="Scratch directory under work_dir for smoke artifacts.",
    )
    args = parser.parse_args(argv)
    cfg = config_from_cli_args(args)
    report = run_all_smoke_checks(args.smoke_dir, pack_id=args.pack, cfg=cfg)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
