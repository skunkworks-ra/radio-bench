"""Unit tests for MCP streamable-http client, SSE parsing, probe rules, and Podman teardown."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

import radio_bench.smoke_containers as sc
from radio_bench.smoke_containers import (
    EXPECTED_SERVER_COUNTS,
    StreamableHTTPMCPClient,
    check_mcp_sidecar_http,
    classify_tool_probe_response,
    load_pinned_tool_manifest,
    parse_mcp_http_body,
    run_offline_container,
)


def test_parse_mcp_http_body_json() -> None:
    body = parse_mcp_http_body(
        "application/json",
        '{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"ms_field_list"}]}}',
    )
    assert body["id"] == 1
    assert body["result"]["tools"][0]["name"] == "ms_field_list"


def test_parse_mcp_http_body_sse() -> None:
    sse = 'event: message\r\ndata: {"jsonrpc":"2.0","id":2,"result":{"ok":true}}\r\n\r\n'
    body = parse_mcp_http_body("text/event-stream; charset=utf-8", sse)
    assert body["id"] == 2
    assert body["result"]["ok"] is True


def test_parse_mcp_http_body_multi_event_and_eof_sse() -> None:
    sse_multi = (
        "event: message\n"
        'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"step":1}}\n\n'
        "event: message\n"
        'data: {"jsonrpc":"2.0","id":7,\n'
        'data: "result":{"tools":[{"name":"ms_antenna_list"}]}}'
    )
    body = parse_mcp_http_body("text/event-stream", sse_multi)
    assert body["id"] == 7
    assert body["result"]["tools"][0]["name"] == "ms_antenna_list"


def test_parse_mcp_http_body_empty_sse_raises() -> None:
    with pytest.raises(ValueError, match="No JSON-RPC event"):
        parse_mcp_http_body("text/event-stream", "event: ping\n\n")


def test_pinned_manifest_has_exact_61_tools() -> None:
    manifest = load_pinned_tool_manifest()
    total = 0
    for server, (_port, expected) in EXPECTED_SERVER_COUNTS.items():
        assert len(manifest[server]) == expected
        total += len(manifest[server])
    assert total == 61


def test_classify_tool_probe_response_rejects_unexpected_and_core_errors() -> None:
    ok_core, reason_core = classify_tool_probe_response(
        "ms_observation_info",
        {"isError": False, "content": [{"type": "text", "text": '{"status":"ok"}'}]},
    )
    assert ok_core is True
    assert reason_core == ""

    bad_unexp, reason_unexp = classify_tool_probe_response(
        "ms_calsol_stats",
        {
            "isError": True,
            "structuredContent": {"error_type": "UNEXPECTED_ERROR"},
            "content": [{"type": "text", "text": '{"error_type":"UNEXPECTED_ERROR"}'}],
        },
    )
    assert bad_unexp is False
    assert "unexpected unhandled exception" in reason_unexp

    bad_core, reason_bad_core = classify_tool_probe_response(
        "ms_observation_info",
        {
            "isError": True,
            "structuredContent": {"error_type": "MS_NOT_FOUND"},
            "content": [{"type": "text", "text": '{"error_type":"MS_NOT_FOUND"}'}],
        },
    )
    assert bad_core is False
    assert "core smoke.ms tool failed" in reason_bad_core

    ok_domain, _ = classify_tool_probe_response(
        "ms_calsol_stats",
        {
            "isError": True,
            "structuredContent": {"error_type": "CALTABLE_NOT_FOUND"},
            "content": [{"type": "text", "text": '{"error_type":"CALTABLE_NOT_FOUND"}'}],
        },
    )
    assert ok_domain is True


def test_classify_tool_probe_response_rejects_import_and_missing_envelope() -> None:
    bad_imp, reason_imp = classify_tool_probe_response(
        "ms_bandpass",
        {
            "isError": True,
            "structuredContent": {"error_type": "COMPUTATION_ERROR"},
            "content": [{"type": "text", "text": "ModuleNotFoundError: No module named 'six'"}],
        },
    )
    assert bad_imp is False
    assert "import failure inside tool" in reason_imp

    bad_imp2, reason_imp2 = classify_tool_probe_response(
        "ms_tclean",
        {
            "isError": False,
            "content": [{"type": "text", "text": "ImportError: libssl.so.3 symbol mismatch"}],
        },
    )
    assert bad_imp2 is False
    assert "import failure inside tool" in reason_imp2

    bad_env, reason_env = classify_tool_probe_response(
        "ms_calsol_stats",
        {
            "isError": True,
            "content": [{"type": "text", "text": "Unhandled FastMCP validation error"}],
        },
    )
    assert bad_env is False
    assert "MCP error without tool envelope" in reason_env


def test_mcp_client_session_id_propagation() -> None:
    seen_requests: list[tuple[str | None, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_requests.append((request.headers.get("Mcp-Session-Id"), payload))
        method = payload["method"]
        if method == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "mcp-session-id": "sess-xyz-999",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"serverInfo": {"name": "m"}},
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202, text="")
        if method == "tools/list":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=(
                    "event: message\n"
                    f'data: {{"jsonrpc":"2.0","id":{payload["id"]},"result":{{"tools":[{{"name":"ms_field_list"}}]}}}}\n\n'
                ),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"isError": False, "content": [{"type": "text", "text": "{}"}]},
            },
        )

    client = StreamableHTTPMCPClient(
        "http://127.0.0.1:18000/mcp",
        transport=httpx.MockTransport(handler),
    )
    try:
        client.initialize()
        assert client.session_id == "sess-xyz-999"
        tools = client.list_tools()
        assert tools == ["ms_field_list"]
        call_res = client.call_tool("ms_field_list", {"params": {"ms_path": "/workspace/smoke.ms"}})
        assert call_res["isError"] is False
    finally:
        client.close()

    assert [sid for sid, _ in seen_requests] == [
        None,
        "sess-xyz-999",
        "sess-xyz-999",
        "sess-xyz-999",
    ]
    assert [p.get("id") for _, p in seen_requests] == [1, None, 2, 3]


def test_mcp_client_jsonrpc_error_and_http_500() -> None:
    def rpc_err_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "error": {"code": -32601, "message": "Method not found"},
            },
        )

    client_err = StreamableHTTPMCPClient(
        "http://127.0.0.1:18000/mcp",
        transport=httpx.MockTransport(rpc_err_handler),
    )
    try:
        with pytest.raises(RuntimeError, match="JSON-RPC error on tools/list"):
            client_err.list_tools()
    finally:
        client_err.close()

    def http_500_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client_500 = StreamableHTTPMCPClient(
        "http://127.0.0.1:18000/mcp",
        transport=httpx.MockTransport(http_500_handler),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client_500.request("tools/list", {})
        with pytest.raises(httpx.HTTPStatusError):
            client_500.notify("notifications/initialized", {})
    finally:
        client_500.close()


def test_run_offline_container_teardown_on_timeout_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], float | None]] = []

    def fake_run_timeout(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(cmd), kwargs.get("timeout")))
        if cmd[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5.0, output="partial", stderr="hung")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sc.subprocess, "run", fake_run_timeout)
    res = run_offline_container("localhost/test:latest", ["python", "-V"], {}, timeout_s=5.0)
    assert res.timed_out is True
    assert res.returncode == 124
    assert "Timed out after 5.0s" in res.stderr
    assert len(calls) == 2
    run_cmd, _ = calls[0]
    rm_cmd, rm_timeout = calls[1]
    cname = run_cmd[run_cmd.index("--name") + 1]
    assert "--network=none" in run_cmd
    assert rm_cmd == ["podman", "rm", "-f", "-t", "0", cname]
    assert rm_timeout == 30

    calls.clear()

    def fake_run_boom(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(cmd), kwargs.get("timeout")))
        if cmd[:2] == ["podman", "run"]:
            raise OSError("podman socket unavailable")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sc.subprocess, "run", fake_run_boom)
    with pytest.raises(OSError, match="podman socket unavailable"):
        run_offline_container("localhost/test:latest", ["python", "-V"], {}, timeout_s=5.0)
    assert len(calls) == 2
    assert calls[1][0][:4] == ["podman", "rm", "-f", "-t"]
    assert calls[1][0][4] == "0"
    assert calls[1][1] == 30


def test_sidecar_http_teardown_on_start_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rm_calls: list[tuple[list[str], float | None]] = []

    monkeypatch.setattr(
        sc,
        "generate_minimal_smoke_ms",
        lambda ms_path: ms_path.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(sc, "load_pinned_tool_manifest", lambda: {"ms-inspect": {}})

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["podman", "run", "-d"]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=60.0)
        if cmd[:2] == ["podman", "rm"]:
            rm_calls.append((list(cmd), kwargs.get("timeout")))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    report = check_mcp_sidecar_http({}, tmp_path)
    assert report["ok"] is False
    assert "Timed out" in report["error"]
    assert len(rm_calls) == 1
    assert rm_calls[0][0][:5] == ["podman", "rm", "-f", "-t", "0"]
    assert rm_calls[0][0][5].startswith("radio-bench-sidecar-smoke-")
    assert rm_calls[0][1] == 30
