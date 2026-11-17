"""Multi-harness ATIF + ClaudeCode sidechain + OpenCode SQLite trace & role attribution."""

from __future__ import annotations

import contextlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Exact 61 MCP tools across radio-analyst at pinned commit 00a11ea.
MS_INSPECT_TOOLS: frozenset[str] = frozenset(
    {
        "ms_antenna_flag_fraction",
        "ms_antenna_list",
        "ms_baseline_lengths",
        "ms_calsol_plot",
        "ms_calsol_stats",
        "ms_calsol_stats_detail",
        "ms_casa_task_lookup",
        "ms_casa_task_spec",
        "ms_casa_tool_spec",
        "ms_corrected_stats",
        "ms_correlator_config",
        "ms_elevation_vs_time",
        "ms_field_list",
        "ms_flag_preflight",
        "ms_flag_summary",
        "ms_gaincal_snr_predict",
        "ms_image_stats",
        "ms_observation_info",
        "ms_online_flag_stats",
        "ms_parallactic_angle_vs_time",
        "ms_phase_cal_lookup",
        "ms_plot_caltable_library",
        "ms_pol_cal_conditions",
        "ms_radiometer_estimate",
        "ms_refant",
        "ms_residual_stats",
        "ms_rfi_channel_stats",
        "ms_scan_intent_summary",
        "ms_scan_list",
        "ms_shadowing_report",
        "ms_spectral_window_list",
        "ms_spw_amp_severity",
        "ms_verify_caltables",
        "ms_verify_import",
        "ms_verify_model",
        "ms_verify_priorcals",
        "ms_vis_amp_time_stats",
        "ms_workflow_status",
    }
)

MS_MODIFY_TOOLS: frozenset[str] = frozenset(
    {
        "ms_apply_initial_rflag",
        "ms_apply_preflag",
        "ms_apply_rflag",
        "ms_applycal",
        "ms_bandpass",
        "ms_casa_task",
        "ms_casa_tool",
        "ms_flag_caltable",
        "ms_flag_data",
        "ms_fluxscale",
        "ms_gaincal",
        "ms_generate_priorcals",
        "ms_initial_bandpass",
        "ms_polcal",
        "ms_postcal_flag",
        "ms_set_intents",
        "ms_setjy",
        "ms_setjy_polcal",
        "ms_tclean",
    }
)

MS_CREATE_TOOLS: frozenset[str] = frozenset(
    {
        "ms_import_asdm",
        "ms_reduction_log",
        "ms_report",
        "ms_sdm_summary",
    }
)

ALL_KNOWN_MCP_TOOLS: frozenset[str] = MS_INSPECT_TOOLS | MS_MODIFY_TOOLS | MS_CREATE_TOOLS

BASH_TOOL_NAMES: frozenset[str] = frozenset(
    {"bash", "shell", "sh", "execute_bash", "run_command", "terminal", "cmd"}
)

DISPATCH_TOOL_NAMES: frozenset[str] = frozenset({"agent", "task", "dispatch_agent"})
PASSIVE_READ_TOOL_NAMES: frozenset[str] = frozenset({"read", "grep", "glob"})
PASSIVE_OR_DISPATCH_TOOL_NAMES: frozenset[str] = DISPATCH_TOOL_NAMES | PASSIVE_READ_TOOL_NAMES

GENERIC_AGENT_MODES: frozenset[str] = frozenset(
    {"build", "plan", "general", "general-purpose", "default", "main"}
)

NONZERO_EXIT_RE = re.compile(
    r"(?:\[exit_code\]|exit\s+code|exit\s+status|returncode)\s*[:=]?\s*(-?[1-9]\d*)\b",
    re.IGNORECASE,
)
RUN_STEP_RC_RE = re.compile(r"\bstatus=failed\b|\brc=(-?[1-9]\d*)\b")
OPENCODE_TASK_ID_RE = re.compile(r"""<task\s+id=["'](ses_[^"']+)["']""")
OPENCODE_TITLE_ROLE_RE = re.compile(r"@([A-Za-z0-9_.-]+)\s+subagent\b")
CLAUDE_AGENT_ID_JSON_RE = re.compile(r'"agentId"\s*:\s*"([^"]+)"')
CLAUDE_AGENT_ID_TEXT_RE = re.compile(r"\bagentId\s*[:=]\s*([A-Za-z0-9_-]+)")

PROTOCOL_ERR_PHRASES: tuple[str, ...] = (
    "unknown tool",
    "unavailable tool",
    "invalid tool",
    "tool not found",
    "no such tool",
    "is not a valid tool",
    "unrecognized tool",
    "mcp error",
    "streamablehttp",
    "connection refused",
    "failed to connect to mcp",
    "inputvalidationerror",
    "validation error for",
    "missing required argument",
    "unexpected keyword argument",
    '"error_type": "validation_error"',
    '"error_type":"validation_error"',
    '"error_type": "invalid_input"',
    '"error_type":"invalid_input"',
)

CASA_SEVERE_PHRASES: tuple[str, ...] = (
    "severe\t",
    "severe ",
    "RuntimeError:",
    "Traceback (most recent call last):",
)


def safe_nonneg_int(val: Any) -> int:
    if isinstance(val, bool) or val is None:
        return 0
    if isinstance(val, (int, float)):
        if not math.isfinite(float(val)):
            return 0
        return max(0, int(round(float(val))))
    return 0


def safe_nonneg_float(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        if math.isfinite(fval) and fval >= 0.0:
            return fval
    return None


def stringify_obs_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _matches_mcp_tool(cand: str, tool: str) -> bool:
    return cand == tool or cand.endswith((f"_{tool}", f"-{tool}", f"/{tool}", f".{tool}"))


def classify_mcp_server(function_name: str) -> str | None:
    """Return `'ms-inspect'`, `'ms-modify'`, `'ms-create'`, `'mcp-other'`, or `None`."""
    raw = function_name.strip()
    if not raw:
        return None
    low = raw.lower()

    if "ms-inspect" in low or "ms_inspect" in low:
        return "ms-inspect"
    if "ms-modify" in low or "ms_modify" in low:
        return "ms-modify"
    if "ms-create" in low or "ms_create" in low:
        return "ms-create"

    candidates = [low]
    for sep in ("__", ":", ".", "/"):
        if sep in low:
            candidates.append(low.rsplit(sep, 1)[-1])
    if low.startswith("mcp_"):
        candidates.append(low[4:])

    for cand in candidates:
        if cand in MS_INSPECT_TOOLS:
            return "ms-inspect"
        if cand in MS_MODIFY_TOOLS:
            return "ms-modify"
        if cand in MS_CREATE_TOOLS:
            return "ms-create"
        for tool in MS_INSPECT_TOOLS:
            if _matches_mcp_tool(cand, tool):
                return "ms-inspect"
        for tool in MS_MODIFY_TOOLS:
            if _matches_mcp_tool(cand, tool):
                return "ms-modify"
        for tool in MS_CREATE_TOOLS:
            if _matches_mcp_tool(cand, tool):
                return "ms-create"

    if low.startswith("mcp__") or low.startswith("mcp_") or low.startswith("ms_"):
        return "mcp-other"
    return None


def extract_exit_code(
    tc_extra: dict[str, Any], obs_extra: dict[str, Any], obs_text: str
) -> int | None:
    for container in (obs_extra, tc_extra):
        for k in ("exit_code", "exitCode", "returncode"):
            v = container.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        meta = container.get("tool_result_metadata")
        if isinstance(meta, dict):
            tur = meta.get("tool_use_result")
            if isinstance(tur, dict):
                for k in ("exitCode", "exit_code", "returncode"):
                    v = tur.get(k)
                    if isinstance(v, int) and not isinstance(v, bool):
                        return v

    m = NONZERO_EXIT_RE.search(obs_text)
    if m:
        with contextlib.suppress(ValueError):
            return int(m.group(1))
    m2 = RUN_STEP_RC_RE.search(obs_text)
    if m2:
        if m2.group(1):
            try:
                return int(m2.group(1))
            except ValueError:
                return 1
        return 1
    return None


def is_tool_error(
    tc_extra: dict[str, Any],
    obs_extra: dict[str, Any],
    obs_text: str,
    *,
    fn_name: str = "",
) -> bool:
    for container in (tc_extra, obs_extra):
        for k in ("is_error", "tool_use_is_error", "tool_result_is_error"):
            if container.get(k) is True:
                return True
        meta = container.get("tool_result_metadata")
        if isinstance(meta, dict) and meta.get("is_error") is True:
            return True

    if fn_name.strip().lower() in PASSIVE_OR_DISPATCH_TOOL_NAMES:
        return False

    low = obs_text.lower()
    if "[error] tool reported failure" in low:
        return True
    if '"iserror": true' in low or '"iserror":true' in low:
        return True
    if ('"status": "error"' in low or '"status":"error"' in low) and '"error_type"' in low:
        return True
    return any(phrase in low for phrase in PROTOCOL_ERR_PHRASES)


def is_protocol_error(fn_name: str, mcp_srv: str | None, obs_text: str, arm: str) -> bool:
    low_fn = fn_name.strip().lower()
    low_obs = obs_text.lower()

    if low_fn == "invalid":
        return True
    if mcp_srv is not None and arm in {"00_none", "01_skills_raw", "01_skills_casa"}:
        return True
    if mcp_srv == "mcp-other" and low_fn.startswith("ms_") and low_fn not in ALL_KNOWN_MCP_TOOLS:
        return True
    if low_fn in PASSIVE_OR_DISPATCH_TOOL_NAMES:
        return False
    return any(phrase in low_obs for phrase in PROTOCOL_ERR_PHRASES)


def extract_dispatched_role(arguments: Any) -> str | None:
    """Extract specialist role name from an `Agent` / `Task` tool call arguments dict."""
    if not isinstance(arguments, dict):
        return None
    sub_type = arguments.get("subagent_type") or arguments.get("agent_type")
    name_arg = arguments.get("name")
    agent_arg = arguments.get("agent") or arguments.get("role")

    if isinstance(sub_type, str) and sub_type.strip():
        st_clean = sub_type.strip()
        if (
            st_clean.lower() in GENERIC_AGENT_MODES
            and isinstance(name_arg, str)
            and name_arg.strip()
        ):
            return name_arg.strip()
        return st_clean
    for cand in (name_arg, agent_arg):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return None


def extract_claude_agent_id_from_obs(obs_extra: dict[str, Any], obs_text: str) -> str | None:
    """Extract spawned ClaudeCode `agentId` from `ObservationResult` metadata or text."""
    meta = obs_extra.get("tool_result_metadata")
    if isinstance(meta, dict):
        tur = meta.get("tool_use_result")
        if isinstance(tur, dict):
            for k in ("agentId", "agent_id"):
                val = tur.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    for k in ("agentId", "agent_id"):
        val = obs_extra.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    m = CLAUDE_AGENT_ID_JSON_RE.search(obs_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m2 = CLAUDE_AGENT_ID_TEXT_RE.search(obs_text)
    if m2 and m2.group(1).strip():
        return m2.group(1).strip()
    return None


def extract_opencode_session_id_from_obs(obs_extra: dict[str, Any], obs_text: str) -> str | None:
    """Extract spawned OpenCode child `ses_...` ID from `ObservationResult` or `<task id=...>`."""
    for k in ("sessionId", "session_id"):
        val = obs_extra.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    meta = obs_extra.get("metadata")
    if isinstance(meta, dict):
        for k in ("sessionId", "session_id"):
            val = meta.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
    m = OPENCODE_TASK_ID_RE.search(obs_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def scan_claude_subagent_meta_files(
    trial_dir: Path | None,
    call_id_to_role: dict[str, str],
) -> tuple[dict[str, str], set[str]]:
    """Scan `<trial_dir>/agent/sessions/projects/**/subagents/agent-*.meta.json` for `agentId -> role`."""
    agent_id_to_role: dict[str, str] = {}
    linked_call_ids: set[str] = set()
    if trial_dir is None or not trial_dir.is_dir():
        return agent_id_to_role, linked_call_ids

    roots: list[Path] = [trial_dir / "agent" / "sessions" / "projects"]
    steps_dir = trial_dir / "steps"
    if steps_dir.is_dir():
        with contextlib.suppress(OSError):
            for sdir in sorted(p for p in steps_dir.iterdir() if p.is_dir()):
                roots.append(sdir / "agent" / "sessions" / "projects")

    for root in roots:
        if not root.is_dir():
            continue
        try:
            meta_files = sorted(root.rglob("subagents/agent-*.meta.json"))
        except OSError:
            continue
        for mf in meta_files:
            try:
                raw = json.loads(mf.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(raw, dict):
                continue
            aid = str(raw.get("agentId") or raw.get("agent_id") or "").strip()
            if not aid and mf.name.startswith("agent-") and mf.name.endswith(".meta.json"):
                aid = mf.name[len("agent-") : -len(".meta.json")].strip()
            if not aid:
                continue
            atype = str(raw.get("agentType") or raw.get("subagent_type") or "").strip()
            tuid = str(raw.get("toolUseId") or raw.get("tool_use_id") or "").strip()
            call_role = call_id_to_role.get(tuid) if tuid else None
            if tuid and tuid in call_id_to_role:
                linked_call_ids.add(tuid)

            chosen_role: str | None = None
            if atype and atype.lower() not in GENERIC_AGENT_MODES:
                chosen_role = atype
            elif call_role:
                chosen_role = call_role
            elif atype:
                chosen_role = atype
            if chosen_role:
                agent_id_to_role[aid] = chosen_role

    return agent_id_to_role, linked_call_ids


def parse_opencode_tool_events(opencode_txt_path: Path | None) -> dict[str, dict[str, Any]]:
    """Parse `agent/opencode.txt` JSONL `tool_use` events indexed by `part.callID` / `part.id`."""
    events_by_id: dict[str, dict[str, Any]] = {}
    if opencode_txt_path is None or not opencode_txt_path.is_file():
        return events_by_id

    try:
        lines = opencode_txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return events_by_id

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "tool_use":
            continue
        part = ev.get("part")
        if not isinstance(part, dict):
            continue
        parsed = _parse_opencode_tool_part_dict(part)
        if parsed is None:
            continue
        cid = parsed["call_id"]
        prev = events_by_id.get(cid)
        events_by_id[cid] = _merge_opencode_tool_event(prev, parsed)
    return events_by_id


def _parse_opencode_tool_part_dict(part: dict[str, Any]) -> dict[str, Any] | None:
    cid = str(part.get("callID") or part.get("id") or "").strip()
    if not cid:
        return None
    state = part.get("state") if isinstance(part.get("state"), dict) else {}
    meta = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    inp = state.get("input") if isinstance(state.get("input"), dict) else {}

    raw_exit = meta.get("exit")
    if raw_exit is None:
        raw_exit = meta.get("exitCode", meta.get("exit_code"))
    exit_code = (
        int(raw_exit) if isinstance(raw_exit, int) and not isinstance(raw_exit, bool) else None
    )

    raw_err = state.get("error")
    if raw_err is None or raw_err is False:
        err_str = ""
    elif isinstance(raw_err, (dict, list)):
        err_str = json.dumps(raw_err)
    else:
        err_str = str(raw_err).strip()

    status_str = str(state.get("status") or "").strip().lower()
    out_str = stringify_obs_content(state.get("output"))
    tool_name = str(part.get("tool") or "").strip()

    child_sid = str(meta.get("sessionId") or meta.get("session_id") or "").strip()
    if not child_sid and out_str:
        m = OPENCODE_TASK_ID_RE.search(out_str)
        if m:
            child_sid = m.group(1).strip()

    dispatched_role = (
        extract_dispatched_role(inp) if tool_name.lower() in DISPATCH_TOOL_NAMES else None
    )

    return {
        "call_id": cid,
        "tool": tool_name,
        "status": status_str,
        "error": err_str,
        "exit": exit_code,
        "output": out_str,
        "input": inp,
        "child_session_id": child_sid or None,
        "dispatched_role": dispatched_role,
    }


def _merge_opencode_tool_event(prev: dict[str, Any] | None, cur: dict[str, Any]) -> dict[str, Any]:
    if prev is None:
        return dict(cur)
    return {
        "call_id": cur["call_id"],
        "tool": cur["tool"] or prev.get("tool", ""),
        "status": cur["status"] or prev.get("status", ""),
        "error": cur["error"] or prev.get("error", ""),
        "exit": cur["exit"] if cur["exit"] is not None else prev.get("exit"),
        "output": cur["output"] or prev.get("output", ""),
        "input": cur["input"] or prev.get("input", {}),
        "child_session_id": cur["child_session_id"] or prev.get("child_session_id"),
        "dispatched_role": cur["dispatched_role"] or prev.get("dispatched_role"),
    }


@dataclass(frozen=True)
class OpenCodeDBTurn:
    """Single recovered inference turn from `opencode.db`."""

    session_id: str
    message_id: str
    model_name: str | None
    uncached_input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reply_output_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    has_token_metrics: bool
    tool_parts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class OpenCodeDBSession:
    """Single root or child session recovered from `opencode.db`."""

    session_id: str
    parent_id: str | None
    title: str
    agent: str
    model_name: str | None
    variant: str | None
    resolved_role: str
    turns: tuple[OpenCodeDBTurn, ...] = ()
    n_step_finish_parts: int = 0
    n_tool_parts: int = 0
    has_unfinished_step: bool = False


@dataclass(frozen=True)
class OpenCodeDBSummary:
    """Complete read-only extraction of `opencode.db`."""

    sessions: tuple[OpenCodeDBSession, ...] = ()
    tool_events_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    child_session_roles_from_parent: dict[str, str] = field(default_factory=dict)
    total_step_finish_events: int = 0
    total_tool_events: int = 0
    has_unfinished_step: bool = False

    @property
    def root_sessions(self) -> list[OpenCodeDBSession]:
        return [s for s in self.sessions if not s.parent_id]

    @property
    def child_sessions(self) -> list[OpenCodeDBSession]:
        return [s for s in self.sessions if bool(s.parent_id)]


def resolve_opencode_db_path(
    trial_dir: Path | None = None,
    explicit_db_path: Path | None = None,
) -> Path | None:
    """Locate `<trial_dir>/agent/opencode/xdg-data/opencode/opencode.db` if present."""
    if explicit_db_path is not None and Path(explicit_db_path).is_file():
        return Path(explicit_db_path)
    if trial_dir is None or not Path(trial_dir).is_dir():
        return None
    td = Path(trial_dir)
    primary = td / "agent" / "opencode" / "xdg-data" / "opencode" / "opencode.db"
    if primary.is_file():
        return primary
    steps_dir = td / "steps"
    if steps_dir.is_dir():
        with contextlib.suppress(OSError):
            step_dirs = sorted(p for p in steps_dir.iterdir() if p.is_dir())
            if step_dirs:
                cand = (
                    step_dirs[-1] / "agent" / "opencode" / "xdg-data" / "opencode" / "opencode.db"
                )
                if cand.is_file():
                    return cand
    return None


def _connect_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    wal_path = db_path.with_name(db_path.name + "-wal")
    has_wal = False
    with contextlib.suppress(OSError):
        has_wal = wal_path.is_file() and wal_path.stat().st_size > 0

    if has_wal:
        try:
            return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error:
            return sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    return sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)


def _parse_opencode_session_model(raw_model: Any) -> tuple[str | None, str | None]:
    if not isinstance(raw_model, str) or not raw_model.strip():
        return None, None
    txt = raw_model.strip()
    if txt.startswith("{"):
        try:
            obj = json.loads(txt)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            mid = str(obj.get("id") or obj.get("modelID") or "").strip()
            prov = str(obj.get("providerID") or obj.get("provider") or "").strip()
            variant = str(obj.get("variant") or "").strip() or None
            if mid and prov and "/" not in mid:
                return f"{prov}/{mid}", variant
            return (mid or None), variant
    return txt, None


def _parse_tokens_obj(tokens: Any) -> tuple[int, int, int, int, int, bool]:
    """Return `(uncached_in, cache_read, cache_write, reply_out, reasoning, has_any)`."""
    if not isinstance(tokens, dict):
        return 0, 0, 0, 0, 0, False
    u_in = safe_nonneg_int(tokens.get("input"))
    out_tok = safe_nonneg_int(tokens.get("output"))
    reas_tok = safe_nonneg_int(tokens.get("reasoning"))
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    c_read = safe_nonneg_int(cache.get("read", tokens.get("cache_read")))
    c_write = safe_nonneg_int(cache.get("write", tokens.get("cache_write")))
    has_any = (u_in + out_tok + reas_tok + c_read + c_write) > 0
    return u_in, c_read, c_write, out_tok, reas_tok, has_any


def read_opencode_sqlite(
    db_path: Path | None,
    *,
    parent_traj_session_roles: dict[str, str] | None = None,
    parent_unlinked_roles: list[str] | None = None,
    parent_ordered_dispatches: list[tuple[str, str]] | None = None,
    parent_linked_cids: set[str] | None = None,
) -> OpenCodeDBSummary | None:
    """Read sessions, turns, tokens, and tool states from `opencode.db` strictly read-only."""
    if db_path is None or not Path(db_path).is_file():
        return None

    con: sqlite3.Connection | None = None
    try:
        con = _connect_readonly_sqlite(Path(db_path))
        tables = {
            str(r[0])
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "session" not in tables:
            return None

        sess_cols = [str(r[1]) for r in con.execute("PRAGMA table_info('session')").fetchall()]
        wanted_sess_cols = [
            c
            for c in (
                "id",
                "parent_id",
                "title",
                "agent",
                "model",
                "cost",
                "tokens_input",
                "tokens_output",
                "tokens_reasoning",
                "tokens_cache_read",
                "tokens_cache_write",
                "time_created",
            )
            if c in sess_cols
        ]
        if "id" not in wanted_sess_cols:
            return None
        order_clause = (
            "ORDER BY time_created ASC, id ASC"
            if "time_created" in wanted_sess_cols
            else "ORDER BY id ASC"
        )
        raw_sess_rows = con.execute(
            f"SELECT {', '.join(wanted_sess_cols)} FROM session {order_clause}"
        ).fetchall()
        sess_dicts = [dict(zip(wanted_sess_cols, row, strict=False)) for row in raw_sess_rows]

        msg_by_id: dict[str, dict[str, Any]] = {}
        msgs_by_session: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        if "message" in tables:
            m_cols = [str(r[1]) for r in con.execute("PRAGMA table_info('message')").fetchall()]
            if {"id", "session_id", "data"}.issubset(m_cols):
                m_order = (
                    "ORDER BY time_created ASC, id ASC"
                    if "time_created" in m_cols
                    else "ORDER BY id ASC"
                )
                for mid, sid, mdata in con.execute(
                    f"SELECT id, session_id, data FROM message {m_order}"
                ).fetchall():
                    if not isinstance(mdata, str):
                        continue
                    try:
                        mobj = json.loads(mdata)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(mobj, dict):
                        mid_s = str(mid)
                        sid_s = str(sid)
                        msg_by_id[mid_s] = mobj
                        msgs_by_session.setdefault(sid_s, []).append((mid_s, mobj))

        parts_by_session: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        tool_events_by_id: dict[str, dict[str, Any]] = {}
        child_roles_from_parent: dict[str, str] = dict(parent_traj_session_roles or {})
        db_linked_cids: set[str] = set(parent_linked_cids or ())
        db_unlinked_dispatches: list[tuple[str, str]] = []

        if "part" in tables:
            p_cols = [str(r[1]) for r in con.execute("PRAGMA table_info('part')").fetchall()]
            if {"id", "session_id", "data"}.issubset(p_cols):
                has_mid = "message_id" in p_cols
                sel_cols = (
                    "id, message_id, session_id, data" if has_mid else "id, '', session_id, data"
                )
                p_order = (
                    "ORDER BY time_created ASC, id ASC"
                    if "time_created" in p_cols
                    else "ORDER BY id ASC"
                )
                for pid, mid, sid, pdata in con.execute(
                    f"SELECT {sel_cols} FROM part {p_order}"
                ).fetchall():
                    if not isinstance(pdata, str):
                        continue
                    try:
                        pobj = json.loads(pdata)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(pobj, dict):
                        continue
                    sid_s = str(sid)
                    mid_s = str(mid or "")
                    parts_by_session.setdefault(sid_s, []).append((str(pid), mid_s, pobj))

                    ptype = str(pobj.get("type") or "").strip().lower()
                    if ptype in ("tool", "tool_use"):
                        parsed_tp = _parse_opencode_tool_part_dict(pobj)
                        if parsed_tp is not None:
                            cid = parsed_tp["call_id"]
                            merged = _merge_opencode_tool_event(
                                tool_events_by_id.get(cid), parsed_tp
                            )
                            tool_events_by_id[cid] = merged
                            csid = merged.get("child_session_id")
                            drole = merged.get("dispatched_role")
                            if csid:
                                if cid:
                                    db_linked_cids.add(cid)
                                if drole:
                                    child_roles_from_parent[csid] = drole
                            elif drole:
                                db_unlinked_dispatches.append((cid, drole))

        unlinked_parent_dispatches: list[str] = []
        seen_unlinked_cids: set[str] = set()
        if parent_ordered_dispatches is not None:
            for cid, rname in parent_ordered_dispatches:
                if cid and cid in db_linked_cids:
                    continue
                if cid:
                    seen_unlinked_cids.add(cid)
                unlinked_parent_dispatches.append(rname)
            for cid, rname in db_unlinked_dispatches:
                if cid and (cid in db_linked_cids or cid in seen_unlinked_cids):
                    continue
                if cid:
                    seen_unlinked_cids.add(cid)
                unlinked_parent_dispatches.append(rname)
        else:
            for cid, rname in db_unlinked_dispatches:
                if cid and cid in db_linked_cids:
                    continue
                if cid:
                    seen_unlinked_cids.add(cid)
                unlinked_parent_dispatches.append(rname)
            if parent_unlinked_roles:
                unlinked_parent_dispatches.extend(parent_unlinked_roles)

        unlinked_cursor = 0
        built_sessions: list[OpenCodeDBSession] = []
        total_sf = 0
        total_tools = 0

        for srow in sess_dicts:
            sid = str(srow.get("id") or "").strip()
            if not sid:
                continue
            raw_pid = srow.get("parent_id")
            parent_id = (
                str(raw_pid).strip() if isinstance(raw_pid, str) and raw_pid.strip() else None
            )
            title = str(srow.get("title") or "").strip()
            agent_col = str(srow.get("agent") or "").strip()
            sess_model, sess_variant = _parse_opencode_session_model(srow.get("model"))

            if parent_id is None:
                resolved_role = "main"
            else:
                if sid in child_roles_from_parent:
                    resolved_role = child_roles_from_parent[sid]
                elif agent_col and agent_col.lower() not in GENERIC_AGENT_MODES:
                    resolved_role = agent_col
                else:
                    m_title = OPENCODE_TITLE_ROLE_RE.search(title)
                    if m_title and m_title.group(1).strip():
                        resolved_role = m_title.group(1).strip()
                    elif unlinked_cursor < len(unlinked_parent_dispatches):
                        resolved_role = unlinked_parent_dispatches[unlinked_cursor]
                        unlinked_cursor += 1
                    elif agent_col:
                        resolved_role = agent_col
                    else:
                        resolved_role = f"subagent_{sid}"

            s_parts = parts_by_session.get(sid, [])
            turns_list: list[OpenCodeDBTurn] = []
            cur_tools: list[dict[str, Any]] = []
            cur_mid = ""
            n_step_starts = 0
            n_sf = 0
            n_tp = 0

            def _model_for_msg(mid_key: str, default_m: str | None = sess_model) -> str | None:
                m_obj = msg_by_id.get(mid_key)
                if isinstance(m_obj, dict):
                    m_id = str(m_obj.get("modelID") or "").strip()
                    p_id = str(m_obj.get("providerID") or "").strip()
                    if m_id and p_id and "/" not in m_id:
                        return f"{p_id}/{m_id}"
                    if m_id:
                        return m_id
                return default_m

            for _pid, mid_s, pobj in s_parts:
                ptype = str(pobj.get("type") or "").strip().lower()
                if ptype in ("step-start", "step_start"):
                    n_step_starts += 1
                    if cur_tools and cur_mid and mid_s and cur_mid != mid_s:
                        m_fb = msg_by_id.get(cur_mid, {})
                        u_i, c_r, c_w, o_t, r_t, has_t = _parse_tokens_obj(m_fb.get("tokens"))
                        turns_list.append(
                            OpenCodeDBTurn(
                                session_id=sid,
                                message_id=cur_mid,
                                model_name=_model_for_msg(cur_mid),
                                uncached_input_tokens=u_i,
                                cache_read_tokens=c_r,
                                cache_write_tokens=c_w,
                                reply_output_tokens=o_t,
                                reasoning_tokens=r_t,
                                cost_usd=safe_nonneg_float(m_fb.get("cost")),
                                has_token_metrics=has_t,
                                tool_parts=tuple(cur_tools),
                            )
                        )
                        cur_tools = []
                    cur_mid = mid_s or cur_mid
                elif ptype in ("tool", "tool_use"):
                    parsed_tp = _parse_opencode_tool_part_dict(pobj)
                    if parsed_tp is not None:
                        n_tp += 1
                        cur_tools.append(parsed_tp)
                        if not cur_mid:
                            cur_mid = mid_s
                elif ptype in ("step-finish", "step_finish"):
                    n_sf += 1
                    eff_mid = mid_s or cur_mid
                    u_i, c_r, c_w, o_t, r_t, has_t = _parse_tokens_obj(pobj.get("tokens"))
                    cost_v = safe_nonneg_float(pobj.get("cost"))
                    if not has_t and eff_mid in msg_by_id:
                        m_fb = msg_by_id[eff_mid]
                        u_i, c_r, c_w, o_t, r_t, has_t = _parse_tokens_obj(m_fb.get("tokens"))
                        if cost_v is None:
                            cost_v = safe_nonneg_float(m_fb.get("cost"))
                    turns_list.append(
                        OpenCodeDBTurn(
                            session_id=sid,
                            message_id=eff_mid,
                            model_name=_model_for_msg(eff_mid),
                            uncached_input_tokens=u_i,
                            cache_read_tokens=c_r,
                            cache_write_tokens=c_w,
                            reply_output_tokens=o_t,
                            reasoning_tokens=r_t,
                            cost_usd=cost_v,
                            has_token_metrics=has_t or cost_v is not None,
                            tool_parts=tuple(cur_tools),
                        )
                    )
                    cur_tools = []
                    cur_mid = ""

            has_unfinished = bool(cur_tools) or (n_step_starts > n_sf)
            if cur_tools:
                m_fb = msg_by_id.get(cur_mid, {})
                u_i, c_r, c_w, o_t, r_t, has_t = _parse_tokens_obj(m_fb.get("tokens"))
                cost_v = safe_nonneg_float(m_fb.get("cost"))
                turns_list.append(
                    OpenCodeDBTurn(
                        session_id=sid,
                        message_id=cur_mid,
                        model_name=_model_for_msg(cur_mid),
                        uncached_input_tokens=u_i,
                        cache_read_tokens=c_r,
                        cache_write_tokens=c_w,
                        reply_output_tokens=o_t,
                        reasoning_tokens=r_t,
                        cost_usd=cost_v,
                        has_token_metrics=has_t or cost_v is not None,
                        tool_parts=tuple(cur_tools),
                    )
                )

            # Fallback 1: assistant `message` rows when `step-finish` parts were omitted
            if not turns_list and sid in msgs_by_session:
                tools_by_mid: dict[str, list[dict[str, Any]]] = {}
                for _pid, mid_s, pobj in s_parts:
                    if str(pobj.get("type") or "").strip().lower() in ("tool", "tool_use"):
                        ptp = _parse_opencode_tool_part_dict(pobj)
                        if ptp is not None:
                            tools_by_mid.setdefault(mid_s, []).append(ptp)
                for mid_s, mobj in msgs_by_session[sid]:
                    if str(mobj.get("role") or "").strip().lower() != "assistant":
                        continue
                    u_i, c_r, c_w, o_t, r_t, has_t = _parse_tokens_obj(mobj.get("tokens"))
                    cost_v = safe_nonneg_float(mobj.get("cost"))
                    turns_list.append(
                        OpenCodeDBTurn(
                            session_id=sid,
                            message_id=mid_s,
                            model_name=_model_for_msg(mid_s),
                            uncached_input_tokens=u_i,
                            cache_read_tokens=c_r,
                            cache_write_tokens=c_w,
                            reply_output_tokens=o_t,
                            reasoning_tokens=r_t,
                            cost_usd=cost_v,
                            has_token_metrics=has_t or cost_v is not None,
                            tool_parts=tuple(tools_by_mid.get(mid_s, [])),
                        )
                    )

            # Fallback 2: session-level token summary when `part` and `message` rows are absent
            if not turns_list:
                s_in = safe_nonneg_int(srow.get("tokens_input"))
                s_out = safe_nonneg_int(srow.get("tokens_output"))
                s_reas = safe_nonneg_int(srow.get("tokens_reasoning"))
                s_cr = safe_nonneg_int(srow.get("tokens_cache_read"))
                s_cw = safe_nonneg_int(srow.get("tokens_cache_write"))
                s_cost = safe_nonneg_float(srow.get("cost"))
                if (s_in + s_out + s_reas + s_cr + s_cw) > 0 or (s_cost is not None and s_cost > 0):
                    turns_list.append(
                        OpenCodeDBTurn(
                            session_id=sid,
                            message_id=f"{sid}_summary",
                            model_name=sess_model,
                            uncached_input_tokens=s_in,
                            cache_read_tokens=s_cr,
                            cache_write_tokens=s_cw,
                            reply_output_tokens=s_out,
                            reasoning_tokens=s_reas,
                            cost_usd=s_cost,
                            has_token_metrics=True,
                            tool_parts=(),
                        )
                    )

            eff_sf = n_sf if n_sf > 0 else len(turns_list)
            if n_step_starts > n_sf:
                eff_sf = max(eff_sf, n_step_starts)
            total_sf += eff_sf
            total_tools += n_tp
            built_sessions.append(
                OpenCodeDBSession(
                    session_id=sid,
                    parent_id=parent_id,
                    title=title,
                    agent=agent_col,
                    model_name=sess_model,
                    variant=sess_variant,
                    resolved_role=resolved_role,
                    turns=tuple(turns_list),
                    n_step_finish_parts=eff_sf,
                    n_tool_parts=n_tp,
                    has_unfinished_step=has_unfinished,
                )
            )

        return OpenCodeDBSummary(
            sessions=tuple(built_sessions),
            tool_events_by_id=tool_events_by_id,
            child_session_roles_from_parent=child_roles_from_parent,
            total_step_finish_events=total_sf,
            total_tool_events=total_tools,
            has_unfinished_step=any(s.has_unfinished_step for s in built_sessions),
        )
    except (sqlite3.Error, OSError, ValueError):
        return None
    finally:
        if con is not None:
            with contextlib.suppress(Exception):
                con.close()
