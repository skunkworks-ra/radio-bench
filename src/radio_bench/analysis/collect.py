"""Harbor 0.25.0 single-job and multi-job trial collector for radio-bench."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from harbor.models.trial.paths import TrialPaths

from radio_bench.analysis.pricing import compute_trial_pricing, normalize_model_key
from radio_bench.analysis.taxonomy import classify_failure
from radio_bench.grading import PASS_REWARD_MIN

CANONICAL_ARMS: tuple[str, ...] = (
    "00_none",
    "10_tools",
    "01_skills_raw",
    "01_skills_casa",
    "11_full",
    "11_orchestrated",
)

BASE_TASK_META: dict[str, tuple[str, str]] = {
    "s1_inspect": ("synth", "inspect"),
    "s1_precal": ("synth", "precal"),
    "s1_calibrate": ("synth", "calibrate"),
    "s1_image": ("synth", "image"),
    "s2_e2e_synth": ("synth", "e2e"),
    "r1_3c391_inspect": ("3c391", "inspect"),
    "r1_3c391_precal": ("3c391", "precal"),
    "r1_3c391_calibrate": ("3c391", "calibrate"),
    "r1_3c391_image": ("3c391", "image"),
    "r2_3c391_e2e": ("3c391", "e2e"),
}

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

_BASH_TOOL_NAMES: frozenset[str] = frozenset(
    {"bash", "shell", "sh", "execute_bash", "run_command", "terminal", "cmd"}
)

_SKIP_SCAN_DIR_NAMES: frozenset[str] = frozenset(
    {
        "analysis",
        "plots",
        "artifacts",
        "agent",
        "user-agent",
        "verifier",
        "steps",
        "__pycache__",
    }
)

_NONZERO_EXIT_RE = re.compile(
    r"(?:\[exit_code\]|exit\s+code|exit\s+status|returncode)\s*[:=]?\s*(-?[1-9]\d*)\b",
    re.IGNORECASE,
)
_RUN_STEP_RC_RE = re.compile(r"\bstatus=failed\b|\brc=(-?[1-9]\d*)\b")

_PROTOCOL_ERR_PHRASES: tuple[str, ...] = (
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

_CASA_SEVERE_PHRASES: tuple[str, ...] = (
    "severe\t",
    "severe ",
    "RuntimeError:",
    "Traceback (most recent call last):",
)

CSV_BASE_COLUMNS: tuple[str, ...] = (
    "job_dir",
    "trial_name",
    "task_name",
    "base_task",
    "suite",
    "stage",
    "is_e2e",
    "arm",
    "agent_name",
    "model_name",
    "attempt_idx",
    "reward",
    "passed",
    "failure_mode",
    "exception_type",
    "exception_message",
    "total_wall_sec",
    "agent_wall_sec",
    "verifier_wall_sec",
    "n_turns",
    "mcp_calls_total",
    "mcp_calls_inspect",
    "mcp_calls_modify",
    "mcp_calls_create",
    "bash_calls",
    "bash_nonzero_exits",
    "tool_errors",
    "protocol_errors",
    "input_tokens",
    "uncached_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "total_tokens",
    "peak_turn_tokens",
    "haiku_cliff_triggered",
    "cliff_turns",
    "input_usd",
    "cache_write_usd",
    "cache_read_usd",
    "output_usd",
    "total_usd",
    "pricing_source",
)


def _strip_dashes(text: str) -> str:
    return text.replace("\u2014", "--").replace("\u2013", "--")


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (OSError, ValueError):
        return None


def sanitize_json_tree(val: Any) -> Any:
    """Recursively ensure all values are finite Python primitives with zero em/en dashes."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, str):
        return _strip_dashes(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        return fval if math.isfinite(fval) else None
    if isinstance(val, np.ndarray):
        return [sanitize_json_tree(v) for v in val.tolist()]
    if isinstance(val, Path):
        return _strip_dashes(str(val))
    if isinstance(val, dict):
        return {_strip_dashes(str(k)): sanitize_json_tree(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [sanitize_json_tree(v) for v in val]
    return _strip_dashes(str(val))


def _is_trial_directory_with_meta(
    candidate: Path,
    res: dict[str, Any] | None,
    cfg: dict[str, Any] | None,
) -> bool:
    tp = TrialPaths(candidate)
    if res is not None and ("trial_name" in res or "task_name" in res):
        return True
    if tp.reward_json_path.is_file() or tp.reward_text_path.is_file():
        return True
    return bool(cfg is not None and isinstance(cfg.get("task"), dict) and "tasks" not in cfg)


def _is_trial_directory(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    tp = TrialPaths(candidate)
    if tp.reward_json_path.is_file() or tp.reward_text_path.is_file():
        return True
    res = _read_json_dict(tp.result_path)
    if res is not None and ("trial_name" in res or "task_name" in res):
        return True
    cfg = _read_json_dict(tp.config_path)
    return _is_trial_directory_with_meta(candidate, res, cfg)


def discover_trial_dirs(
    input_paths: Sequence[Path | str], max_depth: int = 3
) -> list[tuple[Path, Path]]:
    """Discover `(job_dir, trial_dir)` pairs across one or more Harbor job directories."""
    found: list[tuple[Path, Path]] = []
    seen_resolved: set[Path] = set()

    def _walk(cur: Path, depth: int, current_job_dir: Path) -> None:
        if depth > max_depth or not cur.is_dir():
            return
        res = _read_json_dict(cur / "result.json")
        cfg = _read_json_dict(cur / "config.json")
        if _is_trial_directory_with_meta(cur, res, cfg):
            resolved = cur.resolve()
            if resolved not in seen_resolved:
                seen_resolved.add(resolved)
                job_parent = cur.parent if cur == current_job_dir else current_job_dir
                found.append((job_parent, cur))
            return

        is_job_root = (res is not None and ("trial_results" in res or "stats" in res)) or (
            cfg is not None and isinstance(cfg.get("tasks"), list)
        )
        next_job_dir = cur if is_job_root else current_job_dir

        try:
            children = sorted(cur.iterdir(), key=lambda p: p.name)
        except OSError:
            return

        for child in children:
            if not child.is_dir():
                continue
            if child.name.startswith(".") or child.name in _SKIP_SCAN_DIR_NAMES:
                continue
            _walk(child, depth + 1, next_job_dir)

    for raw_p in input_paths:
        p = Path(raw_p)
        if not p.exists():
            raise FileNotFoundError(f"Job or trial directory does not exist: {p}")
        _walk(p, 0, p)

    found.sort(key=lambda pair: (str(pair[0]), pair[1].name))
    return found


def parse_task_variant(
    task_name: str, metrics_stage: str | None = None
) -> tuple[str, str, str, str, bool]:
    """Parse `task_name` into `(base_task, arm, suite, stage, is_e2e)`."""
    clean = task_name.strip().rstrip("/")
    if "/" in clean:
        clean = clean.rsplit("/", 1)[-1]

    base_task = clean
    arm = "00_none"

    for known_arm in sorted(CANONICAL_ARMS, key=len, reverse=True):
        suffix = f"__{known_arm}"
        if clean.endswith(suffix):
            base_task = clean[: -len(suffix)]
            arm = known_arm
            break
    else:
        if "__" in clean:
            parts = clean.split("__")
            cand_arm = parts[1]
            # Resolve exact or Harbor 32-char truncated arm prefixes (e.g. 01_skills_ca -> 01_skills_casa).
            if cand_arm in CANONICAL_ARMS:
                base_task = parts[0]
                arm = cand_arm
            else:
                prefix_matches = [a for a in CANONICAL_ARMS if a.startswith(cand_arm)]
                if len(prefix_matches) == 1 and "_" in cand_arm:
                    base_task = parts[0]
                    arm = prefix_matches[0]
                else:
                    base_task, arm = parts[0], cand_arm

    if base_task in BASE_TASK_META:
        suite, stage = BASE_TASK_META[base_task]
    else:
        low = base_task.lower()
        if low.startswith(("s1_", "s2_", "synth")) or "synth" in low:
            suite = "synth"
        elif "3c391" in low or low.startswith(("r1_", "r2_")):
            suite = "3c391"
        else:
            suite = "unknown"

        if metrics_stage in {"inspect", "precal", "calibrate", "image", "e2e"}:
            stage = metrics_stage
        elif "e2e" in low:
            stage = "e2e"
        elif "inspect" in low:
            stage = "inspect"
        elif "precal" in low:
            stage = "precal"
        elif "calibrate" in low or low.endswith("_cal"):
            stage = "calibrate"
        elif "image" in low:
            stage = "image"
        else:
            stage = "unknown"

    is_e2e = stage == "e2e"
    return base_task, arm, suite, stage, is_e2e


def _parse_iso_dt(val: Any) -> datetime | None:
    if not isinstance(val, str) or not val.strip():
        return None
    try:
        return datetime.fromisoformat(val.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _timing_duration_sec(timing_obj: Any) -> float | None:
    if not isinstance(timing_obj, dict):
        return None
    st = _parse_iso_dt(timing_obj.get("started_at"))
    ft = _parse_iso_dt(timing_obj.get("finished_at"))
    if st is None or ft is None:
        return None
    return max(0.0, (ft - st).total_seconds())


def _extract_durations(result_dict: dict[str, Any] | None) -> tuple[float, float, float]:
    if not isinstance(result_dict, dict):
        return 0.0, 0.0, 0.0

    agent_sec = _timing_duration_sec(result_dict.get("agent_execution"))
    verifier_sec = _timing_duration_sec(result_dict.get("verifier"))

    if (agent_sec is None or verifier_sec is None) and isinstance(
        result_dict.get("step_results"), list
    ):
        step_agent = 0.0
        step_ver = 0.0
        for sr in result_dict["step_results"]:
            if not isinstance(sr, dict):
                continue
            a_d = _timing_duration_sec(sr.get("agent_execution"))
            v_d = _timing_duration_sec(sr.get("verifier"))
            if a_d is not None:
                step_agent += a_d
            if v_d is not None:
                step_ver += v_d
        if agent_sec is None:
            agent_sec = step_agent
        if verifier_sec is None:
            verifier_sec = step_ver

    total_sec = _timing_duration_sec(result_dict)
    if total_sec is None:
        env_sec = _timing_duration_sec(result_dict.get("environment_setup")) or 0.0
        setup_sec = _timing_duration_sec(result_dict.get("agent_setup")) or 0.0
        total_sec = env_sec + setup_sec + (agent_sec or 0.0) + (verifier_sec or 0.0)

    return (
        round(agent_sec or 0.0, 4),
        round(verifier_sec or 0.0, 4),
        round(total_sec or 0.0, 4),
    )


def _matches_mcp_tool(cand: str, tool: str) -> bool:
    return cand == tool or cand.endswith((f"_{tool}", f"-{tool}", f"/{tool}", f".{tool}"))


def _classify_mcp_server(function_name: str) -> str | None:
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


def _stringify_obs_content(content: Any) -> str:
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


def _extract_exit_code(
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

    m = _NONZERO_EXIT_RE.search(obs_text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    m2 = _RUN_STEP_RC_RE.search(obs_text)
    if m2:
        if m2.group(1):
            try:
                return int(m2.group(1))
            except ValueError:
                return 1
        return 1
    return None


def _is_tool_error(tc_extra: dict[str, Any], obs_extra: dict[str, Any], obs_text: str) -> bool:
    for container in (tc_extra, obs_extra):
        for k in ("is_error", "tool_use_is_error", "tool_result_is_error"):
            if container.get(k) is True:
                return True
        meta = container.get("tool_result_metadata")
        if isinstance(meta, dict) and meta.get("is_error") is True:
            return True

    low = obs_text.lower()
    if "[error] tool reported failure" in low:
        return True
    if '"iserror": true' in low or '"iserror":true' in low:
        return True
    if ('"status": "error"' in low or '"status":"error"' in low) and '"error_type"' in low:
        return True
    return any(phrase in low for phrase in _PROTOCOL_ERR_PHRASES)


def _is_protocol_error(fn_name: str, mcp_srv: str | None, obs_text: str, arm: str) -> bool:
    low_fn = fn_name.strip().lower()
    low_obs = obs_text.lower()

    if low_fn == "invalid":
        return True
    if mcp_srv is not None and arm in {"00_none", "01_skills_raw", "01_skills_casa"}:
        return True
    if mcp_srv == "mcp-other" and low_fn.startswith("ms_") and low_fn not in ALL_KNOWN_MCP_TOOLS:
        return True
    return any(phrase in low_obs for phrase in _PROTOCOL_ERR_PHRASES)


def _parse_opencode_tool_events(opencode_txt_path: Path | None) -> dict[str, dict[str, Any]]:
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
        cid = str(part.get("callID") or part.get("id") or "").strip()
        if not cid:
            continue
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        meta = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
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
        out_str = _stringify_obs_content(state.get("output"))

        prev = events_by_id.get(cid)
        events_by_id[cid] = {
            "tool": str(part.get("tool") or (prev["tool"] if prev else "")),
            "status": status_str or (prev["status"] if prev else ""),
            "error": err_str or (prev["error"] if prev else ""),
            "exit": exit_code if exit_code is not None else (prev["exit"] if prev else None),
            "output": out_str or (prev["output"] if prev else ""),
        }
    return events_by_id


def extract_trajectory_telemetry(
    traj_dict: dict[str, Any] | None,
    arm: str = "00_none",
    *,
    opencode_txt_path: Path | None = None,
) -> dict[str, int]:
    """Extract turn count, MCP server call counts, bash calls, and error counts from ATIF JSON."""
    counts = {
        "n_turns": 0,
        "mcp_calls_total": 0,
        "mcp_calls_inspect": 0,
        "mcp_calls_modify": 0,
        "mcp_calls_create": 0,
        "bash_calls": 0,
        "bash_nonzero_exits": 0,
        "tool_errors": 0,
        "protocol_errors": 0,
        "severe_casa_errors": 0,
    }
    if not isinstance(traj_dict, dict):
        return counts

    has_opencode_txt = opencode_txt_path is not None and opencode_txt_path.is_file()
    oc_events = _parse_opencode_tool_events(opencode_txt_path) if has_opencode_txt else {}

    def _walk_traj(t_obj: dict[str, Any], parent_agent: str = "") -> None:
        agent_blk = t_obj.get("agent")
        t_agent = (
            str(agent_blk.get("name") or "").strip().lower() if isinstance(agent_blk, dict) else ""
        ) or parent_agent
        is_opencode = t_agent == "opencode"

        steps = t_obj.get("steps")
        if isinstance(steps, list):
            n_steps = len(steps)
            for step_pos, step in enumerate(steps):
                if not isinstance(step, dict) or step.get("source") != "agent":
                    continue
                counts["n_turns"] += 1
                has_subsequent_steps = step_pos < n_steps - 1

                obs_by_call_id: dict[str, dict[str, Any]] = {}
                unkeyed_obs: list[dict[str, Any]] = []
                obs_obj = step.get("observation")
                if isinstance(obs_obj, dict) and isinstance(obs_obj.get("results"), list):
                    for res_item in obs_obj["results"]:
                        if not isinstance(res_item, dict):
                            continue
                        scid = res_item.get("source_call_id")
                        if isinstance(scid, str) and scid:
                            obs_by_call_id[scid] = res_item
                        else:
                            unkeyed_obs.append(res_item)

                tool_calls = step.get("tool_calls")
                if isinstance(tool_calls, list):
                    for idx, tc in enumerate(tool_calls):
                        if not isinstance(tc, dict):
                            continue
                        fn_name = str(tc.get("function_name") or "")
                        call_id = str(tc.get("tool_call_id") or "")
                        tc_extra = tc.get("extra") if isinstance(tc.get("extra"), dict) else {}

                        obs_item = obs_by_call_id.get(call_id) if call_id else None
                        if obs_item is None and not call_id and idx < len(unkeyed_obs):
                            obs_item = unkeyed_obs[idx]
                        obs_extra = (
                            obs_item.get("extra")
                            if isinstance(obs_item, dict)
                            and isinstance(obs_item.get("extra"), dict)
                            else {}
                        )
                        obs_text = (
                            _stringify_obs_content(obs_item.get("content"))
                            if isinstance(obs_item, dict)
                            else ""
                        )

                        oc_ev = oc_events.get(call_id) if call_id else None
                        eff_fn_name = fn_name or (str(oc_ev["tool"]) if oc_ev else "")
                        text_parts = [p for p in (obs_text,) if p]
                        if oc_ev is not None:
                            if oc_ev["output"] and oc_ev["output"] != obs_text:
                                text_parts.append(str(oc_ev["output"]))
                            if oc_ev["error"]:
                                text_parts.append(str(oc_ev["error"]))
                        combined_obs_text = "\n".join(text_parts)

                        oc_failed = bool(
                            oc_ev is not None
                            and (oc_ev["status"] == "error" or bool(oc_ev["error"]))
                        )
                        missing_oc_obs_err = bool(
                            is_opencode
                            and not has_opencode_txt
                            and bool(call_id)
                            and obs_item is None
                            and has_subsequent_steps
                        )

                        mcp_srv = _classify_mcp_server(eff_fn_name)
                        if mcp_srv is not None:
                            counts["mcp_calls_total"] += 1
                            if mcp_srv == "ms-inspect":
                                counts["mcp_calls_inspect"] += 1
                            elif mcp_srv == "ms-modify":
                                counts["mcp_calls_modify"] += 1
                            elif mcp_srv == "ms-create":
                                counts["mcp_calls_create"] += 1

                        low_fn = eff_fn_name.strip().lower()
                        is_bash = low_fn in _BASH_TOOL_NAMES
                        if is_bash:
                            counts["bash_calls"] += 1
                            ec = _extract_exit_code(tc_extra, obs_extra, combined_obs_text)
                            if ec is None and oc_ev is not None and oc_ev["exit"] is not None:
                                ec = int(oc_ev["exit"])
                            if (
                                (ec is not None and ec != 0)
                                or oc_failed
                                or missing_oc_obs_err
                                or _is_tool_error(tc_extra, obs_extra, combined_obs_text)
                            ):
                                counts["bash_nonzero_exits"] += 1

                        if any(
                            p.lower() in combined_obs_text.lower() for p in _CASA_SEVERE_PHRASES
                        ):
                            counts["severe_casa_errors"] += 1

                        is_proto = _is_protocol_error(eff_fn_name, mcp_srv, combined_obs_text, arm)
                        if is_proto:
                            counts["protocol_errors"] += 1

                        if (
                            is_proto
                            or oc_failed
                            or missing_oc_obs_err
                            or _is_tool_error(tc_extra, obs_extra, combined_obs_text)
                        ):
                            counts["tool_errors"] += 1

        sub_trajs = t_obj.get("subagent_trajectories")
        if isinstance(sub_trajs, list):
            for sub in sub_trajs:
                if isinstance(sub, dict):
                    _walk_traj(sub, t_agent)

    _walk_traj(traj_dict)
    return counts


def _locate_trial_files(tp: TrialPaths) -> tuple[Path, Path, Path, Path | None]:
    """Return `(reward_json, metrics_json, trajectory_json, opencode_txt)` supporting `steps/`."""
    reward_p = tp.reward_json_path
    metrics_p = tp.verifier_dir / "metrics.json"
    traj_p = tp.agent_dir / "trajectory.json"
    oc_p: Path | None = (
        tp.agent_dir / "opencode.txt" if (tp.agent_dir / "opencode.txt").is_file() else None
    )

    if (not reward_p.is_file() or not traj_p.is_file() or oc_p is None) and tp.steps_dir.is_dir():
        try:
            step_dirs = sorted(p for p in tp.steps_dir.iterdir() if p.is_dir())
        except OSError:
            step_dirs = []
        if step_dirs:
            last_step = step_dirs[-1]
            if not reward_p.is_file() and (last_step / "verifier" / "reward.json").is_file():
                reward_p = last_step / "verifier" / "reward.json"
            if not metrics_p.is_file() and (last_step / "verifier" / "metrics.json").is_file():
                metrics_p = last_step / "verifier" / "metrics.json"
            if not traj_p.is_file() and (last_step / "agent" / "trajectory.json").is_file():
                traj_p = last_step / "agent" / "trajectory.json"
            if oc_p is None and (last_step / "agent" / "opencode.txt").is_file():
                oc_p = last_step / "agent" / "opencode.txt"

    return reward_p, metrics_p, traj_p, oc_p


def _resolve_task_name(
    trial_dir: Path,
    result_dict: dict[str, Any] | None,
    config_dict: dict[str, Any] | None,
) -> str:
    if isinstance(result_dict, dict):
        tn = result_dict.get("task_name")
        if isinstance(tn, str) and tn.strip():
            return tn.strip()
        rcfg = result_dict.get("config")
        if isinstance(rcfg, dict) and isinstance(rcfg.get("task"), dict):
            tpath = rcfg["task"].get("path")
            if isinstance(tpath, str) and tpath.strip():
                return Path(tpath.strip()).name

    if isinstance(config_dict, dict) and isinstance(config_dict.get("task"), dict):
        tpath = config_dict["task"].get("path")
        if isinstance(tpath, str) and tpath.strip():
            return Path(tpath.strip()).name

    return trial_dir.name


def _resolve_agent_and_model(
    result_dict: dict[str, Any] | None,
    config_dict: dict[str, Any] | None,
    trajectory_dict: dict[str, Any] | None,
) -> tuple[str, str]:
    agent_name = ""
    raw_model = ""

    if isinstance(result_dict, dict):
        ainfo = result_dict.get("agent_info")
        if isinstance(ainfo, dict):
            if isinstance(ainfo.get("name"), str):
                agent_name = ainfo["name"].strip()
            minfo = ainfo.get("model_info")
            if isinstance(minfo, dict) and isinstance(minfo.get("name"), str):
                m_name = minfo["name"].strip()
                m_prov = minfo.get("provider")
                if isinstance(m_prov, str) and m_prov.strip() and "/" not in m_name:
                    raw_model = f"{m_prov.strip()}/{m_name}"
                else:
                    raw_model = m_name

        rcfg = result_dict.get("config")
        if isinstance(rcfg, dict) and isinstance(rcfg.get("agent"), dict):
            acfg = rcfg["agent"]
            if not agent_name and isinstance(acfg.get("name"), str):
                agent_name = acfg["name"].strip()
            if not raw_model and isinstance(acfg.get("model_name"), str):
                raw_model = acfg["model_name"].strip()

    if isinstance(config_dict, dict) and isinstance(config_dict.get("agent"), dict):
        acfg = config_dict["agent"]
        if not agent_name and isinstance(acfg.get("name"), str):
            agent_name = acfg["name"].strip()
        if not raw_model and isinstance(acfg.get("model_name"), str):
            raw_model = acfg["model_name"].strip()

    if isinstance(trajectory_dict, dict) and isinstance(trajectory_dict.get("agent"), dict):
        tagent = trajectory_dict["agent"]
        if not agent_name and isinstance(tagent.get("name"), str):
            agent_name = tagent["name"].strip()
        if not raw_model and isinstance(tagent.get("model_name"), str):
            raw_model = tagent["model_name"].strip()

    if not agent_name:
        agent_name = "unknown"

    canonical_model = normalize_model_key(raw_model, agent_name=agent_name)
    if not canonical_model:
        canonical_model = "oracle" if agent_name.lower() == "oracle" else (raw_model or "unknown")

    return agent_name, canonical_model


def _extract_exception(
    tp: TrialPaths,
    result_dict: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if isinstance(result_dict, dict):
        exc = result_dict.get("exception_info")
        if not isinstance(exc, dict) and isinstance(result_dict.get("step_results"), list):
            for sr in result_dict["step_results"]:
                if isinstance(sr, dict) and isinstance(sr.get("exception_info"), dict):
                    exc = sr["exception_info"]
                    break
        if isinstance(exc, dict):
            etype = str(exc.get("exception_type") or "").strip() or None
            emsg = str(exc.get("exception_message") or "").strip() or None
            return etype, emsg

    if tp.exception_message_path.is_file():
        try:
            txt = tp.exception_message_path.read_text(encoding="utf-8").strip()
            if txt:
                first_line = txt.splitlines()[0][:300]
                return "TrialException", first_line
        except OSError:
            pass
    return None, None


def collect_single_trial(job_dir: Path, trial_dir: Path) -> dict[str, Any]:
    """Parse a single Harbor 0.25.0 `trial_dir` into a normalized trial record."""
    tp = TrialPaths(trial_dir)
    result_dict = _read_json_dict(tp.result_path)
    config_dict = _read_json_dict(tp.config_path)
    reward_p, metrics_p, traj_p, oc_p = _locate_trial_files(tp)

    reward_dict = _read_json_dict(reward_p)
    if reward_dict is None and isinstance(result_dict, dict):
        vr = result_dict.get("verifier_result")
        if isinstance(vr, dict) and isinstance(vr.get("rewards"), dict):
            reward_dict = dict(vr["rewards"])
    if reward_dict is None and tp.reward_text_path.is_file():
        try:
            r_val = float(tp.reward_text_path.read_text(encoding="utf-8").strip())
            if math.isfinite(r_val):
                reward_dict = {
                    "reward": r_val,
                    "passed": 1 if r_val >= PASS_REWARD_MIN else 0,
                }
        except (OSError, ValueError):
            reward_dict = None

    metrics_dict = _read_json_dict(metrics_p)
    trajectory_dict = _read_json_dict(traj_p)

    task_name = _resolve_task_name(trial_dir, result_dict, config_dict)
    metrics_stage = (
        str(metrics_dict.get("stage"))
        if isinstance(metrics_dict, dict) and isinstance(metrics_dict.get("stage"), str)
        else None
    )
    base_task, arm, suite, stage, is_e2e = parse_task_variant(
        task_name, metrics_stage=metrics_stage
    )

    trial_name = (
        str(result_dict.get("trial_name")).strip()
        if isinstance(result_dict, dict) and isinstance(result_dict.get("trial_name"), str)
        else trial_dir.name
    )
    started_at_str = (
        str(result_dict.get("started_at"))
        if isinstance(result_dict, dict) and isinstance(result_dict.get("started_at"), str)
        else ""
    )

    agent_name, model_name = _resolve_agent_and_model(result_dict, config_dict, trajectory_dict)
    exception_type, exception_message = _extract_exception(tp, result_dict)
    agent_wall_sec, verifier_wall_sec, total_wall_sec = _extract_durations(result_dict)

    raw_reward = 0.0
    passed = 0
    if isinstance(reward_dict, dict) and "reward" in reward_dict:
        rv = reward_dict.get("reward")
        if isinstance(rv, (int, float)) and not isinstance(rv, bool) and math.isfinite(float(rv)):
            raw_reward = round(min(1.0, max(0.0, float(rv))), 6)
        pv = reward_dict.get("passed")
        if isinstance(pv, (int, float)) and not isinstance(pv, bool) and math.isfinite(float(pv)):
            passed = 1 if int(round(float(pv))) != 0 else 0
        else:
            passed = 1 if raw_reward >= PASS_REWARD_MIN else 0

    subscores: dict[str, float] = {}
    if isinstance(metrics_dict, dict) and isinstance(metrics_dict.get("subscores"), dict):
        for k, v in metrics_dict["subscores"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                subscores[str(k)] = round(float(v), 6)
    if isinstance(reward_dict, dict):
        for k, v in reward_dict.items():
            if k in ("reward", "passed"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                subscores[str(k)] = round(float(v), 6)

    telemetry = extract_trajectory_telemetry(trajectory_dict, arm=arm, opencode_txt_path=oc_p)
    pricing = compute_trial_pricing(
        model_name=model_name,
        agent_name=agent_name,
        trajectory_dict=trajectory_dict,
        trial_result_dict=result_dict,
    )

    failure_mode = classify_failure(
        passed=passed,
        stage=stage,
        arm=arm,
        reward=raw_reward,
        reward_dict=reward_dict,
        metrics_dict=metrics_dict,
        exception_type=exception_type,
        exception_message=exception_message,
        tool_errors=telemetry["tool_errors"],
        protocol_errors=telemetry["protocol_errors"],
        mcp_calls_total=telemetry["mcp_calls_total"],
        bash_nonzero_exits=telemetry["bash_nonzero_exits"],
        severe_casa_errors=telemetry["severe_casa_errors"],
    )

    truth_block: dict[str, Any] = {}
    measured_block: dict[str, Any] = {}
    if isinstance(metrics_dict, dict):
        if stage == "e2e":
            cal_b = metrics_dict.get("calibration")
            img_b = metrics_dict.get("image")
            truth_block = sanitize_json_tree(
                {
                    "calibration": cal_b.get("truth") if isinstance(cal_b, dict) else {},
                    "image": img_b.get("truth") if isinstance(img_b, dict) else {},
                }
            )
            measured_block = sanitize_json_tree(
                {
                    "calibration": cal_b.get("measured") if isinstance(cal_b, dict) else {},
                    "image": img_b.get("measured") if isinstance(img_b, dict) else {},
                }
            )
        else:
            if isinstance(metrics_dict.get("truth"), dict):
                truth_block = sanitize_json_tree(metrics_dict["truth"])
            if isinstance(metrics_dict.get("measured"), dict):
                measured_block = sanitize_json_tree(metrics_dict["measured"])

    return sanitize_json_tree(
        {
            "job_dir": str(job_dir),
            "trial_dir": str(trial_dir),
            "trial_name": trial_name,
            "started_at": started_at_str,
            "task_name": task_name,
            "base_task": base_task,
            "suite": suite,
            "stage": stage,
            "is_e2e": is_e2e,
            "arm": arm,
            "agent_name": agent_name,
            "model_name": model_name,
            "attempt_idx": 0,
            "reward": raw_reward,
            "passed": int(passed),
            "failure_mode": failure_mode,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "total_wall_sec": total_wall_sec,
            "agent_wall_sec": agent_wall_sec,
            "verifier_wall_sec": verifier_wall_sec,
            "n_turns": telemetry["n_turns"],
            "mcp_calls_total": telemetry["mcp_calls_total"],
            "mcp_calls_inspect": telemetry["mcp_calls_inspect"],
            "mcp_calls_modify": telemetry["mcp_calls_modify"],
            "mcp_calls_create": telemetry["mcp_calls_create"],
            "bash_calls": telemetry["bash_calls"],
            "bash_nonzero_exits": telemetry["bash_nonzero_exits"],
            "tool_errors": telemetry["tool_errors"],
            "protocol_errors": telemetry["protocol_errors"],
            **pricing.to_dict(),
            "subscores": subscores,
            "truth": truth_block,
            "measured": measured_block,
        }
    )


def _atomic_write_text(target_path: Path, text: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".{target_path.name}.{os.getpid()}.tmp"
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _is_csv_scalar(val: Any) -> bool:
    return isinstance(val, (str, int, float, bool))


def _extract_scalar_measured_map(rec: dict[str, Any]) -> dict[str, Any]:
    meas = rec.get("measured")
    if not isinstance(meas, dict):
        return {}
    out: dict[str, Any] = {}
    is_e2e_rec = bool(rec.get("is_e2e")) or str(rec.get("stage")) == "e2e"
    if is_e2e_rec:
        cal_m = meas.get("calibration")
        if isinstance(cal_m, dict):
            for k, v in cal_m.items():
                if _is_csv_scalar(v):
                    out[f"measured_cal_{k}"] = v
        img_m = meas.get("image")
        if isinstance(img_m, dict):
            for k, v in img_m.items():
                if _is_csv_scalar(v):
                    out[f"measured_img_{k}"] = v
        for k, v in meas.items():
            if k not in ("calibration", "image") and _is_csv_scalar(v):
                out[f"measured_{k}"] = v
    else:
        for k, v in meas.items():
            if _is_csv_scalar(v):
                out[f"measured_{k}"] = v
    return out


def write_trials_outputs(records: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Atomically write `<output_dir>/trials.jsonl` and `<output_dir>/trials.csv` (`allow_nan=False`)."""
    records = [sanitize_json_tree(r) for r in records]
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trials.jsonl"
    csv_path = output_dir / "trials.csv"

    jsonl_lines = [json.dumps(rec, allow_nan=False, sort_keys=False) for rec in records]
    _atomic_write_text(jsonl_path, ("\n".join(jsonl_lines) + "\n") if jsonl_lines else "")

    subscore_keys = sorted(
        {f"subscore_{k}" for rec in records for k in (rec.get("subscores") or {})}
    )
    per_rec_measured = [_extract_scalar_measured_map(rec) for rec in records]
    measured_keys = sorted({mk for m_map in per_rec_measured for mk in m_map})
    fieldnames = [*CSV_BASE_COLUMNS, *subscore_keys, *measured_keys]

    tmp_csv = output_dir / f".trials.csv.{os.getpid()}.tmp"
    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for rec, meas_map in zip(records, per_rec_measured, strict=True):
                row: dict[str, Any] = {}
                for col in CSV_BASE_COLUMNS:
                    val = rec.get(col)
                    row[col] = "" if val is None else val
                subs = rec.get("subscores") or {}
                for sk in subscore_keys:
                    raw_k = sk[len("subscore_") :]
                    sval = subs.get(raw_k)
                    row[sk] = "" if sval is None else sval
                for mk in measured_keys:
                    mval = meas_map.get(mk)
                    row[mk] = "" if mval is None else mval
                writer.writerow(row)
        os.replace(tmp_csv, csv_path)
    finally:
        if tmp_csv.exists():
            tmp_csv.unlink(missing_ok=True)

    return csv_path, jsonl_path


def _extract_regrade_provenance(trial_dir: Path) -> dict[str, Any]:
    tp = TrialPaths(trial_dir)
    cfg = _read_json_dict(tp.config_path) or {}
    res = _read_json_dict(tp.result_path) or {}
    src = cfg.get("source_trial")
    if not isinstance(src, dict) and isinstance(res.get("config"), dict):
        src = res["config"].get("source_trial")
    src_d = src if isinstance(src, dict) else None

    src_path_resolved: Path | None = None
    src_path_name = ""
    src_trial_id = ""
    if src_d is not None:
        raw_sp = src_d.get("path") or src_d.get("trial_dir")
        if isinstance(raw_sp, str) and raw_sp.strip():
            sp = Path(raw_sp.strip())
            src_path_name = sp.name
            try:
                src_path_resolved = sp.resolve()
            except OSError:
                src_path_resolved = sp
        if not src_path_name:
            raw_tname = src_d.get("trial_name")
            if isinstance(raw_tname, str) and raw_tname.strip():
                src_path_name = raw_tname.strip()
        raw_sid = src_d.get("trial_id")
        if raw_sid is not None and str(raw_sid).strip():
            src_trial_id = str(raw_sid).strip()

    ver_obj = res.get("verifier") if isinstance(res.get("verifier"), dict) else {}
    ver_ft = _parse_iso_dt(ver_obj.get("finished_at"))
    res_ft = _parse_iso_dt(res.get("finished_at"))
    res_st = _parse_iso_dt(res.get("started_at"))
    ts = ver_ft or res_ft or res_st
    epoch = ts.timestamp() if ts is not None else 0.0

    try:
        resolved_self = trial_dir.resolve()
    except OSError:
        resolved_self = trial_dir

    return {
        "self_resolved": resolved_self,
        "self_dir_name": trial_dir.name,
        "trial_id": str(res.get("id") or "").strip(),
        "is_regrade": src_d is not None,
        "src_path_resolved": src_path_resolved,
        "src_path_name": src_path_name,
        "src_trial_id": src_trial_id,
        "epoch": epoch,
    }


def _deduplicate_regraded_trials(
    entries: list[tuple[int, dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Drop original trials superseded by Harbor `regrade` (`source_trial`) or duplicate `trial_name`."""
    n = len(entries)
    superseded_indices: set[int] = set()

    # 1. Explicit config.json `source_trial` supersession.
    for i in range(n):
        _ord_i, _rec_i, prov_i = entries[i]
        if not prov_i["is_regrade"]:
            continue
        sp_res = prov_i["src_path_resolved"]
        sp_name = prov_i["src_path_name"]
        s_tid = prov_i["src_trial_id"]
        for j in range(n):
            if i == j:
                continue
            _ord_j, rec_j, prov_j = entries[j]
            if (
                (sp_res is not None and prov_j["self_resolved"] == sp_res)
                or (
                    sp_name
                    and (
                        prov_j["self_dir_name"] == sp_name
                        or str(rec_j.get("trial_name")) == sp_name
                    )
                )
                or (s_tid and prov_j["trial_id"] and prov_j["trial_id"] == s_tid)
            ):
                superseded_indices.add(j)

    surviving = [entries[idx] for idx in range(n) if idx not in superseded_indices]

    # 2. Deduplicate identical trial_name across job directories (keep newest / regrade).
    by_trial_key: dict[
        tuple[str, str, str, str, str, str], tuple[int, dict[str, Any], dict[str, Any]]
    ] = {}
    for ord_idx, rec, prov in surviving:
        t_name = str(rec.get("trial_name") or "")
        key = (
            str(rec.get("suite")),
            str(rec.get("base_task")),
            str(rec.get("arm")),
            str(rec.get("agent_name")),
            str(rec.get("model_name")),
            t_name,
        )
        existing = by_trial_key.get(key)
        if existing is None:
            by_trial_key[key] = (ord_idx, rec, prov)
        else:
            ex_ord, _ex_rec, ex_prov = existing
            cand_rank = (1 if prov["is_regrade"] else 0, prov["epoch"], ord_idx)
            ex_rank = (1 if ex_prov["is_regrade"] else 0, ex_prov["epoch"], ex_ord)
            if cand_rank >= ex_rank:
                by_trial_key[key] = (ord_idx, rec, prov)

    return [item[1] for item in by_trial_key.values()]


def collect_trials(
    job_dirs: Sequence[Path | str],
    *,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect all Harbor trials from `job_dirs`, deduplicate regrades, and assign `attempt_idx`."""
    job_order_map: dict[Path, int] = {}
    for idx, raw_j in enumerate(job_dirs):
        p = Path(raw_j)
        try:
            job_order_map[p.resolve()] = idx
        except OSError:
            job_order_map[p] = idx

    discovered = discover_trial_dirs(job_dirs)
    entries: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for disc_idx, (j_dir, t_dir) in enumerate(discovered):
        rec = collect_single_trial(j_dir, t_dir)
        prov = _extract_regrade_provenance(t_dir)
        try:
            j_res = j_dir.resolve()
        except OSError:
            j_res = j_dir
        ord_rank = job_order_map.get(j_res, disc_idx) * 100_000 + disc_idx
        entries.append((ord_rank, rec, prov))

    records = _deduplicate_regraded_trials(entries)

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (
            str(rec["suite"]),
            str(rec["base_task"]),
            str(rec["arm"]),
            str(rec["agent_name"]),
            str(rec["model_name"]),
        )
        grouped.setdefault(key, []).append(rec)

    ordered_records: list[dict[str, Any]] = []
    for key in sorted(grouped):
        cell_list = grouped[key]
        cell_list.sort(
            key=lambda r: (
                str(r.get("started_at") or ""),
                str(r.get("trial_name") or ""),
                str(r.get("trial_dir") or ""),
            )
        )
        for idx, r in enumerate(cell_list):
            r["attempt_idx"] = idx
            ordered_records.append(r)

    if output_dir is not None:
        write_trials_outputs(ordered_records, Path(output_dir))

    return ordered_records


collect_jobs = collect_trials
