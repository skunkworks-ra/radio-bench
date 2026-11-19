"""Harbor 0.25.0 single-job and multi-job trial collector for radio-bench."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from harbor.models.trial.paths import TrialPaths
except ImportError:

    class TrialPaths:  # type: ignore[no-redef]
        """Minimal Harbor `0.25.0`-compatible `TrialPaths` fallback for isolated containers."""

        def __init__(self, trial_dir: Path) -> None:
            self.trial_dir = Path(trial_dir)

        @property
        def config_path(self) -> Path:
            return self.trial_dir / "config.json"

        @property
        def result_path(self) -> Path:
            return self.trial_dir / "result.json"

        @property
        def verifier_dir(self) -> Path:
            return self.trial_dir / "verifier"

        @property
        def agent_dir(self) -> Path:
            return self.trial_dir / "agent"

        @property
        def steps_dir(self) -> Path:
            return self.trial_dir / "steps"

        @property
        def reward_json_path(self) -> Path:
            return self.verifier_dir / "reward.json"

        @property
        def reward_text_path(self) -> Path:
            return self.verifier_dir / "reward.txt"

        @property
        def exception_message_path(self) -> Path:
            return self.trial_dir / "exception.txt"


from radio_bench.analysis._trace import (
    classify_mcp_server,
    extract_agent_runtime_diagnostics,
    safe_nonneg_int,
)
from radio_bench.analysis.pricing import (
    TrialExecutionAnalysis,
    analyze_trial_execution,
    build_trial_pricing_from_execution,
    normalize_model_key,
)
from radio_bench.analysis.taxonomy import classify_failure
from radio_bench.config import reject_slash_tmp
from radio_bench.grading import PASS_REWARD_MIN

_classify_mcp_server = classify_mcp_server


class TrialRecord(dict[str, Any]):
    """Dict-backed trial record with default runtime health & provider diagnostics fields."""

    def __init__(self, mapping: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if mapping is not None:
            self.update(mapping)
        if kwargs:
            self.update(kwargs)
        self.setdefault("agent_exit_status", "ok")
        self.setdefault("decoder_collapse", False)
        self.setdefault("decoder_collapse_detail", None)
        self.setdefault("output_token_limit_hit", False)
        self.setdefault("last_step_tokens", None)
        self.setdefault("provider_error_kind", None)
        self.setdefault("provider_status_code", None)
        self.setdefault("provider_error_count", 0)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


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

CSV_BASE_COLUMNS: tuple[str, ...] = (
    "job_dir",
    "trial_name",
    "task_name",
    "base_task",
    "pack_id",
    "suite",
    "stage",
    "is_e2e",
    "arm",
    "agent_name",
    "model_name",
    "thinking_level",
    "attempt_idx",
    "reward",
    "passed",
    "failure_mode",
    "exception_type",
    "exception_message",
    "agent_exit_status",
    "decoder_collapse",
    "decoder_collapse_detail",
    "output_token_limit_hit",
    "last_step_tokens",
    "provider_error_kind",
    "provider_status_code",
    "provider_error_count",
    "total_wall_sec",
    "agent_wall_sec",
    "verifier_wall_sec",
    "n_turns",
    "retry_turn_count",
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
    "reasoning_tokens",
    "reply_output_tokens",
    "total_tokens",
    "peak_turn_tokens",
    "haiku_cliff_triggered",
    "cliff_turns",
    "input_usd",
    "cache_write_usd",
    "cache_read_usd",
    "output_usd",
    "first_pass_usd",
    "retry_usd",
    "total_usd",
    "pricing_source",
    "trace_events_expected",
    "trace_events_observed",
    "trace_completeness_ok",
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


def _match_canonical_or_truncated_arm(token: str) -> str | None:
    """Resolve an exact or Harbor 32-char truncated arm token to a canonical arm name."""
    if token in CANONICAL_ARMS:
        return token
    if "_" in token or token in {"00", "10"}:
        prefix_matches = [a for a in CANONICAL_ARMS if a.startswith(token)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
    return None


def parse_task_variant_with_pack(
    task_name: str, metrics_stage: str | None = None
) -> tuple[str, str | None, str, str, str, bool]:
    """Parse `task_name` into `(base_task, pack_id, arm, suite, stage, is_e2e)`."""
    clean = task_name.strip().rstrip("/")
    if "/" in clean:
        clean = clean.rsplit("/", 1)[-1]

    base_task = clean
    pack_id: str | None = None
    arm = "00_none"

    if "__" in clean:
        parts = clean.split("__")
        if len(parts) == 2:
            base_task = parts[0]
            matched_p1 = _match_canonical_or_truncated_arm(parts[1])
            if matched_p1 is not None:
                arm = matched_p1
            elif re.fullmatch(r"[0-9a-f]{6,12}", parts[1]):
                arm = "00_none"
            else:
                arm = parts[1]
        elif len(parts) == 3:
            matched_p2 = _match_canonical_or_truncated_arm(parts[2])
            matched_p1 = _match_canonical_or_truncated_arm(parts[1])
            if matched_p2 is not None:
                base_task = parts[0]
                pack_id = parts[1]
                arm = matched_p2
            elif matched_p1 is not None:
                base_task = parts[0]
                pack_id = None
                arm = matched_p1
            else:
                base_task = parts[0]
                pack_id = parts[1]
                arm = "00_none"
        else:
            matched_p2 = _match_canonical_or_truncated_arm(parts[2])
            matched_last = _match_canonical_or_truncated_arm(parts[-1])
            matched_p1 = _match_canonical_or_truncated_arm(parts[1])
            if matched_p2 is not None:
                base_task = parts[0]
                pack_id = parts[1]
                arm = matched_p2
            elif matched_last is not None:
                base_task = parts[0]
                pack_id = "__".join(parts[1:-1])
                arm = matched_last
            elif matched_p1 is not None:
                base_task = parts[0]
                pack_id = None
                arm = matched_p1
            else:
                base_task = parts[0]
                pack_id = parts[1]
                arm = "00_none"

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

        norm_ms = metrics_stage.strip().lower() if isinstance(metrics_stage, str) else None
        if norm_ms == "image_cont":
            norm_ms = "image"
        canonical_stages = {
            "inspect",
            "precal",
            "calibrate",
            "postcal",
            "polcal",
            "selfcal",
            "image",
            "image_cube",
            "simulate",
            "report",
            "e2e",
        }
        if norm_ms in canonical_stages:
            stage = norm_ms
        elif "e2e" in low:
            stage = "e2e"
        elif "inspect" in low:
            stage = "inspect"
        elif "postcal" in low:
            stage = "postcal"
        elif "polcal" in low:
            stage = "polcal"
        elif "selfcal" in low:
            stage = "selfcal"
        elif "precal" in low:
            stage = "precal"
        elif "calibrate" in low or low.endswith("_cal"):
            stage = "calibrate"
        elif "cube" in low:
            stage = "image_cube"
        elif "image" in low:
            stage = "image"
        elif "simulat" in low:
            stage = "simulate"
        elif "report" in low:
            stage = "report"
        else:
            stage = "unknown"

    is_e2e = stage == "e2e"
    return base_task, pack_id, arm, suite, stage, is_e2e


def parse_task_variant(
    task_name: str, metrics_stage: str | None = None
) -> tuple[str, str, str, str, bool]:
    """Parse `task_name` into `(base_task, arm, suite, stage, is_e2e)`."""
    base_task, _pack_id, arm, suite, stage, is_e2e = parse_task_variant_with_pack(
        task_name, metrics_stage=metrics_stage
    )
    return base_task, arm, suite, stage, is_e2e


def _format_thinking_budget(val: Any) -> str | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if not math.isfinite(float(val)):
            return None
        v_int = int(round(float(val)))
        return "off" if v_int <= 0 else f"budget_{v_int}"
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            v_int = int(s)
            return "off" if v_int <= 0 else f"budget_{v_int}"
        except ValueError:
            return s
    return None


def extract_thinking_level(
    config_dict: dict[str, Any] | None,
    result_dict: dict[str, Any] | None,
    trajectory_dict: dict[str, Any] | None = None,
) -> str:
    """Deterministically extract `thinking_level` across `config.json`, `result.json`, and ATIF."""
    agent_cfgs: list[dict[str, Any]] = []
    if isinstance(config_dict, dict) and isinstance(config_dict.get("agent"), dict):
        agent_cfgs.append(config_dict["agent"])
    if isinstance(result_dict, dict):
        rcfg = result_dict.get("config")
        if isinstance(rcfg, dict) and isinstance(rcfg.get("agent"), dict):
            agent_cfgs.append(rcfg["agent"])

    # 1. kwargs.reasoning_effort
    for acfg in agent_cfgs:
        kw = acfg.get("kwargs")
        if isinstance(kw, dict):
            re_val = kw.get("reasoning_effort")
            if isinstance(re_val, str) and re_val.strip():
                return re_val.strip()

    # 2. kwargs.variant
    for acfg in agent_cfgs:
        kw = acfg.get("kwargs")
        if isinstance(kw, dict):
            var_val = kw.get("variant")
            if isinstance(var_val, str) and var_val.strip():
                return var_val.strip()

    # 3. kwargs.max_thinking_tokens
    for acfg in agent_cfgs:
        kw = acfg.get("kwargs")
        if isinstance(kw, dict) and "max_thinking_tokens" in kw:
            fmt = _format_thinking_budget(kw.get("max_thinking_tokens"))
            if fmt is not None:
                return fmt

    # 4. env.CLAUDE_CODE_EFFORT_LEVEL
    for acfg in agent_cfgs:
        env = acfg.get("env")
        if isinstance(env, dict):
            eff_lvl = env.get("CLAUDE_CODE_EFFORT_LEVEL")
            if isinstance(eff_lvl, str) and eff_lvl.strip():
                return eff_lvl.strip()

    # 5. env.MAX_THINKING_TOKENS
    for acfg in agent_cfgs:
        env = acfg.get("env")
        if isinstance(env, dict) and "MAX_THINKING_TOKENS" in env:
            fmt = _format_thinking_budget(env.get("MAX_THINKING_TOKENS"))
            if fmt is not None:
                return fmt

    # 6. env.CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING == "1"
    for acfg in agent_cfgs:
        env = acfg.get("env")
        if isinstance(env, dict):
            dis = env.get("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING")
            if str(dis or "").strip().lower() in {"1", "true"}:
                return "off"

    # 7. First agent step reasoning_effort
    if isinstance(trajectory_dict, dict) and isinstance(trajectory_dict.get("steps"), list):
        for step in trajectory_dict["steps"]:
            if not isinstance(step, dict) or step.get("source") != "agent":
                continue
            st_re = step.get("reasoning_effort")
            if isinstance(st_re, str) and st_re.strip():
                return st_re.strip()
            st_ex = step.get("extra")
            if isinstance(st_ex, dict):
                ex_re = st_ex.get("reasoning_effort")
                if isinstance(ex_re, str) and ex_re.strip():
                    return ex_re.strip()
            break

    return "default"


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


def extract_trajectory_telemetry(
    traj_dict: dict[str, Any] | None,
    arm: str = "00_none",
    *,
    opencode_txt_path: Path | None = None,
    trial_dir: Path | None = None,
    opencode_db_path: Path | None = None,
) -> dict[str, int]:
    """Extract turn count, MCP server call counts, bash calls, and error counts from ATIF/SQLite."""
    exec_analysis = analyze_trial_execution(
        traj_dict,
        arm=arm,
        trial_dir=trial_dir,
        opencode_txt_path=opencode_txt_path,
        opencode_db_path=opencode_db_path,
    )
    return dict(exec_analysis.telemetry)


def _locate_trial_files(
    tp: TrialPaths,
) -> tuple[Path, Path, Path, Path | None, Path | None, Path | None, Path]:
    """Return `(reward_p, metrics_p, traj_p, oc_txt_p, oc_db_p, claude_txt_p, eff_agent_dir)`."""
    reward_p = tp.reward_json_path
    metrics_p = tp.verifier_dir / "metrics.json"
    traj_p = tp.agent_dir / "trajectory.json"
    eff_agent_dir = tp.agent_dir
    oc_p: Path | None = (
        tp.agent_dir / "opencode.txt" if (tp.agent_dir / "opencode.txt").is_file() else None
    )
    oc_db_cand = tp.agent_dir / "opencode" / "xdg-data" / "opencode" / "opencode.db"
    oc_db_p: Path | None = oc_db_cand if oc_db_cand.is_file() else None
    cc_txt_cand = tp.agent_dir / "claude-code.txt"
    cc_txt_p: Path | None = cc_txt_cand if cc_txt_cand.is_file() else None

    if (
        not reward_p.is_file()
        or not traj_p.is_file()
        or oc_p is None
        or oc_db_p is None
        or cc_txt_p is None
    ) and tp.steps_dir.is_dir():
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
                eff_agent_dir = last_step / "agent"
            if oc_p is None and (last_step / "agent" / "opencode.txt").is_file():
                oc_p = last_step / "agent" / "opencode.txt"
            step_db = last_step / "agent" / "opencode" / "xdg-data" / "opencode" / "opencode.db"
            if oc_db_p is None and step_db.is_file():
                oc_db_p = step_db
            step_cc = last_step / "agent" / "claude-code.txt"
            if cc_txt_p is None and step_cc.is_file():
                cc_txt_p = step_cc

    return reward_p, metrics_p, traj_p, oc_p, oc_db_p, cc_txt_p, eff_agent_dir


def _agent_result_has_activity(result_dict: dict[str, Any] | None) -> bool:
    if not isinstance(result_dict, dict):
        return False
    candidates: list[dict[str, Any]] = []
    if isinstance(result_dict.get("agent_result"), dict):
        candidates.append(result_dict["agent_result"])
    if isinstance(result_dict.get("step_results"), list):
        for sr in result_dict["step_results"]:
            if isinstance(sr, dict) and isinstance(sr.get("agent_result"), dict):
                candidates.append(sr["agent_result"])
    for ar in candidates:
        if (
            safe_nonneg_int(ar.get("n_input_tokens")) > 0
            or safe_nonneg_int(ar.get("n_output_tokens")) > 0
            or (isinstance(ar.get("cost_usd"), (int, float)) and float(ar["cost_usd"]) > 0.0)
        ):
            return True
    return False


def _scan_claude_jsonl_expected_events(agent_dir: Path) -> tuple[int, int, bool] | None:
    """Scan `<agent_dir>/sessions/projects/**/*.jsonl` -> `(n_turns, n_tools, had_corrupt_line)`."""
    proj_root = agent_dir / "sessions" / "projects"
    if not proj_root.is_dir():
        return None
    try:
        jsonl_files = sorted(proj_root.rglob("*.jsonl"))
    except OSError:
        return None
    if not jsonl_files:
        return None

    seen_uuids: set[str] = set()
    msg_ids: set[str] = set()
    tool_ids: set[str] = set()
    corrupt_line = False
    anon_counter = 0

    for jf in jsonl_files:
        try:
            lines = jf.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            corrupt_line = True
            continue
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                corrupt_line = True
                continue
            if not isinstance(ev, dict):
                continue
            uuid = ev.get("uuid")
            if isinstance(uuid, str) and uuid:
                if uuid in seen_uuids:
                    continue
                seen_uuids.add(uuid)
            if ev.get("type") != "assistant":
                continue
            msg = ev.get("message")
            if not isinstance(msg, dict):
                continue
            mid = msg.get("id")
            if isinstance(mid, str) and mid.strip():
                msg_ids.add(mid.strip())
            else:
                anon_counter += 1
                msg_ids.add(f"_anon_{anon_counter}")
            content = msg.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        cid = str(blk.get("id") or blk.get("tool_use_id") or "").strip()
                        if cid:
                            tool_ids.add(cid)

    return len(msg_ids), len(tool_ids), corrupt_line


def _scan_claude_stream_txt(claude_txt_path: Path) -> dict[str, Any] | None:
    if not claude_txt_path.is_file():
        return None
    try:
        lines = claude_txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    msg_ids: set[str] = set()
    tool_ids: set[str] = set()
    saw_result = False
    num_turns: int | None = None
    corrupt_json = False
    anon_idx = 0

    for raw_line in lines:
        s = raw_line.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            corrupt_json = True
            continue
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "assistant":
            msg = ev.get("message")
            if isinstance(msg, dict):
                mid = msg.get("id")
                if isinstance(mid, str) and mid.strip():
                    msg_ids.add(mid.strip())
                else:
                    anon_idx += 1
                    msg_ids.add(f"_stream_anon_{anon_idx}")
                content = msg.get("content")
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            cid = str(blk.get("id") or blk.get("tool_use_id") or "").strip()
                            if cid:
                                tool_ids.add(cid)
        elif etype == "result":
            saw_result = True
            if isinstance(ev.get("num_turns"), int) and not isinstance(ev.get("num_turns"), bool):
                num_turns = int(ev["num_turns"])

    return {
        "n_assistant_msgs": len(msg_ids),
        "n_tool_uses": len(tool_ids),
        "saw_result": saw_result,
        "num_turns": num_turns,
        "corrupt_json": corrupt_json,
    }


def _scan_opencode_stdout_events(opencode_txt_path: Path) -> dict[str, Any] | None:
    if not opencode_txt_path.is_file():
        return None
    try:
        lines = opencode_txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    n_starts = 0
    n_finishes = 0
    tool_ids: set[str] = set()
    corrupt_json = False
    anon_tool = 0

    for raw_line in lines:
        s = raw_line.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            corrupt_json = True
            continue
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type") or "").strip().lower()
        if etype in ("step_start", "step-start"):
            n_starts += 1
        elif etype in ("step_finish", "step-finish"):
            n_finishes += 1
        elif etype in ("tool_use", "tool"):
            part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
            cid = str(part.get("callID") or part.get("id") or "").strip()
            if cid:
                tool_ids.add(cid)
            else:
                anon_tool += 1
                tool_ids.add(f"_oc_anon_{anon_tool}")

    return {
        "n_starts": n_starts,
        "n_finishes": n_finishes,
        "n_tools": len(tool_ids),
        "corrupt_json": corrupt_json,
    }


def reconcile_trace_completeness(
    *,
    agent_name: str,
    model_name: str,
    agent_dir: Path,
    traj_path: Path,
    trajectory_dict: dict[str, Any] | None,
    result_dict: dict[str, Any] | None,
    exec_analysis: TrialExecutionAnalysis,
    opencode_txt_path: Path | None = None,
    claude_txt_path: Path | None = None,
) -> tuple[int, int, bool]:
    """Reconcile ATIF + recovered subagent trace counts against harness-native disk records."""
    ag_low = agent_name.strip().lower()
    if ag_low == "oracle" or model_name.strip().lower() == "oracle":
        return 0, 0, True

    traj_corrupt_on_disk = False
    if traj_path.is_file() and trajectory_dict is None:
        traj_corrupt_on_disk = True

    ar_active = _agent_result_has_activity(result_dict)

    # Check internal ATIF FinalMetrics step truncation if trajectory_dict is present
    fm_step_shortfall = 0
    if isinstance(trajectory_dict, dict) and isinstance(trajectory_dict.get("final_metrics"), dict):
        fm = trajectory_dict["final_metrics"]
        if fm.get("total_steps") is not None:
            fm_steps = safe_nonneg_int(fm.get("total_steps"))
            actual_steps = len(trajectory_dict.get("steps") or [])
            if fm_steps > actual_steps:
                fm_step_shortfall = fm_steps - actual_steps

    # 1. OpenCode harness reconciliation
    if ag_low == "opencode":
        observed = (
            exec_analysis.atif_agent_turns
            + exec_analysis.atif_tool_calls
            + exec_analysis.recovered_child_turns
            + exec_analysis.recovered_child_tool_calls
        )
        if exec_analysis.opencode_db is not None:
            expected = (
                exec_analysis.opencode_db.total_step_finish_events
                + exec_analysis.opencode_db.total_tool_events
            )
            has_root_in_db = any(len(s.turns) > 0 for s in exec_analysis.opencode_db.root_sessions)
            root_traj_missing = has_root_in_db and trajectory_dict is None
            ok = (
                (not traj_corrupt_on_disk)
                and (not root_traj_missing)
                and (not exec_analysis.opencode_db.has_unfinished_step)
                and fm_step_shortfall == 0
                and observed == expected
                and expected > 0
            )
            return expected, observed, ok

        if opencode_txt_path is not None and opencode_txt_path.is_file():
            oc_scan = _scan_opencode_stdout_events(opencode_txt_path)
            if oc_scan is not None:
                n_st = int(oc_scan["n_starts"])
                n_fn = int(oc_scan["n_finishes"])
                n_tl = int(oc_scan["n_tools"])
                corrupt_txt = bool(oc_scan["corrupt_json"])
                if n_st > 0 or n_fn > 0:
                    expected = max(n_st, n_fn) + n_tl
                    ok = (
                        (not traj_corrupt_on_disk)
                        and (trajectory_dict is not None)
                        and (not corrupt_txt)
                        and (n_st <= n_fn)
                        and fm_step_shortfall == 0
                        and observed == expected
                        and expected > 0
                    )
                    return expected, observed, ok
                if n_tl > 0:
                    expected = exec_analysis.atif_agent_turns + n_tl
                    ok = (
                        (not traj_corrupt_on_disk)
                        and (trajectory_dict is not None)
                        and (not corrupt_txt)
                        and fm_step_shortfall == 0
                        and observed == expected
                    )
                    return expected, observed, ok

    # 2. ClaudeCode harness reconciliation
    if ag_low == "claude-code":
        observed = exec_analysis.atif_agent_turns + exec_analysis.atif_tool_calls
        cc_jsonl = _scan_claude_jsonl_expected_events(agent_dir)
        cc_stream = (
            _scan_claude_stream_txt(claude_txt_path) if claude_txt_path is not None else None
        )
        stream_corrupt = bool(cc_stream and cc_stream["corrupt_json"])
        stream_missing_result = bool(
            cc_stream and cc_stream["n_assistant_msgs"] > 0 and not cc_stream["saw_result"]
        )

        if cc_jsonl is not None:
            exp_turns, exp_tools, jsonl_corrupt = cc_jsonl
            expected = exp_turns + exp_tools
            ok = (
                (not traj_corrupt_on_disk)
                and (trajectory_dict is not None)
                and (not jsonl_corrupt)
                and (not stream_corrupt)
                and (not stream_missing_result)
                and fm_step_shortfall == 0
                and observed == expected
                and expected > 0
            )
            return expected, observed, ok

        if cc_stream is not None and (cc_stream["n_assistant_msgs"] > 0 or cc_stream["saw_result"]):
            if cc_stream["n_assistant_msgs"] > 0:
                expected = int(cc_stream["n_assistant_msgs"]) + int(cc_stream["n_tool_uses"])
                ok = (
                    (not traj_corrupt_on_disk)
                    and (trajectory_dict is not None)
                    and (not stream_corrupt)
                    and bool(cc_stream["saw_result"])
                    and fm_step_shortfall == 0
                    and observed == expected
                )
                return expected, observed, ok
            if cc_stream["num_turns"] is not None:
                exp_t = int(cc_stream["num_turns"])
                expected = exp_t + exec_analysis.atif_tool_calls
                ok = (
                    (not traj_corrupt_on_disk)
                    and (trajectory_dict is not None)
                    and (not stream_corrupt)
                    and fm_step_shortfall == 0
                    and exec_analysis.atif_agent_turns == exp_t
                )
                return expected, observed, ok

    # 3. Generic ATIF / fallback reconciliation
    observed_generic = (
        exec_analysis.atif_agent_turns
        + exec_analysis.atif_tool_calls
        + exec_analysis.recovered_child_turns
        + exec_analysis.recovered_child_tool_calls
    )
    if traj_corrupt_on_disk:
        return max(1, observed_generic), observed_generic, False
    if trajectory_dict is None:
        if ar_active:
            return 1, 0, False
        has_exc = isinstance(result_dict, dict) and result_dict.get("exception_info") is not None
        return 0, 0, (not has_exc)
    if fm_step_shortfall > 0:
        return observed_generic + fm_step_shortfall, observed_generic, False
    if ar_active and observed_generic == 0:
        return 1, 0, False
    return observed_generic, observed_generic, True


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
            if not agent_name:
                if isinstance(acfg.get("name"), str) and acfg["name"].strip():
                    agent_name = acfg["name"].strip()
                elif "opencode" in str(acfg.get("import_path") or "").lower():
                    agent_name = "opencode"
            if not raw_model and isinstance(acfg.get("model_name"), str):
                raw_model = acfg["model_name"].strip()

    if isinstance(config_dict, dict) and isinstance(config_dict.get("agent"), dict):
        acfg = config_dict["agent"]
        if not agent_name:
            if isinstance(acfg.get("name"), str) and acfg["name"].strip():
                agent_name = acfg["name"].strip()
            elif "opencode" in str(acfg.get("import_path") or "").lower():
                agent_name = "opencode"
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
        with contextlib.suppress(OSError):
            txt = tp.exception_message_path.read_text(encoding="utf-8").strip()
            if txt:
                first_line = txt.splitlines()[0][:300]
                return "TrialException", first_line
    return None, None


def collect_single_trial(job_dir: Path, trial_dir: Path) -> dict[str, Any]:
    """Parse a single Harbor 0.25.0 `trial_dir` into a normalized trial record."""
    tp = TrialPaths(trial_dir)
    result_dict = _read_json_dict(tp.result_path)
    config_dict = _read_json_dict(tp.config_path)
    reward_p, metrics_p, traj_p, oc_p, oc_db_p, cc_txt_p, eff_agent_dir = _locate_trial_files(tp)

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
    base_task, parsed_pack_id, arm, suite, stage, is_e2e = parse_task_variant_with_pack(
        task_name, metrics_stage=metrics_stage
    )
    pack_id = parsed_pack_id
    if isinstance(config_dict, dict):
        task_cfg = config_dict.get("task")
        task_path_raw = task_cfg.get("path") if isinstance(task_cfg, dict) else None
        if isinstance(task_path_raw, str) and task_path_raw.strip():
            task_toml = Path(task_path_raw) / "task.toml"
            if task_toml.is_file():
                with contextlib.suppress(OSError, ValueError):
                    t_meta = tomllib.loads(task_toml.read_text(encoding="utf-8")).get("metadata")
                    if isinstance(t_meta, dict):
                        if pack_id is None and isinstance(t_meta.get("pack_id"), str):
                            pack_id = t_meta["pack_id"].strip() or None
                        if (
                            arm == "00_none"
                            and isinstance(t_meta.get("arm"), str)
                            and t_meta["arm"].strip()
                        ):
                            arm = t_meta["arm"].strip()
                        if (
                            base_task not in BASE_TASK_META
                            and isinstance(t_meta.get("base_task"), str)
                            and t_meta["base_task"].strip()
                        ):
                            base_task = t_meta["base_task"].strip()
                            if base_task in BASE_TASK_META:
                                suite, stage = BASE_TASK_META[base_task]
                                is_e2e = stage == "e2e"
                            else:
                                if isinstance(t_meta.get("suite"), str) and t_meta["suite"].strip():
                                    suite = t_meta["suite"].strip()
                                if isinstance(t_meta.get("stage"), str) and t_meta["stage"].strip():
                                    stage = t_meta["stage"].strip()
                                    is_e2e = stage == "e2e"
    if (
        suite == "unknown"
        and isinstance(metrics_dict, dict)
        and isinstance(metrics_dict.get("suite_id"), str)
        and metrics_dict["suite_id"].strip()
    ):
        suite = metrics_dict["suite_id"].strip()
    if pack_id is None:
        pack_id = "ra_00a11ea"

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
    thinking_level = extract_thinking_level(config_dict, result_dict, trajectory_dict)
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

    exec_analysis = analyze_trial_execution(
        trajectory_dict,
        model_name=model_name,
        agent_name=agent_name,
        arm=arm,
        trial_dir=trial_dir,
        opencode_txt_path=oc_p,
        opencode_db_path=oc_db_p,
    )
    telemetry = exec_analysis.telemetry
    pricing = build_trial_pricing_from_execution(
        exec_analysis,
        model_name=model_name,
        agent_name=agent_name,
        trajectory_dict=trajectory_dict,
        trial_result_dict=result_dict,
    )
    trace_expected, trace_observed, trace_ok = reconcile_trace_completeness(
        agent_name=agent_name,
        model_name=model_name,
        agent_dir=eff_agent_dir,
        traj_path=traj_p,
        trajectory_dict=trajectory_dict,
        result_dict=result_dict,
        exec_analysis=exec_analysis,
        opencode_txt_path=oc_p,
        claude_txt_path=cc_txt_p,
    )

    runtime_diag = extract_agent_runtime_diagnostics(oc_p, exception_type, exception_message)
    eff_protocol_errors = int(telemetry["protocol_errors"])
    if runtime_diag["provider_error_kind"] == "tool_call_parse_400":
        eff_protocol_errors = max(1, eff_protocol_errors + 1)

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
        protocol_errors=eff_protocol_errors,
        mcp_calls_total=telemetry["mcp_calls_total"],
        bash_nonzero_exits=telemetry["bash_nonzero_exits"],
        severe_casa_errors=telemetry["severe_casa_errors"],
    )

    truth_block: dict[str, Any] = {}
    measured_block: dict[str, Any] = {}
    measurements_block: dict[str, Any] = {}
    rubric_meta_block: dict[str, Any] = {}
    diff_summary_block: dict[str, Any] = {}
    triggered_tags_list: list[str] = []
    if isinstance(metrics_dict, dict):
        if isinstance(metrics_dict.get("measurements"), dict):
            measurements_block = sanitize_json_tree(metrics_dict["measurements"])
        if isinstance(metrics_dict.get("rubric_meta"), dict):
            rubric_meta_block = sanitize_json_tree(metrics_dict["rubric_meta"])
        if isinstance(metrics_dict.get("diff_summary"), dict) and metrics_dict["diff_summary"]:
            diff_summary_block = sanitize_json_tree(metrics_dict["diff_summary"])
        if isinstance(metrics_dict.get("triggered_failure_tags"), list):
            triggered_tags_list = [str(x) for x in metrics_dict["triggered_failure_tags"] if x]

        if stage == "e2e" and (
            isinstance(metrics_dict.get("calibration"), dict)
            or isinstance(metrics_dict.get("image"), dict)
        ):
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

    trial_out: dict[str, Any] = {
        "job_dir": str(job_dir),
        "trial_dir": str(trial_dir),
        "trial_name": trial_name,
        "started_at": started_at_str,
        "task_name": task_name,
        "base_task": base_task,
        "pack_id": pack_id,
        "suite": suite,
        "stage": stage,
        "is_e2e": is_e2e,
        "arm": arm,
        "agent_name": agent_name,
        "model_name": model_name,
        "thinking_level": thinking_level,
        "attempt_idx": 0,
        "reward": raw_reward,
        "passed": int(passed),
        "failure_mode": failure_mode,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "agent_exit_status": runtime_diag["agent_exit_status"],
        "decoder_collapse": bool(runtime_diag["decoder_collapse"]),
        "decoder_collapse_detail": runtime_diag["decoder_collapse_detail"],
        "output_token_limit_hit": bool(runtime_diag["output_token_limit_hit"]),
        "last_step_tokens": runtime_diag["last_step_tokens"],
        "provider_error_kind": runtime_diag["provider_error_kind"],
        "provider_status_code": runtime_diag["provider_status_code"],
        "provider_error_count": runtime_diag["provider_error_count"],
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
        "protocol_errors": eff_protocol_errors,
        **pricing.to_dict(),
        "trace_events_expected": trace_expected,
        "trace_events_observed": trace_observed,
        "trace_completeness_ok": trace_ok,
        "role_breakdown": exec_analysis.role_breakdown,
        "subscores": subscores,
        "truth": truth_block,
        "measured": measured_block,
    }
    if measurements_block:
        trial_out["measurements"] = measurements_block
    if rubric_meta_block:
        trial_out["rubric_meta"] = rubric_meta_block
    if diff_summary_block:
        trial_out["diff_summary"] = diff_summary_block
    if triggered_tags_list:
        trial_out["triggered_failure_tags"] = triggered_tags_list

    return TrialRecord(sanitize_json_tree(trial_out))


collect_trial = collect_single_trial


def summarize_run_health(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize agent exit statuses, benign EPIPE exits, decoder collapses, API errors, and timeouts."""
    clean_completions = 0
    benign_epipe_exits = 0
    provider_api_errors = 0
    recovered_provider_errors = 0
    decoder_collapses = 0
    output_token_limits = 0
    agent_timeouts = 0
    agent_errors = 0
    provider_error_trials: list[dict[str, Any]] = []
    recovered_provider_error_trials: list[dict[str, Any]] = []
    decoder_collapse_trials: list[dict[str, Any]] = []
    output_token_limit_trials: list[dict[str, Any]] = []
    timeout_trials: list[dict[str, Any]] = []
    agent_error_trials: list[dict[str, Any]] = []

    for rec in records:
        status = str(rec.get("agent_exit_status") or "ok").strip() or "ok"
        if status == "ok":
            clean_completions += 1
        elif status == "benign_epipe":
            benign_epipe_exits += 1
        elif status == "provider_api_error":
            provider_api_errors += 1
        elif status == "decoder_collapse":
            decoder_collapses += 1
        elif status == "output_token_limit":
            output_token_limits += 1
        elif status == "agent_timeout":
            agent_timeouts += 1
        else:
            agent_errors += 1

        p_cnt = safe_nonneg_int(rec.get("provider_error_count"))
        p_kind = rec.get("provider_error_kind")
        if status == "provider_api_error":
            provider_error_trials.append(
                {
                    "trial_name": str(rec.get("trial_name") or ""),
                    "base_task": str(rec.get("base_task") or ""),
                    "stage": str(rec.get("stage") or ""),
                    "arm": str(rec.get("arm") or ""),
                    "model_name": str(rec.get("model_name") or ""),
                    "n_turns": safe_nonneg_int(rec.get("n_turns")),
                    "agent_exit_status": status,
                    "provider_error_kind": p_kind,
                    "provider_status_code": rec.get("provider_status_code"),
                    "provider_error_count": p_cnt,
                }
            )
        elif p_cnt > 0 or p_kind is not None:
            recovered_provider_errors += 1
            recovered_provider_error_trials.append(
                {
                    "trial_name": str(rec.get("trial_name") or ""),
                    "base_task": str(rec.get("base_task") or ""),
                    "stage": str(rec.get("stage") or ""),
                    "arm": str(rec.get("arm") or ""),
                    "model_name": str(rec.get("model_name") or ""),
                    "n_turns": safe_nonneg_int(rec.get("n_turns")),
                    "agent_exit_status": status,
                    "provider_error_kind": p_kind,
                    "provider_status_code": rec.get("provider_status_code"),
                    "provider_error_count": p_cnt,
                }
            )
        if status == "decoder_collapse":
            lst_tok = rec.get("last_step_tokens")
            decoder_collapse_trials.append(
                {
                    "trial_name": str(rec.get("trial_name") or ""),
                    "base_task": str(rec.get("base_task") or ""),
                    "stage": str(rec.get("stage") or ""),
                    "arm": str(rec.get("arm") or ""),
                    "model_name": str(rec.get("model_name") or ""),
                    "n_turns": safe_nonneg_int(rec.get("n_turns")),
                    "decoder_collapse_detail": rec.get("decoder_collapse_detail"),
                    "last_step_tokens": (
                        int(lst_tok)
                        if isinstance(lst_tok, (int, float))
                        and not isinstance(lst_tok, bool)
                        and int(lst_tok) > 0
                        else None
                    ),
                    "reward": float(rec.get("reward") or 0.0),
                    "passed": int(rec.get("passed") or 0),
                }
            )
        elif status == "output_token_limit":
            lst_tok = rec.get("last_step_tokens")
            output_token_limit_trials.append(
                {
                    "trial_name": str(rec.get("trial_name") or ""),
                    "base_task": str(rec.get("base_task") or ""),
                    "stage": str(rec.get("stage") or ""),
                    "arm": str(rec.get("arm") or ""),
                    "model_name": str(rec.get("model_name") or ""),
                    "n_turns": safe_nonneg_int(rec.get("n_turns")),
                    "last_step_tokens": (
                        int(lst_tok)
                        if isinstance(lst_tok, (int, float))
                        and not isinstance(lst_tok, bool)
                        and int(lst_tok) > 0
                        else None
                    ),
                    "reward": float(rec.get("reward") or 0.0),
                    "passed": int(rec.get("passed") or 0),
                }
            )
        elif status == "agent_timeout":
            timeout_trials.append(
                {
                    "trial_name": str(rec.get("trial_name") or ""),
                    "base_task": str(rec.get("base_task") or ""),
                    "stage": str(rec.get("stage") or ""),
                    "arm": str(rec.get("arm") or ""),
                    "model_name": str(rec.get("model_name") or ""),
                    "agent_wall_sec": float(rec.get("agent_wall_sec") or 0.0),
                    "reward": float(rec.get("reward") or 0.0),
                    "passed": int(rec.get("passed") or 0),
                }
            )
        elif status not in ("ok", "benign_epipe", "provider_api_error"):
            agent_error_trials.append(
                {
                    "trial_name": str(rec.get("trial_name") or ""),
                    "base_task": str(rec.get("base_task") or ""),
                    "stage": str(rec.get("stage") or ""),
                    "arm": str(rec.get("arm") or ""),
                    "model_name": str(rec.get("model_name") or ""),
                    "exception_type": str(rec.get("exception_type") or ""),
                    "reward": float(rec.get("reward") or 0.0),
                    "passed": int(rec.get("passed") or 0),
                }
            )

    return sanitize_json_tree(
        {
            "n_trials": len(records),
            "clean_completions": clean_completions,
            "benign_epipe_exits": benign_epipe_exits,
            "provider_api_errors": provider_api_errors,
            "recovered_provider_errors": recovered_provider_errors,
            "decoder_collapses": decoder_collapses,
            "output_token_limits": output_token_limits,
            "agent_timeouts": agent_timeouts,
            "agent_errors": agent_errors,
            "provider_error_trials": provider_error_trials,
            "recovered_provider_error_trials": recovered_provider_error_trials,
            "decoder_collapse_trials": decoder_collapse_trials,
            "output_token_limit_trials": output_token_limit_trials,
            "timeout_trials": timeout_trials,
            "agent_error_trials": agent_error_trials,
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
    out_p = Path(output_dir)
    reject_slash_tmp(out_p, label="output_dir")
    records = [sanitize_json_tree(r) for r in records]
    out_p.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_p / "trials.jsonl"
    csv_path = out_p / "trials.csv"

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

    # 2. Deduplicate identical trial_name across job directories (include pack_id & thinking_level).
    by_trial_key: dict[
        tuple[str, str, str, str, str, str, str, str],
        tuple[int, dict[str, Any], dict[str, Any]],
    ] = {}
    for ord_idx, rec, prov in surviving:
        t_name = str(rec.get("trial_name") or "")
        key = (
            str(rec.get("suite")),
            str(rec.get("base_task")),
            str(rec.get("pack_id") or "ra_00a11ea"),
            str(rec.get("arm")),
            str(rec.get("agent_name")),
            str(rec.get("model_name")),
            str(rec.get("thinking_level") or "default"),
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
    if output_dir is not None:
        reject_slash_tmp(Path(output_dir), label="output_dir")
    job_order_map: dict[Path, int] = {}
    for idx, raw_j in enumerate(job_dirs):
        p = Path(raw_j)
        reject_slash_tmp(p, label="job_dir")
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

    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (
            str(rec["suite"]),
            str(rec["base_task"]),
            str(rec.get("pack_id") or "ra_00a11ea"),
            str(rec["arm"]),
            str(rec["agent_name"]),
            str(rec["model_name"]),
            str(rec.get("thinking_level") or "default"),
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
