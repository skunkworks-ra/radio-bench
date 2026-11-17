"""Comprehensive Wave 3 verification tests: trace economics, diff-run, and Pareto frontiers."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.task.id import LocalTaskId
from harbor.models.trajectories import (
    Agent,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import (
    AgentInfo,
    ModelInfo,
    TimingInfo,
    TrialResult,
)
from harbor.models.verifier.result import VerifierResult

from radio_bench import cli as rb_cli
from radio_bench.analysis import (
    analyze_factorial,
    collect_single_trial,
    collect_trials,
    compute_pareto_frontiers,
    compute_parity_summary,
    compute_trial_pricing,
    diff_caltables,
    diff_images,
    diff_visibilities,
    extract_thinking_level,
    summarize_failure_taxonomy,
    write_diagnostic_reports,
)
from radio_bench.config import (
    ModelsConfig,
    load_bench_config,
    load_models_config,
    write_casaconfig_local,
)
from radio_bench.grading.runner import run_verification
from radio_bench.grading.schema import (
    ExtractConfig,
    GeometrySpec,
    LeafRuleSpec,
    RolesSpec,
    RubricSpec,
    SourceApertureSpec,
    SubscoreSpec,
)


def _ensure_local_casa() -> Any:
    cfg = load_bench_config()
    cfile = write_casaconfig_local(cfg)
    if cfile.is_file():
        import os

        os.environ.setdefault("CASASITECONFIG", str(cfile))
    import casatools

    return casatools


def _write_trial_fixture(
    job_dir: Path,
    *,
    task_name: str,
    agent_name: str,
    model_name: str | None,
    agent_kwargs: dict[str, Any] | None = None,
    agent_env: dict[str, str] | None = None,
    reward: float = 0.90,
    passed: int = 1,
    trajectory_dict: dict[str, Any] | None = None,
    raw_trajectory_text: str | None = None,
    agent_context: AgentContext | None = None,
    started_at: datetime | None = None,
    omit_config_kwargs: bool = False,
) -> Path:
    st = started_at or datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)
    ft = st + timedelta(seconds=20.0)
    tasks_base = load_bench_config().tasks_dir

    full_agent_cfg: dict[str, Any] = {"name": agent_name}
    if model_name is not None:
        full_agent_cfg["model_name"] = model_name
    if agent_kwargs:
        full_agent_cfg["kwargs"] = dict(agent_kwargs)
    if agent_env:
        full_agent_cfg["env"] = dict(agent_env)

    tcfg = TrialConfig(
        task={"path": str(tasks_base / task_name)},
        agent=full_agent_cfg,
    )
    trial_dir = job_dir / tcfg.trial_name
    tp = TrialPaths(trial_dir)
    tp.mkdir()

    model_info: ModelInfo | None = None
    if model_name:
        if "/" in model_name:
            prov, bare = model_name.split("/", 1)
            model_info = ModelInfo(name=bare, provider=prov)
        else:
            model_info = ModelInfo(name=model_name, provider=None)

    tr = TrialResult(
        task_name=task_name,
        trial_name=tcfg.trial_name,
        trial_uri=f"file://{trial_dir}",
        task_id=LocalTaskId(path=tasks_base / task_name),
        task_checksum="0011223344556677",
        config=tcfg,
        agent_info=AgentInfo(name=agent_name, version="0.25.0", model_info=model_info),
        agent_result=agent_context,
        verifier_result=VerifierResult(rewards={"reward": reward, "passed": passed}),
        started_at=st,
        finished_at=ft,
        agent_execution=TimingInfo(started_at=st, finished_at=st + timedelta(seconds=15.0)),
        verifier=TimingInfo(started_at=st + timedelta(seconds=15.0), finished_at=ft),
    )

    cfg_dump = json.loads(tcfg.model_dump_json(indent=2))
    if omit_config_kwargs and isinstance(cfg_dump.get("agent"), dict):
        cfg_dump["agent"].pop("kwargs", None)
        cfg_dump["agent"].pop("env", None)

    tp.config_path.write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")
    tp.result_path.write_text(tr.model_dump_json(indent=2), encoding="utf-8")
    tp.reward_json_path.write_text(
        json.dumps({"reward": reward, "passed": passed}, indent=2), encoding="utf-8"
    )
    (tp.verifier_dir / "metrics.json").write_text(
        json.dumps({"schema_version": "0.2", "stage": "inspect", "subscores": {}}, indent=2),
        encoding="utf-8",
    )
    if raw_trajectory_text is not None:
        (tp.agent_dir / "trajectory.json").write_text(raw_trajectory_text, encoding="utf-8")
    elif trajectory_dict is not None:
        (tp.agent_dir / "trajectory.json").write_text(
            json.dumps(trajectory_dict, indent=2, allow_nan=False), encoding="utf-8"
        )
    return trial_dir


# ==============================================================================
# 1. Provider-Aware Reasoning vs. Reply Token Accounting
# ==============================================================================


def test_opencode_exclusive_vs_claudecode_inclusive_reasoning_tokens() -> None:
    """Verify zero OpenCode under-billing, zero ClaudeCode double-billing, and strict None rule."""
    # 1a. OpenCode with reasoning > 0: completion_tokens is reply-only -> billed = reply + reasoning!
    traj_oc = Trajectory(
        session_id="oc-1",
        agent=Agent(name="opencode", version="1.19.5", model_name="nrp/kimi"),
        steps=[
            Step(step_id=1, source="user", message="Solve."),
            Step(
                step_id=2,
                source="agent",
                message="Step 1",
                metrics=Metrics(
                    prompt_tokens=10_000,
                    completion_tokens=800,
                    extra={"reasoning_tokens": 1_200},
                ),
            ),
            Step(
                step_id=3,
                source="agent",
                message="Step 2 (no reasoning)",
                metrics=Metrics(
                    prompt_tokens=12_000,
                    completion_tokens=400,
                ),
            ),
        ],
    )
    p_oc = compute_trial_pricing(
        model_name="nrp/kimi",
        agent_name="opencode",
        trajectory_dict=traj_oc.to_json_dict(),
    )
    # Billed output = (800 + 1200) + (400 + 0) = 2400; reasoning = 1200; reply = 1200
    assert p_oc.output_tokens == 2_400
    assert p_oc.reasoning_tokens == 1_200
    assert p_oc.reply_output_tokens == 1_200
    # nrp/kimi: $0.60/M input, $3.00/M output -> 22k * 0.60/1e6 + 2.4k * 3.00/1e6 = 0.0132 + 0.0072 = 0.0204
    assert math.isclose(p_oc.output_usd, 0.0072, abs_tol=1e-10)
    assert math.isclose(p_oc.total_usd, 0.0204, abs_tol=1e-10)

    # 1b. OpenCode with 0 reasoning across all turns -> reasoning_tokens & reply_output_tokens MUST be None!
    traj_oc_no_reas = Trajectory(
        session_id="oc-2",
        agent=Agent(name="opencode", version="1.19.5", model_name="nrp/kimi"),
        steps=[
            Step(step_id=1, source="user", message="Solve."),
            Step(
                step_id=2,
                source="agent",
                message="Done",
                metrics=Metrics(prompt_tokens=5_000, completion_tokens=500),
            ),
        ],
    )
    p_oc_zero = compute_trial_pricing(
        model_name="nrp/kimi",
        agent_name="opencode",
        trajectory_dict=traj_oc_no_reas.to_json_dict(),
    )
    assert p_oc_zero.output_tokens == 500
    assert p_oc_zero.reasoning_tokens is None
    assert p_oc_zero.reply_output_tokens is None

    # 2a. ClaudeCode inclusive: completion_tokens ALREADY includes thinking_tokens!
    traj_cc = Trajectory(
        session_id="cc-1",
        agent=Agent(name="claude-code", version="2.1.325", model_name="claude-sonnet-4-6"),
        steps=[
            Step(step_id=1, source="user", message="Solve."),
            Step(
                step_id=2,
                source="agent",
                message="Solved.",
                metrics=Metrics(
                    prompt_tokens=10_000,
                    completion_tokens=2_000,
                    extra={"output_tokens_details": {"thinking_tokens": 1_250}},
                ),
            ),
        ],
    )
    p_cc = compute_trial_pricing(
        model_name="anthropic/claude-sonnet-4-6",
        agent_name="claude-code",
        trajectory_dict=traj_cc.to_json_dict(),
    )
    assert p_cc.output_tokens == 2_000
    assert p_cc.reasoning_tokens == 1_250
    assert p_cc.reply_output_tokens == 750
    # Sonnet 4.6: $3/M input, $15/M output -> 10k*3/1e6 + 2k*15/1e6 = 0.03 + 0.03 = 0.06
    assert math.isclose(p_cc.output_usd, 0.03, abs_tol=1e-10)
    assert math.isclose(p_cc.total_usd, 0.06, abs_tol=1e-10)

    # 2b. ClaudeCode 2.1.325 SDK placeholder `thinking_tokens: 0` -> MUST emit None (NEVER 0)!
    traj_cc_sdk_zero = Trajectory(
        session_id="cc-2",
        agent=Agent(name="claude-code", version="2.1.325", model_name="claude-sonnet-4-6"),
        steps=[
            Step(step_id=1, source="user", message="Solve."),
            Step(
                step_id=2,
                source="agent",
                message="Solved.",
                metrics=Metrics(
                    prompt_tokens=10_000,
                    completion_tokens=1_000,
                    extra={"output_tokens_details": {"thinking_tokens": 0}},
                ),
            ),
        ],
    )
    p_cc_sdk = compute_trial_pricing(
        model_name="anthropic/claude-sonnet-4-6",
        agent_name="claude-code",
        trajectory_dict=traj_cc_sdk_zero.to_json_dict(),
    )
    assert p_cc_sdk.output_tokens == 1_000
    assert p_cc_sdk.reasoning_tokens is None
    assert p_cc_sdk.reply_output_tokens is None


# ==============================================================================
# 2. `thinking_level` Extraction & Grouping Across Cells / Contrasts / Parity
# ==============================================================================


def test_thinking_level_extraction_and_grouping(tmp_path: Path) -> None:
    """Verify full `thinking_level` priority chain, result.json fallback, and factorial/parity grouping."""
    assert (
        extract_thinking_level({"agent": {"kwargs": {"reasoning_effort": "high"}}}, None, None)
        == "high"
    )
    assert (
        extract_thinking_level({"agent": {"kwargs": {"variant": "minimal"}}}, None, None)
        == "minimal"
    )
    assert (
        extract_thinking_level({"agent": {"kwargs": {"max_thinking_tokens": 0}}}, None, None)
        == "off"
    )
    assert (
        extract_thinking_level({"agent": {"kwargs": {"max_thinking_tokens": 16000}}}, None, None)
        == "budget_16000"
    )
    assert (
        extract_thinking_level(
            {"agent": {"env": {"CLAUDE_CODE_EFFORT_LEVEL": "medium"}}}, None, None
        )
        == "medium"
    )
    assert (
        extract_thinking_level({"agent": {"env": {"MAX_THINKING_TOKENS": "8192"}}}, None, None)
        == "budget_8192"
    )
    assert (
        extract_thinking_level(
            {"agent": {"env": {"CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1"}}}, None, None
        )
        == "off"
    )
    assert (
        extract_thinking_level(
            {},
            None,
            {"steps": [{"source": "agent", "extra": {"reasoning_effort": "max"}}]},
        )
        == "max"
    )
    assert extract_thinking_level({}, {}, None) == "default"

    # Build a job with Sonnet at both `default` and `high` (with config.json omitting kwargs on `high`
    # so `result.json["config"]["agent"]` fallback is exercised!) + Haiku at `default`.
    job_dir = tmp_path / "thinking_job"
    job_dir.mkdir()
    base_dt = datetime(2026, 11, 17, 10, 0, 0, tzinfo=UTC)
    idx = 0
    for arm, rew_def, rew_high in (
        ("00_none", 0.30, 0.50),
        ("10_tools", 0.50, 0.70),
        ("01_skills_casa", 0.45, 0.65),
        ("11_full", 0.75, 0.95),
    ):
        idx += 1
        _write_trial_fixture(
            job_dir,
            task_name=f"s1_inspect__{arm}",
            agent_name="claude-code",
            model_name="anthropic/claude-sonnet-4-6",
            agent_kwargs=None,
            reward=rew_def,
            passed=int(rew_def >= 0.85),
            started_at=base_dt + timedelta(seconds=idx),
        )
        idx += 1
        _write_trial_fixture(
            job_dir,
            task_name=f"s1_inspect__{arm}",
            agent_name="claude-code",
            model_name="anthropic/claude-sonnet-4-6",
            agent_kwargs={"reasoning_effort": "high"},
            reward=rew_high,
            passed=int(rew_high >= 0.85),
            started_at=base_dt + timedelta(seconds=idx),
            omit_config_kwargs=True,
        )

    # Also add Haiku @ 11_full (`default`, reward = 0.88) for parity comparison against Sonnet `00_none`
    _write_trial_fixture(
        job_dir,
        task_name="s1_inspect__11_full",
        agent_name="claude-code",
        model_name="anthropic/claude-haiku-4-5",
        reward=0.88,
        passed=1,
        started_at=base_dt + timedelta(seconds=99),
    )

    recs = collect_trials([job_dir])
    assert len(recs) == 9
    sonnet_high = [
        r
        for r in recs
        if r["model_name"] == "anthropic/claude-sonnet-4-6" and r["thinking_level"] == "high"
    ]
    sonnet_def = [
        r
        for r in recs
        if r["model_name"] == "anthropic/claude-sonnet-4-6" and r["thinking_level"] == "default"
    ]
    assert len(sonnet_high) == 4
    assert len(sonnet_def) == 4
    # Each (arm, model, thinking_level) has 1 attempt -> attempt_idx == 0 (not merged with default!)
    assert all(r["attempt_idx"] == 0 for r in sonnet_high)

    fac = analyze_factorial(recs, n_boot=200, seed=42)
    assert "anthropic/claude-sonnet-4-6" in fac["by_model"]
    assert "anthropic/claude-sonnet-4-6__default" in fac["by_model_thinking"]
    assert "anthropic/claude-sonnet-4-6__high" in fac["by_model_thinking"]
    assert math.isclose(
        fac["by_model_thinking"]["anthropic/claude-sonnet-4-6__high"]["cells"]["11_full"][
            "reward_task_equal_mean"
        ],
        0.95,
        abs_tol=1e-8,
    )
    assert math.isclose(
        fac["by_model_thinking"]["anthropic/claude-sonnet-4-6__default"]["cells"]["11_full"][
            "reward_task_equal_mean"
        ],
        0.75,
        abs_tol=1e-8,
    )
    # Paired contrast has 2 complete units: (s1_inspect, ra_00a11ea, sonnet, default) & (..., high)
    assert fac["overall"]["contrasts"]["I_casa"]["n_paired_units"] == 2

    # Verify ParitySideSpec `thinking_level` matching: default (0.88 - 0.30 = +0.58) vs high (0.88 - 0.50 = +0.38)
    base_models_cfg = load_models_config()
    from radio_bench.config import ParityPairSpec, ParitySideSpec

    custom_models_cfg = ModelsConfig(
        models=base_models_cfg.models,
        parity_pairs=(
            ParityPairSpec(
                pair_id="haiku_vs_sonnet_def",
                small=ParitySideSpec("claude_haiku_4_5", "11_full", "default"),
                frontier=ParitySideSpec("claude_sonnet_4_6", "00_none", "default"),
            ),
            ParityPairSpec(
                pair_id="haiku_vs_sonnet_high",
                small=ParitySideSpec("claude_haiku_4_5", "11_full", "default"),
                frontier=ParitySideSpec("claude_sonnet_4_6", "00_none", "high"),
            ),
        ),
    )
    par_custom = compute_parity_summary(recs, n_boot=100, seed=7, models_cfg=custom_models_cfg)
    assert math.isclose(
        par_custom["haiku_vs_sonnet_def__11_full_vs_00_none"]["estimate"], 0.58, abs_tol=1e-8
    )
    assert math.isclose(
        par_custom["haiku_vs_sonnet_high__11_full_vs_00_none"]["estimate"], 0.38, abs_tol=1e-8
    )


# ==============================================================================
# 3 & 4. All 3 Subagent Role Paths + Retries + Trace Completeness
# ==============================================================================


def _create_opencode_sqlite_fixture(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                title TEXT NOT NULL,
                agent TEXT,
                model TEXT,
                cost REAL DEFAULT 0,
                tokens_input INTEGER DEFAULT 0,
                tokens_output INTEGER DEFAULT 0,
                tokens_reasoning INTEGER DEFAULT 0,
                tokens_cache_read INTEGER DEFAULT 0,
                tokens_cache_write INTEGER DEFAULT 0,
                time_created INTEGER NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                data TEXT NOT NULL
            )
            """
        )

        # Root session + 2 child subagent sessions
        con.executemany(
            "INSERT INTO session VALUES (?, ?, ?, ?, ?, 0.0, 0, 0, 0, 0, 0, ?)",
            [
                (
                    "ses_root",
                    None,
                    "Orchestrate reduction",
                    "build",
                    '{"id":"kimi","providerID":"nrp","variant":"default"}',
                    1000,
                ),
                (
                    "ses_insp",
                    "ses_root",
                    "Inspect MS",
                    "inspect-stage",
                    '{"id":"kimi","providerID":"nrp","variant":"default"}',
                    2000,
                ),
                (
                    "ses_cal",
                    "ses_root",
                    "Run cal (@calibrate-stage subagent)",
                    "build",
                    '{"id":"kimi","providerID":"nrp","variant":"default"}',
                    3000,
                ),
            ],
        )

        parts = [
            # Root turn 1: dispatches inspect-stage and calibrate-stage
            ("p1", "m_r1", "ses_root", 1001, {"type": "step-start"}),
            (
                "p2",
                "m_r1",
                "ses_root",
                1002,
                {
                    "type": "tool",
                    "tool": "task",
                    "callID": "call_t_insp",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "inspect-stage", "prompt": "inspect"},
                        "output": '<task id="ses_insp" state="completed">ok</task>',
                        "metadata": {"sessionId": "ses_insp"},
                    },
                },
            ),
            (
                "p3",
                "m_r1",
                "ses_root",
                1003,
                {
                    "type": "tool",
                    "tool": "task",
                    "callID": "call_t_cal",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "calibrate-stage", "prompt": "cal"},
                        "output": '<task id="ses_cal" state="completed">ok</task>',
                        "metadata": {"sessionId": "ses_cal"},
                    },
                },
            ),
            (
                "p4",
                "m_r1",
                "ses_root",
                1004,
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 5_000,
                        "output": 300,
                        "reasoning": 200,
                        "cache": {"read": 1_000, "write": 0},
                    },
                },
            ),
            # Child 1 (`inspect-stage`): 1 clean turn calling MCP `ms_field_list`
            ("p5", "m_i1", "ses_insp", 2001, {"type": "step-start"}),
            (
                "p6",
                "m_i1",
                "ses_insp",
                2002,
                {
                    "type": "tool",
                    "tool": "ms-inspect_ms_field_list",
                    "callID": "call_i_mcp",
                    "state": {"status": "completed", "input": {}, "output": '{"fields": []}'},
                },
            ),
            (
                "p7",
                "m_i1",
                "ses_insp",
                2003,
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 8_000,
                        "output": 400,
                        "reasoning": 100,
                        "cache": {"read": 2_000, "write": 500},
                    },
                },
            ),
            # Child 2 (`calibrate-stage`): Turn 1 fails Bash exit=2 -> Turn 2 is retry!
            ("p8", "m_c1", "ses_cal", 3001, {"type": "step-start"}),
            (
                "p9",
                "m_c1",
                "ses_cal",
                3002,
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "call_c_bash_bad",
                    "state": {
                        "status": "completed",
                        "input": {"command": "python bad_cal.py"},
                        "output": "Traceback...",
                        "metadata": {"exit": 2},
                    },
                },
            ),
            (
                "p10",
                "m_c1",
                "ses_cal",
                3003,
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 10_000,
                        "output": 500,
                        "reasoning": 300,
                        "cache": {"read": 0, "write": 1_000},
                    },
                },
            ),
            ("p11", "m_c2", "ses_cal", 3004, {"type": "step-start"}),
            (
                "p12",
                "m_c2",
                "ses_cal",
                3005,
                {
                    "type": "tool",
                    "tool": "ms-modify_ms_gaincal",
                    "callID": "call_c_gain",
                    "state": {"status": "completed", "input": {}, "output": '{"ok": true}'},
                },
            ),
            (
                "p13",
                "m_c2",
                "ses_cal",
                3006,
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 4_000,
                        "output": 250,
                        "reasoning": 150,
                        "cache": {"read": 8_000, "write": 0},
                    },
                },
            ),
            # Root turn 2: NOT a retry because root turn 1 succeeded!
            ("p14", "m_r2", "ses_root", 4001, {"type": "step-start"}),
            (
                "p15",
                "m_r2",
                "ses_root",
                4002,
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 6_000,
                        "output": 200,
                        "reasoning": 50,
                        "cache": {"read": 4_000, "write": 0},
                    },
                },
            ),
        ]
        con.executemany(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
            [(pid, mid, sid, tc, json.dumps(d)) for pid, mid, sid, tc, d in parts],
        )
        con.commit()
    finally:
        con.close()


def test_subagent_roles_retries_and_trace_completeness_across_harnesses(tmp_path: Path) -> None:
    """Verify (a) ClaudeCode sidechains, (b) OpenCode SQLite, (c) ATIF subtrajs, retries, and completeness."""
    # --- (a) ClaudeCode 2.1.x inlined sidechains + .meta.json + retry stream isolation ---
    cc_job = tmp_path / "cc_sidechain_job"
    cc_job.mkdir()
    cc_traj = {
        "schema_version": "ATIF-v1.7",
        "session_id": "sess-cc-main",
        "agent": {"name": "claude-code", "version": "2.1.325", "model_name": "claude-sonnet-4-6"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Orchestrate e2e"},
            # Root turn 1: dispatches inspect-stage (linked via tool_use_result.agentId) & precal-stage (linked via .meta.json)
            {
                "step_id": 2,
                "source": "agent",
                "message": "Launching stages",
                "model_name": "claude-sonnet-4-6",
                "extra": {"is_sidechain": False},
                "metrics": {
                    "prompt_tokens": 15_000,
                    "completion_tokens": 600,
                    "cached_tokens": 5_000,
                    "extra": {
                        "cache_read_input_tokens": 5_000,
                        "cache_creation_input_tokens": 2_000,
                        "output_tokens_details": {"thinking_tokens": 200},
                    },
                },
                "tool_calls": [
                    {
                        "tool_call_id": "tu_insp",
                        "function_name": "Agent",
                        "arguments": {"subagent_type": "inspect-stage", "prompt": "inspect"},
                    },
                    {
                        "tool_call_id": "tu_pre",
                        "function_name": "Task",
                        "arguments": {"subagent_type": "general-purpose", "name": "precal-stage"},
                    },
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "tu_insp",
                            "content": "INSPECT HANDOFF",
                            "extra": {
                                "tool_result_metadata": {
                                    "tool_use_result": {"agentId": "ag_insp_01"}
                                }
                            },
                        },
                        {"source_call_id": "tu_pre", "content": "STAGE HANDOFF"},
                    ]
                },
            },
            # Sidechain 1 (inspect-stage): 1 clean turn
            {
                "step_id": 3,
                "source": "agent",
                "message": "Inspected",
                "model_name": "claude-haiku-4-5",
                "extra": {"is_sidechain": True, "agent_id": "ag_insp_01"},
                "metrics": {"prompt_tokens": 8_000, "completion_tokens": 400},
                "tool_calls": [
                    {
                        "tool_call_id": "tu_mcp_1",
                        "function_name": "mcp__ms-inspect__ms_observation_info",
                        "arguments": {},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "tu_mcp_1", "content": '{"ok": true}'}]
                },
            },
            # Sidechain 2 (precal-stage): Turn 1 fails Bash -> Turn 2 is retry!
            {
                "step_id": 4,
                "source": "agent",
                "message": "Trying precal",
                "model_name": "claude-sonnet-4-6",
                "extra": {"is_sidechain": True, "agent_id": "ag_pre_02"},
                "metrics": {
                    "prompt_tokens": 12_000,
                    "completion_tokens": 500,
                    "extra": {
                        "cache_creation_input_tokens": 4_000,
                        "cache_creation": {"ephemeral_1h_input_tokens": 2_000},
                    },
                },
                "tool_calls": [
                    {
                        "tool_call_id": "tu_bash_err",
                        "function_name": "Bash",
                        "arguments": {"command": "false"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "tu_bash_err",
                            "content": "[exit_code] 1",
                        }
                    ]
                },
            },
            # Root turn 2 interleaved right after sidechain error: MUST NOT be marked retry!
            {
                "step_id": 5,
                "source": "agent",
                "message": "Summary",
                "model_name": "claude-sonnet-4-6",
                "extra": {"is_sidechain": False},
                "metrics": {"prompt_tokens": 16_000, "completion_tokens": 300},
            },
            # Sidechain 2 Turn 2: MUST be marked retry!
            {
                "step_id": 6,
                "source": "agent",
                "message": "Fixed precal",
                "model_name": "claude-sonnet-4-6",
                "extra": {"is_sidechain": True, "agent_id": "ag_pre_02"},
                "metrics": {
                    "prompt_tokens": 14_000,
                    "completion_tokens": 450,
                    "cached_tokens": 10_000,
                    "extra": {"cache_read_input_tokens": 10_000},
                },
            },
        ],
    }
    cc_tdir = _write_trial_fixture(
        cc_job,
        task_name="s2_e2e_synth__11_orchestrated",
        agent_name="claude-code",
        model_name="anthropic/claude-sonnet-4-6",
        trajectory_dict=cc_traj,
    )
    # Write subagent .meta.json and matching session JSONLs for trace completeness reconciliation
    sub_dir = (
        cc_tdir / "agent" / "sessions" / "projects" / "-workspace" / "sess-cc-main" / "subagents"
    )
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-ag_pre_02.meta.json").write_text(
        json.dumps({"agentType": "precal-stage", "toolUseId": "tu_pre"}), encoding="utf-8"
    )
    main_jsonl = sub_dir.parent / "sess-cc-main.jsonl"
    jsonl_events = [
        {
            "uuid": "u1",
            "type": "assistant",
            "message": {
                "id": "m1",
                "content": [
                    {"type": "tool_use", "id": "tu_insp", "name": "Agent", "input": {}},
                    {"type": "tool_use", "id": "tu_pre", "name": "Task", "input": {}},
                ],
            },
        },
        {"uuid": "u2", "type": "assistant", "message": {"id": "m2", "content": []}},
    ]
    main_jsonl.write_text("\n".join(json.dumps(e) for e in jsonl_events) + "\n", encoding="utf-8")
    (sub_dir / "agent-ag_insp_01.jsonl").write_text(
        json.dumps(
            {
                "uuid": "u3",
                "type": "assistant",
                "message": {
                    "id": "m3",
                    "content": [{"type": "tool_use", "id": "tu_mcp_1", "name": "ms", "input": {}}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sub_dir / "agent-ag_pre_02.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "uuid": "u4",
                        "type": "assistant",
                        "message": {
                            "id": "m4",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tu_bash_err",
                                    "name": "Bash",
                                    "input": {},
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {"uuid": "u5", "type": "assistant", "message": {"id": "m5", "content": []}}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cc_rec = collect_single_trial(cc_job, cc_tdir)
    assert cc_rec["n_turns"] == 5
    assert cc_rec["retry_turn_count"] == 1
    assert abs((cc_rec["first_pass_usd"] + cc_rec["retry_usd"]) - cc_rec["total_usd"]) <= 1e-15
    assert cc_rec["reasoning_tokens"] == 200
    assert cc_rec["reply_output_tokens"] == cc_rec["output_tokens"] - 200
    # 5 assistant messages + 4 tool_use calls = 9 events
    assert cc_rec["trace_events_expected"] == 9
    assert cc_rec["trace_events_observed"] == 9
    assert cc_rec["trace_completeness_ok"] is True

    rb_cc = cc_rec["role_breakdown"]
    assert set(rb_cc.keys()) == {"orchestrator", "inspect-stage", "precal-stage"}
    assert rb_cc["orchestrator"]["turns"] == 2
    assert rb_cc["orchestrator"]["retry_turns"] == 0
    assert rb_cc["inspect-stage"]["turns"] == 1
    assert rb_cc["precal-stage"]["turns"] == 2
    assert rb_cc["precal-stage"]["retry_turns"] == 1
    assert rb_cc["precal-stage"]["retry_usd"] == pytest.approx(cc_rec["retry_usd"], abs=1e-8)

    # Truncate trajectory.json by dropping last step -> trace_completeness_ok MUST become False!
    cc_traj_trunc = dict(cc_traj, steps=cc_traj["steps"][:-1])
    (cc_tdir / "agent" / "trajectory.json").write_text(json.dumps(cc_traj_trunc), encoding="utf-8")
    cc_rec_trunc = collect_single_trial(cc_job, cc_tdir)
    assert cc_rec_trunc["trace_events_expected"] == 9
    assert cc_rec_trunc["trace_events_observed"] == 8
    assert cc_rec_trunc["trace_completeness_ok"] is False

    # --- (b) OpenCode 1.19.5 SQLite `opencode.db` child recovery (zero root double-count!) ---
    oc_job = tmp_path / "oc_sqlite_job"
    oc_job.mkdir()
    # Root trajectory.json ONLY contains the 2 root turns (omitting 100% of child subagent turns!)
    oc_root_traj = Trajectory(
        session_id="ses_root",
        agent=Agent(name="opencode", version="1.19.5", model_name="nrp/kimi"),
        steps=[
            Step(step_id=1, source="user", message="Orchestrate"),
            Step(
                step_id=2,
                source="agent",
                message="Dispatching",
                metrics=Metrics(
                    prompt_tokens=6_000,
                    completion_tokens=300,
                    cached_tokens=1_000,
                    extra={"reasoning_tokens": 200},
                ),
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_t_insp",
                        function_name="task",
                        arguments={"subagent_type": "inspect-stage"},
                    ),
                    ToolCall(
                        tool_call_id="call_t_cal",
                        function_name="task",
                        arguments={"subagent_type": "calibrate-stage"},
                    ),
                ],
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id="call_t_insp",
                            content='<task id="ses_insp" state="completed">ok</task>',
                        ),
                        ObservationResult(
                            source_call_id="call_t_cal",
                            content='<task id="ses_cal" state="completed">ok</task>',
                        ),
                    ]
                ),
            ),
            Step(
                step_id=3,
                source="agent",
                message="Done",
                metrics=Metrics(
                    prompt_tokens=10_000,
                    completion_tokens=200,
                    cached_tokens=4_000,
                    extra={"reasoning_tokens": 50},
                ),
            ),
        ],
    )
    oc_tdir = _write_trial_fixture(
        oc_job,
        task_name="s2_e2e_synth__11_orchestrated",
        agent_name="opencode",
        model_name="nrp/kimi",
        trajectory_dict=oc_root_traj.to_json_dict(),
    )
    db_path = oc_tdir / "agent" / "opencode" / "xdg-data" / "opencode" / "opencode.db"
    _create_opencode_sqlite_fixture(db_path)

    oc_rec = collect_single_trial(oc_job, oc_tdir)
    # 2 root turns + 1 inspect child turn + 2 calibrate child turns = 5 turns (root NOT double-counted!)
    assert oc_rec["n_turns"] == 5
    assert oc_rec["mcp_calls_total"] == 2
    assert oc_rec["mcp_calls_inspect"] == 1
    assert oc_rec["mcp_calls_modify"] == 1
    assert oc_rec["bash_calls"] == 1
    assert oc_rec["bash_nonzero_exits"] == 1
    assert oc_rec["retry_turn_count"] == 1
    # Total reasoning: (200 + 50) root + 100 inspect + (300 + 150) calibrate = 800
    # Total reply: (300 + 200) root + 400 inspect + (500 + 250) calibrate = 1650
    # Total billed output: 1650 + 800 = 2450
    assert oc_rec["reasoning_tokens"] == 800
    assert oc_rec["reply_output_tokens"] == 1_650
    assert oc_rec["output_tokens"] == 2_450
    assert abs((oc_rec["first_pass_usd"] + oc_rec["retry_usd"]) - oc_rec["total_usd"]) <= 1e-15

    # 5 step-finish + 5 tool calls = 10 events
    assert oc_rec["trace_events_expected"] == 10
    assert oc_rec["trace_events_observed"] == 10
    assert oc_rec["trace_completeness_ok"] is True

    rb_oc = oc_rec["role_breakdown"]
    assert set(rb_oc.keys()) == {"orchestrator", "inspect-stage", "calibrate-stage"}
    assert rb_oc["orchestrator"]["turns"] == 2
    assert rb_oc["inspect-stage"]["turns"] == 1
    assert rb_oc["calibrate-stage"]["turns"] == 2
    assert rb_oc["calibrate-stage"]["retry_turns"] == 1

    # Corrupt OpenCode trajectory.json -> trace_completeness_ok MUST be False!
    (oc_tdir / "agent" / "trajectory.json").write_text("{truncated_json", encoding="utf-8")
    oc_rec_corrupt = collect_single_trial(oc_job, oc_tdir)
    assert oc_rec_corrupt["trace_completeness_ok"] is False

    # --- (c) Standard ATIF `subagent_trajectories` + Oracle completeness check ---
    atif_job = tmp_path / "atif_subtraj_job"
    atif_job.mkdir()
    sub_img = Trajectory(
        trajectory_id="tid_img",
        session_id="s_img",
        agent=Agent(name="claude-code", version="1.0", model_name="claude-haiku-4-5"),
        steps=[
            Step(step_id=1, source="user", message="Image"),
            Step(
                step_id=2,
                source="agent",
                message="Imaged",
                metrics=Metrics(prompt_tokens=4_000, completion_tokens=200),
            ),
        ],
    )
    sub_rev = Trajectory(
        trajectory_id="tid_rev",
        session_id="s_rev",
        agent=Agent(
            name="claude-code",
            version="1.0",
            model_name="claude-haiku-4-5",
            extra={"subagent_name": "reviewer"},
        ),
        steps=[
            Step(step_id=1, source="user", message="Review"),
            Step(
                step_id=2,
                source="agent",
                message="Approved",
                metrics=Metrics(prompt_tokens=3_000, completion_tokens=150),
            ),
        ],
    )
    root_atif = Trajectory(
        session_id="s_root_atif",
        agent=Agent(name="claude-code", version="1.0", model_name="claude-sonnet-4-6"),
        steps=[
            Step(step_id=1, source="user", message="Go"),
            Step(
                step_id=2,
                source="agent",
                message="Dispatch",
                metrics=Metrics(prompt_tokens=5_000, completion_tokens=250),
                tool_calls=[
                    ToolCall(
                        tool_call_id="c_img",
                        function_name="Task",
                        arguments={"subagent_type": "image-stage"},
                    )
                ],
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id="c_img",
                            content="done",
                            subagent_trajectory_ref=[
                                SubagentTrajectoryRef(trajectory_id="tid_img")
                            ],
                        )
                    ]
                ),
            ),
        ],
        subagent_trajectories=[sub_img, sub_rev],
    )
    atif_tdir = _write_trial_fixture(
        atif_job,
        task_name="s2_e2e_synth__11_orchestrated",
        agent_name="claude-code",
        model_name="anthropic/claude-sonnet-4-6",
        trajectory_dict=root_atif.to_json_dict(),
    )
    atif_rec = collect_single_trial(atif_job, atif_tdir)
    assert set(atif_rec["role_breakdown"].keys()) == {"orchestrator", "image-stage", "reviewer"}
    assert atif_rec["trace_completeness_ok"] is True

    # Oracle trial completeness: (0, 0, True)
    ora_tdir = _write_trial_fixture(
        atif_job,
        task_name="s1_inspect__00_none",
        agent_name="oracle",
        model_name=None,
        trajectory_dict=None,
    )
    ora_rec = collect_single_trial(atif_job, ora_tdir)
    assert ora_rec["trace_events_expected"] == 0
    assert ora_rec["trace_events_observed"] == 0
    assert ora_rec["trace_completeness_ok"] is True
    assert ora_rec["reasoning_tokens"] is None
    assert ora_rec["reply_output_tokens"] is None


# ==============================================================================
# 5. Real CASA 6.7.6.14 `diff-run` Verification (Visibilities, Caltables, Images)
# ==============================================================================


def _create_synthetic_ms_for_diff(
    casatools: Any,
    ms_path: Path,
    *,
    field_names: list[str],
    field_intents: list[str],
    ant_names: list[str],
    chan_freqs_hz: np.ndarray,
    rows: list[dict[str, Any]],
    corr_types: list[int] | None = None,
) -> None:
    """Create a real casacore MS with `FIELD`, `ANTENNA`, `DATA_DESCRIPTION`, `SPECTRAL_WINDOW`, `POLARIZATION`, `STATE`."""
    ms_path.mkdir(parents=True, exist_ok=True)
    nchan = int(chan_freqs_hz.size)
    nrow = len(rows)

    def _sdesc(vtype: str) -> dict[str, Any]:
        return {
            "valueType": vtype,
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "",
        }

    def _adesc(vtype: str, shape: list[int]) -> dict[str, Any]:
        return {
            "valueType": vtype,
            "ndim": len(shape),
            "shape": np.array(shape, dtype=np.int32),
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "",
        }

    tb = casatools.table()
    try:
        assert tb.create(
            str(ms_path),
            {
                "TIME": _sdesc("double"),
                "ANTENNA1": _sdesc("int"),
                "ANTENNA2": _sdesc("int"),
                "FIELD_ID": _sdesc("int"),
                "DATA_DESC_ID": _sdesc("int"),
                "SCAN_NUMBER": _sdesc("int"),
                "STATE_ID": _sdesc("int"),
                "FLAG_ROW": _sdesc("boolean"),
                "DATA": _adesc("complex", [4, nchan]),
                "CORRECTED_DATA": _adesc("complex", [4, nchan]),
                "FLAG": _adesc("boolean", [4, nchan]),
            },
        )
        tb.addrows(nrow)
        tb.putcol("TIME", np.array([r["time"] for r in rows], dtype=np.float64))
        tb.putcol("ANTENNA1", np.array([r["a1"] for r in rows], dtype=np.int32))
        tb.putcol("ANTENNA2", np.array([r["a2"] for r in rows], dtype=np.int32))
        tb.putcol("FIELD_ID", np.array([r["fid"] for r in rows], dtype=np.int32))
        tb.putcol("DATA_DESC_ID", np.zeros(nrow, dtype=np.int32))
        tb.putcol("SCAN_NUMBER", np.array([r["scan"] for r in rows], dtype=np.int32))
        tb.putcol("STATE_ID", np.array([r["fid"] for r in rows], dtype=np.int32))
        tb.putcol("FLAG_ROW", np.zeros(nrow, dtype=bool))
        vis_arr = np.stack([r["vis"] for r in rows], axis=2).astype(np.complex64)
        flg_arr = np.stack([r["flag"] for r in rows], axis=2).astype(bool)
        corr_arr = np.stack([r.get("corr_vis", r["vis"]) for r in rows], axis=2).astype(
            np.complex64
        )
        tb.putcol("DATA", vis_arr)
        tb.putcol("CORRECTED_DATA", corr_arr)
        tb.putcol("FLAG", flg_arr)
    finally:
        with contextlib.suppress(Exception):
            tb.close()
        with contextlib.suppress(Exception):
            tb.done()

    # Subtables
    tb_f = casatools.table()
    try:
        assert tb_f.create(str(ms_path / "FIELD"), {"NAME": _sdesc("string")})
        tb_f.addrows(len(field_names))
        tb_f.putcol("NAME", field_names)
    finally:
        with contextlib.suppress(Exception):
            tb_f.close()
        with contextlib.suppress(Exception):
            tb_f.done()

    tb_a = casatools.table()
    try:
        assert tb_a.create(str(ms_path / "ANTENNA"), {"NAME": _sdesc("string")})
        tb_a.addrows(len(ant_names))
        tb_a.putcol("NAME", ant_names)
    finally:
        with contextlib.suppress(Exception):
            tb_a.close()
        with contextlib.suppress(Exception):
            tb_a.done()

    tb_dd = casatools.table()
    try:
        assert tb_dd.create(
            str(ms_path / "DATA_DESCRIPTION"),
            {"SPECTRAL_WINDOW_ID": _sdesc("int"), "POLARIZATION_ID": _sdesc("int")},
        )
        tb_dd.addrows(1)
        tb_dd.putcol("SPECTRAL_WINDOW_ID", np.array([0], dtype=np.int32))
        tb_dd.putcol("POLARIZATION_ID", np.array([0], dtype=np.int32))
    finally:
        with contextlib.suppress(Exception):
            tb_dd.close()
        with contextlib.suppress(Exception):
            tb_dd.done()

    tb_spw = casatools.table()
    try:
        assert tb_spw.create(
            str(ms_path / "SPECTRAL_WINDOW"),
            {"CHAN_FREQ": _adesc("double", [nchan]), "NUM_CHAN": _sdesc("int")},
        )
        tb_spw.addrows(1)
        tb_spw.putcell("CHAN_FREQ", 0, chan_freqs_hz.astype(np.float64))
        tb_spw.putcol("NUM_CHAN", np.array([nchan], dtype=np.int32))
    finally:
        with contextlib.suppress(Exception):
            tb_spw.close()
        with contextlib.suppress(Exception):
            tb_spw.done()

    eff_ct = corr_types if corr_types is not None else [5, 6, 7, 8]
    tb_p = casatools.table()
    try:
        assert tb_p.create(
            str(ms_path / "POLARIZATION"),
            {"CORR_TYPE": _adesc("int", [len(eff_ct)]), "NUM_CORR": _sdesc("int")},
        )
        tb_p.addrows(1)
        tb_p.putcell("CORR_TYPE", 0, np.array(eff_ct, dtype=np.int32))
        tb_p.putcol("NUM_CORR", np.array([len(eff_ct)], dtype=np.int32))
    finally:
        with contextlib.suppress(Exception):
            tb_p.close()
        with contextlib.suppress(Exception):
            tb_p.done()

    tb_st = casatools.table()
    try:
        assert tb_st.create(str(ms_path / "STATE"), {"OBS_MODE": _sdesc("string")})
        tb_st.addrows(len(field_intents))
        tb_st.putcol("OBS_MODE", field_intents)
    finally:
        with contextlib.suppress(Exception):
            tb_st.close()
        with contextlib.suppress(Exception):
            tb_st.done()


def _create_caltable_fixture(
    casatools: Any,
    ct_path: Path,
    *,
    viscal: str,
    is_complex: bool,
    param: np.ndarray,
    flag: np.ndarray,
    ant_ids: np.ndarray,
    field_ids: np.ndarray,
    times: np.ndarray,
    scans: np.ndarray,
    ant_names: list[str],
    field_names: list[str],
    chan_freqs_hz: np.ndarray,
) -> None:
    ct_path.mkdir(parents=True, exist_ok=True)
    npol, nchan, nrow = param.shape

    def _sdesc(vtype: str) -> dict[str, Any]:
        return {
            "valueType": vtype,
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "",
        }

    p_col = "CPARAM" if is_complex else "FPARAM"
    p_type = "complex" if is_complex else "float"
    tb = casatools.table()
    try:
        assert tb.create(
            str(ct_path),
            {
                "TIME": _sdesc("double"),
                "ANTENNA1": _sdesc("int"),
                "FIELD_ID": _sdesc("int"),
                "SPECTRAL_WINDOW_ID": _sdesc("int"),
                "SCAN_NUMBER": _sdesc("int"),
                p_col: {
                    "valueType": p_type,
                    "ndim": 2,
                    "shape": np.array([npol, nchan], dtype=np.int32),
                    "dataManagerType": "StandardStMan",
                    "dataManagerGroup": "StandardStMan",
                    "option": 0,
                    "maxlen": 0,
                    "comment": "",
                },
                "FLAG": {
                    "valueType": "boolean",
                    "ndim": 2,
                    "shape": np.array([npol, nchan], dtype=np.int32),
                    "dataManagerType": "StandardStMan",
                    "dataManagerGroup": "StandardStMan",
                    "option": 0,
                    "maxlen": 0,
                    "comment": "",
                },
            },
        )
        tb.addrows(nrow)
        tb.putkeyword("VisCal", viscal)
        tb.putcol("TIME", times.astype(np.float64))
        tb.putcol("ANTENNA1", ant_ids.astype(np.int32))
        tb.putcol("FIELD_ID", field_ids.astype(np.int32))
        tb.putcol("SPECTRAL_WINDOW_ID", np.zeros(nrow, dtype=np.int32))
        tb.putcol("SCAN_NUMBER", scans.astype(np.int32))
        tb.putcol(p_col, param.astype(np.complex64 if is_complex else np.float32))
        tb.putcol("FLAG", flag.astype(bool))
    finally:
        with contextlib.suppress(Exception):
            tb.close()
        with contextlib.suppress(Exception):
            tb.done()

    for sub_name, vals in (("ANTENNA", ant_names), ("FIELD", field_names)):
        tbs = casatools.table()
        try:
            assert tbs.create(str(ct_path / sub_name), {"NAME": _sdesc("string")})
            tbs.addrows(len(vals))
            tbs.putcol("NAME", vals)
        finally:
            with contextlib.suppress(Exception):
                tbs.close()
            with contextlib.suppress(Exception):
                tbs.done()

    tb_spw = casatools.table()
    try:
        assert tb_spw.create(
            str(ct_path / "SPECTRAL_WINDOW"),
            {
                "CHAN_FREQ": {
                    "valueType": "double",
                    "ndim": 1,
                    "shape": np.array([int(chan_freqs_hz.size)], dtype=np.int32),
                    "dataManagerType": "StandardStMan",
                    "dataManagerGroup": "StandardStMan",
                    "option": 0,
                    "maxlen": 0,
                    "comment": "",
                }
            },
        )
        tb_spw.addrows(1)
        tb_spw.putcell("CHAN_FREQ", 0, chan_freqs_hz.astype(np.float64))
    finally:
        with contextlib.suppress(Exception):
            tb_spw.close()
        with contextlib.suppress(Exception):
            tb_spw.done()


def _create_convolved_sky_image_set(
    casatools: Any,
    ws_dir: Path,
    stem: str,
    *,
    npix: int,
    cell_arcsec: float,
    b_major_arcsec: float,
    b_minor_arcsec: float,
    b_pa_deg: float,
) -> None:
    """Create a noiseless sky convolved analytically with `(b_major, b_minor, b_pa)` in Jy/beam."""
    from casatasks import imsmooth

    ws_dir.mkdir(parents=True, exist_ok=True)
    # Build true point-like + extended sky on a fine grid in Jy/pixel, then convolve to target beam!
    pix = np.zeros((npix, npix, 1, 1), dtype=np.float64)
    cx = (npix - 1) / 2.0
    cy = (npix - 1) / 2.0
    ix, iy = np.meshgrid(np.arange(npix, dtype=float), np.arange(npix, dtype=float), indexing="ij")
    dx_arcsec = (ix - cx) * (-cell_arcsec)  # RA increases left (negative increment)
    dy_arcsec = (iy - cy) * cell_arcsec

    # Extended Gaussian source centered at (0, 0) + compact source at (+18", +14")
    sigma_ext = 14.0
    ext_jy_per_arcsec2 = 0.80 / (2.0 * math.pi * sigma_ext**2)
    sigma_pt = 3.0
    pt_jy_per_arcsec2 = 0.25 / (2.0 * math.pi * sigma_pt**2)

    sky_jy_per_pix = (
        ext_jy_per_arcsec2 * np.exp(-0.5 * (dx_arcsec**2 + dy_arcsec**2) / sigma_ext**2)
        + pt_jy_per_arcsec2
        * np.exp(-0.5 * ((dx_arcsec - 18.0) ** 2 + (dy_arcsec - 14.0) ** 2) / sigma_pt**2)
    ) * (cell_arcsec**2)
    pix[:, :, 0, 0] = sky_jy_per_pix

    raw_sky_path = ws_dir / f".tmp_{stem}_sky.im"
    ia = casatools.image()
    try:
        ia.fromarray(outfile=str(raw_sky_path), pixels=pix, overwrite=True)
        cs = ia.coordsys()
        try:
            inc_rad = math.radians(cell_arcsec / 3600.0)
            cs.setunits(["rad", "rad", "", "Hz"])
            cs.setreferencepixel([cx, cy, 0.0, 0.0])
            cs.setreferencevalue([math.radians(276.0), math.radians(-9.58), 1.0, 5.0e9])
            cs.setincrement([-inc_rad, inc_rad, 1.0, 1.0e8])
            ia.setcoordsys(cs.torecord())
        finally:
            cs.done()
        ia.setbrightnessunit("Jy/pixel")
    finally:
        ia.close()
        ia.done()

    pbcor_path = ws_dir / f"{stem}.image.pbcor"
    img_path = ws_dir / f"{stem}.image"
    pb_path = ws_dir / f"{stem}.pb"
    mod_path = ws_dir / f"{stem}.model"
    res_path = ws_dir / f"{stem}.residual"

    try:
        imsmooth(
            imagename=str(raw_sky_path),
            kernel="gauss",
            major=f"{b_major_arcsec:.6f}arcsec",
            minor=f"{b_minor_arcsec:.6f}arcsec",
            pa=f"{b_pa_deg:.4f}deg",
            targetres=False,
            outfile=str(pbcor_path),
            overwrite=True,
        )
    finally:
        shutil.rmtree(raw_sky_path, ignore_errors=True)

    # Create matching .pb, .image (= pbcor * pb), .model, .residual
    r_rad = np.hypot(dx_arcsec, dy_arcsec) / 3600.0 * (math.pi / 180.0)
    theta_pb = 1.13 * (299792458.0 / 5.0e9) / 25.0
    pb_2d = np.clip(np.exp(-4.0 * math.log(2.0) * (r_rad / theta_pb) ** 2), 0.25, 1.0)

    ia = casatools.image()
    try:
        assert ia.open(str(pbcor_path))
        pbcor_chunk = np.asarray(ia.getchunk(), dtype=np.float64)
        cs_obj = ia.coordsys()
        try:
            cs_rec = cs_obj.torecord()
        finally:
            cs_obj.done()
        rb = ia.restoringbeam()
    finally:
        with contextlib.suppress(Exception):
            ia.close()
        with contextlib.suppress(Exception):
            ia.done()

    for out_p, arr_4d, bunit, set_rb in (
        (pb_path, pb_2d[:, :, None, None], "", False),
        (img_path, pbcor_chunk * pb_2d[:, :, None, None], "Jy/beam", True),
        (mod_path, pix, "Jy/pixel", False),
        (res_path, np.full_like(pbcor_chunk, 1.0e-5), "Jy/beam", True),
    ):
        ia_w = casatools.image()
        try:
            ia_w.fromarray(outfile=str(out_p), pixels=arr_4d, csys=cs_rec, overwrite=True)
            if bunit:
                ia_w.setbrightnessunit(bunit)
            if set_rb and isinstance(rb, dict) and "major" in rb:
                pa_k = "positionangle" if "positionangle" in rb else "pa"
                ia_w.setrestoringbeam(major=rb["major"], minor=rb["minor"], pa=rb[pa_k])
        finally:
            ia_w.close()
            ia_w.done()


def test_casa_diff_visibilities_caltables_images_and_runner(tmp_path: Path) -> None:
    """Verify self-diff, 3C391 scan-collision & conj vis diff, gauge-fixed K/B/G/F/Df, and crossed-beam images."""
    casatools = _ensure_local_casa()
    rng = np.random.default_rng(20261117)

    freqs = np.linspace(4.5e9, 4.65e9, 16)
    ref_ws = tmp_path / "ref_ws"
    agt_ws = tmp_path / "agt_ws"
    ref_ws.mkdir()
    agt_ws.mkdir()

    # --- 5a. Visibility MS with Scan 1 vs Scan 2 boundary timestamp collisions + sub-ms occ_idx ---
    ref_fields = ["J1331+3030", "J1822-0938", "TARGET_SRC"]
    ref_intents = [
        "CALIBRATE_FLUX,CALIBRATE_BANDPASS",
        "CALIBRATE_PHASE",
        "OBSERVE_TARGET#ON_SOURCE",
    ]
    ref_ants = ["ea01", "ea02", "ea03", "ea04"]

    t_base = 5.1000e9
    ref_rows: list[dict[str, Any]] = []
    # Include exact same boundary timestamp `t_base + 10.0` in Scan 1 AND Scan 2 on Field 0
    # with distinct complex visibilities (650-row 3C391 collision defense!) + duplicate occ_idx in Scan 2!
    for scan_num, scale, times in (
        (1, 7.5 + 0.2j, [t_base, t_base + 10.0]),
        (2, 2.2 - 1.4j, [t_base + 10.0, t_base + 10.0, t_base + 20.0]),
    ):
        for t_idx, tv in enumerate(times):
            for a1 in range(4):
                for a2 in range(a1 + 1, 4):
                    vis = np.zeros((4, 16), dtype=np.complex128)
                    phs = 0.15 * (a2 - a1) + 0.05 * t_idx
                    vis[0] = scale * np.exp(1j * phs)  # RR
                    vis[3] = (scale * 0.98) * np.exp(-1j * phs)  # LL
                    vis[1] = 0.12 + 0.07j  # RL
                    vis[2] = -0.04 + 0.09j  # LR
                    flg = np.zeros((4, 16), dtype=bool)
                    flg[:, 0] = True  # edge channel flagged
                    ref_rows.append(
                        {
                            "time": tv,
                            "a1": a1,
                            "a2": a2,
                            "fid": 0,
                            "scan": scan_num,
                            "vis": vis,
                            "flag": flg,
                        }
                    )

    # Also add phase_cal (fid=1) and target (fid=2) rows
    for fid, sc, amp in ((1, 3, 2.15), (2, 4, 0.35)):
        for a1 in range(4):
            for a2 in range(a1 + 1, 4):
                vis = np.ones((4, 16), dtype=np.complex128) * (amp * np.exp(1j * 0.2 * (a2 - a1)))
                flg = np.zeros((4, 16), dtype=bool)
                ref_rows.append(
                    {
                        "time": t_base + 50.0 + fid * 10.0,
                        "a1": a1,
                        "a2": a2,
                        "fid": fid,
                        "scan": sc,
                        "vis": vis,
                        "flag": flg,
                    }
                )

    _create_synthetic_ms_for_diff(
        casatools,
        ref_ws / "input.ms",
        field_names=ref_fields,
        field_intents=ref_intents,
        ant_names=ref_ants,
        chan_freqs_hz=freqs,
        rows=ref_rows,
    )

    # 1) Self-diff on MS == 0.0 NRMSE, 1.0 Jaccard
    self_vis = diff_visibilities(ref_ws / "input.ms", ref_ws / "input.ms", stage="calibrate")
    assert self_vis["status"] == "ok"
    assert self_vis["overall"]["matched_rows"] == len(ref_rows)
    assert self_vis["overall"]["flag_jaccard"] == 1.0
    assert self_vis["overall"]["parallel_hand_nrmse"] == 0.0
    assert self_vis["overall"]["parallel_hand_phase_rms_deg"] == 0.0

    # 2) Agent MS with reversed antenna table order (`ea04..ea01`), permuted field IDs,
    #    flipped `ANT1 > ANT2` baselines (conjugated + RL<->LR swapped), and shuffled rows
    #    (while keeping intra-scan duplicate occ_idx relative order intact)!
    agt_ants = ["ea04", "ea03", "ea02", "ea01"]
    agt_fields = ["TARGET_SRC", "J1331+3030", "J1822-0938"]
    agt_intents = [
        "OBSERVE_TARGET#ON_SOURCE",
        "CALIBRATE_FLUX,CALIBRATE_BANDPASS",
        "CALIBRATE_PHASE",
    ]
    ref_to_agt_ant = {0: 3, 1: 2, 2: 1, 3: 0}
    ref_to_agt_fid = {0: 1, 1: 2, 2: 0}

    agt_rows: list[dict[str, Any]] = []
    for r in ref_rows:
        # Swap physical baseline direction: ANTENNA1 = agt_idx(a2), ANTENNA2 = agt_idx(a1)
        # so canonical `a1_mapped > a2_mapped` (`ea02 -> ea01` instead of `ea01 -> ea02`),
        # whose physical visibility is `conj(V)` with RL(1) <-> LR(2) swapped!
        aa1 = ref_to_agt_ant[r["a2"]]
        aa2 = ref_to_agt_ant[r["a1"]]
        v_flip = np.conj(r["vis"][[0, 2, 1, 3], :])
        f_flip = r["flag"][[0, 2, 1, 3], :].copy()
        agt_rows.append(
            {
                "time": r["time"],
                "a1": aa1,
                "a2": aa2,
                "fid": ref_to_agt_fid[r["fid"]],
                "scan": r["scan"],
                "vis": v_flip,
                "flag": f_flip,
            }
        )
    # Reverse scan blocks so Scan 4, Scan 3, Scan 2, Scan 1 appear out of order
    agt_rows_reordered = [
        *[r for r in agt_rows if r["scan"] == 4],
        *[r for r in agt_rows if r["scan"] == 2],
        *[r for r in agt_rows if r["scan"] == 3],
        *[r for r in agt_rows if r["scan"] == 1],
    ]
    _create_synthetic_ms_for_diff(
        casatools,
        agt_ws / "input.ms",
        field_names=agt_fields,
        field_intents=agt_intents,
        ant_names=agt_ants,
        chan_freqs_hz=freqs,
        rows=agt_rows_reordered,
    )

    reord_vis = diff_visibilities(agt_ws / "input.ms", ref_ws / "input.ms", stage="calibrate")
    assert reord_vis["overall"]["matched_rows"] == len(ref_rows)
    assert reord_vis["overall"]["flag_jaccard"] == 1.0
    assert reord_vis["overall"]["parallel_hand_nrmse"] < 1e-6
    assert reord_vis["overall"]["parallel_hand_phase_rms_deg"] < 1e-4
    assert reord_vis["overall"]["cross_hand_nrmse"] < 1e-6
    assert set(reord_vis["by_role"].keys()) == {"flux_cal", "bandpass_cal", "phase_cal", "target"}

    # --- 5b. Caltables (`K`, `B`, `G0`, `G`, `F`, `Df`) under changed `refant` (`ea01` vs `ea03`) ---
    n_ant = 4
    ant_ids_4 = np.arange(n_ant, dtype=np.int32)
    fids_0 = np.zeros(n_ant, dtype=np.int32)
    times_0 = np.full(n_ant, t_base, dtype=np.float64)
    scans_1 = np.ones(n_ant, dtype=np.int32)
    flg_k = np.zeros((2, 1, n_ant), dtype=bool)

    # True delays relative to ea01 (idx 0): [0.0, +3.5, -2.0, +6.2] ns
    tau_true = np.array([[[0.0, 3.5, -2.0, 6.2]], [[0.0, -1.5, 4.0, -3.1]]], dtype=np.float64)
    # Agent solved with refant = ea03 (idx 2 in ref_ants -> subtract tau_true[:, :, 2:3])!
    tau_agt_ea03 = tau_true - tau_true[:, :, 2:3]
    _create_caltable_fixture(
        casatools,
        ref_ws / "delay.k",
        viscal="K Jones",
        is_complex=False,
        param=tau_true,
        flag=flg_k,
        ant_ids=ant_ids_4,
        field_ids=fids_0,
        times=times_0,
        scans=scans_1,
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )
    _create_caltable_fixture(
        casatools,
        agt_ws / "custom_k_table.cal",
        viscal="K Jones",
        is_complex=False,
        param=tau_agt_ea03,
        flag=flg_k,
        ant_ids=ant_ids_4,
        field_ids=fids_0,
        times=times_0,
        scans=scans_1,
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )

    # B Jones: change refant to ea03 + multiply agent per-antenna amplitude by 2.5x (`solnorm` invariance!)
    bp_phs = rng.uniform(-0.8, 0.8, size=(2, 16, n_ant))
    bp_phs[:, :, 0] = 0.0  # refant ea01
    bp_amp = 1.0 + 0.08 * np.sin(np.linspace(0, math.pi, 16))[None, :, None]
    bp_ref = bp_amp * np.exp(1j * bp_phs)
    bp_agt = (2.5 * bp_amp) * np.exp(1j * (bp_phs - bp_phs[:, :, 2:3]))
    flg_b = np.zeros((2, 16, n_ant), dtype=bool)
    _create_caltable_fixture(
        casatools,
        ref_ws / "bp.b",
        viscal="B Jones",
        is_complex=True,
        param=bp_ref,
        flag=flg_b,
        ant_ids=ant_ids_4,
        field_ids=fids_0,
        times=times_0,
        scans=scans_1,
        ant_names=ref_fields if False else ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )
    _create_caltable_fixture(
        casatools,
        agt_ws / "my_bp.tbl",
        viscal="B Jones",
        is_complex=True,
        param=bp_agt,
        flag=flg_b,
        ant_ids=ant_ids_4,
        field_ids=fids_0,
        times=times_0,
        scans=scans_1,
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )

    # Df Jones: coupled first-order RIME leakage gauge shift `(D0 - k, D1 + conj(k))` for refant ea03!
    d_ref = rng.normal(0.0, 0.02, size=(2, 16, n_ant)) + 1j * rng.normal(
        0.0, 0.02, size=(2, 16, n_ant)
    )
    d_ref[0, :, 0] = 0.0 + 0.0j  # refant ea01 D0 == 0
    k_shift = d_ref[0, :, 2:3].copy()  # shift so ea03 has D0 == 0!
    d_agt = np.empty_like(d_ref)
    d_agt[0] = d_ref[0] - k_shift[0]
    d_agt[1] = d_ref[1] + np.conj(k_shift[0])
    _create_caltable_fixture(
        casatools,
        ref_ws / "leak.df",
        viscal="Df Jones",
        is_complex=True,
        param=d_ref,
        flag=flg_b,
        ant_ids=ant_ids_4,
        field_ids=fids_0,
        times=times_0,
        scans=scans_1,
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )
    _create_caltable_fixture(
        casatools,
        agt_ws / "agt_leak.cal",
        viscal="Df Jones",
        is_complex=True,
        param=d_agt,
        flag=flg_b,
        ant_ids=ant_ids_4,
        field_ids=fids_0,
        times=times_0,
        scans=scans_1,
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )

    # G + Fluxscale (`S_ref = 2.00 Jy`, `S_agt = 2.40 Jy` -> planted `1.20x` fluxscale shift!)
    # PLUS `G0` with `solint='30s'` (`0, 30, 60s`, refant=ea01) vs `solint='20s'` (`0, 20, 40, 60s`, refant=ea03)
    # on a linearly drifting phase scan so knots `20s` and `40s` exercise true between-knot interpolation!
    g0_phs_ant = np.array([0.0, 0.35, -0.45, 0.20])
    g0_ref_list: list[np.ndarray] = []
    g0_ref_t: list[np.ndarray] = []
    for dt_s in (0.0, 30.0, 60.0):
        drift = np.array(
            [0.0, 0.002 * (dt_s - 30.0), -0.001 * (dt_s - 30.0), 0.003 * (dt_s - 30.0)]
        )
        phs_now = g0_phs_ant + drift
        g0_ref_list.append(np.exp(1j * np.stack([phs_now, -phs_now], axis=0))[:, None, :])
        g0_ref_t.append(np.full(n_ant, t_base + dt_s))
    _create_caltable_fixture(
        casatools,
        ref_ws / "init.g0",
        viscal="G Jones",
        is_complex=True,
        param=np.concatenate(g0_ref_list, axis=2),
        flag=np.zeros((2, 1, 3 * n_ant), dtype=bool),
        ant_ids=np.tile(ant_ids_4, 3),
        field_ids=np.zeros(3 * n_ant, dtype=np.int32),
        times=np.concatenate(g0_ref_t),
        scans=np.ones(3 * n_ant, dtype=np.int32),
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )
    # Agent G0 solved at 4 timestamps (`0, 20, 40, 60s`, strictly straddling `t_base + 30s`)
    # with linear phase drift `+0.002 * (dt_s - 30.0)` and refant = ea03 (`- g0_phs_ant[2]`)!
    g0_agt_list: list[np.ndarray] = []
    g0_t_list: list[np.ndarray] = []
    for dt_s in (0.0, 20.0, 40.0, 60.0):
        drift = np.array(
            [0.0, 0.002 * (dt_s - 30.0), -0.001 * (dt_s - 30.0), 0.003 * (dt_s - 30.0)]
        )
        phs_now = g0_phs_ant + drift
        p_t = np.stack([phs_now - phs_now[2], (-phs_now) - (-phs_now[2])], axis=0)[:, None, :]
        g0_agt_list.append(np.exp(1j * p_t))
        g0_t_list.append(np.full(n_ant, t_base + dt_s))
    _create_caltable_fixture(
        casatools,
        agt_ws / "phase20s.g0",
        viscal="G Jones",
        is_complex=True,
        param=np.concatenate(g0_agt_list, axis=2),
        flag=np.zeros((2, 1, 4 * n_ant), dtype=bool),
        ant_ids=np.tile(ant_ids_4, 4),
        field_ids=np.zeros(4 * n_ant, dtype=np.int32),
        times=np.concatenate(g0_t_list),
        scans=np.ones(4 * n_ant, dtype=np.int32),
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )

    # Complex `ap` G + F (`fid=0` flux_cal + `fid=1` phase_cal): S_ref = 2.0 Jy, S_agt = 2.4 Jy (1.20x)
    sys_g = np.array([0.85, 0.88, 0.82, 0.90])
    s_ref = 2.00
    s_agt = 2.40  # 1.20 * s_ref
    g_fids = np.concatenate([np.zeros(n_ant, dtype=np.int32), np.ones(n_ant, dtype=np.int32)])
    g_ants = np.tile(ant_ids_4, 2)
    g_times = np.concatenate([np.full(n_ant, t_base), np.full(n_ant, t_base + 60.0)])
    g_scans = np.concatenate([np.ones(n_ant, dtype=np.int32), np.full(n_ant, 3, dtype=np.int32)])
    g_flg = np.zeros((2, 1, 2 * n_ant), dtype=bool)

    g_unsc_amp = np.concatenate([sys_g, sys_g * math.sqrt(s_ref)])
    g_unsc_param = np.broadcast_to(g_unsc_amp[None, None, :], (2, 1, 2 * n_ant)).astype(
        np.complex128
    )
    f_ref_amp = np.concatenate([sys_g, sys_g])  # divided by sqrt(s_ref)
    f_ref_param = np.broadcast_to(f_ref_amp[None, None, :], (2, 1, 2 * n_ant)).astype(np.complex128)
    f_agt_amp = np.concatenate([sys_g, (sys_g * math.sqrt(s_ref)) / math.sqrt(s_agt)])
    f_agt_param = np.broadcast_to(f_agt_amp[None, None, :], (2, 1, 2 * n_ant)).astype(np.complex128)

    for ws, g_p, f_p in ((ref_ws, g_unsc_param, f_ref_param), (agt_ws, g_unsc_param, f_agt_param)):
        _create_caltable_fixture(
            casatools,
            ws / "gain.g",
            viscal="G Jones",
            is_complex=True,
            param=g_p,
            flag=g_flg,
            ant_ids=g_ants,
            field_ids=g_fids,
            times=g_times,
            scans=g_scans,
            ant_names=ref_ants,
            field_names=ref_fields,
            chan_freqs_hz=freqs,
        )
        _create_caltable_fixture(
            casatools,
            ws / "flux.cal",
            viscal="G Jones",
            is_complex=True,
            param=f_p,
            flag=g_flg,
            ant_ids=g_ants,
            field_ids=g_fids,
            times=g_times,
            scans=g_scans,
            ant_names=ref_ants,
            field_names=ref_fields,
            chan_freqs_hz=freqs,
        )

    cal_diff = diff_caltables(agt_ws, ref_ws)
    assert cal_diff["status"] == "ok"
    assert cal_diff["K"]["delay_rms_ns"] == pytest.approx(0.0, abs=1e-5)
    assert cal_diff["K"]["delay_max_abs_ns"] == pytest.approx(0.0, abs=1e-5)
    assert cal_diff["B"]["bp_phase_rms_deg"] == pytest.approx(0.0, abs=1e-4)
    assert cal_diff["B"]["bp_amp_scale_ratio"] == pytest.approx(2.5, rel=1e-4)
    assert cal_diff["B"]["bp_norm_amp_nrmse"] == pytest.approx(0.0, abs=1e-5)
    assert cal_diff["D"]["dterm_rms"] == pytest.approx(0.0, abs=1e-6)
    assert cal_diff["G"]["G0_phase_only"]["gain_phase_rms_deg"] == pytest.approx(0.0, abs=1e-4)
    assert cal_diff["G"]["fluxscale_ratio"] == pytest.approx(1.20, rel=1e-4)

    # Plant +5.0 ns delay offset on ea04 (idx 3) in agent K table -> recovered cleanly!
    tau_planted = tau_agt_ea03.copy()
    tau_planted[0, 0, 3] += 5.0
    shutil.rmtree(agt_ws / "custom_k_table.cal")
    _create_caltable_fixture(
        casatools,
        agt_ws / "custom_k_table.cal",
        viscal="K Jones",
        is_complex=False,
        param=tau_planted,
        flag=flg_k,
        ant_ids=ant_ids_4,
        field_ids=fids_0,
        times=times_0,
        scans=scans_1,
        ant_names=ref_ants,
        field_names=ref_fields,
        chan_freqs_hz=freqs,
    )
    cal_diff_planted = diff_caltables(agt_ws, ref_ws)
    assert cal_diff_planted["K"]["delay_max_abs_ns"] == pytest.approx(5.0, abs=1e-4)

    # --- 5c. Image Common-Beam Diff: Identical Beam (no AxialRatio crash!) + Crossed Beams + imregrid ---
    _create_convolved_sky_image_set(
        casatools,
        ref_ws,
        "ref_sky",
        npix=104,
        cell_arcsec=2.5,
        b_major_arcsec=11.0,
        b_minor_arcsec=9.0,
        b_pa_deg=105.0,
    )
    # Self-diff on identical beams -> skips imsmooth cleanly, NCC == 1.0
    self_img_diff = diff_images(ref_ws, ref_ws, scratch_root=tmp_path / "scratch_self")
    assert self_img_diff["status"] == "ok"
    assert self_img_diff["beam_smoothed"] is False
    assert self_img_diff["regridded"] is False
    assert self_img_diff["ncc"] == pytest.approx(1.0, abs=1e-7)
    assert self_img_diff["peak_flux_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert self_img_diff["negative_bowl_delta"] == pytest.approx(0.0, abs=1e-6)

    # Crossed elliptical beam `12x8 @ 15 deg` at DIFFERENT cell (`2.0 arcsec`, `128x128`) vs `11x9 @ 105 deg` (`2.5 arcsec`, `104x104`)!
    _create_convolved_sky_image_set(
        casatools,
        agt_ws,
        "agt_sky",
        npix=128,
        cell_arcsec=2.0,
        b_major_arcsec=12.0,
        b_minor_arcsec=8.0,
        b_pa_deg=15.0,
    )
    sc_cross = tmp_path / "scratch_cross"
    crossed_diff = diff_images(agt_ws, ref_ws, scratch_root=sc_cross)
    assert crossed_diff["status"] == "ok"
    assert crossed_diff["beam_smoothed"] is True
    assert crossed_diff["regridded"] is True
    assert crossed_diff["ncc"] > 0.999
    assert crossed_diff["peak_flux_ratio"] == pytest.approx(1.0, rel=0.02)
    assert crossed_diff["aperture_flux_ratio"] == pytest.approx(1.0, rel=0.02)
    # Zero leftover scratch CASA tables
    assert list(sc_cross.glob("diff_img_*")) == []

    # --- 5d. Batch `--job-dir` CLI & Verifier `runner.py` `/tests/ref_checkpoint` integration ---
    batch_job = tmp_path / "batch_diff_job"
    tdir_b = _write_trial_fixture(
        batch_job,
        task_name="s1_image__11_full",
        agent_name="oracle",
        model_name=None,
    )
    shutil.copytree(agt_ws, tdir_b / "artifacts" / "workspace")
    rc_batch = rb_cli.main(
        [
            "--work-dir",
            str(tmp_path / "cli_work"),
            "diff-run",
            "--job-dir",
            str(batch_job),
            "--ref-dir",
            str(ref_ws),
        ]
    )
    assert rc_batch == 0
    batch_json = batch_job / "analysis" / "diffs" / f"{tdir_b.name}.json"
    assert batch_json.is_file()
    parsed_batch = json.loads(batch_json.read_text(encoding="utf-8"))
    assert parsed_batch["images"]["ncc"] > 0.999

    # Optional `/tests/ref_checkpoint` inside `run_verification`
    tests_dir = tmp_path / "ver_tests"
    tests_dir.mkdir()
    log_dir = tmp_path / "ver_logs"
    ec = ExtractConfig(
        suite_id="synth_vla_d",
        stages=["image"],
        roles=RolesSpec(),
        geometry=GeometrySpec(
            dish_diameter_m=25.0,
            pb_mask_min=0.25,
            source_apertures=[
                SourceApertureSpec(
                    label="ext",
                    kind="extended",
                    ra="276.0deg",
                    dec="-9.58deg",
                    radius_arcsec=40.0,
                )
            ],
        ),
    )
    rb = RubricSpec(
        task_id="synth_img_test",
        stages=["image"],
        subscores=[
            SubscoreSpec(
                id="products_score",
                weight=1.0,
                failure_tag="beam_sampling_or_clean_bowl",
                expr=LeafRuleSpec(
                    observable="image.products_complete",
                    primitive="bool_pass",
                    params={"expected": True},
                ),
            )
        ],
    )
    # Without `/tests/ref_checkpoint` -> `diff_summary == {}`
    _rw1, m1 = run_verification(
        agt_ws,
        tests_dir,
        log_dir,
        extract_config=ec,
        rubric=rb,
        ground_truth={},
        scratch_root=tmp_path / "ver_scratch",
        compute_raw_hash=False,
    )
    assert m1["diff_summary"] == {}

    # With `/tests/ref_checkpoint` -> `diff_summary["images"]["ncc"] > 0.999`
    shutil.copytree(ref_ws, tests_dir / "ref_checkpoint")
    _rw2, m2 = run_verification(
        agt_ws,
        tests_dir,
        log_dir,
        extract_config=ec,
        rubric=rb,
        ground_truth={},
        scratch_root=tmp_path / "ver_scratch",
        compute_raw_hash=False,
    )
    assert m2["diff_summary"]["images"]["ncc"] > 0.999


# ==============================================================================
# 6. Cohort-Stratified Pareto Frontier & 10,000-Bootstrap Membership
# ==============================================================================


def test_cohort_stratified_pareto_oracle_ties_zero_pass_and_bootstrap(tmp_path: Path) -> None:
    """Verify `all_5_arms` vs `e2e_6_arms`, oracle exclusion, $0.00 ties, 0%-pass, and 10k bootstrap."""
    # 1. Oracle-only records -> graceful `no_llm_trials`
    ora_only = [
        {
            "base_task": "s1_inspect",
            "suite": "synth",
            "stage": "inspect",
            "is_e2e": False,
            "arm": "00_none",
            "agent_name": "oracle",
            "model_name": "oracle",
            "reward": 1.0,
            "passed": 1,
            "total_usd": 0.0,
        }
    ]
    p_ora = compute_pareto_frontiers(ora_only, n_boot=1_000, seed=1)
    assert p_ora["all_5_arms"]["status"] == "no_llm_trials"
    assert p_ora["e2e_6_arms"]["status"] == "no_llm_trials"

    # 2. Stage-only records -> `e2e_6_arms` returns `no_e2e_trials` cleanly
    stage_only = [
        {
            "base_task": "s1_inspect",
            "suite": "synth",
            "stage": "inspect",
            "is_e2e": False,
            "arm": "11_full",
            "agent_name": "opencode",
            "model_name": "nrp/kimi",
            "reward": 0.90,
            "passed": 1,
            "total_usd": 0.05,
        }
    ]
    p_stg = compute_pareto_frontiers(stage_only, n_boot=500, seed=1)
    assert p_stg["all_5_arms"]["status"] == "ok"
    assert p_stg["e2e_6_arms"]["status"] == "no_e2e_trials"

    # 3. Full cohort test with:
    #    - Oracle ($0.00, r=1.0) -> MUST be excluded so it doesn't dominate LLM cells!
    #    - Two tied optimal $0.00 local cells (`local/qwen` @ `10_tools` and `11_full`, both $0.00, r=0.80, pass=0.50)
    #      -> BOTH must remain on frontier!
    #    - Dominated $0.00 local cell (`local/qwen` @ `00_none`, $0.00, r=0.40, pass=0.0 -> `dominated_zero_pass`!)
    #    - Paid high-accuracy cell (`nrp/kimi` @ `11_full`, $0.08, r=0.92, pass=1.0) -> on frontier!
    #    - Paid dominated cell (`anthropic/claude-sonnet-4-6` @ `00_none`, $0.25, r=0.55, pass=0.0 -> `dominated_zero_pass`!)
    #    - `11_orchestrated` evaluated ONLY on `s2_e2e_synth` ($0.12, r=0.98, pass=1.0) -> appears ONLY in `e2e_6_arms`,
    #      and `all_5_arms` keeps BOTH `s1_inspect` and `s2_e2e_synth` in `common_tasks`!
    records: list[dict[str, Any]] = [
        # Oracle row
        {
            "base_task": "s1_inspect",
            "suite": "synth",
            "stage": "inspect",
            "is_e2e": False,
            "arm": "00_none",
            "agent_name": "oracle",
            "model_name": "oracle",
            "thinking_level": "default",
            "reward": 1.0,
            "passed": 1,
            "total_usd": 0.0,
        }
    ]

    cell_specs = [
        ("local/qwen", "default", "00_none", 0.0, 0.40, [0, 0]),
        ("local/qwen", "default", "10_tools", 0.0, 0.80, [1, 0]),
        ("local/qwen", "default", "11_full", 0.0, 0.80, [1, 0]),
        ("nrp/kimi", "default", "11_full", 0.08, 0.92, [1, 1]),
        ("anthropic/claude-sonnet-4-6", "default", "00_none", 0.25, 0.55, [0, 0]),
    ]
    for btask, stg, is_e2e in (("s1_inspect", "inspect", False), ("s2_e2e_synth", "e2e", True)):
        for m_name, t_lvl, arm, usd, rew, pass_atts in cell_specs:
            for att_i, p_val in enumerate(pass_atts):
                records.append(
                    {
                        "trial_name": f"{btask}__{m_name}__{arm}__{att_i}",
                        "base_task": btask,
                        "suite": "synth",
                        "stage": stg,
                        "is_e2e": is_e2e,
                        "arm": arm,
                        "agent_name": "opencode",
                        "model_name": m_name,
                        "thinking_level": t_lvl,
                        "reward": rew,
                        "passed": p_val,
                        "total_usd": usd,
                    }
                )

    # Add `11_orchestrated` strictly on `s2_e2e_synth`
    for att_i in range(2):
        records.append(
            {
                "trial_name": f"s2_e2e_synth__kimi__11_orchestrated__{att_i}",
                "base_task": "s2_e2e_synth",
                "suite": "synth",
                "stage": "e2e",
                "is_e2e": True,
                "arm": "11_orchestrated",
                "agent_name": "opencode",
                "model_name": "nrp/kimi",
                "thinking_level": "default",
                "reward": 0.98,
                "passed": 1,
                "total_usd": 0.12,
            }
        )

    fac = analyze_factorial(records, n_boot=10_000, seed=20260413)
    # Strict JSON check (`allow_nan=False`, zero em/en dashes)
    dumped = json.dumps(fac, allow_nan=False)
    assert "\u2014" not in dumped and "\u2013" not in dumped

    p5 = fac["pareto"]["all_5_arms"]
    p6 = fac["pareto"]["e2e_6_arms"]

    assert p5["status"] == "ok"
    assert p5["common_tasks"] == ["s1_inspect", "s2_e2e_synth"]
    assert p5["n_cells"] == 5  # oracle excluded, 11_orchestrated excluded

    by5 = p5["by_cell"]
    # Tied $0.00 r=0.80 local cells BOTH remain on reward & pass frontiers!
    assert by5["local/qwen__default__10_tools"]["is_pareto_reward"] is True
    assert by5["local/qwen__default__11_full"]["is_pareto_reward"] is True
    assert by5["local/qwen__default__10_tools"]["is_pareto_pass"] is True
    assert by5["local/qwen__default__11_full"]["is_pareto_pass"] is True
    assert by5["local/qwen__default__10_tools"]["pareto_reward_boot_prob"] == 1.0
    assert by5["local/qwen__default__11_full"]["pareto_reward_boot_prob"] == 1.0

    # $0.00 r=0.40 0%-pass cell is dominated on reward and `dominated_zero_pass` on pass!
    z_qwen = by5["local/qwen__default__00_none"]
    assert z_qwen["is_pareto_reward"] is False
    assert z_qwen["is_pareto_pass"] is False
    assert z_qwen["pareto_pass_status"] == "dominated_zero_pass"
    assert z_qwen["cost_per_passing_run_usd_task_equal"] is None
    assert z_qwen["pareto_pass_boot_prob"] == 0.0

    # Sonnet 0%-pass cell also `dominated_zero_pass` with `None` cost/pass
    z_son = by5["anthropic/claude-sonnet-4-6__default__00_none"]
    assert z_son["is_pareto_reward"] is False
    assert z_son["is_pareto_pass"] is False
    assert z_son["pareto_pass_status"] == "dominated_zero_pass"
    assert z_son["cost_per_passing_run_usd_task_equal"] is None
    assert z_son["pareto_pass_boot_prob"] == 0.0

    # Paid Kimi @ 11_full ($0.08, r=0.92, pass=1.0) is on BOTH frontiers!
    # In the 2-stage bootstrap, each $0.00 qwen cell (`[1, 0]` on 2 tasks) draws 100% pass with
    # probability `(1/4)^2 = 1/16`, so Kimi remains pass-undominated with analytical probability
    # `(15/16)^2 = 225/256 = 0.87890625`!
    kimi_11 = by5["nrp/kimi__default__11_full"]
    assert kimi_11["is_pareto_reward"] is True
    assert kimi_11["is_pareto_pass"] is True
    assert kimi_11["pareto_reward_boot_prob"] == 1.0
    assert kimi_11["pareto_pass_boot_prob"] == pytest.approx(225.0 / 256.0, abs=0.02)

    # e2e_6_arms includes `11_orchestrated` strictly on `s2_e2e_synth`!
    assert p6["status"] == "ok"
    assert p6["common_tasks"] == ["s2_e2e_synth"]
    assert p6["n_cells"] == 6
    assert p6["by_cell"]["nrp/kimi__default__11_orchestrated"]["is_pareto_reward"] is True
    assert p6["by_cell"]["nrp/kimi__default__11_orchestrated"]["pareto_reward_boot_prob"] == 1.0

    # Verify `write_diagnostic_reports(..., plots=True)` renders Pareto in summary.md & exact 3 PNGs
    tax = summarize_failure_taxonomy(records)
    rep_dir = tmp_path / "pareto_diag"
    rep = write_diagnostic_reports(
        job_dirs=[tmp_path],
        output_dir=rep_dir,
        records=records,
        factorial_summary=fac,
        taxonomy_summary=tax,
        plots=True,
    )
    assert "## 7. Cohort-Stratified Cost-Quality Pareto Frontier" in rep["markdown"]
    assert "dominated_zero_pass" in rep["markdown"]
    png_names = {Path(p).name for p in rep["plot_files"]}
    assert png_names == {
        "reward_by_arm.png",
        "cost_vs_reward.png",
        "failure_taxonomy_by_arm.png",
    }


# ==============================================================================
# 7. Adversarial Review Regressions (Items 1-9)
# ==============================================================================


def test_wave3_review_trace_roles_passive_tools_crash_and_usd_regressions(
    tmp_path: Path,
) -> None:
    """Cover Items 5, 6, 7, 8, and 9 (non-CASA): cursor alignment, passive tools, crash, USD, /tmp."""
    from radio_bench.analysis.cli import run_analysis_pipeline
    from radio_bench.analysis.collect import write_trials_outputs
    from radio_bench.analysis.diagnostics import generate_diagnostic_plots
    from radio_bench.grading import runner as ver_runner

    # --- Item 5 & 6: Mixed linked + unlinked dispatches + repeated "reviewer" + passive Read/Agent ---
    job = tmp_path / "rev_job"
    root_traj = Trajectory(
        session_id="ses_root_mix",
        agent=Agent(name="opencode", version="1.19.5", model_name="nrp/kimi"),
        steps=[
            Step(
                step_id=1,
                source="agent",
                message="Read skill doc and dispatch stages.",
                tool_calls=[
                    # Passive `Read` returning CASA error keywords in doc text -> MUST NOT trigger error/retry!
                    ToolCall(
                        tool_call_id="call_read_1",
                        function_name="Read",
                        arguments={"file_path": "/skills/calibrate/SKILL.md"},
                    ),
                    # Linked dispatch 1 (`inspect-stage` -> `ses_c1`) with handoff prose mentioning `unknown tool`
                    ToolCall(
                        tool_call_id="call_d1",
                        function_name="task",
                        arguments={"subagent_type": "inspect-stage"},
                    ),
                    # Unlinked dispatch 2 (`precal-stage`, no sessionId in obs!)
                    ToolCall(
                        tool_call_id="call_d2",
                        function_name="task",
                        arguments={"subagent_type": "precal-stage"},
                    ),
                ],
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id="call_read_1",
                            content=(
                                "Troubleshooting note: RuntimeError: unknown tool, mcp error, "
                                "SEVERE\tException Reported in CASA logs."
                            ),
                        ),
                        ObservationResult(
                            source_call_id="call_d1",
                            content=(
                                '<task id="ses_c1">INSPECT HANDOFF: checked RuntimeError: '
                                "and unknown tool notes; all clean.</task>"
                            ),
                        ),
                        ObservationResult(
                            source_call_id="call_d2",
                            content="PRECAL HANDOFF: completed flagging.",
                        ),
                    ]
                ),
                metrics=Metrics(
                    prompt_tokens=1_000, completion_tokens=100, extra={"reasoning_tokens": 50}
                ),
            ),
            # Turn 2 MUST NOT be marked as a retry because Turn 1 had ONLY passive `Read` + `task` prose!
            # Also dispatches TWO repeated unlinked `"reviewer"` subagents!
            Step(
                step_id=2,
                source="agent",
                message="Dispatch two independent reviewers.",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_rev1",
                        function_name="task",
                        arguments={"subagent_type": "reviewer"},
                    ),
                    ToolCall(
                        tool_call_id="call_rev2",
                        function_name="task",
                        arguments={"subagent_type": "reviewer"},
                    ),
                ],
                observation=Observation(
                    results=[
                        ObservationResult(source_call_id="call_rev1", content="Reviewer 1 OK."),
                        ObservationResult(source_call_id="call_rev2", content="Reviewer 2 OK."),
                    ]
                ),
                metrics=Metrics(prompt_tokens=1_200, completion_tokens=80),
            ),
        ],
    )
    t_mix = _write_trial_fixture(
        job,
        task_name="s2_e2e_synth__11_orchestrated",
        agent_name="opencode",
        model_name="nrp/kimi",
        trajectory_dict=root_traj.model_dump(mode="json"),
    )
    db_mix = t_mix / "agent" / "opencode" / "xdg-data" / "opencode" / "opencode.db"
    db_mix.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_mix))
    try:
        con.execute(
            "CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, title TEXT, "
            "agent TEXT, model TEXT, time_created INTEGER)"
        )
        con.execute(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER)"
        )
        con.execute(
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, "
            "data TEXT, time_created INTEGER)"
        )
        con.execute(
            "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?)",
            ("ses_root_mix", None, "Root", "build", '{"providerID":"nrp","modelID":"kimi"}', 1),
        )
        for idx_s, sid in enumerate(("ses_c1", "ses_c2", "ses_c3", "ses_c4"), start=2):
            con.execute(
                "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    "ses_root_mix",
                    f"Child {idx_s}",
                    "general",
                    '{"providerID":"nrp","modelID":"kimi"}',
                    idx_s,
                ),
            )
            mid = f"msg_{sid}"
            con.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?)",
                (mid, sid, '{"role":"assistant","providerID":"nrp","modelID":"kimi"}', idx_s * 10),
            )
            con.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                (f"ps_{sid}", mid, sid, '{"type":"step-start"}', idx_s * 10 + 1),
            )
            con.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                (
                    f"pf_{sid}",
                    mid,
                    sid,
                    '{"type":"step-finish","tokens":{"input":500,"output":40,"reasoning":10,"cache":{"read":0,"write":0}}}',
                    idx_s * 10 + 2,
                ),
            )
        con.commit()
    finally:
        con.close()

    from radio_bench.analysis.collect import extract_trajectory_telemetry

    tel_mix = extract_trajectory_telemetry(
        root_traj.model_dump(mode="json"), arm="11_orchestrated", trial_dir=t_mix
    )
    assert tel_mix["severe_casa_errors"] == 0
    rec_mix = collect_single_trial(job, t_mix)
    assert rec_mix["tool_errors"] == 0
    assert rec_mix["protocol_errors"] == 0
    assert rec_mix["retry_turn_count"] == 0
    rb_mix = rec_mix["role_breakdown"]
    assert rb_mix["inspect-stage"]["turns"] == 1
    assert rb_mix["precal-stage"]["turns"] == 1  # NO cursor shift from linked ses_c1!
    assert rb_mix["reviewer"]["turns"] == 2  # BOTH repeated reviewer dispatches preserved!

    # --- Item 5b: ATIF `subagent_trajectories` shared `session_id` + `opencode.db` deduplication ---
    shared_sid_traj = {
        "session_id": "ses_shared_parent",
        "agent": {"name": "opencode", "model_name": "nrp/kimi"},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "c_s1",
                        "function_name": "task",
                        "arguments": {"subagent_type": "inspect-stage"},
                    },
                    {
                        "tool_call_id": "c_s2",
                        "function_name": "task",
                        "arguments": {"subagent_type": "image-stage"},
                    },
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "c_s1",
                            "content": "ok",
                            "subagent_trajectory_ref": [
                                {"trajectory_id": "sub_1", "session_id": "ses_c1"}
                            ],
                        },
                        {
                            "source_call_id": "c_s2",
                            "content": "ok",
                            "subagent_trajectory_ref": [
                                {"trajectory_id": "sub_2", "session_id": "ses_c2"}
                            ],
                        },
                    ]
                },
                "metrics": {"prompt_tokens": 500, "completion_tokens": 50},
            }
        ],
        "subagent_trajectories": [
            {
                "trajectory_id": "sub_1",
                "session_id": "ses_c1",
                "agent": {"name": "opencode", "model_name": "nrp/kimi"},
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "metrics": {"prompt_tokens": 300, "completion_tokens": 30},
                    }
                ],
            },
            {
                "trajectory_id": "sub_2",
                "session_id": "ses_c2",
                "agent": {"name": "opencode", "model_name": "nrp/kimi"},
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "metrics": {"prompt_tokens": 400, "completion_tokens": 40},
                    }
                ],
            },
        ],
    }
    # Pass `db_mix` alongside `shared_sid_traj`: `ses_c1` and `ses_c2` MUST NOT be double-counted!
    t_dedup = _write_trial_fixture(
        job,
        task_name="s1_image__11_full",
        agent_name="opencode",
        model_name="nrp/kimi",
        trajectory_dict=shared_sid_traj,
    )
    dedup_db = t_dedup / "agent" / "opencode" / "xdg-data" / "opencode" / "opencode.db"
    dedup_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_mix, dedup_db)
    rec_dedup = collect_single_trial(job, t_dedup)
    assert rec_dedup["role_breakdown"]["inspect-stage"]["turns"] == 1
    assert rec_dedup["role_breakdown"]["image-stage"]["turns"] == 1

    # --- Item 7: Mid-turn OpenCode crash (`step-start` without `step-finish`), `thinking > completion`, & exact `0.2 + 0.1 == 0.3` USD ---
    t_crash = _write_trial_fixture(
        job,
        task_name="s1_precal__11_full",
        agent_name="opencode",
        model_name="nrp/kimi",
        trajectory_dict={
            "session_id": "ses_crash",
            "agent": {"name": "opencode", "model_name": "nrp/kimi"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "metrics": {"prompt_tokens": 200, "completion_tokens": 20},
                }
            ],
        },
    )
    crash_db = t_crash / "agent" / "opencode" / "xdg-data" / "opencode" / "opencode.db"
    crash_db.parent.mkdir(parents=True, exist_ok=True)
    con2 = sqlite3.connect(str(crash_db))
    try:
        con2.execute(
            "CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, title TEXT, agent TEXT, model TEXT)"
        )
        con2.execute(
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT)"
        )
        con2.execute("INSERT INTO session VALUES ('ses_crash', NULL, 'Root', 'build', 'nrp/kimi')")
        con2.execute(
            "INSERT INTO part VALUES ('p1', 'm1', 'ses_crash', '{\"type\":\"step-start\"}')"
        )
        con2.execute(
            "INSERT INTO part VALUES ('p2', 'm1', 'ses_crash', "
            '\'{"type":"step-finish","tokens":{"input":200,"output":20,"reasoning":0}}\')'
        )
        # Second turn started (`step-start`) then crashed mid-turn before `step-finish`!
        con2.execute(
            "INSERT INTO part VALUES ('p3', 'm2', 'ses_crash', '{\"type\":\"step-start\"}')"
        )
        con2.commit()
    finally:
        con2.close()
    rec_crash = collect_single_trial(job, t_crash)
    assert rec_crash["trace_completeness_ok"] is False

    # Inclusive provider where `thinking_tokens > completion_tokens` + exact `0.2 + 0.1 == 0.3` USD rounding!
    from radio_bench.config import ModelPricingConfig, ModelSpec

    exact_models = ModelsConfig(
        models={
            "exact_dec": ModelSpec(
                key="exact_dec",
                harness="claude-code",
                model_name="test/exact-dec",
                pricing=ModelPricingConfig(
                    input_per_mtok=0.0,
                    output_per_mtok=1000.0,  # $0.001 per token => 200 tok = $0.2, 100 tok = $0.1
                    cache_read_per_mtok=0.0,
                    cache_write_5m_per_mtok=0.0,
                    cache_write_1h_per_mtok=0.0,
                ),
            )
        },
        parity_pairs=(),
    )
    pr_exact = compute_trial_pricing(
        model_name="test/exact-dec",
        agent_name="claude-code",
        trajectory_dict={
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "tool_calls": [
                        {"tool_call_id": "b1", "function_name": "Bash", "arguments": {}}
                    ],
                    "observation": {
                        "results": [{"source_call_id": "b1", "content": "[exit_code]: 1"}]
                    },
                    # raw_completion=50 < thinking_tokens=200 -> billed_out MUST clamp up to 200 ($0.2)!
                    "metrics": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "extra": {"output_tokens_details": {"thinking_tokens": 200}},
                    },
                },
                {
                    "step_id": 2,
                    "source": "agent",
                    "metrics": {
                        "prompt_tokens": 100,
                        "completion_tokens": 100,
                    },  # retry turn = $0.1
                },
            ]
        },
        models_cfg=exact_models,
    )
    assert pr_exact.output_tokens == 300
    assert pr_exact.reasoning_tokens == 200
    assert pr_exact.reply_output_tokens == 100
    assert pr_exact.first_pass_usd == 0.2
    assert pr_exact.retry_usd == 0.1
    assert pr_exact.total_usd == 0.3
    assert repr(pr_exact.total_usd) == "0.3"  # ZERO 0.30000000000000004 noise!

    # --- Item 8: Multi-pack `by_model_thinking` & non-default `thinking_level` in `summary.md` Section 4 ---
    from radio_bench.config import ParityPairSpec, ParitySideSpec

    custom_parity_cfg = ModelsConfig(
        models=load_models_config().models,
        parity_pairs=(
            ParityPairSpec(
                pair_id="haiku_high_vs_sonnet",
                small=ParitySideSpec(
                    model="anthropic/claude-haiku-4-5",
                    arm="11_full",
                    thinking_level="high",
                ),
                frontier=ParitySideSpec(
                    model="anthropic/claude-sonnet-4-6",
                    arm="00_none",
                    thinking_level="default",
                ),
            ),
        ),
    )
    mp_records = [
        {
            "base_task": "s1_inspect",
            "pack_id": "pack_a",
            "suite": "synth",
            "stage": "inspect",
            "is_e2e": False,
            "arm": "11_full",
            "agent_name": "claude-code",
            "model_name": "anthropic/claude-haiku-4-5",
            "thinking_level": "high",
            "reward": 0.92,
            "passed": 1,
            "total_usd": 0.05,
        },
        {
            "base_task": "s1_inspect",
            "pack_id": "pack_b",
            "suite": "synth",
            "stage": "inspect",
            "is_e2e": False,
            "arm": "11_full",
            "agent_name": "claude-code",
            "model_name": "anthropic/claude-haiku-4-5",
            "thinking_level": "high",
            "reward": 0.75,
            "passed": 0,
            "total_usd": 0.06,
        },
        {
            "base_task": "s1_inspect",
            "pack_id": "pack_a",
            "suite": "synth",
            "stage": "inspect",
            "is_e2e": False,
            "arm": "00_none",
            "agent_name": "claude-code",
            "model_name": "anthropic/claude-sonnet-4-6",
            "thinking_level": "default",
            "reward": 0.60,
            "passed": 0,
            "total_usd": 0.18,
        },
    ]
    mp_fac = analyze_factorial(mp_records, n_boot=500, seed=7, models_cfg=custom_parity_cfg)
    assert "pack_a__anthropic/claude-haiku-4-5__high" in mp_fac["by_model_thinking"]
    assert "pack_b__anthropic/claude-haiku-4-5__high" in mp_fac["by_model_thinking"]
    assert (
        mp_fac["by_model_thinking"]["pack_a__anthropic/claude-haiku-4-5__high"]["pack_id"]
        == "pack_a"
    )

    mp_rep = write_diagnostic_reports(
        job_dirs=[tmp_path],
        output_dir=tmp_path / "mp_diag",
        records=mp_records,
        factorial_summary=mp_fac,
        taxonomy_summary=summarize_failure_taxonomy(mp_records),
        plots=True,
    )
    assert "`anthropic/claude-haiku-4-5 (high) @ 11_full`" in mp_rep["markdown"]

    # --- Item 9: `/tmp` guards on public entrypoints & `cli._resolve_diff_caller_paths` ---
    assert rb_cli._resolve_diff_caller_paths(
        ["--agent-dir", "--ref-dir", "rel_ckpt"], tmp_path
    ) == ["--agent-dir", "--ref-dir", "rel_ckpt"]

    with pytest.raises(ValueError, match="/tmp"):
        write_trials_outputs([], Path("/tmp/bad_trials_out"))
    with pytest.raises(ValueError, match="/tmp"):
        collect_trials([Path("/tmp/bad_job_in")])
    with pytest.raises(ValueError, match="/tmp"):
        write_diagnostic_reports(
            job_dirs=[tmp_path],
            output_dir=Path("/tmp/bad_diag_out"),
            records=[],
            factorial_summary={},
            taxonomy_summary={},
        )
    with pytest.raises(ValueError, match="/tmp"):
        generate_diagnostic_plots(
            records=[],
            factorial_summary={},
            taxonomy_summary={},
            plots_dir=Path("/tmp/bad_plots_out"),
        )
    with pytest.raises(ValueError, match="/tmp"):
        run_analysis_pipeline([Path("/tmp/bad_pipe_job")], output_dir=tmp_path / "ok_out")
    with pytest.raises(ValueError, match="/tmp"):
        diff_visibilities(Path("/tmp/bad_a.ms"), tmp_path / "r.ms")
    with pytest.raises(ValueError, match="/tmp"):
        diff_caltables(Path("/tmp/bad_ws_a"), tmp_path / "ws_r")
    with pytest.raises(ValueError, match="/tmp"):
        diff_images(tmp_path / "ws_a", Path("/tmp/bad_ws_r"))
    with pytest.raises(ValueError, match="/tmp"):
        ver_runner.main(
            [
                "--workspace",
                str(tmp_path),
                "--tests-dir",
                str(tmp_path),
                "--log-dir",
                "/tmp/bad_ver_logs",
            ]
        )


def test_wave3_review_diff_checkpoint_caltable_spw_calibrators_and_dirty_image(
    tmp_path: Path,
) -> None:
    """Cover Items 1, 2, 3, 4, and 7 (CASA): data_dir filter, mtime caltables, dirty img, SpW trim, calibrators.ms."""
    from radio_bench.analysis.diff import diff_job_directory, diff_workspaces

    casatools = _ensure_local_casa()
    freqs_16 = np.linspace(4.5e9, 4.65e9, 16)
    freqs_12 = freqs_16[2:14]  # Trimmed channels `2..13` (`mstransform(spw='0:2~13')`!)
    fields = ["J1331+3030", "J1822-0938", "3C391 C1"]
    intents = [
        "CALIBRATE_FLUX,CALIBRATE_BANDPASS",
        "CALIBRATE_PHASE",
        "OBSERVE_TARGET#ON_SOURCE",
    ]
    ants = ["ea01", "ea02", "ea03"]
    t0 = 5.12e9

    # --- Item 4a: `matched_rows == 0` -> `status == "no_common_rows"` & `None` flag metrics ---
    disjoint_a = tmp_path / "disj_a.ms"
    disjoint_r = tmp_path / "disj_r.ms"
    v_one = np.ones((4, 16), dtype=np.complex128)
    f_zero = np.zeros((4, 16), dtype=bool)
    _create_synthetic_ms_for_diff(
        casatools,
        disjoint_r,
        field_names=["J1331+3030"],
        field_intents=["CALIBRATE_FLUX"],
        ant_names=ants,
        chan_freqs_hz=freqs_16,
        rows=[{"time": t0, "a1": 0, "a2": 1, "fid": 0, "scan": 1, "vis": v_one, "flag": f_zero}],
    )
    _create_synthetic_ms_for_diff(
        casatools,
        disjoint_a,
        field_names=["J1331+3030"],
        field_intents=["CALIBRATE_FLUX"],
        ant_names=ants,
        chan_freqs_hz=freqs_16,
        # Different scan (`scan=99`) => 0 matched rows!
        rows=[{"time": t0, "a1": 0, "a2": 1, "fid": 0, "scan": 99, "vis": v_one, "flag": f_zero}],
    )
    no_match = diff_visibilities(disjoint_a, disjoint_r, stage="calibrate")
    assert no_match["status"] == "no_common_rows"
    assert no_match["overall"]["matched_rows"] == 0
    assert no_match["overall"]["flag_jaccard"] is None
    assert no_match["overall"]["flag_recall_vs_ref"] is None
    assert no_match["overall"]["excess_flag_fraction"] is None

    # --- Item 4b: Stokes `[1, 2, 3, 4]` `CORR_TYPE` MUST NOT swap `[1, 2]` (`Q, U`) on baseline flip ---
    stokes_r = tmp_path / "stokes_r.ms"
    stokes_a = tmp_path / "stokes_a.ms"
    v_stokes = np.zeros((4, 16), dtype=np.complex128)
    v_stokes[0] = 1.5 + 0.0j  # I
    v_stokes[1] = 0.4 + 0.0j  # Q (distinct from U!)
    v_stokes[2] = -0.3 + 0.0j  # U
    v_stokes[3] = 0.05 + 0.0j  # V
    _create_synthetic_ms_for_diff(
        casatools,
        stokes_r,
        field_names=["SRC"],
        field_intents=["OBSERVE_TARGET#ON_SOURCE"],
        ant_names=["ea01", "ea02"],
        chan_freqs_hz=freqs_16,
        rows=[{"time": t0, "a1": 0, "a2": 1, "fid": 0, "scan": 1, "vis": v_stokes, "flag": f_zero}],
        corr_types=[1, 2, 3, 4],
    )
    _create_synthetic_ms_for_diff(
        casatools,
        stokes_a,
        field_names=["SRC"],
        field_intents=["OBSERVE_TARGET#ON_SOURCE"],
        ant_names=["ea01", "ea02"],
        chan_freqs_hz=freqs_16,
        # Reversed baseline (a1=1 > a2=0): Stokes visibilities conjugate WITHOUT swapping Q(1)<->U(2)!
        rows=[
            {
                "time": t0,
                "a1": 1,
                "a2": 0,
                "fid": 0,
                "scan": 1,
                "vis": np.conj(v_stokes),
                "flag": f_zero,
            }
        ],
        corr_types=[1, 2, 3, 4],
    )
    stokes_diff = diff_visibilities(stokes_a, stokes_r, stage="calibrate")
    assert stokes_diff["status"] == "ok"
    assert stokes_diff["overall"]["parallel_hand_nrmse"] < 1e-6

    # --- Item 1, 2, 3, 4c, 4d: Top-level `data_root` with BOTH `synth/` and `checkpoints/3c391/`,
    #     edge-trimmed `mstransform` SpW (`12` vs `16` chans), `calibrators.ms`, chronological
    #     caltable sorting (`init_bp.b` vs `bandpass.b`), scalar `T Jones` vs `G Jones`,
    #     multi-timestamp `Df`, and dirty `niter=0` image guard! ---
    fake_data_root = tmp_path / "data_root"
    # Populate an alphabetical trap `synth/s1_image/input.ms` with completely different fields/amps!
    synth_trap_dir = fake_data_root / "synth" / "s1_image"
    _create_synthetic_ms_for_diff(
        casatools,
        synth_trap_dir / "input.ms",
        field_names=["SYNTH_TRAP"],
        field_intents=["OBSERVE_TARGET#ON_SOURCE"],
        ant_names=ants,
        chan_freqs_hz=freqs_16,
        rows=[
            {"time": t0, "a1": 0, "a2": 1, "fid": 0, "scan": 1, "vis": v_one * 99.0, "flag": f_zero}
        ],
    )

    # Populate `checkpoints/3c391/post_calibrate/input.ms` + `checkpoints/3c391/reference_image`
    ref_3c_cal_dir = fake_data_root / "checkpoints" / "3c391" / "post_calibrate"
    ref_3c_img_dir = fake_data_root / "checkpoints" / "3c391" / "reference_image"

    spec_16 = (2.5 + 0.3 * np.sin(np.linspace(0.0, 2.0, 16))) * np.exp(
        1j * np.linspace(-0.2, 0.2, 16)
    )
    vis_cal_16 = np.broadcast_to(spec_16[None, :], (4, 16)).copy()
    vis_tgt_16 = vis_cal_16 * 0.15
    flg_16 = np.zeros((4, 16), dtype=bool)
    flg_16[:, 5] = True  # RFI channel at index 5 in 16-chan (= index 3 in trimmed 12-chan!)

    _create_synthetic_ms_for_diff(
        casatools,
        ref_3c_cal_dir / "input.ms",
        field_names=fields,
        field_intents=intents,
        ant_names=ants,
        chan_freqs_hz=freqs_16,
        rows=[
            {
                "time": t0,
                "a1": 0,
                "a2": 1,
                "fid": 0,
                "scan": 1,
                "vis": vis_cal_16 * 0.5,
                "corr_vis": vis_cal_16,
                "flag": flg_16,
            },
            {
                "time": t0 + 10.0,
                "a1": 0,
                "a2": 1,
                "fid": 2,
                "scan": 2,
                "vis": vis_tgt_16 * 0.5,
                "corr_vis": vis_tgt_16,
                "flag": flg_16,
            },
        ],
    )
    _create_convolved_sky_image_set(
        casatools,
        ref_3c_img_dir,
        "3c391_ref",
        npix=96,
        cell_arcsec=2.5,
        b_major_arcsec=12.0,
        b_minor_arcsec=10.0,
        b_pa_deg=0.0,
    )

    # Reference caltables (`16` channels) in `ref_3c_cal_dir`
    n_ant = 3
    ant_ids_3 = np.arange(n_ant, dtype=np.int32)
    fids_3 = np.zeros(n_ant, dtype=np.int32)
    t_3 = np.full(n_ant, t0, dtype=np.float64)
    sc_3 = np.ones(n_ant, dtype=np.int32)
    bp_true_16 = np.exp(1j * np.outer(np.array([0.0, 0.25, -0.35]), np.linspace(1.0, 1.4, 16)))
    bp_ref_param = np.stack([bp_true_16.T, bp_true_16.conj().T], axis=0)  # (2, 16, 3)
    _create_caltable_fixture(
        casatools,
        ref_3c_cal_dir / "B_ref.cal",
        viscal="B Jones",
        is_complex=True,
        param=bp_ref_param,
        flag=np.zeros((2, 16, n_ant), dtype=bool),
        ant_ids=ant_ids_3,
        field_ids=fids_3,
        times=t_3,
        scans=sc_3,
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_16,
    )
    # Reference `G0` (`npol=2`, phase=[0, 0.3, -0.2] identical on both pols) to compare vs scalar `T Jones` (`npol=1`)!
    t_phs = np.array([0.0, 0.30, -0.20])
    g0_ref_2pol = np.exp(1j * np.stack([t_phs, t_phs], axis=0))[:, None, :]
    _create_caltable_fixture(
        casatools,
        ref_3c_cal_dir / "G0_ref.cal",
        viscal="G Jones",
        is_complex=True,
        param=g0_ref_2pol,
        flag=np.zeros((2, 1, n_ant), dtype=bool),
        ant_ids=ant_ids_3,
        field_ids=fids_3,
        times=t_3,
        scans=sc_3,
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_16,
    )
    # Reference `Df` with 1 timestamp vs Agent `Df` with 2 timestamps (1st slot flagged on ch 0, 2nd unflagged!)
    d_true_16 = np.zeros((2, 16, n_ant), dtype=np.complex128)
    d_true_16[0, :, 1] = 0.02 - 0.01j
    d_true_16[1, :, 1] = -0.015 + 0.008j
    _create_caltable_fixture(
        casatools,
        ref_3c_cal_dir / "Df_ref.cal",
        viscal="Df Jones",
        is_complex=True,
        param=d_true_16,
        flag=np.zeros((2, 16, n_ant), dtype=bool),
        ant_ids=ant_ids_3,
        field_ids=fids_3,
        times=t_3,
        scans=sc_3,
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_16,
    )

    # Now build Agent Harbor trial `r2_3c391_e2e__10_tools` with:
    # - Edge-trimmed `12`-channel `input.ms` where calibrator `fid=0` has UNCALIBRATED `CORRECTED_DATA == DATA`,
    #   while `calibrators.ms` has CALIBRATED `CORRECTED_DATA == vis_cal_12`!
    # - Stale `init_bp.b` (corrupted +45 deg, OLDER mtime, alphabetically AFTER `bandpass.b`!)
    #   and final `bandpass.b` (exact `bp_ref_param[:, 2:14, :]`, NEWER mtime, alphabetically BEFORE `init_bp.b`!)
    # - Scalar `T Jones` (`npol=1`) phase-only table (`phase_t.cal`) AFTER an older stale `init_g0.cal`!
    # - Multi-timestamp `Df` (`2` timestamps on trimmed `12` chans)!
    # - Dirty `niter=0` image WITHOUT restoring beam -> `images.status == "missing_restoring_beam_or_wcs"`
    #   while `visibilities` and `caltables` still succeed!
    agt_job = tmp_path / "agt_3c_job"
    t_agt = _write_trial_fixture(
        agt_job,
        task_name="r2_3c391_e2e__10_tools",
        agent_name="claude-code",
        model_name="anthropic/claude-sonnet-4-6",
    )
    agt_ws = t_agt / "artifacts" / "workspace"
    agt_ws.mkdir(parents=True, exist_ok=True)

    vis_cal_12 = vis_cal_16[:, 2:14]
    vis_tgt_12 = vis_tgt_16[:, 2:14]
    flg_12 = flg_16[:, 2:14]
    _create_synthetic_ms_for_diff(
        casatools,
        agt_ws / "input.ms",
        field_names=fields,
        field_intents=intents,
        ant_names=ants,
        chan_freqs_hz=freqs_12,
        rows=[
            # Uncalibrated in input.ms (`corr_vis == vis * 0.5`)!
            {
                "time": t0,
                "a1": 0,
                "a2": 1,
                "fid": 0,
                "scan": 1,
                "vis": vis_cal_12 * 0.5,
                "corr_vis": vis_cal_12 * 0.5,
                "flag": np.zeros_like(flg_12),
            },
            {
                "time": t0 + 10.0,
                "a1": 0,
                "a2": 1,
                "fid": 2,
                "scan": 2,
                "vis": vis_tgt_12 * 0.5,
                "corr_vis": vis_tgt_12,
                "flag": flg_12,
            },
        ],
    )
    _create_synthetic_ms_for_diff(
        casatools,
        agt_ws / "calibrators.ms",
        field_names=[fields[0]],
        field_intents=[intents[0]],
        ant_names=ants,
        chan_freqs_hz=freqs_12,
        # Calibrated `CORRECTED_DATA == vis_cal_12` + RFI flag in `calibrators.ms`!
        rows=[
            {
                "time": t0,
                "a1": 0,
                "a2": 1,
                "fid": 0,
                "scan": 1,
                "vis": vis_cal_12 * 0.5,
                "corr_vis": vis_cal_12,
                "flag": flg_12,
            }
        ],
    )

    # Stale `init_bp.b` (older mtime = 1000.0) vs final `bandpass.b` (newer mtime = 2000.0)
    bp_bad_12 = bp_ref_param[:, 2:14, :] * np.exp(1j * np.deg2rad(45.0))
    bp_bad_12[:, :, 0] = 1.0 + 0.0j
    _create_caltable_fixture(
        casatools,
        agt_ws / "init_bp.b",
        viscal="B Jones",
        is_complex=True,
        param=bp_bad_12,
        flag=np.zeros((2, 12, n_ant), dtype=bool),
        ant_ids=ant_ids_3,
        field_ids=fids_3,
        times=t_3,
        scans=sc_3,
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_12,
    )
    _create_caltable_fixture(
        casatools,
        agt_ws / "bandpass.b",
        viscal="B Jones",
        is_complex=True,
        param=bp_ref_param[:, 2:14, :],
        flag=np.zeros((2, 12, n_ant), dtype=bool),
        ant_ids=ant_ids_3,
        field_ids=fids_3,
        times=t_3,
        scans=sc_3,
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_12,
    )
    os.utime(agt_ws / "init_bp.b" / "table.dat", (1_700_000_100.0, 1_700_000_100.0))
    os.utime(agt_ws / "bandpass.b" / "table.dat", (1_700_000_900.0, 1_700_000_900.0))

    # Stale `z_stale_g0.cal` (older mtime) vs newer scalar `T Jones` `a_final_t.cal` (`npol=1`, newer mtime!)
    g0_bad = np.exp(1j * np.array([[[0.0, 1.1, -0.9]], [[0.0, 1.1, -0.9]]]))
    _create_caltable_fixture(
        casatools,
        agt_ws / "z_stale_g0.cal",
        viscal="G Jones",
        is_complex=True,
        param=g0_bad,
        flag=np.zeros((2, 1, n_ant), dtype=bool),
        ant_ids=ant_ids_3,
        field_ids=fids_3,
        times=t_3,
        scans=sc_3,
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_12,
    )
    t_good_1pol = np.exp(1j * t_phs[None, None, :])  # shape (1, 1, 3) scalar T Jones!
    _create_caltable_fixture(
        casatools,
        agt_ws / "a_final_t.cal",
        viscal="T Jones",
        is_complex=True,
        param=t_good_1pol,
        flag=np.zeros((1, 1, n_ant), dtype=bool),
        ant_ids=ant_ids_3,
        field_ids=fids_3,
        times=t_3,
        scans=sc_3,
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_12,
    )
    os.utime(agt_ws / "z_stale_g0.cal" / "table.dat", (1_700_000_100.0, 1_700_000_100.0))
    os.utime(agt_ws / "a_final_t.cal" / "table.dat", (1_700_000_900.0, 1_700_000_900.0))

    # Multi-timestamp `Df` on 12 trimmed channels: slot 1 has ch 0 flagged & corrupted, slot 2 unflagged!
    d_slot1 = d_true_16[:, 2:14, :].copy()
    d_slot1[:, 0, 1] = 9.9 + 9.9j  # flagged in slot 1!
    f_slot1 = np.zeros((2, 12, n_ant), dtype=bool)
    f_slot1[:, 0, :] = True
    d_slot2 = d_true_16[:, 2:14, :].copy()
    f_slot2 = np.zeros((2, 12, n_ant), dtype=bool)
    _create_caltable_fixture(
        casatools,
        agt_ws / "leak_multi.df",
        viscal="Df Jones",
        is_complex=True,
        param=np.concatenate([d_slot1, d_slot2], axis=2),
        flag=np.concatenate([f_slot1, f_slot2], axis=2),
        ant_ids=np.tile(ant_ids_3, 2),
        field_ids=np.zeros(2 * n_ant, dtype=np.int32),
        times=np.concatenate([t_3, t_3 + 30.0]),
        scans=np.ones(2 * n_ant, dtype=np.int32),
        ant_names=ants,
        field_names=fields,
        chan_freqs_hz=freqs_12,
    )

    # Dirty `niter=0` image WITHOUT restoring beam!
    ia_dirty = casatools.image()
    try:
        ia_dirty.fromarray(
            outfile=str(agt_ws / "dirty_map.image"),
            pixels=np.zeros((64, 64, 1, 1), dtype=np.float64),
            overwrite=True,
        )
    finally:
        with contextlib.suppress(Exception):
            ia_dirty.close()
        with contextlib.suppress(Exception):
            ia_dirty.done()

    ws_diff = diff_workspaces(t_agt, fake_data_root, scratch_root=tmp_path / "sc_rev")
    assert ws_diff["base_task"] == "r2_3c391_e2e"
    assert ws_diff["visibilities"]["status"] == "ok"
    assert ws_diff["visibilities"]["agent_calibrators_ms"] == "calibrators.ms"
    # Matched `3c391` checkpoint (NOT `synth`!) + `calibrators.ms` calibrated vis + `12`-chan SpW intersection!
    assert ws_diff["visibilities"]["overall"]["matched_rows"] == 2
    assert ws_diff["visibilities"]["overall"]["flag_jaccard"] == 1.0
    assert ws_diff["visibilities"]["overall"]["parallel_hand_nrmse"] < 1e-6

    # Chronological caltable selection (`bandpass.b` > `init_bp.b`, `a_final_t.cal` > `z_stale_g0.cal`)!
    assert ws_diff["caltables"]["B"]["agent_table"] == "bandpass.b"
    assert ws_diff["caltables"]["B"]["bp_phase_rms_deg"] == pytest.approx(0.0, abs=1e-4)
    assert ws_diff["caltables"]["G"]["G0_phase_only"]["agent_table"] == "a_final_t.cal"
    assert ws_diff["caltables"]["G"]["G0_phase_only"]["gain_phase_rms_deg"] == pytest.approx(
        0.0, abs=1e-4
    )
    assert ws_diff["caltables"]["D"]["dterm_rms"] == pytest.approx(0.0, abs=1e-6)
    # Dirty beamless image guarded cleanly without wiping visibilities/caltables!
    assert ws_diff["images"] == {"status": "missing_restoring_beam_or_wcs"}

    # Also verify `precal` flag-merging from `calibrators.ms` into `input.ms`!
    pre_vis = diff_visibilities(agt_ws / "input.ms", ref_3c_cal_dir / "input.ms", stage="precal")
    assert pre_vis["status"] == "ok"
    assert pre_vis["overall"]["flag_jaccard"] == 1.0

    # --- Item 7d: `collect_single_trial` preserves `diff_summary` & `diff_job_directory` exports it when workspace torn down ---
    t_torndown = _write_trial_fixture(
        agt_job,
        task_name="r1_3c391_image__11_full",
        agent_name="opencode",
        model_name="nrp/kimi",
    )
    (t_torndown / "verifier" / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "task_id": "r1_3c391_image",
                "stage": "image",
                "diff_summary": {
                    "schema_version": "0.2",
                    "images": {"status": "ok", "ncc": 0.9985},
                },
            }
        ),
        encoding="utf-8",
    )
    rec_td = collect_single_trial(agt_job, t_torndown)
    assert rec_td["diff_summary"]["images"]["ncc"] == pytest.approx(0.9985)

    batch_out = diff_job_directory(agt_job, fake_data_root, scratch_root=tmp_path / "sc_batch")
    assert batch_out["n_written"] == 2
    assert batch_out["n_skipped_no_workspace"] == 0
    exported_td = json.loads(
        (agt_job / "analysis" / "diffs" / f"{t_torndown.name}.json").read_text(encoding="utf-8")
    )
    assert exported_td["source"] == "verifier_metrics_json"
    assert exported_td["images"]["ncc"] == pytest.approx(0.9985)
