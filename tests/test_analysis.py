"""Unit and Harbor 0.25.0 schema integration tests for radio_bench.analysis."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.job.result import JobResult, JobStats
from harbor.models.task.id import LocalTaskId
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import (
    AgentInfo,
    ExceptionInfo,
    ModelInfo,
    TimingInfo,
    TrialResult,
)
from harbor.models.verifier.result import VerifierResult

from radio_bench.analysis import (
    FAILURE_CATEGORIES,
    FAILURE_NONE,
    analyze_factorial,
    classify_failure,
    collect_trials,
    compute_trial_pricing,
    price_turn,
    summarize_failure_taxonomy,
    wilson_ci,
)
from radio_bench.analysis import (
    main as cli_main,
)


def _write_harbor_trial(
    job_dir: Path,
    *,
    task_name: str,
    agent_name: str,
    model_name: str | None,
    reward_dict: dict[str, float | int] | None,
    metrics_dict: dict[str, Any] | None,
    trajectory: Trajectory | None = None,
    agent_context: AgentContext | None = None,
    exception_info: ExceptionInfo | None = None,
    started_at: datetime | None = None,
    agent_sec: float = 12.0,
    verifier_sec: float = 4.0,
) -> tuple[Path, TrialResult]:
    """Create a realistic Harbor 0.25.0 trial directory validated by Harbor Pydantic models."""
    st = started_at or datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
    agent_start = st + timedelta(seconds=1.0)
    agent_end = agent_start + timedelta(seconds=agent_sec)
    ver_end = agent_end + timedelta(seconds=verifier_sec)

    agent_cfg_dict: dict[str, Any] = {"name": agent_name}
    if model_name is not None:
        agent_cfg_dict["model_name"] = model_name

    tcfg = TrialConfig(
        task={"path": f"/mnt/fast/radio-bench/tasks/{task_name}"},
        agent=agent_cfg_dict,
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

    ver_res = VerifierResult(rewards=reward_dict) if reward_dict is not None else None

    tr = TrialResult(
        task_name=task_name,
        trial_name=tcfg.trial_name,
        trial_uri=f"file://{trial_dir}",
        task_id=LocalTaskId(path=Path(f"/mnt/fast/radio-bench/tasks/{task_name}")),
        task_checksum="deadbeef00112233",
        config=tcfg,
        agent_info=AgentInfo(name=agent_name, version="0.25.0", model_info=model_info),
        agent_result=agent_context,
        verifier_result=ver_res,
        exception_info=exception_info,
        started_at=st,
        finished_at=ver_end,
        agent_execution=TimingInfo(started_at=agent_start, finished_at=agent_end),
        verifier=TimingInfo(started_at=agent_end, finished_at=ver_end),
    )

    tp.config_path.write_text(tcfg.model_dump_json(indent=2), encoding="utf-8")
    tp.result_path.write_text(tr.model_dump_json(indent=2), encoding="utf-8")
    if reward_dict is not None:
        tp.reward_json_path.write_text(
            json.dumps(reward_dict, indent=2, allow_nan=False), encoding="utf-8"
        )
    if metrics_dict is not None:
        (tp.verifier_dir / "metrics.json").write_text(
            json.dumps(metrics_dict, indent=2, allow_nan=False), encoding="utf-8"
        )
    if trajectory is not None:
        (tp.agent_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return trial_dir, tr


def _write_job_result(job_dir: Path, trial_results: list[TrialResult]) -> None:
    st = min(tr.started_at for tr in trial_results if tr.started_at is not None)
    ft = max(tr.finished_at for tr in trial_results if tr.finished_at is not None)
    jr = JobResult(
        id=trial_results[0].id,
        started_at=st,
        finished_at=ft,
        n_total_trials=len(trial_results),
        stats=JobStats.from_trial_results(trial_results, n_total_trials=len(trial_results)),
        trial_results=trial_results,
    )
    (job_dir / "result.json").write_text(jr.model_dump_json(indent=2), encoding="utf-8")


def _assert_no_nan_or_emdash_in_file(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    assert "\u2014" not in raw, f"Found em dash in {path}"
    assert "\u2013" not in raw, f"Found en dash in {path}"
    if path.suffix == ".json":
        parsed = json.loads(
            raw,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {c}")),
        )
        return parsed
    return raw


def test_oracle_job_round_trip(tmp_path: Path) -> None:
    """Oracle job (5 *__00_none tasks, 0 tokens, $0.00) parses cleanly with zero NaNs."""
    job_dir = tmp_path / "oracle_synth"
    job_dir.mkdir(parents=True)

    oracle_tasks = [
        (
            "s1_inspect__00_none",
            "inspect",
            {"bandpass_min_elevation_deg": 41.2},
            {"bandpass_min_elevation_deg": 41.2},
        ),
        (
            "s1_precal__00_none",
            "precal",
            {"expected_j1331_model_amp_jy": 7.67, "flag_recall_min": 0.95, "clean_fpr_max": 0.05},
            {
                "j1331_model_amp_jy": 7.67,
                "must_flag_recall": 1.0,
                "clean_false_positive_rate": 0.008,
            },
        ),
        (
            "s1_calibrate__00_none",
            "calibrate",
            {"expected_pcal_flux_jy": 1.85, "max_slope_deg_100mhz": 5.0},
            {"pcal_flux_jy": 1.85, "j1331_phase_slope_deg_per_100mhz": 0.42},
        ),
        (
            "s1_image__00_none",
            "image",
            {
                "pixels_per_beam_min": 3.0,
                "pixels_per_beam_max": 7.0,
                "max_negative_bowl_ratio": 0.12,
            },
            {"pixels_per_beam": 4.8, "negative_bowl_ratio": 0.009},
        ),
        (
            "s2_e2e_synth__00_none",
            "e2e",
            {
                "calibration": {"expected_pcal_flux_jy": 1.85},
                "image": {"max_negative_bowl_ratio": 0.12},
            },
            {"calibration": {"pcal_flux_jy": 1.85}, "image": {"negative_bowl_ratio": 0.011}},
        ),
    ]

    t_results: list[TrialResult] = []
    base_dt = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
    for idx, (tname, stage, truth, measured) in enumerate(oracle_tasks):
        _, tr = _write_harbor_trial(
            job_dir,
            task_name=tname,
            agent_name="oracle",
            model_name=None,
            reward_dict={"reward": 1.0, "passed": 1},
            metrics_dict={
                "task_id": tname.split("__")[0],
                "stage": stage,
                "table_locked": False,
                "truth": truth,
                "measured": measured,
                "subscores": {},
            },
            trajectory=None,
            started_at=base_dt + timedelta(minutes=idx),
        )
        t_results.append(tr)

    # Include Harbor job-level result.json to ensure discover_trial_dirs skips JobResult.
    _write_job_result(job_dir, t_results)

    rc = cli_main([str(job_dir), "--n-boot", "500"])
    assert rc == 0

    out_dir = job_dir / "analysis"
    summary = _assert_no_nan_or_emdash_in_file(out_dir / "summary.json")
    _assert_no_nan_or_emdash_in_file(out_dir / "summary.md")
    assert not (out_dir / "plots").exists()

    assert summary["n_trials"] == 5
    cell_00 = summary["factorial"]["overall"]["cells"]["00_none"]
    assert cell_00["n_trials"] == 5
    assert cell_00["pass_at_1"] == 1.0
    assert cell_00["pass_at_1_wilson_ci"][1] == 1.0
    assert 0.55 < cell_00["pass_at_1_wilson_ci"][0] < 0.90
    assert cell_00["reward_mean"] == 1.0
    assert cell_00["mean_total_tokens"] == 0.0
    assert cell_00["mean_usd"] == 0.0

    for c_name, c_obj in summary["factorial"]["overall"]["contrasts"].items():
        assert c_obj["status"] == "insufficient_arms", c_name
        assert c_obj["estimate"] is None

    for p_name, p_obj in summary["factorial"]["overall"]["parity"].items():
        assert p_obj["status"] == "insufficient_models", p_name
        assert p_obj["estimate"] is None

    assert summary["taxonomy"]["overall_counts"][FAILURE_NONE] == 5
    assert summary["taxonomy"]["n_failed"] == 0

    # Re-running analyze when <job_dir>/analysis already exists must not double-count or scan analysis/.
    rc_rerun = cli_main([str(job_dir), "--n-boot", "200"])
    assert rc_rerun == 0
    summary_rerun = _assert_no_nan_or_emdash_in_file(out_dir / "summary.json")
    assert summary_rerun["n_trials"] == 5


def test_full_factorial_synergy_and_parity_recovery(tmp_path: Path) -> None:
    """6-arm, 4-model multi-job synthetic ablation recovers exact 2x2 effects, synergy, and parity."""
    stage_job_dir = tmp_path / "suite_r_ablation"
    e2e_job_dir = tmp_path / "suite_r_e2e_ablation"
    stage_job_dir.mkdir(parents=True)
    e2e_job_dir.mkdir(parents=True)

    # Planted mean rewards by arm:
    # R_00 = 0.20, R_10 = 0.45, R_01_raw = 0.35, R_01_casa = 0.50, R_11 = 0.90, R_11_orch = 0.95
    base_arm_rewards = {
        "00_none": 0.20,
        "10_tools": 0.45,
        "01_skills_raw": 0.35,
        "01_skills_casa": 0.50,
        "11_full": 0.90,
        "11_orchestrated": 0.95,
    }
    # Zero-mean symmetric model offsets (+-0.02) so overall arm means stay exact while models differ.
    models = [
        ("claude-code", "anthropic/claude-haiku-4-5", -0.02, 20_000, 1_000),
        ("claude-code", "anthropic/claude-sonnet-4-6", +0.02, 20_000, 1_000),
        ("opencode", "nrp/kimi", -0.01, 25_000, 1_200),
        ("opencode", "nrp/glm-5", +0.01, 25_000, 1_200),
    ]
    # Include 34-char r1_3c391_calibrate__01_skills_casa to test Harbor 32-char trial_name truncation.
    stage_tasks = ["r1_3c391_inspect", "r1_3c391_calibrate"]
    e2e_tasks = ["r2_3c391_e2e"]
    attempt_jitters = [-0.01, 0.0, +0.01]  # Exact zero mean per cell, non-zero attempt variance.

    t_counter = 0
    base_dt = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)

    for task_id in stage_tasks:
        stage_name = "inspect" if "inspect" in task_id else "calibrate"
        for arm in ("00_none", "10_tools", "01_skills_raw", "01_skills_casa", "11_full"):
            for agent_name, model_name, m_off, in_tok, out_tok in models:
                for jitter in attempt_jitters:
                    t_counter += 1
                    rew = round(base_arm_rewards[arm] + m_off + jitter, 6)
                    passed = 1 if rew >= 0.85 else 0
                    traj = Trajectory(
                        session_id=f"s-{t_counter}",
                        agent=Agent(name=agent_name, version="1.0", model_name=model_name),
                        steps=[
                            Step(step_id=1, source="user", message="Run stage"),
                            Step(
                                step_id=2,
                                source="agent",
                                message="Done",
                                model_name=model_name,
                                metrics=Metrics(prompt_tokens=in_tok, completion_tokens=out_tok),
                            ),
                        ],
                    )
                    _write_harbor_trial(
                        stage_job_dir,
                        task_name=f"{task_id}__{arm}",
                        agent_name=agent_name,
                        model_name=model_name,
                        reward_dict={"reward": rew, "passed": passed},
                        metrics_dict={
                            "task_id": task_id,
                            "stage": stage_name,
                            "table_locked": False,
                            "summary_exists": True,
                            "valid_json": True,
                            "has_corrected_data": True,
                            "corrected_distinct_from_data": True,
                            "subscores": {
                                "bandpass_flatness_score": rew,
                                "dead_antenna_score": rew,
                            },
                        },
                        trajectory=traj,
                        started_at=base_dt + timedelta(seconds=t_counter),
                    )

    for task_id in e2e_tasks:
        for arm in (
            "00_none",
            "10_tools",
            "01_skills_raw",
            "01_skills_casa",
            "11_full",
            "11_orchestrated",
        ):
            for agent_name, model_name, m_off, in_tok, out_tok in models:
                for jitter in attempt_jitters:
                    t_counter += 1
                    rew = round(base_arm_rewards[arm] + m_off + jitter, 6)
                    passed = 1 if rew >= 0.85 else 0
                    traj = Trajectory(
                        session_id=f"s-{t_counter}",
                        agent=Agent(name=agent_name, version="1.0", model_name=model_name),
                        steps=[
                            Step(step_id=1, source="user", message="Run e2e"),
                            Step(
                                step_id=2,
                                source="agent",
                                message="Done",
                                model_name=model_name,
                                metrics=Metrics(prompt_tokens=in_tok, completion_tokens=out_tok),
                            ),
                        ],
                    )
                    _write_harbor_trial(
                        e2e_job_dir,
                        task_name=f"{task_id}__{arm}",
                        agent_name=agent_name,
                        model_name=model_name,
                        reward_dict={"reward": rew, "passed": passed},
                        metrics_dict={
                            "task_id": task_id,
                            "stage": "e2e",
                            "table_locked": False,
                            "calibration": {
                                "has_corrected_data": True,
                                "corrected_distinct_from_data": True,
                                "subscores": {"bandpass_flatness_score": rew},
                            },
                            "image": {
                                "measured": {"image_sets_found": 1, "negative_bowl_ratio": 0.02},
                                "subscores": {"products_score": 1.0, "beam_sampling_score": rew},
                            },
                            "subscores": {"calibration_score": rew, "delivered_image_score": rew},
                        },
                        trajectory=traj,
                        started_at=base_dt + timedelta(seconds=t_counter),
                    )

    records = collect_trials([stage_job_dir, e2e_job_dir])
    # 2 stage tasks * 5 arms * 4 models * 3 attempts (120) + 1 e2e * 6 arms * 4 models * 3 attempts (72) = 192
    assert len(records) == 192
    # Verify 34-char task name was NOT truncated to 01_skills_ca.
    assert any(
        r["base_task"] == "r1_3c391_calibrate" and r["arm"] == "01_skills_casa" for r in records
    )

    fact_1 = analyze_factorial(records, n_boot=10_000, seed=20260413)
    fact_2 = analyze_factorial(records, n_boot=10_000, seed=20260413)

    contrasts = fact_1["overall"]["contrasts"]
    assert math.isclose(contrasts["I_casa"]["estimate"], 0.15, abs_tol=1e-9)
    assert math.isclose(contrasts["I_raw"]["estimate"], 0.30, abs_tol=1e-9)
    assert math.isclose(contrasts["Delta_PromptAdapt"]["estimate"], 0.15, abs_tol=1e-9)
    assert math.isclose(contrasts["Delta_Orch"]["estimate"], 0.05, abs_tol=1e-9)
    assert math.isclose(contrasts["Delta_Tools_casa"]["estimate"], 0.325, abs_tol=1e-9)
    assert math.isclose(contrasts["Delta_Skills_casa"]["estimate"], 0.375, abs_tol=1e-9)
    assert math.isclose(contrasts["Delta_Tools_raw"]["estimate"], 0.40, abs_tol=1e-9)
    assert math.isclose(contrasts["Delta_Skills_raw"]["estimate"], 0.30, abs_tol=1e-9)

    for cname, cobj in contrasts.items():
        assert cobj["status"] == "ok", cname
        assert cobj["ci_low"] <= cobj["estimate"] <= cobj["ci_high"], cname
        # Exact deterministic bootstrap reproducibility across runs.
        assert cobj["ci_low"] == fact_2["overall"]["contrasts"][cname]["ci_low"]
        assert cobj["ci_high"] == fact_2["overall"]["contrasts"][cname]["ci_high"]

    # Parity: Haiku @ 11_full (0.90 - 0.02 = 0.88) vs Sonnet @ 00_none (0.20 + 0.02 = 0.22) -> +0.66
    # Haiku cost (20k * $1 + 1k * $5)/1e6 = $0.025 vs Sonnet (20k * $3 + 1k * $15)/1e6 = $0.075 -> ratio 1/3
    parity_claude = fact_1["overall"]["parity"]["anthropic_haiku_vs_sonnet__11_full_vs_00_none"]
    assert parity_claude["status"] == "ok"
    assert math.isclose(parity_claude["estimate"], 0.66, abs_tol=1e-9)
    assert math.isclose(parity_claude["cost_ratio"], 1.0 / 3.0, rel_tol=1e-6)

    # Parity: Kimi @ 11_full (0.89) vs GLM-5 @ 00_none (0.21) -> +0.68
    parity_nrp = fact_1["overall"]["parity"]["nrp_kimi_vs_glm5__11_full_vs_00_none"]
    assert parity_nrp["status"] == "ok"
    assert math.isclose(parity_nrp["estimate"], 0.68, abs_tol=1e-9)


def test_pricing_and_haiku_100k_cliff() -> None:
    """Verify exact list-price calculations, cache read/write TTLs, NRP models, and Haiku 100K 5x cliff."""
    # 1. Exact boundary <= 100,000 input tokens on Haiku: NO cliff (1x).
    # 80,000 uncached ($1/M) + 10,000 cache_read ($0.10/M) + 10,000 cache_write_5m ($1.25/M) = 100,000 prompt
    # + 2,000 output ($5/M) -> 0.08 + 0.001 + 0.0125 + 0.01 = $0.1035
    below = price_turn(
        "anthropic/claude-haiku-4-5",
        uncached_input_tokens=80_000,
        cache_read_tokens=10_000,
        cache_write_tokens=10_000,
        output_tokens=2_000,
    )
    assert below.input_tokens == 100_000
    assert below.cliff_triggered is False
    assert math.isclose(below.total_usd, 0.1035, abs_tol=1e-10)

    # 2. Above 100,000 input tokens on Haiku (e.g. 120,000 total prompt tokens): 5x cliff!
    # 20,000 uncached ($5/M) + 90,000 cache_read ($0.50/M) + 10,000 cache_write_5m ($6.25/M)
    # + 2,000 output ($25/M) -> 0.10 + 0.045 + 0.0625 + 0.05 = $0.2575
    above = price_turn(
        "anthropic/claude-haiku-4-5",
        uncached_input_tokens=20_000,
        cache_read_tokens=90_000,
        cache_write_tokens=10_000,
        output_tokens=2_000,
    )
    assert above.input_tokens == 120_000
    assert above.cliff_triggered is True
    assert math.isclose(above.input_usd, 0.10, abs_tol=1e-10)
    assert math.isclose(above.cache_read_usd, 0.045, abs_tol=1e-10)
    assert math.isclose(above.cache_write_usd, 0.0625, abs_tol=1e-10)
    assert math.isclose(above.output_usd, 0.05, abs_tol=1e-10)
    assert math.isclose(above.total_usd, 0.2575, abs_tol=1e-10)

    # 3. Sonnet 4.6 at 150,000 tokens has NO 100K cliff, and honors 1h cache write (2.0x = $6/M).
    sonnet = price_turn(
        "anthropic/claude-sonnet-4-6",
        uncached_input_tokens=10_000,
        cache_read_tokens=100_000,
        cache_write_tokens=40_000,
        cache_write_1h_tokens=40_000,
        output_tokens=4_000,
    )
    assert sonnet.cliff_triggered is False
    # 10k*3 + 100k*0.30 + 40k*6.00 + 4k*15 = 30k + 30k + 240k + 60k = $0.36
    assert math.isclose(sonnet.total_usd, 0.36, abs_tol=1e-10)

    # 4. NRP open-weights models (nrp/kimi and nrp/glm-5) + oracle = $0.00.
    kimi = price_turn(
        "nrp/kimi",
        uncached_input_tokens=100_000,
        cache_read_tokens=50_000,
        output_tokens=10_000,
    )
    # 100k*0.60 + 50k*0.10 + 10k*3.00 = 60k + 5k + 30k = $0.095
    assert math.isclose(kimi.total_usd, 0.095, abs_tol=1e-10)

    glm5 = price_turn(
        "nrp/glm-5",
        uncached_input_tokens=100_000,
        cache_read_tokens=50_000,
        output_tokens=10_000,
    )
    # 100k*1.00 + 50k*0.20 + 10k*3.20 = 100k + 10k + 32k = $0.142
    assert math.isclose(glm5.total_usd, 0.142, abs_tol=1e-10)

    ora = compute_trial_pricing(model_name=None, agent_name="oracle")
    assert ora.total_usd == 0.0
    assert ora.total_tokens == 0

    # 5. Multi-turn trajectory with 1 normal turn + 1 cliff turn + subagent trajectory.
    traj = Trajectory(
        session_id="main",
        agent=Agent(name="claude-code", version="1.0", model_name="claude-haiku-4-5"),
        steps=[
            Step(step_id=1, source="user", message="Start"),
            Step(
                step_id=2,
                source="agent",
                message="Turn 1",
                model_name="claude-haiku-4-5",
                metrics=Metrics(
                    prompt_tokens=100_000,
                    completion_tokens=2_000,
                    cached_tokens=10_000,
                    extra={
                        "cache_read_input_tokens": 10_000,
                        "cache_creation_input_tokens": 10_000,
                    },
                ),
            ),
            Step(
                step_id=3,
                source="agent",
                message="Turn 2 crosses 100K",
                model_name="claude-haiku-4-5",
                metrics=Metrics(
                    prompt_tokens=120_000,
                    completion_tokens=2_000,
                    cached_tokens=90_000,
                    extra={
                        "cache_read_input_tokens": 90_000,
                        "cache_creation_input_tokens": 10_000,
                    },
                ),
            ),
        ],
    )
    tp = compute_trial_pricing(
        model_name="anthropic/claude-haiku-4-5",
        agent_name="claude-code",
        trajectory_dict=traj.to_json_dict(),
    )
    assert tp.peak_turn_tokens == 120_000
    assert tp.haiku_cliff_triggered is True
    assert tp.cliff_turns == 1
    assert math.isclose(tp.total_usd, 0.1035 + 0.2575, abs_tol=1e-8)

    # 6. Fallback to AgentContext when trajectory.json is absent.
    fallback_tp = compute_trial_pricing(
        model_name="anthropic/claude-sonnet-4-6",
        agent_name="claude-code",
        trajectory_dict=None,
        trial_result_dict={
            "agent_result": {
                "n_input_tokens": 50_000,
                "n_cache_tokens": 20_000,
                "n_output_tokens": 2_000,
            }
        },
    )
    # 30k uncached * $3 + 20k cache_read * $0.30 + 2k out * $15 = 0.09 + 0.006 + 0.03 = $0.126
    assert fallback_tp.pricing_source == "agent_context"
    assert math.isclose(fallback_tp.total_usd, 0.126, abs_tol=1e-8)


def test_trajectory_tool_and_mcp_telemetry(tmp_path: Path) -> None:
    """Verify exact MCP server classification, bash non-zero exits, and embedded subagent turns."""
    sub_traj = Trajectory(
        trajectory_id="sub-1",
        session_id="sess-1",
        agent=Agent(name="claude-code", version="1.0", model_name="claude-haiku-4-5"),
        steps=[
            Step(step_id=1, source="user", message="Subagent inspect"),
            Step(
                step_id=2,
                source="agent",
                message="Calling inspect + create",
                model_name="claude-haiku-4-5",
                tool_calls=[
                    ToolCall(
                        tool_call_id="sc1",
                        function_name="mcp__ms-inspect__ms_field_list",
                        arguments={},
                    ),
                    ToolCall(
                        tool_call_id="sc2",
                        function_name="ms_reduction_log",
                        arguments={},
                    ),
                ],
                observation=Observation(
                    results=[
                        ObservationResult(source_call_id="sc1", content='{"ok": true}'),
                        ObservationResult(source_call_id="sc2", content='{"ok": true}'),
                    ]
                ),
            ),
        ],
    )
    root_traj = Trajectory(
        session_id="sess-1",
        agent=Agent(name="claude-code", version="1.0", model_name="claude-sonnet-4-6"),
        steps=[
            Step(step_id=1, source="user", message="Reduce MS"),
            Step(
                step_id=2,
                source="agent",
                message="Modify + bash calls",
                model_name="claude-sonnet-4-6",
                tool_calls=[
                    ToolCall(
                        tool_call_id="c1",
                        function_name="mcp__ms-modify__ms_setjy",
                        arguments={},
                    ),
                    ToolCall(
                        tool_call_id="c2",
                        function_name="Bash",
                        arguments={"command": "python bad.py"},
                    ),
                    ToolCall(
                        tool_call_id="c3",
                        function_name="Bash",
                        arguments={"command": "python ok.py"},
                    ),
                ],
                observation=Observation(
                    results=[
                        ObservationResult(source_call_id="c1", content='{"ok": true}'),
                        ObservationResult(
                            source_call_id="c2",
                            content="[stderr]\nSyntaxError\n[exit_code] 1",
                        ),
                        ObservationResult(source_call_id="c3", content="[stdout]\nall good"),
                    ]
                ),
            ),
        ],
        subagent_trajectories=[sub_traj],
    )

    job_dir = tmp_path / "telemetry_job"
    job_dir.mkdir()
    _write_harbor_trial(
        job_dir,
        task_name="s2_e2e_synth__11_orchestrated",
        agent_name="claude-code",
        model_name="anthropic/claude-sonnet-4-6",
        reward_dict={"reward": 0.92, "passed": 1},
        metrics_dict={"stage": "e2e", "table_locked": False},
        trajectory=root_traj,
    )
    recs = collect_trials([job_dir])
    assert len(recs) == 1
    r = recs[0]
    assert r["n_turns"] == 2
    assert r["mcp_calls_total"] == 3
    assert r["mcp_calls_inspect"] == 1
    assert r["mcp_calls_modify"] == 1
    assert r["mcp_calls_create"] == 1
    assert r["bash_calls"] == 2
    assert r["bash_nonzero_exits"] == 1
    assert r["failure_mode"] == FAILURE_NONE


def test_failure_taxonomy_all_categories() -> None:
    """Verify 100% deterministic classification across passing ('none') and all 7 SPEC.md failure modes."""
    # 0. Passing trial -> 'none'
    assert (
        classify_failure(
            passed=1,
            stage="calibrate",
            reward=0.95,
            exception_type="AgentTimeoutError",  # ignored when passed == 1
        )
        == FAILURE_NONE
    )

    # 1a. lock_or_timeout via AgentTimeoutError
    assert (
        classify_failure(
            passed=0,
            stage="calibrate",
            reward=0.0,
            exception_type="AgentTimeoutError",
        )
        == "lock_or_timeout"
    )
    # 1b. lock_or_timeout via active casacore table lock
    assert (
        classify_failure(
            passed=0,
            stage="precal",
            reward=0.0,
            metrics_dict={"stage": "precal", "table_locked": True},
        )
        == "lock_or_timeout"
    )

    # 2. tool_protocol_error (attempting MCP tool in 01_skills_raw where MCP is absent)
    assert (
        classify_failure(
            passed=0,
            stage="inspect",
            arm="01_skills_raw",
            reward=0.0,
            mcp_calls_total=2,
            protocol_errors=2,
            metrics_dict={"stage": "inspect", "summary_exists": False, "valid_json": False},
        )
        == "tool_protocol_error"
    )

    # 3a. script_or_casa_crash via missing primary deliverable (inspect_summary.json missing)
    assert (
        classify_failure(
            passed=0,
            stage="inspect",
            arm="00_none",
            reward=0.0,
            reward_dict={"reward": 0.0, "passed": 0, "valid_json_score": 0.0},
            metrics_dict={"stage": "inspect", "summary_exists": False, "valid_json": False},
        )
        == "script_or_casa_crash"
    )
    # 3b. script_or_casa_crash via NonZeroAgentExitCodeError
    assert (
        classify_failure(
            passed=0,
            stage="image",
            arm="10_tools",
            reward=0.0,
            exception_type="NonZeroAgentExitCodeError",
        )
        == "script_or_casa_crash"
    )

    # 4. flux_model_or_fluxscale_error (precal model_data_score < 0.5 & calibrate fluxscale_score < 0.5)
    assert (
        classify_failure(
            passed=0,
            stage="precal",
            arm="00_none",
            reward=0.60,
            reward_dict={
                "reward": 0.60,
                "passed": 0,
                "calibrators_ms_score": 1.0,
                "model_data_score": 0.10,
                "flag_recall_score": 0.95,
                "clean_fpr_score": 1.0,
            },
            metrics_dict={"stage": "precal", "measured": {"calibrators_ms_exists": True}},
        )
        == "flux_model_or_fluxscale_error"
    )
    assert (
        classify_failure(
            passed=0,
            stage="calibrate",
            arm="10_tools",
            reward=0.65,
            reward_dict={
                "reward": 0.65,
                "passed": 0,
                "retention_and_dead_ant_score": 1.0,
                "bandpass_flatness_score": 0.95,
                "fluxscale_score": 0.20,
                "target_fidelity_score": 0.60,
            },
            metrics_dict={
                "stage": "calibrate",
                "has_corrected_data": True,
                "corrected_distinct_from_data": True,
            },
        )
        == "flux_model_or_fluxscale_error"
    )

    # 5. skipped_delay_or_bandpass (bandpass_flatness_score < 0.5)
    assert (
        classify_failure(
            passed=0,
            stage="calibrate",
            arm="00_none",
            reward=0.55,
            reward_dict={
                "reward": 0.55,
                "passed": 0,
                "retention_and_dead_ant_score": 1.0,
                "bandpass_flatness_score": 0.15,
                "fluxscale_score": 0.95,
                "target_fidelity_score": 0.30,
            },
            metrics_dict={
                "stage": "calibrate",
                "has_corrected_data": True,
                "corrected_distinct_from_data": True,
            },
        )
        == "skipped_delay_or_bandpass"
    )

    # 6. flagging_recall_or_overflag (precal flag_recall_score < 0.5 & inspect dead_antenna_score < 0.5)
    assert (
        classify_failure(
            passed=0,
            stage="precal",
            arm="01_skills_casa",
            reward=0.65,
            reward_dict={
                "reward": 0.65,
                "passed": 0,
                "calibrators_ms_score": 1.0,
                "model_data_score": 1.0,
                "flag_recall_score": 0.25,
                "clean_fpr_score": 1.0,
            },
            metrics_dict={"stage": "precal", "measured": {"calibrators_ms_exists": True}},
        )
        == "flagging_recall_or_overflag"
    )
    assert (
        classify_failure(
            passed=0,
            stage="inspect",
            arm="00_none",
            reward=0.80,
            reward_dict={
                "reward": 0.80,
                "passed": 0,
                "valid_json_score": 1.0,
                "flux_bandpass_score": 1.0,
                "phase_vs_target_score": 1.0,
                "dead_antenna_score": 0.0,
                "elevation_score": 1.0,
            },
            metrics_dict={"stage": "inspect", "summary_exists": True, "valid_json": True},
        )
        == "flagging_recall_or_overflag"
    )

    # 7. beam_sampling_or_clean_bowl (negative_bowl_ratio >= 0.12 or beam_sampling_score < 0.5)
    assert (
        classify_failure(
            passed=0,
            stage="image",
            arm="00_none",
            reward=0.62,
            reward_dict={
                "reward": 0.62,
                "passed": 0,
                "products_score": 1.0,
                "beam_sampling_score": 0.90,
                "residual_rms_score": 0.85,
                "deconv_and_bowl_score": 0.20,
            },
            metrics_dict={
                "stage": "image",
                "measured": {"image_sets_found": 1, "negative_bowl_ratio": 0.24},
            },
        )
        == "beam_sampling_or_clean_bowl"
    )

    # Fallback check: all subscores >= 0.5 (e.g. 0.72..0.82) so composite < 0.85 -> lowest subscore wins.
    fallback_cat = classify_failure(
        passed=0,
        stage="image",
        arm="10_tools",
        reward=0.78,
        reward_dict={
            "reward": 0.78,
            "passed": 0,
            "products_score": 1.0,
            "beam_sampling_score": 0.65,
            "residual_rms_score": 0.80,
            "deconv_and_bowl_score": 0.80,
        },
        metrics_dict={
            "stage": "image",
            "measured": {"image_sets_found": 1, "negative_bowl_ratio": 0.04},
        },
    )
    assert fallback_cat in FAILURE_CATEGORIES
    assert fallback_cat == "beam_sampling_or_clean_bowl"

    # Verify summarize_failure_taxonomy counts all 7 categories + none accurately.
    synth_records = [
        {"arm": "00_none", "model_name": "nrp/kimi", "failure_mode": cat}
        for cat in (FAILURE_NONE, *FAILURE_CATEGORIES)
    ]
    tax = summarize_failure_taxonomy(synth_records)
    assert tax["n_trials"] == 8
    assert tax["n_failed"] == 7
    for cat in (FAILURE_NONE, *FAILURE_CATEGORIES):
        assert tax["overall_counts"][cat] == 1


def test_cli_outputs_and_plot_hygiene(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI file layout, truth-vs-measured table ordering, zero em/en dashes, and --plots gating."""
    job_dir = tmp_path / "cli_job"
    job_dir.mkdir()

    _write_harbor_trial(
        job_dir,
        task_name="s1_inspect__00_none",
        agent_name="claude-code",
        model_name="anthropic/claude-haiku-4-5",
        reward_dict={
            "reward": 0.65,
            "passed": 0,
            "valid_json_score": 1.0,
            "flux_bandpass_score": 1.0,
            "phase_vs_target_score": 1.0,
            "dead_antenna_score": 0.0,
            "elevation_score": 0.5,
        },
        metrics_dict={
            "task_id": "s1_inspect",
            "stage": "inspect",
            "summary_exists": True,
            "valid_json": True,
            "truth": {
                "flux_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "dead_antennas": ["ea07"],
                "bandpass_min_elevation_deg": 41.25,
                "elevation_tol_deg": 0.25,
            },
            "measured": {
                "flux_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "dead_antennas": [],
                "bandpass_min_elevation_deg": 42.10,
            },
            "subscores": {"dead_antenna_score": 0.0},
        },
    )
    _write_harbor_trial(
        job_dir,
        task_name="s1_inspect__11_full",
        agent_name="claude-code",
        model_name="anthropic/claude-haiku-4-5",
        reward_dict={
            "reward": 1.0,
            "passed": 1,
            "valid_json_score": 1.0,
            "flux_bandpass_score": 1.0,
            "phase_vs_target_score": 1.0,
            "dead_antenna_score": 1.0,
            "elevation_score": 1.0,
        },
        metrics_dict={
            "task_id": "s1_inspect",
            "stage": "inspect",
            "summary_exists": True,
            "valid_json": True,
            "truth": {
                "flux_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "dead_antennas": ["ea07"],
                "bandpass_min_elevation_deg": 41.25,
                "elevation_tol_deg": 0.25,
            },
            "measured": {
                "flux_calibrator": "J1331+3030",
                "phase_calibrator": "J1822-0938",
                "dead_antennas": ["ea07"],
                "bandpass_min_elevation_deg": 41.25,
            },
            "subscores": {"dead_antenna_score": 1.0},
        },
    )

    # 1. Default invocation WITHOUT --plots: zero PNGs created anywhere.
    rc = cli_main([str(job_dir), "--n-boot", "200"])
    assert rc == 0
    stdout_text = capsys.readouterr().out
    assert "\u2014" not in stdout_text and "\u2013" not in stdout_text

    out_dir = job_dir / "analysis"
    assert (out_dir / "trials.csv").is_file()
    assert (out_dir / "trials.jsonl").is_file()
    summary = _assert_no_nan_or_emdash_in_file(out_dir / "summary.json")
    md_text = _assert_no_nan_or_emdash_in_file(out_dir / "summary.md")

    # Truth-vs-Measured table MUST appear before Cell Summary, Contrasts, and Taxonomy.
    idx_truth = md_text.index("## 1. Truth vs Measured Physical Metrics")
    idx_cells = md_text.index("## 2. Capability Arm Cell Summary")
    idx_contr = md_text.index("## 3. 2x2 Factorial Contrasts and Synergy")
    idx_tax = md_text.index("## 6. Deterministic Failure-Mode Taxonomy by Arm")
    assert idx_truth < idx_cells < idx_contr < idx_tax

    assert summary["plot_files"] == []
    assert list(job_dir.rglob("*.png")) == []

    # Verify trials.csv and trials.jsonl contents.
    jsonl_rows = [
        json.loads(line)
        for line in (out_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_rows) == 2
    with (out_dir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2

    # 2. Subprocess CLI invocation WITH --plots: PNGs land strictly inside <job_dir>/analysis/plots/.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "radio_bench.analysis.cli",
            str(job_dir),
            "--plots",
            "--n-boot",
            "200",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    plots_dir = out_dir / "plots"
    assert plots_dir.is_dir()
    pngs = sorted(job_dir.rglob("*.png"))
    assert len(pngs) == 3
    for png in pngs:
        assert png.parent == plots_dir
        assert png.stat().st_size > 1_000


def test_wilson_ci_edge_cases() -> None:
    """Wilson score interval handles n=0, k=0, and k=n without NaN or rounding overshoot."""
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0
    assert 0.25 < hi0 < 0.35
    lo1, hi1 = wilson_ci(10, 10)
    assert 0.65 < lo1 < 0.75
    assert hi1 == 1.0


def test_aggregate_token_fallback_and_crashed_trial_recovery(tmp_path: Path) -> None:
    """Aggregate FinalMetrics/AgentContext fallback and missing-verifier crash recovery work cleanly."""
    # 1. Trajectory without per-step metrics falls back to FinalMetrics cleanly.
    traj_final_only = Trajectory(
        session_id="sess-final-only",
        agent=Agent(name="opencode", version="1.0", model_name="nrp/kimi"),
        steps=[
            Step(step_id=1, source="user", message="Inspect MS."),
            Step(step_id=2, source="agent", message="Done."),
        ],
        final_metrics=FinalMetrics(
            total_prompt_tokens=25_000,
            total_completion_tokens=4_000,
            total_cached_tokens=5_000,
            total_steps=2,
            extra={"total_cache_write_tokens": 0},
        ),
    )
    priced = compute_trial_pricing(
        model_name="nrp/kimi",
        agent_name="opencode",
        trajectory_dict=traj_final_only.model_dump(mode="json"),
        trial_result_dict=None,
    )
    # OpenCode: total_prompt_tokens (25,000) = uncached (20,000) + cache_read (5,000).
    assert priced.uncached_input_tokens == 20_000
    assert priced.cache_read_tokens == 5_000
    assert priced.output_tokens == 4_000
    expected_kimi_usd = 0.020 * 0.60 + 0.005 * 0.10 + 0.004 * 3.00
    assert math.isclose(priced.total_usd, expected_kimi_usd, rel_tol=0.0, abs_tol=1e-9)

    # 2. Crashed trial directory with missing verifier/reward.json and missing trajectory.json
    #    falls back to AgentContext tokens and reward=0.0, passed=0.
    crash_job = tmp_path / "crash_job"
    now_dt = datetime(2026, 11, 14, 15, 0, 0, tzinfo=UTC)
    _write_harbor_trial(
        crash_job,
        task_name="s2_precal__10_tools",
        agent_name="claude-code",
        model_name="anthropic/claude-sonnet-4-6",
        reward_dict=None,
        metrics_dict=None,
        trajectory=None,
        agent_context=AgentContext(
            n_input_tokens=15_000,
            n_cache_tokens=5_000,
            n_output_tokens=1_000,
        ),
        exception_info=ExceptionInfo(
            exception_type="AgentRuntimeError",
            exception_message="Container exited with status 137 (OOMKilled)",
            exception_traceback="Traceback ...",
            occurred_at=now_dt + timedelta(seconds=11.5),
        ),
        started_at=now_dt,
    )

    rows = collect_trials([crash_job])
    assert len(rows) == 1
    row = rows[0]
    assert row["reward"] == 0.0
    assert row["passed"] == 0
    assert row["failure_mode"] == "script_or_casa_crash"
    assert row["uncached_input_tokens"] == 10_000
    assert row["cache_read_tokens"] == 5_000
    assert row["output_tokens"] == 1_000
    expected_sonnet_usd = 0.010 * 3.00 + 0.005 * 0.30 + 0.001 * 15.00
    assert math.isclose(row["total_usd"], expected_sonnet_usd, rel_tol=0.0, abs_tol=1e-9)


# === Wave 5 Regression Tests ===


def test_wave5_pricing_opencode_cache_write_aggregate_cliff_and_unknown_model() -> None:
    """Verify OpenCode cache_write non-subtraction, aggregate 1.0x cliff guard, and unknown model source."""
    from radio_bench.analysis import price_from_agent_context

    # 1. OpenCode step-level: prompt_tokens (60,000) = input_tok (50,000) + cache_read (10,000);
    #    cache_write_tokens (20,000) is separate and MUST NOT be subtracted from uncached_input_tokens.
    traj_oc_under = Trajectory(
        session_id="sess-oc-under",
        agent=Agent(name="opencode", version="1.0", model_name="anthropic/claude-haiku-4-5"),
        steps=[
            Step(step_id=1, source="user", message="Run inspect."),
            Step(
                step_id=2,
                source="agent",
                message="Inspected.",
                metrics=Metrics(
                    prompt_tokens=60_000,
                    completion_tokens=4_000,
                    cached_tokens=10_000,
                    extra={"cache_write_tokens": 20_000},
                ),
            ),
        ],
    )
    p_under = compute_trial_pricing(
        model_name="anthropic/claude-haiku-4-5",
        agent_name="opencode",
        trajectory_dict=traj_oc_under.model_dump(mode="json"),
    )
    assert p_under.uncached_input_tokens == 50_000
    assert p_under.cache_read_tokens == 10_000
    assert p_under.cache_write_tokens == 20_000
    assert p_under.input_tokens == 80_000
    assert p_under.haiku_cliff_triggered is False
    expected_under = 0.050 * 1.00 + 0.010 * 0.10 + 0.020 * 1.25 + 0.004 * 5.00
    assert math.isclose(p_under.total_usd, expected_under, rel_tol=0.0, abs_tol=1e-9)

    # 2. OpenCode step-level crossing 100K Haiku cliff via cache_write_tokens:
    #    prompt_tokens=85,000 (75K uncached + 10K cache_read) + 20K cache_write => 105,000 > 100,000!
    traj_oc_cliff = Trajectory(
        session_id="sess-oc-cliff",
        agent=Agent(name="opencode", version="1.0", model_name="anthropic/claude-haiku-4-5"),
        steps=[
            Step(step_id=1, source="user", message="Run calibrate."),
            Step(
                step_id=2,
                source="agent",
                message="Calibrated.",
                metrics=Metrics(
                    prompt_tokens=85_000,
                    completion_tokens=4_000,
                    cached_tokens=10_000,
                    extra={"cache_write_tokens": 20_000},
                ),
            ),
        ],
    )
    p_cliff = compute_trial_pricing(
        model_name="anthropic/claude-haiku-4-5",
        agent_name="opencode",
        trajectory_dict=traj_oc_cliff.model_dump(mode="json"),
    )
    assert p_cliff.uncached_input_tokens == 75_000
    assert p_cliff.input_tokens == 105_000
    assert p_cliff.haiku_cliff_triggered is True
    assert p_cliff.cliff_turns == 1
    expected_cliff = (0.075 * 1.00 + 0.010 * 0.10 + 0.020 * 1.25 + 0.004 * 5.00) * 5.0
    assert math.isclose(p_cliff.total_usd, expected_cliff, rel_tol=0.0, abs_tol=1e-9)

    # 3. OpenCode FinalMetrics fallback also preserves uncached = 50,000 and input = 80,000.
    traj_oc_final = Trajectory(
        session_id="sess-oc-final",
        agent=Agent(name="opencode", version="1.0", model_name="anthropic/claude-haiku-4-5"),
        steps=[
            Step(step_id=1, source="user", message="Inspect."),
            Step(step_id=2, source="agent", message="Step 1."),
            Step(step_id=3, source="agent", message="Step 2."),
        ],
        final_metrics=FinalMetrics(
            total_prompt_tokens=60_000,
            total_completion_tokens=4_000,
            total_cached_tokens=10_000,
            total_steps=3,
            extra={"total_cache_write_tokens": 20_000},
        ),
    )
    p_oc_final = compute_trial_pricing(
        model_name="anthropic/claude-haiku-4-5",
        agent_name="opencode",
        trajectory_dict=traj_oc_final.model_dump(mode="json"),
    )
    assert p_oc_final.uncached_input_tokens == 50_000
    assert p_oc_final.input_tokens == 80_000
    assert p_oc_final.haiku_cliff_triggered is False

    # 4. Aggregate fallback (AgentContext & multi-step FinalMetrics) with 150,000 cumulative input tokens
    #    on Haiku MUST NOT trigger the single-turn 100K 5x cliff unless n_agent_steps == 1.
    p_ctx = price_from_agent_context(
        {"n_input_tokens": 150_000, "n_cache_tokens": 30_000, "n_output_tokens": 10_000},
        model_name="anthropic/claude-haiku-4-5",
        agent_name="claude-code",
    )
    assert p_ctx.input_tokens == 150_000
    assert p_ctx.uncached_input_tokens == 120_000
    assert p_ctx.haiku_cliff_triggered is False
    assert p_ctx.cliff_turns == 0
    assert p_ctx.peak_turn_tokens == 0
    expected_ctx_1x = 0.120 * 1.00 + 0.030 * 0.10 + 0.010 * 5.00
    assert math.isclose(p_ctx.total_usd, expected_ctx_1x, rel_tol=0.0, abs_tol=1e-9)

    p_ctx_single = price_from_agent_context(
        {"n_input_tokens": 150_000, "n_cache_tokens": 30_000, "n_output_tokens": 10_000},
        model_name="anthropic/claude-haiku-4-5",
        agent_name="claude-code",
        n_agent_steps=1,
    )
    assert p_ctx_single.haiku_cliff_triggered is True
    assert p_ctx_single.cliff_turns == 1
    assert p_ctx_single.peak_turn_tokens == 150_000

    traj_haiku_multi_final = Trajectory(
        session_id="sess-haiku-multi-final",
        agent=Agent(name="claude-code", version="1.0", model_name="anthropic/claude-haiku-4-5"),
        steps=[
            Step(step_id=1, source="user", message="Start."),
            Step(step_id=2, source="agent", message="Turn 1."),
            Step(step_id=3, source="agent", message="Turn 2."),
        ],
        final_metrics=FinalMetrics(
            total_prompt_tokens=150_000,
            total_completion_tokens=10_000,
            total_cached_tokens=30_000,
            total_steps=3,
        ),
    )
    p_multi_fm = compute_trial_pricing(
        model_name="anthropic/claude-haiku-4-5",
        agent_name="claude-code",
        trajectory_dict=traj_haiku_multi_final.model_dump(mode="json"),
    )
    assert p_multi_fm.haiku_cliff_triggered is False
    assert p_multi_fm.cliff_turns == 0
    assert p_multi_fm.peak_turn_tokens == 0
    assert math.isclose(p_multi_fm.total_usd, expected_ctx_1x, rel_tol=0.0, abs_tol=1e-9)

    # 5. Unknown model pricing_source: "agent_reported_cost" when fallback_cost_usd used, else "unknown_model".
    p_unk_fb = price_from_agent_context(
        {"n_input_tokens": 5_000, "n_output_tokens": 1_000, "cost_usd": 0.37},
        model_name="custom/mystery-llm-v9",
    )
    assert p_unk_fb.pricing_source == "agent_reported_cost"
    assert math.isclose(p_unk_fb.total_usd, 0.37, rel_tol=0.0, abs_tol=1e-9)

    p_unk_zero = price_from_agent_context(
        {"n_input_tokens": 5_000, "n_output_tokens": 1_000},
        model_name="custom/mystery-llm-v9",
    )
    assert p_unk_zero.pricing_source == "unknown_model"
    assert p_unk_zero.total_usd == 0.0


def test_wave5_two_stage_bootstrap_no_variance_inflation_and_wilson_validation() -> None:
    """Confirm two-stage bootstrap SE matches theoretical sqrt((sigma_w^2 / n_att) / K) without (2 - 1/K)x inflation."""
    for bad_k, bad_n in [(-1, 5), (0, -1), (6, 5), (True, 5), (1, False)]:
        with pytest.raises(ValueError):
            wilson_ci(bad_k, bad_n)  # type: ignore[arg-type]

    # Zero between-task variance across K = 6 tasks, n_att = 4 attempts per arm.
    # Deterministic 0.20 on 00_none, 10_tools, 01_skills_casa; [0.40, 0.40, 0.80, 0.80] on 11_full
    # (population within-task variance sigma_w^2 = 0.04).
    # For Delta_Tools_casa = 0.5 * (R_11 - R_01 + R_10 - R_00) = 0.5 * R_11 - 0.10:
    # True SE = 0.5 * sqrt((0.04 / 4) / 6) = 0.0204124145 (buggy SE = 0.02763854, +35.4% inflated).
    k_tasks = 6
    n_att = 4
    records: list[dict[str, Any]] = []
    for t_idx in range(k_tasks):
        base_task = f"synth_task_{t_idx}"
        for arm, rewards in (
            ("00_none", [0.20, 0.20, 0.20, 0.20]),
            ("10_tools", [0.20, 0.20, 0.20, 0.20]),
            ("01_skills_casa", [0.20, 0.20, 0.20, 0.20]),
            ("11_full", [0.40, 0.40, 0.80, 0.80]),
        ):
            for a_idx, r_val in enumerate(rewards):
                records.append(
                    {
                        "trial_name": f"{base_task}__{arm}__{a_idx}",
                        "base_task": base_task,
                        "suite": "synth",
                        "stage": "calibrate",
                        "is_e2e": False,
                        "arm": arm,
                        "model_name": "anthropic/claude-haiku-4-5",
                        "reward": r_val,
                        "passed": int(r_val >= 0.65),
                        "total_usd": float("nan") if (t_idx == 0 and a_idx == 0) else 0.05,
                        "total_tokens": float("inf") if (t_idx == 0 and a_idx == 1) else 10_000,
                        "total_wall_sec": 15.0,
                    }
                )
        for a_idx in range(n_att):
            records.append(
                {
                    "trial_name": f"{base_task}__sonnet_00__{a_idx}",
                    "base_task": base_task,
                    "suite": "synth",
                    "stage": "calibrate",
                    "is_e2e": False,
                    "arm": "00_none",
                    "model_name": "anthropic/claude-sonnet-4-6",
                    "reward": 0.50,
                    "passed": 0,
                    "total_usd": 0.25,
                    "total_tokens": 20_000,
                    "total_wall_sec": 30.0,
                }
            )

    summary = analyze_factorial(records, n_boot=6_000, seed=20260413)
    dt = summary["overall"]["contrasts"]["Delta_Tools_casa"]
    se_true_contrast = 0.5 * math.sqrt((0.04 / n_att) / k_tasks)
    assert abs(dt["boot_se"] - se_true_contrast) / se_true_contrast < 0.04

    par = summary["overall"]["parity"]["anthropic_haiku_vs_sonnet__11_full_vs_00_none"]
    se_true_parity = math.sqrt((0.04 / n_att) / k_tasks)
    assert abs(par["boot_se"] - se_true_parity) / se_true_parity < 0.04

    c11 = summary["overall"]["cells"]["11_full"]
    assert math.isfinite(c11["mean_usd"])
    assert math.isfinite(c11["mean_total_tokens"])


def test_wave5_unbalanced_attempts_task_equal_and_parity_pass_at_k_strata() -> None:
    """Verify task-equal weighting under 5 stage vs 3 e2e attempts, e2e_only flag, pass@3/5, and strata parity."""
    records: list[dict[str, Any]] = []

    stage_tasks = [("s3_cal", "calibrate", False), ("s4_img", "image", False)]
    e2e_tasks = [("r3c391_e2e", "e2e", True)]

    for btask, stg, is_e2e in stage_tasks + e2e_tasks:
        n_att = 3 if is_e2e else 5
        for a_idx in range(n_att):
            s_pass = int(a_idx < (1 if is_e2e else 4))
            s_rew = 0.30 if is_e2e else 0.90
            s_usd = 0.90 if is_e2e else 0.10
            s_tok = 90_000 if is_e2e else 10_000
            s_wall = 90.0 if is_e2e else 10.0
            records.append(
                {
                    "trial_name": f"{btask}__11_full__haiku_{a_idx}",
                    "base_task": btask,
                    "suite": "r3c391",
                    "stage": stg,
                    "is_e2e": is_e2e,
                    "arm": "11_full",
                    "model_name": "anthropic/claude-haiku-4-5",
                    "reward": s_rew,
                    "passed": s_pass,
                    "total_usd": s_usd,
                    "total_tokens": s_tok,
                    "total_wall_sec": s_wall,
                }
            )
            f_pass = int(a_idx < (0 if is_e2e else 2))
            f_rew = 0.20 if is_e2e else 0.60
            f_usd = 1.80 if is_e2e else 0.40
            f_tok = 180_000 if is_e2e else 40_000
            f_wall = 120.0 if is_e2e else 20.0
            records.append(
                {
                    "trial_name": f"{btask}__00_none__sonnet_{a_idx}",
                    "base_task": btask,
                    "suite": "r3c391",
                    "stage": stg,
                    "is_e2e": is_e2e,
                    "arm": "00_none",
                    "model_name": "anthropic/claude-sonnet-4-6",
                    "reward": f_rew,
                    "passed": f_pass,
                    "total_usd": f_usd,
                    "total_tokens": f_tok,
                    "total_wall_sec": f_wall,
                }
            )
            if is_e2e:
                records.append(
                    {
                        "trial_name": f"{btask}__11_orchestrated__haiku_{a_idx}",
                        "base_task": btask,
                        "suite": "r3c391",
                        "stage": stg,
                        "is_e2e": True,
                        "arm": "11_orchestrated",
                        "model_name": "anthropic/claude-haiku-4-5",
                        "reward": 0.75,
                        "passed": 1,
                        "total_usd": 1.10,
                        "total_tokens": 110_000,
                        "total_wall_sec": 100.0,
                    }
                )

    res = analyze_factorial(records, n_boot=400, seed=99)
    cells = res["overall"]["cells"]
    assert cells["11_full"]["e2e_only"] is False
    assert cells["11_orchestrated"]["e2e_only"] is True

    assert math.isclose(cells["11_full"]["reward_task_equal_mean"], 0.70, rel_tol=0.0, abs_tol=1e-7)
    expected_p1_te = (0.8 + 0.8 + (1.0 / 3.0)) / 3.0
    assert math.isclose(
        cells["11_full"]["pass_at_1_task_equal"], expected_p1_te, rel_tol=0.0, abs_tol=1e-7
    )
    expected_usd_te = (0.10 + 0.10 + 0.90) / 3.0
    assert math.isclose(
        cells["11_full"]["mean_usd_task_equal"], expected_usd_te, rel_tol=0.0, abs_tol=1e-7
    )
    assert math.isclose(
        cells["11_full"]["mean_total_tokens_task_equal"],
        (10_000 + 10_000 + 90_000) / 3.0,
        rel_tol=0.0,
        abs_tol=1e-3,
    )
    assert math.isclose(
        cells["11_full"]["mean_wall_sec_task_equal"],
        (10.0 + 10.0 + 90.0) / 3.0,
        rel_tol=0.0,
        abs_tol=1e-4,
    )

    par = res["overall"]["parity"]["anthropic_haiku_vs_sonnet__11_full_vs_00_none"]
    assert math.isclose(par["small_mean_usd"], (0.10 + 0.10 + 0.90) / 3.0, abs_tol=1e-7)
    assert math.isclose(par["large_mean_usd"], (0.40 + 0.40 + 1.80) / 3.0, abs_tol=1e-7)
    assert math.isclose(
        par["cost_ratio"], par["small_mean_usd"] / par["large_mean_usd"], abs_tol=1e-6
    )
    assert isinstance(par["pass_at_1_ci95"], list) and len(par["pass_at_1_ci95"]) == 2

    assert math.isclose(par["small_pass_at_3"], 1.0, abs_tol=1e-7)
    assert math.isclose(par["large_pass_at_3"], 0.6, abs_tol=1e-7)
    assert math.isclose(par["pass_at_3_gap"], 0.4, abs_tol=1e-7)

    assert math.isclose(par["small_pass_at_5"], 1.0, abs_tol=1e-7)
    assert math.isclose(par["large_pass_at_5"], 1.0, abs_tol=1e-7)
    assert math.isclose(par["pass_at_5_gap"], 0.0, abs_tol=1e-7)

    assert "parity" in res["by_stage_group"]["stage"]
    assert "parity" in res["by_stage_group"]["e2e"]
    assert (
        res["by_stage_group"]["e2e"]["parity"][
            "anthropic_haiku_vs_sonnet__11_orchestrated_vs_00_none"
        ]["status"]
        == "ok"
    )
    assert "parity" in res["by_task"]["s3_cal"]
    assert "parity" in res["by_task"]["r3c391_e2e"]


def test_wave5_opencode_log_recovery_truncation_regrade_and_csv_sanitization(
    tmp_path: Path,
) -> None:
    """Verify opencode.txt telemetry recovery, 32-char arm truncation, regrade dedup, and CSV measured_*."""
    import numpy as np

    from radio_bench.analysis.cli import run_analysis_pipeline
    from radio_bench.analysis.collect import (
        _classify_mcp_server,
        collect_jobs,
        extract_trajectory_telemetry,
        parse_task_variant,
        sanitize_json_tree,
        write_trials_outputs,
    )

    assert parse_task_variant("r3c391_s3_calibrate__01_skills_ca__a1b2c3d")[1] == "01_skills_casa"
    assert parse_task_variant("vla_continuum_e2e__01_skills_ra__9f8e7d6")[1] == "01_skills_raw"
    assert parse_task_variant("r3c391_continuum_e2e__11_orchestra__1234567")[1] == "11_orchestrated"

    assert _classify_mcp_server("fooms_setjy") is None
    assert _classify_mcp_server("radio-analyst_ms-modify__ms_setjy") == "ms-modify"
    assert _classify_mcp_server("mcp_ms-inspect_ms_observation_info") == "ms-inspect"

    oc_log = tmp_path / "opencode.txt"
    oc_events = [
        {
            "type": "tool_use",
            "part": {
                "callID": "call_gain_err",
                "tool": "radio-analyst_ms-modify_ms_gaincal",
                "state": {"status": "error", "error": "Gaincal failed: table locked"},
            },
        },
        {
            "type": "tool_use",
            "part": {
                "callID": "call_proto_err",
                "tool": "invalid",
                "state": {"status": "error", "error": "unavailable tool ms_nonexistent"},
            },
        },
        {
            "type": "tool_use",
            "part": {
                "callID": "call_bash_2",
                "tool": "bash",
                "state": {"status": "completed", "metadata": {"exit": 2}},
            },
        },
    ]
    oc_log.write_text("\n".join(json.dumps(e) for e in oc_events) + "\n", encoding="utf-8")

    traj_oc = Trajectory(
        session_id="sess-oc-err",
        agent=Agent(name="opencode", version="1.0", model_name="nrp/kimi"),
        steps=[
            Step(step_id=1, source="user", message="Calibrate."),
            Step(
                step_id=2,
                source="agent",
                message="Calling tools.",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_gain_err",
                        function_name="radio-analyst_ms-modify_ms_gaincal",
                        arguments={},
                    ),
                    ToolCall(
                        tool_call_id="call_proto_err",
                        function_name="invalid",
                        arguments={},
                    ),
                    ToolCall(
                        tool_call_id="call_bash_2",
                        function_name="bash",
                        arguments={"command": "false"},
                    ),
                ],
                observation=Observation(
                    results=[ObservationResult(source_call_id=None, content="unrelated ok")]
                ),
            ),
            Step(step_id=3, source="agent", message="Done."),
        ],
    )
    telem = extract_trajectory_telemetry(
        traj_oc.model_dump(mode="json"),
        arm="11_full",
        opencode_txt_path=oc_log,
    )
    assert telem["tool_errors"] == 2
    assert telem["protocol_errors"] == 1
    assert telem["bash_nonzero_exits"] == 1
    assert telem["mcp_calls_modify"] == 1

    telem_no_log = extract_trajectory_telemetry(
        traj_oc.model_dump(mode="json"),
        arm="11_full",
        opencode_txt_path=None,
    )
    assert telem_no_log["tool_errors"] == 3

    job1 = tmp_path / "job_orig"
    job2 = tmp_path / "job_regraded"
    orig_dir, orig_tr = _write_harbor_trial(
        job1,
        task_name="s1_inspect__11_full",
        agent_name="opencode",
        model_name="nrp/kimi",
        reward_dict={"reward": 0.25, "passed": 0},
        metrics_dict={
            "stage": "inspect",
            "truth": {"flux_calibrator": "3C286"},
            "measured": {"flux_calibrator": "wrong", "bandpass_min_elevation_deg": 41.2},
            "subscores": {"flux_bandpass_score": 0.0},
            "errors": [],
        },
    )
    regrade_dir, _ = _write_harbor_trial(
        job2,
        task_name="s1_inspect__11_full",
        agent_name="opencode",
        model_name="nrp/kimi",
        reward_dict={"reward": 0.92, "passed": 1},
        metrics_dict={
            "stage": "inspect",
            "truth": {"flux_calibrator": "3C286"},
            "measured": {
                "flux_calibrator": "3C286",
                "bandpass_min_elevation_deg": 42.5,
                "note": "clean" + chr(0x2014) + "pass" + chr(0x2013) + "ok",
            },
            "subscores": {"flux_bandpass_score": 1.0},
            "errors": [],
        },
        started_at=datetime(2026, 11, 14, 14, 0, 0, tzinfo=UTC),
    )
    rcfg_path = regrade_dir / "config.json"
    rcfg = json.loads(rcfg_path.read_text(encoding="utf-8"))
    rcfg["source_trial"] = {
        "action": "regrade",
        "path": str(orig_dir),
        "trial_id": str(orig_tr.id),
    }
    rcfg_path.write_text(json.dumps(rcfg, indent=2), encoding="utf-8")

    deduped = collect_jobs([job1, job2])
    assert len(deduped) == 1
    assert math.isclose(deduped[0]["reward"], 0.92, abs_tol=1e-9)

    e2e_rec = dict(deduped[0])
    e2e_rec["trial_name"] = "e2e_trial_0"
    e2e_rec["stage"] = "e2e"
    e2e_rec["is_e2e"] = np.bool_(True)
    e2e_rec["n_agent_turns"] = np.int64(7)
    e2e_rec["reward"] = np.float64(0.88)
    e2e_rec["measured"] = {
        "calibration": {"flux_error_pct": np.float64(3.2)},
        "image": {
            "off_source_residual_rms": np.float64(4.5e-5),
            "arr": np.array([1.0, float("nan")]),
        },
    }
    clean_tree = sanitize_json_tree(e2e_rec)
    assert clean_tree["is_e2e"] is True
    assert clean_tree["measured"]["image"]["arr"] == [1.0, None]

    csv_out_dir = tmp_path / "csv_check"
    csv_p, jsonl_p = write_trials_outputs([deduped[0], e2e_rec], csv_out_dir)
    csv_txt = csv_p.read_text(encoding="utf-8")
    jsonl_txt = jsonl_p.read_text(encoding="utf-8")
    assert chr(0x2014) not in csv_txt and chr(0x2013) not in csv_txt
    assert chr(0x2014) not in jsonl_txt and chr(0x2013) not in jsonl_txt

    with csv_p.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "measured_bandpass_min_elevation_deg" in rows[0]
    assert rows[0]["measured_flux_calibrator"] == "3C286"
    assert rows[0]["measured_note"] == "clean--pass--ok"
    assert rows[1]["measured_cal_flux_error_pct"] == "3.2"
    assert rows[1]["measured_img_off_source_residual_rms"] == "4.5e-05"

    single_res = run_analysis_pipeline([regrade_dir], plots=False, n_boot=50)
    assert Path(single_res["output_dir"]) == regrade_dir.parent / "analysis"
    assert (regrade_dir.parent / "analysis" / "summary.json").is_file()


def test_wave5_taxonomy_diagnostics_and_scientific_overlays(tmp_path: Path) -> None:
    """Verify taxonomy e2e/image/inspect rules, diagnostics Truth-vs-Measured & Markdown, and CASA overlays."""
    import os

    import numpy as np

    from radio_bench.analysis.diagnostics import (
        build_truth_vs_measured_table,
        write_diagnostic_reports,
    )

    f_e2e_flux = classify_failure(
        passed=0,
        stage="e2e",
        arm="11_full",
        reward=0.15,
        reward_dict={
            "reward": 0.15,
            "passed": 0,
            "calibration_score": 0.30,
            "delivered_image_score": 0.0,
            "cal_fluxscale_score": 0.10,
            "cal_bandpass_flatness_score": 0.90,
            "cal_retention_and_dead_ant_score": 0.90,
            "img_products_score": 0.0,
        },
        metrics_dict={
            "stage": "e2e",
            "measured": {
                "calibration": {"has_corrected_data": True, "corrected_distinct_from_data": True},
                "image": {"chosen_prefix": None, "image_sets_found": 0},
            },
        },
    )
    assert f_e2e_flux == "flux_model_or_fluxscale_error"

    f_img_beam = classify_failure(
        passed=0,
        stage="image",
        arm="00_none",
        reward=0.25,
        reward_dict={
            "reward": 0.25,
            "passed": 0,
            "products_score": 0.0,
            "beam_sampling_score": 0.20,
            "rms_score": 0.80,
        },
        metrics_dict={
            "stage": "image",
            "truth": {"ppb_min": 3.0, "ppb_max": 7.0, "bowl_max": 0.05},
            "measured": {
                "chosen_prefix": "clean_img",
                "image_sets_found": 1,
                "pixels_per_beam": 1.6,
                "negative_bowl_ratio": 0.08,
            },
        },
    )
    assert f_img_beam == "beam_sampling_or_clean_bowl"

    assert (
        classify_failure(
            passed=0,
            stage="inspect",
            reward=0.40,
            reward_dict={
                "flux_bandpass_score": 0.0,
                "elevation_score": 0.0,
                "dead_antenna_score": 0.0,
            },
        )
        == "flux_model_or_fluxscale_error"
    )
    assert (
        classify_failure(
            passed=0,
            stage="inspect",
            reward=0.50,
            reward_dict={
                "flux_bandpass_score": 1.0,
                "elevation_score": 0.0,
                "dead_antenna_score": 0.0,
            },
        )
        == "skipped_delay_or_bandpass"
    )

    assert (
        classify_failure(
            passed=0,
            stage="e2e",
            reward=0.0,
            metrics_dict={
                "measured": {"calibration": {"has_corrected_data": False}},
                "calibration": {"errors": ["RuntimeError: casacore write lock on table.lock"]},
            },
        )
        == "lock_or_timeout"
    )
    assert (
        classify_failure(
            passed=0,
            stage="calibrate",
            reward=0.0,
            exception_type="RuntimeError",
            exception_message="libshmcp shared buffer allocation failed",
        )
        == "script_or_casa_crash"
    )
    assert (
        classify_failure(
            passed=0,
            stage="calibrate",
            reward=0.0,
            exception_type="RuntimeError",
            exception_message="mcp server connection closed unexpectedly",
        )
        == "tool_protocol_error"
    )

    assert (
        classify_failure(
            passed=0,
            stage="e2e",
            reward=0.58,
            reward_dict={
                "calibration_score": 0.40,
                "delivered_image_score": 0.70,
                "cal_fluxscale_score": 0.55,
                "cal_bandpass_flatness_score": 0.90,
                "cal_retention_and_dead_ant_score": 0.90,
                "img_rms_score": 0.80,
            },
            metrics_dict={
                "measured": {
                    "calibration": {
                        "has_corrected_data": True,
                        "corrected_distinct_from_data": True,
                    },
                    "image": {"chosen_prefix": "img0", "image_sets_found": 1},
                }
            },
        )
        == "flux_model_or_fluxscale_error"
    )

    cfg_file = Path("/mnt/fast/radio-bench/casaconfig_local.py")
    if cfg_file.is_file():
        os.environ.setdefault("CASASITECONFIG", str(cfg_file))
    import casatools

    ws_trial = tmp_path / "overlay_trial"
    ws_dir = ws_trial / "artifacts" / "workspace"
    ws_dir.mkdir(parents=True)

    cal_path = ws_dir / "bandpass.bcal"
    tb = casatools.table()
    col_desc = {
        "ANTENNA1": {
            "valueType": "int",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "",
        },
        "CPARAM": {
            "valueType": "complex",
            "ndim": 2,
            "shape": np.array([2, 8], dtype=np.int32),
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "",
        },
        "FLAG": {
            "valueType": "boolean",
            "ndim": 2,
            "shape": np.array([2, 8], dtype=np.int32),
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "",
        },
    }
    assert tb.create(str(cal_path), col_desc)
    tb.addrows(4)
    tb.putcol("ANTENNA1", np.array([0, 1, 2, 3], dtype=np.int32))
    tb.putcol("CPARAM", np.ones((2, 8, 4), dtype=np.complex128) * (1.02 + 0.15j))
    tb.putcol("FLAG", np.zeros((2, 8, 4), dtype=bool))
    tb.close()

    ia = casatools.image()
    rng = np.random.default_rng(42)
    ia.fromarray(
        outfile=str(ws_dir / "clean_img.residual"),
        pixels=rng.normal(0.0, 3.0e-5, size=(32, 32, 1, 1)),
        overwrite=True,
    )
    ia.close()
    ia.fromarray(
        outfile=str(ws_dir / "clean_img.pb"),
        pixels=np.ones((32, 32, 1, 1), dtype=np.float64) * 0.9,
        overwrite=True,
    )
    ia.close()

    diag_records: list[dict[str, Any]] = [
        {
            "trial_name": "insp_0",
            "trial_dir": str(tmp_path / "no_ws"),
            "base_task": "s1_inspect|pipe",
            "suite": "vla",
            "stage": "inspect",
            "is_e2e": False,
            "arm": "11_full",
            "model_name": "anthropic/claude-haiku-4-5",
            "reward": 0.95,
            "passed": 1,
            "total_usd": 0.05,
            "total_tokens": 12_000,
            "total_wall_sec": 10.0,
            "failure_mode": FAILURE_NONE,
            "truth": {
                "flux_calibrator": "J1331+3030",
                "bandpass_calibrator": "J1331+3030",
                "target_fields": ["3C391"],
            },
            "measured": {
                "flux_calibrator": ["J1331+3030"],
                "bandpass_calibrator": ["J1331+3030"],
                "target_fields": ["3C391"],
            },
        },
        {
            "trial_name": "e2e_0",
            "trial_dir": str(ws_trial),
            "base_task": "r3c391_e2e",
            "suite": "r3c391",
            "stage": "e2e",
            "is_e2e": True,
            "arm": "11_orchestrated",
            "model_name": "anthropic/claude-haiku-4-5",
            "reward": 0.88,
            "passed": 1,
            "total_usd": 0.45,
            "total_tokens": 55_000,
            "total_wall_sec": 75.0,
            "failure_mode": FAILURE_NONE,
            "truth": {
                "calibration": {
                    "dead_antenna_flagged_min": 0.99,
                    "max_coh_phs_deg": 10.0,
                },
                "image": {
                    "ppb_min": 3.5,
                    "ppb_max": 6.5,
                    "bowl_max": 0.08,
                },
            },
            "measured": {
                "calibration": {
                    "dead_antenna_flagged_fraction": 1.0,
                    "coherent_phase_deg": 1.2,
                },
                "image": {
                    "pixels_per_beam": 4.8,
                    "negative_bowl_ratio": 0.03,
                    "off_source_residual_rms": 3.0e-5,
                },
            },
        },
        {
            "trial_name": "e2e_sonnet_0",
            "trial_dir": str(tmp_path / "no_ws_2"),
            "base_task": "r3c391_e2e",
            "suite": "r3c391",
            "stage": "e2e",
            "is_e2e": True,
            "arm": "00_none",
            "model_name": "anthropic/claude-sonnet-4-6",
            "reward": 0.40,
            "passed": 0,
            "total_usd": 1.20,
            "total_tokens": 95_000,
            "total_wall_sec": 110.0,
            "failure_mode": "beam_sampling_or_clean_bowl",
            "truth": {
                "calibration": {
                    "dead_antenna_flagged_min": 0.99,
                    "max_coh_phs_deg": 10.0,
                },
                "image": {
                    "ppb_min": 3.5,
                    "ppb_max": 6.5,
                    "bowl_max": 0.08,
                },
            },
            "measured": {
                "calibration": {},
                "image": {"pixels_per_beam": 1.8, "negative_bowl_ratio": 0.19},
            },
        },
    ]

    tvm = build_truth_vs_measured_table(diag_records)
    by_label = {(r["base_task"], r["metric"]): r for r in tvm}
    assert by_label[("s1_inspect|pipe", "flux_calibrator")]["by_arm"]["11_full"] == "J1331+3030"
    assert (
        by_label[("s1_inspect|pipe", "flux_calibrator")]["by_arm"]["11_full"]
        == by_label[("s1_inspect|pipe", "flux_calibrator")]["truth"]
    )
    assert by_label[("s1_inspect|pipe", "bandpass_calibrator")]["by_arm"]["11_full"] == "J1331+3030"
    assert by_label[("s1_inspect|pipe", "target_fields")]["by_arm"]["11_full"] == '["3C391"]'
    assert by_label[("r3c391_e2e", "img.pixels_per_beam")]["truth"] == "[3.5, 6.5]"
    assert by_label[("r3c391_e2e", "img.negative_bowl_ratio")]["truth"] == "< 0.08"
    assert ("r3c391_e2e", "cal.dead_antenna_flagged_fraction") in by_label
    assert ("r3c391_e2e", "cal.coherent_phase_deg") in by_label

    fac = analyze_factorial(diag_records, n_boot=100, seed=11)
    tax = summarize_failure_taxonomy(diag_records)
    out_dir = tmp_path / "diag_out"
    rep = write_diagnostic_reports(
        job_dirs=[tmp_path],
        output_dir=out_dir,
        records=diag_records,
        factorial_summary=fac,
        taxonomy_summary=tax,
        plots=True,
    )
    md_text = rep["markdown"]
    assert chr(0x2014) not in md_text and chr(0x2013) not in md_text
    assert r"s1_inspect\|pipe" in md_text
    assert "(e2e-only)" in md_text
    assert "Pass@3" in md_text and "Pass@5" in md_text

    plot_names = {Path(p).name for p in rep["plot_files"]}
    assert plot_names == {
        "reward_by_arm.png",
        "cost_vs_reward.png",
        "failure_taxonomy_by_arm.png",
        "caltable_amp_phase_overlay.png",
        "offsource_residual_hist.png",
    }
    assert list((out_dir / "plots").glob("*.tmp")) == []

    no_ws_records = [dict(r, trial_dir=str(tmp_path / "missing_ws")) for r in diag_records]
    out_dir_fb = tmp_path / "diag_out_fb"
    rep_fb = write_diagnostic_reports(
        job_dirs=[tmp_path],
        output_dir=out_dir_fb,
        records=no_ws_records,
        factorial_summary=fac,
        taxonomy_summary=tax,
        plots=True,
    )
    fb_names = {Path(p).name for p in rep_fb["plot_files"]}
    assert "offsource_residual_hist.png" in fb_names
    assert "caltable_amp_phase_overlay.png" not in fb_names


def test_wave6_pre_oracle_diagnostics_and_cli_no_runpy_warning() -> None:
    """Verify lazy CLI import (zero runpy warning) and Suite S vs Suite R calibrate truth specs."""
    import subprocess
    import sys

    from radio_bench.analysis.diagnostics import _extract_stage_metric_specs

    proc = subprocess.run(
        [sys.executable, "-W", "error", "-m", "radio_bench.analysis.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"cli --help failed with -W error: {proc.stderr}"

    specs_s = dict(
        (lbl, tgt)
        for lbl, tgt, _path in _extract_stage_metric_specs(
            "calibrate",
            {
                "retention_min": 0.85,
                "retention_max": None,
                "verifier_tclean_dr_min": None,
                "max_coh_amp_err": 0.08,
                "max_coh_phs_deg": 10.0,
            },
        )
    )
    assert specs_s["target_row_retention"] == ">= 0.85"
    assert "verifier_tclean_dynamic_range" not in specs_s
    assert specs_s["coherent_amp_rel_err"] == "<= 0.080"
    assert specs_s["coherent_phase_deg"] == "<= 10.00 deg"

    specs_r = dict(
        (lbl, tgt)
        for lbl, tgt, _path in _extract_stage_metric_specs(
            "calibrate",
            {
                "retention_min": 0.70,
                "retention_max": 0.90,
                "verifier_tclean_dr_min": 180.0,
                "max_coh_amp_err": None,
                "max_coh_phs_deg": None,
            },
        )
    )
    assert specs_r["target_row_retention"] == "[0.70, 0.90]"
    assert specs_r["verifier_tclean_dynamic_range"] == ">= 180.0"
    assert "coherent_amp_rel_err" not in specs_r
    assert "coherent_phase_deg" not in specs_r
