"""Isolation, arm-purity, anti-contamination, and Harbor schema tests for Wave 4."""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.opencode import OpenCode
from harbor.lint.models import COMPOSE_RULES, DEFAULT_RULES, LintRule
from harbor.lint.task import lint_task
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from radio_bench.build_tasks import (
    AGENT_IMAGE,
    AGENT_NETWORK_MODE,
    ARM_FULL,
    ARM_NONE,
    ARM_ORCHESTRATED,
    ARM_SKILLS_CASA,
    ARM_SKILLS_RAW,
    ARM_TOOLS,
    DEFAULT_SPECS_DIR,
    DEFAULT_TASKS_DIR,
    SKILL_ARMS,
    TOOL_ARMS,
    VERIFIER_IMAGE,
    VERIFIER_NETWORK_MODE,
    arms_for_spec,
    build_all_tasks,
    load_task_spec,
)
from radio_bench.config import load_bench_config
from radio_bench.export_skills import (
    CASA_FORBIDDEN_SUBSTRINGS,
    DEFAULT_EXPORT_DIR,
    PYTHON_FENCE_RE,
    RADIO_ANALYST_SHA,
    RAW_TRANSLATION_NOTE,
)
from radio_bench.podman_env import ensure_docker_compose_plugin
from radio_bench.render_jobs import render_preset_job

BENCH_CFG = load_bench_config()
REPO_ROOT = BENCH_CFG.repo_root

STAGE_TASKS = (
    "s1_inspect",
    "s1_precal",
    "s1_calibrate",
    "s1_image",
    "r1_3c391_inspect",
    "r1_3c391_precal",
    "r1_3c391_calibrate",
    "r1_3c391_image",
)
E2E_TASKS = (
    "s2_e2e_synth",
    "r2_3c391_e2e",
)
CALIBRATE_TASKS = frozenset({"s1_calibrate", "r1_3c391_calibrate"})
ALL_BASE_TASKS = STAGE_TASKS + E2E_TASKS
FORBIDDEN_AGENT_FILENAMES = frozenset({"ground_truth.json", "verify.py"})
MS_TOOL_TOKEN_RE = re.compile(r"\bms_[a-z0-9_]+\b")
EM_OR_EN_DASH_RE = re.compile(r"[\u2013\u2014]")

FORBIDDEN_INSTRUCTION_TOKENS: tuple[str, ...] = (
    "G0",
    "B0",
    "solnorm",
    "Perley-Butler",
    "negativethreshold",
    "auto-multithresh",
    "multiscale",
    "robust=0.5",
    "bowl_ratio",
    "0.75 mJy",
    "mcp__plugin_radio-analyst_",
    "ea13",
    "ea07",
    "3C391",
    "J1822",
    "J0319+4130",
    "3C84",
    "ground_truth",
    "reward.json",
)


def _tasks_root() -> Path:
    override = os.environ.get("RADIO_BENCH_TASKS_DIR", "").strip()
    return Path(override) if override else DEFAULT_TASKS_DIR


def _compiled_task_dirs() -> list[Path]:
    return sorted(p for p in _tasks_root().iterdir() if p.is_dir())


def test_compiled_task_matrix_counts() -> None:
    """Verify exact 52 compiled Harbor task variants (8 * 5 + 2 * 6) with 00_none."""
    task_dirs = _compiled_task_dirs()
    assert len(task_dirs) == 52
    assert all("00_vanilla" not in d.name for d in task_dirs)

    expected: set[str] = set()
    for base in STAGE_TASKS:
        for arm in (ARM_NONE, ARM_SKILLS_RAW, ARM_SKILLS_CASA, ARM_TOOLS, ARM_FULL):
            expected.add(f"{base}__{arm}")
    for base in E2E_TASKS:
        for arm in (
            ARM_NONE,
            ARM_SKILLS_RAW,
            ARM_SKILLS_CASA,
            ARM_TOOLS,
            ARM_FULL,
            ARM_ORCHESTRATED,
        ):
            expected.add(f"{base}__{arm}")

    actual = {d.name for d in task_dirs}
    assert actual == expected


def test_arms_for_spec_modes() -> None:
    """Verify configurable --skills-mode and --include-orchestrated arm selection."""
    stage_spec = load_task_spec(DEFAULT_SPECS_DIR / "s1_inspect")
    e2e_spec = load_task_spec(DEFAULT_SPECS_DIR / "s2_e2e_synth")

    assert arms_for_spec(stage_spec, skills_mode="raw", include_orchestrated=True) == [
        ARM_NONE,
        ARM_SKILLS_RAW,
        ARM_TOOLS,
        ARM_FULL,
    ]
    assert arms_for_spec(stage_spec, skills_mode="casa", include_orchestrated=True) == [
        ARM_NONE,
        ARM_SKILLS_CASA,
        ARM_TOOLS,
        ARM_FULL,
    ]
    assert arms_for_spec(e2e_spec, skills_mode="both", include_orchestrated=False) == [
        ARM_NONE,
        ARM_SKILLS_RAW,
        ARM_SKILLS_CASA,
        ARM_TOOLS,
        ARM_FULL,
    ]
    assert arms_for_spec(e2e_spec, skills_mode="both", include_orchestrated=True) == [
        ARM_NONE,
        ARM_SKILLS_RAW,
        ARM_SKILLS_CASA,
        ARM_TOOLS,
        ARM_FULL,
        ARM_ORCHESTRATED,
    ]


def test_build_tasks_prunes_unselected_arms_on_scratch(tmp_path: Path) -> None:
    """Verify build_all_tasks cleanly emits and prunes subsets in a scratch output dir."""
    specs_subset = tmp_path / "specs_subset"
    specs_subset.mkdir()
    for tid in ("s1_inspect", "s2_e2e_synth"):
        os.symlink(DEFAULT_SPECS_DIR / tid, specs_subset / tid)

    out_dir = tmp_path / "tasks_out"
    stale_vanilla = out_dir / "s1_inspect__00_vanilla"
    stale_vanilla.mkdir(parents=True)

    res_all = build_all_tasks(
        specs_dir=specs_subset,
        skills_dir=DEFAULT_EXPORT_DIR,
        output_dir=out_dir,
        skills_mode="both",
        include_orchestrated=True,
    )
    assert res_all["compiled_tasks"] == 11
    assert not stale_vanilla.exists()

    res_casa_only = build_all_tasks(
        specs_dir=specs_subset,
        skills_dir=DEFAULT_EXPORT_DIR,
        output_dir=out_dir,
        skills_mode="casa",
        include_orchestrated=False,
    )
    assert res_casa_only["compiled_tasks"] == 8
    remaining = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert len(remaining) == 8
    assert all(
        "__01_skills_raw" not in name and "__11_orchestrated" not in name for name in remaining
    )


def test_arm_skill_and_mcp_isolation() -> None:
    """Verify orthogonal separation of skills, MCP sidecars, TMPDIR, OpenMP, and native volumes across 52 tasks."""
    specs = {tid: load_task_spec(DEFAULT_SPECS_DIR / tid) for tid in ALL_BASE_TASKS}
    for task_dir in _compiled_task_dirs():
        base_task, arm = task_dir.name.split("__", 1)
        spec = specs[base_task]
        expected_omp = str(spec.cpus)
        expected_ver_omp = str(spec.effective_verifier_cpus)

        raw_toml = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        cfg = TaskConfig.model_validate(raw_toml)
        env_compose = yaml.safe_load(
            (task_dir / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
        )
        ver_compose = yaml.safe_load(
            (task_dir / "tests" / "docker-compose.yaml").read_text(encoding="utf-8")
        )
        services = env_compose["services"]
        main_volumes = services["main"]["volumes"]
        main_env = services["main"]["environment"]
        main_cmd = " ".join(services["main"]["command"])

        assert raw_toml["artifacts"] == ["/workspace"]
        assert cfg.agent.user == "analyst"
        assert cfg.environment.docker_image == AGENT_IMAGE
        assert cfg.environment.cpus == spec.cpus
        assert cfg.verifier.environment_mode == "separate"
        assert cfg.verifier.environment is not None
        assert cfg.verifier.environment.docker_image == VERIFIER_IMAGE
        assert cfg.verifier.environment.cpus == spec.effective_verifier_cpus
        assert cfg.verifier.environment.healthcheck is None

        for env_map, omp_val in (
            (cfg.environment.env, expected_omp),
            (cfg.verifier.env, expected_ver_omp),
            (cfg.verifier.environment.env, expected_ver_omp),
            (main_env, expected_omp),
            (ver_compose["services"]["main"]["environment"], expected_ver_omp),
        ):
            assert env_map.get("TMPDIR") == "/var/tmp"
            assert env_map.get("OMP_NUM_THREADS") == omp_val
            assert env_map.get("OMP_WAIT_POLICY") == "PASSIVE"
            assert env_map.get("OPENBLAS_NUM_THREADS") == "1"
            assert env_map.get("MKL_NUM_THREADS") == "1"
            assert env_map.get("NUMEXPR_NUM_THREADS") == "1"

        assert "workspace_vol:/workspace" in main_volumes
        assert "agent_tmp:/var/tmp" in main_volumes
        assert "workspace_vol" in env_compose["volumes"]
        assert "agent_tmp" in env_compose["volumes"]
        assert "chmod 1777 /var/tmp" in main_cmd

        assert AGENT_NETWORK_MODE == "pasta"
        assert VERIFIER_NETWORK_MODE == "none"
        assert env_compose["services"]["main"]["network_mode"] == "pasta"
        assert "networks" not in env_compose
        assert ver_compose["services"]["main"]["network_mode"] == "none"
        assert "networks" not in ver_compose

        ver_volumes = ver_compose["services"]["main"]["volumes"]
        assert "verifier_ws:/workspace" in ver_volumes
        assert "verifier_tmp:/var/tmp" in ver_volumes
        assert "verifier_ws" in ver_compose["volumes"]
        assert "verifier_tmp" in ver_compose["volumes"]

        assert raw_toml["metadata"]["base_task"] == base_task
        assert raw_toml["metadata"]["arm"] == arm
        assert raw_toml["metadata"]["radio_analyst_sha"] == RADIO_ANALYST_SHA

        skills_path = task_dir / "environment" / "skills"
        seed_ws_path = task_dir / "environment" / "seed_workspace"

        if arm in SKILL_ARMS:
            assert skills_path.is_dir(), f"{task_dir.name} missing environment/skills"
            assert seed_ws_path.is_dir(), f"{task_dir.name} missing environment/seed_workspace"
            assert cfg.environment.skills_dir == "/skills"
            assert any(v.startswith("./skills:/skills:ro") for v in main_volumes)

            # Simulator skills must never appear in any arm; orchestrate ONLY in 11_orchestrated.
            assert not (skills_path / "ms-simulator").exists()
            assert not (skills_path / "simulate").exists()
            if arm == ARM_ORCHESTRATED:
                assert (skills_path / "orchestrate" / "SKILL.md").is_file()
            else:
                assert not (skills_path / "orchestrate").exists(), (
                    f"Single-agent arm {task_dir.name} leaked orchestrate skill"
                )
        else:
            assert arm in {ARM_NONE, ARM_TOOLS}
            assert not skills_path.exists(), f"{task_dir.name} leaked environment/skills"
            assert not seed_ws_path.exists(), f"{task_dir.name} leaked seed_workspace"
            assert cfg.environment.skills_dir is None
            assert all("/skills" not in v for v in main_volumes)

        if arm in TOOL_ARMS:
            assert "mcp-servers" in services, f"{task_dir.name} missing mcp-servers service"
            assert "mcp-servers:127.0.0.1" in services["main"]["extra_hosts"]
            assert services["mcp-servers"]["network_mode"] == "service:main"
            sidecar_env = services["mcp-servers"]["environment"]
            assert sidecar_env["RADIO_MCP_HOST"] == "127.0.0.1"
            assert sidecar_env.get("OMP_NUM_THREADS") == expected_omp
            assert sidecar_env.get("OMP_WAIT_POLICY") == "PASSIVE"
            assert sidecar_env.get("OPENBLAS_NUM_THREADS") == "1"
            assert sidecar_env.get("MKL_NUM_THREADS") == "1"
            assert sidecar_env.get("NUMEXPR_NUM_THREADS") == "1"
            assert "agent_tmp:/var/tmp" in services["mcp-servers"]["volumes"]
            mcp_names = [s.name for s in cfg.environment.mcp_servers]
            assert mcp_names == ["ms-inspect", "ms-modify", "ms-create"]
            assert all(s.transport == "streamable-http" for s in cfg.environment.mcp_servers)
        else:
            assert arm in {ARM_NONE, ARM_SKILLS_RAW, ARM_SKILLS_CASA}
            assert "mcp-servers" not in services, f"{task_dir.name} leaked mcp-servers service"
            assert "extra_hosts" not in services["main"]
            assert cfg.environment.mcp_servers == []


def test_casa_vs_raw_skill_token_purity_and_anti_leakage() -> None:
    """Verify 01_skills_casa has zero ms_*, zero truth/eval leaks, valid Python blocks, and clean MCP prefixes."""
    for task_dir in _compiled_task_dirs():
        _, arm = task_dir.name.split("__", 1)
        if arm == ARM_SKILLS_CASA:
            files_to_scan = [task_dir / "instruction.md"] + sorted(
                p
                for sub in ("skills", "seed_workspace")
                for p in (task_dir / "environment" / sub).rglob("*")
                if p.is_file()
            )
            assert len(files_to_scan) >= 25
            combined_casa = []
            parsed_blocks = 0
            for fpath in files_to_scan:
                text = fpath.read_text(encoding="utf-8")
                matches = MS_TOOL_TOKEN_RE.findall(text)
                assert matches == [], f"Forbidden ms_* tokens {matches} in {fpath}"
                assert EM_OR_EN_DASH_RE.search(text) is None, f"Em/en dash found in {fpath}"
                lower_text = text.lower()
                for bad_sub in CASA_FORBIDDEN_SUBSTRINGS:
                    assert bad_sub not in lower_text, (
                        f"Forbidden leaked substring {bad_sub!r} in {fpath}"
                    )
                if fpath.suffix == ".md":
                    for match in PYTHON_FENCE_RE.finditer(text):
                        ast.parse(match.group(1))
                        parsed_blocks += 1
                combined_casa.append(text)

            assert parsed_blocks >= 15
            joined = "\n".join(combined_casa)
            for term in (
                "Perley-Butler 2017",
                "fluxscale",
                "multiscale",
                "flatnoise",
                "pbcor",
                "negativethreshold",
                "solnorm=False",
            ):
                assert term in joined, f"Missing scientific term {term!r} in {task_dir.name}"

        elif arm in {ARM_SKILLS_RAW, ARM_FULL, ARM_ORCHESTRATED}:
            skill_files = sorted(
                p for p in (task_dir / "environment" / "skills").rglob("*.md") if p.is_file()
            )
            raw_tokens: set[str] = set()
            for p in skill_files:
                txt = p.read_text(encoding="utf-8")
                assert "mcp__plugin_radio-analyst_" not in txt, (
                    f"Stale mcp__plugin_radio-analyst_ prefix in {p}"
                )
                raw_tokens.update(MS_TOOL_TOKEN_RE.findall(txt))

            assert "ms_observation_info" in raw_tokens
            assert "ms_bandpass" in raw_tokens
            assert "ms_tclean" in raw_tokens

            if arm == ARM_SKILLS_RAW:
                for guide_name in ("CLAUDE.md", "AGENTS.md"):
                    guide_text = (
                        task_dir / "environment" / "seed_workspace" / guide_name
                    ).read_text(encoding="utf-8")
                    assert RAW_TRANSLATION_NOTE in guide_text


def test_orchestrated_arm_scoping_and_subagents() -> None:
    """Verify 11_orchestrated exists only on e2e tasks and stages Claude + OpenCode agents."""
    orchestrated_dirs = [
        d for d in _compiled_task_dirs() if d.name.endswith(f"__{ARM_ORCHESTRATED}")
    ]
    assert [d.name for d in orchestrated_dirs] == [
        f"r2_3c391_e2e__{ARM_ORCHESTRATED}",
        f"s2_e2e_synth__{ARM_ORCHESTRATED}",
    ]

    expected_stages = {
        "inspect-stage.md",
        "precal-stage.md",
        "calibrate-stage.md",
        "postcal-stage.md",
        "selfcal-stage.md",
        "image-stage.md",
    }
    for d in orchestrated_dirs:
        seed_ws = d / "environment" / "seed_workspace"
        claude_agents = {p.name for p in (seed_ws / ".claude" / "agents").glob("*.md")}
        opencode_agents = {p.name for p in (seed_ws / ".opencode" / "agents").glob("*.md")}
        claude_cmds = {p.name for p in (seed_ws / ".claude" / "commands").glob("*.md")}
        assert expected_stages.issubset(claude_agents)
        assert expected_stages.issubset(opencode_agents)
        assert {"inspect.md", "precal.md", "calibrate.md", "image.md"}.issubset(claude_cmds)

        env_compose = yaml.safe_load(
            (d / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
        )
        main_cmd = " ".join(env_compose["services"]["main"]["command"])
        assert "/home/analyst/.claude" in main_cmd

        orch_skill = (d / "environment" / "skills" / "orchestrate" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert "mcp__plugin_radio-analyst_" not in orch_skill
        assert "radio-analyst:inspect-stage" not in orch_skill

        for agent_md in list((seed_ws / ".claude" / "agents").glob("*.md")) + list(
            (seed_ws / ".opencode" / "agents").glob("*.md")
        ):
            txt = agent_md.read_text(encoding="utf-8")
            assert "mcp__plugin_radio-analyst_" not in txt
            assert EM_OR_EN_DASH_RE.search(txt) is None

    for d in _compiled_task_dirs():
        if not d.name.endswith(f"__{ARM_ORCHESTRATED}"):
            seed_ws = d / "environment" / "seed_workspace"
            assert not (seed_ws / ".claude" / "agents").exists()
            assert not (seed_ws / ".opencode" / "agents").exists()


def test_zero_verifier_leakage_permissions_symlinks_and_calibrators_ms() -> None:
    """Verify zero test/truth leakage, 0o755 scripts, no escaping links, and calibrators.ms staging."""
    specs = {tid: load_task_spec(DEFAULT_SPECS_DIR / tid) for tid in ALL_BASE_TASKS}

    for task_dir in _compiled_task_dirs():
        task_root = task_dir.resolve()
        base_task, _ = task_dir.name.split("__", 1)
        spec = specs[base_task]

        # 1. No symlinks escaping task_root anywhere in task_dir
        for root_str, dirnames, filenames in os.walk(task_dir, followlinks=False):
            cur = Path(root_str)
            for entry_name in [*dirnames, *filenames]:
                entry = cur / entry_name
                if entry.is_symlink():
                    resolved = entry.resolve()
                    assert resolved.is_relative_to(task_root), (
                        f"Escaping symlink {entry} -> {resolved}"
                    )

        # 2. Zero ground_truth.json, verify.py, or .npz in environment/ or solution/
        for restricted_sub in ("environment", "solution"):
            sub_dir = task_dir / restricted_sub
            for p in sub_dir.rglob("*"):
                assert p.name not in FORBIDDEN_AGENT_FILENAMES, f"Leaked {p.name} inside {p}"
                assert p.suffix != ".npz", f"Leaked .npz file inside {p}"

        # 3. Verify tests/ contains ground_truth.json, verify.py, and all extra_test_files
        tests_dir = task_dir / "tests"
        assert (tests_dir / "ground_truth.json").is_file()
        assert (tests_dir / "verify.py").is_file()
        for extra in spec.extra_test_files:
            assert (tests_dir / extra).is_file(), f"Missing {extra} in {tests_dir}"

        # 4. Executable 0o755 mode on tests/test.sh and solution/solve.sh
        for script_rel in ("tests/test.sh", "solution/solve.sh"):
            script_path = task_dir / script_rel
            assert script_path.is_file()
            mode = stat.S_IMODE(script_path.stat().st_mode)
            assert mode == 0o755, f"Expected 0o755 on {script_path}, got {oct(mode)}"

        # 5. Hardlink inode identity on input.ms/table.dat and calibrate-stage calibrators.ms
        staged_table_dat = task_dir / "environment" / "data" / "input.ms" / "table.dat"
        source_table_dat = spec.input_ms / "table.dat"
        assert staged_table_dat.stat().st_ino == source_table_dat.stat().st_ino

        staged_cal_ms = task_dir / "environment" / "data" / "calibrators.ms"
        env_compose = yaml.safe_load(
            (task_dir / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
        )
        main_cmd = " ".join(env_compose["services"]["main"]["command"])
        if base_task in CALIBRATE_TASKS:
            assert spec.input_calibrators_ms is not None
            assert staged_cal_ms.is_dir(), f"Missing staged calibrators.ms in {task_dir.name}"
            assert (staged_cal_ms / "table.info").is_file()
            assert (staged_cal_ms / "table.dat").stat().st_ino == (
                spec.input_calibrators_ms / "table.dat"
            ).stat().st_ino
            assert "/seed_data/calibrators.ms /workspace/calibrators.ms" in main_cmd
        else:
            assert spec.input_calibrators_ms is None
            assert not staged_cal_ms.exists()
            assert "/seed_data/calibrators.ms" not in main_cmd


def test_vault_instruction_symlinks_cross_arm_identity_and_no_playbook_leakage() -> None:
    """Verify task_specs/*/instruction.md resolves cleanly, matches all arms, and has zero Skills=1 leaks."""
    for base_task in ALL_BASE_TASKS:
        spec_inst = DEFAULT_SPECS_DIR / base_task / "instruction.md"
        assert spec_inst.exists(), f"Missing instruction.md at {spec_inst}"
        resolved_inst = spec_inst.resolve(strict=True)
        assert resolved_inst.is_file()
        if spec_inst.is_symlink():
            assert resolved_inst != spec_inst

        canonical_text = resolved_inst.read_text(encoding="utf-8")
        assert canonical_text.strip()
        assert EM_OR_EN_DASH_RE.search(canonical_text) is None
        for forbidden in FORBIDDEN_INSTRUCTION_TOKENS:
            assert forbidden not in canonical_text, (
                f"Forbidden leaked procedural/dataset token {forbidden!r} in {base_task}/instruction.md"
            )

        arm_dirs = [d for d in _compiled_task_dirs() if d.name.startswith(f"{base_task}__")]
        assert len(arm_dirs) in {5, 6}
        for d in arm_dirs:
            compiled_inst = d / "instruction.md"
            assert not compiled_inst.is_symlink()
            assert compiled_inst.read_text(encoding="utf-8") == canonical_text


def test_harbor_lint_clean_across_all_52_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Harbor lint rules pass with zero findings on all 52 tasks (even in bare pytest)."""
    cfg_dir = ensure_docker_compose_plugin()
    assert cfg_dir is not None, "docker-compose CLI plugin could not be configured"
    monkeypatch.setenv("DOCKER_CONFIG", str(cfg_dir))
    env_bin = str(Path(sys.executable).resolve().parent)
    monkeypatch.setenv("PATH", f"{env_bin}:{os.environ.get('PATH', '')}")

    non_compose_rules = (DEFAULT_RULES | {LintRule.ROOT_AGENT}) - COMPOSE_RULES
    full_rules = DEFAULT_RULES | {LintRule.ROOT_AGENT}

    for task_dir in _compiled_task_dirs():
        findings = lint_task(task_dir, selected_rules=non_compose_rules)
        assert findings == [], f"Lint findings in {task_dir.name}: {findings}"

    tasks_root = _tasks_root()
    representative = [
        tasks_root / f"s1_inspect__{ARM_NONE}",
        tasks_root / f"s1_inspect__{ARM_SKILLS_RAW}",
        tasks_root / f"s1_inspect__{ARM_SKILLS_CASA}",
        tasks_root / f"s1_inspect__{ARM_TOOLS}",
        tasks_root / f"s1_inspect__{ARM_FULL}",
        tasks_root / f"s1_calibrate__{ARM_TOOLS}",
        tasks_root / f"s2_e2e_synth__{ARM_ORCHESTRATED}",
    ]
    for task_dir in representative:
        findings = lint_task(task_dir, selected_rules=full_rules)
        assert findings == [], f"Compose lint findings in {task_dir.name}: {findings}"


@pytest.mark.parametrize(
    ("filename", "expected_tasks", "expected_attempts", "expected_agents", "max_concurrency"),
    [
        ("oracle_synth.yaml", 5, 1, 1, 4),
        ("oracle_3c391.yaml", 5, 1, 1, 1),
        ("suite_s_ablation.yaml", 20, 5, 4, 4),
        ("suite_s_e2e_ablation.yaml", 6, 3, 4, 4),
        ("suite_r_ablation.yaml", 20, 5, 4, 1),
        ("suite_r_e2e_ablation.yaml", 6, 3, 4, 1),
    ],
)
def test_job_yaml_configs_and_opencode_nrp_provider(
    filename: str,
    expected_tasks: int,
    expected_attempts: int,
    expected_agents: int,
    max_concurrency: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all 6 Harbor job YAML configs, n_attempts split, and OpenCode nrp/* provider config."""
    monkeypatch.setenv("NRP_API_KEY", "dummy-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")

    preset_name = filename.removesuffix(".yaml")
    job_path = render_preset_job(preset_name, output_dir=tmp_path / "rendered_jobs", cfg=BENCH_CFG)
    assert job_path.is_file()
    raw_text = job_path.read_text(encoding="utf-8")
    assert EM_OR_EN_DASH_RE.search(raw_text) is None
    assert "00_vanilla" not in raw_text

    cfg = JobConfig.model_validate(yaml.safe_load(raw_text))
    assert cfg.jobs_dir == BENCH_CFG.runs_dir
    assert cfg.environment.type == EnvironmentType.PODMAN
    assert cfg.environment.delete is True
    assert cfg.n_attempts == expected_attempts
    assert 1 <= cfg.n_concurrent_trials <= max_concurrency
    assert len(cfg.tasks) == expected_tasks
    assert len(cfg.agents) == expected_agents

    for task_entry in cfg.tasks:
        assert task_entry.path is not None
        assert task_entry.path.is_dir(), f"Missing task directory in {filename}: {task_entry.path}"

    if expected_agents == 4:
        pairs = [(a.name, a.model_name) for a in cfg.agents]
        assert pairs == [
            ("claude-code", "anthropic/claude-haiku-4-5"),
            ("claude-code", "anthropic/claude-sonnet-4-6"),
            ("opencode", "nrp/kimi"),
            ("opencode", "nrp/glm-5"),
        ]
        for a in cfg.agents:
            AgentFactory.run_preflight(a)
            assert a.kwargs.get("disable_web_search") is True
            if a.name == "opencode":
                assert "reasoning_effort" not in a.kwargs
                assert "config" not in a.kwargs
                assert "model_api_key_name" not in a.kwargs
                assert a.base_url == "https://ellm.nrp-nautilus.io/v1"
                assert a.model_api_key_name == "NRP_API_KEY"
                assert a.model_api_format == "openai_chat_completions"
                assert a.kwargs.get("version") == "1.19.5"

                created = AgentFactory.create_agent_from_config(
                    a,
                    logs_dir=tmp_path / f"oc_{a.model_name.replace('/', '_')}",
                )
                assert isinstance(created, OpenCode)
                oc = created
                assert oc.version() == "1.19.5"
                assert "dummy-test-key" in oc.model_connection.env.values()
                reg_cmd = oc._build_register_config_command()
                assert reg_cmd is not None
                assert "@ai-sdk/openai-compatible" in reg_cmd
                assert "https://ellm.nrp-nautilus.io/v1" in reg_cmd

                # Extract and parse JSON payload written to ~/.config/opencode/opencode.json
                prefix = "mkdir -p ~/.config/opencode && echo "
                suffix = " > ~/.config/opencode/opencode.json"
                assert reg_cmd.startswith(prefix) and reg_cmd.endswith(suffix)
                import shlex

                tokens = shlex.split(reg_cmd[len("mkdir -p ~/.config/opencode && ") : -len(suffix)])
                assert tokens[0] == "echo"
                oc_cfg = json.loads(tokens[1])
                nrp_prov = oc_cfg["provider"]["nrp"]
                assert nrp_prov["npm"] == "@ai-sdk/openai-compatible"
                assert nrp_prov["options"]["baseURL"] == "https://ellm.nrp-nautilus.io/v1"
                model_short = a.model_name.split("/", 1)[1]
                assert model_short in nrp_prov["models"]
                api_key_ref = nrp_prov["options"]["apiKey"]
                assert api_key_ref.startswith("{env:") and api_key_ref.endswith("}")
                forwarded_var = api_key_ref[len("{env:") : -1]
                assert oc.model_connection.env.get(forwarded_var) == "dummy-test-key"


def test_openmp_anti_thrashing_in_specs_containerfile_solvers_and_verifier_tclean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 3C391 CPU/timeout budgets, Containerfile ENV, solve.py, and verifier tclean OpenMP env."""
    import subprocess
    from typing import Any

    import radio_bench.grading.cal_metrics as cm

    for tid in (
        "r1_3c391_inspect",
        "r1_3c391_precal",
        "r1_3c391_calibrate",
        "r1_3c391_image",
        "r2_3c391_e2e",
    ):
        spec = load_task_spec(DEFAULT_SPECS_DIR / tid)
        assert spec.cpus == 8
        assert spec.effective_verifier_cpus == 8
        if tid in {"r1_3c391_calibrate", "r2_3c391_e2e"}:
            assert spec.verifier_timeout_sec == 1200.0

    for cf_name in ("agent-base.Containerfile", "verifier.Containerfile"):
        cf_text = (REPO_ROOT / "images" / cf_name).read_text(encoding="utf-8")
        for token in (
            "OMP_WAIT_POLICY=PASSIVE",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ):
            assert token in cf_text

    for tid in ALL_BASE_TASKS:
        solve_text = (DEFAULT_SPECS_DIR / tid / "solve.py").read_text(encoding="utf-8")
        assert 'os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")' in solve_text

    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        req_data = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        Path(req_data["out_json"]).write_text(
            json.dumps(
                {
                    "verifier_tclean_peak_jy": 0.12,
                    "verifier_tclean_offsource_rms_jy": 0.0004,
                    "verifier_tclean_dynamic_range": 300.0,
                    "verifier_tclean_physical_valid": True,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    dummy_ms = tmp_path / "dummy.ms"
    dummy_ms.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    res = cm._run_verifier_tclean_recipe(dummy_ms, {}, scratch)
    assert res["verifier_tclean_returncode"] == 0
    assert captured["timeout"] == 900.0
    assert "os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')" in captured["cmd"][2]
    assert captured["env"]["OMP_NUM_THREADS"] == "8"
    assert captured["env"]["OMP_WAIT_POLICY"] == "PASSIVE"
    assert captured["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert captured["env"]["MKL_NUM_THREADS"] == "1"
    assert captured["env"]["NUMEXPR_NUM_THREADS"] == "1"
