"""Comprehensive unit tests for zero-hardcode config, multi-pack isolation, and model registry."""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.lint.models import DEFAULT_RULES, LintRule
from harbor.lint.task import lint_task
from harbor.models.job.config import JobConfig

import radio_bench.build_images as bi
import radio_bench.build_tasks as bt
import radio_bench.config as rb_cfg
import radio_bench.render_jobs as rj
import radio_bench.smoke_containers as sc
from radio_bench.analysis.cli import run_analysis_pipeline
from radio_bench.analysis.collect import (
    collect_single_trial,
    collect_trials,
    parse_task_variant,
    parse_task_variant_with_pack,
)
from radio_bench.analysis.factorial import (
    PARITY_MODEL_PAIRS,
    analyze_factorial,
    resolve_parity_model_pairs,
)
from radio_bench.analysis.pricing import (
    compute_trial_pricing,
    get_model_rate,
    normalize_model_key,
    price_from_agent_context,
    price_turn,
)
from radio_bench.build_tasks import (
    ARM_FULL,
    ARM_NONE,
    ARM_ORCHESTRATED,
    ARM_SKILLS_CASA,
    ARM_SKILLS_RAW,
    ARM_TOOLS,
    _hardlink_file,
    build_all_tasks,
    compile_variant,
    load_task_spec,
)
from radio_bench.config import (
    ModelPricingConfig,
    ModelsConfig,
    ModelSpec,
    PackConfig,
    ParityPairSpec,
    ParitySideSpec,
    is_remote_git_url,
    load_bench_config,
    load_models_config,
    materialize_pack_source,
    prepare_runtime_env,
    reject_slash_tmp,
    resolve_git_ref,
)
from radio_bench.podman_env import ensure_docker_compose_plugin, smoke_compose_teardown
from radio_bench.render_jobs import compile_job_dict, render_all_presets


def _clean_bench_env() -> dict[str, str]:
    """Return an isolated environment mapping with zero RADIO_BENCH_* overrides."""
    return {"PATH": os.environ.get("PATH", "/usr/bin")}


def test_fresh_clone_repo_relative_defaults_and_missing_toml_fallback(tmp_path: Path) -> None:
    """Verify repo-relative defaults when bench.local.toml and RADIO_BENCH_* env vars are absent."""
    fake_repo = tmp_path / "fresh_clone"
    fake_repo.mkdir()
    committed_bench = (rb_cfg.resolve_repo_root() / "bench.toml").read_text(encoding="utf-8")
    (fake_repo / "bench.toml").write_text(committed_bench, encoding="utf-8")

    cfg = load_bench_config(repo_root=fake_repo, env=_clean_bench_env(), uid=4242)
    assert cfg.repo_root == fake_repo.resolve()
    assert cfg.work_dir == (fake_repo / ".work").resolve()
    assert cfg.data_dir == (fake_repo / "data").resolve()
    assert cfg.runs_dir == (fake_repo / "runs").resolve()
    assert cfg.casaguides_dir == (fake_repo / "data" / "casaguides").resolve()
    assert cfg.prompts_dir == (fake_repo / "task_specs").resolve()
    assert cfg.casa_data_dir == Path(os.path.expanduser("~/.casa/data")).resolve()
    assert cfg.docker_host == "unix:///run/user/4242/podman/podman.sock"
    assert cfg.container_tmpdir == (fake_repo / ".work" / "podman_tmp").resolve()
    assert cfg.default_pack == "ra_00a11ea"
    assert set(cfg.packs) == {"ra_00a11ea", "ra_rescue"}
    # Side-effect-free load: zero directories created on disk.
    assert not (fake_repo / ".work").exists()
    assert not (fake_repo / "data").exists()

    # Fallback when bench.toml and configs/models.toml are absent on disk.
    bare_dir = tmp_path / "bare_no_toml"
    bare_dir.mkdir()
    bare_cfg = load_bench_config(repo_root=bare_dir, env=_clean_bench_env(), uid=1001)
    assert bare_cfg.work_dir == (bare_dir / ".work").resolve()
    assert bare_cfg.docker_host == "unix:///run/user/1001/podman/podman.sock"
    bare_models = load_models_config(repo_root=bare_dir, env=_clean_bench_env())
    assert "claude_haiku_4_5" in bare_models.models
    assert "local_llama_qwen" in bare_models.models


def test_precedence_order_and_dynamic_container_tmpdir(tmp_path: Path) -> None:
    """Verify strict precedence CLI > env > bench.local.toml > bench.toml and container_tmpdir tracking."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bench.toml").write_text(
        "\n".join(
            [
                'default_pack = "ra_00a11ea"',
                "[paths]",
                'work_dir = "./toml_work"',
                'data_dir = "./toml_data"',
                'runs_dir = "./toml_runs"',
                "[runtime]",
                'container_tmpdir = "./toml_work/podman_tmp"',
                "[packs.ra_00a11ea]",
                'analyst_src = "https://github.com/skunkworks-ra/radio-analyst.git"',
                'analyst_ref = "00a11ea"',
                'sidecar_image = "localhost/radio-bench-mcp-sidecar:00a11ea"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # 1. bench.local.toml overrides work_dir & data_dir; container_tmpdir tracks work_dir dynamically.
    local_work = tmp_path / "local_work"
    local_data = tmp_path / "local_data"
    (repo / "bench.local.toml").write_text(
        "\n".join(
            [
                "[paths]",
                f'work_dir = "{local_work}"',
                f'data_dir = "{local_data}"',
                "[packs.ra_00a11ea]",
                f'analyst_src = "{tmp_path / "local_ra"}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg_local = load_bench_config(repo_root=repo, env=_clean_bench_env())
    assert cfg_local.work_dir == local_work.resolve()
    assert cfg_local.data_dir == local_data.resolve()
    assert cfg_local.runs_dir == (repo / "toml_runs").resolve()
    assert cfg_local.container_tmpdir == (local_work / "podman_tmp").resolve()
    assert cfg_local.get_pack("ra_00a11ea").analyst_src == str(tmp_path / "local_ra")
    assert cfg_local.get_pack("ra_00a11ea").analyst_ref == "00a11ea"

    # 2. Environment variables override bench.local.toml; container_tmpdir tracks env work_dir.
    env_work = tmp_path / "env_work"
    env_runs = tmp_path / "env_runs"
    env_vars = {
        **_clean_bench_env(),
        "RADIO_BENCH_WORK_DIR": str(env_work),
        "RADIO_BENCH_RUNS_DIR": str(env_runs),
    }
    cfg_env = load_bench_config(repo_root=repo, env=env_vars)
    assert cfg_env.work_dir == env_work.resolve()
    assert cfg_env.runs_dir == env_runs.resolve()
    assert cfg_env.data_dir == local_data.resolve()
    assert cfg_env.container_tmpdir == (env_work / "podman_tmp").resolve()

    # Explicit CONTAINER_TMPDIR in env is preserved when work_dir is not overridden at CLI rank.
    explicit_ctmp = tmp_path / "explicit_ctmp"
    cfg_env_ctmp = load_bench_config(
        repo_root=repo,
        env={**env_vars, "CONTAINER_TMPDIR": str(explicit_ctmp)},
    )
    assert cfg_env_ctmp.container_tmpdir == explicit_ctmp.resolve()

    # 3. Explicit CLI/function args override env and local config.
    cli_work = tmp_path / "cli_work"
    cfg_cli = load_bench_config(
        repo_root=repo,
        env={**env_vars, "CONTAINER_TMPDIR": str(explicit_ctmp)},
        work_dir=cli_work,
    )
    assert cfg_cli.work_dir == cli_work.resolve()
    assert cfg_cli.container_tmpdir == (cli_work / "podman_tmp").resolve()

    cli_ctmp = tmp_path / "cli_ctmp"
    cfg_cli_both = load_bench_config(
        repo_root=repo,
        env=env_vars,
        work_dir=cli_work,
        container_tmpdir=cli_ctmp,
    )
    assert cfg_cli_both.container_tmpdir == cli_ctmp.resolve()


def test_slash_tmp_rejection_direct_and_symlink_while_permitting_var_tmp(tmp_path: Path) -> None:
    """Verify /tmp is rejected syntactically and via symlinks while /var/tmp is allowed."""
    for bad_raw in ("/tmp", "/tmp/", "/tmp/radio_bench_forbidden"):
        with pytest.raises(ValueError, match="/tmp"):
            reject_slash_tmp(bad_raw, label="test_path")

    # Symlink inside non-/tmp tmp_path pointing at /tmp (creates zero files in /tmp).
    sym_link = tmp_path / "sneaky_symlink"
    os.symlink("/tmp/nonexistent_radio_bench_target", sym_link)
    with pytest.raises(ValueError, match="/tmp"):
        reject_slash_tmp(sym_link, label="symlink_path")

    with pytest.raises(ValueError, match="/tmp"):
        load_bench_config(
            repo_root=tmp_path,
            env={**_clean_bench_env(), "RADIO_BENCH_WORK_DIR": str(sym_link)},
            ignore_local_config=True,
        )

    with pytest.raises(ValueError, match="/tmp"):
        load_bench_config(
            repo_root=tmp_path,
            env={**_clean_bench_env(), "CONTAINER_TMPDIR": "/tmp/bad_podman"},
            ignore_local_config=True,
        )

    ok_cfg = load_bench_config(
        repo_root=tmp_path,
        work_dir=tmp_path / "safe_work",
        data_dir=tmp_path / "safe_data",
        runs_dir=tmp_path / "safe_runs",
        env=_clean_bench_env(),
        ignore_local_config=True,
    )
    with pytest.raises(ValueError, match="/tmp"):
        prepare_runtime_env(ok_cfg, tmpdir="/tmp/bad_runtime_tmp")

    # /var/tmp is NOT /tmp and must be permitted (container scratch invariant).
    resolved_var_tmp = reject_slash_tmp("/var/tmp/radio_bench_container_ok", label="var_tmp")
    assert str(resolved_var_tmp).startswith("/var/tmp/")

    prepared = prepare_runtime_env(ok_cfg, base_env=_clean_bench_env())
    assert prepared["TMPDIR"] == str(ok_cfg.tmp_dir)
    assert ok_cfg.casaconfig_path.is_file()
    casa_cfg_text = ok_cfg.casaconfig_path.read_text(encoding="utf-8")
    assert "data_auto_update = False" in casa_cfg_text
    assert "measures_auto_update = False" in casa_cfg_text
    assert "nologfile = True" in casa_cfg_text


def test_lazy_pack_validation_git_head_worktree_and_remote_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify lazy ra_rescue validation plus HEAD, branch, worktree, and remote clone materialization."""
    missing_rescue = tmp_path / "does_not_exist_ra_rescue"
    repo = tmp_path / "cfg_repo"
    repo.mkdir()
    (repo / "bench.toml").write_text(
        "\n".join(
            [
                'default_pack = "ra_00a11ea"',
                "[paths]",
                f'work_dir = "{tmp_path / "work"}"',
                "[packs.ra_00a11ea]",
                'analyst_src = "https://github.com/skunkworks-ra/radio-analyst.git"',
                'analyst_ref = "00a11ea"',
                'sidecar_image = "localhost/radio-bench-mcp-sidecar:00a11ea"',
                "[packs.ra_rescue]",
                f'analyst_src = "{missing_rescue}"',
                'analyst_ref = "HEAD"',
                'sidecar_image = "localhost/radio-bench-mcp-sidecar:ra_rescue"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Missing ra_rescue path must NOT fail load_bench_config().
    cfg = load_bench_config(repo_root=repo, env=_clean_bench_env())
    rescue_pack = cfg.get_pack("ra_rescue")
    with pytest.raises(FileNotFoundError, match="ra_rescue"):
        materialize_pack_source(rescue_pack, cfg.work_dir, repo_root=cfg.repo_root)

    # Create a real temporary git repo + git worktree (where .git is a file, not a dir).
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    src_repo = tmp_path / "mock_analyst_repo"
    src_repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src_repo)], env=git_env, check=True)
    (src_repo / "README.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src_repo), "add", "README.txt"], env=git_env, check=True)
    subprocess.run(
        ["git", "-C", str(src_repo), "commit", "-m", "Initial commit"],
        env=git_env,
        check=True,
    )
    expected_full_sha = (
        subprocess.run(
            ["git", "-C", str(src_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .lower()
    )

    wt_dir = tmp_path / "mock_analyst_worktree"
    subprocess.run(
        ["git", "-C", str(src_repo), "worktree", "add", "-b", "rescue-branch", str(wt_dir), "HEAD"],
        env=git_env,
        check=True,
    )
    assert (wt_dir / ".git").is_file()

    wt_pack = PackConfig(
        pack_id="ra_rescue",
        description="Worktree pack",
        analyst_src=str(wt_dir),
        analyst_ref="HEAD",
        sidecar_image="localhost/radio-bench-mcp-sidecar:ra_rescue",
    )
    mat_path, full_sha, short_sha = materialize_pack_source(
        wt_pack, cfg.work_dir, repo_root=cfg.repo_root
    )
    assert mat_path == wt_dir.resolve()
    assert full_sha == expected_full_sha
    assert short_sha == expected_full_sha[:7]

    branch_pack = PackConfig(
        pack_id="ra_branch",
        description="Branch ref pack",
        analyst_src=str(wt_dir),
        analyst_ref="rescue-branch",
        sidecar_image="localhost/radio-bench-mcp-sidecar:ra_branch",
    )
    _, branch_full, branch_short = materialize_pack_source(
        branch_pack, cfg.work_dir, repo_root=cfg.repo_root
    )
    assert branch_full == expected_full_sha
    assert branch_short == expected_full_sha[:7]

    # Verify remote URL detection and offline clone materialization into <work_dir>/tmp/pack_src_<id>.
    assert is_remote_git_url("https://github.com/skunkworks-ra/radio-analyst.git") is True
    assert is_remote_git_url("git@github.com:skunkworks-ra/radio-analyst.git") is True
    assert is_remote_git_url("../radio-analyst-rescue") is False

    monkeypatch.setattr(rb_cfg, "is_remote_git_url", lambda s: s == str(src_repo))
    remote_pack = PackConfig(
        pack_id="ra_remote_sim",
        description="Simulated remote clone",
        analyst_src=str(src_repo),
        analyst_ref="main",
        sidecar_image="localhost/radio-bench-mcp-sidecar:ra_remote_sim",
    )
    cloned_dir, cloned_full, _ = materialize_pack_source(
        remote_pack, cfg.work_dir, repo_root=cfg.repo_root
    )
    assert cloned_dir == (cfg.work_dir / "tmp" / "pack_src_ra_remote_sim").resolve()
    assert (cloned_dir / ".git").exists()
    assert cloned_full == expected_full_sha


def test_cross_device_hardlink_fallback_warns_once_and_copies_0644(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify EXDEV / cross-filesystem fallback logs a warning once and writes regular 0o644 files."""
    bt._WARNED_CROSS_DEVICE_PAIRS.clear()
    src_a = tmp_path / "src_a.dat"
    src_b = tmp_path / "src_b.dat"
    src_a.write_bytes(b"visibility_a")
    src_b.write_bytes(b"visibility_b")
    src_a.chmod(0o600)
    src_b.chmod(0o600)

    dst_dir = tmp_path / "dst"
    dst_a = dst_dir / "a.dat"
    dst_b = dst_dir / "b.dat"

    def _raise_exdev(_s: str | os.PathLike[str], _d: str | os.PathLike[str]) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", _raise_exdev)
    with caplog.at_level(logging.WARNING, logger="radio_bench.build_tasks"):
        _hardlink_file(src_a, dst_a)
        _hardlink_file(src_b, dst_b)

    warnings = [r for r in caplog.records if "Cross-filesystem staging detected" in r.message]
    assert len(warnings) == 1
    assert dst_a.read_bytes() == b"visibility_a"
    assert dst_b.read_bytes() == b"visibility_b"
    assert dst_a.stat().st_ino != src_a.stat().st_ino
    assert stat.S_IMODE(dst_a.stat().st_mode) == 0o644
    assert stat.S_IMODE(dst_b.stat().st_mode) == 0o644


def _make_minimal_exported_skills_tree(root: Path, sha: str) -> None:
    for rel in (
        "raw/skills/inspect",
        "casa/skills/inspect",
        "orchestrated_skills/orchestrate",
        "agents/claude",
        "agents/commands",
        "agents/opencode",
        "full",
        "orchestrated",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "raw/skills/inspect/SKILL.md").write_text("# inspect\n", encoding="utf-8")
    (root / "casa/skills/inspect/SKILL.md").write_text("# casa inspect\n", encoding="utf-8")
    (root / "orchestrated_skills/orchestrate/SKILL.md").write_text(
        "# orchestrate\n", encoding="utf-8"
    )
    (root / "agents/claude/inspect-stage.md").write_text("# stage\n", encoding="utf-8")
    (root / "agents/commands/inspect.md").write_text("# cmd\n", encoding="utf-8")
    (root / "agents/opencode/inspect-stage.md").write_text("# oc stage\n", encoding="utf-8")
    for mode in ("raw", "casa", "full", "orchestrated"):
        for fname in ("CLAUDE.md", "AGENTS.md"):
            (root / mode / fname).write_text(f"# {mode}\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"radio_analyst_sha": sha, "radio_analyst_commit": sha * 5 + "abcde"}) + "\n",
        encoding="utf-8",
    )


def test_single_vs_multipack_task_compilation_lint_and_collector_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify single-pack (<task>__<arm>) vs multi-pack (<task>__<pack>__<arm>), lint, and collect."""
    live_cfg = load_bench_config()
    compose_cfg = ensure_docker_compose_plugin(live_cfg.docker_config_dir)
    if compose_cfg is not None:
        monkeypatch.setenv("DOCKER_CONFIG", str(compose_cfg))

    fake_repo = tmp_path / "mp_repo"
    specs_sub = fake_repo / "task_specs"
    specs_sub.mkdir(parents=True)
    os.symlink(live_cfg.repo_root / "task_specs" / "s1_inspect", specs_sub / "s1_inspect")
    os.symlink(live_cfg.repo_root / "task_specs" / "s2_e2e_synth", specs_sub / "s2_e2e_synth")

    work_dir = tmp_path / "mp_work"
    skills_root = work_dir / "exported_skills"
    _make_minimal_exported_skills_tree(skills_root / "ra_00a11ea", "00a11ea")
    _make_minimal_exported_skills_tree(skills_root / "ra_rescue", "7f6e5d4")

    (fake_repo / "bench.toml").write_text(
        "\n".join(
            [
                'default_pack = "ra_00a11ea"',
                "[paths]",
                f'work_dir = "{work_dir}"',
                f'data_dir = "{live_cfg.data_dir}"',
                f'runs_dir = "{tmp_path / "mp_runs"}"',
                "[packs.ra_00a11ea]",
                'analyst_src = "https://github.com/skunkworks-ra/radio-analyst.git"',
                'analyst_ref = "00a11ea"',
                'sidecar_image = "localhost/radio-bench-mcp-sidecar:00a11ea"',
                "mcp_ports = { ms-inspect = 8000, ms-modify = 8001, ms-create = 8002 }",
                "[packs.ra_rescue]",
                f'analyst_src = "{tmp_path / "unused_lazy_rescue"}"',
                'analyst_ref = "HEAD"',
                'sidecar_image = "localhost/radio-bench-mcp-sidecar:ra_rescue"',
                "mcp_ports = { ms-inspect = 9000, ms-modify = 9001, ms-create = 9002 }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mp_cfg = load_bench_config(repo_root=fake_repo, env=_clean_bench_env())

    # 1. Single-pack compile -> <task_id>__<arm>
    out_single = work_dir / "tasks_single"
    summary_single = build_all_tasks(
        specs_dir=specs_sub,
        skills_dir=skills_root,
        output_dir=out_single,
        pack_id="ra_00a11ea",
        cfg=mp_cfg,
    )
    assert summary_single["compiled_tasks"] == 11
    assert summary_single["include_pack_in_name"] is False
    single_tool_dir = out_single / f"s1_inspect__{ARM_TOOLS}"
    assert single_tool_dir.is_dir()
    single_toml = tomllib.loads((single_tool_dir / "task.toml").read_text(encoding="utf-8"))
    assert single_toml["metadata"]["pack_id"] == "ra_00a11ea"
    assert single_toml["metadata"]["analyst_sha"] == "00a11ea"
    assert single_toml["metadata"]["radio_analyst_sha"] == "00a11ea"

    # 2. Multi-pack compile -> <task_id>__<pack_id>__<arm>
    out_multi = work_dir / "tasks_multi"
    summary_multi = build_all_tasks(
        specs_dir=specs_sub,
        skills_dir=skills_root,
        output_dir=out_multi,
        pack_ids=["ra_00a11ea", "ra_rescue"],
        cfg=mp_cfg,
    )
    assert summary_multi["compiled_tasks"] == 22
    assert summary_multi["include_pack_in_name"] is True
    assert summary_multi["by_pack"] == {"ra_00a11ea": 11, "ra_rescue": 11}

    rescue_full_dir = out_multi / f"s2_e2e_synth__ra_rescue__{ARM_ORCHESTRATED}"
    assert rescue_full_dir.is_dir()
    rescue_toml = tomllib.loads((rescue_full_dir / "task.toml").read_text(encoding="utf-8"))
    assert rescue_toml["metadata"]["base_task"] == "s2_e2e_synth"
    assert rescue_toml["metadata"]["pack_id"] == "ra_rescue"
    assert rescue_toml["metadata"]["analyst_sha"] == "7f6e5d4"
    assert rescue_toml["metadata"]["arm"] == ARM_ORCHESTRATED
    urls = [s["url"] for s in rescue_toml["environment"]["mcp_servers"]]
    assert urls == [
        "http://127.0.0.1:9000/mcp",
        "http://127.0.0.1:9001/mcp",
        "http://127.0.0.1:9002/mcp",
    ]

    rescue_compose = yaml.safe_load(
        (rescue_full_dir / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    assert (
        rescue_compose["services"]["mcp-servers"]["image"]
        == "${RADIO_BENCH_SIDECAR_IMAGE:-localhost/radio-bench-mcp-sidecar:ra_rescue}"
    )
    assert (
        rescue_compose["services"]["mcp-servers"]["environment"]["RADIO_MCP_PORT_MS_INSPECT"]
        == "9000"
    )

    # Full Harbor lint (including UNPINNED_IMAGE compose rule) must pass with zero findings!
    full_rules = DEFAULT_RULES | {LintRule.ROOT_AGENT}
    assert lint_task(single_tool_dir, selected_rules=full_rules) == []
    assert lint_task(rescue_full_dir, selected_rules=full_rules) == []

    # 3. Collector round-trip across single-pack, multi-pack, and truncated Harbor trial names.
    assert parse_task_variant("s1_inspect__11_full") == (
        "s1_inspect",
        ARM_FULL,
        "synth",
        "inspect",
        False,
    )
    assert parse_task_variant_with_pack("s1_inspect__ra_rescue__01_skills_casa") == (
        "s1_inspect",
        "ra_rescue",
        ARM_SKILLS_CASA,
        "synth",
        "inspect",
        False,
    )
    assert parse_task_variant_with_pack("s2_e2e_synth__ra_rescue__11_orchestra__9a8b7c6") == (
        "s2_e2e_synth",
        "ra_rescue",
        ARM_ORCHESTRATED,
        "synth",
        "e2e",
        True,
    )
    assert parse_task_variant_with_pack("r1_3c391_inspect__01_skills_ra__1122334") == (
        "r1_3c391_inspect",
        None,
        ARM_SKILLS_RAW,
        "3c391",
        "inspect",
        False,
    )

    fake_job = tmp_path / "fake_mp_job"
    fake_trial = fake_job / "s1_inspect__ra_rescue__11_full__abc1234"
    (fake_trial / "verifier").mkdir(parents=True)
    (fake_trial / "config.json").write_text(
        json.dumps(
            {
                "task": {"path": str(out_multi / "s1_inspect__ra_rescue__11_full")},
                "agent": {"name": "oracle"},
            }
        ),
        encoding="utf-8",
    )
    (fake_trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "s1_inspect__ra_rescue__11_full",
                "trial_name": fake_trial.name,
                "agent_info": {"name": "oracle"},
            }
        ),
        encoding="utf-8",
    )
    (fake_trial / "verifier" / "reward.json").write_text(
        json.dumps({"reward": 1.0, "passed": 1}), encoding="utf-8"
    )
    rec = collect_single_trial(fake_job, fake_trial)
    assert rec["base_task"] == "s1_inspect"
    assert rec["pack_id"] == "ra_rescue"
    assert rec["arm"] == ARM_FULL
    assert rec["passed"] == 1


def test_models_registry_pricing_parity_and_render_jobs_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify models.toml pricing, thinking levels, concurrency caps, and pasta host-gateway overlay."""
    # 1. Provider-aware model normalization & pricing rates.
    assert normalize_model_key("tacc/kimi") == "tacc/kimi"
    assert normalize_model_key("nrp/kimi") == "nrp/kimi"
    assert normalize_model_key("local/qwen") == "local/qwen"
    assert normalize_model_key("claude-haiku-4-5") == "anthropic/claude-haiku-4-5"
    assert normalize_model_key("claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"

    assert get_model_rate("tacc/kimi") is not None
    assert price_turn(
        "tacc/kimi", uncached_input_tokens=100_000, output_tokens=10_000
    ).total_usd == pytest.approx(0.09)
    assert price_turn(
        "nrp/kimi", uncached_input_tokens=100_000, output_tokens=10_000
    ).total_usd == pytest.approx(0.09)
    local_p = price_turn("local/qwen", uncached_input_tokens=200_000, output_tokens=20_000)
    assert local_p.rate_known is True
    assert local_p.total_usd == 0.0
    assert local_p.cliff_triggered is False

    haiku_cliff = price_turn("claude_haiku_4_5", uncached_input_tokens=100_001, output_tokens=0)
    assert haiku_cliff.cliff_triggered is True
    assert haiku_cliff.total_usd == pytest.approx(100_001 * 5.0 / 1_000_000.0)

    assert PARITY_MODEL_PAIRS == (
        ("anthropic_haiku_vs_sonnet", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6"),
        ("nrp_kimi_vs_glm5", "nrp/kimi", "nrp/glm-5"),
    )
    assert resolve_parity_model_pairs() == PARITY_MODEL_PAIRS

    # 2. Canonical preset rendering validates against Harbor JobConfig.
    preset_dir = tmp_path / "preset_jobs"
    rendered = render_all_presets(output_dir=preset_dir)
    assert set(rendered) == {
        "oracle_synth",
        "oracle_3c391",
        "suite_s_ablation",
        "suite_s_e2e_ablation",
        "suite_r_ablation",
        "suite_r_e2e_ablation",
    }
    for p_path in rendered.values():
        JobConfig.model_validate(yaml.safe_load(p_path.read_text(encoding="utf-8")))

    # 3. Missing TACC_BASE_URL raises ValueError; setting it validates cleanly.
    with pytest.raises(ValueError, match="TACC_BASE_URL"):
        compile_job_dict(
            job_name="missing_tacc_url",
            suites=["synth"],
            arms=[ARM_NONE],
            model_keys=["tacc_kimi"],
            jobs_out_dir=tmp_path / "jobs_err",
            env=_clean_bench_env(),
        )

    # 4. Combined job across ALL 6 models + thinking_levels=["default", "high"] + local gateway:
    #    - local_llama_qwen caps job n_concurrent_trials to 1 (even on synth with suite_cap=2)
    #    - LOCAL_LLM_API_KEY unset -> env={"LOCAL_LLM_API_KEY": "local-placeholder"} passes preflight
    #    - claude_sonnet_4_6 emits both default (no reasoning_effort) and high (reasoning_effort="high")
    #    - host_gateway_compose.yaml written and wired into environment.extra_docker_compose
    custom_models_toml = tmp_path / "custom_models.toml"
    base_models_text = (rb_cfg.resolve_repo_root() / "configs" / "models.toml").read_text(
        encoding="utf-8"
    )
    # Also test opencode non-default thinking level -> kwargs.variant (never reasoning_effort)
    custom_models_toml.write_text(
        base_models_text.replace(
            '[models.nrp_kimi]\nharness = "opencode"\nmodel_name = "nrp/kimi"\napi_format = "openai_chat_completions"\nbase_url = "https://ellm.nrp-nautilus.io/v1"\napi_key_env = "NRP_API_KEY"\nopencode_version = "1.19.5"\nthinking_levels = ["default"]',
            '[models.nrp_kimi]\nharness = "opencode"\nmodel_name = "nrp/kimi"\napi_format = "openai_chat_completions"\nbase_url = "https://ellm.nrp-nautilus.io/v1"\napi_key_env = "NRP_API_KEY"\nopencode_version = "1.19.5"\nthinking_levels = ["default", "high"]',
        ),
        encoding="utf-8",
    )
    c_models = load_models_config(config_path=custom_models_toml, env=_clean_bench_env())

    test_env: dict[str, str] = {
        **_clean_bench_env(),
        "ANTHROPIC_API_KEY": "dummy-ant",
        "NRP_API_KEY": "dummy-nrp",
        "TACC_API_KEY": "dummy-tacc",
        "TACC_BASE_URL": "https://tacc-inference.example.org/v1",
    }
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-ant")
    monkeypatch.setenv("NRP_API_KEY", "dummy-nrp")
    monkeypatch.setenv("TACC_API_KEY", "dummy-tacc")
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)

    gw_jobs_dir = tmp_path / "gw_jobs"
    all_models_job: dict[str, Any] = compile_job_dict(
        job_name="all_endpoints_smoke",
        suites=["synth"],
        arms=[ARM_NONE, ARM_FULL],
        task_scope="stage",
        model_keys=[
            "claude_haiku_4_5",
            "claude_sonnet_4_6",
            "nrp_kimi",
            "nrp_glm5",
            "tacc_kimi",
            "local_llama_qwen",
        ],
        thinking_levels=None,
        pack_ids=["ra_00a11ea", "ra_rescue"],
        jobs_out_dir=gw_jobs_dir,
        models_cfg=c_models,
        env=test_env,
    )
    yaml_path = gw_jobs_dir / "all_endpoints_smoke.yaml"
    yaml_path.write_text(yaml.safe_dump(all_models_job, sort_keys=False), encoding="utf-8")

    validated = JobConfig.model_validate(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))
    assert validated.n_concurrent_trials == 1
    # 4 synth stage tasks * 2 packs * 2 arms = 16 multi-pack tasks
    assert len(validated.tasks) == 16
    assert any("s1_inspect__ra_rescue__11_full" in str(t.path) for t in validated.tasks)

    overlay_file = gw_jobs_dir / "host_gateway_compose.yaml"
    assert overlay_file.is_file()
    assert validated.environment.extra_docker_compose == [overlay_file]
    overlay_data = yaml.safe_load(overlay_file.read_text(encoding="utf-8"))
    assert overlay_data["services"]["main"]["extra_hosts"] == [
        "host.containers.internal:host-gateway"
    ]

    # 6 models + 1 extra 'high' variant on sonnet + 1 extra 'high' variant on nrp_kimi = 8 agents
    assert len(validated.agents) == 8
    sonnet_agents = [a for a in validated.agents if a.model_name == "anthropic/claude-sonnet-4-6"]
    assert len(sonnet_agents) == 2
    assert "reasoning_effort" not in sonnet_agents[0].kwargs
    assert sonnet_agents[1].kwargs["reasoning_effort"] == "high"
    assert sonnet_agents[0].model_api_format is None

    kimi_agents = [a for a in validated.agents if a.model_name == "nrp/kimi"]
    assert len(kimi_agents) == 2
    assert "variant" not in kimi_agents[0].kwargs
    assert kimi_agents[1].kwargs["variant"] == "high"
    assert "reasoning_effort" not in kimi_agents[1].kwargs

    local_agents = [a for a in validated.agents if a.model_name == "local/qwen"]
    assert len(local_agents) == 1
    assert local_agents[0].base_url == "http://host.containers.internal:8080/v1"
    assert local_agents[0].env == {"LOCAL_LLM_API_KEY": "local-placeholder"}

    for agent_cfg in validated.agents:
        AgentFactory.run_preflight(agent_cfg)


def test_cli_config_flags_and_child_process_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--config`, `--work-dir`, `--data-dir`, `--runs-dir` propagate to child processes."""
    custom_work = tmp_path / "cfg_work"
    custom_data = tmp_path / "cfg_data"
    custom_runs = tmp_path / "cfg_runs"
    custom_toml = tmp_path / "custom_bench.toml"
    custom_toml.write_text(
        "\n".join(
            [
                'default_pack = "ra_00a11ea"',
                "[paths]",
                f'work_dir = "{custom_work}"',
                f'data_dir = "{custom_data}"',
                f'runs_dir = "{custom_runs}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repo_root = rb_cfg.resolve_repo_root()
    child_env = {
        **_clean_bench_env(),
        "PYTHONPATH": str(repo_root / "src"),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "radio_bench.cli",
            "--config",
            str(custom_toml),
            "render-jobs",
            "--preset",
            "oracle_synth",
        ],
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    rendered_payload = json.loads(proc.stdout)
    expected_yaml = (custom_work / "jobs" / "oracle_synth.yaml").resolve()
    assert Path(rendered_payload["rendered_presets"]["oracle_synth"]).resolve() == expected_yaml
    assert expected_yaml.is_file()
    job_doc = yaml.safe_load(expected_yaml.read_text(encoding="utf-8"))
    assert Path(job_doc["jobs_dir"]).resolve() == custom_runs.resolve()
    assert all(str((custom_work / "tasks").resolve()) in str(t["path"]) for t in job_doc["tasks"])

    # Verify CLI `--work-dir` and `--runs-dir` placed AFTER subcommand also override `--config`.
    cli_work_override = tmp_path / "cli_override_work"
    cli_runs_override = tmp_path / "cli_override_runs"
    proc_override = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "radio_bench.cli",
            "render-jobs",
            "--config",
            str(custom_toml),
            "--work-dir",
            str(cli_work_override),
            "--runs-dir",
            str(cli_runs_override),
            "--preset",
            "oracle_3c391",
        ],
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    override_payload = json.loads(proc_override.stdout)
    expected_override_yaml = (cli_work_override / "jobs" / "oracle_3c391.yaml").resolve()
    assert (
        Path(override_payload["rendered_presets"]["oracle_3c391"]).resolve()
        == expected_override_yaml
    )
    override_doc = yaml.safe_load(expected_override_yaml.read_text(encoding="utf-8"))
    assert Path(override_doc["jobs_dir"]).resolve() == cli_runs_override.resolve()

    # Verify `radio_bench.build_images` wrapper forwards flags + prepared runtime env cleanly.
    recorded_cmds: list[tuple[list[str], dict[str, str]]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append((list(cmd), dict(kwargs.get("env") or {})))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(bi.subprocess, "run", _fake_run)
    rc_bi = bi.main(
        [
            "--config",
            str(custom_toml),
            "--work-dir",
            str(cli_work_override),
            "--pack",
            "ra_00a11ea",
            "--verifier-only",
            "--skip-smoke",
        ]
    )
    assert rc_bi == 0
    assert len(recorded_cmds) == 1
    sent_cmd, sent_env = recorded_cmds[0]
    assert sent_cmd[0] == "bash"
    assert sent_cmd[1].endswith("images/build_images.sh")
    assert "--verifier-only" in sent_cmd
    assert "--skip-smoke" in sent_cmd
    assert sent_env["RADIO_BENCH_WORK_DIR"] == str(cli_work_override.resolve())
    assert sent_env["RADIO_BENCH_CONFIG"] == str(custom_toml.resolve())


def test_lazy_module_imports_with_broken_local_toml_and_cli_rescue(tmp_path: Path) -> None:
    """Importing modules must not evaluate broken bench.local.toml/models.local.toml before CLI parse."""
    fake_repo = tmp_path / "broken_local_repo"
    (fake_repo / "configs").mkdir(parents=True)
    # Intentionally invalid /tmp work_dir in bench.local.toml and invalid concurrency=0 in models.local.toml
    (fake_repo / "bench.local.toml").write_text(
        '[paths]\nwork_dir = "/tmp/forbidden_on_import"\n',
        encoding="utf-8",
    )
    (fake_repo / "models.local.toml").write_text(
        "[models.claude_haiku_4_5]\nmax_concurrent_trials = 0\n",
        encoding="utf-8",
    )

    real_repo = rb_cfg.resolve_repo_root()
    import_env = {
        **_clean_bench_env(),
        "RADIO_BENCH_REPO_ROOT": str(fake_repo),
        "TMPDIR": "/tmp",
        "PYTHONPATH": f"{real_repo}:{real_repo / 'src'}",
    }
    import_snippet = (
        "import sys; "
        "import radio_bench.export_skills, radio_bench.build_tasks, "
        "radio_bench.analysis.pricing, radio_bench.analysis.factorial, "
        "data_builders.sim_common, data_builders.build_synthetic_suite, "
        "data_builders.build_3c391_checkpoints; "
        "assert 'casatools' not in sys.modules; "
        "print('imports_clean')"
    )
    proc_imp = subprocess.run(
        [sys.executable, "-B", "-c", import_snippet],
        env=import_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_imp.returncode == 0, proc_imp.stderr
    assert "imports_clean" in proc_imp.stdout

    # Now rescue via explicit `--config` and `--models-config` in that same broken repo!
    valid_bench = tmp_path / "rescue_bench.toml"
    rescue_work = tmp_path / "rescue_work"
    valid_bench.write_text(
        f'[paths]\nwork_dir = "{rescue_work}"\nruns_dir = "{tmp_path / "rescue_runs"}"\n',
        encoding="utf-8",
    )
    valid_models = real_repo / "configs" / "models.toml"
    proc_rescue = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "radio_bench.render_jobs",
            "--config",
            str(valid_bench),
            "--models-config",
            str(valid_models),
            "--preset",
            "oracle_synth",
        ],
        env=import_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_rescue.returncode == 0, proc_rescue.stderr
    assert (rescue_work / "jobs" / "oracle_synth.yaml").is_file()


def test_multipack_collect_trials_truncation_and_factorial_separation(tmp_path: Path) -> None:
    """Verify 32-char Harbor truncation guard, per-pack attempt_idx/dedup, and factorial by_pack/parity."""
    # 1. Harbor 32-char truncated trial directory name where `__<arm>` was cut off before `__<uuid>`:
    assert parse_task_variant_with_pack("r1_3c391_calibrate__ra_00a11ea__a1b2c3d") == (
        "r1_3c391_calibrate",
        "ra_00a11ea",
        ARM_NONE,
        "3c391",
        "calibrate",
        False,
    )

    # Verify collect_single_trial recovers `metadata.arm` from task.toml when trial_dir.name was truncated.
    fake_task_dir = tmp_path / "compiled_task"
    fake_task_dir.mkdir()
    (fake_task_dir / "task.toml").write_text(
        '[metadata]\nbase_task = "r1_3c391_calibrate"\npack_id = "ra_00a11ea"\narm = "11_full"\n',
        encoding="utf-8",
    )
    trunc_job = tmp_path / "trunc_job"
    trunc_trial = trunc_job / "r1_3c391_calibrate__ra_00a11ea__a1b2c3d"
    (trunc_trial / "verifier").mkdir(parents=True)
    (trunc_trial / "config.json").write_text(
        json.dumps({"task": {"path": str(fake_task_dir)}, "agent": {"name": "oracle"}}),
        encoding="utf-8",
    )
    # Intentionally omit `task_name` in result.json so _resolve_task_name uses config task.path / task.toml
    (trunc_trial / "result.json").write_text(
        json.dumps({"trial_name": trunc_trial.name, "agent_info": {"name": "oracle"}}),
        encoding="utf-8",
    )
    (trunc_trial / "verifier" / "reward.json").write_text(
        json.dumps({"reward": 0.95, "passed": 1}), encoding="utf-8"
    )
    trunc_rec = collect_single_trial(trunc_job, trunc_trial)
    assert trunc_rec["pack_id"] == "ra_00a11ea"
    assert trunc_rec["arm"] == "11_full"

    # 2. Multi-pack collect_trials + analyze_factorial separation across ra_00a11ea and ra_rescue.
    mp_job = tmp_path / "mp_ablation_job"
    arms = (ARM_NONE, ARM_SKILLS_RAW, ARM_SKILLS_CASA, ARM_TOOLS, ARM_FULL)
    for pack_id, reward_boost in (("ra_00a11ea", 0.0), ("ra_rescue", 0.20)):
        for arm in arms:
            for att in range(3):
                # Deliberately share `trial_name` across packs to verify pack_id is in dedup key!
                shared_tname = f"s1_inspect__{arm}__att{att}"
                tdir = mp_job / f"{pack_id}__{shared_tname}"
                (tdir / "verifier").mkdir(parents=True)
                base_r = 0.50 if arm in {ARM_NONE, ARM_SKILLS_RAW} else 0.75
                r_val = min(1.0, base_r + reward_boost)
                (tdir / "result.json").write_text(
                    json.dumps(
                        {
                            "task_name": f"s1_inspect__{pack_id}__{arm}",
                            "trial_name": shared_tname,
                            "started_at": f"2026-11-17T00:00:0{att}Z",
                            "agent_info": {
                                "name": "claude-code",
                                "model_info": {"name": "anthropic/claude-haiku-4-5"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (tdir / "verifier" / "reward.json").write_text(
                    json.dumps({"reward": r_val, "passed": 1 if r_val >= 0.80 else 0}),
                    encoding="utf-8",
                )

            # Frontier trials on 01_skills_casa for custom parity arm test
            f_tname = f"s1_inspect__{pack_id}__front_{arm}"
            if arm == ARM_SKILLS_CASA:
                for att in range(3):
                    ftdir = mp_job / f"{f_tname}_{att}"
                    (ftdir / "verifier").mkdir(parents=True)
                    (ftdir / "result.json").write_text(
                        json.dumps(
                            {
                                "task_name": f"s1_inspect__{pack_id}__{arm}",
                                "trial_name": ftdir.name,
                                "started_at": f"2026-11-17T00:01:0{att}Z",
                                "agent_info": {
                                    "name": "claude-code",
                                    "model_info": {"name": "anthropic/claude-sonnet-4-6"},
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    (ftdir / "verifier" / "reward.json").write_text(
                        json.dumps({"reward": 0.60, "passed": 0}),
                        encoding="utf-8",
                    )

    records = collect_trials([mp_job])
    haiku_full = [
        r
        for r in records
        if r["model_name"] == "anthropic/claude-haiku-4-5" and r["arm"] == ARM_FULL
    ]
    # 3 attempts in ra_00a11ea + 3 attempts in ra_rescue = 6 surviving records, each indexed 0..2!
    assert len(haiku_full) == 6
    by_p = {
        pid: [r["attempt_idx"] for r in haiku_full if r["pack_id"] == pid]
        for pid in ("ra_00a11ea", "ra_rescue")
    }
    assert by_p == {"ra_00a11ea": [0, 1, 2], "ra_rescue": [0, 1, 2]}

    custom_models = ModelsConfig(
        models={
            "claude_haiku_4_5": ModelSpec(
                key="claude_haiku_4_5",
                harness="claude-code",
                model_name="anthropic/claude-haiku-4-5",
            ),
            "claude_sonnet_4_6": ModelSpec(
                key="claude_sonnet_4_6",
                harness="claude-code",
                model_name="anthropic/claude-sonnet-4-6",
            ),
        },
        parity_pairs=(
            ParityPairSpec(
                pair_id="custom_tools_vs_casa",
                small=ParitySideSpec(model="claude_haiku_4_5", arm=ARM_TOOLS),
                frontier=ParitySideSpec(model="claude_sonnet_4_6", arm=ARM_SKILLS_CASA),
            ),
        ),
    )
    fact = analyze_factorial(records, n_boot=50, seed=42, models_cfg=custom_models)
    # Because each (base_task, pack_id, model) unit has 3 attempts (not 6!), pass_at_5 must be None!
    assert fact["overall"]["cells"][ARM_FULL]["pass_at_3"] is not None
    assert fact["overall"]["cells"][ARM_FULL]["pass_at_5"] is None
    assert "by_pack" in fact
    assert set(fact["by_pack"]) == {"ra_00a11ea", "ra_rescue"}
    assert fact["by_pack"]["ra_00a11ea"]["cells"][ARM_FULL]["reward_mean"] == pytest.approx(0.75)
    assert fact["by_pack"]["ra_rescue"]["cells"][ARM_FULL]["reward_mean"] == pytest.approx(0.95)

    custom_parity_key = "custom_tools_vs_casa__10_tools_vs_01_skills_casa"
    assert custom_parity_key in fact["overall"]["parity"]
    p_entry = fact["overall"]["parity"][custom_parity_key]
    assert p_entry["status"] == "ok"
    assert p_entry["small_arm"] == ARM_TOOLS
    assert p_entry["frontier_arm"] == ARM_SKILLS_CASA
    # Mean of (0.75, 0.95) = 0.85 vs frontier 0.60 -> +0.25
    assert p_entry["estimate"] == pytest.approx(0.25)

    # Single-pack factorial output omits `by_pack` to preserve exact legacy schema.
    single_pack_recs = [r for r in records if r["pack_id"] == "ra_00a11ea"]
    fact_single = analyze_factorial(single_pack_recs, n_boot=20, seed=42, models_cfg=custom_models)
    assert "by_pack" not in fact_single


def test_gitignore_symlinks_analyst_src_tmp_guard_and_remote_branch_fallback(
    tmp_path: Path,
) -> None:
    """Verify .gitignore symlink matching, analyst_src /tmp symlink rejection, and origin/<branch> ref resolution."""
    repo_root = rb_cfg.resolve_repo_root()
    for ignored_rel in (
        ".work",
        "data",
        "runs",
        "bench.local.toml",
        "models.local.toml",
        "configs/models.local.toml",
    ):
        chk = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "-q", ignored_rel],
            check=False,
        )
        assert chk.returncode == 0, f"Expected .gitignore to ignore {ignored_rel!r}"

    # Symlink to /tmp in [packs.<id>].analyst_src must fail immediately at load_bench_config()!
    sneaky_pack_link = tmp_path / "sneaky_pack_src"
    os.symlink("/tmp/nonexistent_pack_repo", sneaky_pack_link)
    bad_repo = tmp_path / "bad_pack_repo"
    bad_repo.mkdir()
    (bad_repo / "bench.toml").write_text(
        f'[packs.ra_rescue]\nanalyst_src = "{sneaky_pack_link}"\nanalyst_ref = "HEAD"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="/tmp"):
        load_bench_config(repo_root=bad_repo, env=_clean_bench_env())

    # Remote `--no-checkout` clone resolving non-default branch via `origin/<branch>^{commit}`:
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    origin_repo = tmp_path / "origin_repo"
    origin_repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(origin_repo)], env=git_env, check=True)
    (origin_repo / "f.txt").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(origin_repo), "add", "f.txt"], env=git_env, check=True)
    subprocess.run(
        ["git", "-C", str(origin_repo), "commit", "-m", "main commit"], env=git_env, check=True
    )
    subprocess.run(
        ["git", "-C", str(origin_repo), "checkout", "-b", "feat-nondefault"],
        env=git_env,
        check=True,
    )
    (origin_repo / "f.txt").write_text("feature\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(origin_repo), "add", "f.txt"], env=git_env, check=True)
    subprocess.run(
        ["git", "-C", str(origin_repo), "commit", "-m", "feature commit"], env=git_env, check=True
    )
    feat_sha = (
        subprocess.run(
            ["git", "-C", str(origin_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .lower()
    )
    # Switch origin HEAD back to `main` so `--no-checkout` clone has NO local `feat-nondefault` ref!
    subprocess.run(["git", "-C", str(origin_repo), "checkout", "main"], env=git_env, check=True)

    no_checkout_clone = tmp_path / "no_checkout_clone"
    subprocess.run(
        ["git", "clone", "--no-checkout", "--quiet", str(origin_repo), str(no_checkout_clone)],
        check=True,
    )
    resolved_full, resolved_short = resolve_git_ref(no_checkout_clone, "feat-nondefault")
    assert resolved_full == feat_sha
    assert resolved_short == feat_sha[:7]


def test_prompts_dir_override_readonly_copy_models_merges_and_explicit_tmp_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prompts_dir precedence, read-only cross-device overwrite, models merges, and /tmp guards."""
    live_cfg = load_bench_config()

    # 1. `prompts_dir` override takes precedence over `spec_dir / "instruction.md"`.
    custom_prompts = tmp_path / "custom_prompts"
    (custom_prompts / "s1_inspect").mkdir(parents=True)
    override_inst = custom_prompts / "s1_inspect" / "instruction.md"
    override_inst.write_text("# Custom Overridden Prompt\n", encoding="utf-8")
    override_cfg = load_bench_config(prompts_dir=custom_prompts)
    spec = load_task_spec(live_cfg.repo_root / "task_specs" / "s1_inspect", cfg=override_cfg)
    assert spec.effective_instruction_path.resolve() == override_inst.resolve()

    # 2. Cross-device `_hardlink_file` fallback overwrites an existing read-only 0o444 file cleanly.
    src_f = tmp_path / "src_ro.txt"
    dst_f = tmp_path / "dst_ro.txt"
    src_f.write_text("updated_payload", encoding="utf-8")
    dst_f.write_text("old_readonly", encoding="utf-8")
    dst_f.chmod(0o444)

    def _raise_exdev(_s: str | os.PathLike[str], _d: str | os.PathLike[str]) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", _raise_exdev)
    _hardlink_file(src_f, dst_f)
    assert dst_f.read_text(encoding="utf-8") == "updated_payload"
    assert stat.S_IMODE(dst_f.stat().st_mode) == 0o644

    # 3. Models config edge cases:
    #    - `api_format = "anthropic"` normalizes to `"anthropic_messages"`
    #    - overriding `base_url` with `base_url_env` in `models.local.toml` clears `base_url`
    #    - `max_concurrent_trials = 0` raises ValueError
    #    - `n_concurrent` is emitted explicitly on `local_llama_qwen` even when `job_n_concurrent == 1`
    #    - `compute_trial_pricing` and `price_from_agent_context` honor passed `models_cfg`
    m_repo = tmp_path / "models_edge_repo"
    (m_repo / "configs").mkdir(parents=True)
    (m_repo / "configs" / "models.toml").write_text(
        "\n".join(
            [
                "[models.custom_ant]",
                'harness = "opencode"',
                'model_name = "custom/claude"',
                'api_format = "anthropic"',
                'base_url = "https://old.example.com/v1"',
                'api_key_env = "CUSTOM_KEY"',
                "[models.local_llama_qwen]",
                'harness = "opencode"',
                'model_name = "local/qwen"',
                'api_format = "openai_chat_completions"',
                'base_url = "http://host.containers.internal:8080/v1"',
                'api_key_env = "LOCAL_LLM_API_KEY"',
                "max_concurrent_trials = 1",
                "host_gateway_required = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (m_repo / "models.local.toml").write_text(
        '[models.custom_ant]\nbase_url_env = "CUSTOM_NEW_URL"\n',
        encoding="utf-8",
    )
    edge_models = load_models_config(repo_root=m_repo, env=_clean_bench_env())
    assert edge_models.models["custom_ant"].api_format == "anthropic_messages"
    assert edge_models.models["custom_ant"].base_url is None
    assert edge_models.models["custom_ant"].base_url_env == "CUSTOM_NEW_URL"

    suite_r_job = compile_job_dict(
        job_name="local_3c391_cap",
        suites=["3c391"],
        arms=[ARM_NONE],
        task_scope="stage",
        model_keys=["local_llama_qwen"],
        jobs_out_dir=tmp_path / "edge_jobs",
        models_cfg=edge_models,
        env=_clean_bench_env(),
    )
    assert suite_r_job["n_concurrent_trials"] == 1
    assert suite_r_job["agents"][0]["n_concurrent"] == 1

    bad_models_file = tmp_path / "bad_conc_models.toml"
    bad_models_file.write_text(
        '[models.bad]\nharness = "claude-code"\nmodel_name = "x/y"\nmax_concurrent_trials = 0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_concurrent_trials"):
        load_models_config(config_path=bad_models_file, env=_clean_bench_env())

    injected_models = ModelsConfig(
        models={
            "custom_paid": ModelSpec(
                key="custom_paid",
                harness="opencode",
                model_name="custom/paid-v1",
                pricing=ModelPricingConfig(
                    input_per_mtok=10.0,
                    output_per_mtok=20.0,
                    cache_read_per_mtok=1.0,
                    cache_write_5m_per_mtok=10.0,
                    cache_write_1h_per_mtok=10.0,
                ),
            )
        },
        parity_pairs=(),
    )
    ctx_priced = price_from_agent_context(
        {"n_input_tokens": 100_000, "n_cache_tokens": 0, "n_output_tokens": 50_000},
        model_name="custom/paid-v1",
        agent_name="opencode",
        n_agent_steps=1,
        models_cfg=injected_models,
    )
    assert ctx_priced.pricing_source == "agent_context"
    assert ctx_priced.total_usd == pytest.approx(2.0)

    traj_priced = compute_trial_pricing(
        model_name="custom/paid-v1",
        agent_name="opencode",
        trajectory_dict={
            "steps": [
                {
                    "source": "agent",
                    "metrics": {"prompt_tokens": 100_000, "completion_tokens": 50_000},
                }
            ]
        },
        models_cfg=injected_models,
    )
    assert traj_priced.pricing_source == "trajectory_steps"
    assert traj_priced.total_usd == pytest.approx(2.0)

    # 4. Explicit `/tmp` path arguments are rejected across all helper entrypoints.
    with pytest.raises(ValueError, match="/tmp"):
        ensure_docker_compose_plugin(Path("/tmp/bad_docker_cfg"))
    with pytest.raises(ValueError, match="/tmp"):
        smoke_compose_teardown(scratch_root=Path("/tmp/bad_smoke_scratch"))
    with pytest.raises(ValueError, match="/tmp"):
        sc.check_mcp_sidecar_http({}, Path("/tmp/bad_sidecar_smoke"))
    with pytest.raises(ValueError, match="/tmp"):
        sc.run_all_smoke_checks(Path("/tmp/bad_all_smoke"))
    with pytest.raises(ValueError, match="/tmp"):
        compile_variant(
            spec,
            ARM_NONE,
            output_dir=Path("/tmp/bad_tasks_out"),
            exported_skills_dir=live_cfg.exported_skills_dir,
        )
    with pytest.raises(ValueError, match="/tmp"):
        rj.main(["--suites", "synth", "--arms", "00_none", "--output", "/tmp/bad_job.yaml"])
    with pytest.raises(ValueError, match="/tmp"):
        run_analysis_pipeline([tmp_path], output_dir=Path("/tmp/bad_analysis_out"))


def test_part3_opencode_mcp_timeout_safe_retries_and_pixi_run_job(tmp_path: Path) -> None:
    """Verify safe retry whitelist, opencode limit.context, tool-arm opencode.jsonc, and pixi run-job."""
    repo_root = rb_cfg.resolve_repo_root()
    pixi_doc = tomllib.loads((repo_root / "pixi.toml").read_text(encoding="utf-8"))
    assert pixi_doc["tasks"]["run-job"] == "python -B -m radio_bench.cli run-job"

    # 1. Safe retry.include_exceptions whitelist (NEVER NonZeroAgentExitCodeError or AgentTimeoutError)
    custom_models = ModelsConfig(
        models={
            "oc_custom": ModelSpec(
                key="oc_custom",
                harness="opencode",
                model_name="minimax/MiniMax-M2.5",
                context_window_tokens=196_608,
                opencode_config={
                    "provider": {
                        "minimax": {
                            "models": {
                                "MiniMax-M2.5": {
                                    "limit": {"output": 40_000},
                                    "options": {"temperature": 0.1},
                                }
                            }
                        }
                    },
                    "mcp": {
                        "should_be_stripped": {
                            "type": "remote",
                            "url": "http://mcp-servers:9999/mcp",
                        }
                    },
                },
            )
        },
        parity_pairs=(),
    )
    job_dict = compile_job_dict(
        job_name="safe_retry_test",
        suites=["synth"],
        arms=["00_none", "11_full"],
        model_keys=["oc_custom"],
        tasks_dir=tmp_path / "tasks",
        runs_dir=tmp_path / "runs",
        jobs_out_dir=tmp_path / "jobs",
        models_cfg=custom_models,
        env=_clean_bench_env(),
    )
    # Validate against Harbor 0.25.0 JobConfig Pydantic schema
    validated_job = JobConfig.model_validate(job_dict)
    assert validated_job.retry.max_retries == 2
    inc_exc = list(job_dict["retry"]["include_exceptions"])
    assert inc_exc == [
        "ApiRateLimitError",
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiConnectionClosedError",
        "ApiResponseStalledError",
        "NetworkConnectionError",
    ]
    assert "NonZeroAgentExitCodeError" not in inc_exc
    assert "AgentTimeoutError" not in inc_exc

    # 2. Verify opencode_config.provider.<prov>.models.<id>.limit + user deep-merge + zero mcp leak
    agent_entry = rj._build_agent_entry(
        custom_models.models["oc_custom"],
        "default",
        job_n_concurrent=2,
        env=_clean_bench_env(),
    )
    oc_cfg = agent_entry["kwargs"]["opencode_config"]
    assert "mcp" not in oc_cfg
    lim = oc_cfg["provider"]["minimax"]["models"]["MiniMax-M2.5"]["limit"]
    assert lim["context"] == 196_608
    # User override on limit.output wins while auto-populated limit.context is preserved
    assert lim["output"] == 40_000
    assert oc_cfg["provider"]["minimax"]["models"]["MiniMax-M2.5"]["options"] == {
        "temperature": 0.1
    }

    # Verify default output token cap min(32000, max(8192, ctx // 4)) when no override is supplied
    auto_spec = ModelSpec(
        key="oc_auto",
        harness="opencode",
        model_name="openrouter/qwen-coder",
        context_window_tokens=131_072,
    )
    auto_entry = rj._build_agent_entry(
        auto_spec, "default", job_n_concurrent=1, env=_clean_bench_env()
    )
    assert "name" not in auto_entry
    assert auto_entry["import_path"] == "radio_bench.agents:RadioBenchOpenCode"
    assert "env" not in auto_entry
    assert auto_entry["kwargs"]["opencode_config"] == {
        "provider": {
            "openrouter": {
                "models": {
                    "qwen-coder": {
                        "limit": {
                            "context": 131_072,
                            "output": 32_000,
                        }
                    }
                }
            }
        }
    }

    # 3. Verify build_tasks.py writes /home/analyst/.config/opencode/opencode.jsonc ONLY on tool arms
    custom_ports = {"ms-inspect": 9101, "ms-modify": 9102, "ms-create": 9103}

    for non_tool_arm in (ARM_NONE, ARM_SKILLS_RAW, ARM_SKILLS_CASA):
        compose_nt = bt._build_environment_compose(
            non_tool_arm,
            cpus=4,
            has_seed_workspace=(non_tool_arm in (ARM_SKILLS_RAW, ARM_SKILLS_CASA)),
            has_calibrators_ms=False,
            sidecar_image="localhost/radio-bench-mcp-sidecar:latest",
            mcp_ports=custom_ports,
        )
        cmd_nt = compose_nt["services"]["main"]["command"][-1]
        assert "opencode.jsonc" not in cmd_nt
        assert "1200000" not in cmd_nt

    for tool_arm in (ARM_TOOLS, ARM_FULL, ARM_ORCHESTRATED):
        compose_t = bt._build_environment_compose(
            tool_arm,
            cpus=4,
            has_seed_workspace=(tool_arm in (ARM_FULL, ARM_ORCHESTRATED)),
            has_calibrators_ms=False,
            sidecar_image="localhost/radio-bench-mcp-sidecar:latest",
            mcp_ports=custom_ports,
        )
        cmd_t = compose_t["services"]["main"]["command"][-1]
        assert "/home/analyst/.config/opencode/opencode.jsonc" in cmd_t
        assert '"timeout": 1200000' in cmd_t
        for srv_name, port in custom_ports.items():
            assert f"http://127.0.0.1:{port}/mcp" in cmd_t
            assert f'"{srv_name}"' in cmd_t


def test_opencode_max_output_tokens_and_experimental_env_cap(tmp_path: Path) -> None:
    """Verify ModelSpec.max_output_tokens TOML parsing and OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX."""
    from harbor.models.job.config import AgentConfig

    toml_file = tmp_path / "models_max_out.toml"
    toml_file.write_text(
        "\n".join(
            [
                "[models.reasoning_large]",
                'harness = "opencode"',
                'model_name = "tejas/MiniMax-M3"',
                'api_format = "openai_chat_completions"',
                'base_url = "https://example.org/v1"',
                'api_key_env = "TACC_API_KEY"',
                "context_window_tokens = 1048576",
                "max_output_tokens = 131072",
                "",
                "[models.local_small]",
                'harness = "opencode"',
                'model_name = "local/qwen"',
                'base_url = "http://host.containers.internal:8080/v1"',
                'api_key_env = "LOCAL_LLM_API_KEY"',
                "host_gateway_required = true",
                "context_window_tokens = 65536",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_models_config(config_path=toml_file, env=_clean_bench_env())
    m_large = loaded.models["reasoning_large"]
    assert m_large.max_output_tokens == 131_072
    entry_large = rj._build_agent_entry(
        m_large, "default", job_n_concurrent=2, env={"TACC_API_KEY": "dummy"}
    )
    validated_large = AgentConfig.model_validate(entry_large)
    assert validated_large.env == {"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "131072"}
    assert entry_large["kwargs"]["opencode_config"]["provider"]["tejas"]["models"]["MiniMax-M3"][
        "limit"
    ] == {"context": 1_048_576, "output": 131_072}

    # <= 32000 output cap does NOT inject OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX
    m_small = loaded.models["local_small"]
    assert m_small.max_output_tokens is None
    entry_small = rj._build_agent_entry(
        m_small, "default", job_n_concurrent=1, env=_clean_bench_env()
    )
    validated_small = AgentConfig.model_validate(entry_small)
    assert validated_small.env == {"LOCAL_LLM_API_KEY": "local-placeholder"}

    # host_gateway_required + max_output_tokens > 32000 preserves BOTH env keys
    m_both = ModelSpec(
        key="local_reasoning",
        harness="opencode",
        model_name="local/qwq",
        api_key_env="LOCAL_LLM_API_KEY",
        host_gateway_required=True,
        context_window_tokens=131_072,
        max_output_tokens=96_000,
    )
    entry_both = rj._build_agent_entry(
        m_both, "default", job_n_concurrent=1, env=_clean_bench_env()
    )
    validated_both = AgentConfig.model_validate(entry_both)
    assert validated_both.env == {
        "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "96000",
        "LOCAL_LLM_API_KEY": "local-placeholder",
    }

    # Invalid max_output_tokens (< 1) raises ValueError
    bad_toml = tmp_path / "bad_max_out.toml"
    bad_toml.write_text(
        "\n".join(
            [
                "[models.bad_m]",
                'harness = "opencode"',
                'model_name = "tejas/MiniMax-M3"',
                "max_output_tokens = 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_output_tokens"):
        load_models_config(config_path=bad_toml, env=_clean_bench_env())


def test_radiobench_opencode_factory_and_runtime_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RadioBenchOpenCode factory resolution, log pull, and retryable error mapping."""
    import asyncio
    from types import SimpleNamespace

    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.base import (
        ApiError,
        ApiInternalServerError,
        ApiRateLimitError,
        ApiResponseStalledError,
        NonZeroAgentExitCodeError,
    )
    from harbor.agents.installed.opencode import OpenCode
    from harbor.models.agent.context import AgentContext
    from harbor.models.job.config import AgentConfig

    from radio_bench.agents import RadioBenchOpenCode

    monkeypatch.setenv("TACC_API_KEY", "dummy-tacc-key")

    spec = ModelSpec(
        key="tejas_minimax_m3",
        harness="opencode",
        model_name="tejas/MiniMax-M3",
        api_format="openai_chat_completions",
        base_url="https://example.org/v1",
        api_key_env="TACC_API_KEY",
        opencode_version="1.19.5",
        context_window_tokens=1_048_576,
        max_output_tokens=16_384,
    )
    agent_dict = rj._build_agent_entry(spec, "default", job_n_concurrent=1)
    assert "name" not in agent_dict
    assert agent_dict["import_path"] == "radio_bench.agents:RadioBenchOpenCode"
    assert "env" not in agent_dict

    agent_cfg = AgentConfig.model_validate(agent_dict)
    assert agent_cfg.name is None
    assert agent_cfg.import_path == "radio_bench.agents:RadioBenchOpenCode"
    AgentFactory.run_preflight(agent_cfg)

    def _make_agent_and_env(
        case_name: str,
        log_payload: str,
    ) -> tuple[RadioBenchOpenCode, Any, list[tuple[str, Path]]]:
        logs_dir = tmp_path / f"logs_{case_name}"
        created = AgentFactory.create_agent_from_config(agent_cfg, logs_dir=logs_dir)
        assert isinstance(created, RadioBenchOpenCode)
        assert created.name() == "opencode"
        downloads: list[tuple[str, Path]] = []

        async def _fake_download_file(src: str, dst: Path) -> None:
            downloads.append((src, Path(dst)))
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_text(log_payload, encoding="utf-8")

        env = SimpleNamespace(download_file=_fake_download_file)
        return created, env, downloads

    # 1. Decoder collapse (\x0f loop on exit 0) -> ApiInternalServerError (retryable)
    collapse_log = (
        json.dumps({"type": "step_start"})
        + "\n"
        + json.dumps({"type": "reasoning", "part": {"text": "Thinking...\n" + ("\x0f" * 128)}})
        + "\n"
        + json.dumps(
            {
                "type": "step_finish",
                "part": {
                    "reason": "length",
                    "tokens": {"input": 1000, "output": 1, "reasoning": 31999, "total": 33000},
                },
            }
        )
        + "\n"
    )
    ag_col, env_col, dl_col = _make_agent_and_env("collapse", collapse_log)

    async def _ok_super_run(self: Any, instruction: str, environment: Any, context: Any) -> None:
        return None

    monkeypatch.setattr(OpenCode, "run", _ok_super_run)
    with pytest.raises(ApiInternalServerError, match="control_byte_loop"):
        asyncio.run(ag_col.run("do task", env_col, AgentContext()))
    assert len(dl_col) == 1
    assert dl_col[0][0].endswith("/opencode.txt")

    # 2. HTTP 500 where super().run() raised NonZeroAgentExitCodeError -> ApiInternalServerError
    err500_log = (
        json.dumps(
            {
                "type": "error",
                "error": {"data": {"statusCode": 500, "message": "Internal Server Error"}},
            }
        )
        + "\n"
    )
    ag_500, env_500, dl_500 = _make_agent_and_env("http500", err500_log)

    async def _fail_nonzero(self: Any, instruction: str, environment: Any, context: Any) -> None:
        raise NonZeroAgentExitCodeError("exit 1")

    monkeypatch.setattr(OpenCode, "run", _fail_nonzero)
    with pytest.raises(ApiInternalServerError, match="HTTP 500"):
        asyncio.run(ag_500.run("do task", env_500, AgentContext()))
    assert len(dl_500) == 1

    # 3. HTTP 504 where super().run() raised NonZeroAgentExitCodeError -> ApiResponseStalledError
    err504_log = (
        json.dumps(
            {
                "type": "error",
                "error": {"data": {"statusCode": 504, "message": "504 Gateway Time-out"}},
            }
        )
        + "\n"
    )
    ag_504, env_504, _ = _make_agent_and_env("http504", err504_log)
    with pytest.raises(ApiResponseStalledError, match="HTTP 504"):
        asyncio.run(ag_504.run("do task", env_504, AgentContext()))

    # 4. HTTP 429 where super().run() raised NonZeroAgentExitCodeError -> ApiRateLimitError
    err429_log = (
        json.dumps(
            {
                "type": "error",
                "error": {"data": {"statusCode": 429, "message": "Too Many Requests"}},
            }
        )
        + "\n"
    )
    ag_429, env_429, _ = _make_agent_and_env("http429", err429_log)
    with pytest.raises(ApiRateLimitError, match="HTTP 429"):
        asyncio.run(ag_429.run("do task", env_429, AgentContext()))

    # 5. HTTP 400 malformed tool call with x-ratelimit headers where Harbor falsely raised
    # ApiRateLimitError -> downgrades to plain NonZeroAgentExitCodeError (not ApiError)
    err400_log = (
        json.dumps(
            {
                "type": "error",
                "error": {
                    "name": "APIError",
                    "data": {
                        "statusCode": 400,
                        "message": "Error code: 400 - failed to parse tool call arguments",
                        "responseHeaders": {"x-ratelimit-remaining": "99"},
                    },
                },
            }
        )
        + "\n"
    )
    ag_400, env_400, _ = _make_agent_and_env("http400", err400_log)

    async def _false_ratelimit(self: Any, instruction: str, environment: Any, context: Any) -> None:
        raise ApiRateLimitError("matched x-ratelimit header")

    monkeypatch.setattr(OpenCode, "run", _false_ratelimit)
    with pytest.raises(NonZeroAgentExitCodeError, match="HTTP 400") as exc_400_info:
        asyncio.run(ag_400.run("do task", env_400, AgentContext()))
    assert not isinstance(exc_400_info.value, ApiError)

    # 6. Benign shutdown EPIPE (NonZeroAgentExitCodeError, zero API errors) -> swallowed cleanly
    epipe_log = (
        json.dumps({"type": "step_start"})
        + "\n"
        + json.dumps({"type": "text", "part": {"text": "Finished calibration cleanly."}})
        + "\n"
        + json.dumps({"type": "step_finish", "part": {"reason": "stop"}})
        + "\n"
        + "Error: EPIPE: broken pipe, write\n"
    )
    ag_ep, env_ep, _ = _make_agent_and_env("epipe", epipe_log)

    async def _fail_epipe(self: Any, instruction: str, environment: Any, context: Any) -> None:
        raise NonZeroAgentExitCodeError("stderr: EPIPE: broken pipe")

    monkeypatch.setattr(OpenCode, "run", _fail_epipe)
    assert asyncio.run(ag_ep.run("do task", env_ep, AgentContext())) is None

    # 7. Transient recovered HTTP 500 followed by clean stop on exit 0 -> returns cleanly
    recovered_log = (
        json.dumps(
            {
                "type": "error",
                "error": {"data": {"statusCode": 500, "message": "Internal Server Error"}},
            }
        )
        + "\n"
        + json.dumps({"type": "step_start"})
        + "\n"
        + json.dumps({"type": "text", "part": {"text": "Recovered and completed."}})
        + "\n"
        + json.dumps({"type": "step_finish", "part": {"reason": "stop"}})
        + "\n"
    )
    ag_rec, env_rec, _ = _make_agent_and_env("recovered", recovered_log)
    monkeypatch.setattr(OpenCode, "run", _ok_super_run)
    assert asyncio.run(ag_rec.run("do task", env_rec, AgentContext())) is None

    # 8. Normal clean exit 0 -> returns cleanly
    clean_log = (
        json.dumps({"type": "step_start"})
        + "\n"
        + json.dumps({"type": "text", "part": {"text": "All deliverables verified."}})
        + "\n"
        + json.dumps({"type": "step_finish", "part": {"reason": "stop"}})
        + "\n"
    )
    ag_ok, env_ok, dl_ok = _make_agent_and_env("clean", clean_log)
    assert asyncio.run(ag_ok.run("do task", env_ok, AgentContext())) is None
    assert len(dl_ok) == 1

    # 9. Recovered mid-session 503 + clean stop where OpenCode._error_messages() raised -> swallowed
    ag_rec_exc, env_rec_exc, _ = _make_agent_and_env("recovered_emitted", recovered_log)

    async def _fail_emitted(self: Any, instruction: str, environment: Any, context: Any) -> None:
        raise NonZeroAgentExitCodeError("OpenCode emitted error event(s): Internal Server Error")

    monkeypatch.setattr(OpenCode, "run", _fail_emitted)
    assert asyncio.run(ag_rec_exc.run("do task", env_rec_exc, AgentContext())) is None

    # 10. Recovered mid-session 503 + clean stop + shutdown EPIPE -> swallowed
    rec_epipe_log = recovered_log + "error: EPIPE: broken pipe, write\n"
    ag_rec_ep, env_rec_ep, _ = _make_agent_and_env("recovered_epipe", rec_epipe_log)
    monkeypatch.setattr(OpenCode, "run", _fail_epipe)
    assert asyncio.run(ag_rec_ep.run("do task", env_rec_ep, AgentContext())) is None

    # 11. Fatal provider_api_error (HTTP 401 on exit 0; HTTP 422 mislabeled as ApiRateLimitError)
    err401_log = (
        json.dumps(
            {
                "type": "error",
                "error": {"data": {"statusCode": 401, "message": "Invalid API key"}},
            }
        )
        + "\n"
    )
    ag_401, env_401, _ = _make_agent_and_env("http401", err401_log)
    monkeypatch.setattr(OpenCode, "run", _ok_super_run)
    with pytest.raises(NonZeroAgentExitCodeError, match="HTTP 401") as exc_401_info:
        asyncio.run(ag_401.run("do task", env_401, AgentContext()))
    assert not isinstance(exc_401_info.value, ApiError)

    err422_log = (
        json.dumps(
            {
                "type": "error",
                "error": {
                    "data": {
                        "statusCode": 422,
                        "message": "Unprocessable Entity: context length exceeded",
                        "responseHeaders": {"x-ratelimit-remaining": "50"},
                    }
                },
            }
        )
        + "\n"
    )
    ag_422, env_422, _ = _make_agent_and_env("http422", err422_log)
    monkeypatch.setattr(OpenCode, "run", _false_ratelimit)
    with pytest.raises(NonZeroAgentExitCodeError, match="HTTP 422") as exc_422_info:
        asyncio.run(ag_422.run("do task", env_422, AgentContext()))
    assert not isinstance(exc_422_info.value, ApiError)

    # 12. Unmounted pre-existing stale host_log (--resume) is refreshed; failed refresh preserves it
    ag_res, env_res, dl_res = _make_agent_and_env("resume_refresh", collapse_log)
    ag_res.logs_dir.mkdir(parents=True, exist_ok=True)
    stale_host_log = ag_res.logs_dir / ag_res._OUTPUT_FILENAME
    stale_host_log.write_text(clean_log, encoding="utf-8")
    env_res.is_path_mounted = lambda _remote, _local: False
    monkeypatch.setattr(OpenCode, "run", _ok_super_run)
    with pytest.raises(ApiInternalServerError, match="control_byte_loop"):
        asyncio.run(ag_res.run("resume task", env_res, AgentContext()))
    assert len(dl_res) == 1

    # Failed download_file preserves pre-existing host_log and still classifies it
    ag_keep_logs = tmp_path / "logs_keep_existing"
    ag_keep = AgentFactory.create_agent_from_config(agent_cfg, logs_dir=ag_keep_logs)
    assert isinstance(ag_keep, RadioBenchOpenCode)
    ag_keep_logs.mkdir(parents=True, exist_ok=True)
    (ag_keep_logs / ag_keep._OUTPUT_FILENAME).write_text(collapse_log, encoding="utf-8")

    async def _broken_download(src: str, dst: Path) -> None:
        raise OSError("container stopped")

    env_broken = SimpleNamespace(
        download_file=_broken_download,
        is_path_mounted=lambda _remote, _local: False,
    )
    with pytest.raises(ApiInternalServerError, match="control_byte_loop"):
        asyncio.run(ag_keep.run("resume task", env_broken, AgentContext()))
    assert (ag_keep_logs / ag_keep._OUTPUT_FILENAME).read_text(encoding="utf-8") == collapse_log

    # 13. Missing host_log + download_file failure still classifies caught_exc unconditionally
    ag_nolog = AgentFactory.create_agent_from_config(agent_cfg, logs_dir=tmp_path / "logs_nolog")
    assert isinstance(ag_nolog, RadioBenchOpenCode)

    async def _false_rl_400(self: Any, instruction: str, environment: Any, context: Any) -> None:
        raise ApiRateLimitError("Error code: 400 - failed to parse tool call")

    monkeypatch.setattr(OpenCode, "run", _false_rl_400)
    with pytest.raises(NonZeroAgentExitCodeError, match="HTTP 400") as exc_nolog_info:
        asyncio.run(ag_nolog.run("do task", env_broken, AgentContext()))
    assert not isinstance(exc_nolog_info.value, ApiError)
