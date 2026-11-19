"""Dynamic Harbor 0.25.0 job YAML compiler from bench.toml + configs/models.toml."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from harbor.models.job.config import JobConfig

from radio_bench.build_tasks import (
    ARM_FULL,
    ARM_NONE,
    ARM_ORCHESTRATED,
    ARM_SKILLS_CASA,
    ARM_SKILLS_RAW,
    ARM_TOOLS,
    format_variant_name,
)
from radio_bench.config import (
    BenchConfig,
    ModelsConfig,
    ModelSpec,
    add_config_cli_args,
    config_from_cli_args,
    load_bench_config,
    load_models_config,
    reject_slash_tmp,
)

SUITE_STAGE_TASKS: dict[str, tuple[str, ...]] = {
    "synth": ("s1_inspect", "s1_precal", "s1_calibrate", "s1_image"),
    "3c391": (
        "r1_3c391_inspect",
        "r1_3c391_precal",
        "r1_3c391_calibrate",
        "r1_3c391_image",
    ),
}

SUITE_E2E_TASKS: dict[str, tuple[str, ...]] = {
    "synth": ("s2_e2e_synth",),
    "3c391": ("r2_3c391_e2e",),
}

SUITE_DEFAULT_CONCURRENCY: dict[str, int] = {
    "synth": 2,
    "3c391": 1,
}

STAGE_ABLATION_ARMS: tuple[str, ...] = (
    ARM_NONE,
    ARM_SKILLS_RAW,
    ARM_SKILLS_CASA,
    ARM_TOOLS,
    ARM_FULL,
)

E2E_ABLATION_ARMS: tuple[str, ...] = (
    ARM_NONE,
    ARM_SKILLS_RAW,
    ARM_SKILLS_CASA,
    ARM_TOOLS,
    ARM_FULL,
    ARM_ORCHESTRATED,
)

DEFAULT_ABLATION_MODELS: tuple[str, ...] = (
    "claude_haiku_4_5",
    "claude_sonnet_4_6",
    "nrp_kimi",
    "nrp_glm5",
)

RETRYABLE_API_EXCEPTIONS: tuple[str, ...] = (
    "ApiRateLimitError",
    "ApiInternalServerError",
    "ApiOverloadedError",
    "ApiConnectionClosedError",
    "ApiResponseStalledError",
    "NetworkConnectionError",
)


@dataclass(frozen=True)
class JobPreset:
    """Canonical benchmark job preset specification."""

    name: str
    suites: tuple[str, ...]
    task_scope: str  # "all", "stage", or "e2e"
    arms: tuple[str, ...]
    oracle: bool
    model_keys: tuple[str, ...]
    thinking_levels: tuple[str, ...] | None
    n_attempts: int
    suite_concurrency: int
    max_retries: int


CANONICAL_PRESETS: dict[str, JobPreset] = {
    "oracle_synth": JobPreset(
        name="oracle_synth",
        suites=("synth",),
        task_scope="all",
        arms=(ARM_NONE,),
        oracle=True,
        model_keys=(),
        thinking_levels=None,
        n_attempts=1,
        suite_concurrency=2,
        max_retries=0,
    ),
    "oracle_3c391": JobPreset(
        name="oracle_3c391",
        suites=("3c391",),
        task_scope="all",
        arms=(ARM_NONE,),
        oracle=True,
        model_keys=(),
        thinking_levels=None,
        n_attempts=1,
        suite_concurrency=1,
        max_retries=0,
    ),
    "suite_s_ablation": JobPreset(
        name="suite_s_ablation",
        suites=("synth",),
        task_scope="stage",
        arms=STAGE_ABLATION_ARMS,
        oracle=False,
        model_keys=DEFAULT_ABLATION_MODELS,
        thinking_levels=("default",),
        n_attempts=5,
        suite_concurrency=2,
        max_retries=2,
    ),
    "suite_s_e2e_ablation": JobPreset(
        name="suite_s_e2e_ablation",
        suites=("synth",),
        task_scope="e2e",
        arms=E2E_ABLATION_ARMS,
        oracle=False,
        model_keys=DEFAULT_ABLATION_MODELS,
        thinking_levels=("default",),
        n_attempts=3,
        suite_concurrency=2,
        max_retries=2,
    ),
    "suite_r_ablation": JobPreset(
        name="suite_r_ablation",
        suites=("3c391",),
        task_scope="stage",
        arms=STAGE_ABLATION_ARMS,
        oracle=False,
        model_keys=DEFAULT_ABLATION_MODELS,
        thinking_levels=("default",),
        n_attempts=5,
        suite_concurrency=1,
        max_retries=2,
    ),
    "suite_r_e2e_ablation": JobPreset(
        name="suite_r_e2e_ablation",
        suites=("3c391",),
        task_scope="e2e",
        arms=E2E_ABLATION_ARMS,
        oracle=False,
        model_keys=DEFAULT_ABLATION_MODELS,
        thinking_levels=("default",),
        n_attempts=3,
        suite_concurrency=1,
        max_retries=2,
    ),
}


def write_host_gateway_compose(jobs_dir: Path) -> Path:
    """Write Podman pasta host-gateway Compose overlay for local host inference servers."""
    reject_slash_tmp(jobs_dir, label="jobs_dir")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = jobs_dir / "host_gateway_compose.yaml"
    overlay_spec = {
        "services": {
            "main": {
                "extra_hosts": ["host.containers.internal:host-gateway"],
            }
        }
    }
    overlay_path.write_text(yaml.safe_dump(overlay_spec, sort_keys=False), encoding="utf-8")
    return overlay_path


def _resolve_base_tasks(suites: Sequence[str], task_scope: str) -> list[tuple[str, bool]]:
    """Return ordered `(base_task_id, is_e2e)` pairs for requested suites and scope."""
    if task_scope not in {"all", "stage", "e2e"}:
        raise ValueError(f"Unsupported task_scope={task_scope!r}")
    out: list[tuple[str, bool]] = []
    for suite in suites:
        if suite not in SUITE_STAGE_TASKS:
            raise ValueError(
                f"Unknown suite {suite!r}; expected one of {sorted(SUITE_STAGE_TASKS)}"
            )
        if task_scope in {"all", "stage"}:
            for tid in SUITE_STAGE_TASKS[suite]:
                out.append((tid, False))
        if task_scope in {"all", "e2e"}:
            for tid in SUITE_E2E_TASKS[suite]:
                out.append((tid, True))
    return out


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {str(k): v for k, v in base.items()}
    for k, v in override.items():
        ks = str(k)
        if isinstance(v, Mapping) and isinstance(out.get(ks), Mapping):
            out[ks] = _deep_merge_dicts(out[ks], v)
        elif isinstance(v, Mapping):
            out[ks] = _deep_merge_dicts({}, v)
        else:
            out[ks] = v
    return out


def _build_single_agent_dict(
    model: ModelSpec,
    thinking_level: str = "default",
    *,
    job_n_concurrent: int = 1,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Translate one `(ModelSpec, thinking_level)` into a Harbor 0.25.0 AgentConfig dict."""
    eff_env = os.environ if env is None else env
    resolved_base_url = model.base_url or ""
    if not resolved_base_url and model.base_url_env:
        resolved_base_url = eff_env.get(model.base_url_env, "").strip()
        if not resolved_base_url:
            raise ValueError(
                f"Model {model.key!r} requires environment variable {model.base_url_env} "
                "(or base_url in models.local.toml) to render job configuration"
            )

    if model.harness == "claude-code":
        agent_dict: dict[str, Any] = {
            "name": "claude-code",
            "model_name": model.model_name,
        }
        if resolved_base_url:
            agent_dict["base_url"] = resolved_base_url
            if model.api_key_env:
                agent_dict["model_api_key_name"] = model.api_key_env
            if model.api_format:
                agent_dict["model_api_format"] = model.api_format
        kwargs: dict[str, Any] = {
            "disable_web_search": bool(model.disable_web_search),
        }
        if thinking_level != "default":
            kwargs["reasoning_effort"] = thinking_level
        agent_dict["kwargs"] = kwargs
    elif model.harness == "opencode":
        agent_dict = {
            "name": "opencode",
            "model_name": model.model_name,
        }
        if resolved_base_url:
            agent_dict["base_url"] = resolved_base_url
        if model.api_key_env:
            agent_dict["model_api_key_name"] = model.api_key_env
        if model.api_format:
            agent_dict["model_api_format"] = model.api_format
        kwargs = {}
        if model.opencode_version:
            kwargs["version"] = model.opencode_version
        kwargs["disable_web_search"] = bool(model.disable_web_search)
        if thinking_level != "default":
            kwargs["variant"] = thinking_level

        oc_cfg: dict[str, Any] = {}
        if int(model.context_window_tokens) > 0 and "/" in model.model_name:
            provider_id, model_id = model.model_name.split("/", 1)
            if provider_id.strip() and model_id.strip():
                ctx_tokens = int(model.context_window_tokens)
                out_tokens = min(65536, max(8192, ctx_tokens // 4))
                oc_cfg = {
                    "provider": {
                        provider_id.strip(): {
                            "models": {
                                model_id.strip(): {
                                    "limit": {
                                        "context": ctx_tokens,
                                        "output": out_tokens,
                                    }
                                }
                            }
                        }
                    }
                }
        user_oc_cfg = getattr(model, "opencode_config", None)
        if isinstance(user_oc_cfg, Mapping) and user_oc_cfg:
            oc_cfg = _deep_merge_dicts(oc_cfg, user_oc_cfg)
        # Never inject MCP definitions into job-level opencode_config (those belong only on tool arms).
        oc_cfg.pop("mcp", None)
        if oc_cfg:
            kwargs["opencode_config"] = oc_cfg

        agent_dict["kwargs"] = kwargs
    else:
        raise ValueError(f"Unsupported harness {model.harness!r} on model {model.key!r}")

    if model.max_concurrent_trials is not None:
        agent_dict["n_concurrent"] = min(int(model.max_concurrent_trials), int(job_n_concurrent))

    if (
        model.host_gateway_required
        and model.api_key_env
        and not eff_env.get(model.api_key_env, "").strip()
    ):
        agent_dict["env"] = {model.api_key_env: "local-placeholder"}

    return agent_dict


_build_agent_entry = _build_single_agent_dict


def compile_job_dict(
    *,
    job_name: str,
    suites: Sequence[str],
    arms: Sequence[str],
    task_scope: str = "all",
    oracle: bool = False,
    model_keys: Sequence[str] = DEFAULT_ABLATION_MODELS,
    thinking_levels: Sequence[str] | None = None,
    pack_ids: Sequence[str] | None = None,
    force_pack_in_name: bool = False,
    n_attempts: int = 1,
    suite_concurrency: int | None = None,
    max_retries: int | None = None,
    tasks_dir: Path | None = None,
    runs_dir: Path | None = None,
    jobs_out_dir: Path | None = None,
    cfg: BenchConfig | None = None,
    models_cfg: ModelsConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build and validate a Harbor 0.25.0 JobConfig dictionary."""
    eff_env = os.environ if env is None else env
    resolved_cfg = cfg if cfg is not None else load_bench_config(env=eff_env)
    resolved_models = models_cfg if models_cfg is not None else load_models_config(env=eff_env)

    eff_tasks_dir = tasks_dir if tasks_dir is not None else resolved_cfg.tasks_dir
    eff_runs_dir = runs_dir if runs_dir is not None else resolved_cfg.runs_dir
    eff_jobs_dir = jobs_out_dir if jobs_out_dir is not None else resolved_cfg.jobs_dir

    reject_slash_tmp(eff_tasks_dir, label="tasks_dir")
    reject_slash_tmp(eff_runs_dir, label="runs_dir")
    reject_slash_tmp(eff_jobs_dir, label="jobs_dir")

    eff_packs = (
        [p.strip() for p in pack_ids if p.strip()]
        if pack_ids is not None and len(pack_ids) > 0
        else [resolved_cfg.default_pack]
    )
    for pid in eff_packs:
        resolved_cfg.get_pack(pid)
    include_pack = bool(force_pack_in_name or len(eff_packs) > 1)

    if suite_concurrency is not None:
        base_conc = max(1, int(suite_concurrency))
    else:
        base_conc = min(SUITE_DEFAULT_CONCURRENCY.get(s, 2) for s in suites)

    selected_models: list[ModelSpec] = []
    if not oracle:
        for mkey in model_keys:
            if mkey not in resolved_models.models:
                available = ", ".join(sorted(resolved_models.models))
                raise KeyError(f"Unknown model key {mkey!r}; configured models: {available}")
            selected_models.append(resolved_models.models[mkey])

    model_caps = [
        int(m.max_concurrent_trials)
        for m in selected_models
        if m.max_concurrent_trials is not None and m.max_concurrent_trials > 0
    ]
    n_concurrent_trials = max(1, min([base_conc, *model_caps]))

    agents_list: list[dict[str, Any]] = []
    if oracle:
        agents_list.append({"name": "oracle"})
    else:
        allow_all_levels = thinking_levels is None or "all" in thinking_levels
        for m in selected_models:
            levels = (
                list(m.thinking_levels)
                if allow_all_levels
                else [lvl for lvl in m.thinking_levels if lvl in thinking_levels]  # type: ignore[operator]
            )
            for lvl in levels:
                agents_list.append(
                    _build_single_agent_dict(
                        m,
                        lvl,
                        job_n_concurrent=n_concurrent_trials,
                        env=eff_env,
                    )
                )
        if not agents_list:
            raise ValueError(
                f"No agent configurations matched models={list(model_keys)} and "
                f"thinking_levels={list(thinking_levels) if thinking_levels else None}"
            )

    base_tasks = _resolve_base_tasks(suites, task_scope)
    task_entries: list[dict[str, str]] = []
    for base_tid, is_e2e in base_tasks:
        for pid in eff_packs:
            for arm in arms:
                if arm == ARM_ORCHESTRATED and not is_e2e:
                    continue
                vname = format_variant_name(
                    base_tid,
                    arm,
                    pack_id=pid,
                    include_pack_in_name=include_pack,
                )
                task_entries.append({"path": str(eff_tasks_dir / vname)})

    if not task_entries:
        raise ValueError(f"Job {job_name!r} resolved zero tasks")

    eff_retries = (0 if oracle else 2) if max_retries is None else int(max_retries)
    retry_dict: dict[str, Any] = {"max_retries": eff_retries}
    if eff_retries > 0:
        retry_dict.update(
            {
                "min_wait_sec": 5.0,
                "max_wait_sec": 60.0,
                "wait_multiplier": 2.0,
                "include_exceptions": list(RETRYABLE_API_EXCEPTIONS),
            }
        )

    env_dict: dict[str, Any] = {
        "type": "podman",
        "delete": True,
    }
    if any(m.host_gateway_required for m in selected_models):
        overlay_path = write_host_gateway_compose(eff_jobs_dir)
        env_dict["extra_docker_compose"] = [str(overlay_path)]

    job_dict: dict[str, Any] = {
        "job_name": job_name,
        "jobs_dir": str(eff_runs_dir),
        "n_attempts": int(n_attempts),
        "n_concurrent_trials": n_concurrent_trials,
        "retry": retry_dict,
        "environment": env_dict,
        "agents": agents_list,
        "tasks": task_entries,
    }

    # Strict Harbor 0.25.0 Pydantic validation prior to returning.
    JobConfig.model_validate(job_dict)
    return job_dict


def render_preset_job(
    preset_name: str,
    *,
    output_dir: Path | None = None,
    pack_ids: Sequence[str] | None = None,
    force_pack_in_name: bool = False,
    cfg: BenchConfig | None = None,
    models_cfg: ModelsConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Render one canonical preset YAML file into `<work_dir>/jobs/<preset>.yaml`."""
    if preset_name not in CANONICAL_PRESETS:
        raise KeyError(
            f"Unknown preset {preset_name!r}; available presets: {sorted(CANONICAL_PRESETS)}"
        )
    preset = CANONICAL_PRESETS[preset_name]
    eff_env = os.environ if env is None else env
    resolved_cfg = cfg if cfg is not None else load_bench_config(env=eff_env)
    resolved_models = models_cfg if models_cfg is not None else load_models_config(env=eff_env)
    target_dir = output_dir if output_dir is not None else resolved_cfg.jobs_dir
    reject_slash_tmp(target_dir, label="output_dir")
    target_dir.mkdir(parents=True, exist_ok=True)

    job_dict = compile_job_dict(
        job_name=preset.name,
        suites=preset.suites,
        arms=preset.arms,
        task_scope=preset.task_scope,
        oracle=preset.oracle,
        model_keys=preset.model_keys,
        thinking_levels=preset.thinking_levels,
        pack_ids=pack_ids,
        force_pack_in_name=force_pack_in_name,
        n_attempts=preset.n_attempts,
        suite_concurrency=preset.suite_concurrency,
        max_retries=preset.max_retries,
        jobs_out_dir=target_dir,
        cfg=resolved_cfg,
        models_cfg=resolved_models,
        env=eff_env,
    )
    out_path = target_dir / f"{preset.name}.yaml"
    out_path.write_text(yaml.safe_dump(job_dict, sort_keys=False), encoding="utf-8")
    return out_path


def render_all_presets(
    *,
    output_dir: Path | None = None,
    pack_ids: Sequence[str] | None = None,
    force_pack_in_name: bool = False,
    cfg: BenchConfig | None = None,
    models_cfg: ModelsConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Render all 6 canonical preset job YAML files under `<work_dir>/jobs/`."""
    eff_env = os.environ if env is None else env
    resolved_cfg = cfg if cfg is not None else load_bench_config(env=eff_env)
    resolved_models = models_cfg if models_cfg is not None else load_models_config(env=eff_env)
    written: dict[str, Path] = {}
    for preset_name in CANONICAL_PRESETS:
        written[preset_name] = render_preset_job(
            preset_name,
            output_dir=output_dir,
            pack_ids=pack_ids,
            force_pack_in_name=force_pack_in_name,
            cfg=resolved_cfg,
            models_cfg=resolved_models,
            env=eff_env,
        )
    return written


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render validated Harbor 0.25.0 job YAML files into <work_dir>/jobs/."
    )
    add_config_cli_args(parser)
    parser.add_argument(
        "--models-config",
        type=Path,
        default=None,
        help="Explicit models.toml path.",
    )
    parser.add_argument(
        "--preset",
        choices=("all", *CANONICAL_PRESETS.keys()),
        default=None,
        help="Canonical job preset to render (default: all when no custom --suites/--models given).",
    )
    parser.add_argument("--job-name", type=str, default=None, help="Custom job_name.")
    parser.add_argument(
        "--suites",
        type=str,
        default=None,
        help="Comma-separated suites for custom job (synth,3c391).",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "stage", "e2e"),
        default="all",
        help="Task scope for custom job (all, stage, or e2e).",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=None,
        help="Comma-separated capability arms for custom job.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys from configs/models.toml (or 'oracle').",
    )
    parser.add_argument(
        "--thinking-levels",
        type=str,
        default=None,
        help="Comma-separated thinking levels (e.g. default,high or all).",
    )
    parser.add_argument(
        "--packs",
        type=str,
        default=None,
        help="Comma-separated pack IDs.",
    )
    parser.add_argument(
        "--force-pack-in-name",
        action="store_true",
        help="Emit <task_id>__<pack_id>__<arm> task paths even for a single pack.",
    )
    parser.add_argument("--n-attempts", type=int, default=1, help="Attempts per cell.")
    parser.add_argument(
        "--n-concurrent-trials",
        type=int,
        default=None,
        help="Optional upper bound on concurrent trials.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for preset YAML files (default: <work_dir>/jobs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output YAML file path for a custom job.",
    )
    args = parser.parse_args(argv)

    cfg = config_from_cli_args(args)
    models_cfg = load_models_config(config_path=args.models_config, repo_root=cfg.repo_root)
    packs = _split_csv(args.packs) or None

    is_custom = any(x is not None for x in (args.suites, args.arms, args.models, args.job_name))
    if not is_custom or args.preset is not None:
        eff_preset = args.preset or "all"
        if eff_preset == "all":
            written = render_all_presets(
                output_dir=args.output_dir,
                pack_ids=packs,
                force_pack_in_name=args.force_pack_in_name,
                cfg=cfg,
                models_cfg=models_cfg,
            )
            print(
                json.dumps(
                    {"rendered_presets": {k: str(v) for k, v in written.items()}},
                    indent=2,
                )
            )
        else:
            out_file = render_preset_job(
                eff_preset,
                output_dir=args.output_dir,
                pack_ids=packs,
                force_pack_in_name=args.force_pack_in_name,
                cfg=cfg,
                models_cfg=models_cfg,
            )
            print(json.dumps({"rendered_presets": {eff_preset: str(out_file)}}, indent=2))
        return 0

    suites = _split_csv(args.suites) or ["synth"]
    arms = _split_csv(args.arms) or list(E2E_ABLATION_ARMS)
    model_tokens = _split_csv(args.models)
    is_oracle = model_tokens == ["oracle"]
    eff_models = () if is_oracle else (tuple(model_tokens) or DEFAULT_ABLATION_MODELS)
    t_levels = _split_csv(args.thinking_levels) or None
    job_name = args.job_name or "custom_job"

    if args.output is not None:
        reject_slash_tmp(args.output, label="output")
    target_dir = (
        args.output.parent if args.output is not None else (args.output_dir or cfg.jobs_dir)
    )
    reject_slash_tmp(target_dir, label="output_dir")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output if args.output is not None else (target_dir / f"{job_name}.yaml")

    job_dict = compile_job_dict(
        job_name=job_name,
        suites=suites,
        arms=arms,
        task_scope=args.scope,
        oracle=is_oracle,
        model_keys=eff_models,
        thinking_levels=t_levels,
        pack_ids=packs,
        force_pack_in_name=args.force_pack_in_name,
        n_attempts=args.n_attempts,
        suite_concurrency=args.n_concurrent_trials,
        jobs_out_dir=target_dir,
        cfg=cfg,
        models_cfg=models_cfg,
    )
    out_path.write_text(yaml.safe_dump(job_dict, sort_keys=False), encoding="utf-8")
    print(json.dumps({"rendered_job": str(out_path), "job_name": job_name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
