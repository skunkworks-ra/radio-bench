"""List-price token cost model and Claude Haiku 100K 5x cliff accounting."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from radio_bench.config import ModelsConfig, load_models_config, resolve_repo_root

HAIKU_CLIFF_TOKEN_THRESHOLD = 100_000
HAIKU_CLIFF_MULTIPLIER = 5.0


@dataclass(frozen=True)
class ModelRate:
    """Per-million-token USD list-price schedule for one model family."""

    canonical_model: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_5m_per_mtok: float
    cache_write_1h_per_mtok: float
    cliff_prompt_tokens: int | None = None
    cliff_multiplier: float = 1.0


ORACLE_MODEL_RATE = ModelRate(
    canonical_model="oracle",
    input_per_mtok=0.0,
    output_per_mtok=0.0,
    cache_read_per_mtok=0.0,
    cache_write_5m_per_mtok=0.0,
    cache_write_1h_per_mtok=0.0,
    cliff_prompt_tokens=None,
    cliff_multiplier=1.0,
)


def build_model_rates(
    models_cfg: ModelsConfig | None = None,
) -> tuple[dict[str, ModelRate], dict[str, str]]:
    """Build `(rates_by_canonical_model, alias_to_canonical)` from ModelsConfig."""
    cfg = models_cfg if models_cfg is not None else load_models_config()
    rates: dict[str, ModelRate] = {"oracle": ORACLE_MODEL_RATE}
    aliases: dict[str, str] = {"oracle": "oracle"}
    for mkey, spec in cfg.models.items():
        canon = spec.model_name.strip().lower()
        p = spec.pricing
        rates[canon] = ModelRate(
            canonical_model=canon,
            input_per_mtok=p.input_per_mtok,
            output_per_mtok=p.output_per_mtok,
            cache_read_per_mtok=p.cache_read_per_mtok,
            cache_write_5m_per_mtok=p.cache_write_5m_per_mtok,
            cache_write_1h_per_mtok=p.cache_write_1h_per_mtok,
            cliff_prompt_tokens=p.cliff_prompt_tokens,
            cliff_multiplier=p.cliff_multiplier,
        )
        aliases[mkey.strip().lower()] = canon
        aliases[canon] = canon
        for al in spec.aliases:
            aliases[al.strip().lower()] = canon
    return rates, aliases


_RATES_CACHE: dict[tuple[Any, ...], tuple[dict[str, ModelRate], dict[str, str]]] = {}


def _stat_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _current_rates_and_aliases(
    models_cfg: ModelsConfig | None = None,
) -> tuple[dict[str, ModelRate], dict[str, str]]:
    if models_cfg is not None:
        return build_model_rates(models_cfg)

    env_models = os.environ.get("RADIO_BENCH_MODELS_CONFIG", "").strip()
    env_repo = os.environ.get("RADIO_BENCH_REPO_ROOT", "").strip()
    if env_models:
        explicit = Path(os.path.expanduser(env_models))
        key: tuple[Any, ...] = ("explicit", str(explicit), _stat_mtime_ns(explicit))
    else:
        repo = resolve_repo_root()
        c1 = repo / "configs" / "models.toml"
        c2 = repo / "configs" / "models.local.toml"
        c3 = repo / "models.local.toml"
        key = (
            "repo",
            env_repo,
            str(repo),
            _stat_mtime_ns(c1),
            _stat_mtime_ns(c2),
            _stat_mtime_ns(c3),
        )

    cached = _RATES_CACHE.get(key)
    if cached is None:
        cached = build_model_rates()
        _RATES_CACHE[key] = cached
    return cached


def __getattr__(name: str) -> Any:
    if name == "MODEL_RATES":
        rates, _ = _current_rates_and_aliases()
        return rates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class TurnPricing:
    """Token and list-price USD breakdown for a single LLM inference turn."""

    model_name: str
    input_tokens: int
    uncached_input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int
    output_tokens: int
    input_usd: float
    cache_write_usd: float
    cache_read_usd: float
    output_usd: float
    total_usd: float
    cliff_triggered: bool
    rate_known: bool = True
    used_fallback_cost: bool = False


@dataclass(frozen=True)
class TrialPricing:
    """Structured per-trial token and list-price USD breakdown."""

    input_tokens: int
    uncached_input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    total_tokens: int
    peak_turn_tokens: int
    haiku_cliff_triggered: bool
    cliff_turns: int
    input_usd: float
    cache_write_usd: float
    cache_read_usd: float
    output_usd: float
    total_usd: float
    pricing_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_model_key(
    model_name: str | None,
    agent_name: str | None = None,
    *,
    models_cfg: ModelsConfig | None = None,
) -> str:
    """Resolve provider/date-suffixed model identifiers to canonical rate keys."""
    if (agent_name or "").strip().lower() == "oracle":
        return "oracle"
    raw = (model_name or "").strip().lower()
    if not raw or raw == "oracle":
        return "oracle" if (agent_name or "").strip().lower() == "oracle" else raw

    rates, aliases = _current_rates_and_aliases(models_cfg)

    if raw in rates:
        return raw
    if raw in aliases:
        return aliases[raw]

    provider, bare = raw.split("/", 1) if "/" in raw else ("", raw)
    bare = bare.split("@", 1)[0]
    if bare in aliases and not provider:
        return aliases[bare]

    if "haiku-4-5" in bare or "haiku-4.5" in bare or bare.startswith("claude-haiku"):
        return "anthropic/claude-haiku-4-5"
    if (
        "sonnet-4-6" in bare
        or "sonnet-4.6" in bare
        or "sonnet-4-5" in bare
        or bare.startswith("claude-sonnet")
    ):
        return "anthropic/claude-sonnet-4-6"
    if "kimi" in bare:
        if provider == "tacc" and "tacc/kimi" in rates:
            return "tacc/kimi"
        if provider in ("", "nrp"):
            return "nrp/kimi"
    if ("glm-5" in bare or bare.startswith("glm")) and provider in ("", "nrp"):
        return "nrp/glm-5"
    if ("qwen" in bare) and provider == "local" and "local/qwen" in rates:
        return "local/qwen"
    return raw


def get_model_rate(
    model_name: str | None,
    agent_name: str | None = None,
    *,
    models_cfg: ModelsConfig | None = None,
) -> ModelRate | None:
    """Look up the list-price rate schedule for `model_name`, or None if unknown."""
    rates, _ = _current_rates_and_aliases(models_cfg)
    key = normalize_model_key(model_name, agent_name=agent_name, models_cfg=models_cfg)
    return rates.get(key)


def _safe_nonneg_int(val: Any) -> int:
    if isinstance(val, bool) or val is None:
        return 0
    if isinstance(val, (int, float)):
        if not math.isfinite(float(val)):
            return 0
        return max(0, int(round(float(val))))
    return 0


def _safe_nonneg_float(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        if math.isfinite(fval) and fval >= 0.0:
            return fval
    return None


def _resolve_pricing_source(default_source: str, turns: list[TurnPricing]) -> str:
    if all(t.rate_known for t in turns):
        return default_source
    unknown_turns = [t for t in turns if not t.rate_known]
    if any(t.used_fallback_cost for t in unknown_turns):
        return "agent_reported_cost"
    return "unknown_model"


def price_turn(
    model_name: str | None,
    *,
    uncached_input_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    output_tokens: int = 0,
    agent_name: str | None = None,
    fallback_cost_usd: float | None = None,
    allow_cliff: bool = True,
    models_cfg: ModelsConfig | None = None,
) -> TurnPricing:
    """Price a single LLM turn using list prices and configured prompt cliff rules."""
    u_in = _safe_nonneg_int(uncached_input_tokens)
    c_read = _safe_nonneg_int(cache_read_tokens)
    c_write = _safe_nonneg_int(cache_write_tokens)
    c_write_1h = min(_safe_nonneg_int(cache_write_1h_tokens), c_write)
    c_write_5m = c_write - c_write_1h
    out_tok = _safe_nonneg_int(output_tokens)
    total_in = u_in + c_read + c_write

    canonical = normalize_model_key(model_name, agent_name=agent_name, models_cfg=models_cfg)
    rate = get_model_rate(canonical, agent_name=agent_name, models_cfg=models_cfg)
    if rate is None:
        fb_val = _safe_nonneg_float(fallback_cost_usd)
        fb = fb_val if fb_val is not None else 0.0
        return TurnPricing(
            model_name=canonical or "unknown",
            input_tokens=total_in,
            uncached_input_tokens=u_in,
            cache_read_tokens=c_read,
            cache_write_tokens=c_write,
            cache_write_1h_tokens=c_write_1h,
            output_tokens=out_tok,
            input_usd=round(fb, 8),
            cache_write_usd=0.0,
            cache_read_usd=0.0,
            output_usd=0.0,
            total_usd=round(fb, 8),
            cliff_triggered=False,
            rate_known=False,
            used_fallback_cost=fb_val is not None,
        )

    cliff_hit = (
        allow_cliff
        and rate.cliff_prompt_tokens is not None
        and rate.cliff_multiplier > 1.0
        and total_in > rate.cliff_prompt_tokens
    )
    mult = rate.cliff_multiplier if cliff_hit else 1.0

    in_usd = (u_in * rate.input_per_mtok * mult) / 1_000_000.0
    cw_usd = (
        (c_write_5m * rate.cache_write_5m_per_mtok + c_write_1h * rate.cache_write_1h_per_mtok)
        * mult
    ) / 1_000_000.0
    cr_usd = (c_read * rate.cache_read_per_mtok * mult) / 1_000_000.0
    out_usd = (out_tok * rate.output_per_mtok * mult) / 1_000_000.0
    tot_usd = in_usd + cw_usd + cr_usd + out_usd

    return TurnPricing(
        model_name=rate.canonical_model,
        input_tokens=total_in,
        uncached_input_tokens=u_in,
        cache_read_tokens=c_read,
        cache_write_tokens=c_write,
        cache_write_1h_tokens=c_write_1h,
        output_tokens=out_tok,
        input_usd=in_usd,
        cache_write_usd=cw_usd,
        cache_read_usd=cr_usd,
        output_usd=out_usd,
        total_usd=tot_usd,
        cliff_triggered=cliff_hit,
        rate_known=True,
        used_fallback_cost=False,
    )


def _extract_step_token_counts(
    metrics_dict: dict[str, Any],
    *,
    agent_name: str | None = None,
) -> tuple[int, int, int, int, int, float | None]:
    """Extract `(uncached_in, cache_read, cache_write, cache_write_1h, output, fallback_usd)` from ATIF Metrics."""
    extra = metrics_dict.get("extra")
    extra_d = extra if isinstance(extra, dict) else {}

    raw_prompt = _safe_nonneg_int(metrics_dict.get("prompt_tokens"))
    out_tok = _safe_nonneg_int(metrics_dict.get("completion_tokens"))

    if "cache_read_input_tokens" in extra_d:
        cache_read = _safe_nonneg_int(extra_d.get("cache_read_input_tokens"))
    else:
        cache_read = _safe_nonneg_int(metrics_dict.get("cached_tokens"))

    has_claude_creation = "cache_creation_input_tokens" in extra_d
    has_opencode_write = ("cache_write_tokens" in extra_d) and not has_claude_creation
    if has_claude_creation:
        cache_write = _safe_nonneg_int(extra_d.get("cache_creation_input_tokens"))
    elif "cache_write_tokens" in extra_d:
        cache_write = _safe_nonneg_int(extra_d.get("cache_write_tokens"))
    else:
        cache_write = 0

    cache_creation_obj = extra_d.get("cache_creation")
    if isinstance(cache_creation_obj, dict):
        cache_write_1h = min(
            _safe_nonneg_int(cache_creation_obj.get("ephemeral_1h_input_tokens")),
            cache_write,
        )
    else:
        cache_write_1h = 0

    is_opencode = (agent_name or "").strip().lower() == "opencode"
    # In Harbor 0.25.0 ATIF:
    # - ClaudeCode sets prompt_tokens = uncached + cache_read + cache_creation.
    # - OpenCode sets prompt_tokens = uncached + cache_read (excluding cache_write in extra).
    if has_opencode_write:
        uncached_in = max(0, raw_prompt - cache_read)
    elif has_claude_creation:
        if raw_prompt >= cache_read + cache_write:
            uncached_in = raw_prompt - cache_read - cache_write
        else:
            uncached_in = max(0, raw_prompt - cache_read)
    elif not is_opencode and raw_prompt >= cache_read + cache_write:
        uncached_in = raw_prompt - cache_read - cache_write
    else:
        uncached_in = max(0, raw_prompt - cache_read)

    fallback_usd = _safe_nonneg_float(metrics_dict.get("cost_usd"))
    return uncached_in, cache_read, cache_write, cache_write_1h, out_tok, fallback_usd


def _count_trajectory_agent_steps(traj_dict: dict[str, Any] | None) -> int | None:
    if not isinstance(traj_dict, dict):
        return None
    has_any_steps_list = False
    count = 0

    def _walk(t_obj: dict[str, Any]) -> None:
        nonlocal has_any_steps_list, count
        steps = t_obj.get("steps")
        if isinstance(steps, list) and steps:
            has_any_steps_list = True
            for st in steps:
                if isinstance(st, dict) and st.get("source") == "agent":
                    count += 1
        sub_trajs = t_obj.get("subagent_trajectories")
        if isinstance(sub_trajs, list):
            for sub in sub_trajs:
                if isinstance(sub, dict):
                    _walk(sub)

    _walk(traj_dict)
    return count if has_any_steps_list else None


def _collect_trajectory_turns(
    traj_dict: dict[str, Any],
    default_model: str | None,
    agent_name: str | None,
    *,
    models_cfg: ModelsConfig | None = None,
) -> list[TurnPricing]:
    turns: list[TurnPricing] = []
    agent_block = traj_dict.get("agent")
    traj_model = (
        agent_block.get("model_name")
        if isinstance(agent_block, dict) and isinstance(agent_block.get("model_name"), str)
        else None
    ) or default_model
    traj_agent = (
        agent_block.get("name")
        if isinstance(agent_block, dict) and isinstance(agent_block.get("name"), str)
        else None
    ) or agent_name

    steps = traj_dict.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("source") != "agent":
                continue
            metrics = step.get("metrics")
            if not isinstance(metrics, dict):
                continue
            u_in, c_read, c_write, c_write_1h, out_tok, fb_usd = _extract_step_token_counts(
                metrics, agent_name=traj_agent
            )
            if u_in == 0 and c_read == 0 and c_write == 0 and out_tok == 0 and fb_usd is None:
                continue
            step_model = (
                step.get("model_name") if isinstance(step.get("model_name"), str) else None
            ) or traj_model
            turns.append(
                price_turn(
                    step_model,
                    uncached_input_tokens=u_in,
                    cache_read_tokens=c_read,
                    cache_write_tokens=c_write,
                    cache_write_1h_tokens=c_write_1h,
                    output_tokens=out_tok,
                    agent_name=traj_agent,
                    fallback_cost_usd=fb_usd,
                    models_cfg=models_cfg,
                )
            )

    sub_trajs = traj_dict.get("subagent_trajectories")
    if isinstance(sub_trajs, list):
        for sub in sub_trajs:
            if isinstance(sub, dict):
                turns.extend(
                    _collect_trajectory_turns(sub, traj_model, traj_agent, models_cfg=models_cfg)
                )
    return turns


def price_from_agent_context(
    agent_context: dict[str, Any] | list[dict[str, Any]] | Any,
    *,
    model_name: str | None = None,
    agent_name: str | None = None,
    n_agent_steps: int | None = None,
    models_cfg: ModelsConfig | None = None,
) -> TrialPricing:
    """Price cumulative `AgentContext` usage without false multi-turn Haiku cliff inflation."""
    canonical_top = normalize_model_key(model_name, agent_name=agent_name, models_cfg=models_cfg)
    if canonical_top == "oracle":
        return TrialPricing(
            input_tokens=0,
            uncached_input_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=0,
            total_tokens=0,
            peak_turn_tokens=0,
            haiku_cliff_triggered=False,
            cliff_turns=0,
            input_usd=0.0,
            cache_write_usd=0.0,
            cache_read_usd=0.0,
            output_usd=0.0,
            total_usd=0.0,
            pricing_source="oracle_zero",
        )

    raw_list = agent_context if isinstance(agent_context, list) else [agent_context]
    contexts: list[dict[str, Any]] = []
    for item in raw_list:
        if isinstance(item, dict):
            contexts.append(item)
        elif hasattr(item, "model_dump") and callable(item.model_dump):
            dumped = item.model_dump(mode="json")
            if isinstance(dumped, dict):
                contexts.append(dumped)

    single_turn = n_agent_steps == 1
    ctx_turns: list[TurnPricing] = []
    for ctx in contexts:
        mu = ctx.get("model_usage")
        if isinstance(mu, dict) and mu:
            for mu_model, usage in mu.items():
                if not isinstance(usage, dict):
                    continue
                n_in = _safe_nonneg_int(usage.get("n_input_tokens"))
                n_cache = _safe_nonneg_int(usage.get("n_cache_tokens"))
                n_out = _safe_nonneg_int(usage.get("n_output_tokens"))
                fb = _safe_nonneg_float(usage.get("cost_usd"))
                if n_in == 0 and n_cache == 0 and n_out == 0 and fb is None:
                    continue
                ctx_turns.append(
                    price_turn(
                        mu_model,
                        uncached_input_tokens=max(0, n_in - n_cache),
                        cache_read_tokens=n_cache,
                        output_tokens=n_out,
                        agent_name=agent_name,
                        fallback_cost_usd=fb,
                        allow_cliff=single_turn,
                        models_cfg=models_cfg,
                    )
                )
        else:
            n_in = _safe_nonneg_int(ctx.get("n_input_tokens"))
            n_cache = _safe_nonneg_int(ctx.get("n_cache_tokens"))
            n_out = _safe_nonneg_int(ctx.get("n_output_tokens"))
            fb = _safe_nonneg_float(ctx.get("cost_usd"))
            if n_in > 0 or n_cache > 0 or n_out > 0 or fb is not None:
                ctx_turns.append(
                    price_turn(
                        model_name,
                        uncached_input_tokens=max(0, n_in - n_cache),
                        cache_read_tokens=n_cache,
                        output_tokens=n_out,
                        agent_name=agent_name,
                        fallback_cost_usd=fb,
                        allow_cliff=single_turn,
                        models_cfg=models_cfg,
                    )
                )

    if ctx_turns:
        in_tok = sum(t.input_tokens for t in ctx_turns)
        u_in = sum(t.uncached_input_tokens for t in ctx_turns)
        c_read = sum(t.cache_read_tokens for t in ctx_turns)
        c_write = sum(t.cache_write_tokens for t in ctx_turns)
        out_tok = sum(t.output_tokens for t in ctx_turns)
        peak_tok = max(t.input_tokens for t in ctx_turns) if single_turn else 0
        cliff_turns = sum(1 for t in ctx_turns if t.cliff_triggered) if single_turn else 0
        in_usd = sum(t.input_usd for t in ctx_turns)
        cw_usd = sum(t.cache_write_usd for t in ctx_turns)
        cr_usd = sum(t.cache_read_usd for t in ctx_turns)
        out_usd = sum(t.output_usd for t in ctx_turns)
        tot_usd = in_usd + cw_usd + cr_usd + out_usd
        return TrialPricing(
            input_tokens=in_tok,
            uncached_input_tokens=u_in,
            cache_read_tokens=c_read,
            cache_write_tokens=c_write,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            peak_turn_tokens=peak_tok,
            haiku_cliff_triggered=cliff_turns > 0,
            cliff_turns=cliff_turns,
            input_usd=round(in_usd, 8),
            cache_write_usd=round(cw_usd, 8),
            cache_read_usd=round(cr_usd, 8),
            output_usd=round(out_usd, 8),
            total_usd=round(tot_usd, 8),
            pricing_source=_resolve_pricing_source("agent_context", ctx_turns),
        )

    empty_src = (
        "unknown_model"
        if (
            canonical_top
            and get_model_rate(canonical_top, agent_name=agent_name, models_cfg=models_cfg) is None
        )
        else "empty"
    )
    return TrialPricing(
        input_tokens=0,
        uncached_input_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=0,
        total_tokens=0,
        peak_turn_tokens=0,
        haiku_cliff_triggered=False,
        cliff_turns=0,
        input_usd=0.0,
        cache_write_usd=0.0,
        cache_read_usd=0.0,
        output_usd=0.0,
        total_usd=0.0,
        pricing_source=empty_src,
    )


def compute_trial_pricing(
    *,
    model_name: str | None,
    agent_name: str | None = None,
    trajectory_dict: dict[str, Any] | None = None,
    trial_result_dict: dict[str, Any] | None = None,
    models_cfg: ModelsConfig | None = None,
) -> TrialPricing:
    """Compute per-trial token and list-price USD metrics from ATIF trace or TrialResult."""
    canonical = normalize_model_key(model_name, agent_name=agent_name, models_cfg=models_cfg)
    if canonical == "oracle":
        return TrialPricing(
            input_tokens=0,
            uncached_input_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=0,
            total_tokens=0,
            peak_turn_tokens=0,
            haiku_cliff_triggered=False,
            cliff_turns=0,
            input_usd=0.0,
            cache_write_usd=0.0,
            cache_read_usd=0.0,
            output_usd=0.0,
            total_usd=0.0,
            pricing_source="oracle_zero",
        )

    n_agent_steps = _count_trajectory_agent_steps(trajectory_dict)

    # 1. Primary source: per-step ATIF trajectory metrics (preserves exact PeakTurnTokens & cliff).
    if isinstance(trajectory_dict, dict):
        turns = _collect_trajectory_turns(
            trajectory_dict, model_name, agent_name, models_cfg=models_cfg
        )
        if turns:
            in_tok = sum(t.input_tokens for t in turns)
            u_in = sum(t.uncached_input_tokens for t in turns)
            c_read = sum(t.cache_read_tokens for t in turns)
            c_write = sum(t.cache_write_tokens for t in turns)
            out_tok = sum(t.output_tokens for t in turns)
            peak_tok = max(t.input_tokens for t in turns)
            cliff_turns = sum(1 for t in turns if t.cliff_triggered)
            in_usd = sum(t.input_usd for t in turns)
            cw_usd = sum(t.cache_write_usd for t in turns)
            cr_usd = sum(t.cache_read_usd for t in turns)
            out_usd = sum(t.output_usd for t in turns)
            tot_usd = in_usd + cw_usd + cr_usd + out_usd
            return TrialPricing(
                input_tokens=in_tok,
                uncached_input_tokens=u_in,
                cache_read_tokens=c_read,
                cache_write_tokens=c_write,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
                peak_turn_tokens=peak_tok,
                haiku_cliff_triggered=cliff_turns > 0,
                cliff_turns=cliff_turns,
                input_usd=round(in_usd, 8),
                cache_write_usd=round(cw_usd, 8),
                cache_read_usd=round(cr_usd, 8),
                output_usd=round(out_usd, 8),
                total_usd=round(tot_usd, 8),
                pricing_source=_resolve_pricing_source("trajectory_steps", turns),
            )

        # 2. Fallback: aggregate ATIF FinalMetrics when per-step metrics are absent.
        fm = trajectory_dict.get("final_metrics")
        if isinstance(fm, dict):
            agent_blk = trajectory_dict.get("agent")
            eff_agent = (
                agent_blk.get("name")
                if isinstance(agent_blk, dict) and isinstance(agent_blk.get("name"), str)
                else None
            ) or agent_name
            is_opencode = (eff_agent or "").strip().lower() == "opencode"

            extra_fm = fm.get("extra") if isinstance(fm.get("extra"), dict) else {}
            raw_prompt = _safe_nonneg_int(fm.get("total_prompt_tokens"))
            out_tok = _safe_nonneg_int(fm.get("total_completion_tokens"))
            if "total_cache_read_input_tokens" in extra_fm:
                c_read = _safe_nonneg_int(extra_fm.get("total_cache_read_input_tokens"))
            else:
                c_read = _safe_nonneg_int(fm.get("total_cached_tokens"))
            has_cc = "total_cache_creation_input_tokens" in extra_fm
            has_oc_write = ("total_cache_write_tokens" in extra_fm) and not has_cc
            if has_cc:
                c_write = _safe_nonneg_int(extra_fm.get("total_cache_creation_input_tokens"))
            else:
                c_write = _safe_nonneg_int(extra_fm.get("total_cache_write_tokens"))
            fb_usd = _safe_nonneg_float(fm.get("total_cost_usd"))
            if raw_prompt > 0 or out_tok > 0 or c_read > 0 or c_write > 0 or fb_usd is not None:
                if has_oc_write:
                    u_in = max(0, raw_prompt - c_read)
                elif has_cc:
                    if raw_prompt >= c_read + c_write:
                        u_in = raw_prompt - c_read - c_write
                    else:
                        u_in = max(0, raw_prompt - c_read)
                elif not is_opencode and raw_prompt >= c_read + c_write:
                    u_in = raw_prompt - c_read - c_write
                else:
                    u_in = max(0, raw_prompt - c_read)

                eff_steps = n_agent_steps
                if eff_steps is None and _safe_nonneg_int(fm.get("total_steps")) == 1:
                    eff_steps = 1
                single_turn = eff_steps == 1

                tp = price_turn(
                    model_name,
                    uncached_input_tokens=u_in,
                    cache_read_tokens=c_read,
                    cache_write_tokens=c_write,
                    output_tokens=out_tok,
                    agent_name=eff_agent,
                    fallback_cost_usd=fb_usd,
                    allow_cliff=single_turn,
                    models_cfg=models_cfg,
                )
                return TrialPricing(
                    input_tokens=tp.input_tokens,
                    uncached_input_tokens=tp.uncached_input_tokens,
                    cache_read_tokens=tp.cache_read_tokens,
                    cache_write_tokens=tp.cache_write_tokens,
                    output_tokens=tp.output_tokens,
                    total_tokens=tp.input_tokens + tp.output_tokens,
                    peak_turn_tokens=tp.input_tokens if single_turn else 0,
                    haiku_cliff_triggered=tp.cliff_triggered if single_turn else False,
                    cliff_turns=(1 if tp.cliff_triggered else 0) if single_turn else 0,
                    input_usd=round(tp.input_usd, 8),
                    cache_write_usd=round(tp.cache_write_usd, 8),
                    cache_read_usd=round(tp.cache_read_usd, 8),
                    output_usd=round(tp.output_usd, 8),
                    total_usd=round(tp.total_usd, 8),
                    pricing_source=_resolve_pricing_source("trajectory_final_metrics", [tp]),
                )

    # 3. Fallback: TrialResult.agent_result (or step_results[*].agent_result) AgentContext.
    if isinstance(trial_result_dict, dict):
        contexts: list[dict[str, Any]] = []
        ar = trial_result_dict.get("agent_result")
        if isinstance(ar, dict):
            contexts.append(ar)
        elif isinstance(trial_result_dict.get("step_results"), list):
            for sr in trial_result_dict["step_results"]:
                if isinstance(sr, dict) and isinstance(sr.get("agent_result"), dict):
                    contexts.append(sr["agent_result"])

        if contexts:
            eff_ctx_steps = (
                len(contexts) if (len(contexts) > 1 and n_agent_steps == 1) else n_agent_steps
            )
            priced_ctx = price_from_agent_context(
                contexts,
                model_name=model_name,
                agent_name=agent_name,
                n_agent_steps=eff_ctx_steps,
                models_cfg=models_cfg,
            )
            if (
                priced_ctx.total_tokens > 0
                or priced_ctx.total_usd > 0.0
                or priced_ctx.pricing_source == "agent_reported_cost"
            ):
                return priced_ctx

    empty_src = (
        "unknown_model"
        if (
            canonical
            and get_model_rate(canonical, agent_name=agent_name, models_cfg=models_cfg) is None
        )
        else "empty"
    )
    return TrialPricing(
        input_tokens=0,
        uncached_input_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=0,
        total_tokens=0,
        peak_turn_tokens=0,
        haiku_cliff_triggered=False,
        cliff_turns=0,
        input_usd=0.0,
        cache_write_usd=0.0,
        cache_read_usd=0.0,
        output_usd=0.0,
        total_usd=0.0,
        pricing_source=empty_src,
    )
