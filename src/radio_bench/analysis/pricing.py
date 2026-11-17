"""List-price token cost model, provider-aware reasoning accounting, and retry/role attribution."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radio_bench.analysis._trace import (
    BASH_TOOL_NAMES,
    CASA_SEVERE_PHRASES,
    DISPATCH_TOOL_NAMES,
    GENERIC_AGENT_MODES,
    PASSIVE_OR_DISPATCH_TOOL_NAMES,
    OpenCodeDBSummary,
    classify_mcp_server,
    extract_claude_agent_id_from_obs,
    extract_dispatched_role,
    extract_exit_code,
    extract_opencode_session_id_from_obs,
    is_protocol_error,
    is_tool_error,
    parse_opencode_tool_events,
    read_opencode_sqlite,
    resolve_opencode_db_path,
    safe_nonneg_float,
    safe_nonneg_int,
    scan_claude_subagent_meta_files,
    stringify_obs_content,
)
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
    reasoning_tokens: int | None = None
    reply_output_tokens: int | None = None
    role: str = "main"
    stream_id: str = "main"
    is_retry: bool = False


@dataclass(frozen=True)
class TrialPricing:
    """Structured per-trial token, reasoning, retry, and list-price USD breakdown."""

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
    reasoning_tokens: int | None = None
    reply_output_tokens: int | None = None
    retry_turn_count: int = 0
    first_pass_usd: float = 0.0
    retry_usd: float = 0.0

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


_safe_nonneg_int = safe_nonneg_int
_safe_nonneg_float = safe_nonneg_float


def _resolve_pricing_source(default_source: str, turns: list[TurnPricing]) -> str:
    if all(t.rate_known for t in turns):
        return default_source
    unknown_turns = [t for t in turns if not t.rate_known]
    if any(t.used_fallback_cost for t in unknown_turns):
        return "agent_reported_cost"
    return "unknown_model"


def _finalize_retry_usd_split(
    total_usd_rounded: float,
    turns: list[TurnPricing],
) -> tuple[int, float, float]:
    """Compute `(retry_turn_count, first_pass_usd, retry_usd)` with exact `first + retry == total`."""
    retry_cnt = sum(1 for t in turns if t.is_retry)
    if retry_cnt == 0 or not turns:
        return 0, total_usd_rounded, 0.0
    if retry_cnt == len(turns):
        return retry_cnt, 0.0, total_usd_rounded

    raw_retry = sum(t.total_usd for t in turns if t.is_retry)
    retry_rounded = round(raw_retry, 8)
    first_rounded = round(total_usd_rounded - retry_rounded, 8)
    # Re-anchor total_usd if binary64 subtraction differs by 1e-8 boundary
    return retry_cnt, first_rounded, retry_rounded


def _finalize_reasoning_totals(
    billed_output_tokens: int,
    turn_reasonings: list[int],
) -> tuple[int | None, int | None]:
    """Emit integer `(reasoning_tokens, reply_output_tokens)` iff >= 1 turn reported reasoning > 0."""
    if any(r > 0 for r in turn_reasonings):
        reas_sum = sum(turn_reasonings)
        return reas_sum, max(0, billed_output_tokens - reas_sum)
    return None, None


def price_turn(
    model_name: str | None,
    *,
    uncached_input_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int | None = None,
    agent_name: str | None = None,
    fallback_cost_usd: float | None = None,
    allow_cliff: bool = True,
    role: str = "main",
    stream_id: str = "main",
    is_retry: bool = False,
    models_cfg: ModelsConfig | None = None,
) -> TurnPricing:
    """Price a single LLM turn using list prices (`output_tokens` is inclusive billed output)."""
    u_in = safe_nonneg_int(uncached_input_tokens)
    c_read = safe_nonneg_int(cache_read_tokens)
    c_write = safe_nonneg_int(cache_write_tokens)
    c_write_1h = min(safe_nonneg_int(cache_write_1h_tokens), c_write)
    c_write_5m = c_write - c_write_1h
    out_tok = safe_nonneg_int(output_tokens)
    total_in = u_in + c_read + c_write

    reas_val: int | None = None
    reply_val: int | None = None
    if reasoning_tokens is not None and safe_nonneg_int(reasoning_tokens) > 0:
        reas_val = safe_nonneg_int(reasoning_tokens)
        reply_val = max(0, out_tok - reas_val)

    canonical = normalize_model_key(model_name, agent_name=agent_name, models_cfg=models_cfg)
    rate = get_model_rate(canonical, agent_name=agent_name, models_cfg=models_cfg)
    if rate is None:
        fb_val = safe_nonneg_float(fallback_cost_usd)
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
            reasoning_tokens=reas_val,
            reply_output_tokens=reply_val,
            role=role,
            stream_id=stream_id,
            is_retry=is_retry,
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
        reasoning_tokens=reas_val,
        reply_output_tokens=reply_val,
        role=role,
        stream_id=stream_id,
        is_retry=is_retry,
    )


def _extract_raw_reasoning_tokens(metrics_dict: dict[str, Any], extra_d: dict[str, Any]) -> int:
    """Extract raw reasoning/thinking token count from ATIF `Metrics` / `extra`."""
    for top_key in ("reasoning_tokens", "thinking_tokens"):
        if top_key in metrics_dict and metrics_dict.get(top_key) is not None:
            return safe_nonneg_int(metrics_dict.get(top_key))
    for ext_key in ("reasoning_tokens", "thinking_tokens"):
        if ext_key in extra_d and extra_d.get(ext_key) is not None:
            return safe_nonneg_int(extra_d.get(ext_key))
    for det_key in ("output_tokens_details", "completion_tokens_details"):
        det = extra_d.get(det_key)
        if isinstance(det, dict):
            for sub_key in ("thinking_tokens", "reasoning_tokens"):
                if sub_key in det and det.get(sub_key) is not None:
                    return safe_nonneg_int(det.get(sub_key))
    return 0


def _extract_step_token_counts_with_reasoning(
    metrics_dict: dict[str, Any],
    *,
    agent_name: str | None = None,
) -> tuple[int, int, int, int, int, int, float | None]:
    """Extract `(uncached_in, cache_read, cache_write, cache_write_1h, billed_out, reasoning, fb_usd)`.

    Provider-aware output accounting:
    - Harbor `OpenCode` (`1.19.5`) sets `Metrics.completion_tokens = tokens.output` (reply-only,
      having already subtracted `tokens.reasoning`), with `reasoning_tokens` in `Metrics.extra`.
      Thus `billed_out = raw_completion + raw_reasoning`.
    - `claude-code` and all inclusive ATIF harnesses set `Metrics.completion_tokens` inclusive of
      thinking/reasoning tokens. Thus `billed_out = raw_completion`.
    """
    extra = metrics_dict.get("extra")
    extra_d = extra if isinstance(extra, dict) else {}

    raw_prompt = safe_nonneg_int(metrics_dict.get("prompt_tokens"))
    raw_completion = safe_nonneg_int(metrics_dict.get("completion_tokens"))
    raw_reasoning = _extract_raw_reasoning_tokens(metrics_dict, extra_d)

    if "cache_read_input_tokens" in extra_d:
        cache_read = safe_nonneg_int(extra_d.get("cache_read_input_tokens"))
    else:
        cache_read = safe_nonneg_int(metrics_dict.get("cached_tokens"))

    has_claude_creation = "cache_creation_input_tokens" in extra_d
    has_opencode_write = ("cache_write_tokens" in extra_d) and not has_claude_creation
    if has_claude_creation:
        cache_write = safe_nonneg_int(extra_d.get("cache_creation_input_tokens"))
    elif "cache_write_tokens" in extra_d:
        cache_write = safe_nonneg_int(extra_d.get("cache_write_tokens"))
    else:
        cache_write = 0

    cache_creation_obj = extra_d.get("cache_creation")
    if isinstance(cache_creation_obj, dict):
        cache_write_1h = min(
            safe_nonneg_int(cache_creation_obj.get("ephemeral_1h_input_tokens")),
            cache_write,
        )
    else:
        cache_write_1h = 0

    is_opencode = (agent_name or "").strip().lower() == "opencode"
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

    if is_opencode:
        billed_out = raw_completion + raw_reasoning
        turn_reasoning = raw_reasoning
    else:
        billed_out = max(raw_completion, raw_reasoning)
        turn_reasoning = raw_reasoning

    fallback_usd = safe_nonneg_float(metrics_dict.get("cost_usd"))
    return (
        uncached_in,
        cache_read,
        cache_write,
        cache_write_1h,
        billed_out,
        turn_reasoning,
        fallback_usd,
    )


def _extract_step_token_counts(
    metrics_dict: dict[str, Any],
    *,
    agent_name: str | None = None,
) -> tuple[int, int, int, int, int, float | None]:
    """Backward-compatible 6-tuple step token extractor returning billed `output_tokens`."""
    u_in, c_read, c_write, c_write_1h, billed_out, _reas, fb_usd = (
        _extract_step_token_counts_with_reasoning(metrics_dict, agent_name=agent_name)
    )
    return u_in, c_read, c_write, c_write_1h, billed_out, fb_usd


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


@dataclass
class _RoleAccumulator:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    turn_reasonings: list[int] = field(default_factory=list)
    total_usd_raw: float = 0.0
    retry_usd_raw: float = 0.0
    tool_calls: int = 0
    mcp_calls_total: int = 0
    bash_calls: int = 0
    bash_nonzero_exits: int = 0
    tool_errors: int = 0
    retry_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        tot_r = round(self.total_usd_raw, 8)
        if self.retry_turns == 0:
            ret_r = 0.0
            first_r = tot_r
        elif self.retry_turns == self.turns:
            ret_r = tot_r
            first_r = 0.0
        else:
            ret_r = round(self.retry_usd_raw, 8)
            first_r = round(tot_r - ret_r, 8)
            tot_r = round(first_r + ret_r, 8)
        reas_tot, reply_tot = _finalize_reasoning_totals(self.output_tokens, self.turn_reasonings)
        return {
            "turns": self.turns,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": reas_tot,
            "reply_output_tokens": reply_tot,
            "total_usd": tot_r,
            "first_pass_usd": first_r,
            "tool_calls": self.tool_calls,
            "mcp_calls_total": self.mcp_calls_total,
            "bash_calls": self.bash_calls,
            "bash_nonzero_exits": self.bash_nonzero_exits,
            "tool_errors": self.tool_errors,
            "retry_turns": self.retry_turns,
            "retry_usd": ret_r,
        }


@dataclass(frozen=True)
class TrialExecutionAnalysis:
    """Unified multi-harness turn, role, retry, tool, and token extraction."""

    telemetry: dict[str, int]
    priced_turns: list[TurnPricing]
    turn_reasonings: list[int]
    role_breakdown: dict[str, dict[str, Any]]
    opencode_db: OpenCodeDBSummary | None
    atif_agent_turns: int
    atif_tool_calls: int
    recovered_child_turns: int
    recovered_child_tool_calls: int


def _resolve_step_model_for_pricing(
    step_model: str | None,
    default_model: str | None,
    agent_name: str | None,
    models_cfg: ModelsConfig | None,
) -> str | None:
    if not step_model:
        return default_model
    if get_model_rate(step_model, agent_name=agent_name, models_cfg=models_cfg) is not None:
        return step_model
    if (
        default_model
        and get_model_rate(default_model, agent_name=agent_name, models_cfg=models_cfg) is not None
    ):
        return default_model
    return step_model


def _build_obs_index(
    step: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
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
    return obs_by_call_id, unkeyed_obs


def _scan_parent_dispatches(
    steps: list[Any],
    oc_events: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    list[tuple[str, str]],
    set[str],
    set[str],
]:
    """Scan parent steps for subagent dispatches across ClaudeCode, OpenCode, and ATIF refs.

    Returns:
    - `call_id_to_role`
    - `claude_agent_id_to_role`
    - `opencode_session_to_role`
    - `atif_subtraj_id_to_role`
    - `ordered_dispatches`: `[(call_id, role), ...]`
    - `oc_linked_cids`: parent tool_call_ids linked to an OpenCode child `session_id`
    - `subtraj_linked_cids`: parent tool_call_ids linked to an ATIF `subagent_trajectory_ref`
    """
    call_id_to_role: dict[str, str] = {}
    claude_agent_id_to_role: dict[str, str] = {}
    opencode_session_to_role: dict[str, str] = {}
    atif_subtraj_id_to_role: dict[str, str] = {}
    atif_subtraj_sid_candidates: dict[str, list[tuple[str, str]]] = {}
    ordered_dispatches: list[tuple[str, str]] = []
    oc_linked_cids: set[str] = set()
    subtraj_linked_cids: set[str] = set()

    for step in steps:
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        obs_by_id, unkeyed_obs = _build_obs_index(step)
        tcs = step.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for idx, tc in enumerate(tcs):
            if not isinstance(tc, dict):
                continue
            fn_name = str(tc.get("function_name") or "").strip()
            call_id = str(tc.get("tool_call_id") or "").strip()
            oc_ev = oc_events.get(call_id) if call_id else None
            eff_fn = fn_name or (str(oc_ev.get("tool") or "") if oc_ev else "")
            if eff_fn.strip().lower() not in DISPATCH_TOOL_NAMES:
                continue

            role = extract_dispatched_role(tc.get("arguments"))
            if not role and oc_ev is not None:
                role = oc_ev.get("dispatched_role") or extract_dispatched_role(oc_ev.get("input"))
            if not role:
                continue

            if call_id:
                call_id_to_role[call_id] = role
            ordered_dispatches.append((call_id, role))

            obs_item = obs_by_id.get(call_id) if call_id else None
            if obs_item is None and not call_id and idx < len(unkeyed_obs):
                obs_item = unkeyed_obs[idx]
            obs_extra = (
                obs_item.get("extra")
                if isinstance(obs_item, dict) and isinstance(obs_item.get("extra"), dict)
                else {}
            )
            obs_text = (
                stringify_obs_content(obs_item.get("content")) if isinstance(obs_item, dict) else ""
            )

            # (a) ClaudeCode agentId link
            c_aid = extract_claude_agent_id_from_obs(obs_extra, obs_text)
            if c_aid:
                claude_agent_id_to_role[c_aid] = role

            # (b) OpenCode child sessionId link
            oc_sid = extract_opencode_session_id_from_obs(obs_extra, obs_text)
            if not oc_sid and oc_ev is not None and oc_ev.get("child_session_id"):
                oc_sid = str(oc_ev["child_session_id"])
            if oc_sid:
                opencode_session_to_role[oc_sid] = role
                if call_id:
                    oc_linked_cids.add(call_id)

            # (c) Standard ATIF subagent_trajectory_ref link
            if isinstance(obs_item, dict):
                refs = obs_item.get("subagent_trajectory_ref")
                ref_list = refs if isinstance(refs, list) else ([refs] if refs is not None else [])
                for ref in ref_list:
                    if isinstance(ref, dict):
                        tid = str(ref.get("trajectory_id") or "").strip()
                        rsid = str(ref.get("session_id") or "").strip()
                        if tid:
                            atif_subtraj_id_to_role[tid] = role
                            if call_id:
                                subtraj_linked_cids.add(call_id)
                        elif rsid:
                            atif_subtraj_sid_candidates.setdefault(rsid, []).append((call_id, role))

    for rsid, cid_role_list in atif_subtraj_sid_candidates.items():
        if len(cid_role_list) == 1 and rsid not in atif_subtraj_id_to_role:
            cid0, role0 = cid_role_list[0]
            atif_subtraj_id_to_role[rsid] = role0
            if cid0:
                subtraj_linked_cids.add(cid0)

    return (
        call_id_to_role,
        claude_agent_id_to_role,
        opencode_session_to_role,
        atif_subtraj_id_to_role,
        ordered_dispatches,
        oc_linked_cids,
        subtraj_linked_cids,
    )


def analyze_trial_execution(
    traj_dict: dict[str, Any] | None,
    *,
    model_name: str | None = None,
    agent_name: str | None = None,
    arm: str = "00_none",
    trial_dir: Path | None = None,
    opencode_txt_path: Path | None = None,
    opencode_db_path: Path | None = None,
    models_cfg: ModelsConfig | None = None,
) -> TrialExecutionAnalysis:
    """Walk ATIF trajectory + ClaudeCode sidechains + OpenCode SQLite for unified telemetry & pricing."""
    telemetry = {
        "n_turns": 0,
        "tool_calls_total": 0,
        "mcp_calls_total": 0,
        "mcp_calls_inspect": 0,
        "mcp_calls_modify": 0,
        "mcp_calls_create": 0,
        "bash_calls": 0,
        "bash_nonzero_exits": 0,
        "tool_errors": 0,
        "protocol_errors": 0,
        "severe_casa_errors": 0,
        "retry_turn_count": 0,
    }

    eff_agent_top = (agent_name or "").strip()
    if isinstance(traj_dict, dict) and isinstance(traj_dict.get("agent"), dict):
        t_ag_name = traj_dict["agent"].get("name")
        if isinstance(t_ag_name, str) and t_ag_name.strip() and not eff_agent_top:
            eff_agent_top = t_ag_name.strip()

    if normalize_model_key(model_name, agent_name=eff_agent_top, models_cfg=models_cfg) == "oracle":
        return TrialExecutionAnalysis(
            telemetry=telemetry,
            priced_turns=[],
            turn_reasonings=[],
            role_breakdown={},
            opencode_db=None,
            atif_agent_turns=0,
            atif_tool_calls=0,
            recovered_child_turns=0,
            recovered_child_tool_calls=0,
        )

    has_opencode_txt = opencode_txt_path is not None and Path(opencode_txt_path).is_file()
    oc_events = parse_opencode_tool_events(opencode_txt_path) if has_opencode_txt else {}

    root_steps = (
        traj_dict.get("steps")
        if isinstance(traj_dict, dict) and isinstance(traj_dict.get("steps"), list)
        else []
    )
    (
        call_id_to_role,
        claude_aid_to_role,
        oc_sid_to_role,
        subtraj_id_to_role,
        ordered_dispatches,
        oc_linked_cids,
        subtraj_linked_cids,
    ) = _scan_parent_dispatches(root_steps, oc_events)

    resolved_db_path = resolve_opencode_db_path(trial_dir, opencode_db_path)
    oc_db: OpenCodeDBSummary | None = None
    if resolved_db_path is not None:
        oc_db = read_opencode_sqlite(
            resolved_db_path,
            parent_traj_session_roles=oc_sid_to_role,
            parent_ordered_dispatches=ordered_dispatches,
            parent_linked_cids=oc_linked_cids,
        )
        if oc_db is not None:
            for cid, db_ev in oc_db.tool_events_by_id.items():
                if cid not in oc_events:
                    oc_events[cid] = db_ev
                else:
                    # Enrich missing fields from SQLite
                    for k in ("tool", "status", "error", "exit", "output"):
                        if not oc_events[cid].get(k) and db_ev.get(k) is not None:
                            oc_events[cid][k] = db_ev[k]

    # Link ClaudeCode subagent `.meta.json` files on disk if present
    meta_aid_roles, meta_linked_cids = scan_claude_subagent_meta_files(trial_dir, call_id_to_role)
    linked_dispatch_cids: set[str] = set(meta_linked_cids)
    for step in root_steps:
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        obs_by_id, _ = _build_obs_index(step)
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            cid = str(tc.get("tool_call_id") or "").strip()
            if cid and cid in call_id_to_role and cid in obs_by_id:
                obs_it = obs_by_id[cid]
                obs_ex = obs_it.get("extra") if isinstance(obs_it.get("extra"), dict) else {}
                obs_tx = stringify_obs_content(obs_it.get("content"))
                if extract_claude_agent_id_from_obs(obs_ex, obs_tx):
                    linked_dispatch_cids.add(cid)

    for aid, rname in meta_aid_roles.items():
        claude_aid_to_role.setdefault(aid, rname)

    # Identify all inlined sidechain `agent_id`s in chronological order and resolve unmapped ones
    sidechain_aids_ordered: list[str] = []
    has_any_sidechain_steps = False
    for step in root_steps:
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        st_ex = step.get("extra") if isinstance(step.get("extra"), dict) else {}
        is_sc = bool(st_ex.get("is_sidechain"))
        aid = str(st_ex.get("agent_id") or "").strip()
        if is_sc or aid:
            has_any_sidechain_steps = True
            key_aid = aid or "sidechain"
            if key_aid not in sidechain_aids_ordered:
                sidechain_aids_ordered.append(key_aid)

    remaining_dispatch_roles = [
        role for cid, role in ordered_dispatches if not cid or cid not in linked_dispatch_cids
    ]
    rem_cursor = 0
    for aid in sidechain_aids_ordered:
        if aid not in claude_aid_to_role:
            if rem_cursor < len(remaining_dispatch_roles):
                claude_aid_to_role[aid] = remaining_dispatch_roles[rem_cursor]
                rem_cursor += 1
            else:
                claude_aid_to_role[aid] = f"subagent_{aid}" if aid != "sidechain" else "subagent"

    sub_trajs = (
        traj_dict.get("subagent_trajectories")
        if isinstance(traj_dict, dict) and isinstance(traj_dict.get("subagent_trajectories"), list)
        else []
    )
    has_oc_children = oc_db is not None and len(oc_db.child_sessions) > 0
    has_any_subagents = bool(has_any_sidechain_steps or len(sub_trajs) > 0 or has_oc_children)
    root_role = "orchestrator" if (has_any_subagents or arm == "11_orchestrated") else "main"

    stream_prev_had_error: dict[str, bool] = {}
    role_accums: dict[str, _RoleAccumulator] = {}
    priced_turns: list[TurnPricing] = []
    turn_reasonings: list[int] = []

    atif_agent_turns = 0
    atif_tool_calls = 0
    recovered_child_turns = 0
    recovered_child_tool_calls = 0

    has_any_oc_event_source = has_opencode_txt or (
        oc_db is not None and bool(oc_db.tool_events_by_id)
    )

    def _process_atif_trajectory(
        t_obj: dict[str, Any],
        *,
        fixed_role: str | None,
        fixed_stream_prefix: str,
        parent_model: str | None,
        parent_agent: str | None,
        sub_positional_roles: list[str],
    ) -> None:
        nonlocal atif_agent_turns, atif_tool_calls
        agent_blk = t_obj.get("agent") if isinstance(t_obj.get("agent"), dict) else {}
        t_model = (
            agent_blk.get("model_name")
            if isinstance(agent_blk.get("model_name"), str) and agent_blk.get("model_name")
            else None
        ) or parent_model
        t_agent = (
            agent_blk.get("name")
            if isinstance(agent_blk.get("name"), str) and agent_blk.get("name")
            else None
        ) or parent_agent
        is_opencode = (t_agent or "").strip().lower() == "opencode"

        steps_list = t_obj.get("steps")
        if isinstance(steps_list, list):
            n_steps = len(steps_list)
            for step_pos, step in enumerate(steps_list):
                if not isinstance(step, dict) or step.get("source") != "agent":
                    continue
                atif_agent_turns += 1
                telemetry["n_turns"] += 1
                has_subsequent_steps = step_pos < n_steps - 1

                st_ex = step.get("extra") if isinstance(step.get("extra"), dict) else {}
                if fixed_role is not None:
                    role = fixed_role
                    stream_id = fixed_stream_prefix
                else:
                    is_sc = bool(st_ex.get("is_sidechain"))
                    aid = str(st_ex.get("agent_id") or "").strip()
                    if is_sc or aid:
                        key_aid = aid or "sidechain"
                        role = claude_aid_to_role.get(key_aid, f"subagent_{key_aid}")
                        stream_id = f"sidechain:{key_aid}"
                    else:
                        role = root_role
                        stream_id = "root"

                is_retry = stream_prev_had_error.get(stream_id, False)
                if is_retry:
                    telemetry["retry_turn_count"] += 1

                r_acc = role_accums.setdefault(role, _RoleAccumulator())
                r_acc.turns += 1
                if is_retry:
                    r_acc.retry_turns += 1

                obs_by_call_id, unkeyed_obs = _build_obs_index(step)
                tcs = step.get("tool_calls")
                turn_had_error = False

                if isinstance(tcs, list):
                    for idx, tc in enumerate(tcs):
                        if not isinstance(tc, dict):
                            continue
                        atif_tool_calls += 1
                        telemetry["tool_calls_total"] += 1
                        r_acc.tool_calls += 1

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
                            stringify_obs_content(obs_item.get("content"))
                            if isinstance(obs_item, dict)
                            else ""
                        )

                        oc_ev = oc_events.get(call_id) if call_id else None
                        eff_fn_name = fn_name or (str(oc_ev["tool"]) if oc_ev else "")
                        text_parts = [p for p in (obs_text,) if p]
                        if oc_ev is not None:
                            if oc_ev.get("output") and oc_ev["output"] != obs_text:
                                text_parts.append(str(oc_ev["output"]))
                            if oc_ev.get("error"):
                                text_parts.append(str(oc_ev["error"]))
                        combined_obs_text = "\n".join(text_parts)

                        oc_failed = bool(
                            oc_ev is not None
                            and (oc_ev.get("status") == "error" or bool(oc_ev.get("error")))
                        )
                        missing_oc_obs_err = bool(
                            is_opencode
                            and not has_any_oc_event_source
                            and bool(call_id)
                            and obs_item is None
                            and has_subsequent_steps
                        )

                        mcp_srv = classify_mcp_server(eff_fn_name)
                        if mcp_srv is not None:
                            telemetry["mcp_calls_total"] += 1
                            r_acc.mcp_calls_total += 1
                            if mcp_srv == "ms-inspect":
                                telemetry["mcp_calls_inspect"] += 1
                            elif mcp_srv == "ms-modify":
                                telemetry["mcp_calls_modify"] += 1
                            elif mcp_srv == "ms-create":
                                telemetry["mcp_calls_create"] += 1

                        low_fn = eff_fn_name.strip().lower()
                        is_bash = low_fn in BASH_TOOL_NAMES
                        bash_failed = False
                        if is_bash:
                            telemetry["bash_calls"] += 1
                            r_acc.bash_calls += 1
                            ec = extract_exit_code(tc_extra, obs_extra, combined_obs_text)
                            if ec is None and oc_ev is not None and oc_ev.get("exit") is not None:
                                ec = int(oc_ev["exit"])
                            if (
                                (ec is not None and ec != 0)
                                or oc_failed
                                or missing_oc_obs_err
                                or is_tool_error(tc_extra, obs_extra, combined_obs_text)
                            ):
                                telemetry["bash_nonzero_exits"] += 1
                                r_acc.bash_nonzero_exits += 1
                                bash_failed = True

                        low_eff_fn = eff_fn_name.strip().lower()
                        if low_eff_fn not in PASSIVE_OR_DISPATCH_TOOL_NAMES and any(
                            p.lower() in combined_obs_text.lower() for p in CASA_SEVERE_PHRASES
                        ):
                            telemetry["severe_casa_errors"] += 1

                        is_proto = is_protocol_error(eff_fn_name, mcp_srv, combined_obs_text, arm)
                        if is_proto:
                            telemetry["protocol_errors"] += 1

                        tool_failed = bool(
                            is_proto
                            or oc_failed
                            or missing_oc_obs_err
                            or is_tool_error(
                                tc_extra, obs_extra, combined_obs_text, fn_name=eff_fn_name
                            )
                        )
                        if tool_failed:
                            telemetry["tool_errors"] += 1
                            r_acc.tool_errors += 1

                        if bash_failed or tool_failed:
                            turn_had_error = True

                stream_prev_had_error[stream_id] = turn_had_error

                metrics = step.get("metrics")
                if isinstance(metrics, dict):
                    u_in, c_read, c_write, c_write_1h, billed_out, t_reas, fb_usd = (
                        _extract_step_token_counts_with_reasoning(metrics, agent_name=t_agent)
                    )
                    if (
                        u_in > 0
                        or c_read > 0
                        or c_write > 0
                        or billed_out > 0
                        or fb_usd is not None
                    ):
                        raw_step_m = (
                            step.get("model_name")
                            if isinstance(step.get("model_name"), str)
                            else None
                        )
                        eff_step_m = _resolve_step_model_for_pricing(
                            raw_step_m, t_model, t_agent, models_cfg
                        )
                        tp = price_turn(
                            eff_step_m,
                            uncached_input_tokens=u_in,
                            cache_read_tokens=c_read,
                            cache_write_tokens=c_write,
                            cache_write_1h_tokens=c_write_1h,
                            output_tokens=billed_out,
                            reasoning_tokens=t_reas,
                            agent_name=t_agent,
                            fallback_cost_usd=fb_usd,
                            role=role,
                            stream_id=stream_id,
                            is_retry=is_retry,
                            models_cfg=models_cfg,
                        )
                        priced_turns.append(tp)
                        turn_reasonings.append(t_reas)
                        r_acc.input_tokens += tp.input_tokens
                        r_acc.output_tokens += tp.output_tokens
                        r_acc.turn_reasonings.append(t_reas)
                        r_acc.total_usd_raw += tp.total_usd
                        if is_retry:
                            r_acc.retry_usd_raw += tp.total_usd

        # Standard ATIF `subagent_trajectories`
        nested = t_obj.get("subagent_trajectories")
        if isinstance(nested, list):
            sid_counts: dict[str, int] = {}
            for sub_it in nested:
                if isinstance(sub_it, dict):
                    s_val = str(sub_it.get("session_id") or "").strip()
                    if s_val:
                        sid_counts[s_val] = sid_counts.get(s_val, 0) + 1

            pos_cursor = 0
            for s_idx, sub in enumerate(nested):
                if not isinstance(sub, dict):
                    continue
                sub_tid = str(sub.get("trajectory_id") or "").strip()
                sub_sid = str(sub.get("session_id") or "").strip()
                if sub_tid:
                    processed_subtraj_session_ids.add(sub_tid)
                if sub_sid:
                    processed_subtraj_session_ids.add(sub_sid)

                sub_ag = sub.get("agent") if isinstance(sub.get("agent"), dict) else {}
                sub_ag_ex = sub_ag.get("extra") if isinstance(sub_ag.get("extra"), dict) else {}
                sub_extra_name = str(
                    sub_ag_ex.get("subagent_name")
                    or sub_ag_ex.get("agent_type")
                    or sub_ag_ex.get("role")
                    or ""
                ).strip()
                sub_ag_name = str(sub_ag.get("name") or "").strip()

                resolved_sub_role: str | None = None
                if sub_tid and sub_tid in subtraj_id_to_role:
                    resolved_sub_role = subtraj_id_to_role[sub_tid]
                elif sub_sid and sid_counts.get(sub_sid, 0) == 1 and sub_sid in subtraj_id_to_role:
                    resolved_sub_role = subtraj_id_to_role[sub_sid]
                elif sub_extra_name:
                    resolved_sub_role = sub_extra_name
                elif sub_ag_name and sub_ag_name.lower() not in {
                    "claude-code",
                    "opencode",
                    "oracle",
                    *GENERIC_AGENT_MODES,
                }:
                    resolved_sub_role = sub_ag_name
                elif pos_cursor < len(sub_positional_roles):
                    resolved_sub_role = sub_positional_roles[pos_cursor]
                    pos_cursor += 1
                else:
                    resolved_sub_role = f"subagent_{s_idx + 1}"

                sub_stream = f"{fixed_stream_prefix}/subtraj:{sub_tid or sub_sid or s_idx}"
                _process_atif_trajectory(
                    sub,
                    fixed_role=resolved_sub_role,
                    fixed_stream_prefix=sub_stream,
                    parent_model=t_model,
                    parent_agent=t_agent,
                    sub_positional_roles=[],
                )

    processed_subtraj_session_ids: set[str] = set()
    if isinstance(traj_dict, dict):
        sub_pos_roles = [
            r for cid, r in ordered_dispatches if not cid or cid not in subtraj_linked_cids
        ]
        _process_atif_trajectory(
            traj_dict,
            fixed_role=None,
            fixed_stream_prefix="root",
            parent_model=model_name,
            parent_agent=eff_agent_top,
            sub_positional_roles=sub_pos_roles,
        )

    # Recover OpenCode SQLite sessions without double-counting root or ATIF-present child turns!
    if oc_db is not None:
        recover_sessions = (
            list(oc_db.sessions) if atif_agent_turns == 0 else list(oc_db.child_sessions)
        )
        for sess in recover_sessions:
            if sess.parent_id and sess.session_id in processed_subtraj_session_ids:
                continue
            is_child_sess = bool(sess.parent_id)
            s_role = sess.resolved_role if is_child_sess else root_role
            s_stream = f"opencode:{sess.session_id}"

            for db_turn in sess.turns:
                telemetry["n_turns"] += 1
                if is_child_sess:
                    recovered_child_turns += 1

                is_retry = stream_prev_had_error.get(s_stream, False)
                if is_retry:
                    telemetry["retry_turn_count"] += 1

                r_acc = role_accums.setdefault(s_role, _RoleAccumulator())
                r_acc.turns += 1
                if is_retry:
                    r_acc.retry_turns += 1

                turn_had_error = False
                for tp_part in db_turn.tool_parts:
                    telemetry["tool_calls_total"] += 1
                    r_acc.tool_calls += 1
                    if is_child_sess:
                        recovered_child_tool_calls += 1

                    fn_name = str(tp_part.get("tool") or "")
                    out_txt = str(tp_part.get("output") or "")
                    err_txt = str(tp_part.get("error") or "")
                    comb_txt = "\n".join(p for p in (out_txt, err_txt) if p)
                    oc_failed = bool(tp_part.get("status") == "error" or err_txt)

                    mcp_srv = classify_mcp_server(fn_name)
                    if mcp_srv is not None:
                        telemetry["mcp_calls_total"] += 1
                        r_acc.mcp_calls_total += 1
                        if mcp_srv == "ms-inspect":
                            telemetry["mcp_calls_inspect"] += 1
                        elif mcp_srv == "ms-modify":
                            telemetry["mcp_calls_modify"] += 1
                        elif mcp_srv == "ms-create":
                            telemetry["mcp_calls_create"] += 1

                    is_bash = fn_name.strip().lower() in BASH_TOOL_NAMES
                    bash_failed = False
                    if is_bash:
                        telemetry["bash_calls"] += 1
                        r_acc.bash_calls += 1
                        ec = tp_part.get("exit")
                        if ec is None:
                            ec = extract_exit_code({}, {}, comb_txt)
                        if (
                            (ec is not None and ec != 0)
                            or oc_failed
                            or is_tool_error({}, {}, comb_txt)
                        ):
                            telemetry["bash_nonzero_exits"] += 1
                            r_acc.bash_nonzero_exits += 1
                            bash_failed = True

                    low_db_fn = fn_name.strip().lower()
                    if low_db_fn not in PASSIVE_OR_DISPATCH_TOOL_NAMES and any(
                        p.lower() in comb_txt.lower() for p in CASA_SEVERE_PHRASES
                    ):
                        telemetry["severe_casa_errors"] += 1

                    is_proto = is_protocol_error(fn_name, mcp_srv, comb_txt, arm)
                    if is_proto:
                        telemetry["protocol_errors"] += 1

                    tool_failed = bool(
                        is_proto or oc_failed or is_tool_error({}, {}, comb_txt, fn_name=fn_name)
                    )
                    if tool_failed:
                        telemetry["tool_errors"] += 1
                        r_acc.tool_errors += 1

                    if bash_failed or tool_failed:
                        turn_had_error = True

                stream_prev_had_error[s_stream] = turn_had_error

                billed_out = db_turn.reply_output_tokens + db_turn.reasoning_tokens
                t_reas = db_turn.reasoning_tokens
                if (
                    db_turn.has_token_metrics
                    or db_turn.uncached_input_tokens > 0
                    or db_turn.cache_read_tokens > 0
                    or db_turn.cache_write_tokens > 0
                    or billed_out > 0
                ):
                    eff_m = _resolve_step_model_for_pricing(
                        db_turn.model_name, model_name, "opencode", models_cfg
                    )
                    tp_obj = price_turn(
                        eff_m,
                        uncached_input_tokens=db_turn.uncached_input_tokens,
                        cache_read_tokens=db_turn.cache_read_tokens,
                        cache_write_tokens=db_turn.cache_write_tokens,
                        output_tokens=billed_out,
                        reasoning_tokens=t_reas,
                        agent_name="opencode",
                        fallback_cost_usd=db_turn.cost_usd,
                        role=s_role,
                        stream_id=s_stream,
                        is_retry=is_retry,
                        models_cfg=models_cfg,
                    )
                    priced_turns.append(tp_obj)
                    turn_reasonings.append(t_reas)
                    r_acc.input_tokens += tp_obj.input_tokens
                    r_acc.output_tokens += tp_obj.output_tokens
                    r_acc.turn_reasonings.append(t_reas)
                    r_acc.total_usd_raw += tp_obj.total_usd
                    if is_retry:
                        r_acc.retry_usd_raw += tp_obj.total_usd

    role_breakdown_out = {rname: acc.to_dict() for rname, acc in role_accums.items()}
    return TrialExecutionAnalysis(
        telemetry=telemetry,
        priced_turns=priced_turns,
        turn_reasonings=turn_reasonings,
        role_breakdown=role_breakdown_out,
        opencode_db=oc_db,
        atif_agent_turns=atif_agent_turns,
        atif_tool_calls=atif_tool_calls,
        recovered_child_turns=recovered_child_turns,
        recovered_child_tool_calls=recovered_child_tool_calls,
    )


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
            reasoning_tokens=None,
            reply_output_tokens=None,
            retry_turn_count=0,
            first_pass_usd=0.0,
            retry_usd=0.0,
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
    ctx_reasonings: list[int] = []
    is_opencode = (agent_name or "").strip().lower() == "opencode"

    for ctx in contexts:
        mu = ctx.get("model_usage")
        if isinstance(mu, dict) and mu:
            for mu_model, usage in mu.items():
                if not isinstance(usage, dict):
                    continue
                n_in = safe_nonneg_int(usage.get("n_input_tokens"))
                n_cache = safe_nonneg_int(usage.get("n_cache_tokens"))
                raw_out = safe_nonneg_int(usage.get("n_output_tokens"))
                n_reas = safe_nonneg_int(usage.get("n_reasoning_tokens"))
                billed_out = (raw_out + n_reas) if is_opencode else raw_out
                fb = safe_nonneg_float(usage.get("cost_usd"))
                if n_in == 0 and n_cache == 0 and billed_out == 0 and fb is None:
                    continue
                ctx_turns.append(
                    price_turn(
                        mu_model,
                        uncached_input_tokens=max(0, n_in - n_cache),
                        cache_read_tokens=n_cache,
                        output_tokens=billed_out,
                        reasoning_tokens=n_reas,
                        agent_name=agent_name,
                        fallback_cost_usd=fb,
                        allow_cliff=single_turn,
                        models_cfg=models_cfg,
                    )
                )
                ctx_reasonings.append(n_reas)
        else:
            n_in = safe_nonneg_int(ctx.get("n_input_tokens"))
            n_cache = safe_nonneg_int(ctx.get("n_cache_tokens"))
            raw_out = safe_nonneg_int(ctx.get("n_output_tokens"))
            n_reas = safe_nonneg_int(ctx.get("n_reasoning_tokens"))
            billed_out = (raw_out + n_reas) if is_opencode else raw_out
            fb = safe_nonneg_float(ctx.get("cost_usd"))
            if n_in > 0 or n_cache > 0 or billed_out > 0 or fb is not None:
                ctx_turns.append(
                    price_turn(
                        model_name,
                        uncached_input_tokens=max(0, n_in - n_cache),
                        cache_read_tokens=n_cache,
                        output_tokens=billed_out,
                        reasoning_tokens=n_reas,
                        agent_name=agent_name,
                        fallback_cost_usd=fb,
                        allow_cliff=single_turn,
                        models_cfg=models_cfg,
                    )
                )
                ctx_reasonings.append(n_reas)

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
        tot_usd = round(in_usd + cw_usd + cr_usd + out_usd, 8)
        reas_tot, reply_tot = _finalize_reasoning_totals(out_tok, ctx_reasonings)
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
            total_usd=tot_usd,
            pricing_source=_resolve_pricing_source("agent_context", ctx_turns),
            reasoning_tokens=reas_tot,
            reply_output_tokens=reply_tot,
            retry_turn_count=0,
            first_pass_usd=tot_usd,
            retry_usd=0.0,
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
        reasoning_tokens=None,
        reply_output_tokens=None,
        retry_turn_count=0,
        first_pass_usd=0.0,
        retry_usd=0.0,
    )


def build_trial_pricing_from_execution(
    exec_analysis: TrialExecutionAnalysis,
    *,
    model_name: str | None,
    agent_name: str | None = None,
    trajectory_dict: dict[str, Any] | None = None,
    trial_result_dict: dict[str, Any] | None = None,
    models_cfg: ModelsConfig | None = None,
) -> TrialPricing:
    """Build `TrialPricing` from an already-computed `TrialExecutionAnalysis` (with fallbacks)."""
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
            reasoning_tokens=None,
            reply_output_tokens=None,
            retry_turn_count=0,
            first_pass_usd=0.0,
            retry_usd=0.0,
        )

    turns = exec_analysis.priced_turns
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
        tot_usd = round(in_usd + cw_usd + cr_usd + out_usd, 8)

        retry_cnt, first_usd, ret_usd = _finalize_retry_usd_split(tot_usd, turns)
        # Maintain exact 8-decimal identity first_pass_usd + retry_usd == total_usd
        eff_tot_usd = round(first_usd + ret_usd, 8)
        reas_tot, reply_tot = _finalize_reasoning_totals(out_tok, exec_analysis.turn_reasonings)

        default_src = (
            "opencode_sqlite"
            if (exec_analysis.atif_agent_turns == 0 and exec_analysis.opencode_db is not None)
            else "trajectory_steps"
        )
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
            total_usd=eff_tot_usd,
            pricing_source=_resolve_pricing_source(default_src, turns),
            reasoning_tokens=reas_tot,
            reply_output_tokens=reply_tot,
            retry_turn_count=retry_cnt,
            first_pass_usd=first_usd,
            retry_usd=ret_usd,
        )

    n_agent_steps = _count_trajectory_agent_steps(trajectory_dict)
    obs_retry_cnt = exec_analysis.telemetry.get("retry_turn_count", 0)

    # 2. Fallback: aggregate ATIF FinalMetrics when per-step metrics are absent.
    if isinstance(trajectory_dict, dict):
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
            raw_prompt = safe_nonneg_int(fm.get("total_prompt_tokens"))
            raw_out = safe_nonneg_int(fm.get("total_completion_tokens"))
            fm_reas = safe_nonneg_int(
                extra_fm.get("total_reasoning_tokens", extra_fm.get("reasoning_tokens"))
            )
            billed_out = (raw_out + fm_reas) if is_opencode else max(raw_out, fm_reas)

            if "total_cache_read_input_tokens" in extra_fm:
                c_read = safe_nonneg_int(extra_fm.get("total_cache_read_input_tokens"))
            else:
                c_read = safe_nonneg_int(fm.get("total_cached_tokens"))
            has_cc = "total_cache_creation_input_tokens" in extra_fm
            has_oc_write = ("total_cache_write_tokens" in extra_fm) and not has_cc
            if has_cc:
                c_write = safe_nonneg_int(extra_fm.get("total_cache_creation_input_tokens"))
            else:
                c_write = safe_nonneg_int(extra_fm.get("total_cache_write_tokens"))
            fb_usd = safe_nonneg_float(fm.get("total_cost_usd"))
            if raw_prompt > 0 or billed_out > 0 or c_read > 0 or c_write > 0 or fb_usd is not None:
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
                if eff_steps is None and safe_nonneg_int(fm.get("total_steps")) == 1:
                    eff_steps = 1
                single_turn = eff_steps == 1

                tp = price_turn(
                    model_name,
                    uncached_input_tokens=u_in,
                    cache_read_tokens=c_read,
                    cache_write_tokens=c_write,
                    output_tokens=billed_out,
                    reasoning_tokens=fm_reas,
                    agent_name=eff_agent,
                    fallback_cost_usd=fb_usd,
                    allow_cliff=single_turn,
                    models_cfg=models_cfg,
                )
                tot_u = round(tp.total_usd, 8)
                reas_tot, reply_tot = _finalize_reasoning_totals(billed_out, [fm_reas])
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
                    total_usd=tot_u,
                    pricing_source=_resolve_pricing_source("trajectory_final_metrics", [tp]),
                    reasoning_tokens=reas_tot,
                    reply_output_tokens=reply_tot,
                    retry_turn_count=obs_retry_cnt,
                    first_pass_usd=tot_u,
                    retry_usd=0.0,
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
        reasoning_tokens=None,
        reply_output_tokens=None,
        retry_turn_count=obs_retry_cnt,
        first_pass_usd=0.0,
        retry_usd=0.0,
    )


def compute_trial_pricing(
    *,
    model_name: str | None,
    agent_name: str | None = None,
    trajectory_dict: dict[str, Any] | None = None,
    trial_result_dict: dict[str, Any] | None = None,
    trial_dir: Path | None = None,
    opencode_txt_path: Path | None = None,
    opencode_db_path: Path | None = None,
    arm: str = "00_none",
    models_cfg: ModelsConfig | None = None,
) -> TrialPricing:
    """Compute per-trial token, reasoning, retry, and list-price USD metrics."""
    exec_analysis = analyze_trial_execution(
        trajectory_dict,
        model_name=model_name,
        agent_name=agent_name,
        arm=arm,
        trial_dir=trial_dir,
        opencode_txt_path=opencode_txt_path,
        opencode_db_path=opencode_db_path,
        models_cfg=models_cfg,
    )
    return build_trial_pricing_from_execution(
        exec_analysis,
        model_name=model_name,
        agent_name=agent_name,
        trajectory_dict=trajectory_dict,
        trial_result_dict=trial_result_dict,
        models_cfg=models_cfg,
    )
