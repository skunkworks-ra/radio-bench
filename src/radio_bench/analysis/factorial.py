"""2x2 factorial estimators, paired task bootstrap CIs, Wilson pass@1, parity, and Pareto frontiers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from radio_bench.analysis.collect import CANONICAL_ARMS, sanitize_json_tree
from radio_bench.analysis.pricing import normalize_model_key
from radio_bench.config import ModelsConfig, load_models_config

DEFAULT_N_BOOT = 10_000
DEFAULT_BOOT_SEED = 20260413
WILSON_Z_95 = 1.959963984540054

PARETO_ALL_5_ARMS: tuple[str, ...] = (
    "00_none",
    "01_skills_raw",
    "01_skills_casa",
    "10_tools",
    "11_full",
)

PARETO_E2E_6_ARMS: tuple[str, ...] = (
    "00_none",
    "01_skills_raw",
    "01_skills_casa",
    "10_tools",
    "11_full",
    "11_orchestrated",
)

PARETO_EPS = 1e-12


def resolve_parity_model_pairs(
    models_cfg: ModelsConfig | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Load `(pair_id, small_canonical_model, frontier_canonical_model)` from ModelsConfig."""
    cfg = models_cfg if models_cfg is not None else load_models_config()
    pairs: list[tuple[str, str, str]] = []
    for pair in cfg.parity_pairs:
        small_raw = cfg.resolve_model_name(pair.small.model)
        front_raw = cfg.resolve_model_name(pair.frontier.model)
        small_m = normalize_model_key(small_raw, models_cfg=cfg)
        front_m = normalize_model_key(front_raw, models_cfg=cfg)
        pairs.append((pair.pair_id, small_m, front_m))
    return tuple(pairs)


def __getattr__(name: str) -> Any:
    if name == "PARITY_MODEL_PAIRS":
        return resolve_parity_model_pairs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Linear contrast specifications: (name, formula_str, arm_weights, e2e_only).
CONTRAST_SPECS: tuple[tuple[str, str, dict[str, float], bool], ...] = (
    (
        "Delta_Tools_casa",
        "0.5 * ((R_11 - R_01_casa) + (R_10 - R_00))",
        {"11_full": 0.5, "01_skills_casa": -0.5, "10_tools": 0.5, "00_none": -0.5},
        False,
    ),
    (
        "Delta_Skills_casa",
        "0.5 * ((R_11 - R_10) + (R_01_casa - R_00))",
        {"11_full": 0.5, "10_tools": -0.5, "01_skills_casa": 0.5, "00_none": -0.5},
        False,
    ),
    (
        "I_casa",
        "R_11 - R_10 - R_01_casa + R_00",
        {"11_full": 1.0, "10_tools": -1.0, "01_skills_casa": -1.0, "00_none": 1.0},
        False,
    ),
    (
        "Delta_Tools_raw",
        "0.5 * ((R_11 - R_01_raw) + (R_10 - R_00))",
        {"11_full": 0.5, "01_skills_raw": -0.5, "10_tools": 0.5, "00_none": -0.5},
        False,
    ),
    (
        "Delta_Skills_raw",
        "0.5 * ((R_11 - R_10) + (R_01_raw - R_00))",
        {"11_full": 0.5, "10_tools": -0.5, "01_skills_raw": 0.5, "00_none": -0.5},
        False,
    ),
    (
        "I_raw",
        "R_11 - R_10 - R_01_raw + R_00",
        {"11_full": 1.0, "10_tools": -1.0, "01_skills_raw": -1.0, "00_none": 1.0},
        False,
    ),
    (
        "Delta_PromptAdapt",
        "R_01_casa - R_01_raw",
        {"01_skills_casa": 1.0, "01_skills_raw": -1.0},
        False,
    ),
    (
        "Delta_Orch",
        "R_11_orchestrated - R_11_full",
        {"11_orchestrated": 1.0, "11_full": -1.0},
        True,
    ),
)


def _finite_float(val: Any, default: float = 0.0) -> float:
    """Coerce `val` to a finite Python float, returning `default` on None/bool/NaN/Inf."""
    if isinstance(val, bool) or val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        fval = float(val)
        if math.isfinite(fval):
            return fval
    return default


def wilson_ci(k: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Compute Wilson score confidence interval `[low, high]` for `k` successes in `n` trials."""
    if isinstance(k, bool) or isinstance(n, bool) or n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial counts for wilson_ci: k={k}, n={n}")
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radicand = max(0.0, (p_hat * (1.0 - p_hat)) / n + z2 / (4.0 * n * n))
    half = (z / denom) * math.sqrt(radicand)

    low = 0.0 if k == 0 else max(0.0, center - half)
    high = 1.0 if k == n else min(1.0, center + half)
    return round(low, 8), round(high, 8)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimator `1 - comb(n - c, k) / comb(n, k)` when `n >= k >= 1`."""
    if k < 1 or n < k:
        return None
    c_clamped = max(0, min(int(c), int(n)))
    if n - c_clamped < k:
        return 1.0
    return round(1.0 - (math.comb(n - c_clamped, k) / math.comb(n, k)), 8)


def _ordered_present_arms(
    records: list[dict[str, Any]],
    arms_order: tuple[str, ...] = CANONICAL_ARMS,
) -> list[str]:
    present = {str(r.get("arm")) for r in records if r.get("arm")}
    ordered = [arm for arm in arms_order if arm in present]
    extras = sorted(present - set(ordered))
    return [*ordered, *extras]


def summarize_arm_cells(
    records: list[dict[str, Any]],
    arms_order: tuple[str, ...] = CANONICAL_ARMS,
) -> dict[str, dict[str, Any]]:
    """Compute cell summary statistics across all present arms for a set of trial records."""
    cells: dict[str, dict[str, Any]] = {}
    for arm in _ordered_present_arms(records, arms_order=arms_order):
        arm_recs = [r for r in records if str(r.get("arm")) == arm]
        n = len(arm_recs)
        if n == 0:
            continue

        rewards = np.array([_finite_float(r.get("reward")) for r in arm_recs], dtype=np.float64)
        passed_arr = np.array([int(bool(r.get("passed"))) for r in arm_recs], dtype=np.int64)
        k_pass = int(np.sum(passed_arr))
        w_low, w_high = wilson_ci(k_pass, n)

        # Unit-level grouping (base_task, pack_id, model_name, thinking_level) for task-equal metrics.
        by_unit: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for r in arm_recs:
            ukey = (
                str(r.get("base_task")),
                str(r.get("pack_id") or "ra_00a11ea"),
                str(r.get("model_name")),
                str(r.get("thinking_level") or "default"),
            )
            by_unit.setdefault(ukey, []).append(r)

        unit_reward_means = [
            float(np.mean([_finite_float(x.get("reward")) for x in u_recs]))
            for u_recs in by_unit.values()
        ]
        unit_pass_means = [
            float(np.mean([1.0 if bool(x.get("passed")) else 0.0 for x in u_recs]))
            for u_recs in by_unit.values()
        ]
        unit_usd_means = [
            float(np.mean([_finite_float(x.get("total_usd")) for x in u_recs]))
            for u_recs in by_unit.values()
        ]
        unit_tok_means = [
            float(np.mean([_finite_float(x.get("total_tokens")) for x in u_recs]))
            for u_recs in by_unit.values()
        ]
        unit_wall_means = [
            float(
                np.mean([_finite_float(x.get("total_wall_sec", x.get("wall_sec"))) for x in u_recs])
            )
            for u_recs in by_unit.values()
        ]

        p3_vals = [
            val
            for u_recs in by_unit.values()
            if (
                val := pass_at_k_unbiased(
                    len(u_recs),
                    sum(1 for x in u_recs if bool(x.get("passed"))),
                    3,
                )
            )
            is not None
        ]
        p5_vals = [
            val
            for u_recs in by_unit.values()
            if (
                val := pass_at_k_unbiased(
                    len(u_recs),
                    sum(1 for x in u_recs if bool(x.get("passed"))),
                    5,
                )
            )
            is not None
        ]

        in_tok = np.array(
            [_finite_float(r.get("input_tokens")) for r in arm_recs], dtype=np.float64
        )
        out_tok = np.array(
            [_finite_float(r.get("output_tokens")) for r in arm_recs], dtype=np.float64
        )
        tot_tok = np.array(
            [_finite_float(r.get("total_tokens")) for r in arm_recs], dtype=np.float64
        )
        peak_tok = np.array(
            [_finite_float(r.get("peak_turn_tokens")) for r in arm_recs], dtype=np.float64
        )
        usd_arr = np.array([_finite_float(r.get("total_usd")) for r in arm_recs], dtype=np.float64)
        wall_arr = np.array(
            [_finite_float(r.get("total_wall_sec", r.get("wall_sec"))) for r in arm_recs],
            dtype=np.float64,
        )
        agent_wall_arr = np.array(
            [_finite_float(r.get("agent_wall_sec")) for r in arm_recs], dtype=np.float64
        )
        ver_wall_arr = np.array(
            [_finite_float(r.get("verifier_wall_sec")) for r in arm_recs], dtype=np.float64
        )
        turns_arr = np.array([_finite_float(r.get("n_turns")) for r in arm_recs], dtype=np.float64)
        cliff_arr = np.array(
            [1.0 if r.get("haiku_cliff_triggered") else 0.0 for r in arm_recs], dtype=np.float64
        )

        total_usd_sum = float(np.sum(usd_arr))
        cost_per_pass = round(total_usd_sum / k_pass, 8) if k_pass > 0 else None
        p1_te_raw = float(np.mean(unit_pass_means))
        usd_te_raw = float(np.mean(unit_usd_means))
        cost_per_pass_te = round(usd_te_raw / p1_te_raw, 8) if p1_te_raw > 0.0 else None
        arm_is_e2e_only = all(
            bool(r.get("is_e2e")) or str(r.get("stage")) == "e2e" for r in arm_recs
        )

        cells[arm] = {
            "arm": arm,
            "e2e_only": arm_is_e2e_only,
            "n_trials": n,
            "n_tasks": len({str(r.get("base_task")) for r in arm_recs}),
            "n_models": len({str(r.get("model_name")) for r in arm_recs}),
            "n_passed": k_pass,
            "reward_mean": round(float(np.mean(rewards)), 8),
            "reward_task_equal_mean": round(float(np.mean(unit_reward_means)), 8),
            "reward_median": round(float(np.median(rewards)), 8),
            "reward_sd": round(float(np.std(rewards, ddof=1)), 8) if n >= 2 else 0.0,
            "pass_at_1": round(k_pass / n, 8),
            "pass_at_1_task_equal": round(p1_te_raw, 8),
            "pass_at_1_wilson_ci": [w_low, w_high],
            "pass_at_3": round(float(np.mean(p3_vals)), 8) if p3_vals else None,
            "pass_at_5": round(float(np.mean(p5_vals)), 8) if p5_vals else None,
            "mean_input_tokens": round(float(np.mean(in_tok)), 3),
            "mean_output_tokens": round(float(np.mean(out_tok)), 3),
            "mean_total_tokens": round(float(np.mean(tot_tok)), 3),
            "mean_total_tokens_task_equal": round(float(np.mean(unit_tok_means)), 3),
            "mean_peak_turn_tokens": round(float(np.mean(peak_tok)), 3),
            "mean_usd": round(float(np.mean(usd_arr)), 8),
            "mean_usd_task_equal": round(usd_te_raw, 8),
            "total_usd": round(total_usd_sum, 8),
            "cost_per_passing_run_usd": cost_per_pass,
            "cost_per_passing_run_usd_task_equal": cost_per_pass_te,
            "mean_wall_sec": round(float(np.mean(wall_arr)), 4),
            "mean_wall_sec_task_equal": round(float(np.mean(unit_wall_means)), 4),
            "mean_agent_wall_sec": round(float(np.mean(agent_wall_arr)), 4),
            "mean_verifier_wall_sec": round(float(np.mean(ver_wall_arr)), 4),
            "mean_turns": round(float(np.mean(turns_arr)), 3),
            "haiku_cliff_rate": round(float(np.mean(cliff_arr)), 6),
        }
    return cells


def _bootstrap_paired_linear_contrast(
    records: list[dict[str, Any]],
    *,
    name: str,
    formula: str,
    arm_weights: dict[str, float],
    e2e_only: bool,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Compute a paired linear contrast across `(base_task, pack_id, model, thinking_level)` units."""
    filtered = (
        [r for r in records if bool(r.get("is_e2e")) or str(r.get("stage")) == "e2e"]
        if e2e_only
        else list(records)
    )
    req_arms = sorted(arm_weights.keys())

    # Group into (base_task, pack_id, model_name, thinking_level) -> arm -> list of (reward, passed).
    unit_arm_vals: dict[tuple[str, str, str, str], dict[str, list[tuple[float, float]]]] = {}
    for r in filtered:
        arm = str(r.get("arm"))
        if arm not in arm_weights:
            continue
        ukey = (
            str(r.get("base_task")),
            str(r.get("pack_id") or "ra_00a11ea"),
            str(r.get("model_name")),
            str(r.get("thinking_level") or "default"),
        )
        unit_arm_vals.setdefault(ukey, {}).setdefault(arm, []).append(
            (_finite_float(r.get("reward")), 1.0 if bool(r.get("passed")) else 0.0)
        )

    complete_units = sorted(
        ukey
        for ukey, arm_map in unit_arm_vals.items()
        if all(a in arm_map and len(arm_map[a]) > 0 for a in req_arms)
    )
    if not complete_units:
        return {
            "name": name,
            "formula": formula,
            "status": "insufficient_arms",
            "required_arms": req_arms,
            "e2e_only": e2e_only,
            "n_paired_units": 0,
            "n_paired_tasks": 0,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "boot_se": None,
            "pass_at_1_estimate": None,
            "pass_at_1_ci_low": None,
            "pass_at_1_ci_high": None,
        }

    arm_reward_means: dict[str, float] = {}
    arm_pass_means: dict[str, float] = {}
    for arm in req_arms:
        arm_reward_means[arm] = float(
            np.mean([np.mean([x[0] for x in unit_arm_vals[u][arm]]) for u in complete_units])
        )
        arm_pass_means[arm] = float(
            np.mean([np.mean([x[1] for x in unit_arm_vals[u][arm]]) for u in complete_units])
        )

    point_est = sum(w * arm_reward_means[a] for a, w in arm_weights.items())
    pass_est = sum(w * arm_pass_means[a] for a, w in arm_weights.items())

    tasks_sorted = sorted({u[0] for u in complete_units})
    units_by_task: dict[str, list[tuple[str, str, str, str]]] = {t: [] for t in tasks_sorted}
    for u in complete_units:
        units_by_task[u[0]].append(u)

    k_tasks = len(tasks_sorted)
    if n_boot <= 0:
        return {
            "name": name,
            "formula": formula,
            "status": "ok",
            "required_arms": req_arms,
            "e2e_only": e2e_only,
            "n_paired_units": len(complete_units),
            "n_paired_tasks": k_tasks,
            "estimate": round(point_est, 10),
            "ci_low": round(point_est, 10),
            "ci_high": round(point_est, 10),
            "boot_se": 0.0,
            "pass_at_1_estimate": round(pass_est, 10),
            "pass_at_1_ci_low": round(pass_est, 10),
            "pass_at_1_ci_high": round(pass_est, 10),
        }

    rng = np.random.default_rng(seed)

    if k_tasks == 1:
        cluster_idx = np.zeros((n_boot, 1), dtype=np.int64)
    else:
        cluster_idx = rng.integers(0, k_tasks, size=(n_boot, k_tasks))

    flat_clusters = cluster_idx.ravel()
    total_slots = flat_clusters.size
    slot_reward_sum = np.zeros(total_slots, dtype=np.float64)
    slot_pass_sum = np.zeros(total_slots, dtype=np.float64)
    task_unit_counts = np.zeros(k_tasks, dtype=np.float64)

    for k_idx, task_id in enumerate(tasks_sorted):
        t_units = units_by_task[task_id]
        task_unit_counts[k_idx] = float(len(t_units))
        pos = np.flatnonzero(flat_clusters == k_idx)
        m_slots = int(pos.size)
        if m_slots == 0:
            continue
        for u in t_units:
            for arm in req_arms:
                w = arm_weights[arm]
                pairs = unit_arm_vals[u][arm]
                n_att = len(pairs)
                r_vec = np.array([p[0] for p in pairs], dtype=np.float64)
                p_vec = np.array([p[1] for p in pairs], dtype=np.float64)
                if n_att == 1:
                    slot_reward_sum[pos] += w * r_vec[0]
                    slot_pass_sum[pos] += w * p_vec[0]
                else:
                    att_idx = rng.integers(0, n_att, size=(m_slots, n_att))
                    slot_reward_sum[pos] += w * r_vec[att_idx].mean(axis=1)
                    slot_pass_sum[pos] += w * p_vec[att_idx].mean(axis=1)

    sampled_counts = task_unit_counts[cluster_idx].sum(axis=1)
    boot_reward = slot_reward_sum.reshape(n_boot, k_tasks).sum(axis=1) / sampled_counts
    boot_pass = slot_pass_sum.reshape(n_boot, k_tasks).sum(axis=1) / sampled_counts

    r_low, r_high = np.percentile(boot_reward, [2.5, 97.5])
    p_low, p_high = np.percentile(boot_pass, [2.5, 97.5])
    boot_se = float(np.std(boot_reward, ddof=1)) if n_boot >= 2 else 0.0

    return {
        "name": name,
        "formula": formula,
        "status": "ok",
        "required_arms": req_arms,
        "e2e_only": e2e_only,
        "n_paired_units": len(complete_units),
        "n_paired_tasks": k_tasks,
        "estimate": round(float(point_est), 10),
        "ci_low": round(float(r_low), 10),
        "ci_high": round(float(r_high), 10),
        "boot_se": round(boot_se, 10),
        "pass_at_1_estimate": round(float(pass_est), 10),
        "pass_at_1_ci_low": round(float(p_low), 10),
        "pass_at_1_ci_high": round(float(p_high), 10),
    }


def compute_stratum_contrasts(
    records: list[dict[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
) -> dict[str, dict[str, Any]]:
    """Compute all 8 SPEC.md Section 5.2 factorial, prompt-adapt, and orchestration contrasts."""
    out: dict[str, dict[str, Any]] = {}
    for idx, (name, formula, weights, e2e_only) in enumerate(CONTRAST_SPECS):
        out[name] = _bootstrap_paired_linear_contrast(
            records,
            name=name,
            formula=formula,
            arm_weights=weights,
            e2e_only=e2e_only,
            n_boot=n_boot,
            seed=seed + idx * 1009,
        )
    return out


def _paired_pass_at_k(
    paired_units: list[tuple[str, str]],
    small_by_unit: dict[tuple[str, str], list[dict[str, Any]]],
    front_by_unit: dict[tuple[str, str], list[dict[str, Any]]],
    k: int,
) -> tuple[float | None, float | None, float | None]:
    s_vals: list[float] = []
    f_vals: list[float] = []
    gaps: list[float] = []
    for u in paired_units:
        s_recs = small_by_unit[u]
        f_recs = front_by_unit[u]
        vs = pass_at_k_unbiased(len(s_recs), sum(1 for x in s_recs if bool(x.get("passed"))), k)
        vf = pass_at_k_unbiased(len(f_recs), sum(1 for x in f_recs if bool(x.get("passed"))), k)
        if vs is not None:
            s_vals.append(vs)
        if vf is not None:
            f_vals.append(vf)
        if vs is not None and vf is not None:
            gaps.append(vs - vf)

    s_mean = round(float(np.mean(s_vals)), 8) if s_vals else None
    f_mean = round(float(np.mean(f_vals)), 8) if f_vals else None
    gap_mean = round(float(np.mean(gaps)), 10) if gaps else None
    return s_mean, f_mean, gap_mean


def _compute_single_parity(
    records: list[dict[str, Any]],
    *,
    pair_id: str,
    small_model: str,
    small_arm: str,
    frontier_model: str,
    frontier_arm: str,
    small_thinking_level: str = "default",
    frontier_thinking_level: str = "default",
    e2e_only: bool,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Compare `(small_model, small_arm, small_thinking)` vs `(frontier_model, frontier_arm, frontier_thinking)`."""
    filtered = (
        [r for r in records if bool(r.get("is_e2e")) or str(r.get("stage")) == "e2e"]
        if e2e_only
        else list(records)
    )
    small_recs = [
        r
        for r in filtered
        if str(r.get("model_name")) == small_model
        and str(r.get("arm")) == small_arm
        and str(r.get("thinking_level") or "default") == small_thinking_level
    ]
    front_recs = [
        r
        for r in filtered
        if str(r.get("model_name")) == frontier_model
        and str(r.get("arm")) == frontier_arm
        and str(r.get("thinking_level") or "default") == frontier_thinking_level
    ]

    small_by_unit: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in small_recs:
        ukey = (str(r.get("base_task")), str(r.get("pack_id") or "ra_00a11ea"))
        small_by_unit.setdefault(ukey, []).append(r)

    front_by_unit: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in front_recs:
        ukey = (str(r.get("base_task")), str(r.get("pack_id") or "ra_00a11ea"))
        front_by_unit.setdefault(ukey, []).append(r)

    paired_units = sorted(set(small_by_unit) & set(front_by_unit))
    if not paired_units:
        return {
            "pair_id": pair_id,
            "small_model": small_model,
            "small_arm": small_arm,
            "small_thinking_level": small_thinking_level,
            "frontier_model": frontier_model,
            "frontier_arm": frontier_arm,
            "frontier_thinking_level": frontier_thinking_level,
            "status": "insufficient_models",
            "n_paired_tasks": 0,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "boot_se": None,
            "pass_at_1_diff": None,
            "pass_at_1_ci95": None,
            "small_reward_mean": None,
            "frontier_reward_mean": None,
            "large_reward_mean": None,
            "small_pass_at_1": None,
            "frontier_pass_at_1": None,
            "large_pass_at_1": None,
            "small_pass_at_3": None,
            "frontier_pass_at_3": None,
            "large_pass_at_3": None,
            "pass_at_3_gap": None,
            "small_pass_at_5": None,
            "frontier_pass_at_5": None,
            "large_pass_at_5": None,
            "pass_at_5_gap": None,
            "small_mean_usd": None,
            "frontier_mean_usd": None,
            "large_mean_usd": None,
            "cost_ratio": None,
            "small_cost_per_pass_usd": None,
            "frontier_cost_per_pass_usd": None,
            "large_cost_per_pass_usd": None,
            "cost_per_pass_ratio": None,
            "small_mean_tokens": None,
            "frontier_mean_tokens": None,
            "large_mean_tokens": None,
            "token_ratio": None,
            "small_mean_wall_sec": None,
            "frontier_mean_wall_sec": None,
            "large_mean_wall_sec": None,
            "wall_sec_ratio": None,
            "wall_time_ratio": None,
        }

    s_task_r = [
        float(np.mean([_finite_float(x.get("reward")) for x in small_by_unit[u]]))
        for u in paired_units
    ]
    f_task_r = [
        float(np.mean([_finite_float(x.get("reward")) for x in front_by_unit[u]]))
        for u in paired_units
    ]
    s_task_p = [
        float(np.mean([1.0 if bool(x.get("passed")) else 0.0 for x in small_by_unit[u]]))
        for u in paired_units
    ]
    f_task_p = [
        float(np.mean([1.0 if bool(x.get("passed")) else 0.0 for x in front_by_unit[u]]))
        for u in paired_units
    ]
    s_task_usd = [
        float(np.mean([_finite_float(x.get("total_usd")) for x in small_by_unit[u]]))
        for u in paired_units
    ]
    f_task_usd = [
        float(np.mean([_finite_float(x.get("total_usd")) for x in front_by_unit[u]]))
        for u in paired_units
    ]
    s_task_tok = [
        float(np.mean([_finite_float(x.get("total_tokens")) for x in small_by_unit[u]]))
        for u in paired_units
    ]
    f_task_tok = [
        float(np.mean([_finite_float(x.get("total_tokens")) for x in front_by_unit[u]]))
        for u in paired_units
    ]
    s_task_wall = [
        float(
            np.mean(
                [
                    _finite_float(x.get("total_wall_sec", x.get("wall_sec")))
                    for x in small_by_unit[u]
                ]
            )
        )
        for u in paired_units
    ]
    f_task_wall = [
        float(
            np.mean(
                [
                    _finite_float(x.get("total_wall_sec", x.get("wall_sec")))
                    for x in front_by_unit[u]
                ]
            )
        )
        for u in paired_units
    ]

    s_rmean = float(np.mean(s_task_r))
    f_rmean = float(np.mean(f_task_r))
    delta_parity = s_rmean - f_rmean
    s_pmean = float(np.mean(s_task_p))
    f_pmean = float(np.mean(f_task_p))
    pass_diff = s_pmean - f_pmean

    s_p3, f_p3, p3_gap = _paired_pass_at_k(paired_units, small_by_unit, front_by_unit, 3)
    s_p5, f_p5, p5_gap = _paired_pass_at_k(paired_units, small_by_unit, front_by_unit, 5)

    s_usd = float(np.mean(s_task_usd))
    f_usd = float(np.mean(f_task_usd))
    cost_ratio = round(s_usd / f_usd, 8) if f_usd > 0.0 else None

    s_cpp = round(s_usd / s_pmean, 8) if s_pmean > 0.0 else None
    f_cpp = round(f_usd / f_pmean, 8) if f_pmean > 0.0 else None
    cpp_ratio = (
        round(s_cpp / f_cpp, 8)
        if (s_cpp is not None and f_cpp is not None and f_cpp > 0.0)
        else None
    )

    s_tok = float(np.mean(s_task_tok))
    f_tok = float(np.mean(f_task_tok))
    tok_ratio = round(s_tok / f_tok, 8) if f_tok > 0.0 else None

    s_wall = float(np.mean(s_task_wall))
    f_wall = float(np.mean(f_task_wall))
    wall_ratio = round(s_wall / f_wall, 8) if f_wall > 0.0 else None

    tasks_sorted = sorted({u[0] for u in paired_units})
    units_by_task: dict[str, list[tuple[str, str]]] = {t: [] for t in tasks_sorted}
    for u in paired_units:
        units_by_task[u[0]].append(u)

    k_tasks = len(tasks_sorted)
    if n_boot <= 0:
        ci_low, ci_high = delta_parity, delta_parity
        p1_ci = [round(pass_diff, 10), round(pass_diff, 10)]
        boot_se = 0.0
    else:
        rng = np.random.default_rng(seed)
        if k_tasks == 1:
            c_idx = np.zeros((n_boot, 1), dtype=np.int64)
        else:
            c_idx = rng.integers(0, k_tasks, size=(n_boot, k_tasks))

        flat_c = c_idx.ravel()
        slot_r_diff = np.zeros(flat_c.size, dtype=np.float64)
        slot_p_diff = np.zeros(flat_c.size, dtype=np.float64)
        task_unit_counts = np.zeros(k_tasks, dtype=np.float64)

        for k_idx, task_id in enumerate(tasks_sorted):
            t_units = units_by_task[task_id]
            task_unit_counts[k_idx] = float(len(t_units))
            pos = np.flatnonzero(flat_c == k_idx)
            m_slots = int(pos.size)
            if m_slots == 0:
                continue

            for u in t_units:
                s_r_vec = np.array(
                    [_finite_float(x.get("reward")) for x in small_by_unit[u]], dtype=np.float64
                )
                s_p_vec = np.array(
                    [1.0 if bool(x.get("passed")) else 0.0 for x in small_by_unit[u]],
                    dtype=np.float64,
                )
                f_r_vec = np.array(
                    [_finite_float(x.get("reward")) for x in front_by_unit[u]], dtype=np.float64
                )
                f_p_vec = np.array(
                    [1.0 if bool(x.get("passed")) else 0.0 for x in front_by_unit[u]],
                    dtype=np.float64,
                )

                if len(s_r_vec) == 1:
                    s_r_b = s_r_vec[0]
                    s_p_b = s_p_vec[0]
                else:
                    s_att = rng.integers(0, len(s_r_vec), size=(m_slots, len(s_r_vec)))
                    s_r_b = s_r_vec[s_att].mean(axis=1)
                    s_p_b = s_p_vec[s_att].mean(axis=1)

                if len(f_r_vec) == 1:
                    f_r_b = f_r_vec[0]
                    f_p_b = f_p_vec[0]
                else:
                    f_att = rng.integers(0, len(f_r_vec), size=(m_slots, len(f_r_vec)))
                    f_r_b = f_r_vec[f_att].mean(axis=1)
                    f_p_b = f_p_vec[f_att].mean(axis=1)

                slot_r_diff[pos] += s_r_b - f_r_b
                slot_p_diff[pos] += s_p_b - f_p_b

        sampled_counts = task_unit_counts[c_idx].sum(axis=1)
        boot_delta = slot_r_diff.reshape(n_boot, k_tasks).sum(axis=1) / sampled_counts
        boot_p_delta = slot_p_diff.reshape(n_boot, k_tasks).sum(axis=1) / sampled_counts
        r_lo, r_hi = np.percentile(boot_delta, [2.5, 97.5])
        p_lo, p_hi = np.percentile(boot_p_delta, [2.5, 97.5])
        ci_low, ci_high = float(r_lo), float(r_hi)
        p1_ci = [round(float(p_lo), 10), round(float(p_hi), 10)]
        boot_se = float(np.std(boot_delta, ddof=1)) if n_boot >= 2 else 0.0

    return {
        "pair_id": pair_id,
        "small_model": small_model,
        "small_arm": small_arm,
        "small_thinking_level": small_thinking_level,
        "frontier_model": frontier_model,
        "frontier_arm": frontier_arm,
        "frontier_thinking_level": frontier_thinking_level,
        "status": "ok",
        "n_paired_tasks": k_tasks,
        "estimate": round(delta_parity, 10),
        "ci_low": round(ci_low, 10),
        "ci_high": round(ci_high, 10),
        "boot_se": round(boot_se, 10),
        "pass_at_1_diff": round(pass_diff, 10),
        "pass_at_1_ci95": p1_ci,
        "small_reward_mean": round(s_rmean, 8),
        "frontier_reward_mean": round(f_rmean, 8),
        "large_reward_mean": round(f_rmean, 8),
        "small_pass_at_1": round(s_pmean, 8),
        "frontier_pass_at_1": round(f_pmean, 8),
        "large_pass_at_1": round(f_pmean, 8),
        "small_pass_at_3": s_p3,
        "frontier_pass_at_3": f_p3,
        "large_pass_at_3": f_p3,
        "pass_at_3_gap": p3_gap,
        "small_pass_at_5": s_p5,
        "frontier_pass_at_5": f_p5,
        "large_pass_at_5": f_p5,
        "pass_at_5_gap": p5_gap,
        "small_mean_usd": round(s_usd, 8),
        "frontier_mean_usd": round(f_usd, 8),
        "large_mean_usd": round(f_usd, 8),
        "cost_ratio": cost_ratio,
        "small_cost_per_pass_usd": s_cpp,
        "frontier_cost_per_pass_usd": f_cpp,
        "large_cost_per_pass_usd": f_cpp,
        "cost_per_pass_ratio": cpp_ratio,
        "small_mean_tokens": round(s_tok, 3),
        "frontier_mean_tokens": round(f_tok, 3),
        "large_mean_tokens": round(f_tok, 3),
        "token_ratio": tok_ratio,
        "small_mean_wall_sec": round(s_wall, 4),
        "frontier_mean_wall_sec": round(f_wall, 4),
        "large_mean_wall_sec": round(f_wall, 4),
        "wall_sec_ratio": wall_ratio,
        "wall_time_ratio": wall_ratio,
    }


def compute_parity_summary(
    records: list[dict[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
    models_cfg: ModelsConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute Small + Full Stack vs Frontier Bare parity contrasts across model families."""
    cfg = models_cfg if models_cfg is not None else load_models_config()
    out: dict[str, dict[str, Any]] = {}
    for idx, pair in enumerate(cfg.parity_pairs):
        small_raw = cfg.resolve_model_name(pair.small.model)
        front_raw = cfg.resolve_model_name(pair.frontier.model)
        small_m = normalize_model_key(small_raw, models_cfg=cfg)
        front_m = normalize_model_key(front_raw, models_cfg=cfg)
        small_arm = pair.small.arm
        front_arm = pair.frontier.arm
        small_tl = str(pair.small.thinking_level or "default").strip() or "default"
        front_tl = str(pair.frontier.thinking_level or "default").strip() or "default"
        primary_e2e_only = small_arm == "11_orchestrated" or front_arm == "11_orchestrated"

        base_key = f"{pair.pair_id}__{small_arm}_vs_{front_arm}"
        if base_key in out:
            base_key = f"{pair.pair_id}__{small_arm}_{small_tl}_vs_{front_arm}_{front_tl}"

        out[base_key] = _compute_single_parity(
            records,
            pair_id=pair.pair_id,
            small_model=small_m,
            small_arm=small_arm,
            frontier_model=front_m,
            frontier_arm=front_arm,
            small_thinking_level=small_tl,
            frontier_thinking_level=front_tl,
            e2e_only=primary_e2e_only,
            n_boot=n_boot,
            seed=seed + 50_000 + idx * 997,
        )
        if small_arm == "11_full":
            orch_key = f"{pair.pair_id}__11_orchestrated_vs_{front_arm}"
            if orch_key in out:
                orch_key = f"{pair.pair_id}__11_orchestrated_{small_tl}_vs_{front_arm}_{front_tl}"
            out[orch_key] = _compute_single_parity(
                records,
                pair_id=pair.pair_id,
                small_model=small_m,
                small_arm="11_orchestrated",
                frontier_model=front_m,
                frontier_arm=front_arm,
                small_thinking_level=small_tl,
                frontier_thinking_level=front_tl,
                e2e_only=True,
                n_boot=n_boot,
                seed=seed + 60_000 + idx * 997,
            )
    return out


def _is_oracle_record(r: dict[str, Any]) -> bool:
    return (
        str(r.get("arm") or "").strip().lower() == "oracle"
        or str(r.get("model_name") or "").strip().lower() == "oracle"
        or str(r.get("agent_name") or "").strip().lower() == "oracle"
    )


def _compute_pareto_cohort(
    records: list[dict[str, Any]],
    *,
    cohort_name: str,
    allowed_arms: tuple[str, ...],
    e2e_only: bool,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Classify Pareto-optimal `(pack_id, model_name, thinking_level, arm)` cells on common paired tasks."""
    llm_recs = [r for r in records if not _is_oracle_record(r)]
    if not llm_recs:
        return {
            "cohort": cohort_name,
            "status": "no_llm_trials",
            "allowed_arms": list(allowed_arms),
            "e2e_only": e2e_only,
            "n_cells": 0,
            "n_common_tasks": 0,
            "common_tasks": [],
            "cells": [],
            "by_cell": {},
            "reward_frontier_cells": [],
            "pass_frontier_cells": [],
        }

    scope_recs = (
        [r for r in llm_recs if bool(r.get("is_e2e")) or str(r.get("stage")) == "e2e"]
        if e2e_only
        else llm_recs
    )
    if not scope_recs:
        return {
            "cohort": cohort_name,
            "status": "no_e2e_trials" if e2e_only else "no_cells",
            "allowed_arms": list(allowed_arms),
            "e2e_only": e2e_only,
            "n_cells": 0,
            "n_common_tasks": 0,
            "common_tasks": [],
            "cells": [],
            "by_cell": {},
            "reward_frontier_cells": [],
            "pass_frontier_cells": [],
        }

    allowed_set = set(allowed_arms)
    cohort_recs = [r for r in scope_recs if str(r.get("arm")) in allowed_set]
    if not cohort_recs:
        return {
            "cohort": cohort_name,
            "status": "no_cells",
            "allowed_arms": list(allowed_arms),
            "e2e_only": e2e_only,
            "n_cells": 0,
            "n_common_tasks": 0,
            "common_tasks": [],
            "cells": [],
            "by_cell": {},
            "reward_frontier_cells": [],
            "pass_frontier_cells": [],
        }

    arm_rank = {a: idx for idx, a in enumerate(allowed_arms)}
    # Cell key: (pack_id, model_name, thinking_level, arm) -> task_id -> list[(reward, passed, usd)]
    cell_task_vals: dict[
        tuple[str, str, str, str], dict[str, list[tuple[float, float, float]]]
    ] = {}
    for r in cohort_recs:
        b_task = str(r.get("base_task") or "").strip()
        if not b_task:
            continue
        ckey = (
            str(r.get("pack_id") or "ra_00a11ea"),
            str(r.get("model_name") or "unknown"),
            str(r.get("thinking_level") or "default"),
            str(r.get("arm")),
        )
        cell_task_vals.setdefault(ckey, {}).setdefault(b_task, []).append(
            (
                _finite_float(r.get("reward")),
                1.0 if bool(r.get("passed")) else 0.0,
                _finite_float(r.get("total_usd")),
            )
        )

    sorted_cell_keys = sorted(
        cell_task_vals.keys(),
        key=lambda ck: (ck[0], ck[1], ck[2], arm_rank.get(ck[3], 999), ck[3]),
    )
    if not sorted_cell_keys:
        return {
            "cohort": cohort_name,
            "status": "no_cells",
            "allowed_arms": list(allowed_arms),
            "e2e_only": e2e_only,
            "n_cells": 0,
            "n_common_tasks": 0,
            "common_tasks": [],
            "cells": [],
            "by_cell": {},
            "reward_frontier_cells": [],
            "pass_frontier_cells": [],
        }

    common_tasks = sorted(
        set.intersection(*(set(cell_task_vals[ck].keys()) for ck in sorted_cell_keys))
    )
    if not common_tasks:
        return {
            "cohort": cohort_name,
            "status": "empty_task_intersection",
            "allowed_arms": list(allowed_arms),
            "e2e_only": e2e_only,
            "n_cells": len(sorted_cell_keys),
            "n_common_tasks": 0,
            "common_tasks": [],
            "cells": [],
            "by_cell": {},
            "reward_frontier_cells": [],
            "pass_frontier_cells": [],
        }

    distinct_packs = {ck[0] for ck in sorted_cell_keys}
    multi_pack = len(distinct_packs) > 1

    n_cells = len(sorted_cell_keys)
    k_tasks = len(common_tasks)

    reward_te = np.zeros(n_cells, dtype=np.float64)
    pass_te = np.zeros(n_cells, dtype=np.float64)
    usd_te = np.zeros(n_cells, dtype=np.float64)
    cell_trial_counts: list[int] = []

    # Pre-materialize numpy vectors per (cell_idx, task_idx) for fast vectorized bootstrap
    cell_task_arrays: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for c_idx, ck in enumerate(sorted_cell_keys):
        t_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        r_means: list[float] = []
        p_means: list[float] = []
        u_means: list[float] = []
        tot_att = 0
        for tid in common_tasks:
            triples = cell_task_vals[ck][tid]
            tot_att += len(triples)
            rv = np.array([x[0] for x in triples], dtype=np.float64)
            pv = np.array([x[1] for x in triples], dtype=np.float64)
            uv = np.array([x[2] for x in triples], dtype=np.float64)
            t_list.append((rv, pv, uv))
            r_means.append(float(np.mean(rv)))
            p_means.append(float(np.mean(pv)))
            u_means.append(float(np.mean(uv)))
        cell_task_arrays.append(t_list)
        cell_trial_counts.append(tot_att)
        reward_te[c_idx] = float(np.mean(r_means))
        pass_te[c_idx] = float(np.mean(p_means))
        usd_te[c_idx] = float(np.mean(u_means))

    # Pair A (`reward`): (mean_usd_task_equal, reward_task_equal_mean)
    c_i = usd_te[:, None]
    c_j = usd_te[None, :]
    r_i = reward_te[:, None]
    r_j = reward_te[None, :]
    dom_reward = (
        (c_j <= c_i + PARETO_EPS)
        & (r_j >= r_i - PARETO_EPS)
        & ((c_j < c_i - PARETO_EPS) | (r_j > r_i + PARETO_EPS))
    )
    is_pareto_reward = ~np.any(dom_reward, axis=1)

    # Pair B (`pass`): (cost_per_passing_run_usd_task_equal, pass_at_1_task_equal)
    valid_pass = pass_te > 0.0
    cpp_arr = np.where(valid_pass, usd_te / np.maximum(pass_te, 1e-15), np.inf)
    cpp_i = cpp_arr[:, None]
    cpp_j = cpp_arr[None, :]
    p_i = pass_te[:, None]
    p_j = pass_te[None, :]
    dom_pass = (
        valid_pass[None, :]
        & (cpp_j <= cpp_i + PARETO_EPS)
        & (p_j >= p_i - PARETO_EPS)
        & ((cpp_j < cpp_i - PARETO_EPS) | (p_j > p_i + PARETO_EPS))
    )
    is_pareto_pass = valid_pass & (~np.any(dom_pass, axis=1))

    # Vectorized 2-stage cluster + attempt bootstrap for frontier membership probabilities
    if n_boot <= 0:
        boot_prob_reward = is_pareto_reward.astype(np.float64)
        boot_prob_pass = is_pareto_pass.astype(np.float64)
    else:
        rng = np.random.default_rng(seed)
        if k_tasks == 1:
            cluster_idx = np.zeros((n_boot, 1), dtype=np.int64)
        else:
            cluster_idx = rng.integers(0, k_tasks, size=(n_boot, k_tasks))

        flat_c = cluster_idx.ravel()
        n_slots = flat_c.size
        slot_r = np.zeros((n_slots, n_cells), dtype=np.float64)
        slot_p = np.zeros((n_slots, n_cells), dtype=np.float64)
        slot_u = np.zeros((n_slots, n_cells), dtype=np.float64)

        for k_idx in range(k_tasks):
            pos = np.flatnonzero(flat_c == k_idx)
            m_slots = int(pos.size)
            if m_slots == 0:
                continue
            for c_idx in range(n_cells):
                rv, pv, uv = cell_task_arrays[c_idx][k_idx]
                n_att = int(rv.size)
                if n_att == 1:
                    slot_r[pos, c_idx] = rv[0]
                    slot_p[pos, c_idx] = pv[0]
                    slot_u[pos, c_idx] = uv[0]
                else:
                    att_idx = rng.integers(0, n_att, size=(m_slots, n_att))
                    slot_r[pos, c_idx] = rv[att_idx].mean(axis=1)
                    slot_p[pos, c_idx] = pv[att_idx].mean(axis=1)
                    slot_u[pos, c_idx] = uv[att_idx].mean(axis=1)

        boot_r = slot_r.reshape(n_boot, k_tasks, n_cells).mean(axis=1)
        boot_p = slot_p.reshape(n_boot, k_tasks, n_cells).mean(axis=1)
        boot_u = slot_u.reshape(n_boot, k_tasks, n_cells).mean(axis=1)

        front_r_counts = np.zeros(n_cells, dtype=np.int64)
        front_p_counts = np.zeros(n_cells, dtype=np.int64)
        chunk_b = 2_000
        for b_start in range(0, n_boot, chunk_b):
            b_end = min(n_boot, b_start + chunk_b)
            br = boot_r[b_start:b_end]
            bp = boot_p[b_start:b_end]
            bu = boot_u[b_start:b_end]

            bu_i = bu[:, :, None]
            bu_j = bu[:, None, :]
            br_i = br[:, :, None]
            br_j = br[:, None, :]
            b_dom_r = (
                (bu_j <= bu_i + PARETO_EPS)
                & (br_j >= br_i - PARETO_EPS)
                & ((bu_j < bu_i - PARETO_EPS) | (br_j > br_i + PARETO_EPS))
            )
            front_r_counts += np.sum(~np.any(b_dom_r, axis=2), axis=0)

            b_val_p = bp > 0.0
            b_cpp = np.where(b_val_p, bu / np.maximum(bp, 1e-15), np.inf)
            b_cpp_i = b_cpp[:, :, None]
            b_cpp_j = b_cpp[:, None, :]
            bp_i = bp[:, :, None]
            bp_j = bp[:, None, :]
            b_dom_p = (
                b_val_p[:, None, :]
                & (b_cpp_j <= b_cpp_i + PARETO_EPS)
                & (bp_j >= bp_i - PARETO_EPS)
                & ((b_cpp_j < b_cpp_i - PARETO_EPS) | (bp_j > bp_i + PARETO_EPS))
            )
            b_front_p = b_val_p & (~np.any(b_dom_p, axis=2))
            front_p_counts += np.sum(b_front_p, axis=0)

        boot_prob_reward = front_r_counts.astype(np.float64) / float(n_boot)
        boot_prob_pass = front_p_counts.astype(np.float64) / float(n_boot)

    cells_list: list[dict[str, Any]] = []
    by_cell: dict[str, dict[str, Any]] = {}

    for c_idx, (pack_id, model_name, thinking_level, arm) in enumerate(sorted_cell_keys):
        cell_id = (
            f"{pack_id}__{model_name}__{thinking_level}__{arm}"
            if multi_pack
            else f"{model_name}__{thinking_level}__{arm}"
        )
        is_pr = bool(is_pareto_reward[c_idx])
        is_pp = bool(is_pareto_pass[c_idx])
        vp = bool(valid_pass[c_idx])
        cpp_json: float | None = round(float(cpp_arr[c_idx]), 8) if vp else None
        p_status = "dominated_zero_pass" if not vp else ("frontier" if is_pp else "dominated")
        r_status = "frontier" if is_pr else "dominated"

        entry = {
            "cell_id": cell_id,
            "pack_id": pack_id,
            "model_name": model_name,
            "thinking_level": thinking_level,
            "arm": arm,
            "n_common_tasks": k_tasks,
            "n_trials_on_common_tasks": cell_trial_counts[c_idx],
            "mean_usd_task_equal": round(float(usd_te[c_idx]), 8),
            "reward_task_equal_mean": round(float(reward_te[c_idx]), 8),
            "pass_at_1_task_equal": round(float(pass_te[c_idx]), 8),
            "cost_per_passing_run_usd_task_equal": cpp_json,
            "is_pareto_reward": is_pr,
            "pareto_reward_status": r_status,
            "pareto_reward_boot_prob": round(float(boot_prob_reward[c_idx]), 6),
            "is_pareto_pass": is_pp,
            "pareto_pass_status": p_status,
            "pareto_pass_boot_prob": round(float(boot_prob_pass[c_idx]), 6),
        }
        cells_list.append(entry)
        by_cell[cell_id] = entry

    cells_sorted_cost = sorted(
        cells_list,
        key=lambda d: (
            float(d["mean_usd_task_equal"]),
            -float(d["reward_task_equal_mean"]),
            str(d["cell_id"]),
        ),
    )
    reward_frontier_ids = [
        str(d["cell_id"]) for d in cells_sorted_cost if bool(d["is_pareto_reward"])
    ]
    pass_frontier_ids = [
        str(d["cell_id"])
        for d in sorted(
            (d for d in cells_list if bool(d["is_pareto_pass"])),
            key=lambda d: (
                float(d["cost_per_passing_run_usd_task_equal"] or 0.0),
                -float(d["pass_at_1_task_equal"]),
                str(d["cell_id"]),
            ),
        )
    ]

    return {
        "cohort": cohort_name,
        "status": "ok",
        "allowed_arms": list(allowed_arms),
        "e2e_only": e2e_only,
        "n_cells": n_cells,
        "n_common_tasks": k_tasks,
        "common_tasks": common_tasks,
        "cells": cells_sorted_cost,
        "by_cell": by_cell,
        "reward_frontier_cells": reward_frontier_ids,
        "pass_frontier_cells": pass_frontier_ids,
    }


def compute_pareto_frontiers(
    records: list[dict[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
) -> dict[str, dict[str, Any]]:
    """Compute cohort-stratified (`all_5_arms` and `e2e_6_arms`) Pareto frontiers & bootstrap probs."""
    all_5 = _compute_pareto_cohort(
        records,
        cohort_name="all_5_arms",
        allowed_arms=PARETO_ALL_5_ARMS,
        e2e_only=False,
        n_boot=n_boot,
        seed=seed + 701,
    )
    e2e_6 = _compute_pareto_cohort(
        records,
        cohort_name="e2e_6_arms",
        allowed_arms=PARETO_E2E_6_ARMS,
        e2e_only=True,
        n_boot=n_boot,
        seed=seed + 1_709,
    )
    return {
        "all_5_arms": sanitize_json_tree(all_5),
        "e2e_6_arms": sanitize_json_tree(e2e_6),
    }


def analyze_factorial(
    records: list[dict[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
    models_cfg: ModelsConfig | None = None,
) -> dict[str, Any]:
    """Run complete 2x2 factorial, prompt-adaptation, orchestration, parity, and Pareto analysis."""
    overall_cells = summarize_arm_cells(records)
    overall_contrasts = compute_stratum_contrasts(records, n_boot=n_boot, seed=seed)
    overall_parity = compute_parity_summary(
        records, n_boot=n_boot, seed=seed, models_cfg=models_cfg
    )
    pareto_summary = compute_pareto_frontiers(records, n_boot=n_boot, seed=seed + 700_000)

    suites = sorted({str(r.get("suite")) for r in records if r.get("suite")})
    models = sorted({str(r.get("model_name")) for r in records if r.get("model_name")})
    tasks = sorted({str(r.get("base_task")) for r in records if r.get("base_task")})
    packs = sorted({str(r.get("pack_id") or "ra_00a11ea") for r in records})
    multi_pack = len(packs) > 1
    model_thinkings = sorted(
        {
            (
                str(r.get("pack_id") or "ra_00a11ea"),
                str(r.get("model_name")),
                str(r.get("thinking_level") or "default"),
            )
            for r in records
            if r.get("model_name")
        }
    )

    by_suite: dict[str, Any] = {}
    for s_idx, suite in enumerate(suites):
        s_recs = [r for r in records if str(r.get("suite")) == suite]
        by_suite[suite] = {
            "n_trials": len(s_recs),
            "cells": summarize_arm_cells(s_recs),
            "contrasts": compute_stratum_contrasts(
                s_recs, n_boot=n_boot, seed=seed + 100_000 + s_idx * 101
            ),
            "parity": compute_parity_summary(
                s_recs, n_boot=n_boot, seed=seed + 110_000 + s_idx * 101, models_cfg=models_cfg
            ),
        }

    by_stage_group: dict[str, Any] = {}
    for g_idx, (grp_name, pred) in enumerate(
        (
            ("stage", lambda r: not bool(r.get("is_e2e")) and str(r.get("stage")) != "e2e"),
            ("e2e", lambda r: bool(r.get("is_e2e")) or str(r.get("stage")) == "e2e"),
        )
    ):
        g_recs = [r for r in records if pred(r)]
        if not g_recs:
            continue
        by_stage_group[grp_name] = {
            "n_trials": len(g_recs),
            "cells": summarize_arm_cells(g_recs),
            "contrasts": compute_stratum_contrasts(
                g_recs, n_boot=n_boot, seed=seed + 150_000 + g_idx * 101
            ),
            "parity": compute_parity_summary(
                g_recs, n_boot=n_boot, seed=seed + 160_000 + g_idx * 101, models_cfg=models_cfg
            ),
        }

    by_model: dict[str, Any] = {}
    for m_idx, model in enumerate(models):
        m_recs = [r for r in records if str(r.get("model_name")) == model]
        by_model[model] = {
            "n_trials": len(m_recs),
            "cells": summarize_arm_cells(m_recs),
            "contrasts": compute_stratum_contrasts(
                m_recs, n_boot=n_boot, seed=seed + 200_000 + m_idx * 101
            ),
        }

    by_model_thinking: dict[str, Any] = {}
    for mt_idx, (pack_id, model, t_lvl) in enumerate(model_thinkings):
        mt_recs = [
            r
            for r in records
            if str(r.get("pack_id") or "ra_00a11ea") == pack_id
            and str(r.get("model_name")) == model
            and str(r.get("thinking_level") or "default") == t_lvl
        ]
        mt_key = f"{pack_id}__{model}__{t_lvl}" if multi_pack else f"{model}__{t_lvl}"
        by_model_thinking[mt_key] = {
            "pack_id": pack_id,
            "model_name": model,
            "thinking_level": t_lvl,
            "n_trials": len(mt_recs),
            "cells": summarize_arm_cells(mt_recs),
            "contrasts": compute_stratum_contrasts(
                mt_recs, n_boot=n_boot, seed=seed + 250_000 + mt_idx * 101
            ),
        }

    by_suite_model: dict[str, Any] = {}
    sm_idx = 0
    for suite in suites:
        for model in models:
            sm_recs = [
                r
                for r in records
                if str(r.get("suite")) == suite and str(r.get("model_name")) == model
            ]
            if not sm_recs:
                continue
            sm_key = f"{suite}__{model}"
            by_suite_model[sm_key] = {
                "suite": suite,
                "model_name": model,
                "n_trials": len(sm_recs),
                "cells": summarize_arm_cells(sm_recs),
                "contrasts": compute_stratum_contrasts(
                    sm_recs, n_boot=n_boot, seed=seed + 300_000 + sm_idx * 101
                ),
            }
            sm_idx += 1

    by_task: dict[str, Any] = {}
    for t_idx, task_id in enumerate(tasks):
        t_recs = [r for r in records if str(r.get("base_task")) == task_id]
        by_task[task_id] = {
            "base_task": task_id,
            "suite": str(t_recs[0].get("suite")),
            "stage": str(t_recs[0].get("stage")),
            "n_trials": len(t_recs),
            "cells": summarize_arm_cells(t_recs),
            "contrasts": compute_stratum_contrasts(
                t_recs, n_boot=n_boot, seed=seed + 400_000 + t_idx * 101
            ),
            "parity": compute_parity_summary(
                t_recs, n_boot=n_boot, seed=seed + 410_000 + t_idx * 101, models_cfg=models_cfg
            ),
        }

    result_payload: dict[str, Any] = {
        "config": {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "wilson_z": WILSON_Z_95,
        },
        "n_trials": len(records),
        "overall": {
            "cells": overall_cells,
            "contrasts": overall_contrasts,
            "parity": overall_parity,
        },
        "pareto": pareto_summary,
        "by_suite": by_suite,
        "by_stage_group": by_stage_group,
        "by_model": by_model,
        "by_model_thinking": by_model_thinking,
        "by_suite_model": by_suite_model,
        "by_task": by_task,
    }

    if len(packs) > 1:
        by_pack: dict[str, Any] = {}
        for p_idx, pack_id in enumerate(packs):
            p_recs = [r for r in records if str(r.get("pack_id") or "ra_00a11ea") == pack_id]
            by_pack[pack_id] = {
                "pack_id": pack_id,
                "n_trials": len(p_recs),
                "cells": summarize_arm_cells(p_recs),
                "contrasts": compute_stratum_contrasts(
                    p_recs, n_boot=n_boot, seed=seed + 500_000 + p_idx * 101
                ),
                "parity": compute_parity_summary(
                    p_recs, n_boot=n_boot, seed=seed + 510_000 + p_idx * 101, models_cfg=models_cfg
                ),
            }
        result_payload["by_pack"] = by_pack

    return sanitize_json_tree(result_payload)
