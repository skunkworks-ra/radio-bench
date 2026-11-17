"""Harbor 2x2 capability ablation benchmark for radio interferometric data reduction."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DATA_ROOT",
    "RADIO_ANALYST_SHA",
    "REPO_ROOT",
    "RUNS_ROOT",
    "TEST_ROOT",
]


def __getattr__(name: str) -> Any:
    if name in {"REPO_ROOT", "TEST_ROOT", "DATA_ROOT", "RUNS_ROOT", "RADIO_ANALYST_SHA"}:
        from radio_bench.config import load_bench_config

        cfg = load_bench_config()
        if name == "REPO_ROOT":
            return cfg.repo_root
        if name == "TEST_ROOT":
            return cfg.work_dir
        if name == "DATA_ROOT":
            return cfg.data_dir
        if name == "RUNS_ROOT":
            return cfg.runs_dir
        if name == "RADIO_ANALYST_SHA":
            return cfg.get_pack().analyst_ref
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
