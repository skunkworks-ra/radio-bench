"""Natural-weighting thermal floor: estimator accuracy, ratio math, and not-applicable gating."""

from __future__ import annotations

import math
import os
from pathlib import Path

from radio_bench.config import is_under_slash_tmp, load_bench_config, write_casaconfig_local

BENCH_CFG = load_bench_config()
if is_under_slash_tmp(os.environ.get("TMPDIR", "")) or not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = str(BENCH_CFG.tmp_dir)
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
write_casaconfig_local(BENCH_CFG)
os.environ.setdefault("CASASITECONFIG", str(BENCH_CFG.casaconfig_path))

import pytest  # noqa: E402

from radio_bench.grading.extractors.common import (  # noqa: E402
    natural_thermal_noise_jy,
    resolve_field_ids_by_names,
    thermal_floor_ratio,
)
from radio_bench.grading.rubric import evaluate_rule_expr  # noqa: E402
from radio_bench.grading.schema import LeafRuleSpec, ObservableRecord  # noqa: E402

SYNTH_IMAGE_MS = BENCH_CFG.data_dir / "synth" / "s1_image" / "input.ms"


@pytest.mark.skipif(not SYNTH_IMAGE_MS.is_dir(), reason="Suite S data not built")
def test_estimator_recovers_injected_synth_noise() -> None:
    # Suite S injects sigma_vis = 0.0527 Jy per real/imag component on every visibility.
    fids = resolve_field_ids_by_names(SYNTH_IMAGE_MS, ["SYNTH_MOSAIC_P1", "SYNTH_MOSAIC_P2"])
    sigma, info = natural_thermal_noise_jy(SYNTH_IMAGE_MS, fids)
    expected = 0.0527 / math.sqrt(info["n_samples"])
    assert sigma is not None
    assert sigma == pytest.approx(expected, rel=0.02)


def test_floor_ratio_math() -> None:
    # 200 beams: floor = sigma * (1 - 3/20) = 0.85 sigma.
    assert thermal_floor_ratio(0.85e-4, 1.0e-4, 200.0) == pytest.approx(1.0)
    assert thermal_floor_ratio(0.5e-4, 1.0e-4, 200.0) < 1.0
    assert thermal_floor_ratio(1.0e-4, None, 200.0) is None
    assert thermal_floor_ratio(1.0e-4, 1.0e-4, 4.0) is None


def _gate_score(rec: ObservableRecord) -> float:
    leaf = LeafRuleSpec(
        observable="image.offsource_rms_over_thermal_floor",
        primitive="threshold_gate",
        params={"op": ">=", "limit": 1.0, "pass_if_not_applicable": True},
    )
    return evaluate_rule_expr(
        leaf,
        {"image.offsource_rms_over_thermal_floor": rec},
        {},
        stage="image",
        truth_prefix=None,
        named_scores={},
        leaf_traces=[],
    )


def test_gate_scores() -> None:
    assert _gate_score(ObservableRecord(value=1.3, status="measured")) == 1.0
    assert _gate_score(ObservableRecord(value=0.4, status="measured")) == 0.0
    assert _gate_score(ObservableRecord(value=None, status="not_applicable")) == 1.0
    assert _gate_score(ObservableRecord(value=None, status="missing_artifact")) == 0.0


@pytest.mark.skipif(not SYNTH_IMAGE_MS.is_dir(), reason="Suite S data not built")
def test_target_retention_matches_direct_flag_count() -> None:
    import casatools
    import numpy as np

    from radio_bench.grading.extractors.common import target_parallel_hand_retention

    fids = resolve_field_ids_by_names(SYNTH_IMAGE_MS, ["SYNTH_MOSAIC_P1", "SYNTH_MOSAIC_P2"])
    tb = casatools.table()
    try:
        tb.open(str(SYNTH_IMAGE_MS))
        sub = tb.query(f"FIELD_ID IN [{','.join(map(str, fids))}] && ANTENNA1 != ANTENNA2")
        try:
            flg = np.asarray(sub.getcol("FLAG"), dtype=bool)[[0, -1]]
            flg |= np.asarray(sub.getcol("FLAG_ROW"), dtype=bool)[None, None, :]
        finally:
            sub.close()
    finally:
        tb.close()
        tb.done()
    expected = float(np.mean(~flg))
    assert target_parallel_hand_retention(SYNTH_IMAGE_MS, fids) == pytest.approx(expected)
