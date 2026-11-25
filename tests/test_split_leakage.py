"""Reference/held-out dataset names must not appear in exported skills."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from radio_bench.build_tasks import find_split_leakage


def _spec(split: str, roles: dict[str, list[str]]) -> SimpleNamespace:
    return SimpleNamespace(split=split, extract_config_json=json.dumps({"roles": roles}))


def test_flags_reference_target_and_phase_cal(tmp_path: Path) -> None:
    (tmp_path / "skill").mkdir()
    (tmp_path / "skill" / "SKILL.md").write_text(
        "Image twhya the same way as J1037-2934.\n", encoding="utf-8"
    )
    specs = [_spec("reference", {"targets": ["TW Hya", "twhya"], "phase_cal": ["J1037-2934"]})]
    hits = find_split_leakage(specs, tmp_path)
    assert sorted(h.rsplit(": ", 1)[1] for h in hits) == ["J1037-2934", "twhya"]


def test_ignores_dev_suites_and_standard_calibrators(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("Use 3C286 for flux. Mosaic like 3C391.\n", encoding="utf-8")
    specs = [
        _spec("dev", {"targets": ["3C391"]}),
        _spec("held_out", {"targets": ["3C286"], "flux_cal": ["3C286"]}),
    ]
    assert find_split_leakage(specs, tmp_path) == []


def test_no_partial_token_match(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("J0336+32180 and PER_FIELD_710\n", encoding="utf-8")
    specs = [_spec("reference", {"targets": ["PER_FIELD_71"], "phase_cal": ["J0336+3218"]})]
    assert find_split_leakage(specs, tmp_path) == []


def test_standard_calibrator_as_target_is_exempt(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("Set the 3C286 flux model with setjy.\n", encoding="utf-8")
    specs = [_spec("reference", {"targets": ["3c286"], "phase_cal": ["J1310+3220"]})]
    assert find_split_leakage(specs, tmp_path) == []
