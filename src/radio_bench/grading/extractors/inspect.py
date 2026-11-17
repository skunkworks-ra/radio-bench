"""Pure metric extractor for `inspect` stage (`inspect_summary.json` / `stage_output.json`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from radio_bench.grading.schema import (
    ExtractConfig,
    ObservableRecord,
    StageOutputManifest,
    finite_observable,
    loads_strict_json,
)

FLUX_CAL_ALIASES = (
    "flux_calibrator",
    "flux_cal",
    "flux_calibrators",
    "primary_calibrator",
    "flux_density_calibrator",
)
BP_CAL_ALIASES = (
    "bandpass_calibrator",
    "bandpass_cal",
    "bp_calibrator",
    "bp_cal",
    "bandpass_calibrators",
)
COMBINED_FLUX_BP_ALIASES = (
    "flux_bandpass_calibrator",
    "flux_and_bandpass_calibrator",
)
PHASE_CAL_ALIASES = (
    "phase_calibrator",
    "phase_cal",
    "phase_calibrators",
    "gain_calibrator",
    "complex_gain_calibrator",
)
TARGET_ALIASES = (
    "target_fields",
    "mosaic_target_fields",
    "targets",
    "science_targets",
    "mosaic_fields",
    "target_pointings",
    "mosaic_pointings",
    "target_field_names",
)
DEAD_ANT_ALIASES = (
    "dead_antennas",
    "dead_antenna",
    "bad_antennas",
    "bad_antenna",
    "flagged_antennas",
    "corrupted_antennas",
    "zero_amplitude_antennas",
)
MISSING_STATION_ALIASES = (
    "missing_stations",
    "missing_antennas",
    "missing_vla_pads",
    "missing_pads",
)
ELEV_ALIASES = (
    "inspect.bandpass_min_el_deg",
    "bandpass_min_elevation_deg",
    "bp_min_elevation_deg",
    "min_bandpass_elevation_deg",
    "bandpass_calibrator_min_elevation_deg",
    "bandpass_min_el_deg",
)


def norm_field_name(val: Any) -> str:
    if isinstance(val, dict):
        for k in ("name", "field", "field_name", "source"):
            if k in val and val[k] is not None:
                val = val[k]
                break
    s = str(val).strip().strip("'\"")
    return " ".join(s.split())


def norm_ant_name(val: Any) -> str:
    if isinstance(val, dict):
        for k in ("name", "antenna", "antenna_name", "station"):
            if k in val and val[k] is not None:
                val = val[k]
                break
    return str(val).strip().strip("'\"").lower()


def _extract_alias_value(payload: dict[str, Any], aliases: tuple[str, ...]) -> tuple[bool, Any]:
    for key in aliases:
        if key in payload and payload[key] is not None:
            return True, payload[key]
    for section_key in (
        "decisions",
        "reported_metrics",
        "fields",
        "calibrators",
        "antennas",
        "elevation",
        "summary",
    ):
        sub = payload.get(section_key)
        if isinstance(sub, dict):
            for key in aliases:
                if key in sub and sub[key] is not None:
                    return True, sub[key]
    return False, None


def _extract_elev_fallback(payload: dict[str, Any]) -> tuple[bool, Any]:
    found, val = _extract_alias_value(payload, ELEV_ALIASES)
    if found:
        return True, val
    for k, v in payload.items():
        low_k = str(k).lower()
        if (
            low_k.endswith("_min_elevation_deg")
            and not low_k.startswith(("phase_", "pol_", "target_"))
            and v is not None
        ):
            return True, v
    return False, None


def _to_str_list(raw: Any, normalizer: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict) and "value" in raw and len(raw) <= 4:
        raw = raw["value"]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [normalizer(p) for p in parts if normalizer(p)]
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            normed = normalizer(item)
            if normed:
                out.append(normed)
        return out
    if isinstance(raw, dict):
        normed = normalizer(raw)
        return [normed] if normed else []
    return []


def _to_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    if isinstance(raw, str):
        cleaned = raw.strip().lower().replace("deg", "").replace("degrees", "").strip()
        try:
            val = float(cleaned)
            return val if math.isfinite(val) else None
        except ValueError:
            return None
    if isinstance(raw, dict):
        for k in ("value", "min_deg", "min_elevation_deg", "deg"):
            if k in raw:
                return _to_float(raw[k])
    return None


def extract_inspect(
    workspace_dir: Path,
    extract_config: ExtractConfig,
    ref_arrays_dir: Path | None = None,
    *,
    table_locked: bool = False,
) -> tuple[dict[str, ObservableRecord], list[str]]:
    """Extract raw observables from `stage_output.json` or `inspect_summary.json`."""
    _ = extract_config, ref_arrays_dir
    ws = Path(workspace_dir)
    errors: list[str] = []
    obs: dict[str, ObservableRecord] = {}

    if table_locked:
        for key, unit in (
            ("inspect.summary_valid", None),
            ("inspect.valid_sections_count", None),
            ("inspect.flux_cal_list", None),
            ("inspect.bandpass_cal_list", None),
            ("inspect.phase_cal_list", None),
            ("inspect.target_fields_list", None),
            ("inspect.dead_antennas_list", None),
            ("inspect.missing_stations_list", None),
            ("inspect.bandpass_min_el_deg", "deg"),
        ):
            obs[key] = ObservableRecord(
                value=False if key == "inspect.summary_valid" else None,
                status="locked_or_unreadable",
                unit=unit,
                provenance="input.ms",
            )
        return obs, errors

    stage_out_path = ws / "stage_output.json"
    summary_path = ws / "inspect_summary.json"

    chosen_path: Path | None = None
    if summary_path.is_file():
        chosen_path = summary_path
    elif stage_out_path.is_file():
        chosen_path = stage_out_path

    if chosen_path is None:
        errors.append(f"Missing summary file: {summary_path}")
        obs["inspect.summary_valid"] = ObservableRecord(
            value=False, status="measured", provenance="inspect_summary.json"
        )
        for key, unit in (
            ("inspect.valid_sections_count", None),
            ("inspect.flux_cal_list", None),
            ("inspect.bandpass_cal_list", None),
            ("inspect.phase_cal_list", None),
            ("inspect.target_fields_list", None),
            ("inspect.dead_antennas_list", None),
            ("inspect.missing_stations_list", None),
            ("inspect.bandpass_min_el_deg", "deg"),
        ):
            obs[key] = ObservableRecord(
                value=None, status="missing_artifact", unit=unit, provenance="inspect_summary.json"
            )
        return obs, errors

    prov = chosen_path.name
    try:
        payload = loads_strict_json(chosen_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Malformed JSON in {chosen_path}: {exc}")
        obs["inspect.summary_valid"] = ObservableRecord(
            value=False, status="measured", provenance=prov
        )
        for key, unit in (
            ("inspect.valid_sections_count", None),
            ("inspect.flux_cal_list", None),
            ("inspect.bandpass_cal_list", None),
            ("inspect.phase_cal_list", None),
            ("inspect.target_fields_list", None),
            ("inspect.dead_antennas_list", None),
            ("inspect.missing_stations_list", None),
            ("inspect.bandpass_min_el_deg", "deg"),
        ):
            obs[key] = ObservableRecord(
                value=None, status="extractor_error", unit=unit, provenance=prov
            )
        return obs, errors

    if not isinstance(payload, dict) or len(payload) == 0:
        errors.append(f"{prov} must be a non-empty JSON object")
        obs["inspect.summary_valid"] = ObservableRecord(
            value=False, status="measured", provenance=prov
        )
        for key, unit in (
            ("inspect.valid_sections_count", None),
            ("inspect.flux_cal_list", None),
            ("inspect.bandpass_cal_list", None),
            ("inspect.phase_cal_list", None),
            ("inspect.target_fields_list", None),
            ("inspect.dead_antennas_list", None),
            ("inspect.missing_stations_list", None),
            ("inspect.bandpass_min_el_deg", "deg"),
        ):
            obs[key] = ObservableRecord(
                value=None, status="extractor_error", unit=unit, provenance=prov
            )
        return obs, errors

    # Validate StageOutputManifest strictly when reading stage_output.json
    if chosen_path == stage_out_path:
        try:
            manifest = StageOutputManifest.model_validate(payload)
            if manifest.status != "completed":
                obs["inspect.summary_valid"] = ObservableRecord(
                    value=False, status="measured", provenance=prov
                )
                for key, unit in (
                    ("inspect.valid_sections_count", None),
                    ("inspect.flux_cal_list", None),
                    ("inspect.bandpass_cal_list", None),
                    ("inspect.phase_cal_list", None),
                    ("inspect.target_fields_list", None),
                    ("inspect.dead_antennas_list", None),
                    ("inspect.missing_stations_list", None),
                    ("inspect.bandpass_min_el_deg", "deg"),
                ):
                    obs[key] = ObservableRecord(
                        value=None, status="not_applicable", unit=unit, provenance=prov
                    )
                return obs, errors
        except Exception as exc:
            errors.append(f"Invalid StageOutputManifest in {chosen_path}: {exc}")
            obs["inspect.summary_valid"] = ObservableRecord(
                value=False, status="measured", provenance=prov
            )
            for key, unit in (
                ("inspect.valid_sections_count", None),
                ("inspect.flux_cal_list", None),
                ("inspect.bandpass_cal_list", None),
                ("inspect.phase_cal_list", None),
                ("inspect.target_fields_list", None),
                ("inspect.dead_antennas_list", None),
                ("inspect.missing_stations_list", None),
                ("inspect.bandpass_min_el_deg", "deg"),
            ):
                obs[key] = ObservableRecord(
                    value=None, status="extractor_error", unit=unit, provenance=prov
                )
            return obs, errors

    has_flux, raw_flux = _extract_alias_value(payload, FLUX_CAL_ALIASES)
    has_bp, raw_bp = _extract_alias_value(payload, BP_CAL_ALIASES)
    if not has_flux or not has_bp:
        has_comb, raw_comb = _extract_alias_value(payload, COMBINED_FLUX_BP_ALIASES)
        if has_comb:
            if not has_flux:
                has_flux, raw_flux = True, raw_comb
            if not has_bp:
                has_bp, raw_bp = True, raw_comb

    _has_phase, raw_phase = _extract_alias_value(payload, PHASE_CAL_ALIASES)
    _has_targets, raw_targets = _extract_alias_value(payload, TARGET_ALIASES)
    has_dead, raw_dead = _extract_alias_value(payload, DEAD_ANT_ALIASES)
    _has_missing, raw_missing = _extract_alias_value(payload, MISSING_STATION_ALIASES)
    has_elev, raw_elev = _extract_elev_fallback(payload)

    pred_flux = _to_str_list(raw_flux, norm_field_name)
    pred_bp = _to_str_list(raw_bp, norm_field_name)
    pred_phase = _to_str_list(raw_phase, norm_field_name)
    pred_targets = _to_str_list(raw_targets, norm_field_name)
    pred_dead = _to_str_list(raw_dead, norm_ant_name)
    pred_missing = _to_str_list(raw_missing, norm_ant_name)
    pred_elev = _to_float(raw_elev)

    sections_present = sum(
        [
            1 if (pred_flux or pred_bp) else 0,
            1 if (pred_phase and pred_targets) else 0,
            1 if has_dead else 0,
            1 if (has_elev and pred_elev is not None) else 0,
        ]
    )

    obs["inspect.summary_valid"] = ObservableRecord(value=True, status="measured", provenance=prov)
    obs["inspect.valid_sections_count"] = ObservableRecord(
        value=int(sections_present), status="measured", provenance=prov
    )
    obs["inspect.flux_cal_list"] = ObservableRecord(
        value=pred_flux, status="measured", provenance=prov
    )
    obs["inspect.bandpass_cal_list"] = ObservableRecord(
        value=pred_bp, status="measured", provenance=prov
    )
    obs["inspect.phase_cal_list"] = ObservableRecord(
        value=pred_phase, status="measured", provenance=prov
    )
    obs["inspect.target_fields_list"] = ObservableRecord(
        value=pred_targets, status="measured", provenance=prov
    )
    obs["inspect.dead_antennas_list"] = ObservableRecord(
        value=pred_dead, status="measured", provenance=prov
    )
    obs["inspect.missing_stations_list"] = ObservableRecord(
        value=pred_missing, status="measured", provenance=prov
    )
    obs["inspect.bandpass_min_el_deg"] = finite_observable(
        pred_elev,
        unit="deg",
        provenance=prov,
        missing_status="extractor_error" if has_elev else "missing_artifact",
    )
    return obs, errors
