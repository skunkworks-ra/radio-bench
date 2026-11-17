"""Dataset-blind physical metric extractors for `radio-bench`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from radio_bench.grading.extractors.calibrate import extract_calibrate
from radio_bench.grading.extractors.common import (
    compute_raw_data_blake2b,
    discover_caltables_by_viscal,
    discover_fluxscale_from_caltables,
    has_active_table_lock,
    parse_stage_manifest,
    resolve_scratch_root,
    scan_workspace_table_locks,
)
from radio_bench.grading.extractors.image import discover_image_sets, extract_image
from radio_bench.grading.extractors.inspect import extract_inspect
from radio_bench.grading.extractors.precal import extract_precal
from radio_bench.grading.schema import ExtractConfig, ObservableRecord, StageOutputManifest


@dataclass
class WorkspaceExtractionResult:
    """Raw extracted observables, Layer B manifest (if present), and non-fatal diagnostics."""

    observables: dict[str, ObservableRecord] = field(default_factory=dict)
    manifest: StageOutputManifest | None = None
    raw_manifest_dict: dict[str, Any] | None = None
    locked_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def extract_workspace(
    workspace_dir: Path,
    extract_config: ExtractConfig,
    ref_arrays_dir: Path | None = None,
    *,
    scratch_root: Path | None = None,
    compute_raw_hash: bool = True,
) -> WorkspaceExtractionResult:
    """Run pure physical metric extractors for `extract_config.stages` over `workspace_dir`.

    Never reads truth scalar files, never applies pass/fail tolerances, and never branches
    on `suite_id`.
    """
    ws = Path(workspace_dir)
    obs: dict[str, ObservableRecord] = {}
    errors: list[str] = []

    locked_paths = scan_workspace_table_locks(ws)
    table_locked = len(locked_paths) > 0
    if locked_paths:
        for lp in locked_paths:
            errors.append(f"Active casacore write lock on {lp}")

    obs["proc.table_locked"] = ObservableRecord(
        value=table_locked,
        status="measured",
        provenance="workspace",
    )

    manifest, raw_manifest_dict, manifest_obs, manifest_errs = parse_stage_manifest(
        ws, stages=extract_config.stages
    )
    obs.update(manifest_obs)
    errors.extend(manifest_errs)

    input_ms = ws / "input.ms"
    if compute_raw_hash and (input_ms / "table.dat").is_file() and not table_locked:
        try:
            digest = compute_raw_data_blake2b(input_ms)
            obs["proc.raw_data_blake2b"] = ObservableRecord(
                value=digest,
                status="measured",
                provenance="input.ms",
            )
        except Exception as exc:
            errors.append(f"Failed computing raw_data_blake2b on input.ms: {exc}")
            obs["proc.raw_data_blake2b"] = ObservableRecord(
                value=None,
                status="extractor_error",
                provenance="input.ms",
            )
    elif table_locked:
        obs["proc.raw_data_blake2b"] = ObservableRecord(
            value=None,
            status="locked_or_unreadable",
            provenance="input.ms",
        )
    else:
        obs["proc.raw_data_blake2b"] = ObservableRecord(
            value=None,
            status="not_applicable" if not compute_raw_hash else "missing_artifact",
            provenance="input.ms",
        )

    stage_set = {
        "image" if s.strip().lower() == "image_cont" else s.strip().lower()
        for s in extract_config.stages
    }

    # On multi-stage e2e pipelines, intermediate inspect/precal steps are audited via terminal
    # calibration and imaging deliverables unless explicitly evaluated as standalone stages.
    is_e2e_chain = "calibrate" in stage_set and "image" in stage_set and len(stage_set) >= 4

    if "inspect" in stage_set and not is_e2e_chain:
        try:
            s_obs, s_err = extract_inspect(
                ws, extract_config, ref_arrays_dir, table_locked=table_locked
            )
            obs.update(s_obs)
            errors.extend(s_err)
        except Exception as exc:
            errors.append(f"Extractor 'inspect' crashed ({type(exc).__name__}): {exc}")

    if "precal" in stage_set and not is_e2e_chain:
        try:
            s_obs, s_err = extract_precal(
                ws, extract_config, ref_arrays_dir, table_locked=table_locked
            )
            obs.update(s_obs)
            errors.extend(s_err)
        except Exception as exc:
            errors.append(f"Extractor 'precal' crashed ({type(exc).__name__}): {exc}")

    if "calibrate" in stage_set:
        try:
            s_obs, s_err = extract_calibrate(
                ws,
                extract_config,
                ref_arrays_dir,
                scratch_root=scratch_root,
                table_locked=table_locked,
            )
            obs.update(s_obs)
            errors.extend(s_err)
        except Exception as exc:
            errors.append(f"Extractor 'calibrate' crashed ({type(exc).__name__}): {exc}")

    if "image" in stage_set:
        try:
            s_obs, s_err = extract_image(
                ws, extract_config, ref_arrays_dir, table_locked=table_locked
            )
            obs.update(s_obs)
            errors.extend(s_err)
        except Exception as exc:
            errors.append(f"Extractor 'image' crashed ({type(exc).__name__}): {exc}")

    return WorkspaceExtractionResult(
        observables=obs,
        manifest=manifest,
        raw_manifest_dict=raw_manifest_dict,
        locked_paths=locked_paths,
        errors=errors,
    )


__all__ = [
    "WorkspaceExtractionResult",
    "compute_raw_data_blake2b",
    "discover_caltables_by_viscal",
    "discover_fluxscale_from_caltables",
    "discover_image_sets",
    "extract_calibrate",
    "extract_image",
    "extract_inspect",
    "extract_precal",
    "extract_workspace",
    "has_active_table_lock",
    "parse_stage_manifest",
    "resolve_scratch_root",
    "scan_workspace_table_locks",
]
