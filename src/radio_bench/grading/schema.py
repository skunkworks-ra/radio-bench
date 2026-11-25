"""Pydantic v2 schemas for Layer B manifests, observables, suite manifests, and rubrics."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StatusLiteral = Literal["completed", "halted", "failed"]

HaltCodeLiteral = Literal[
    "missing_calibrator",
    "corrupted_flux_standard",
    "insufficient_parallactic_angle",
    "failed_priorcals",
    "zero_unflagged_data",
    "missing_input_file",
    "other_physical_blocker",
]

ObservableStatusLiteral = Literal[
    "measured",
    "not_applicable",
    "missing_artifact",
    "locked_or_unreadable",
    "extractor_error",
]
ObservableStatus = ObservableStatusLiteral

PrimitiveLiteral = Literal[
    "bool_pass",
    "exact_set",
    "set_jaccard",
    "set_f1",
    "piecewise_floor_ramp",
    "discrete_map",
    "upper_ramp",
    "lower_ramp",
    "rel_err_ramp",
    "abs_err_ramp",
    "window_ramp",
    "threshold_gate",
]

CombineLiteral = Literal[
    "min_of",
    "max_of",
    "weighted_sum",
    "zero_if_any_zero",
    "multiplicative_cap",
]


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {token!r} is forbidden")


def loads_strict_json(raw_text: str) -> Any:
    """Parse UTF-8 JSON rejecting `NaN`, `Infinity`, and `-Infinity` tokens."""
    return json.loads(raw_text, parse_constant=_reject_json_constant)


def ensure_finite_json_value(val: Any, path: str = "value") -> Any:
    """Convert numpy scalars/arrays to Python primitives and reject non-finite numbers."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite float at {path}: {val!r}")
        return fval
    if isinstance(val, str):
        return val
    if isinstance(val, np.ndarray):
        return ensure_finite_json_value(val.tolist(), path=path)
    if isinstance(val, (list, tuple)):
        return [
            ensure_finite_json_value(item, path=f"{path}[{idx}]") for idx, item in enumerate(val)
        ]
    if isinstance(val, dict):
        return {str(k): ensure_finite_json_value(v, path=f"{path}.{k}") for k, v in val.items()}
    raise ValueError(f"Unsupported value type at {path}: {type(val).__name__}")


def finite_float_or_none(val: Any, *, ndigits: int | None = None) -> float | None:
    """Return finite Python `float` (optionally rounded), or `None` on `None`/`bool`/`NaN`/`Inf`."""
    if val is None or isinstance(val, (bool, np.bool_)):
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(fv):
        return None
    return round(fv, ndigits) if ndigits is not None else fv


def finite_observable(
    raw_val: Any,
    *,
    unit: str | None = None,
    provenance: str | None = None,
    missing_status: ObservableStatusLiteral = "missing_artifact",
    ndigits: int | None = None,
) -> ObservableRecord:
    """Build a float `ObservableRecord`, converting `NaN`/`Inf` overflows into `extractor_error`."""
    if raw_val is None:
        return ObservableRecord(
            value=None,
            status=missing_status,
            unit=unit,
            provenance=provenance,
        )
    fv = finite_float_or_none(raw_val, ndigits=ndigits)
    if fv is None:
        return ObservableRecord(
            value=None,
            status="extractor_error",
            unit=unit,
            provenance=provenance,
        )
    return ObservableRecord(
        value=fv,
        status="measured",
        unit=unit,
        provenance=provenance,
    )


def _check_nonfinite_string(val: str) -> str:
    stripped = val.strip()
    low = stripped.lower()
    if low in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        raise ValueError(f"ReportedMetric.value string encodes non-finite value: {val!r}")
    try:
        fv = float(stripped)
    except ValueError:
        return val
    if not math.isfinite(fv):
        raise ValueError(f"ReportedMetric.value string parses to non-finite float: {val!r}")
    return val


def _validate_reported_metric_value(val: Any) -> float | int | str | list[str] | list[float] | bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"ReportedMetric.value must be finite, got {val!r}")
        return fval
    if isinstance(val, str):
        return _check_nonfinite_string(val)
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        if not val:
            return []
        if all(isinstance(x, str) for x in val):
            return [_check_nonfinite_string(str(x)) for x in val]
        if all(
            isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
            for x in val
        ):
            out_floats: list[float] = []
            for idx, item in enumerate(val):
                fv = float(item)
                if not math.isfinite(fv):
                    raise ValueError(f"Non-finite float at ReportedMetric.value[{idx}]: {item!r}")
                out_floats.append(fv)
            return out_floats
        raise ValueError(
            "ReportedMetric.value list must contain uniformly strings or finite numbers"
        )
    raise ValueError(f"Invalid ReportedMetric.value type: {type(val).__name__}")


class ReportedMetric(BaseModel):
    """Self-reported scalar or list metric inside `stage_output.json`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    value: Any
    unit: str | None = None
    source: Literal["tool", "casa_script", "derived"] | None = None
    step_ref: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _check_value(cls, v: Any) -> Any:
        return _validate_reported_metric_value(v)


class StageDecisions(BaseModel):
    """Structured reduction decisions inside `stage_output.json`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    flux_calibrator: str | None = None
    bandpass_calibrator: str | None = None
    phase_calibrator: str | None = None
    pol_angle_calibrator: str | None = None
    pol_leakage_calibrator: str | None = None
    target_fields: list[str] = Field(default_factory=list)
    dead_antennas: list[str] = Field(default_factory=list)
    missing_stations: list[str] = Field(default_factory=list)
    refant: str | None = None
    uvrange: str | None = None
    solints: dict[str, str] = Field(default_factory=dict)
    imaging: dict[str, Any] = Field(default_factory=dict)

    @field_validator("imaging", mode="before")
    @classmethod
    def _check_imaging(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("StageDecisions.imaging must be a JSON object")
        checked = ensure_finite_json_value(v, path="decisions.imaging")
        assert isinstance(checked, dict)
        return checked


class StageArtifacts(BaseModel):
    """On-disk CASA/HTML product pointers inside `stage_output.json`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    ms_path: str = "input.ms"
    calibrators_ms_path: str | None = None
    caltables: list[str] = Field(default_factory=list)
    image_prefix: str | None = None
    report_path: str | None = None


class StageOutputManifest(BaseModel):
    """Layer B arm-neutral stage output manifest (`/workspace/stage_output.json`)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal["0.2"] = "0.2"
    task_id: str | None = None
    stage: str
    status: StatusLiteral
    halt_code: HaltCodeLiteral | None = None
    halt_reason: str | None = None
    unresolved_checks: list[str] = Field(default_factory=list)
    decisions: StageDecisions = Field(default_factory=StageDecisions)
    artifacts: StageArtifacts = Field(default_factory=StageArtifacts)
    reported_metrics: dict[str, ReportedMetric] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_halt_consistency(self) -> StageOutputManifest:
        if not self.stage.strip():
            raise ValueError("stage must be non-empty")
        if self.status in ("halted", "failed"):
            if self.halt_code is None:
                raise ValueError(f"halt_code is required when status == {self.status!r}")
            if self.halt_reason is None or not self.halt_reason.strip():
                raise ValueError(f"halt_reason must be non-empty when status == {self.status!r}")
        elif self.status == "completed" and self.halt_code is not None:
            raise ValueError("halt_code must be null when status == 'completed'")
        return self


class ObservableRecord(BaseModel):
    """Single canonical raw physical observable emitted by an extractor or runner."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    value: Any = None
    status: ObservableStatusLiteral
    unit: str | None = None
    provenance: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _check_finite_value(cls, v: Any) -> Any:
        return ensure_finite_json_value(v, path="ObservableRecord.value")


# ---------------------------------------------------------------------------
# Suite Manifest (`suites/<suite_id>/suite.toml`) & `ExtractConfig` Projection
# ---------------------------------------------------------------------------


# Dataset tiers. dev: capability development, traces are read. reference: public walkthrough
# (CASA Guide, VLA pipeline) with a human-verified result; likely known to the LLM, never used to
# drive skill edits. held_out: no public walkthrough, not seen by the LLM, run once per frozen pack.
SplitTier = Literal["dev", "reference", "held_out"]


class ArchiveSpec(BaseModel):
    """Archival tarball checksum entry in `[suite.archives]`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    name: str
    rel_path: str
    sha256: str


class SuiteInfoSpec(BaseModel):
    """Dataset metadata block `[suite]` in `suite.toml`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    suite_id: str
    legacy_suite_alias: str | None = None
    split: SplitTier = "dev"
    telescope: str = "VLA"
    band: str = ""
    pol_basis: Literal["circular", "linear"] = "circular"
    description: str = ""
    category: str = "astronomy"
    tags: list[str] = Field(default_factory=lambda: ["radio-astronomy", "casa", "interferometry"])
    archives: list[ArchiveSpec] = Field(default_factory=list)


class RolesSpec(BaseModel):
    """Semantic field and antenna roles `[roles]` (zero truth scalar expectations)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    flux_cal: list[str] = Field(default_factory=list)
    bandpass_cal: list[str] = Field(default_factory=list)
    phase_cal: list[str] = Field(default_factory=list)
    pol_angle_cal: list[str] = Field(default_factory=list)
    pol_leakage_cal: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    dead_antennas: list[str] = Field(default_factory=list)
    missing_stations: list[str] = Field(default_factory=list)
    target_retention_excludes_dead_antenna: bool = True

    @property
    def calibrator_fields(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in (
            *self.flux_cal,
            *self.bandpass_cal,
            *self.phase_cal,
            *self.pol_angle_cal,
            *self.pol_leakage_cal,
        ):
            clean = raw.strip()
            if clean and clean.upper() not in seen:
                seen.add(clean.upper())
                out.append(clean)
        return out

    @property
    def target_fields(self) -> list[str]:
        return [t.strip() for t in self.targets if t.strip()]


class SourceApertureSpec(BaseModel):
    """Sky/cube measurement aperture geometry (coordinates & radii only; zero flux truth)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    label: str
    kind: Literal["extended", "point", "region", "cube"] = "extended"
    ra: str
    dec: str
    radius_arcsec: float
    offsource_exclusion_arcsec: float | None = None
    fallback_exclusion_arcsec: float | None = None
    v_min_kms: float | None = None
    v_max_kms: float | None = None


class VerifierTcleanRecipe(BaseModel):
    """Fixed-recipe verifier `tclean` imaging parameters (zero pass/fail thresholds)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    field: str
    spw: str = ""
    phasecenter: str | None = None
    specmode: str = "mfs"
    gridder: str = "standard"
    deconvolver: str = "hogbom"
    scales: list[int] | None = None
    smallscalebias: float | None = None
    imsize: list[int]
    cell: str | None = None
    cell_arcsec: float | None = None
    weighting: str = "briggs"
    robust: float = 0.5
    pblimit: float = 0.2
    usemask: str = "pb"
    pbmask: float = 0.2
    sidelobethreshold: float | None = None
    noisethreshold: float | None = None
    lownoisethreshold: float | None = None
    negativethreshold: float | None = None
    minbeamfrac: float | None = None
    growiterations: int | None = None
    dogrowprune: bool | None = None
    niter: int = 500
    cycleniter: int | None = None
    nmajor: int | None = None
    threshold: str = "0.5mJy"
    pbcor: bool = True
    nterms: int | None = None
    off_source_min_radius_arcsec: float | None = None
    timeout_sec: float = 900.0

    @model_validator(mode="after")
    def _validate_cell(self) -> VerifierTcleanRecipe:
        if self.cell is None and self.cell_arcsec is None:
            raise ValueError("VerifierTcleanRecipe requires either `cell` or `cell_arcsec`")
        if len(self.imsize) == 1:
            self.imsize = [int(self.imsize[0]), int(self.imsize[0])]
        elif len(self.imsize) != 2:
            raise ValueError(f"imsize must have 1 or 2 integers, got {self.imsize!r}")
        return self

    @property
    def effective_cell_arcsec(self) -> float:
        if self.cell_arcsec is not None:
            return float(self.cell_arcsec)
        assert self.cell is not None
        cleaned = self.cell.strip().lower()
        if cleaned.endswith("arcsec"):
            return float(cleaned[: -len("arcsec")].strip())
        if cleaned.endswith("arcmin"):
            return float(cleaned[: -len("arcmin")].strip()) * 60.0
        if cleaned.endswith("deg"):
            return float(cleaned[: -len("deg")].strip()) * 3600.0
        return float(cleaned)


class GeometrySpec(BaseModel):
    """Spatial/primary-beam measurement geometry `[geometry]` in `suite.toml`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    dish_diameter_m: float = 25.0
    pb_mask_min: float = 0.20
    offsource_inner_exclusion_beams: float = 5.0
    phasecenter: str | None = None
    off_source_min_radius_arcsec: float | None = None
    source_apertures: list[SourceApertureSpec] = Field(default_factory=list)
    verifier_tclean_recipe: VerifierTcleanRecipe | None = None


class ResourcesSpec(BaseModel):
    """Container CPU, RAM, and concurrency budget `[resources]` in `suite.toml`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    cpus: int = 4
    verifier_cpus: int | None = None
    memory_mb: int = 8192
    n_concurrent_trials: int = 1


class CheckpointEntrySpec(BaseModel):
    """Named checkpoint path mapping in `[checkpoints.entries.<name>]`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    input_ms: str
    input_calibrators_ms: str | None = None


class CheckpointsSpec(BaseModel):
    """Checkpoint builder and named checkpoint paths `[checkpoints]` in `suite.toml`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    builder: str | None = None
    entries: dict[str, CheckpointEntrySpec] = Field(default_factory=dict)


class SuiteTaskSpec(BaseModel):
    """Single task definition `[tasks.<task_id>]` inside `suite.toml`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    stages: list[str]
    stage: str | None = None
    checkpoint_in: str = "raw"
    data_dir: str
    input_ms: str | None = None
    input_calibrators_ms: str | None = None
    ground_truth: str
    extra_test_files: list[str] = Field(default_factory=list)
    ref_flags_file: str | None = None
    true_target_vis_file: str | None = None
    allow_virtual_model: bool = False
    use_verifier_tclean_recipe: bool | None = None
    difficulty: str = "medium"
    agent_timeout_sec: float = 900.0
    verifier_timeout_sec: float = 600.0
    cpus: int | None = None
    verifier_cpus: int | None = None
    memory_mb: int | None = None
    instruction: str
    oracle_script: str
    rubric: str

    @model_validator(mode="after")
    def _validate_stages(self) -> SuiteTaskSpec:
        if not self.stages:
            raise ValueError("SuiteTaskSpec.stages must be non-empty")
        return self

    @property
    def canonical_stage(self) -> str:
        if self.stage:
            return self.stage
        if len(self.stages) > 1:
            return "e2e"
        s0 = self.stages[0].strip().lower()
        return "image" if s0 == "image_cont" else s0

    @property
    def is_e2e(self) -> bool:
        return self.canonical_stage == "e2e" or len(self.stages) >= 4


class ExtractConfig(BaseModel):
    """Dataset-blind projection serialized to `/tests/extract_config.json` (zero truth scalars)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal["0.2"] = "0.2"
    suite_id: str
    split: SplitTier = "dev"
    telescope: str = "VLA"
    pol_basis: Literal["circular", "linear"] = "circular"
    stages: list[str]
    allow_virtual_model: bool = False
    target_retention_excludes_dead_antenna: bool = True
    roles: RolesSpec = Field(default_factory=RolesSpec)
    geometry: GeometrySpec = Field(default_factory=GeometrySpec)


class SuiteManifest(BaseModel):
    """Full parsed and validated `suites/<suite_id>/suite.toml` manifest."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    suite: SuiteInfoSpec
    roles: RolesSpec = Field(default_factory=RolesSpec)
    geometry: GeometrySpec = Field(default_factory=GeometrySpec)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    checkpoints: CheckpointsSpec = Field(default_factory=CheckpointsSpec)
    tasks: dict[str, SuiteTaskSpec]

    def to_extract_config(self, task_id: str) -> ExtractConfig:
        if task_id not in self.tasks:
            raise KeyError(f"Unknown task_id {task_id!r} in suite {self.suite.suite_id!r}")
        tspec = self.tasks[task_id]
        norm_stages = [
            "image" if s.strip().lower() == "image_cont" else s.strip().lower()
            for s in tspec.stages
        ]
        # On pure inspect tasks, field roles are discovered by the agent and compared in runner.
        eff_roles = RolesSpec() if norm_stages == ["inspect"] else self.roles.model_copy(deep=True)
        include_tclean = (
            tspec.use_verifier_tclean_recipe
            if tspec.use_verifier_tclean_recipe is not None
            else ("calibrate" in norm_stages and self.geometry.verifier_tclean_recipe is not None)
        )
        eff_geom = self.geometry.model_copy(deep=True)
        if not include_tclean:
            eff_geom.verifier_tclean_recipe = None

        return ExtractConfig(
            suite_id=self.suite.suite_id,
            split=self.suite.split,
            telescope=self.suite.telescope,
            pol_basis=self.suite.pol_basis,
            stages=norm_stages,
            allow_virtual_model=bool(tspec.allow_virtual_model),
            target_retention_excludes_dead_antenna=bool(
                self.roles.target_retention_excludes_dead_antenna
            ),
            roles=eff_roles,
            geometry=eff_geom,
        )


# ---------------------------------------------------------------------------
# Declarative Rubric (`rubric.json`) Schema
# ---------------------------------------------------------------------------


ALLOWED_THRESHOLD_OPS: frozenset[str] = frozenset({">=", ">", "<=", "<", "=="})

ALLOWED_PRIMITIVE_PARAMS: dict[str, frozenset[str]] = {
    "bool_pass": frozenset({"expected", "truth_key"}),
    "exact_set": frozenset(
        {
            "truth_key",
            "expected",
            "norm",
            "ignore_set",
            "ignore_truth_key",
            "partial_match_sets",
            "forbid_overlap_truth_keys",
            "overlap_penalty_factor",
        }
    ),
    "set_jaccard": frozenset(
        {
            "truth_key",
            "expected",
            "norm",
            "ignore_set",
            "ignore_truth_key",
            "forbid_overlap_truth_keys",
            "overlap_penalty_factor",
        }
    ),
    "set_f1": frozenset(
        {
            "truth_key",
            "expected",
            "norm",
            "ignore_set",
            "ignore_truth_key",
            "forbid_overlap_truth_keys",
            "overlap_penalty_factor",
        }
    ),
    "piecewise_floor_ramp": frozenset({"base_floor", "denom"}),
    "discrete_map": frozenset({"mapping", "default"}),
    "upper_ramp": frozenset(
        {
            "pass_max",
            "pass_max_key",
            "pass_max_scale",
            "pass_max_key_scale_key",
            "pass_max_mult_key",
            "zero_at",
            "zero_at_key",
            "zero_at_scale",
            "zero_at_key_scale_key",
            "decay_span",
            "decay_span_mult_of_pass_max",
            "strict_pass",
            "scale_above_pass",
            "use_abs",
            "spectrum_truth_key",
        }
    ),
    "lower_ramp": frozenset(
        {
            "pass_min",
            "pass_min_key",
            "pass_min_scale",
            "pass_min_key_scale_key",
            "zero_at",
            "zero_at_key",
            "zero_at_scale",
            "zero_at_key_scale_key",
            "zero_at_scale_of_pass_min",
            "decay_span",
            "min_denom",
        }
    ),
    "rel_err_ramp": frozenset(
        {
            "target",
            "truth_key",
            "target_scale",
            "truth_key_scale_key",
            "pass_rel_tol",
            "pass_rel_tol_key",
            "zero_at_rel_err",
            "zero_at_rel_err_key",
            "decay_span",
            "decay_span_mult_of_tol",
        }
    ),
    "abs_err_ramp": frozenset(
        {
            "target",
            "truth_key",
            "target_scale",
            "truth_key_scale_key",
            "pass_abs_tol",
            "pass_abs_tol_key",
            "zero_at_abs_err",
            "zero_at_abs_err_key",
            "decay_span",
        }
    ),
    "window_ramp": frozenset(
        {
            "min",
            "min_key",
            "min_scale",
            "min_key_scale_key",
            "max",
            "max_key",
            "max_scale",
            "max_key_scale_key",
            "pad_low",
            "pad_high",
            "low_zero_at",
            "high_zero_at",
            "strict_positive",
        }
    ),
    "threshold_gate": frozenset(
        {
            "op",
            "limit",
            "limit_key",
            "scale",
            "mult_key",
            "use_abs",
            "truth_key",
            "pass_if_not_applicable",
        }
    ),
}


def validate_primitive_params(primitive: str, params: dict[str, Any]) -> None:
    """Validate parameter keys and operator literals for `primitive` at schema load time."""
    allowed = ALLOWED_PRIMITIVE_PARAMS.get(primitive)
    if allowed is None:
        raise ValueError(f"Unknown primitive: {primitive!r}")
    unknown = sorted(set(params.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown params {unknown} for primitive {primitive!r}; allowed: {sorted(allowed)}"
        )
    if primitive == "threshold_gate":
        op = params.get("op", ">=")
        if not isinstance(op, str) or op.strip() not in ALLOWED_THRESHOLD_OPS:
            raise ValueError(
                f"Invalid threshold_gate op {op!r}; expected one of {sorted(ALLOWED_THRESHOLD_OPS)}"
            )


class LeafRuleSpec(BaseModel):
    """Leaf primitive scoring rule inside a rubric expression tree."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    id: str | None = None
    observable: str
    primitive: PrimitiveLiteral
    params: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None
    unit: str | None = None

    @field_validator("params", mode="before")
    @classmethod
    def _validate_params(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("LeafRuleSpec.params must be a dict")
        checked = ensure_finite_json_value(v, path="LeafRuleSpec.params")
        assert isinstance(checked, dict)
        return checked

    @model_validator(mode="after")
    def _validate_primitive_param_keys(self) -> LeafRuleSpec:
        validate_primitive_params(self.primitive, self.params)
        return self


class CombineRuleSpec(BaseModel):
    """Combinator node (`min_of`, `max_of`, `weighted_sum`, `zero_if_any_zero`, `multiplicative_cap`)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    id: str | None = None
    combine: CombineLiteral
    weights: list[float] | None = None
    children: list[RuleExpr]

    @model_validator(mode="after")
    def _validate_children(self) -> CombineRuleSpec:
        if not self.children:
            raise ValueError(f"Combinator {self.combine!r} requires at least one child")
        if self.weights is not None:
            if len(self.weights) != len(self.children):
                raise ValueError(
                    f"Combinator {self.combine!r} weights length {len(self.weights)} "
                    f"!= children length {len(self.children)}"
                )
            if any(w < 0.0 or not math.isfinite(w) for w in self.weights):
                raise ValueError(f"Combinator {self.combine!r} weights must be finite >= 0")
            if self.combine in ("weighted_sum", "zero_if_any_zero") and sum(self.weights) <= 0.0:
                raise ValueError(
                    f"Combinator {self.combine!r} requires at least one positive weight"
                )
        return self


RuleExpr = Annotated[LeafRuleSpec | CombineRuleSpec, Field(union_mode="left_to_right")]
CombineRuleSpec.model_rebuild()


class HardGateSpec(BaseModel):
    """Boolean or threshold hard veto forcing `passed = 0` (and optional `reward = 0.0`)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    id: str
    observable: str
    primitive: PrimitiveLiteral = "bool_pass"
    params: dict[str, Any] = Field(default_factory=dict)
    zero_reward_on_fail: bool = True
    zero_subscores_on_fail: bool = False
    optional_when_truth_missing: bool = False
    optional_when_not_measured: bool = False
    failure_tag: str

    @field_validator("params", mode="before")
    @classmethod
    def _validate_gate_params_finite(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("HardGateSpec.params must be a dict")
        checked = ensure_finite_json_value(v, path="HardGateSpec.params")
        assert isinstance(checked, dict)
        return checked

    @model_validator(mode="after")
    def _validate_gate_param_keys(self) -> HardGateSpec:
        validate_primitive_params(self.primitive, self.params)
        return self


class SubscoreSpec(BaseModel):
    """Top-level or stage-block weighted subscore definition."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    id: str
    weight: float
    failure_tag: str
    symptom_only: bool = False
    label: str | None = None
    expr: RuleExpr

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"Subscore weight must be finite and >= 0, got {v!r}")
        return float(v)


class StageBlockSpec(BaseModel):
    """Multi-stage composite rubric block (e.g. `calibration_score` + `delivered_image_score` on `e2e`)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    id: str
    stage: str
    weight: float
    subscore_prefix: str = ""
    truth_prefix: str | None = None
    failure_tag: str
    core_subscore_floors: dict[str, float] = Field(default_factory=dict)
    subscores: list[SubscoreSpec]

    @field_validator("weight")
    @classmethod
    def _validate_block_weight(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"StageBlock weight must be finite and >= 0, got {v!r}")
        return float(v)


class DisplayMetricSpec(BaseModel):
    """Optional explicit Truth-vs-Measured row descriptor in `rubric.json`."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    id: str
    observable: str
    label: str | None = None
    unit: str | None = None
    stage_block: str | None = None
    scale: float = 1.0
    digits: int = 4
    truth_key: str | list[str] | None = None
    target_format: str | None = None


class RubricSpec(BaseModel):
    """Declarative scoring rubric (`/tests/rubric.json`)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal["0.2"] = "0.2"
    task_id: str
    stages: list[str]
    pass_reward_min: float = 0.85
    expect_halt: bool = False
    expected_halt_codes: list[HaltCodeLiteral] = Field(default_factory=list)
    required_reported_metrics: list[str] = Field(default_factory=list)
    claim_rel_tol: float = 0.05
    claim_scale_floors: dict[str, float] = Field(default_factory=dict)
    hard_gates: list[HardGateSpec] = Field(default_factory=list)
    core_subscore_floors: dict[str, float] = Field(default_factory=dict)
    subscores: list[SubscoreSpec] = Field(default_factory=list)
    stage_blocks: list[StageBlockSpec] = Field(default_factory=list)
    display_metrics: list[DisplayMetricSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_structure(self) -> RubricSpec:
        if not self.stages:
            raise ValueError("RubricSpec.stages must be non-empty")
        if not self.subscores and not self.stage_blocks and not self.expect_halt:
            raise ValueError(
                "RubricSpec must define `subscores` or `stage_blocks` when `expect_halt` is False"
            )
        return self
