"""Declarative `rubric.json` evaluator, Layer B Claim Auditor, and self-describing `rubric_meta`."""

from __future__ import annotations

import html.parser
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.grading.extractors import WorkspaceExtractionResult
from radio_bench.grading.extractors.inspect import norm_ant_name, norm_field_name
from radio_bench.grading.schema import (
    CombineRuleSpec,
    DisplayMetricSpec,
    HardGateSpec,
    LeafRuleSpec,
    ObservableRecord,
    RubricSpec,
    RuleExpr,
    StageBlockSpec,
    StageOutputManifest,
    SubscoreSpec,
    finite_float_or_none,
    finite_observable,
)

_HIDDEN_STYLE_RE = re.compile(
    r"(?:^|;)\s*(?:"
    r"display\s*:\s*none\b"
    r"|visibility\s*:\s*(?:hidden|collapse)\b"
    r"|opacity\s*:\s*0(?:\.0+)?(?=\s*(?:;|!important|$))"
    r"|font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%|vw|vh|vmin|vmax|ch|ex)?(?=\s*(?:;|!important|$))"
    r")",
    re.IGNORECASE,
)

_HIDDEN_HTML_TAGS: frozenset[str] = frozenset({"script", "style", "template", "noscript", "head"})
_VOID_HTML_TAGS: frozenset[str] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_P_CLOSING_TAGS: frozenset[str] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "details",
        "div",
        "dl",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "table",
        "ul",
    }
)


@dataclass
class HTMLClaimSpan:
    """Single visible `<span data-observable="...">` DOM claim extracted from `report.html`."""

    observable: str
    raw_text: str
    unit: str | None = None


class _VisibleClaimHTMLParser(html.parser.HTMLParser):
    """Extract `<tag data-observable="...">` claims strictly from visible HTML DOM nodes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Stack entries: (tag_lower, is_hidden, active_claim_idx_or_none)
        self._stack: list[tuple[str, bool, int | None]] = []
        self.claims: list[HTMLClaimSpan] = []

    def _parent_hidden(self) -> bool:
        return bool(self._stack[-1][1]) if self._stack else False

    def _active_claim_indices(self) -> list[int]:
        return [c_idx for _t, hid, c_idx in self._stack if not hid and c_idx is not None]

    def _pop_if_open_before_boundary(
        self, target_tags: frozenset[str], boundary_tags: frozenset[str]
    ) -> None:
        for idx in range(len(self._stack) - 1, -1, -1):
            st_tag = self._stack[idx][0]
            if st_tag in boundary_tags:
                return
            if st_tag in target_tags:
                del self._stack[idx:]
                return

    def _auto_close_optional_siblings(self, t_low: str) -> None:
        if t_low in _P_CLOSING_TAGS:
            self._pop_if_open_before_boundary(
                frozenset({"p"}),
                frozenset({"table", "thead", "tbody", "tfoot", "tr", "td", "th"}),
            )
        if t_low == "li":
            self._pop_if_open_before_boundary(
                frozenset({"li"}), frozenset({"ul", "ol", "menu", "table"})
            )
        elif t_low in ("td", "th"):
            self._pop_if_open_before_boundary(
                frozenset({"td", "th"}),
                frozenset({"tr", "table", "thead", "tbody", "tfoot"}),
            )
        elif t_low == "tr":
            self._pop_if_open_before_boundary(
                frozenset({"tr"}), frozenset({"table", "thead", "tbody", "tfoot"})
            )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t_low = tag.lower()
        self._auto_close_optional_siblings(t_low)
        attr_map = {k.lower(): v for k, v in attrs}
        has_hidden_attr = "hidden" in attr_map
        aria_hidden = (attr_map.get("aria-hidden") or "").strip().lower() == "true"
        style_str = attr_map.get("style") or ""
        has_hidden_style = bool(_HIDDEN_STYLE_RE.search(style_str))
        is_hidden = (
            self._parent_hidden()
            or t_low in _HIDDEN_HTML_TAGS
            or has_hidden_attr
            or aria_hidden
            or has_hidden_style
        )

        claim_idx: int | None = None
        obs_id = (attr_map.get("data-observable") or "").strip()
        if obs_id and not is_hidden:
            unit_val = (attr_map.get("data-unit") or "").strip() or None
            claim_idx = len(self.claims)
            self.claims.append(HTMLClaimSpan(observable=obs_id, raw_text="", unit=unit_val))

        # Void HTML5 elements do not push onto stack
        if t_low not in _VOID_HTML_TAGS:
            self._stack.append((t_low, is_hidden, claim_idx))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # In HTML5, `<div style="display:none" />` is NOT self-closing on non-void elements!
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        t_low = tag.lower()
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx][0] == t_low:
                del self._stack[idx:]
                break

    def handle_data(self, data: str) -> None:
        if self._parent_hidden():
            return
        for c_idx in self._active_claim_indices():
            self.claims[c_idx].raw_text += data


def extract_visible_html_claims(report_html_path: Path) -> list[HTMLClaimSpan]:
    """Parse visible `<span data-observable="...">` DOM claims from `report.html`."""
    raw_html = report_html_path.read_text(encoding="utf-8", errors="replace")
    parser = _VisibleClaimHTMLParser()
    parser.feed(raw_html)
    parser.close()
    for c in parser.claims:
        c.raw_text = c.raw_text.strip()
    return parser.claims


def default_scale_floor(observable_id: str, unit: str | None = None) -> float:
    """Return a physical scale floor preventing near-zero division blowups in Claim Auditor."""
    low = observable_id.lower()
    u_low = (unit or "").strip().lower()
    if any(
        tok in low
        for tok in ("false_positive", "overflag", "bowl_ratio", "rel_err", "nrmse", "flag_growth")
    ):
        return 0.01
    if any(
        tok in low for tok in ("recall", "retention", "fraction", "pearson_r", "ncc", "pol_frac")
    ):
        return 0.05
    if u_low in ("deg", "deg/100mhz") or "deg" in low or "slope" in low:
        return 1.0
    if "rms_jy" in low:
        return 1e-5
    if u_low in ("jy", "jy/beam") or low.endswith("_jy") or "_jy_" in low:
        return 1e-3
    if "dynamic_range" in low or low.endswith("_dr") or low.endswith("_snr"):
        return 5.0
    return 1e-3


def _canon_unit_token(u: str | None) -> str:
    if not u:
        return ""
    norm = u.strip().replace("\u03bc", "u").replace("\u00b5", "u").lower().replace(" ", "")
    if norm in ("dimensionless", "none", "unitless", "1"):
        return ""
    return norm


def normalize_claim_unit(
    val: float,
    reported_unit: str | None,
    canonical_unit: str | None,
) -> tuple[bool, float]:
    """Convert `val` from `reported_unit` into `canonical_unit`; return `(unit_ok, converted)`."""
    if reported_unit is None or not reported_unit.strip():
        return True, val
    ru = _canon_unit_token(reported_unit)
    cu = _canon_unit_token(canonical_unit)
    if ru == cu:
        return True, val

    if cu in ("jy", "jy/beam"):
        if ru in ("mjy", "mjy/beam"):
            return True, val * 1e-3
        if ru in ("ujy", "ujy/beam", "microjy", "microjy/beam"):
            return True, val * 1e-6
        if ru in ("kjy", "kjy/beam"):
            return True, val * 1e3
        if ru in ("jy", "jy/beam"):
            return True, val
        return False, val

    if cu in ("jykm/s", "jy*km/s", "jy.km/s"):
        if ru in ("jykm/s", "jy*km/s", "jy.km/s"):
            return True, val
        if ru in ("mjykm/s", "mjy*km/s", "mjy.km/s"):
            return True, val * 1e-3
        if ru in ("ujykm/s", "ujy*km/s", "ujy.km/s", "microjykm/s"):
            return True, val * 1e-6
        return False, val

    if cu == "deg/100mhz":
        if ru == "deg/100mhz":
            return True, val
        if ru == "deg/ghz":
            return True, val * 0.1
        if ru == "deg/mhz":
            return True, val * 100.0
        return False, val

    if cu == "arcsec":
        if ru in ("arcsec", "asec", '"', "''"):
            return True, val
        if ru in ("arcmin", "amin", "'"):
            return True, val * 60.0
        if ru in ("deg", "degree", "degrees"):
            return True, val * 3600.0
        if ru == "mas":
            return True, val * 1e-3
        return False, val

    if cu == "deg":
        if ru in ("deg", "degree", "degrees"):
            return True, val
        if ru in ("rad", "radian", "radians"):
            return True, math.degrees(val)
        if ru in ("arcmin", "amin"):
            return True, val / 60.0
        if ru in ("arcsec", "asec"):
            return True, val / 3600.0
        return False, val

    if cu == "hz":
        if ru == "hz":
            return True, val
        if ru == "khz":
            return True, val * 1e3
        if ru == "mhz":
            return True, val * 1e6
        if ru == "ghz":
            return True, val * 1e9
        return False, val

    if cu == "":
        if ru in ("%", "pct", "percent"):
            return True, val * 0.01
        return False, val

    return False, val


_CLAIM_NUM_WITH_UNIT_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*([A-Za-z\u03bc\u00b5%'\"][A-Za-z0-9\u03bc\u00b5%/'\"*._ -]*))?$"
)


def _parse_claim_scalar_with_unit(raw_val: Any) -> tuple[float | None, str | None]:
    if isinstance(raw_val, (bool, np.bool_)) or raw_val is None:
        return None, None
    if isinstance(raw_val, (int, float, np.integer, np.floating)):
        fv = float(raw_val)
        return (fv, None) if math.isfinite(fv) else (None, None)
    if isinstance(raw_val, str):
        cleaned = raw_val.strip().replace(",", "")
        m = _CLAIM_NUM_WITH_UNIT_RE.fullmatch(cleaned)
        if m:
            try:
                fv = float(m.group(1))
            except ValueError:
                return None, None
            if not math.isfinite(fv):
                return None, None
            inline_u = (m.group(2) or "").strip() or None
            return fv, inline_u
    return None, None


def _parse_claim_scalar(raw_val: Any) -> float | None:
    fv, _u = _parse_claim_scalar_with_unit(raw_val)
    return fv


def audit_claims(
    workspace_dir: Path,
    rubric: RubricSpec,
    manifest: StageOutputManifest | None,
    observables: dict[str, ObservableRecord],
    *,
    raw_manifest_dict: dict[str, Any] | None = None,
) -> tuple[dict[str, ObservableRecord], list[str], set[str]]:
    """Audit Layer B `stage_output.json["reported_metrics"]` and visible `report.html` claims."""
    errors: list[str] = []
    manifest_claims: dict[str, list[tuple[Any, str | None, str]]] = {}
    html_claims: dict[str, list[tuple[Any, str | None, str]]] = {}

    if manifest is not None and manifest.reported_metrics:
        for obs_key, rep_metric in manifest.reported_metrics.items():
            manifest_claims.setdefault(obs_key, []).append(
                (rep_metric.value, rep_metric.unit, "stage_output.json")
            )
    elif isinstance(raw_manifest_dict, dict) and isinstance(
        raw_manifest_dict.get("reported_metrics"), dict
    ):
        for obs_key, raw_entry in raw_manifest_dict["reported_metrics"].items():
            if isinstance(raw_entry, dict):
                manifest_claims.setdefault(str(obs_key), []).append(
                    (raw_entry.get("value"), raw_entry.get("unit"), "stage_output.json")
                )
            else:
                manifest_claims.setdefault(str(obs_key), []).append(
                    (raw_entry, None, "stage_output.json")
                )

    report_rel = (
        manifest.artifacts.report_path
        if (manifest is not None and manifest.artifacts.report_path)
        else "report.html"
    )
    report_path = workspace_dir / report_rel
    # Guard against path traversal outside workspace_dir
    try:
        if not report_path.resolve().is_relative_to(workspace_dir.resolve()):
            report_path = workspace_dir / "report.html"
    except Exception:
        report_path = workspace_dir / "report.html"

    html_exists = report_path.is_file()
    if html_exists:
        try:
            spans = extract_visible_html_claims(report_path)
            for sp in spans:
                html_claims.setdefault(sp.observable, []).append(
                    (sp.raw_text, sp.unit, report_path.name)
                )
        except Exception as exc:
            errors.append(f"Failed parsing {report_path.name}: {exc}")

    is_report_stage = "report" in {s.strip().lower() for s in rubric.stages}
    if is_report_stage:
        observables["report.html_valid"] = ObservableRecord(
            value=bool(html_exists),
            status="measured",
            provenance=report_path.name,
        )

    missing_req = 0
    for req_key in rubric.required_reported_metrics:
        if is_report_stage:
            present = req_key in html_claims
        else:
            present = (req_key in manifest_claims) or (req_key in html_claims)
        if not present:
            missing_req += 1
            errors.append(f"Missing required reported metric claim: {req_key}")

    all_claim_entries: list[tuple[str, Any, str | None, str]] = []
    for k, entries in manifest_claims.items():
        for val, unit, src in entries:
            all_claim_entries.append((k, val, unit, src))
    for k, entries in html_claims.items():
        for val, unit, src in entries:
            all_claim_entries.append((k, val, unit, src))

    claimed_keys: set[str] = {k for k, _v, _u, _s in all_claim_entries}

    unsupported_cnt = 0
    max_scale_err = 0.0
    audited_any = bool(all_claim_entries or rubric.required_reported_metrics)

    for obs_key, rep_val, rep_unit, src_name in all_claim_entries:
        ext_rec = observables.get(obs_key)
        if ext_rec is None or ext_rec.status != "measured" or ext_rec.value is None:
            unsupported_cnt += 1
            max_scale_err = max(max_scale_err, 1.0)
            errors.append(
                f"Unsupported claim {obs_key!r} in {src_name}: observable status is "
                f"{None if ext_rec is None else ext_rec.status!r}"
            )
            continue

        ext_val = ext_rec.value
        if isinstance(ext_val, bool):
            rep_bool: bool | None = None
            if isinstance(rep_val, bool):
                rep_bool = rep_val
            elif isinstance(rep_val, str) and rep_val.strip().lower() in ("true", "false"):
                rep_bool = rep_val.strip().lower() == "true"
            if rep_bool is None or rep_bool != ext_val:
                unsupported_cnt += 1
                max_scale_err = max(max_scale_err, 1.0)
                errors.append(
                    f"Claim mismatch for {obs_key!r} in {src_name}: reported {rep_val!r} != {ext_val!r}"
                )
        elif isinstance(ext_val, (int, float)):
            rep_num, inline_u = _parse_claim_scalar_with_unit(rep_val)
            if rep_num is None:
                unsupported_cnt += 1
                max_scale_err = max(max_scale_err, 1.0)
                errors.append(
                    f"Non-numeric claim for numeric observable {obs_key!r} in {src_name}: {rep_val!r}"
                )
                continue
            eff_unit = rep_unit if (rep_unit is not None and str(rep_unit).strip()) else inline_u
            unit_ok, norm_rep = normalize_claim_unit(
                rep_num, str(eff_unit) if eff_unit is not None else None, ext_rec.unit
            )
            if not unit_ok or not math.isfinite(norm_rep):
                unsupported_cnt += 1
                max_scale_err = max(max_scale_err, 1.0)
                errors.append(
                    f"Incompatible unit {eff_unit!r} for {obs_key!r} (expected {ext_rec.unit!r}) in {src_name}"
                )
                continue
            ext_f = float(ext_val)
            sf = float(
                rubric.claim_scale_floors.get(obs_key, default_scale_floor(obs_key, ext_rec.unit))
            )
            scale_err = abs(norm_rep - ext_f) / max(abs(ext_f), sf, 1e-12)
            if not math.isfinite(scale_err):
                unsupported_cnt += 1
                max_scale_err = max(max_scale_err, 1.0)
                continue
            max_scale_err = max(max_scale_err, scale_err)
            if scale_err > float(rubric.claim_rel_tol):
                unsupported_cnt += 1
                errors.append(
                    f"Unsupported numeric claim {obs_key!r} in {src_name}: "
                    f"reported={norm_rep:.6g}, measured={ext_f:.6g}, scale_err={scale_err:.4f} "
                    f"> tol={rubric.claim_rel_tol}"
                )
        elif isinstance(ext_val, list):
            is_num_list = bool(ext_val) and all(
                isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
                for x in ext_val
            )
            if is_num_list:
                raw_elems = (
                    [s.strip() for s in rep_val.split(",") if s.strip()]
                    if isinstance(rep_val, str)
                    else (list(rep_val) if isinstance(rep_val, list) else [])
                )
                if len(raw_elems) != len(ext_val):
                    unsupported_cnt += 1
                    max_scale_err = max(max_scale_err, 1.0)
                    errors.append(
                        f"Claim list length mismatch for {obs_key!r} in {src_name}: "
                        f"{len(raw_elems)} != {len(ext_val)}"
                    )
                    continue
                sf = float(
                    rubric.claim_scale_floors.get(
                        obs_key, default_scale_floor(obs_key, ext_rec.unit)
                    )
                )
                elem_failed = False
                for r_el, e_el in zip(raw_elems, ext_val, strict=True):
                    rn, in_u = _parse_claim_scalar_with_unit(r_el)
                    if rn is None:
                        elem_failed = True
                        break
                    eff_u = rep_unit if (rep_unit is not None and str(rep_unit).strip()) else in_u
                    u_ok, norm_rn = normalize_claim_unit(
                        rn, str(eff_u) if eff_u is not None else None, ext_rec.unit
                    )
                    if not u_ok or not math.isfinite(norm_rn):
                        elem_failed = True
                        break
                    ef = float(e_el)
                    s_err = abs(norm_rn - ef) / max(abs(ef), sf, 1e-12)
                    if not math.isfinite(s_err):
                        elem_failed = True
                        break
                    max_scale_err = max(max_scale_err, s_err)
                    if s_err > float(rubric.claim_rel_tol):
                        elem_failed = True
                if elem_failed:
                    unsupported_cnt += 1
                    max_scale_err = max(max_scale_err, 1.0)
                    errors.append(f"Unsupported numeric list claim for {obs_key!r} in {src_name}")
            else:
                if isinstance(rep_val, str):
                    rep_items = {x.strip().upper() for x in rep_val.split(",") if x.strip()}
                elif isinstance(rep_val, list):
                    rep_items = {str(x).strip().upper() for x in rep_val if str(x).strip()}
                else:
                    rep_items = set()
                ext_items = {str(x).strip().upper() for x in ext_val if str(x).strip()}
                if rep_items != ext_items:
                    unsupported_cnt += 1
                    max_scale_err = max(max_scale_err, 1.0)
                    errors.append(
                        f"Claim set mismatch for {obs_key!r} in {src_name}: {sorted(rep_items)} != {sorted(ext_items)}"
                    )
        else:
            if str(rep_val).strip().upper() != str(ext_val).strip().upper():
                unsupported_cnt += 1
                max_scale_err = max(max_scale_err, 1.0)
                errors.append(
                    f"Claim string mismatch for {obs_key!r} in {src_name}: {rep_val!r} != {ext_val!r}"
                )

    if missing_req > 0:
        max_scale_err = max(max_scale_err, 1.0)

    if not audited_any:
        scale_err_rec = ObservableRecord(
            value=None,
            status="not_applicable",
            provenance="stage_output.json+report.html",
        )
    elif not all_claim_entries and missing_req > 0:
        scale_err_rec = ObservableRecord(
            value=None,
            status="missing_artifact",
            provenance="stage_output.json+report.html",
        )
    else:
        scale_err_rec = finite_observable(
            max_scale_err,
            provenance="stage_output.json+report.html",
            missing_status="extractor_error",
            ndigits=6,
        )

    claim_obs = {
        "proc.claim_max_scale_err": scale_err_rec,
        "proc.unsupported_claim_count": ObservableRecord(
            value=int(unsupported_cnt) if audited_any else None,
            status="measured" if audited_any else "not_applicable",
            provenance="stage_output.json+report.html",
        ),
        "proc.missing_required_claim_count": ObservableRecord(
            value=int(missing_req) if audited_any else 0,
            status="measured" if audited_any else "not_applicable",
            provenance="stage_output.json+report.html",
        ),
    }
    return claim_obs, errors, claimed_keys


# ---------------------------------------------------------------------------
# Dotted `ground_truth.json` Lookup Helpers
# ---------------------------------------------------------------------------


def _lookup_single_dotted_path(root: dict[str, Any], dotted_key: str) -> tuple[bool, Any]:
    cur: Any = root
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and 0 <= int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return (cur is not None), cur


def resolve_truth_value(
    ground_truth: dict[str, Any],
    key_spec: str | list[str] | None,
    *,
    truth_prefix: str | None = None,
    default: Any = None,
) -> tuple[bool, Any]:
    """Resolve a dotted path (or fallback list of dotted paths) inside `ground_truth`."""
    if key_spec is None:
        return (default is not None), default

    keys = [key_spec] if isinstance(key_spec, str) else list(key_spec)
    scopes: list[dict[str, Any]] = []
    if truth_prefix and isinstance(ground_truth.get(truth_prefix), dict):
        scopes.append(ground_truth[truth_prefix])
    scopes.append(ground_truth)

    for scope in scopes:
        for k in keys:
            found, val = _lookup_single_dotted_path(scope, str(k))
            if found:
                return True, val
    return (default is not None), default


def _resolve_scalar_param(
    params: dict[str, Any],
    direct_key: str,
    truth_key_param: str,
    ground_truth: dict[str, Any],
    truth_prefix: str | None,
    *,
    scale_param: str | None = None,
    default: float | None = None,
) -> tuple[bool, float]:
    scale = 1.0
    if scale_param and scale_param in params and params[scale_param] is not None:
        sf = finite_float_or_none(params[scale_param])
        if sf is None:
            return False, 0.0
        scale = sf
    if truth_key_param in params and params[truth_key_param] is not None:
        found, raw = resolve_truth_value(
            ground_truth, params[truth_key_param], truth_prefix=truth_prefix
        )
        if found and raw is not None:
            rv = finite_float_or_none(raw)
            if rv is not None:
                if scale_param is None and f"{truth_key_param}_scale_key" in params:
                    s_found, s_val = resolve_truth_value(
                        ground_truth,
                        params[f"{truth_key_param}_scale_key"],
                        truth_prefix=truth_prefix,
                    )
                    if s_found and s_val is not None:
                        sv = finite_float_or_none(s_val)
                        if sv is None:
                            return False, 0.0
                        scale *= sv
                prod = rv * scale
                if math.isfinite(prod):
                    return True, prod
    if direct_key in params and params[direct_key] is not None:
        dv = finite_float_or_none(params[direct_key])
        if dv is not None and math.isfinite(dv * scale):
            return True, dv * scale
        return False, 0.0
    if default is not None:
        df = finite_float_or_none(default)
        if df is not None and math.isfinite(df * scale):
            return True, df * scale
    return False, 0.0


def _to_normalized_set(raw: Any, *, mode: str = "upper") -> set[str]:
    if raw is None:
        return set()
    items: list[Any]
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple, set)):
        items = [x for x in raw if x is not None]
    else:
        items = [raw]

    out: set[str] = set()
    for it in items:
        if mode == "ant":
            s = norm_ant_name(it)
        elif mode == "field":
            s = norm_field_name(it).upper()
        else:
            s = str(it).strip().upper()
        if s:
            out.add(s)
    return out


# ---------------------------------------------------------------------------
# Leaf Primitive & Combinator Evaluators
# ---------------------------------------------------------------------------


@dataclass
class LeafEvalTrace:
    """Evaluation trace for one leaf primitive rule (drives `rubric_meta.display_rows`)."""

    observable: str
    primitive: str
    label: str
    unit: str | None
    stage: str
    truth_display: str
    truth_value: Any
    measured_value: Any
    status: str
    score: float


def _evaluate_leaf(
    leaf: LeafRuleSpec,
    observables: dict[str, ObservableRecord],
    ground_truth: dict[str, Any],
    *,
    stage: str,
    truth_prefix: str | None,
    named_scores: dict[str, float],
    leaf_traces: list[LeafEvalTrace],
    subscore_prefix: str = "",
) -> float:
    rec = observables.get(leaf.observable)
    meas_ok = rec is not None and rec.status == "measured" and rec.value is not None
    val = rec.value if meas_ok and rec is not None else None
    status_str = rec.status if rec is not None else "missing_artifact"
    unit = leaf.unit if leaf.unit is not None else (rec.unit if rec is not None else None)
    label = leaf.label or leaf.observable
    p = leaf.params

    score = 0.0
    truth_disp = ""
    truth_val: Any = None

    prim = leaf.primitive

    if prim == "bool_pass":
        expected = bool(p.get("expected", True))
        truth_val = expected
        truth_disp = str(expected)
        if meas_ok and isinstance(val, bool):
            score = 1.0 if (val is expected) else 0.0

    elif prim in ("exact_set", "set_jaccard", "set_f1"):
        norm_mode = str(
            p.get(
                "norm",
                "ant" if "antenna" in leaf.observable or "station" in leaf.observable else "field",
            )
        )
        _, raw_truth = resolve_truth_value(
            ground_truth, p.get("truth_key"), truth_prefix=truth_prefix, default=p.get("expected")
        )
        true_set = _to_normalized_set(raw_truth, mode=norm_mode)
        truth_val = sorted(true_set)
        truth_disp = ", ".join(sorted(true_set)) if true_set else "[]"

        ignore_set: set[str] = set()
        if "ignore_set" in p:
            ignore_set |= _to_normalized_set(p["ignore_set"], mode=norm_mode)
        if "ignore_truth_key" in p:
            ig_keys = (
                p["ignore_truth_key"]
                if isinstance(p["ignore_truth_key"], list)
                else [p["ignore_truth_key"]]
            )
            for ik in ig_keys:
                f_ig, raw_ig = resolve_truth_value(ground_truth, ik, truth_prefix=truth_prefix)
                if f_ig and raw_ig is not None:
                    ignore_set |= _to_normalized_set(raw_ig, mode=norm_mode)

        if meas_ok:
            pred_raw_set = _to_normalized_set(val, mode=norm_mode)
            eff_pred = pred_raw_set - (ignore_set - true_set)

            if prim == "exact_set":
                if eff_pred == true_set:
                    score = 1.0
                else:
                    for pm in p.get("partial_match_sets", []):
                        if not isinstance(pm, dict):
                            continue
                        pm_keys = pm.get("truth_keys", [])
                        pm_union: set[str] = set()
                        all_pm_present = True
                        for pmk in pm_keys:
                            f_pm, r_pm = resolve_truth_value(
                                ground_truth, pmk, truth_prefix=truth_prefix
                            )
                            s_pm = _to_normalized_set(r_pm, mode=norm_mode) if f_pm else set()
                            if not s_pm:
                                all_pm_present = False
                                break
                            pm_union |= s_pm
                        if all_pm_present and pm_union and eff_pred == pm_union:
                            score = float(pm.get("score", 0.85))
                            break
            elif prim == "set_jaccard":
                if not true_set:
                    score = 1.0 if not eff_pred else 0.0
                elif not eff_pred:
                    score = 0.0
                else:
                    inter = len(eff_pred & true_set)
                    union = len(eff_pred | true_set)
                    score = float(inter / max(union, 1))
            else:  # set_f1
                if not true_set:
                    score = 1.0 if not eff_pred else 0.0
                elif not eff_pred:
                    score = 0.0
                else:
                    tp = len(eff_pred & true_set)
                    fp = len(eff_pred - true_set)
                    fn = len(true_set - eff_pred)
                    score = float((2 * tp) / max(2 * tp + fp + fn, 1))

            # Optional cross-role overlap penalty (e.g. calibrator confused as target or vice-versa)
            forbid_keys = p.get("forbid_overlap_truth_keys", [])
            if forbid_keys and score > 0.0:
                forbidden_set: set[str] = set()
                for fk in forbid_keys:
                    f_forb, r_forb = resolve_truth_value(
                        ground_truth, fk, truth_prefix=truth_prefix
                    )
                    if f_forb and r_forb is not None:
                        forbidden_set |= _to_normalized_set(r_forb, mode=norm_mode)
                if eff_pred & forbidden_set:
                    pen = float(p.get("overlap_penalty_factor", 0.0))
                    score *= pen

    elif prim == "piecewise_floor_ramp":
        base_floor = float(p.get("base_floor", 0.20))
        denom = float(p.get("denom", 4.0))
        truth_val = denom
        truth_disp = f"{denom:g}"
        if meas_ok and isinstance(val, (int, float)) and not isinstance(val, bool):
            frac = min(1.0, max(0.0, float(val) / max(denom, 1e-12)))
            score = base_floor + (1.0 - base_floor) * frac

    elif prim == "discrete_map":
        mapping: dict[str, Any] = p.get("mapping", {})
        truth_disp = "complete"
        if meas_ok and val is not None:
            k_str = (
                str(int(val))
                if isinstance(val, (int, np.integer)) and not isinstance(val, bool)
                else str(val)
            )
            score = float(mapping.get(k_str, p.get("default", 0.0)))

    elif prim == "upper_ramp":
        ok_p, pass_max = _resolve_scalar_param(
            p, "pass_max", "pass_max_key", ground_truth, truth_prefix, scale_param="pass_max_scale"
        )
        if "pass_max_mult_key" in p:
            f_m, v_m = resolve_truth_value(
                ground_truth, p["pass_max_mult_key"], truth_prefix=truth_prefix
            )
            if f_m and v_m is not None:
                mv = finite_float_or_none(v_m)
                if mv is not None and math.isfinite(pass_max * mv):
                    pass_max *= mv
                else:
                    ok_p = False
        if "zero_at" in p or "zero_at_key" in p:
            _, zero_at = _resolve_scalar_param(
                p, "zero_at", "zero_at_key", ground_truth, truth_prefix, scale_param="zero_at_scale"
            )
        elif "decay_span_mult_of_pass_max" in p:
            zero_at = pass_max + float(p["decay_span_mult_of_pass_max"]) * pass_max
        else:
            zero_at = pass_max + float(p.get("decay_span", 0.0))

        strict_pass = bool(p.get("strict_pass", False))
        scale_above = float(p.get("scale_above_pass", 1.0))
        use_abs = bool(p.get("use_abs", False))
        op_sym = "<" if strict_pass else "<="
        truth_val = pass_max if ok_p else None
        truth_disp = f"{op_sym} {pass_max:.4g}" if ok_p else ""

        if ok_p and meas_ok and isinstance(val, (int, float)) and not isinstance(val, bool):
            x = abs(float(val)) if use_abs else float(val)
            in_pass = (x < pass_max) if strict_pass else (x <= pass_max)
            if in_pass:
                score = 1.0
            elif zero_at <= pass_max:
                score = 0.0
            else:
                score = scale_above * max(0.0, 1.0 - (x - pass_max) / (zero_at - pass_max))

    elif prim == "lower_ramp":
        ok_p, pass_min = _resolve_scalar_param(
            p, "pass_min", "pass_min_key", ground_truth, truth_prefix, scale_param="pass_min_scale"
        )
        if "zero_at" in p or "zero_at_key" in p:
            _, zero_at = _resolve_scalar_param(
                p, "zero_at", "zero_at_key", ground_truth, truth_prefix, scale_param="zero_at_scale"
            )
        elif "zero_at_scale_of_pass_min" in p:
            zero_at = float(p["zero_at_scale_of_pass_min"]) * pass_min
        elif "decay_span" in p:
            zero_at = pass_min - float(p["decay_span"])
        else:
            zero_at = 0.0

        truth_val = pass_min if ok_p else None
        truth_disp = f">= {pass_min:.4g}" if ok_p else ""
        if ok_p and meas_ok and isinstance(val, (int, float)) and not isinstance(val, bool):
            x = float(val)
            if x >= pass_min:
                score = 1.0
            elif pass_min <= zero_at:
                score = 0.0
            else:
                # Match v0.1 max(denom, 1.0) vs max(denom, 1e-6) smoothly
                min_denom = float(p.get("min_denom", 1e-6))
                score = max(0.0, (x - zero_at) / max(pass_min - zero_at, min_denom))

    elif prim == "rel_err_ramp":
        ok_t, target = _resolve_scalar_param(
            p, "target", "truth_key", ground_truth, truth_prefix, scale_param="target_scale"
        )
        ok_tol, pass_tol = _resolve_scalar_param(
            p, "pass_rel_tol", "pass_rel_tol_key", ground_truth, truth_prefix, default=0.05
        )
        if "zero_at_rel_err" in p or "zero_at_rel_err_key" in p:
            _, zero_at = _resolve_scalar_param(
                p, "zero_at_rel_err", "zero_at_rel_err_key", ground_truth, truth_prefix
            )
        elif "decay_span_mult_of_tol" in p:
            zero_at = pass_tol + float(p["decay_span_mult_of_tol"]) * pass_tol
        else:
            zero_at = pass_tol + float(p.get("decay_span", 2.0 * pass_tol))

        truth_val = target if ok_t else None
        truth_disp = f"{target:.4g} +/-{pass_tol * 100.0:.1f}%" if (ok_t and ok_tol) else ""
        if (
            ok_t
            and ok_tol
            and meas_ok
            and isinstance(val, (int, float))
            and not isinstance(val, bool)
        ):
            rel_err = abs(float(val) - target) / max(abs(target), 1e-12)
            if rel_err <= pass_tol:
                score = 1.0
            elif zero_at <= pass_tol:
                score = 0.0
            else:
                score = max(0.0, 1.0 - (rel_err - pass_tol) / max(zero_at - pass_tol, 1e-6))

    elif prim == "abs_err_ramp":
        ok_t, target = _resolve_scalar_param(
            p, "target", "truth_key", ground_truth, truth_prefix, scale_param="target_scale"
        )
        ok_tol, pass_tol = _resolve_scalar_param(
            p, "pass_abs_tol", "pass_abs_tol_key", ground_truth, truth_prefix, default=0.25
        )
        if "zero_at_abs_err" in p or "zero_at_abs_err_key" in p:
            _, zero_at = _resolve_scalar_param(
                p, "zero_at_abs_err", "zero_at_abs_err_key", ground_truth, truth_prefix
            )
        else:
            zero_at = pass_tol + float(p.get("decay_span", 1.25))

        truth_val = target if ok_t else None
        truth_disp = f"{target:.4g} +/-{pass_tol:.4g}" if (ok_t and ok_tol) else ""
        if (
            ok_t
            and ok_tol
            and meas_ok
            and isinstance(val, (int, float))
            and not isinstance(val, bool)
        ):
            abs_err = abs(float(val) - target)
            if abs_err <= pass_tol:
                score = 1.0
            elif zero_at <= pass_tol:
                score = 0.0
            else:
                score = max(0.0, 1.0 - (abs_err - pass_tol) / max(zero_at - pass_tol, 1e-6))

    elif prim == "window_ramp":
        ok_lo, w_min = _resolve_scalar_param(
            p, "min", "min_key", ground_truth, truth_prefix, scale_param="min_scale"
        )
        ok_hi, w_max = _resolve_scalar_param(
            p, "max", "max_key", ground_truth, truth_prefix, scale_param="max_scale"
        )
        pad_low = float(p.get("pad_low", 0.0))
        pad_high = float(p.get("pad_high", 0.0))
        low_zero_at = float(p["low_zero_at"]) if "low_zero_at" in p else (w_min - pad_low)
        high_zero_at = float(p["high_zero_at"]) if "high_zero_at" in p else (w_max + pad_high)

        truth_val = [w_min, w_max] if (ok_lo and ok_hi) else (w_min if ok_lo else None)
        truth_disp = (
            f"[{w_min:.4g}, {w_max:.4g}]"
            if (ok_lo and ok_hi)
            else (f">= {w_min:.4g}" if ok_lo else "")
        )
        if ok_lo and meas_ok and isinstance(val, (int, float)) and not isinstance(val, bool):
            x = float(val)
            strict_positive = bool(p.get("strict_positive", False))
            if strict_positive and x <= 0.0:
                score = 0.0
            elif not ok_hi:
                if x >= w_min:
                    score = 1.0
                elif w_min <= low_zero_at:
                    score = 0.0
                else:
                    score = max(0.0, (x - low_zero_at) / max(w_min - low_zero_at, 1e-6))
            elif w_min <= x <= w_max:
                score = 1.0
            elif x < w_min:
                if w_min <= low_zero_at:
                    score = 0.0
                else:
                    score = max(0.0, (x - low_zero_at) / max(w_min - low_zero_at, 1e-6))
            else:
                if high_zero_at <= w_max:
                    score = 0.0
                else:
                    score = max(0.0, 1.0 - (x - w_max) / max(high_zero_at - w_max, 1e-6))

    elif prim == "threshold_gate":
        op = str(p.get("op", ">=")).strip()
        ok_lim, limit = _resolve_scalar_param(
            p, "limit", "limit_key", ground_truth, truth_prefix, scale_param="scale"
        )
        if "mult_key" in p:
            f_mk, v_mk = resolve_truth_value(ground_truth, p["mult_key"], truth_prefix=truth_prefix)
            if f_mk and v_mk is not None:
                mk_f = finite_float_or_none(v_mk)
                if mk_f is not None and math.isfinite(limit * mk_f):
                    limit *= mk_f
                else:
                    ok_lim = False
        truth_val = limit if ok_lim else None
        truth_disp = f"{op} {limit:.4g}" if ok_lim else ""
        if ok_lim and meas_ok and isinstance(val, (int, float)) and not isinstance(val, bool):
            x = abs(float(val)) if bool(p.get("use_abs", False)) else float(val)
            if op == ">=":
                score = 1.0 if x >= limit else 0.0
            elif op == ">":
                score = 1.0 if x > limit else 0.0
            elif op == "<=":
                score = 1.0 if x <= limit else 0.0
            elif op == "<":
                score = 1.0 if x < limit else 0.0
            elif op == "==":
                score = 1.0 if abs(x - limit) <= 1e-9 else 0.0
            else:
                raise ValueError(f"Unsupported threshold_gate op: {op!r}")
        elif (
            bool(p.get("pass_if_not_applicable", False))
            and rec is not None
            and rec.status == "not_applicable"
        ):
            score = 1.0

    score = min(1.0, max(0.0, float(score)))
    if leaf.id:
        named_scores[leaf.id] = score
        if subscore_prefix:
            named_scores[f"{subscore_prefix}{leaf.id}"] = score

    leaf_traces.append(
        LeafEvalTrace(
            observable=leaf.observable,
            primitive=prim,
            label=label,
            unit=unit,
            stage=stage,
            truth_display=truth_disp,
            truth_value=truth_val,
            measured_value=val,
            status=status_str,
            score=round(score, 6),
        )
    )
    return score


def evaluate_rule_expr(
    expr: RuleExpr,
    observables: dict[str, ObservableRecord],
    ground_truth: dict[str, Any],
    *,
    stage: str,
    truth_prefix: str | None,
    named_scores: dict[str, float],
    leaf_traces: list[LeafEvalTrace],
    subscore_prefix: str = "",
) -> float:
    """Recursively evaluate a `LeafRuleSpec` or `CombineRuleSpec` expression tree."""
    if isinstance(expr, LeafRuleSpec):
        return _evaluate_leaf(
            expr,
            observables,
            ground_truth,
            stage=stage,
            truth_prefix=truth_prefix,
            named_scores=named_scores,
            leaf_traces=leaf_traces,
            subscore_prefix=subscore_prefix,
        )

    assert isinstance(expr, CombineRuleSpec)
    child_scores = [
        evaluate_rule_expr(
            ch,
            observables,
            ground_truth,
            stage=stage,
            truth_prefix=truth_prefix,
            named_scores=named_scores,
            leaf_traces=leaf_traces,
            subscore_prefix=subscore_prefix,
        )
        for ch in expr.children
    ]

    comb = expr.combine
    if comb == "min_of":
        res = min(child_scores)
    elif comb == "max_of":
        res = max(child_scores)
    elif comb == "weighted_sum":
        if expr.weights is not None:
            w_sum = sum(expr.weights)
            res = (
                sum(w * s for w, s in zip(expr.weights, child_scores, strict=True)) / w_sum
                if w_sum > 0.0
                else 0.0
            )
        else:
            res = sum(child_scores) / len(child_scores)
    elif comb == "zero_if_any_zero":
        if any(s <= 0.0 for s in child_scores):
            res = 0.0
        elif expr.weights is not None:
            w_sum = sum(expr.weights)
            res = (
                sum(w * s for w, s in zip(expr.weights, child_scores, strict=True)) / w_sum
                if w_sum > 0.0
                else 0.0
            )
        else:
            res = sum(child_scores) / len(child_scores)
    elif comb == "multiplicative_cap":
        res = 1.0
        for s in child_scores:
            res *= s
    else:
        raise ValueError(f"Unsupported combinator: {comb!r}")

    res = min(1.0, max(0.0, float(res)))
    if expr.id:
        named_scores[expr.id] = res
        if subscore_prefix:
            named_scores[f"{subscore_prefix}{expr.id}"] = res
    return res


# ---------------------------------------------------------------------------
# Full Rubric Evaluation (`evaluate_rubric`)
# ---------------------------------------------------------------------------


@dataclass
class RubricEvaluationResult:
    """Complete result of evaluating `RubricSpec` over extracted observables + `ground_truth`."""

    reward: float
    passed: int
    reward_dict: dict[str, float | int]
    subscores: dict[str, float]
    observables: dict[str, ObservableRecord]
    rubric_meta: dict[str, Any]
    triggered_failure_tags: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _derive_cross_role_confusion_observable(
    observables: dict[str, ObservableRecord],
    ground_truth: dict[str, Any],
) -> None:
    """Populate `inspect.cross_role_confusion` when `inspect.*_list` observables exist."""
    p_rec = observables.get("inspect.phase_cal_list")
    t_rec = observables.get("inspect.target_fields_list")
    if p_rec is None or t_rec is None or p_rec.status != "measured" or t_rec.status != "measured":
        return

    pred_phase = _to_normalized_set(p_rec.value, mode="field")
    pred_targets = _to_normalized_set(t_rec.value, mode="field")

    cal_set: set[str] = set()
    for k in (
        "flux_calibrator",
        "fields.flux_calibrator",
        "bandpass_calibrator",
        "fields.bandpass_calibrator",
        "phase_calibrator",
        "fields.phase_calibrator",
        "pol_leakage_calibrator",
        "fields.pol_leakage_calibrator",
        "pol_angle_calibrator",
    ):
        found, v = resolve_truth_value(ground_truth, k)
        if found and v is not None:
            cal_set |= _to_normalized_set(v, mode="field")

    true_targets: set[str] = set()
    for tk in ("target_fields", "fields.target_fields", "mosaic_target_fields"):
        found, v = resolve_truth_value(ground_truth, tk)
        if found and v is not None:
            true_targets |= _to_normalized_set(v, mode="field")

    confused = bool((pred_targets & cal_set) or (pred_phase & true_targets))
    observables["inspect.cross_role_confusion"] = ObservableRecord(
        value=confused,
        status="measured",
        provenance=p_rec.provenance,
    )


def _derive_raw_data_unmodified_observable(
    observables: dict[str, ObservableRecord],
    ground_truth: dict[str, Any],
) -> tuple[str | None]:
    """Compare `proc.raw_data_blake2b` against `ground_truth` digest when present."""
    found, exp_hash = resolve_truth_value(
        ground_truth, ["raw_data_blake2b", "truth.raw_data_blake2b", "tolerances.raw_data_blake2b"]
    )
    if not found or not isinstance(exp_hash, str) or not exp_hash.strip():
        observables["proc.raw_data_unmodified"] = ObservableRecord(
            value=None,
            status="not_applicable",
            provenance="ground_truth.json",
        )
        return None

    meas_rec = observables.get("proc.raw_data_blake2b")
    if meas_rec is None or meas_rec.status != "measured" or not isinstance(meas_rec.value, str):
        observables["proc.raw_data_unmodified"] = ObservableRecord(
            value=False,
            status=meas_rec.status if meas_rec is not None else "missing_artifact",
            provenance="input.ms",
        )
        return "Unable to verify raw_data_blake2b against ground_truth.json"

    unmodified = meas_rec.value.strip().lower() == exp_hash.strip().lower()
    observables["proc.raw_data_unmodified"] = ObservableRecord(
        value=unmodified,
        status="measured",
        provenance="input.ms vs ground_truth.raw_data_blake2b",
    )
    if not unmodified:
        return (
            f"Raw input.ms DATA/row-keys mutated: measured blake2b={meas_rec.value[:16]}... "
            f"!= expected={exp_hash.strip()[:16]}..."
        )
    return None


def _derive_spec_nrmse_if_needed(
    observables: dict[str, ObservableRecord],
    ground_truth: dict[str, Any],
    rubric: RubricSpec,
) -> None:
    """Compute `precal.flux_cal_model_spec_nrmse` from `ground_truth` spectrum if referenced."""
    cur = observables.get("precal.flux_cal_model_spec_nrmse")
    spec_rec = observables.get("precal.flux_cal_model_spectrum_jy")
    if (
        cur is not None
        and cur.status == "measured"
        or spec_rec is None
        or spec_rec.status != "measured"
        or not isinstance(spec_rec.value, list)
    ):
        return

    # Check if any leaf rule on `precal.flux_cal_model_spec_nrmse` specifies `spectrum_truth_key`
    def _find_spec_key(expr: RuleExpr) -> str | list[str] | None:
        if isinstance(expr, LeafRuleSpec):
            if expr.observable == "precal.flux_cal_model_spec_nrmse":
                return expr.params.get("spectrum_truth_key")
            return None
        for ch in expr.children:
            res = _find_spec_key(ch)
            if res is not None:
                return res
        return None

    spec_key: str | list[str] | None = None
    spec_truth_prefix: str | None = None
    for sub in rubric.subscores:
        spec_key = _find_spec_key(sub.expr)
        if spec_key is not None:
            break
    if spec_key is None:
        for blk in rubric.stage_blocks:
            for sub in blk.subscores:
                spec_key = _find_spec_key(sub.expr)
                if spec_key is not None:
                    spec_truth_prefix = blk.truth_prefix
                    break
            if spec_key is not None:
                break
    if spec_key is None:
        return

    found, raw_ref = resolve_truth_value(ground_truth, spec_key, truth_prefix=spec_truth_prefix)
    if not found or not isinstance(raw_ref, list) or len(raw_ref) != len(spec_rec.value):
        return
    ref_arr = np.asarray(raw_ref, dtype=np.float64)
    meas_arr = np.asarray(spec_rec.value, dtype=np.float64)
    denom = float(np.mean(np.abs(ref_arr)))
    if denom > 1e-12:
        nrmse = float(np.sqrt(np.mean((meas_arr - ref_arr) ** 2)) / denom)
        observables["precal.flux_cal_model_spec_nrmse"] = finite_observable(
            nrmse,
            provenance=spec_rec.provenance,
            missing_status="extractor_error",
            ndigits=6,
        )


def _evaluate_hard_gate(
    gate: HardGateSpec,
    observables: dict[str, ObservableRecord],
    ground_truth: dict[str, Any],
) -> tuple[bool, bool]:
    """Return `(skipped, gate_passed)` for a `HardGateSpec`."""
    if gate.optional_when_truth_missing:
        t_key = gate.params.get("truth_key")
        if t_key is not None:
            f_t, _ = resolve_truth_value(ground_truth, t_key)
            if not f_t:
                return True, True
        rec_chk = observables.get(gate.observable)
        if rec_chk is not None and rec_chk.status == "not_applicable":
            return True, True

    rec = observables.get(gate.observable)
    if gate.optional_when_not_measured and (rec is None or rec.status != "measured"):
        return True, True

    dummy_leaf = LeafRuleSpec(
        observable=gate.observable,
        primitive=gate.primitive,
        params=gate.params,
    )
    sc = _evaluate_leaf(
        dummy_leaf,
        observables,
        ground_truth,
        stage="proc",
        truth_prefix=None,
        named_scores={},
        leaf_traces=[],
    )
    return False, (sc >= 1.0 - 1e-9)


def _build_display_rows(
    rubric: RubricSpec,
    leaf_traces: list[LeafEvalTrace],
    observables: dict[str, ObservableRecord],
    ground_truth: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build deduplicated Truth-vs-Measured rows for `diagnostics.py`."""
    rows: list[dict[str, Any]] = []
    if rubric.display_metrics:
        for dm in rubric.display_metrics:
            assert isinstance(dm, DisplayMetricSpec)
            rec = observables.get(dm.observable)
            m_val = rec.value if (rec is not None and rec.status == "measured") else None
            if (
                m_val is not None
                and isinstance(m_val, (int, float))
                and not isinstance(m_val, bool)
            ):
                m_val = round(float(m_val) * float(dm.scale), int(dm.digits))
            t_disp = dm.target_format or ""
            t_val: Any = None
            if dm.truth_key is not None:
                found, raw_t = resolve_truth_value(ground_truth, dm.truth_key)
                if found:
                    t_val = raw_t
                    if not t_disp:
                        if isinstance(raw_t, (int, float)) and not isinstance(raw_t, bool):
                            t_disp = f"{float(raw_t) * float(dm.scale):.{dm.digits}g}"
                        else:
                            t_disp = str(raw_t)
            rows.append(
                {
                    "id": dm.id,
                    "observable": dm.observable,
                    "stage": dm.stage_block
                    or (rubric.stages[0] if len(rubric.stages) == 1 else "e2e"),
                    "label": dm.label or dm.observable,
                    "unit": dm.unit if dm.unit is not None else (rec.unit if rec else None),
                    "truth_display": t_disp,
                    "truth_value": t_val,
                    "measured_value": m_val,
                    "status": rec.status if rec is not None else "missing_artifact",
                }
            )
        return rows

    seen_keys: set[tuple[str, str]] = set()
    for tr in leaf_traces:
        # Skip internal boolean plumbing gates when richer physical observables exist,
        # except primary stage-validity booleans
        key = (tr.stage, tr.observable)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(
            {
                "id": tr.observable,
                "observable": tr.observable,
                "stage": tr.stage,
                "label": tr.label,
                "unit": tr.unit,
                "truth_display": tr.truth_display,
                "truth_value": tr.truth_value,
                "measured_value": tr.measured_value,
                "status": tr.status,
                "score": tr.score,
            }
        )
    return rows


_STAGE_CLAIM_PREFIXES: dict[str, tuple[str, ...]] = {
    "image": ("image.",),
    "image_cont": ("image.",),
    "calibrate": ("cal.",),
    "precal": ("precal.",),
    "polcal": ("polcal.",),
    "postcal": ("postcal.",),
    "selfcal": ("selfcal.",),
    "image_cube": ("cube.",),
    "simulate": ("sim.",),
}


def _has_hedged_downstream_artifacts_or_claims(
    terminal_stage: str,
    observables: dict[str, ObservableRecord],
    claimed_keys: set[str],
    required_reported_metrics: list[str],
) -> tuple[bool, str | None]:
    """Detect hedged downstream deliverable fabrication on an `expect_halt=True` task."""
    st = terminal_stage.strip().lower()
    if st in ("image", "image_cont"):
        vp = observables.get("image.valid_product_count")
        if vp is not None and vp.status == "measured" and int(vp.value or 0) > 0:
            return True, "Downstream CASA image products were fabricated despite expect_halt=True"
    elif st == "calibrate":
        cd = observables.get("cal.corrected_distinct_from_data")
        if cd is not None and cd.status == "measured" and cd.value is True:
            return True, "Downstream CORRECTED_DATA was fabricated despite expect_halt=True"
    elif st == "precal":
        nr = observables.get("precal.calibrators_ms_nrows")
        if nr is not None and nr.status == "measured" and int(nr.value or 0) > 0:
            return True, "Downstream calibrators.ms was fabricated despite expect_halt=True"

    allowed_req = set(required_reported_metrics)
    forbid_prefixes = _STAGE_CLAIM_PREFIXES.get(st, ())
    if forbid_prefixes:
        for ck in sorted(claimed_keys):
            if ck not in allowed_req and ck.startswith(forbid_prefixes):
                return (
                    True,
                    f"Downstream product claim {ck!r} reported despite expect_halt=True on stage {st!r}",
                )
    return False, None


def evaluate_rubric(
    workspace_dir: Path,
    extraction: WorkspaceExtractionResult,
    rubric: RubricSpec,
    ground_truth: dict[str, Any],
) -> RubricEvaluationResult:
    """Evaluate `rubric` against `extraction` and `ground_truth`."""
    ws = Path(workspace_dir)
    obs: dict[str, ObservableRecord] = {
        k: v.model_copy(deep=True) for k, v in extraction.observables.items()
    }
    errors: list[str] = list(extraction.errors)
    triggered_tags: list[str] = []

    def _add_tag(tag: str) -> None:
        if tag and tag not in triggered_tags:
            triggered_tags.append(tag)

    # 1. Runner-derived cross-cutting observables
    raw_err = _derive_raw_data_unmodified_observable(obs, ground_truth)
    if raw_err:
        errors.append(raw_err)
    _derive_cross_role_confusion_observable(obs, ground_truth)
    _derive_spec_nrmse_if_needed(obs, ground_truth, rubric)

    claim_obs, claim_errs, claimed_keys = audit_claims(
        ws,
        rubric,
        extraction.manifest,
        obs,
        raw_manifest_dict=extraction.raw_manifest_dict,
    )
    obs.update(claim_obs)
    errors.extend(claim_errs)

    table_locked = bool(
        obs.get("proc.table_locked") is not None and obs["proc.table_locked"].value is True
    )

    # 2. Layer B Halt / Abstention & Manifest Schema Consistency
    manifest = extraction.manifest
    raw_manifest = extraction.raw_manifest_dict
    raw_status = (
        manifest.status
        if manifest is not None
        else (
            str(raw_manifest.get("status")).strip().lower()
            if isinstance(raw_manifest, dict) and isinstance(raw_manifest.get("status"), str)
            else None
        )
    )

    manifest_schema_rec = obs.get("proc.manifest_schema_valid")
    manifest_schema_invalid = bool(
        (ws / "stage_output.json").is_file()
        and manifest_schema_rec is not None
        and manifest_schema_rec.status == "measured"
        and manifest_schema_rec.value is False
    )
    if manifest_schema_invalid:
        _add_tag("unsupported_or_missing_claim")

    invalid_halt_triggered = False
    manifest_failed_triggered = False

    if rubric.expect_halt:
        unsup_cnt = int(
            (
                obs.get("proc.unsupported_claim_count")
                or ObservableRecord(value=0, status="not_applicable")
            ).value
            or 0
        )
        miss_cnt = int(
            (
                obs.get("proc.missing_required_claim_count")
                or ObservableRecord(value=0, status="not_applicable")
            ).value
            or 0
        )
        halt_code_ok = (
            manifest is not None
            and manifest.halt_code is not None
            and (not rubric.expected_halt_codes or manifest.halt_code in rubric.expected_halt_codes)
        )
        terminal_stage = rubric.stages[-1] if rubric.stages else ""
        hedged_bad, hedged_msg = _has_hedged_downstream_artifacts_or_claims(
            terminal_stage,
            obs,
            claimed_keys,
            rubric.required_reported_metrics,
        )
        if hedged_bad and hedged_msg:
            errors.append(hedged_msg)
        abstention_ok = bool(
            manifest is not None
            and not manifest_schema_invalid
            and manifest.status == "halted"
            and halt_code_ok
            and bool(manifest.halt_reason and manifest.halt_reason.strip())
            and not table_locked
            and not hedged_bad
            and unsup_cnt == 0
            and miss_cnt == 0
        )
        obs["proc.abstention_correct"] = ObservableRecord(
            value=abstention_ok,
            status="measured",
            provenance="stage_output.json",
        )
        if not abstention_ok:
            _add_tag("invalid_abstention_or_halt")
            errors.append(
                "Task requires valid physical halt (expect_halt=True), but workspace did not halt cleanly"
            )
    else:
        obs["proc.abstention_correct"] = ObservableRecord(
            value=None,
            status="not_applicable",
            provenance="stage_output.json",
        )
        if raw_status == "halted":
            invalid_halt_triggered = True
            _add_tag("invalid_abstention_or_halt")
            errors.append("Agent halted (status='halted') on a solvable task (expect_halt=False)")
        elif raw_status == "failed":
            manifest_failed_triggered = True
            _add_tag("script_or_casa_crash")
            errors.append("Agent manifest reported status='failed'")

    # Check raw_data_unmodified explicitly if measured False
    raw_unmod_rec = obs.get("proc.raw_data_unmodified")
    if (
        raw_unmod_rec is not None
        and raw_unmod_rec.status == "measured"
        and raw_unmod_rec.value is False
    ):
        _add_tag("raw_data_or_calflag_mutation")

    # Check Claim Auditor failures whenever claims are audited or required
    unsup_rec = obs.get("proc.unsupported_claim_count")
    miss_rec = obs.get("proc.missing_required_claim_count")
    claim_audit_failed = bool(
        (unsup_rec is not None and unsup_rec.status == "measured" and int(unsup_rec.value or 0) > 0)
        or (
            miss_rec is not None
            and miss_rec.status == "measured"
            and rubric.required_reported_metrics
            and int(miss_rec.value or 0) > 0
        )
    )
    if claim_audit_failed:
        _add_tag("unsupported_or_missing_claim")

    # 3. Evaluate `hard_gates`
    hard_gate_meta: list[dict[str, Any]] = []
    hard_veto_failed = False
    zero_reward_veto = False
    zero_subscores_veto = False

    for hg in rubric.hard_gates:
        skipped, hg_ok = _evaluate_hard_gate(hg, obs, ground_truth)
        hard_gate_meta.append(
            {
                "id": hg.id,
                "observable": hg.observable,
                "skipped": skipped,
                "passed": hg_ok,
                "zero_reward_on_fail": hg.zero_reward_on_fail,
                "failure_tag": hg.failure_tag,
            }
        )
        if not skipped and not hg_ok:
            hard_veto_failed = True
            if hg.zero_reward_on_fail:
                zero_reward_veto = True
            if hg.zero_subscores_on_fail:
                zero_subscores_veto = True
            _add_tag(hg.failure_tag)

    if table_locked:
        hard_veto_failed = True
        zero_reward_veto = True
        zero_subscores_veto = True
        _add_tag("lock_or_timeout")

    # 4. Evaluate Subscores & Multi-Stage Blocks
    named_scores: dict[str, float] = {}
    leaf_traces: list[LeafEvalTrace] = []
    subscores_out: dict[str, float] = {}
    subscore_meta_list: list[dict[str, Any]] = []
    stage_block_meta_list: list[dict[str, Any]] = []

    if rubric.expect_halt:
        ab_ok = bool(obs["proc.abstention_correct"].value is True)
        ab_score = 1.0 if ab_ok else 0.0
        subscores_out["abstention_score"] = ab_score
        named_scores["abstention_score"] = ab_score
        subscore_meta_list.append(
            {
                "id": "abstention_score",
                "stage": rubric.stages[-1],
                "stage_block_id": None,
                "weight": 1.0,
                "score": ab_score,
                "failure_tag": "invalid_abstention_or_halt",
                "symptom_only": False,
                "label": "Valid Physical Halt / Abstention",
            }
        )
        composite = ab_score
    elif rubric.stage_blocks:
        total_block_w = sum(b.weight for b in rubric.stage_blocks)
        composite_acc = 0.0
        for blk in rubric.stage_blocks:
            assert isinstance(blk, StageBlockSpec)
            blk_w_sum = sum(s.weight for s in blk.subscores)
            blk_acc = 0.0
            for sub in blk.subscores:
                assert isinstance(sub, SubscoreSpec)
                if zero_subscores_veto:
                    raw_s = 0.0
                else:
                    raw_s = evaluate_rule_expr(
                        sub.expr,
                        obs,
                        ground_truth,
                        stage=blk.stage,
                        truth_prefix=blk.truth_prefix,
                        named_scores=named_scores,
                        leaf_traces=leaf_traces,
                        subscore_prefix=blk.subscore_prefix,
                    )
                s_rounded = round(raw_s, 6)
                out_key = f"{blk.subscore_prefix}{sub.id}"
                named_scores[sub.id] = raw_s
                named_scores[out_key] = raw_s
                blk_acc += sub.weight * raw_s
                subscores_out[out_key] = s_rounded
                subscore_meta_list.append(
                    {
                        "id": out_key,
                        "local_id": sub.id,
                        "stage": blk.stage,
                        "stage_block_id": blk.id,
                        "weight": round(blk.weight * (sub.weight / max(blk_w_sum, 1e-12)), 6),
                        "local_weight": sub.weight,
                        "score": s_rounded,
                        "failure_tag": sub.failure_tag,
                        "symptom_only": sub.symptom_only,
                        "label": sub.label or out_key,
                    }
                )
            blk_norm = blk_acc / blk_w_sum if blk_w_sum > 0.0 else 0.0
            blk_score = 0.0 if zero_subscores_veto else round(min(1.0, max(0.0, blk_norm)), 6)
            named_scores[blk.id] = blk_score
            stage_block_meta_list.append(
                {
                    "id": blk.id,
                    "stage": blk.stage,
                    "weight": blk.weight,
                    "score": blk_score,
                    "failure_tag": blk.failure_tag,
                }
            )
            composite_acc += blk.weight * blk_score

        # Preserve exact v0.1 e2e key order: block scores first, then prefixed subscores
        ordered_e2e_subscores: dict[str, float] = {}
        for bm in stage_block_meta_list:
            ordered_e2e_subscores[str(bm["id"])] = float(bm["score"])
        for k, v in subscores_out.items():
            ordered_e2e_subscores[k] = v
        subscores_out = ordered_e2e_subscores
        composite = composite_acc / total_block_w if total_block_w > 0.0 else 0.0
    else:
        pos_w_sum = sum(s.weight for s in rubric.subscores)
        composite_acc = 0.0
        stage_name = rubric.stages[0]
        for sub in rubric.subscores:
            if zero_subscores_veto:
                raw_s = 0.0
            else:
                raw_s = evaluate_rule_expr(
                    sub.expr,
                    obs,
                    ground_truth,
                    stage=stage_name,
                    truth_prefix=None,
                    named_scores=named_scores,
                    leaf_traces=leaf_traces,
                )
            s_rounded = round(raw_s, 6)
            named_scores[sub.id] = raw_s
            if sub.weight > 0.0 or len(rubric.subscores) == 1:
                subscores_out[sub.id] = s_rounded
            composite_acc += sub.weight * raw_s
            subscore_meta_list.append(
                {
                    "id": sub.id,
                    "stage": stage_name,
                    "stage_block_id": None,
                    "weight": sub.weight,
                    "score": s_rounded,
                    "failure_tag": sub.failure_tag,
                    "symptom_only": sub.symptom_only,
                    "label": sub.label or sub.id,
                }
            )
        composite = composite_acc / pos_w_sum if pos_w_sum > 0.0 else 0.0

    # 5. Check `core_subscore_floors` (including named child nodes such as `phase_part` & `target_part`)
    floors_ok = True
    floor_results: dict[str, dict[str, Any]] = {}
    if not rubric.expect_halt:
        for node_id, min_floor in rubric.core_subscore_floors.items():
            actual = named_scores.get(node_id, 0.0)
            ok_f = actual >= float(min_floor) - 1e-9
            floor_results[node_id] = {
                "min_floor": float(min_floor),
                "actual": round(float(actual), 6),
                "passed": ok_f,
            }
            if not ok_f:
                floors_ok = False

        for blk in rubric.stage_blocks:
            for fk, fv in blk.core_subscore_floors.items():
                pref_key = f"{blk.subscore_prefix}{fk}" if blk.subscore_prefix else fk
                eff_key = pref_key if pref_key in named_scores else fk
                actual = named_scores.get(eff_key, 0.0)
                ok_f = actual >= float(fv) - 1e-9
                floor_results[pref_key] = {
                    "min_floor": float(fv),
                    "actual": round(float(actual), 6),
                    "passed": ok_f,
                }
                if not ok_f:
                    floors_ok = False

    # 6. Finalize `reward`, `passed`, and physics failure tags in DAG & root-cause order
    if (
        zero_reward_veto
        or invalid_halt_triggered
        or manifest_failed_triggered
        or (rubric.expect_halt and obs["proc.abstention_correct"].value is not True)
    ):
        reward = 0.0
    else:
        reward = round(min(1.0, max(0.0, composite)), 6)

    pass_min = float(rubric.pass_reward_min)
    passed = (
        1
        if (
            reward >= pass_min
            and floors_ok
            and not hard_veto_failed
            and not invalid_halt_triggered
            and not manifest_failed_triggered
            and not manifest_schema_invalid
            and not claim_audit_failed
            and (not rubric.expect_halt or obs["proc.abstention_correct"].value is True)
        )
        else 0
    )

    if passed == 0:
        # Collect physics failure tags in stage-block DAG order:
        # within each stage, evaluate root-cause (`symptom_only == False`) subscores first,
        # sorted by weighted deficit `weight * (1 - score)` descending, before `symptom_only`.
        stages_in_order: list[str] = []
        for sm in subscore_meta_list:
            st = str(sm["stage"])
            if st not in stages_in_order:
                stages_in_order.append(st)

        for st in stages_in_order:
            st_subs = [
                sm
                for sm in subscore_meta_list
                if sm["stage"] == st and float(sm["weight"]) > 0.0 and float(sm["score"]) < 0.85
            ]
            root_subs = sorted(
                [sm for sm in st_subs if not sm.get("symptom_only")],
                key=lambda x: float(x["weight"]) * (1.0 - float(x["score"])),
                reverse=True,
            )
            symp_subs = sorted(
                [sm for sm in st_subs if sm.get("symptom_only")],
                key=lambda x: float(x["weight"]) * (1.0 - float(x["score"])),
                reverse=True,
            )
            for sm in (*root_subs, *symp_subs):
                _add_tag(str(sm["failure_tag"]))

    reward_dict: dict[str, float | int] = {
        "reward": reward,
        "passed": int(passed),
        **subscores_out,
    }

    display_rows = _build_display_rows(rubric, leaf_traces, obs, ground_truth)
    rubric_meta: dict[str, Any] = {
        "schema_version": "0.2",
        "task_id": rubric.task_id,
        "stages": list(rubric.stages),
        "pass_reward_min": pass_min,
        "expect_halt": rubric.expect_halt,
        "required_reported_metrics": list(rubric.required_reported_metrics),
        "hard_gates": hard_gate_meta,
        "core_subscore_floors": floor_results,
        "stage_blocks": stage_block_meta_list,
        "subscores": subscore_meta_list,
        "named_scores": {k: round(v, 6) for k, v in named_scores.items()},
        "display_rows": display_rows,
    }

    return RubricEvaluationResult(
        reward=reward,
        passed=int(passed),
        reward_dict=reward_dict,
        subscores=subscores_out,
        observables=obs,
        rubric_meta=rubric_meta,
        triggered_failure_tags=triggered_tags,
        errors=errors,
    )
