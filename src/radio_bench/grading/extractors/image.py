"""Pure metric extractor for `image` (`image_cont`) stage (zero dataset branches)."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.grading.extractors.common import (
    check_table_multiused,
    has_active_table_lock,
    natural_thermal_noise_jy,
    parse_sky_coord_rad,
    resolve_ant_ids_by_names,
    resolve_field_ids_by_names,
    target_parallel_hand_retention,
    thermal_floor_ratio,
    verify_pb_taper,
)
from radio_bench.grading.schema import ExtractConfig, ObservableRecord, finite_observable

_SKIP_DIR_SUFFIXES: tuple[str, ...] = (
    ".ms",
    ".cal",
    ".g0",
    ".k",
    ".b",
    ".g",
    ".flux",
    ".cl",
    ".flagversions",
    ".psf",
    ".sumwt",
    ".weight",
    ".mask",
    ".pb",
    ".residual",
    ".model",
    ".pbcor",
    ".tt1",
    ".tt2",
    ".alpha",
)


@dataclass
class DiscoveredImageSet:
    """Discovered CASA continuum image product bundle for one stem."""

    prefix: str
    image_path: Path
    pb_path: Path | None
    residual_path: Path | None
    model_path: Path | None
    pbcor_path: Path | None
    has_nonzero_model: int
    mtime: float

    @property
    def n_present(self) -> int:
        return sum(
            1
            for p in (
                self.image_path,
                self.pb_path,
                self.residual_path,
                self.model_path,
                self.pbcor_path,
            )
            if p is not None
        )


def _is_casa_table_dir(p: Path) -> bool:
    return p.is_dir() and (p / "table.dat").is_file()


def _find_first_table(candidates: list[Path]) -> Path | None:
    for cand in candidates:
        if _is_casa_table_dir(cand):
            return cand
    return None


def _probe_nonzero_model(model_path: Path | None) -> int:
    """Return `1` if `model_path` opens cleanly and contains any non-zero clean component."""
    if model_path is None or has_active_table_lock(model_path):
        return 0
    import casatools

    ia = casatools.image()
    opened = False
    try:
        opened = bool(ia.open(str(model_path)))
        if not opened:
            return 0
        arr = np.asarray(ia.getchunk(), dtype=np.float64)
        return 1 if bool(np.any(np.isfinite(arr) & (np.abs(arr) > 1e-12))) else 0
    except Exception:
        return 0
    finally:
        if opened:
            with contextlib.suppress(Exception):
                ia.close()
        with contextlib.suppress(Exception):
            ia.done()


def discover_image_sets(workspace_dir: Path, *, max_depth: int = 3) -> list[DiscoveredImageSet]:
    """Discover CASA continuum image sets (`.image` or `.image.tt0`, excluding `*pbcor*` stems).

    Two-pass `.pbcor` matching prevents an incomplete stem (e.g. `dirty_test.image`) from
    stealing another stem's matched `final.image.pbcor`, and ranks candidates by
    `(n_present, has_nonzero_model, mtime)`.
    """
    restored_candidates: list[tuple[Path, bool]] = []

    def _scan(cur_dir: Path, depth: int) -> None:
        if depth > max_depth or not cur_dir.is_dir():
            return
        try:
            entries = sorted(cur_dir.iterdir(), key=lambda x: x.name)
        except OSError:
            return
        for entry in entries:
            name = entry.name
            low = name.lower()
            if name.startswith(".") or name.startswith("TempLattice"):
                continue
            if not entry.is_dir():
                continue
            if low in ("input.ms", "calibrators.ms") or low.endswith(".ms"):
                continue
            if low.endswith(".image.tt0") and _is_casa_table_dir(entry):
                stem_low = low[: -len(".image.tt0")]
                if "pbcor" not in stem_low and stem_low:
                    restored_candidates.append((entry, True))
                continue
            if low.endswith(".image") and _is_casa_table_dir(entry):
                stem_low = low[: -len(".image")]
                if "pbcor" not in stem_low and stem_low:
                    restored_candidates.append((entry, False))
                continue
            if any(low.endswith(sfx) for sfx in _SKIP_DIR_SUFFIXES) or _is_casa_table_dir(entry):
                continue
            _scan(entry, depth + 1)

    _scan(workspace_dir, 0)
    if not restored_candidates:
        return []

    # Pass 1: match stem-explicit companion products and record claimed `.pbcor` paths
    prelim: list[dict[str, Any]] = []
    claimed_pbcors: set[Path] = set()

    for img_dir, is_tt0 in restored_candidates:
        parent = img_dir.parent
        if is_tt0:
            stem = img_dir.name[: -len(".image.tt0")]
            pb = _find_first_table([parent / f"{stem}.pb.tt0", parent / f"{stem}.pb"])
            res = _find_first_table([parent / f"{stem}.residual.tt0", parent / f"{stem}.residual"])
            mod = _find_first_table([parent / f"{stem}.model.tt0", parent / f"{stem}.model"])
            pbcor = _find_first_table(
                [
                    parent / f"{stem}.image.tt0.pbcor",
                    parent / f"{stem}.image.pbcor.tt0",
                    parent / f"{stem}.pbcor.tt0",
                    parent / f"{stem}.image.pbcor",
                    parent / f"{stem}.pbcor",
                ]
            )
        else:
            stem = img_dir.name[: -len(".image")]
            pb = _find_first_table([parent / f"{stem}.pb", parent / f"{stem}.pb.tt0"])
            res = _find_first_table([parent / f"{stem}.residual", parent / f"{stem}.residual.tt0"])
            mod = _find_first_table([parent / f"{stem}.model", parent / f"{stem}.model.tt0"])
            pbcor = _find_first_table(
                [
                    parent / f"{stem}.image.pbcor",
                    parent / f"{stem}.pbcor",
                    parent / f"{stem}.image.pbcor.tt0",
                    parent / f"{stem}.image.tt0.pbcor",
                ]
            )

        if pbcor is not None:
            claimed_pbcors.add(pbcor.resolve())

        try:
            mtime = float((img_dir / "table.dat").stat().st_mtime)
        except OSError:
            mtime = 0.0

        prelim.append(
            {
                "parent": parent,
                "stem": stem,
                "image_path": img_dir,
                "pb_path": pb,
                "residual_path": res,
                "model_path": mod,
                "pbcor_path": pbcor,
                "mtime": mtime,
            }
        )

    # Pass 2: allow custom-named `.pbcor` fallback ONLY when unclaimed and unambiguous in `parent`
    missing_pbcor_per_parent: dict[Path, int] = {}
    for item in prelim:
        if item["pbcor_path"] is None:
            p_key = item["parent"].resolve()
            missing_pbcor_per_parent[p_key] = missing_pbcor_per_parent.get(p_key, 0) + 1

    sets: list[DiscoveredImageSet] = []
    for item in prelim:
        parent = item["parent"]
        pbcor = item["pbcor_path"]
        if pbcor is None and missing_pbcor_per_parent.get(parent.resolve(), 0) == 1:
            try:
                unclaimed = [
                    p
                    for p in sorted(parent.iterdir(), key=lambda x: x.name)
                    if p.name.lower().endswith(".pbcor")
                    and _is_casa_table_dir(p)
                    and p.resolve() not in claimed_pbcors
                ]
            except OSError:
                unclaimed = []
            if len(unclaimed) == 1:
                pbcor = unclaimed[0]
                claimed_pbcors.add(pbcor.resolve())

        has_nz_mod = _probe_nonzero_model(item["model_path"])
        sets.append(
            DiscoveredImageSet(
                prefix=str(parent / item["stem"]),
                image_path=item["image_path"],
                pb_path=item["pb_path"],
                residual_path=item["residual_path"],
                model_path=item["model_path"],
                pbcor_path=pbcor,
                has_nonzero_model=has_nz_mod,
                mtime=float(item["mtime"]),
            )
        )

    sets.sort(key=lambda s: (s.n_present, s.has_nonzero_model, s.mtime), reverse=True)
    return sets


def read_image_2d(
    img_path: Path,
    *,
    need_header: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Read 2D Stokes-I continuum image plane, boolean mask, and optional WCS/beam header."""
    import casatools

    ia = casatools.image()
    qa = casatools.quanta() if need_header else None
    opened = False
    try:
        opened = bool(ia.open(str(img_path)))
        if not opened:
            raise RuntimeError(f"casatools.image.open returned False for {img_path}")
        raw_arr = ia.getchunk()
        raw_msk = ia.getchunk(getmask=True)
        while raw_arr.ndim > 2:
            raw_arr = raw_arr[..., 0]
            raw_msk = raw_msk[..., 0]
        arr = raw_arr.astype(np.float64).copy()
        msk = raw_msk.astype(bool).copy()

        hdr: dict[str, Any] | None = None
        if need_header and qa is not None:
            rb = ia.restoringbeam()
            if isinstance(rb, dict) and "beams" in rb:
                beams = rb["beams"]
                first_c = "*0" if "*0" in beams else next(iter(beams))
                sub_b = beams[first_c]
                first_p = "*0" if "*0" in sub_b else next(iter(sub_b))
                rb = sub_b[first_p]
            if not isinstance(rb, dict) or "major" not in rb or "minor" not in rb:
                raise RuntimeError(f"Missing restoring beam in {img_path}")
            b_maj = float(qa.convert(rb["major"], "arcsec")["value"])
            b_min = float(qa.convert(rb["minor"], "arcsec")["value"])
            pa_key = "positionangle" if "positionangle" in rb else "pa"
            b_pa = float(qa.convert(rb[pa_key], "deg")["value"]) if pa_key in rb else 0.0

            cs = ia.coordsys()
            try:
                units = [str(u).strip() for u in cs.units()]
                refpix = [float(x) for x in cs.referencepixel()["numeric"]]
                raw_refval = [float(x) for x in cs.referencevalue()["numeric"]]
                raw_inc = [float(x) for x in cs.increment()["numeric"]]
            finally:
                with contextlib.suppress(Exception):
                    cs.done()

            u0 = units[0] if len(units) > 0 and units[0] else "rad"
            u1 = units[1] if len(units) > 1 and units[1] else "rad"
            rv0_rad = float(qa.convert({"value": raw_refval[0], "unit": u0}, "rad")["value"])
            rv1_rad = float(qa.convert({"value": raw_refval[1], "unit": u1}, "rad")["value"])
            inc0_rad = float(qa.convert({"value": raw_inc[0], "unit": u0}, "rad")["value"])
            inc1_rad = float(qa.convert({"value": raw_inc[1], "unit": u1}, "rad")["value"])
            if abs(inc0_rad) < 1e-18 or abs(inc1_rad) < 1e-18:
                raise RuntimeError(f"Degenerate zero WCS increment in {img_path}")

            freq_hz: float | None = None
            for ax_idx in range(2, min(len(units), len(raw_refval))):
                u_ax = units[ax_idx]
                if "hz" in u_ax.lower():
                    freq_hz = float(
                        qa.convert({"value": raw_refval[ax_idx], "unit": u_ax}, "Hz")["value"]
                    )
                    break

            refval = [rv0_rad, rv1_rad, *raw_refval[2:]]
            inc = [inc0_rad, inc1_rad, *raw_inc[2:]]

            cell_x_arcsec = abs(math.degrees(inc0_rad) * 3600.0)
            cell_y_arcsec = abs(math.degrees(inc1_rad) * 3600.0)
            cell_arcsec = math.sqrt(max(cell_x_arcsec * cell_y_arcsec, 1e-18))
            hdr = {
                "b_major_arcsec": b_maj,
                "b_minor_arcsec": b_min,
                "b_pa_deg": b_pa,
                "cell_x_arcsec": cell_x_arcsec,
                "cell_y_arcsec": cell_y_arcsec,
                "cell_arcsec": cell_arcsec,
                "refpix": refpix,
                "refval": refval,
                "inc": inc,
                "freq_hz": freq_hz,
            }
        return arr, msk, hdr
    finally:
        if opened:
            with contextlib.suppress(Exception):
                ia.close()
        with contextlib.suppress(Exception):
            ia.done()
        if qa is not None:
            with contextlib.suppress(Exception):
                qa.done()


def sky_to_pix(hdr: dict[str, Any], ra_rad: float, dec_rad: float) -> tuple[float, float]:
    """Project `(ra_rad, dec_rad)` to fractional pixel coordinates `(px, py)` using SIN WCS."""
    ra0 = float(hdr["refval"][0])
    dec0 = float(hdr["refval"][1])
    inc_x = float(hdr["inc"][0])
    inc_y = float(hdr["inc"][1])
    if abs(inc_x) < 1e-18 or abs(inc_y) < 1e-18:
        raise ValueError("Degenerate zero WCS increment in image header")
    l_rad = math.cos(dec_rad) * math.sin(ra_rad - ra0)
    m_rad = math.sin(dec_rad) * math.cos(dec0) - math.cos(dec_rad) * math.sin(dec0) * math.cos(
        ra_rad - ra0
    )
    px = float(hdr["refpix"][0] + l_rad / inc_x)
    py = float(hdr["refpix"][1] + m_rad / inc_y)
    return px, py


def _measure_convolved_sky_ncc(
    pbcor_arr: np.ndarray,
    valid_mask: np.ndarray,
    ref_arrays_dir: Path | None,
) -> float | None:
    """Compute normalized cross-correlation against `true_sky.npz` when staged."""
    if ref_arrays_dir is None:
        return None
    cand = (
        ref_arrays_dir
        if (ref_arrays_dir.is_file() and ref_arrays_dir.name == "true_sky.npz")
        else (ref_arrays_dir / "true_sky.npz" if ref_arrays_dir.is_dir() else None)
    )
    if cand is None or not cand.is_file():
        return None
    try:
        with np.load(cand, allow_pickle=False) as z:
            key = (
                "convolved_sky_jy_per_beam"
                if "convolved_sky_jy_per_beam" in z.files
                else "true_sky"
            )
            if key not in z.files:
                return None
            ref_img = np.asarray(z[key], dtype=np.float64)
        if ref_img.shape != pbcor_arr.shape:
            return None
        sel = valid_mask & np.isfinite(pbcor_arr) & np.isfinite(ref_img)
        if int(np.sum(sel)) < 20:
            return None
        a = pbcor_arr[sel] - float(np.mean(pbcor_arr[sel]))
        b = ref_img[sel] - float(np.mean(ref_img[sel]))
        da = float(np.sum(a**2))
        db = float(np.sum(b**2))
        if da <= 1e-24 or db <= 1e-24:
            return None
        return float(np.sum(a * b) / math.sqrt(da * db))
    except Exception:
        return None


def extract_image(
    workspace_dir: Path,
    extract_config: ExtractConfig,
    ref_arrays_dir: Path | None = None,
    *,
    table_locked: bool = False,
) -> tuple[dict[str, ObservableRecord], list[str]]:
    """Extract raw continuum image observables from `/workspace` (zero dataset branches)."""
    from casatasks import clearstat

    with contextlib.suppress(Exception):
        clearstat()

    ws = Path(workspace_dir)
    geom = extract_config.geometry
    errors: list[str] = []
    obs: dict[str, ObservableRecord] = {}

    all_image_keys = (
        ("image.products_complete", None),
        ("image.valid_product_count", None),
        ("image.shapes_and_wcs_valid", None),
        ("image.pb_taper_consistent", None),
        ("image.pixels_per_beam", None),
        ("image.cell_arcsec", "arcsec"),
        ("image.b_major_arcsec", "arcsec"),
        ("image.b_minor_arcsec", "arcsec"),
        ("image.b_pa_deg", "deg"),
        ("image.offsource_residual_rms_jy", "Jy/beam"),
        ("image.offsource_residual_mad_std_jy", "Jy/beam"),
        ("image.offsource_residual_mad_over_rms", None),
        ("image.thermal_noise_natural_jy", "Jy/beam"),
        ("image.offsource_rms_over_thermal_floor", None),
        ("image.target_row_retention", None),
        ("image.peak_jy_per_beam", "Jy/beam"),
        ("image.dynamic_range", None),
        ("image.source_integrated_flux_jy", "Jy"),
        ("image.point_source_snr", None),
        ("image.negative_bowl_ratio", None),
        ("image.convolved_sky_ncc", None),
    )

    if table_locked:
        for key, unit in all_image_keys:
            obs[key] = ObservableRecord(
                value=None,
                status="locked_or_unreadable",
                unit=unit,
                provenance="workspace",
            )
        return obs, errors

    candidates = discover_image_sets(ws)
    if not candidates:
        errors.append("No CASA .image or .image.tt0 products found in /workspace")
        for key, unit in all_image_keys:
            default_val: Any = None
            if key in (
                "image.products_complete",
                "image.shapes_and_wcs_valid",
                "image.pb_taper_consistent",
            ):
                default_val = False
            elif key == "image.valid_product_count":
                default_val = 0
            obs[key] = ObservableRecord(
                value=default_val,
                status="measured" if default_val is not None else "missing_artifact",
                unit=unit,
                provenance="workspace",
            )
        return obs, errors

    chosen = candidates[0]
    prov = Path(chosen.prefix).name
    for p in (
        chosen.image_path,
        chosen.pb_path,
        chosen.residual_path,
        chosen.model_path,
        chosen.pbcor_path,
    ):
        if p is not None and check_table_multiused(p):
            errors.append(f"Active casacore write lock on {p}")
            for key, unit in all_image_keys:
                obs[key] = ObservableRecord(
                    value=None,
                    status="locked_or_unreadable",
                    unit=unit,
                    provenance=prov,
                )
            return obs, errors

    img_arr: np.ndarray | None = None
    img_msk: np.ndarray | None = None
    hdr: dict[str, Any] | None = None
    pb_arr: np.ndarray | None = None
    pb_msk: np.ndarray | None = None
    res_arr: np.ndarray | None = None
    res_msk: np.ndarray | None = None
    mod_arr: np.ndarray | None = None
    pbcor_arr: np.ndarray | None = None
    pbcor_msk: np.ndarray | None = None

    try:
        img_arr, img_msk, hdr = read_image_2d(chosen.image_path, need_header=True)
    except Exception as exc:
        errors.append(f"Failed reading restored image {chosen.image_path.name}: {exc}")

    if chosen.pb_path is not None:
        try:
            pb_arr, pb_msk, _ = read_image_2d(chosen.pb_path)
        except Exception as exc:
            errors.append(f"Failed reading pb {chosen.pb_path.name}: {exc}")
    else:
        errors.append("Missing .pb image product")

    if chosen.residual_path is not None:
        try:
            res_arr, res_msk, _ = read_image_2d(chosen.residual_path)
        except Exception as exc:
            errors.append(f"Failed reading residual {chosen.residual_path.name}: {exc}")
    else:
        errors.append("Missing .residual image product")

    if chosen.model_path is not None:
        try:
            mod_arr, _mod_msk, _ = read_image_2d(chosen.model_path)
        except Exception as exc:
            errors.append(f"Failed reading model {chosen.model_path.name}: {exc}")
    else:
        errors.append("Missing .model image product")

    if chosen.pbcor_path is not None:
        try:
            pbcor_arr, pbcor_msk, _ = read_image_2d(chosen.pbcor_path)
        except Exception as exc:
            errors.append(f"Failed reading pbcor {chosen.pbcor_path.name}: {exc}")
    else:
        errors.append("Missing .image.pbcor / .pbcor image product")

    if img_arr is None or hdr is None:
        for key, unit in all_image_keys:
            default_val = None
            if key in (
                "image.products_complete",
                "image.shapes_and_wcs_valid",
                "image.pb_taper_consistent",
            ):
                default_val = False
            elif key == "image.valid_product_count":
                default_val = 0
            obs[key] = ObservableRecord(
                value=default_val,
                status="measured" if default_val is not None else "extractor_error",
                unit=unit,
                provenance=prov,
            )
        return obs, errors

    shape_2d = img_arr.shape
    shapes_match = all(
        a is None or a.shape == shape_2d for a in (pb_arr, res_arr, mod_arr, pbcor_arr)
    )
    if not shapes_match:
        errors.append("Image product spatial shapes do not match restored image")

    pb_cut = float(geom.pb_mask_min)
    if pb_arr is not None and pb_arr.shape == shape_2d:
        pb_gate = np.isfinite(pb_arr) & (pb_arr >= pb_cut)
        if pb_msk is not None and pb_msk.shape == shape_2d:
            pb_gate = pb_gate & pb_msk
    elif pbcor_arr is not None and pbcor_arr.shape == shape_2d:
        with np.errstate(divide="ignore", invalid="ignore"):
            pb_est = np.where(np.abs(pbcor_arr) > 1e-12, img_arr / pbcor_arr, 0.0)
        pb_gate = np.isfinite(pb_est) & (pb_est >= pb_cut)
    else:
        pb_gate = np.zeros(shape_2d, dtype=bool)

    valid_pb20 = pb_gate & np.isfinite(img_arr)
    if img_msk is not None:
        valid_pb20 = valid_pb20 & img_msk
    if pbcor_arr is not None and pbcor_arr.shape == shape_2d:
        valid_pb20 = valid_pb20 & np.isfinite(pbcor_arr)
        if pbcor_msk is not None:
            valid_pb20 = valid_pb20 & pbcor_msk
    if res_arr is not None and res_arr.shape == shape_2d:
        valid_pb20 = valid_pb20 & np.isfinite(res_arr)
        if res_msk is not None:
            valid_pb20 = valid_pb20 & res_msk

    valid_pb20_count = int(np.sum(valid_pb20))
    shapes_and_wcs_valid = bool(
        shapes_match
        and valid_pb20_count >= 100
        and hdr["b_major_arcsec"] > 0.0
        and hdr["b_minor_arcsec"] > 0.0
        and hdr["cell_arcsec"] > 0.0
    )

    pb_taper_valid = False
    pb_stats: dict[str, float] = {}
    if pb_arr is not None and pb_arr.shape == shape_2d and int(np.sum(pb_gate)) >= 100:
        pb_taper_valid, pb_stats = verify_pb_taper(
            pb_arr,
            pb_gate,
            inc_x_rad=float(hdr["inc"][0]),
            inc_y_rad=float(hdr["inc"][1]),
            freq_hz=hdr.get("freq_hz"),
            dish_diameter_m=float(geom.dish_diameter_m),
        )
        if not pb_taper_valid:
            errors.append(
                "Unphysical .pb taper "
                f"(max={pb_stats.get('pb_max_pb20')}, min_pb20={pb_stats.get('pb_min_pb20')}, "
                f"min_full={pb_stats.get('pb_min_full')}, std={pb_stats.get('pb_std_pb20')})"
            )
    else:
        errors.append("Missing or empty .pb for primary-beam taper verification")

    pbcor_consistent = False
    if (
        pb_arr is not None
        and pb_arr.shape == shape_2d
        and pbcor_arr is not None
        and pbcor_arr.shape == shape_2d
        and valid_pb20_count >= 100
    ):
        pb25_mask = valid_pb20 & (pb_arr >= max(pb_cut + 0.05, 0.25))
        if int(np.sum(pb25_mask)) >= 50:
            peak_abs_img = float(np.nanmax(np.abs(img_arr[valid_pb20])))
            pbcor_abs_diff = np.abs(pbcor_arr[pb25_mask] * pb_arr[pb25_mask] - img_arr[pb25_mask])
            pbcor_recon_med_err_jy = float(np.nanmedian(pbcor_abs_diff))
            pbcor_recon_p999_err_jy = float(np.nanpercentile(pbcor_abs_diff, 99.9))
            pbcor_tol_jy = 0.03 * max(abs(peak_abs_img), 1e-6)
            pbcor_consistent = (
                math.isfinite(pbcor_recon_med_err_jy)
                and math.isfinite(pbcor_recon_p999_err_jy)
                and pbcor_recon_med_err_jy <= pbcor_tol_jy
                and pbcor_recon_p999_err_jy <= pbcor_tol_jy
            )
            if not pbcor_consistent:
                errors.append(
                    "Inconsistent .image.pbcor * .pb vs .image "
                    f"(median abs diff={pbcor_recon_med_err_jy:.6e} Jy/beam)"
                )
        else:
            errors.append(
                "Insufficient unmasked pixels with pb >= 0.25 for pbcor consistency check"
            )
    else:
        errors.append("Missing .pb or .image.pbcor for pbcor * pb == image verification")

    present_parts = sum(
        1
        for arr in (img_arr, pb_arr, res_arr, mod_arr, pbcor_arr)
        if arr is not None and arr.shape == shape_2d
    )
    ppb = float(hdr["b_minor_arcsec"] / max(hdr["cell_arcsec"], 1e-12))

    nx, ny = shape_2d
    ix, iy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    cell_x = float(hdr["cell_x_arcsec"])
    cell_y = float(hdr["cell_y_arcsec"])

    # Precompute pixel distances to configured source apertures (if any)
    ap_dists: list[tuple[Any, np.ndarray]] = []
    for ap in geom.source_apertures:
        ra_r, dec_r = parse_sky_coord_rad(ap.ra, ap.dec)
        ap_px, ap_py = sky_to_pix(hdr, ra_r, dec_r)
        dist_ap = np.hypot((ix - ap_px) * cell_x, (iy - ap_py) * cell_y)
        ap_dists.append((ap, dist_ap))

    # Off-source residual RMS & MAD/RMS
    offsource_rms_jy: float | None = None
    offsource_mad_std_jy: float | None = None
    offsource_mad_over_rms: float | None = None
    n_offsource_beams: float | None = None

    if res_arr is not None and res_arr.shape == shape_2d and valid_pb20_count >= 100:
        if geom.off_source_min_radius_arcsec is not None:
            if geom.phasecenter:
                ra_c, dec_c = parse_sky_coord_rad(geom.phasecenter)
                px_c, py_c = sky_to_pix(hdr, ra_c, dec_c)
            else:
                px_c, py_c = (nx - 1) / 2.0, (ny - 1) / 2.0
            dist_c = np.hypot((ix - px_c) * cell_x, (iy - py_c) * cell_y)
            min_r = float(geom.off_source_min_radius_arcsec)
            if mod_arr is not None and mod_arr.shape == shape_2d:
                off_mask = valid_pb20 & (dist_c >= min_r) & (np.abs(mod_arr) == 0.0)
            else:
                off_mask = valid_pb20 & (dist_c >= min_r)
            if int(np.sum(off_mask)) < 500:
                off_mask = valid_pb20 & (dist_c >= min_r)
            if int(np.sum(off_mask)) < 200 and mod_arr is not None and mod_arr.shape == shape_2d:
                off_mask = valid_pb20 & (np.abs(mod_arr) == 0.0)
            if int(np.sum(off_mask)) < 50:
                off_mask = valid_pb20
        elif ap_dists:
            beam_excl_arcsec = float(geom.offsource_inner_exclusion_beams) * float(
                hdr["b_major_arcsec"]
            )
            off_mask = valid_pb20.copy()
            for ap, dist_ap in ap_dists:
                excl_r = (
                    float(ap.offsource_exclusion_arcsec)
                    if ap.offsource_exclusion_arcsec is not None
                    else max(beam_excl_arcsec, 1.5 * float(ap.radius_arcsec))
                )
                off_mask = off_mask & (dist_ap > excl_r)
            if int(np.sum(off_mask)) < 100:
                off_mask = valid_pb20.copy()
                for ap, dist_ap in ap_dists:
                    fb_r = (
                        float(ap.fallback_exclusion_arcsec)
                        if ap.fallback_exclusion_arcsec is not None
                        else max(0.7 * beam_excl_arcsec, 1.1 * float(ap.radius_arcsec))
                    )
                    off_mask = off_mask & (dist_ap > fb_r)
        else:
            masked_img = np.where(valid_pb20, img_arr, -np.inf)
            pk_flat = int(np.argmax(masked_img))
            px_pk, py_pk = np.unravel_index(pk_flat, shape_2d)
            dist_pk = np.hypot((ix - float(px_pk)) * cell_x, (iy - float(py_pk)) * cell_y)
            excl_arcsec = float(geom.offsource_inner_exclusion_beams) * float(hdr["b_major_arcsec"])
            if mod_arr is not None and mod_arr.shape == shape_2d:
                off_mask = valid_pb20 & (dist_pk >= excl_arcsec) & (np.abs(mod_arr) == 0.0)
            else:
                off_mask = valid_pb20 & (dist_pk >= excl_arcsec)
            if int(np.sum(off_mask)) < 50:
                off_mask = valid_pb20

        if int(np.sum(off_mask)) >= 20:
            off_vals = res_arr[off_mask]
            offsource_rms_jy = float(np.sqrt(np.mean(off_vals**2)))
            off_med = float(np.median(off_vals))
            offsource_mad_std_jy = float(1.4826 * np.median(np.abs(off_vals - off_med)))
            if math.isfinite(offsource_rms_jy) and offsource_rms_jy > 0.0:
                offsource_mad_over_rms = float(offsource_mad_std_jy / offsource_rms_jy)
            beam_pix = (math.pi * hdr["b_major_arcsec"] * hdr["b_minor_arcsec"]) / (
                4.0 * math.log(2.0) * max(cell_x * cell_y, 1e-24)
            )
            if beam_pix > 0.0:
                n_offsource_beams = float(np.sum(off_mask)) / beam_pix

    # Thermal floor from the agent's own unflagged target visibilities
    sigma_nat_jy: float | None = None
    target_retention: float | None = None
    target_names = [t.strip() for t in extract_config.roles.targets if t.strip()]
    if target_names and (ws / "input.ms" / "table.dat").is_file():
        try:
            tgt_fids = resolve_field_ids_by_names(ws / "input.ms", target_names)
            sigma_nat_jy, _ = natural_thermal_noise_jy(ws / "input.ms", tgt_fids)
            dead_ids = (
                resolve_ant_ids_by_names(ws / "input.ms", list(extract_config.roles.dead_antennas))
                if extract_config.target_retention_excludes_dead_antenna
                else []
            )
            target_retention = target_parallel_hand_retention(
                ws / "input.ms", tgt_fids, exclude_ant_ids=dead_ids
            )
        except Exception as exc:
            errors.append(f"Thermal noise / retention estimate failed: {type(exc).__name__}: {exc}")
    floor_ratio = thermal_floor_ratio(offsource_rms_jy, sigma_nat_jy, n_offsource_beams)

    # Clean model negative-bowl ratio
    negative_bowl_ratio: float | None = None
    if mod_arr is not None and mod_arr.shape == shape_2d and valid_pb20_count >= 100:
        mod_pb = mod_arr[pb_gate & np.isfinite(mod_arr)]
        pos_model_sum_jy = float(np.sum(mod_pb[mod_pb > 0.0]))
        neg_model_abs_sum_jy = float(np.abs(np.sum(mod_pb[mod_pb < 0.0])))
        if pos_model_sum_jy > 1e-12:
            negative_bowl_ratio = float(neg_model_abs_sum_jy / pos_model_sum_jy)
        else:
            negative_bowl_ratio = 1.0
            errors.append("Clean .model has zero positive flux (uncleaned dirty image)")

    # Peak intensity & dynamic range
    peak_img_jy: float | None = None
    dynamic_range: float | None = None
    if valid_pb20_count >= 100:
        peak_img_jy = float(np.nanmax(img_arr[valid_pb20]))
        if offsource_rms_jy is not None and offsource_rms_jy > 0.0 and math.isfinite(peak_img_jy):
            dynamic_range = float(peak_img_jy / offsource_rms_jy)

    # Source integrated flux & point-source SNR from configured `source_apertures`
    ext_apertures = [(ap, d) for ap, d in ap_dists if ap.kind in ("extended", "region")]
    pt_apertures = [(ap, d) for ap, d in ap_dists if ap.kind == "point"]

    source_integrated_flux_jy: float | None = None
    if (
        ext_apertures
        and pbcor_arr is not None
        and pbcor_arr.shape == shape_2d
        and valid_pb20_count >= 100
    ):
        beam_area_pix = (math.pi * hdr["b_major_arcsec"] * hdr["b_minor_arcsec"]) / max(
            4.0 * math.log(2.0) * cell_x * cell_y, 1e-12
        )
        ap0, dist_ext = ext_apertures[0]
        ext_ap = valid_pb20 & (dist_ext <= float(ap0.radius_arcsec))
        if np.any(ext_ap):
            source_integrated_flux_jy = float(np.sum(pbcor_arr[ext_ap]) / beam_area_pix)

    point_source_snr: float | None = None
    if (
        pt_apertures
        and valid_pb20_count >= 100
        and offsource_rms_jy is not None
        and offsource_rms_jy > 0.0
    ):
        ap_pt, dist_pt = pt_apertures[0]
        pt_box = valid_pb20 & (dist_pt <= float(ap_pt.radius_arcsec))
        if np.any(pt_box):
            pt_peak_img_jy = float(np.max(img_arr[pt_box]))
            point_source_snr = float(pt_peak_img_jy / offsource_rms_jy)

    ncc_val = (
        _measure_convolved_sky_ncc(pbcor_arr, valid_pb20, ref_arrays_dir)
        if (pbcor_arr is not None and pbcor_arr.shape == shape_2d)
        else None
    )

    obs["image.products_complete"] = ObservableRecord(
        value=bool(present_parts == 5),
        status="measured",
        provenance=prov,
    )
    obs["image.valid_product_count"] = ObservableRecord(
        value=int(present_parts),
        status="measured",
        provenance=prov,
    )
    obs["image.shapes_and_wcs_valid"] = ObservableRecord(
        value=shapes_and_wcs_valid,
        status="measured",
        provenance=prov,
    )
    obs["image.pb_taper_consistent"] = ObservableRecord(
        value=bool(pb_taper_valid and pbcor_consistent),
        status="measured",
        provenance=f"{chosen.pb_path.name if chosen.pb_path else prov}+{chosen.pbcor_path.name if chosen.pbcor_path else prov}",
    )
    obs["image.pixels_per_beam"] = finite_observable(
        ppb,
        provenance=chosen.image_path.name,
        missing_status="extractor_error",
    )
    obs["image.cell_arcsec"] = finite_observable(
        hdr["cell_arcsec"],
        unit="arcsec",
        provenance=chosen.image_path.name,
        missing_status="extractor_error",
        ndigits=6,
    )
    obs["image.b_major_arcsec"] = finite_observable(
        hdr["b_major_arcsec"],
        unit="arcsec",
        provenance=chosen.image_path.name,
        missing_status="extractor_error",
        ndigits=6,
    )
    obs["image.b_minor_arcsec"] = finite_observable(
        hdr["b_minor_arcsec"],
        unit="arcsec",
        provenance=chosen.image_path.name,
        missing_status="extractor_error",
        ndigits=6,
    )
    obs["image.b_pa_deg"] = finite_observable(
        hdr["b_pa_deg"],
        unit="deg",
        provenance=chosen.image_path.name,
        missing_status="extractor_error",
        ndigits=4,
    )
    obs["image.offsource_residual_rms_jy"] = finite_observable(
        offsource_rms_jy,
        unit="Jy/beam",
        provenance=chosen.residual_path.name if chosen.residual_path else prov,
        missing_status="missing_artifact",
    )
    obs["image.offsource_residual_mad_std_jy"] = finite_observable(
        offsource_mad_std_jy,
        unit="Jy/beam",
        provenance=chosen.residual_path.name if chosen.residual_path else prov,
        missing_status="missing_artifact",
    )
    obs["image.offsource_residual_mad_over_rms"] = finite_observable(
        offsource_mad_over_rms,
        provenance=chosen.residual_path.name if chosen.residual_path else prov,
        missing_status="missing_artifact",
    )
    obs["image.thermal_noise_natural_jy"] = (
        finite_observable(sigma_nat_jy, unit="Jy/beam", provenance="input.ms")
        if sigma_nat_jy is not None
        else ObservableRecord(value=None, status="not_applicable", unit="Jy/beam")
    )
    obs["image.offsource_rms_over_thermal_floor"] = (
        finite_observable(
            floor_ratio,
            provenance=f"{chosen.residual_path.name if chosen.residual_path else prov} vs input.ms",
        )
        if floor_ratio is not None
        else ObservableRecord(value=None, status="not_applicable")
    )
    obs["image.target_row_retention"] = (
        finite_observable(target_retention, provenance="input.ms::FLAG", ndigits=6)
        if target_retention is not None
        else ObservableRecord(value=None, status="not_applicable")
    )
    obs["image.peak_jy_per_beam"] = finite_observable(
        peak_img_jy,
        unit="Jy/beam",
        provenance=chosen.image_path.name,
        missing_status="missing_artifact",
    )
    obs["image.dynamic_range"] = finite_observable(
        dynamic_range,
        provenance=f"{chosen.image_path.name}/{chosen.residual_path.name if chosen.residual_path else 'residual'}",
        missing_status="missing_artifact",
    )
    obs["image.source_integrated_flux_jy"] = (
        ObservableRecord(
            value=None,
            status="not_applicable",
            unit="Jy",
            provenance=chosen.pbcor_path.name if chosen.pbcor_path else prov,
        )
        if not ext_apertures
        else finite_observable(
            source_integrated_flux_jy,
            unit="Jy",
            provenance=chosen.pbcor_path.name if chosen.pbcor_path else prov,
            missing_status="missing_artifact",
        )
    )
    obs["image.point_source_snr"] = (
        ObservableRecord(
            value=None,
            status="not_applicable",
            provenance=chosen.image_path.name,
        )
        if not pt_apertures
        else finite_observable(
            point_source_snr,
            provenance=chosen.image_path.name,
            missing_status="missing_artifact",
        )
    )
    obs["image.negative_bowl_ratio"] = finite_observable(
        negative_bowl_ratio,
        provenance=chosen.model_path.name if chosen.model_path else prov,
        missing_status="missing_artifact",
    )
    obs["image.convolved_sky_ncc"] = finite_observable(
        ncc_val,
        provenance=f"{chosen.pbcor_path.name if chosen.pbcor_path else prov} vs true_sky.npz",
        missing_status="not_applicable",
    )
    return obs, errors
