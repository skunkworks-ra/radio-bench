"""Deterministic scientific verifier for image-stage tasks (s1_image, r1_3c391_image)."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.grading.cal_metrics import _parse_sky_coord_rad
from radio_bench.grading.precal_metrics import has_active_table_lock

PASS_REWARD_MIN_DEFAULT = 0.85

_SKIP_DIR_SUFFIXES = (
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
    prefix: str
    image_path: Path
    pb_path: Path | None
    residual_path: Path | None
    model_path: Path | None
    pbcor_path: Path | None
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


def _discover_image_sets(workspace_dir: Path, max_depth: int = 3) -> list[DiscoveredImageSet]:
    """Discover CASA continuum image sets (`.image` or `.image.tt0`) under `workspace_dir`."""
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
                restored_candidates.append((entry, True))
                continue
            if low.endswith(".image") and _is_casa_table_dir(entry):
                restored_candidates.append((entry, False))
                continue
            if any(low.endswith(sfx) for sfx in _SKIP_DIR_SUFFIXES) or _is_casa_table_dir(entry):
                continue
            _scan(entry, depth + 1)

    _scan(workspace_dir, 0)

    sets: list[DiscoveredImageSet] = []
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
                ]
            )

        # Fallback if agent wrote a single custom-named *.pbcor in the same directory
        if pbcor is None:
            alt_pbcors = [
                p
                for p in parent.iterdir()
                if p.name.lower().endswith(".pbcor") and _is_casa_table_dir(p)
            ]
            if len(alt_pbcors) == 1:
                pbcor = alt_pbcors[0]

        try:
            mtime = float((img_dir / "table.dat").stat().st_mtime)
        except OSError:
            mtime = 0.0

        sets.append(
            DiscoveredImageSet(
                prefix=str(parent / stem),
                image_path=img_dir,
                pb_path=pb,
                residual_path=res,
                model_path=mod,
                pbcor_path=pbcor,
                mtime=mtime,
            )
        )

    sets.sort(key=lambda s: (s.n_present, s.mtime), reverse=True)
    return sets


def _read_image_2d(
    img_path: Path,
    *,
    need_header: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
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


def _sky_to_pix(hdr: dict[str, Any], ra_rad: float, dec_rad: float) -> tuple[float, float]:
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


def _extract_image_truth(ground_truth: dict[str, Any]) -> dict[str, Any]:
    top_gt = ground_truth
    if "image_ground_truth" in ground_truth and isinstance(
        ground_truth["image_ground_truth"], dict
    ):
        gt = ground_truth["image_ground_truth"]
    else:
        gt = ground_truth

    suite_raw = str(gt.get("suite") or top_gt.get("suite") or "").strip().lower()
    is_suite_r = suite_raw == "r" or "3c391" in str(gt.get("task_id", top_gt.get("task_id", "")))

    tols = gt.get("tolerances") if isinstance(gt.get("tolerances"), dict) else {}
    top_tols = top_gt.get("tolerances") if isinstance(top_gt.get("tolerances"), dict) else {}

    ppb_min = float(tols.get("pixels_per_beam_min", tols.get("min_pixels_per_beam", 3.0)))
    ppb_max = float(tols.get("pixels_per_beam_max", tols.get("max_pixels_per_beam", 7.0)))
    pb_cutoff = float(tols.get("pb_cutoff", 0.2))
    bowl_max = float(
        tols.get("negative_model_bowl_ratio_max", tols.get("max_negative_bowl_ratio", 0.12))
    )
    pass_min = float(
        gt.get(
            "pass_reward_min",
            tols.get("pass_reward_min", top_tols.get("pass_reward_min", PASS_REWARD_MIN_DEFAULT)),
        )
    )

    w_src = (
        gt.get("image_weights")
        if isinstance(gt.get("image_weights"), dict)
        else (gt.get("weights") if isinstance(gt.get("weights"), dict) else {})
    )
    weights = {
        "products_valid": float(w_src.get("products_valid", 0.15)),
        "beam_sampling": float(w_src.get("beam_sampling", 0.20)),
        "off_source_residual_rms": float(w_src.get("off_source_residual_rms", 0.25)),
        "deconv_and_bowl": float(
            w_src.get(
                "flux_snr_and_bowl_guard",
                w_src.get("dynamic_range_and_bowl_guard", 0.40),
            )
        ),
    }

    if is_suite_r:
        min_rms_jy = float(tols.get("off_source_residual_rms_min_jy_per_beam", 0.08e-3))
        max_rms_jy = float(tols.get("off_source_residual_rms_max_jy_per_beam", 0.00075))
        dr_min = float(tols.get("dynamic_range_min", 200.0))
        phasecenter = str(gt.get("phasecenter", "J2000 282.35101667deg -0.92793889deg"))
        off_min_rad = float(gt.get("off_source_min_radius_arcsec", 220.0))
        return {
            "is_suite_r": True,
            "ppb_min": ppb_min,
            "ppb_max": ppb_max,
            "pb_cutoff": pb_cutoff,
            "min_offsource_rms_jy": min_rms_jy,
            "max_offsource_rms_jy": max_rms_jy,
            "dynamic_range_min": dr_min,
            "phasecenter": phasecenter,
            "off_source_min_radius_arcsec": off_min_rad,
            "bowl_max": bowl_max,
            "pass_reward_min": pass_min,
            "weights": weights,
        }

    noise = gt.get("noise") if isinstance(gt.get("noise"), dict) else top_gt.get("noise", {})
    sigma_th_jy = float(noise.get("sigma_th_jy_per_beam", 0.00014895))
    rms_mult = float(
        tols.get(
            "max_offsource_rms_sigma_th_mult",
            noise.get("residual_rms_max_sigma_th_mult", 1.5),
        )
    )
    sky = (
        gt.get("sky_model")
        if isinstance(gt.get("sky_model"), dict)
        else top_gt.get("sky_model", {})
    )
    ext = sky.get("extended_gaussian", {})
    pt = sky.get("point_source", {})
    return {
        "is_suite_r": False,
        "ppb_min": ppb_min,
        "ppb_max": ppb_max,
        "pb_cutoff": pb_cutoff,
        "sigma_th_jy": sigma_th_jy,
        "min_offsource_rms_jy": float(tols.get("min_offsource_rms_sigma_th_mult", 0.35))
        * sigma_th_jy,
        "max_offsource_rms_sigma_th_mult": rms_mult,
        "max_offsource_rms_jy": rms_mult * sigma_th_jy,
        "ext_ra": str(ext.get("ra", "18h24m00.000s")),
        "ext_dec": str(ext.get("dec", "-09d35m00.000s")),
        "ext_flux_jy": float(ext.get("flux_jy", 0.075)),
        "ext_aperture_arcsec": float(ext.get("aperture_radius_arcsec", 50.0)),
        "max_ext_flux_rel_err": float(tols.get("max_ext_flux_rel_err", 0.15)),
        "pt_ra": str(pt.get("ra", "18h24m08.200s")),
        "pt_dec": str(pt.get("dec", "-09d34m20.000s")),
        "pt_flux_jy": float(pt.get("flux_jy", 0.00210422)),
        "min_point_source_snr": float(
            tols.get("min_point_source_snr", pt.get("min_snr_threshold", 5.0))
        ),
        "bowl_max": bowl_max,
        "pass_reward_min": pass_min,
        "weights": weights,
    }


def grade_image(
    workspace_dir: Path,
    ground_truth: dict[str, Any],
    *,
    masks_npz_path: Path | None = None,
    task_id: str,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Grade delivered CASA continuum image products inside `/workspace`."""
    from casatasks import clearstat

    _ = masks_npz_path
    with contextlib.suppress(Exception):
        clearstat()

    ws = Path(workspace_dir)
    truth = _extract_image_truth(ground_truth)
    weights = truth["weights"]
    pass_min = truth["pass_reward_min"]
    errors: list[str] = []

    candidates = _discover_image_sets(ws)
    if not candidates:
        errors.append("No CASA .image or .image.tt0 products found in /workspace")
        reward_dict: dict[str, float | int] = {
            "reward": 0.0,
            "passed": 0,
            "products_score": 0.0,
            "beam_sampling_score": 0.0,
            "residual_rms_score": 0.0,
            "deconv_and_bowl_score": 0.0,
        }
        metrics_dict: dict[str, Any] = {
            "task_id": task_id,
            "stage": "image",
            "pass_reward_min": pass_min,
            "table_locked": False,
            "truth": truth,
            "measured": {"image_sets_found": 0},
            "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
            "weights": weights,
            "errors": errors,
        }
        return reward_dict, metrics_dict

    chosen = candidates[0]
    table_locked = False
    for p in (
        chosen.image_path,
        chosen.pb_path,
        chosen.residual_path,
        chosen.model_path,
        chosen.pbcor_path,
    ):
        if p is not None and has_active_table_lock(p):
            table_locked = True
            errors.append(f"Active casacore write lock on {p}")

    if table_locked:
        reward_dict = {
            "reward": 0.0,
            "passed": 0,
            "products_score": 0.0,
            "beam_sampling_score": 0.0,
            "residual_rms_score": 0.0,
            "deconv_and_bowl_score": 0.0,
        }
        metrics_dict = {
            "task_id": task_id,
            "stage": "image",
            "pass_reward_min": pass_min,
            "table_locked": True,
            "truth": truth,
            "measured": {"chosen_prefix": chosen.prefix},
            "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
            "weights": weights,
            "errors": errors,
        }
        return reward_dict, metrics_dict

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
        img_arr, img_msk, hdr = _read_image_2d(chosen.image_path, need_header=True)
    except Exception as exc:
        errors.append(f"Failed reading restored image {chosen.image_path.name}: {exc}")

    if chosen.pb_path is not None:
        try:
            pb_arr, pb_msk, _ = _read_image_2d(chosen.pb_path)
        except Exception as exc:
            errors.append(f"Failed reading pb {chosen.pb_path.name}: {exc}")
    else:
        errors.append("Missing .pb image product")

    if chosen.residual_path is not None:
        try:
            res_arr, res_msk, _ = _read_image_2d(chosen.residual_path)
        except Exception as exc:
            errors.append(f"Failed reading residual {chosen.residual_path.name}: {exc}")
    else:
        errors.append("Missing .residual image product")

    if chosen.model_path is not None:
        try:
            mod_arr, _mod_msk, _ = _read_image_2d(chosen.model_path)
        except Exception as exc:
            errors.append(f"Failed reading model {chosen.model_path.name}: {exc}")
    else:
        errors.append("Missing .model image product")

    if chosen.pbcor_path is not None:
        try:
            pbcor_arr, pbcor_msk, _ = _read_image_2d(chosen.pbcor_path)
        except Exception as exc:
            errors.append(f"Failed reading pbcor {chosen.pbcor_path.name}: {exc}")
    else:
        errors.append("Missing .image.pbcor / .pbcor image product")

    if img_arr is None or hdr is None:
        reward_dict = {
            "reward": 0.0,
            "passed": 0,
            "products_score": 0.0,
            "beam_sampling_score": 0.0,
            "residual_rms_score": 0.0,
            "deconv_and_bowl_score": 0.0,
        }
        metrics_dict = {
            "task_id": task_id,
            "stage": "image",
            "pass_reward_min": pass_min,
            "table_locked": False,
            "truth": truth,
            "measured": {"chosen_prefix": chosen.prefix},
            "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
            "weights": weights,
            "errors": errors,
        }
        return reward_dict, metrics_dict

    shape_2d = img_arr.shape
    shapes_match = all(
        a is None or a.shape == shape_2d for a in (pb_arr, res_arr, mod_arr, pbcor_arr)
    )
    if not shapes_match:
        errors.append("Image product spatial shapes do not match restored image")

    pb_cut = truth["pb_cutoff"]
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

    # Physical Primary Beam (.pb) mosaic taper & (.image.pbcor * .pb == .image) consistency checks
    pb_max_20: float | None = None
    pb_min_20: float | None = None
    pb_min_full: float | None = None
    pb_std_20: float | None = None
    pb_taper_valid = False
    if pb_arr is not None and pb_arr.shape == shape_2d and int(np.sum(pb_gate)) >= 100:
        pb_in_gate = pb_arr[pb_gate]
        pb_max_20 = float(np.max(pb_in_gate))
        pb_min_20 = float(np.min(pb_in_gate))
        pb_fin_all = pb_arr[np.isfinite(pb_arr)]
        pb_min_full = float(np.min(pb_fin_all)) if pb_fin_all.size > 0 else 1.0
        pb_std_20 = float(np.std(pb_in_gate))
        pb_taper_valid = (
            (0.92 <= pb_max_20 <= 1.05)
            and (pb_min_20 <= 0.65 or pb_min_full < 0.85)
            and (pb_std_20 >= 0.05)
        )
        if not pb_taper_valid:
            errors.append(
                f"Unphysical .pb taper (max={pb_max_20:.4f}, min_pb20={pb_min_20:.4f}, "
                f"min_full={pb_min_full:.4f}, std={pb_std_20:.4f})"
            )
    else:
        errors.append("Missing or empty .pb for mosaic primary-beam taper verification")

    pbcor_recon_med_err_jy: float | None = None
    pbcor_consistent = False
    if (
        pb_arr is not None
        and pb_arr.shape == shape_2d
        and pbcor_arr is not None
        and pbcor_arr.shape == shape_2d
        and valid_pb20_count >= 100
    ):
        pb25_mask = valid_pb20 & (pb_arr >= 0.25)
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
                    f"Inconsistent .image.pbcor * .pb vs .image "
                    f"(median abs diff={pbcor_recon_med_err_jy:.6e} Jy/beam)"
                )
        else:
            errors.append(
                "Insufficient unmasked pixels with pb >= 0.25 for pbcor consistency check"
            )
    else:
        errors.append("Missing .pb or .image.pbcor for pbcor * pb == image verification")

    # 1. Products valid (0.15 weight)
    present_parts = sum(
        1
        for arr in (img_arr, pb_arr, res_arr, mod_arr, pbcor_arr)
        if arr is not None and arr.shape == shape_2d
    )
    if (
        present_parts == 5
        and valid_pb20_count >= 100
        and shapes_match
        and pb_taper_valid
        and pbcor_consistent
    ):
        products_score = 1.0
    elif valid_pb20_count >= 100 and shapes_match and pb_taper_valid and pbcor_consistent:
        products_score = 0.2 * (present_parts - 1)
    else:
        products_score = 0.0

    # 2. Beam sampling 3.0 <= b_minor / cell <= 7.0 (0.20 weight)
    ppb = hdr["b_minor_arcsec"] / max(hdr["cell_arcsec"], 1e-12)
    ppb_min = truth["ppb_min"]
    ppb_max = truth["ppb_max"]
    if ppb_min <= ppb <= ppb_max:
        beam_sampling_score = 1.0
    elif ppb < ppb_min:
        beam_sampling_score = max(0.0, (ppb - 1.5) / max(ppb_min - 1.5, 1e-6))
    else:
        beam_sampling_score = max(0.0, 1.0 - (ppb - ppb_max) / 5.0)

    # Build sky-offset coordinate grids
    nx, ny = shape_2d
    ix, iy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    cell_x = hdr["cell_x_arcsec"]
    cell_y = hdr["cell_y_arcsec"]

    # 3. Off-source flatnoise residual RMS inside pb >= 0.2 (0.25 weight)
    offsource_rms_jy: float | None = None
    offsource_mad_std_jy: float | None = None
    offsource_pixel_count = 0
    residual_physically_valid = False
    residual_rms_score = 0.0

    if res_arr is not None and res_arr.shape == shape_2d and valid_pb20_count >= 100:
        if truth["is_suite_r"]:
            ra_c, dec_c = _parse_sky_coord_rad(truth["phasecenter"])
            px_c, py_c = _sky_to_pix(hdr, ra_c, dec_c)
            dist_c = np.hypot((ix - px_c) * cell_x, (iy - py_c) * cell_y)
            min_r = truth["off_source_min_radius_arcsec"]
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
        else:
            ext_ra_r, ext_dec_r = _parse_sky_coord_rad(truth["ext_ra"], truth["ext_dec"])
            pt_ra_r, pt_dec_r = _parse_sky_coord_rad(truth["pt_ra"], truth["pt_dec"])
            ext_px, ext_py = _sky_to_pix(hdr, ext_ra_r, ext_dec_r)
            pt_px, pt_py = _sky_to_pix(hdr, pt_ra_r, pt_dec_r)
            dist_ext = np.hypot((ix - ext_px) * cell_x, (iy - ext_py) * cell_y)
            dist_pt = np.hypot((ix - pt_px) * cell_x, (iy - pt_py) * cell_y)
            off_mask = valid_pb20 & (dist_ext > 75.0) & (dist_pt > 35.0)
            if int(np.sum(off_mask)) < 100:
                off_mask = valid_pb20 & (dist_ext > 55.0) & (dist_pt > 25.0)

        offsource_pixel_count = int(np.sum(off_mask))
        if offsource_pixel_count >= 20:
            off_vals = res_arr[off_mask]
            offsource_rms_jy = float(np.sqrt(np.mean(off_vals**2)))
            off_med = float(np.median(off_vals))
            offsource_mad_std_jy = float(1.4826 * np.median(np.abs(off_vals - off_med)))
            min_rms = truth["min_offsource_rms_jy"]
            max_rms = truth["max_offsource_rms_jy"]
            residual_physically_valid = (
                math.isfinite(offsource_rms_jy)
                and math.isfinite(offsource_mad_std_jy)
                and offsource_rms_jy >= min_rms
                and offsource_mad_std_jy >= 0.50 * offsource_rms_jy
            )
            if not residual_physically_valid:
                residual_rms_score = 0.0
                errors.append(
                    f"Unphysical off-source residual (rms={offsource_rms_jy:.6e} Jy/beam, "
                    f"min_required={min_rms:.6e}, mad_std={offsource_mad_std_jy:.6e})"
                )
            elif offsource_rms_jy < max_rms:
                residual_rms_score = 1.0
            else:
                residual_rms_score = max(0.0, 1.0 - (offsource_rms_jy - max_rms) / (1.7 * max_rms))

    # 4. Target flux / DR & negative-model bowl guard (0.40 weight)
    pos_model_sum_jy: float | None = None
    neg_model_abs_sum_jy: float | None = None
    negative_bowl_ratio: float | None = None
    bowl_sub = 0.0
    bowl_cap_mult = 0.0

    if mod_arr is not None and mod_arr.shape == shape_2d and valid_pb20_count >= 100:
        mod_pb = mod_arr[pb_gate & np.isfinite(mod_arr)]
        pos_model_sum_jy = float(np.sum(mod_pb[mod_pb > 0.0]))
        neg_model_abs_sum_jy = float(np.abs(np.sum(mod_pb[mod_pb < 0.0])))
        if pos_model_sum_jy > 1e-12:
            negative_bowl_ratio = neg_model_abs_sum_jy / pos_model_sum_jy
            bowl_max = truth["bowl_max"]
            if negative_bowl_ratio < bowl_max:
                bowl_sub = 1.0
                bowl_cap_mult = 1.0
            else:
                bowl_sub = 0.0
                bowl_cap_mult = max(0.0, 1.0 - (negative_bowl_ratio - bowl_max) / bowl_max) * 0.25
        else:
            negative_bowl_ratio = 1.0
            errors.append("Clean .model has zero positive flux (uncleaned dirty image)")

    extra_measured: dict[str, Any] = {}
    deconv_and_bowl_score = 0.0

    if truth["is_suite_r"]:
        peak_img_jy: float | None = None
        dynamic_range: float | None = None
        if valid_pb20_count >= 100 and offsource_rms_jy is not None and offsource_rms_jy > 0.0:
            peak_img_jy = float(np.nanmax(img_arr[valid_pb20]))
            dynamic_range = peak_img_jy / offsource_rms_jy
            dr_min = truth["dynamic_range_min"]
            if dynamic_range >= dr_min:
                dr_sub = 1.0
            else:
                dr_sub = max(0.0, (dynamic_range - 0.4 * dr_min) / max(0.6 * dr_min, 1.0))
            deconv_and_bowl_score = (0.70 * dr_sub + 0.30 * bowl_sub) * bowl_cap_mult
        extra_measured["peak_image_jy_per_beam"] = (
            round(peak_img_jy, 6) if peak_img_jy is not None else None
        )
        extra_measured["dynamic_range"] = (
            round(dynamic_range, 3) if dynamic_range is not None else None
        )
    else:
        ext_flux_jy: float | None = None
        ext_rel_err: float | None = None
        pt_peak_img_jy: float | None = None
        pt_snr_img: float | None = None

        ext_ra_r, ext_dec_r = _parse_sky_coord_rad(truth["ext_ra"], truth["ext_dec"])
        pt_ra_r, pt_dec_r = _parse_sky_coord_rad(truth["pt_ra"], truth["pt_dec"])
        ext_px, ext_py = _sky_to_pix(hdr, ext_ra_r, ext_dec_r)
        pt_px, pt_py = _sky_to_pix(hdr, pt_ra_r, pt_dec_r)
        dist_ext = np.hypot((ix - ext_px) * cell_x, (iy - ext_py) * cell_y)
        dist_pt = np.hypot((ix - pt_px) * cell_x, (iy - pt_py) * cell_y)

        ext_sub = 0.0
        if pbcor_arr is not None and pbcor_arr.shape == shape_2d and valid_pb20_count >= 100:
            beam_area_pix = (math.pi * hdr["b_major_arcsec"] * hdr["b_minor_arcsec"]) / max(
                4.0 * math.log(2.0) * cell_x * cell_y, 1e-12
            )
            ext_ap = valid_pb20 & (dist_ext <= truth["ext_aperture_arcsec"])
            if np.any(ext_ap):
                ext_flux_jy = float(np.sum(pbcor_arr[ext_ap]) / beam_area_pix)
                ext_rel_err = abs(ext_flux_jy - truth["ext_flux_jy"]) / truth["ext_flux_jy"]
                e_tol = truth["max_ext_flux_rel_err"]
                if ext_rel_err <= e_tol:
                    ext_sub = 1.0
                else:
                    ext_sub = max(0.0, 1.0 - (ext_rel_err - e_tol) / 0.30)

        pt_sub = 0.0
        if valid_pb20_count >= 100 and offsource_rms_jy is not None and offsource_rms_jy > 0.0:
            pt_box = valid_pb20 & (dist_pt <= 20.0)
            if np.any(pt_box):
                pt_peak_img_jy = float(np.max(img_arr[pt_box]))
                pt_snr_img = pt_peak_img_jy / offsource_rms_jy
                snr_min = truth["min_point_source_snr"]
                if pt_snr_img >= snr_min:
                    pt_sub = 1.0
                else:
                    pt_sub = max(0.0, (pt_snr_img - 2.0) / max(snr_min - 2.0, 1e-6))

        deconv_and_bowl_score = (0.40 * ext_sub + 0.35 * pt_sub + 0.25 * bowl_sub) * bowl_cap_mult
        extra_measured.update(
            {
                "ext_integrated_flux_jy": round(ext_flux_jy, 6)
                if ext_flux_jy is not None
                else None,
                "ext_flux_rel_err": round(ext_rel_err, 6) if ext_rel_err is not None else None,
                "pt_peak_image_jy_per_beam": round(pt_peak_img_jy, 7)
                if pt_peak_img_jy is not None
                else None,
                "pt_snr_image": round(pt_snr_img, 3) if pt_snr_img is not None else None,
                "offsource_rms_over_sigma_th": round(offsource_rms_jy / truth["sigma_th_jy"], 4)
                if offsource_rms_jy is not None
                else None,
            }
        )

    if not (pb_taper_valid and pbcor_consistent and residual_physically_valid):
        deconv_and_bowl_score = 0.0

    composite = (
        weights["products_valid"] * products_score
        + weights["beam_sampling"] * beam_sampling_score
        + weights["off_source_residual_rms"] * residual_rms_score
        + weights["deconv_and_bowl"] * deconv_and_bowl_score
    )
    reward = round(min(1.0, max(0.0, composite)), 6)
    passed = 1 if reward >= pass_min else 0

    reward_dict = {
        "reward": reward,
        "passed": passed,
        "products_score": round(products_score, 6),
        "beam_sampling_score": round(beam_sampling_score, 6),
        "residual_rms_score": round(residual_rms_score, 6),
        "deconv_and_bowl_score": round(deconv_and_bowl_score, 6),
    }
    metrics_dict = {
        "task_id": task_id,
        "stage": "image",
        "pass_reward_min": pass_min,
        "table_locked": False,
        "truth": truth,
        "measured": {
            "chosen_prefix": chosen.prefix,
            "image_path": str(chosen.image_path),
            "pb_path": str(chosen.pb_path) if chosen.pb_path else None,
            "residual_path": str(chosen.residual_path) if chosen.residual_path else None,
            "model_path": str(chosen.model_path) if chosen.model_path else None,
            "pbcor_path": str(chosen.pbcor_path) if chosen.pbcor_path else None,
            "imsize": [int(shape_2d[0]), int(shape_2d[1])],
            "cell_arcsec": round(hdr["cell_arcsec"], 4),
            "b_major_arcsec": round(hdr["b_major_arcsec"], 4),
            "b_minor_arcsec": round(hdr["b_minor_arcsec"], 4),
            "b_pa_deg": round(hdr["b_pa_deg"], 3),
            "pixels_per_beam": round(ppb, 4),
            "valid_pb20_pixel_count": valid_pb20_count,
            "offsource_pixel_count": offsource_pixel_count,
            "offsource_residual_rms_jy_per_beam": round(offsource_rms_jy, 8)
            if offsource_rms_jy is not None
            else None,
            "offsource_residual_rms_mjy_per_beam": round(offsource_rms_jy * 1e3, 5)
            if offsource_rms_jy is not None
            else None,
            "offsource_residual_mad_std_jy_per_beam": round(offsource_mad_std_jy, 8)
            if offsource_mad_std_jy is not None
            else None,
            "residual_physically_valid": residual_physically_valid,
            "pb_taper_valid": pb_taper_valid,
            "pb_max_pb20": round(pb_max_20, 6) if pb_max_20 is not None else None,
            "pb_min_pb20": round(pb_min_20, 6) if pb_min_20 is not None else None,
            "pb_std_pb20": round(pb_std_20, 6) if pb_std_20 is not None else None,
            "pbcor_consistent": pbcor_consistent,
            "pbcor_recon_median_abs_err_jy": round(pbcor_recon_med_err_jy, 10)
            if pbcor_recon_med_err_jy is not None
            else None,
            "model_positive_sum_jy": round(pos_model_sum_jy, 6)
            if pos_model_sum_jy is not None
            else None,
            "model_negative_abs_sum_jy": round(neg_model_abs_sum_jy, 6)
            if neg_model_abs_sum_jy is not None
            else None,
            "negative_bowl_ratio": round(negative_bowl_ratio, 6)
            if negative_bowl_ratio is not None
            else None,
            **extra_measured,
        },
        "subscores": {k: v for k, v in reward_dict.items() if k not in ("reward", "passed")},
        "weights": weights,
        "errors": errors,
    }
    return reward_dict, metrics_dict
