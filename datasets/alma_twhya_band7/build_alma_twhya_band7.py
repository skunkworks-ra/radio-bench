"""Deterministic checkpoint + ground-truth builder for the alma_twhya_band7 reference suite.

Two phases, both writing only under <data_dir>/casaguides/alma_twhya_band7/:

- `checkpoints` (casatools/casatasks/numpy only; runs in casa-base on OSG):
  copies the guide's calibrated MS to checkpoints/post_calibrate/input.ms with HISTORY cleared,
  then reproduces the guide's non-interactive continuum image of the science target under
  reference/ (split width=8, hogbom, 250 x 0.1 arcsec, briggs 0.5, box mask, 15 mJy threshold).
- `truth` (needs radio_bench importable: pixi env or the verifier image):
  measures the reference image with the same `extract_image` the verifier uses, hashes the
  checkpoint DATA with `compute_raw_data_blake2b`, checks golden headroom, and writes
  g1_twhya_image/ground_truth.json.

Tolerances are guide-anchored (REFERENCE.json holds provenance for every constant below).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SUITE_ID = "alma_twhya_band7"
IMAGE_TASK = "g1_twhya_image"
HERE = Path(__file__).resolve().parent
SUITE_TOML = HERE / "suites" / SUITE_ID / "suite.toml"

SRC_MS_NAME = "twhya_calibrated.ms"
TARGET_FIELD_NAME = "TW Hya"
TARGET_FIELD_ID = "5"  # guide: split(field='5'); field 4 shares the name and has zero rows

# Guide continuum recipe (First Look at Imaging CASA 6.6.6, non-interactive twhya_cont_auto),
# with pbcor=True in place of the separate impbcor step so the .image.pbcor product exists.
SPLIT_WIDTH = 8
REFERENCE_TCLEAN: dict[str, Any] = {
    "field": "0",
    "spw": "",
    "specmode": "mfs",
    "gridder": "standard",
    "deconvolver": "hogbom",
    "imsize": [250, 250],
    "cell": ["0.1arcsec"],
    "mask": "box [[100pix, 100pix], [150pix, 150pix]]",
    "weighting": "briggs",
    "robust": 0.5,
    "threshold": "15mJy",
    "niter": 10000,
    "interactive": False,
    "pbcor": True,
}
REFERENCE_PREFIX = "twhya_cont_auto"

# Guide-published values (REFERENCE.json status GUIDE, with quotes).
GUIDE_RMS_NO_SELFCAL_JY = 0.0066
GUIDE_RMS_IMAGING_APPROX_JY = 0.007
GUIDE_BEAM_NO_SELFCAL_ARCSEC = (0.560, 0.416)
GUIDE_BEAM_PA_NO_SELFCAL_DEG = -55.974
GUIDE_RMS_SELFCAL_FINAL_JY = 0.0020

# Tolerance conventions (REFERENCE.json status CONVENTION).
RMS_MAX_FACTOR = 2.0  # vla_24a376_lband: off_source_residual_rms_max = ref_rms * 2.0
PIXELS_PER_BEAM_MIN = 3.0  # vla_3c391_cband SPEC PIXELS_PER_BEAM_MIN
PIXELS_PER_BEAM_MAX = 7.0  # vla_3c391_cband SPEC PIXELS_PER_BEAM_MAX
PB_CUTOFF = 0.2  # vla_3c391_cband SPEC PB_CUTOFF
NEGATIVE_MODEL_BOWL_RATIO_MAX = 0.12  # vla_3c391_cband SPEC NEGATIVE_MODEL_BOWL_RATIO_MAX
GOLDEN_RMS_HEADROOM = 0.60 / 0.75  # vla_3c391_cband GOLDEN_..._RMS_MAX_MJY / SPEC ..._MAX_MJY
GOLDEN_BOWL_MAX = 0.05  # vla_3c391_cband GOLDEN_NEGATIVE_MODEL_BOWL_RATIO_MAX
MAD_OVER_RMS_MIN = 0.50  # vla_3c391_cband image rubric threshold_gate limit

RMS_MAX_JY = round(GUIDE_RMS_NO_SELFCAL_JY * RMS_MAX_FACTOR, 6)
# The guide publishes no peak, so there is no dynamic_range_min and no rms floor (both null).


def suite_root(data_dir: Path) -> Path:
    return data_dir / "casaguides" / SUITE_ID


def setup_env(root: Path) -> None:
    tmp = root / "_tmp"
    (tmp / "mplconfig").mkdir(parents=True, exist_ok=True)
    for key, val in (("TMPDIR", tmp), ("MPLCONFIGDIR", tmp / "mplconfig")):
        cur = os.environ.get(key, "")
        if not cur or cur == "/tmp" or cur.startswith("/tmp/"):
            os.environ[key] = str(val)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


@contextlib.contextmanager
def open_tb(path: Path | str, nomodify: bool = True) -> Iterator[Any]:
    from casatools import table

    tb = table()
    opened = False
    try:
        tb.open(str(path), nomodify=nomodify)
        opened = True
        yield tb
    finally:
        if opened:
            with contextlib.suppress(Exception):
                tb.close()
        with contextlib.suppress(Exception):
            tb.done()


def clear_history(ms_path: Path) -> None:
    hist = ms_path / "HISTORY"
    if (hist / "table.dat").is_file():
        with open_tb(hist, nomodify=False) as tb:
            n = int(tb.nrows())
            if n > 0:
                tb.removerows(list(range(n)))


def assert_unlocked(path: Path) -> None:
    import casatasks

    casatasks.clearstat()
    with open_tb(path) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active table lock on {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ms_meta(ms_path: Path) -> dict[str, Any]:
    import numpy as np
    from casatools import msmetadata

    with open_tb(ms_path) as tb:
        nrows = int(tb.nrows())
        cols = sorted(tb.colnames())
        field_id = tb.getcol("FIELD_ID")
        flagged_frac = float(np.mean(tb.getcol("FLAG")))
        a1 = tb.getcol("ANTENNA1")
        a2 = tb.getcol("ANTENNA2")
    with open_tb(ms_path / "HISTORY") as tb:
        hist_rows = int(tb.nrows())
    md = msmetadata()
    try:
        md.open(str(ms_path))
        names = [str(x) for x in md.fieldnames()]
        intents = {str(i): sorted(str(x) for x in md.intentsforfield(i)) for i in range(len(names))}
        nchan = [int(md.nchan(i)) for i in range(md.nspw())]
        ant_names = [str(x) for x in md.antennanames()]
    finally:
        with contextlib.suppress(Exception):
            md.close()
        with contextlib.suppress(Exception):
            md.done()
    active = sorted({int(x) for x in np.unique(np.concatenate([a1, a2]))})
    return {
        "ms_path": str(ms_path),
        "nrows": nrows,
        "columns": cols,
        "field_names": names,
        "field_intents": intents,
        "rows_per_field": {str(int(f)): int(np.sum(field_id == f)) for f in np.unique(field_id)},
        "spw_nchan": nchan,
        "active_antennas": [ant_names[i] for i in active],
        "flagged_vis_fraction": round(flagged_frac, 6),
        "history_nrows": hist_rows,
    }


def casa_image_stats(prefix: Path) -> dict[str, Any]:
    """CASA-only header + imstat summary of the reference products (informational)."""
    import casatasks
    from casatools import image, quanta

    ia = image()
    qa = quanta()
    beam: dict[str, float] = {}
    try:
        ia.open(str(prefix.with_name(prefix.name + ".image")))
        rb = ia.restoringbeam()
        beam = {
            "b_major_arcsec": round(float(qa.convert(rb["major"], "arcsec")["value"]), 4),
            "b_minor_arcsec": round(float(qa.convert(rb["minor"], "arcsec")["value"]), 4),
            "b_pa_deg": round(float(qa.convert(rb["positionangle"], "deg")["value"]), 3),
        }
    finally:
        with contextlib.suppress(Exception):
            ia.close()
        with contextlib.suppress(Exception):
            ia.done()
        with contextlib.suppress(Exception):
            qa.done()
    img = casatasks.imstat(imagename=str(prefix.with_name(prefix.name + ".image")))
    res = casatasks.imstat(imagename=str(prefix.with_name(prefix.name + ".residual")))
    mod = casatasks.imstat(imagename=str(prefix.with_name(prefix.name + ".model")))
    casatasks.clearstat()
    return {
        **beam,
        "image_max_jy_per_beam": round(float(img["max"][0]), 6),
        "residual_rms_full_jy_per_beam": round(float(res["rms"][0]), 6),
        "model_sum_jy": round(float(mod["sum"][0]), 6),
    }


# ---------------------------------------------------------------------------
# Phase 1: CASA-only checkpoints + reference image
# ---------------------------------------------------------------------------


def phase_checkpoints(root: Path) -> None:
    setup_env(root)
    import casatasks

    src_ms = root / "extracted" / SRC_MS_NAME
    if not (src_ms / "table.dat").is_file():
        raise SystemExit(f"missing {src_ms}; run `run.py fetch` first")

    ckpt = root / "checkpoints" / "post_calibrate"
    in_ms = ckpt / "input.ms"
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True)
    shutil.copytree(src_ms, in_ms)
    for fv in ckpt.glob("*.flagversions"):
        shutil.rmtree(fv)
    clear_history(in_ms)
    assert_unlocked(in_ms)
    meta = ms_meta(in_ms)
    print(f"[1/2] checkpoint {in_ms}: {meta['nrows']} rows, history={meta['history_nrows']}")

    ref = root / "reference"
    if ref.exists():
        shutil.rmtree(ref)
    ref.mkdir(parents=True)
    smoothed = ref / "twhya_smoothed.ms"
    casatasks.split(
        vis=str(in_ms),
        field=TARGET_FIELD_ID,
        width=str(SPLIT_WIDTH),
        outputvis=str(smoothed),
        datacolumn="data",
    )
    prefix = ref / REFERENCE_PREFIX
    casatasks.tclean(
        vis=str(smoothed), imagename=str(prefix), datacolumn="data", **REFERENCE_TCLEAN
    )
    casatasks.clearstat()
    for temp_lat in ref.glob("TempLattice*"):
        shutil.rmtree(temp_lat, ignore_errors=True)
    assert_unlocked(smoothed)
    stats = casa_image_stats(prefix)
    write_json(
        root / "casa_summary.json",
        {
            "source_ms": str(src_ms),
            "checkpoint": meta,
            "reference_split": {"field": TARGET_FIELD_ID, "width": SPLIT_WIDTH},
            "reference_tclean": REFERENCE_TCLEAN,
            "reference_prefix": str(prefix),
            "reference_casa_stats": stats,
            "products": sorted(p.name for p in ref.iterdir()),
        },
    )
    print(f"[2/2] reference image {prefix}: {stats}")


# ---------------------------------------------------------------------------
# Phase 2: verifier-identical measurements + ground truth (needs radio_bench)
# ---------------------------------------------------------------------------


def phase_truth(root: Path) -> None:
    setup_env(root)
    try:
        from radio_bench.grading.extractors.common import compute_raw_data_blake2b
        from radio_bench.grading.extractors.image import extract_image
        from radio_bench.grading.schema import SuiteManifest
    except ImportError as exc:
        raise SystemExit(
            f"truth phase needs radio_bench importable (pixi env or verifier image): {exc}"
        ) from exc

    in_ms = root / "checkpoints" / "post_calibrate" / "input.ms"
    ref = root / "reference"
    if not (in_ms / "table.dat").is_file() or not ref.is_dir():
        raise SystemExit("missing checkpoint or reference image; run the checkpoints phase first")

    manifest = SuiteManifest.model_validate(tomllib.loads(SUITE_TOML.read_text(encoding="utf-8")))
    xcfg = manifest.to_extract_config(IMAGE_TASK)
    obs, errs = extract_image(ref, xcfg)
    o = {k: v.value for k, v in obs.items()}
    digest = compute_raw_data_blake2b(in_ms)

    ref_rms = o.get("image.offsource_residual_rms_jy")
    if ref_rms is None:
        raise SystemExit(f"reference image measurement failed: {errs}")

    checks = [
        ("valid_product_count == 5", o.get("image.valid_product_count") == 5),
        ("shapes_and_wcs_valid", o.get("image.shapes_and_wcs_valid") is True),
        ("pb_taper_consistent", o.get("image.pb_taper_consistent") is True),
        (
            f"pixels_per_beam in [{PIXELS_PER_BEAM_MIN}, {PIXELS_PER_BEAM_MAX}]",
            PIXELS_PER_BEAM_MIN <= float(o["image.pixels_per_beam"]) <= PIXELS_PER_BEAM_MAX,
        ),
        (
            f"ref rms <= {GOLDEN_RMS_HEADROOM:.2f} * rms_max ({RMS_MAX_JY})",
            float(ref_rms) <= GOLDEN_RMS_HEADROOM * RMS_MAX_JY,
        ),
        (
            f"negative_bowl_ratio <= {GOLDEN_BOWL_MAX}",
            # missing fails; a measured 0.0 must stay 0.0
            o.get("image.negative_bowl_ratio") is not None
            and float(o["image.negative_bowl_ratio"]) <= GOLDEN_BOWL_MAX,
        ),
        (
            f"mad_over_rms >= {MAD_OVER_RMS_MIN}",
            float(o.get("image.offsource_residual_mad_over_rms") or 0.0) >= MAD_OVER_RMS_MIN,
        ),
    ]
    measured = {
        "raw_data_blake2b": digest,
        "extractor_observables": o,
        "extractor_errors": errs,
        "golden_checks": [{"check": c, "pass": bool(p)} for c, p in checks],
    }
    write_json(root / "truth_summary.json", measured)
    failed = [c for c, p in checks if not p]
    if failed:
        raise SystemExit(f"golden headroom checks failed (no ground truth written): {failed}")

    gt = {
        "task_id": IMAGE_TASK,
        "suite": SUITE_ID,
        "raw_data_blake2b": digest,
        "guide_reference": {
            "rms_no_selfcal_jy_per_beam": GUIDE_RMS_NO_SELFCAL_JY,
            "rms_imaging_approx_jy_per_beam": GUIDE_RMS_IMAGING_APPROX_JY,
            "beam_no_selfcal_arcsec": list(GUIDE_BEAM_NO_SELFCAL_ARCSEC),
            "beam_pa_no_selfcal_deg": GUIDE_BEAM_PA_NO_SELFCAL_DEG,
            "rms_selfcal_final_jy_per_beam": GUIDE_RMS_SELFCAL_FINAL_JY,
        },
        "reference_image_summary": {
            "note": "builder reproduction of the guide recipe; informational, not a target",
            "image_prefix": f"reference/{REFERENCE_PREFIX}",
            "tclean": REFERENCE_TCLEAN,
            "split_width": SPLIT_WIDTH,
            "b_major_arcsec": o.get("image.b_major_arcsec"),
            "b_minor_arcsec": o.get("image.b_minor_arcsec"),
            "b_pa_deg": o.get("image.b_pa_deg"),
            "pixels_per_beam": o.get("image.pixels_per_beam"),
            "peak_image_jy_per_beam": o.get("image.peak_jy_per_beam"),
            "offsource_residual_rms_jy": ref_rms,
            "dynamic_range": o.get("image.dynamic_range"),
            "negative_bowl_ratio": o.get("image.negative_bowl_ratio"),
        },
        "tolerances": {
            "pixels_per_beam_min": PIXELS_PER_BEAM_MIN,
            "pixels_per_beam_max": PIXELS_PER_BEAM_MAX,
            "pb_cutoff": PB_CUTOFF,
            "off_source_residual_rms_max_jy_per_beam": RMS_MAX_JY,
            "negative_model_bowl_ratio_max": NEGATIVE_MODEL_BOWL_RATIO_MAX,
            "pass_reward_min": 0.85,
        },
    }
    write_json(root / IMAGE_TASK / "ground_truth.json", gt)
    print(
        f"ground truth -> {root / IMAGE_TASK / 'ground_truth.json'}: rms_max={RMS_MAX_JY}, "
        f"ref_rms={ref_rms}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("checkpoints", "truth", "all"))
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = suite_root(args.data_dir.resolve())
    if str(root).startswith("/tmp"):
        raise SystemExit("refusing to build under /tmp")
    if args.phase in ("checkpoints", "all"):
        phase_checkpoints(root)
    if args.phase in ("truth", "all"):
        phase_truth(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
