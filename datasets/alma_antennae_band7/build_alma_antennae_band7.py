"""Checkpoint + ground-truth builder for the alma_antennae_band7 reference suite.

Needs only casatools, casatasks, numpy (runs inside radio-bench-casa-base on OSG).
Inputs come from `run.py fetch` under <suite-root>/extracted/. Outputs, all under <suite-root>:
  post_calibrate_{north,south}/input.ms      guide calibrated MS copy, HISTORY rows removed
  g1_antennae_image_{north,south}/ground_truth.json
  reference/reference_measurements.json     imstat of the guide-shipped continuum images
Target values are the guide's published numbers (see REFERENCE.json for quotes); tolerance
factors are copied from the 3C391 / 24A-376 suites.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

GUIDE_IMAGING_URL = "https://casaguides.nrao.edu/index.php/AntennaeBand7_Imaging_6.6.6"

# Guide-published continuum numbers (quotes and sections in REFERENCE.json).
MOSAICS: dict[str, dict[str, Any]] = {
    "north": {
        "task_id": "g1_antennae_image_north",
        "src_ms": "Antennae_Band7_CalibratedData/Antennae_North.cal.ms",
        "ref_image": "Antennae_North.Cont.Clean.image.fits",
        "guide": {
            "peak_jy_per_beam": 0.00380,
            "rms_jy_per_beam": 0.00049,
            "b_major_arcsec": 1.23,
            "b_minor_arcsec": 0.72,
            "b_pa_deg": 85.4,
        },
    },
    "south": {
        "task_id": "g1_antennae_image_south",
        "src_ms": "Antennae_Band7_CalibratedData/Antennae_South.cal.ms",
        "ref_image": "Antennae_South.Cont.Clean.image.fits",
        "guide": {
            "peak_jy_per_beam": 0.00478,
            "rms_jy_per_beam": 0.00047,
            "b_major_arcsec": 1.14,
            "b_minor_arcsec": 0.66,
            "b_pa_deg": 61.5,
        },
    },
}

# Tolerance conventions (sources named in REFERENCE.json).
PIXELS_PER_BEAM_MIN = 3.0  # 3C391 r1_3c391_image ground_truth tolerances.pixels_per_beam_min
PIXELS_PER_BEAM_MAX = 7.0  # 3C391 r1_3c391_image ground_truth tolerances.pixels_per_beam_max
RMS_MAX_FACTOR = 2.0  # 24A-376 builder: off_source_residual_rms_max = ref_rms * 2.0
RMS_MIN_JY = 8e-05  # 3C391 r1_3c391_image.json residual_rms_score threshold_gate limit
DR_MIN_FACTOR = 0.55  # 24A-376 builder: dynamic_range_min = max(150, 0.55 * ref_dr)
BOWL_MAX = 0.12  # 3C391 + 24A-376 tolerances.negative_model_bowl_ratio_max
PEAK_MAX_FACTOR = 2.0  # reuse of the 24A-376 rms_max factor as a line-contamination guard
PRODUCT_GUIDE_REL_TOL = 0.05  # builder self-check: product imstat vs guide text


def _setup_env(suite_root: Path) -> None:
    tmp = Path(os.environ.get("TMPDIR", "") or suite_root / "_tmp")
    if tmp == Path("/tmp") or str(tmp).startswith("/tmp/"):
        tmp = suite_root / "_tmp"
    mpl = Path(os.environ.get("MPLCONFIGDIR", "") or tmp / "mplconfig")
    tmp.mkdir(parents=True, exist_ok=True)
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


@contextlib.contextmanager
def open_tb(path: Path | str, nomodify: bool = True) -> Iterator[Any]:
    from casatools import table

    tb = table()
    opened = False
    try:
        opened = bool(tb.open(str(path), nomodify=nomodify))
        if not opened:
            raise RuntimeError(f"table.open returned False for {path}")
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


def assert_unlocked(ms_path: Path) -> None:
    import casatasks

    casatasks.clearstat()
    with open_tb(ms_path) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active table lock on {ms_path}")


def compute_raw_data_blake2b(ms_path: Path, chunk_size: int = 50_000) -> str:
    """Same digest as radio_bench.grading.extractors.common.compute_raw_data_blake2b."""
    cols = ("TIME", "ANTENNA1", "ANTENNA2", "FIELD_ID", "DATA_DESC_ID", "DATA")
    hasher = hashlib.blake2b(digest_size=32)
    from casatools import table

    tb = table()
    opened = False
    try:
        opened = bool(tb.open(str(ms_path), nomodify=True, lockoptions={"option": "usernoread"}))
        if not opened:
            raise RuntimeError(f"table.open returned False for {ms_path}")
        missing = [c for c in cols if c not in set(tb.colnames())]
        if missing:
            raise KeyError(f"Missing immutable MS columns in {ms_path.name}: {missing}")
        hasher.update(f"total_nrows={int(tb.nrows())};".encode())
        ddids = sorted(int(x) for x in np.unique(np.asarray(tb.getcol("DATA_DESC_ID"))))
        for ddid in ddids:
            sub = tb.query(f"DATA_DESC_ID == {ddid}")
            try:
                n = int(sub.nrows())
                for sr in range(0, n, chunk_size):
                    nr = min(chunk_size, n - sr)
                    hasher.update(f"ddid={ddid}:sr={sr}:nr={nr};".encode())
                    for col in cols:
                        arr = np.ascontiguousarray(
                            np.asarray(sub.getcol(col, startrow=sr, nrow=nr))
                        )
                        hasher.update(f"{col}:{arr.dtype.str}:{arr.shape};".encode())
                        hasher.update(memoryview(arr))
            finally:
                with contextlib.suppress(Exception):
                    sub.close()
                with contextlib.suppress(Exception):
                    sub.done()
    finally:
        if opened:
            with contextlib.suppress(Exception):
                tb.close()
        with contextlib.suppress(Exception):
            tb.done()
    return hasher.hexdigest()


def measure_reference_image(fits_path: Path) -> dict[str, Any]:
    """imstat-equivalent numbers of a guide-shipped restored image (flat noise, not pbcor)."""
    from casatools import image

    ia = image()
    try:
        ia.open(str(fits_path))
        st = ia.statistics(robust=False)
        rb = ia.restoringbeam()
        shape = [int(x) for x in ia.shape()]
        cs = ia.coordsys()
        try:
            cell_arcsec = abs(float(cs.increment()["numeric"][1])) * 180.0 / math.pi * 3600.0
        finally:
            cs.done()
    finally:
        ia.close()
        ia.done()
    peak = float(np.asarray(st["max"]).ravel()[0])
    rms = float(np.asarray(st["rms"]).ravel()[0])
    return {
        "file": fits_path.name,
        "shape": shape,
        "cell_arcsec": round(cell_arcsec, 6),
        "peak_jy_per_beam": round(peak, 8),
        "rms_jy_per_beam": round(rms, 8),
        "dynamic_range": round(peak / rms, 3),
        "flux_full_image_jy": round(float(np.asarray(st["flux"]).ravel()[0]), 6),
        "npts": int(np.asarray(st["npts"]).ravel()[0]),
        "b_major_arcsec": round(float(rb["major"]["value"]), 4),
        "b_minor_arcsec": round(float(rb["minor"]["value"]), 4),
        "b_pa_deg": round(float(rb["positionangle"]["value"]), 3),
        "pixels_per_beam": round(float(rb["minor"]["value"]) / cell_arcsec, 3),
        "definition": "casatools image.statistics over all unmasked pixels; DR = max / rms",
    }


def check_product_vs_guide(side: str, prod: dict[str, Any], guide: dict[str, Any]) -> None:
    for pkey, gkey in (
        ("peak_jy_per_beam", "peak_jy_per_beam"),
        ("rms_jy_per_beam", "rms_jy_per_beam"),
    ):
        rel = abs(prod[pkey] - guide[gkey]) / guide[gkey]
        if rel > PRODUCT_GUIDE_REL_TOL:
            raise RuntimeError(
                f"{side}: product {pkey}={prod[pkey]} vs guide {guide[gkey]} ({rel:.3f})"
            )


def make_checkpoint(src_ms: Path, out_dir: Path) -> Path:
    final_ms = out_dir / "input.ms"
    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    stage_ms = staging / "input.ms"
    shutil.copytree(src_ms, stage_ms, symlinks=False)
    for lock in stage_ms.rglob("table.lock"):
        lock.unlink()
    clear_history(stage_ms)
    assert_unlocked(stage_ms)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    return final_ms


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()
    suite_root = args.suite_root.resolve()
    extracted = suite_root / "extracted"
    if not extracted.is_dir():
        raise SystemExit(f"missing {extracted}; run `run.py fetch` first")
    _setup_env(suite_root)

    summary: dict[str, Any] = {"guide_imaging_url": GUIDE_IMAGING_URL, "mosaics": {}}
    for side, m in MOSAICS.items():
        src_ms = extracted / m["src_ms"]
        ref_fits = extracted / m["ref_image"]
        if not src_ms.is_dir() or not ref_fits.is_file():
            raise SystemExit(f"{side}: missing {src_ms} or {ref_fits}")

        prod = measure_reference_image(ref_fits)
        check_product_vs_guide(side, prod, m["guide"])

        ms_path = make_checkpoint(src_ms, suite_root / f"post_calibrate_{side}")
        digest = compute_raw_data_blake2b(ms_path)
        guide = dict(m["guide"])
        gt = {
            "raw_data_blake2b": digest,
            "guide_published": guide,
            "reference_product": prod,
            "tolerances": {
                "pixels_per_beam_min": PIXELS_PER_BEAM_MIN,
                "pixels_per_beam_max": PIXELS_PER_BEAM_MAX,
                "off_source_residual_rms_max_jy_per_beam": round(
                    RMS_MAX_FACTOR * guide["rms_jy_per_beam"], 8
                ),
                "off_source_residual_rms_min_jy_per_beam": RMS_MIN_JY,
                "dynamic_range_min": round(DR_MIN_FACTOR * prod["dynamic_range"], 3),
                "negative_model_bowl_ratio_max": BOWL_MAX,
                "continuum_peak_max_jy_per_beam": round(
                    PEAK_MAX_FACTOR * guide["peak_jy_per_beam"], 8
                ),
            },
        }
        write_json(suite_root / m["task_id"] / "ground_truth.json", gt)
        summary["mosaics"][side] = {"input_ms": str(ms_path), "raw_data_blake2b": digest, **gt}
        print(
            f"[{side}] checkpoint {ms_path}  blake2b={digest[:16]}  DR_ref={prod['dynamic_range']}"
        )

    write_json(suite_root / "reference" / "reference_measurements.json", summary)
    print("build done")


if __name__ == "__main__":
    main()
