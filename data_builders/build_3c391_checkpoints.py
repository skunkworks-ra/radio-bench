"""Deterministic builder for Suite R golden 3C391 checkpoints and ground-truth JSONs.

Extracts the untouched archival tarball `3c391_ctm_mosaic_10s_spw0.ms.tgz` strictly read-only
(falling back to copying `.ms` + resetting scratch/flags/STATE/HISTORY if only `.ms` exists) and
generates under `<data_dir>/checkpoints/3c391/`:
  - `raw/input.ms` (clean uncalibrated MS with zero scratch columns or prior-run history)
  - `post_precal/input.ms`, `post_precal/calibrators.ms`, `post_precal/precal_masks.npz`
  - `post_calibrate/input.ms`, `post_calibrate/caltables/{G0,K,B,G,fluxscale}.cal`
  - `reference_image/3c391_mosaic.{image,image.pbcor,pb,residual,model,psf,sumwt}`
  - `ground_truth.json` and per-task `{task_id}/ground_truth.json` (+ staged `precal_masks.npz`)
    for all 5 Suite R tasks.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import importlib  # noqa: E402

import numpy as np  # noqa: E402

from data_builders.sim_common import configure_builder_casa_env  # noqa: E402
from radio_bench.config import (  # noqa: E402
    BenchConfig,
    add_config_cli_args,
    config_from_cli_args,
    load_bench_config,
)


class _LazyCasaProxy:
    """Defer importing casatools/casatasks until runtime env and CASASITECONFIG are prepared."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._mod: Any = None

    def __getattr__(self, attr: str) -> Any:
        if self._mod is None:
            configure_builder_casa_env()
            self._mod = importlib.import_module(self._module_name)
        return getattr(self._mod, attr)


casatasks: Any = _LazyCasaProxy("casatasks")
casatools: Any = _LazyCasaProxy("casatools")


def resolve_default_3c391_paths(cfg: BenchConfig | None = None) -> tuple[Path, Path]:
    """Locate default archival 3C391 `.ms` and `.ms.tgz` paths without host hardcoding."""
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    env_ms = os.environ.get("RADIO_BENCH_3C391_MS", "").strip()
    env_tgz = os.environ.get("RADIO_BENCH_3C391_TGZ", "").strip()
    if env_ms or env_tgz:
        ms_p = Path(env_ms) if env_ms else Path(env_tgz).with_suffix("")
        tgz_p = Path(env_tgz) if env_tgz else Path(f"{ms_p}.tgz")
        return ms_p, tgz_p

    search_dirs = (
        resolved_cfg.casaguides_dir,
        resolved_cfg.casaguides_dir / "vla_3c391_adv",
        resolved_cfg.data_dir / "datasets" / "3c391_fresh",
        resolved_cfg.data_dir.parent / "datasets" / "3c391_fresh",
    )
    for base in search_dirs:
        cand_ms = base / "3c391_ctm_mosaic_10s_spw0.ms"
        cand_tgz = base / "3c391_ctm_mosaic_10s_spw0.ms.tgz"
        if cand_ms.exists() or cand_tgz.exists():
            return cand_ms, cand_tgz

    fallback_base = resolved_cfg.casaguides_dir
    return (
        fallback_base / "3c391_ctm_mosaic_10s_spw0.ms",
        fallback_base / "3c391_ctm_mosaic_10s_spw0.ms.tgz",
    )


def build_forbidden_history_substrings(cfg: BenchConfig | None = None) -> tuple[str, ...]:
    """Build forbidden MS HISTORY substrings dynamically from runtime host roots."""
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    dynamic_roots: list[str] = []
    for root in (resolved_cfg.work_dir.parent, resolved_cfg.data_dir, Path.home()):
        s = str(root).rstrip("/")
        if s and s != "/":
            dynamic_roots.append(f"{s}/")
    ordered = [
        "Perley-Butler",
        "phase_cal.cl",
        "!ea13",
        *dynamic_roots,
        "cal.K",
        "cal.B",
        "3c391_run",
        "3c391_fresh",
    ]
    return tuple(dict.fromkeys(ordered))


RAW_SCRATCH_COLUMNS = ("CORRECTED_DATA", "MODEL_DATA", "IMAGING_WEIGHT")
ARCHIVAL_HISTORY_ROWS = 14


def __getattr__(name: str) -> Any:
    if name in {
        "BENCH_CFG",
        "DEFAULT_SOURCE_MS",
        "DEFAULT_SOURCE_TGZ",
        "DEFAULT_OUT_DIR",
        "FORBIDDEN_HISTORY_SUBSTRINGS",
    }:
        cfg = load_bench_config()
        if name == "BENCH_CFG":
            return cfg
        if name == "DEFAULT_OUT_DIR":
            return cfg.data_dir / "checkpoints" / "3c391"
        if name == "FORBIDDEN_HISTORY_SUBSTRINGS":
            return build_forbidden_history_substrings(cfg)
        ms_p, tgz_p = resolve_default_3c391_paths(cfg)
        return ms_p if name == "DEFAULT_SOURCE_MS" else tgz_p
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


FLUX_CAL_FIELD = "J1331+3030"
PHASE_CAL_FIELD = "J1822-0938"
POL_CAL_FIELD = "J0319+4130"
TARGET_FIELD_SEL = "2~8"
TARGET_FIELD_IDS = [2, 3, 4, 5, 6, 7, 8]
CALIBRATOR_FIELD_SEL = "0,1,9"
DEAD_ANTENNA_NAME = "ea13"
DEAD_ANTENNA_ID = 10
REFANT_NAME = "ea21"
EXPECTED_J1822_FLUX_JY = 2.30
INTENT_DELIMITERS = [",", ";"]

# SPEC.md Section 4 tolerances (single source of truth for ground_truth.json & validation)
PASS_REWARD_MIN = 0.85
ELEVATION_ABS_TOL_DEG = 0.25
MODEL_AMP_REL_TOL = 0.01
FLAG_RECALL_MIN = 0.95
CLEAN_FALSE_POSITIVE_MAX = 0.05
TARGET_UNFLAGGED_ROWS_MIN = 0.70
TARGET_UNFLAGGED_ROWS_MAX = 0.90
DEAD_ANTENNA_FLAGGED_MIN = 0.99
J1331_CHANNEL_PHASE_RMS_MAX_DEG = 8.0
J1822_FLUX_REL_TOL = 0.08
VERIFIER_TCLEAN_DR_MIN = 180.0
PIXELS_PER_BEAM_MIN = 3.0
PIXELS_PER_BEAM_MAX = 7.0
PB_CUTOFF = 0.2
OFF_SOURCE_RESIDUAL_RMS_MAX_MJY = 0.75
OFF_SOURCE_RESIDUAL_RMS_MAX_JY = 0.00075
DYNAMIC_RANGE_MIN = 200.0
NEGATIVE_MODEL_BOWL_RATIO_MAX = 0.12

# Golden reference headroom requirements inside SPEC tolerances
GOLDEN_OFF_SOURCE_RESIDUAL_RMS_MAX_MJY = 0.60
GOLDEN_DYNAMIC_RANGE_MIN = 230.0
GOLDEN_NEGATIVE_MODEL_BOWL_RATIO_MAX = 0.05


@contextlib.contextmanager
def open_table(path: Path | str, *, nomodify: bool = True) -> Iterator[Any]:
    tb = casatools.table()
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


@contextlib.contextmanager
def open_msmd(path: Path | str) -> Iterator[Any]:
    msmd = casatools.msmetadata()
    opened = False
    try:
        msmd.open(str(path))
        opened = True
        yield msmd
    finally:
        if opened:
            with contextlib.suppress(Exception):
                msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()


@contextlib.contextmanager
def open_image(path: Path | str) -> Iterator[Any]:
    ia = casatools.image()
    opened = False
    try:
        ia.open(str(path))
        opened = True
        yield ia
    finally:
        if opened:
            with contextlib.suppress(Exception):
                ia.close()
        with contextlib.suppress(Exception):
            ia.done()


@contextlib.contextmanager
def open_measures() -> Iterator[tuple[Any, Any]]:
    me = casatools.measures()
    qa = casatools.quanta()
    try:
        yield me, qa
    finally:
        with contextlib.suppress(Exception):
            me.done()
        with contextlib.suppress(Exception):
            qa.done()


def make_tree_writable(path: Path) -> None:
    """Ensure owner write bits are set recursively before updating or removing directories."""
    if not path.exists():
        return
    with contextlib.suppress(OSError):
        path.chmod(path.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR)
    if path.is_dir() and not path.is_symlink():
        for root, dirs, files in os.walk(path):
            root_p = Path(root)
            with contextlib.suppress(OSError):
                root_p.chmod(root_p.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            for dname in dirs:
                dp = root_p / dname
                if not dp.is_symlink():
                    with contextlib.suppress(OSError):
                        dp.chmod(dp.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            for fname in files:
                fp = root_p / fname
                if not fp.is_symlink():
                    with contextlib.suppress(OSError):
                        fp.chmod(fp.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR)


def reset_dir(path: Path) -> None:
    if path.exists():
        make_tree_writable(path)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        with contextlib.suppress(OSError):
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        path.unlink()
    elif path.is_dir():
        make_tree_writable(path)
        shutil.rmtree(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Serialize JSON rejecting NaN/Inf and atomically replace target file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp_path.write_text(serialized, encoding="utf-8")
    os.replace(tmp_path, path)


def stage_file_atomic(src: Path, dst: Path) -> None:
    """Copy a sidecar file atomically into a task directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    remove_path(tmp_path)
    shutil.copy2(src, tmp_path)
    os.replace(tmp_path, dst)


def assert_unlocked(ms_path: Path) -> None:
    casatasks.clearstat()
    with open_table(ms_path, nomodify=True) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"CASA table has active locks: {ms_path}")


def assert_no_history_leaks(ms_path: Path) -> None:
    """Verify MS HISTORY subtable contains zero leaked builder/prior-run strings."""
    hist_path = ms_path / "HISTORY"
    with open_table(hist_path, nomodify=True) as tb_hist:
        hist_cols = set(tb_hist.colnames())
        nrows = int(tb_hist.nrows())
        texts: list[str] = []
        if nrows > 0:
            for col in ("MESSAGE", "CLI_COMMAND", "APP_PARAMS", "APPLICATION", "ORIGIN"):
                if col in hist_cols:
                    vals = tb_hist.getcol(col)
                    texts.extend(str(x) for x in np.ravel(vals))
    for bad in build_forbidden_history_substrings():
        leaked = [t for t in texts if bad in t]
        if leaked:
            raise RuntimeError(
                f"{ms_path}/HISTORY leaked forbidden substring {bad!r} "
                f"in {len(leaked)} entries (first: {leaked[0]!r})"
            )


def sanitize_ms_history(ms_path: Path, *, keep_rows: int = ARCHIVAL_HISTORY_ROWS) -> None:
    """Truncate MS HISTORY subtable to archival rows (`0..keep_rows-1`) and verify zero leaks."""
    hist_path = ms_path / "HISTORY"
    with open_table(hist_path, nomodify=False) as tb_hist:
        nrows = int(tb_hist.nrows())
        if nrows > keep_rows:
            tb_hist.removerows(list(range(keep_rows, nrows)))
    assert_no_history_leaks(ms_path)


def resolve_source_archive(source_ms: Path) -> Path | None:
    """Return untouched `.tgz` archive path if available alongside or as `source_ms`."""
    if source_ms.is_file() and (
        source_ms.name.endswith(".tgz") or source_ms.name.endswith(".tar.gz")
    ):
        return source_ms
    candidate = Path(f"{source_ms}.tgz")
    if candidate.is_file():
        return candidate
    return None


def assert_clean_raw_ms(raw_ms: Path) -> None:
    """Verify staged raw/input.ms matches untouched archival state with zero leaks."""
    with open_table(raw_ms, nomodify=True) as tb:
        cols = set(tb.colnames())
        raw_flag = tb.getcol("FLAG")
        raw_flag_row = tb.getcol("FLAG_ROW")
    leaked_cols = [c for c in RAW_SCRATCH_COLUMNS if c in cols]
    if leaked_cols:
        raise RuntimeError(f"raw/input.ms still contains scratch columns: {leaked_cols}")
    if np.any(raw_flag) or np.any(raw_flag_row):
        raise RuntimeError("raw/input.ms has non-zero FLAG or FLAG_ROW entries")

    with open_table(raw_ms / "STATE", nomodify=True) as tb_state:
        if int(tb_state.nrows()) != 0:
            raise RuntimeError(
                f"raw/input.ms STATE subtable must have 0 rows, got {tb_state.nrows()}"
            )

    assert_no_history_leaks(raw_ms)


def stage_clean_raw_ms(source_ms: Path, raw_ms: Path) -> Path:
    """Populate `raw/input.ms` from `.tgz` archive (or state-equivalent `.ms` fallback)."""
    archive = resolve_source_archive(source_ms)
    if archive is not None:
        raw_ms.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(raw_ms)],
            check=True,
        )
        make_tree_writable(raw_ms)
        with open_table(raw_ms, nomodify=False) as tb:
            drop_cols = [c for c in RAW_SCRATCH_COLUMNS if c in tb.colnames()]
            if drop_cols:
                tb.removecols(drop_cols)
        staged_source = archive
    elif source_ms.is_dir():
        shutil.copytree(source_ms, raw_ms)
        make_tree_writable(raw_ms)
        with open_table(raw_ms, nomodify=False) as tb:
            drop_cols = [c for c in RAW_SCRATCH_COLUMNS if c in tb.colnames()]
            if drop_cols:
                tb.removecols(drop_cols)
            flag_arr = tb.getcol("FLAG")
            tb.putcol("FLAG", np.zeros_like(flag_arr, dtype=bool))
            flag_row_arr = tb.getcol("FLAG_ROW")
            tb.putcol("FLAG_ROW", np.zeros_like(flag_row_arr, dtype=bool))
            if "STATE_ID" in tb.colnames():
                tb.putcol("STATE_ID", np.zeros(int(tb.nrows()), dtype=np.int32))
        with open_table(raw_ms / "STATE", nomodify=False) as tb_state:
            if int(tb_state.nrows()) > 0:
                tb_state.removerows(list(range(int(tb_state.nrows()))))
        staged_source = source_ms
    else:
        raise FileNotFoundError(
            f"Neither archival tarball {source_ms}.tgz nor MS directory found: {source_ms}"
        )

    sanitize_ms_history(raw_ms, keep_rows=ARCHIVAL_HISTORY_ROWS)
    assert_clean_raw_ms(raw_ms)
    assert_unlocked(raw_ms)
    return staged_source


def compute_field_elevation_bounds(
    times_sec: np.ndarray,
    field_ids: np.ndarray,
    phase_dirs_rad: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Compute (el_min_deg, el_max_deg) per field ID at VLA using CASA measures."""
    bounds: dict[int, tuple[float, float]] = {}
    with open_measures() as (me, qa):
        me.doframe(me.observatory("VLA"))
        for fid in sorted({int(x) for x in np.unique(field_ids)}):
            ra_rad = float(phase_dirs_rad[0, 0, fid])
            dec_rad = float(phase_dirs_rad[1, 0, fid])
            sky_dir = me.direction(
                "J2000",
                qa.quantity(ra_rad, "rad"),
                qa.quantity(dec_rad, "rad"),
            )
            unique_times = np.unique(times_sec[field_ids == fid])
            elevations: list[float] = []
            for t_sec in unique_times:
                me.doframe(me.epoch("UTC", qa.quantity(float(t_sec) / 86400.0, "d")))
                azel = me.measure(sky_dir, "AZELGEO")
                el_deg = float(qa.convert(azel["m1"], "deg")["value"])
                elevations.append(el_deg)
            bounds[fid] = (float(min(elevations)), float(max(elevations)))
    return bounds


def inspect_pristine_ms(ms_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read-only extraction of observation metadata and raw visibility flag state."""
    with open_table(ms_path, nomodify=True) as tb:
        n_rows = int(tb.nrows())
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        field_id = tb.getcol("FIELD_ID")
        scan_num = tb.getcol("SCAN_NUMBER")
        times_sec = tb.getcol("TIME")
        raw_flag = tb.getcol("FLAG")
        raw_flag_row = tb.getcol("FLAG_ROW")
        raw_data = tb.getcol("DATA")

    with open_table(ms_path / "ANTENNA", nomodify=True) as tb_ant:
        ant_names = [str(x) for x in tb_ant.getcol("NAME")]

    with open_table(ms_path / "FIELD", nomodify=True) as tb_fld:
        field_names = [str(x) for x in tb_fld.getcol("NAME")]
        phase_dirs_rad = tb_fld.getcol("PHASE_DIR")

    with open_table(ms_path / "SPECTRAL_WINDOW", nomodify=True) as tb_spw:
        chan_freq_hz = tb_spw.getcol("CHAN_FREQ")[:, 0]
        chan_width_hz = tb_spw.getcol("CHAN_WIDTH")[:, 0]

    with open_table(ms_path / "POLARIZATION", nomodify=True) as tb_pol:
        corr_type_codes = tb_pol.getcol("CORR_TYPE")[:, 0].tolist()

    corr_map = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
    correlations = [corr_map.get(int(c), str(c)) for c in corr_type_codes]

    with open_msmd(ms_path) as msmd:
        field_intents = {
            fid: sorted(str(x) for x in msmd.intentsforfield(fid))
            for fid in range(len(field_names))
        }

    el_bounds = compute_field_elevation_bounds(times_sec, field_id, phase_dirs_rad)

    # Identify missing VLA pads (ea01..ea28)
    expected_vla = {f"ea{i:02d}" for i in range(1, 29)}
    missing_pads = sorted(expected_vla - set(ant_names))

    # Measure per-antenna median cross-hand parallel amplitude on J1331+3030 scans 2,3,37
    good_j1331_rows = (field_id == 0) & (scan_num != 1) & (ant1 != ant2)
    parallel_amp = 0.5 * (np.abs(raw_data[0, :, :]) + np.abs(raw_data[3, :, :]))
    ant_median_amp_jy: dict[str, float] = {}
    for aid, aname in enumerate(ant_names):
        mask_a = good_j1331_rows & ((ant1 == aid) | (ant2 == aid))
        vals = parallel_amp[:, mask_a]
        ant_median_amp_jy[aname] = float(np.median(vals)) if vals.size > 0 else 0.0

    role_by_field = {
        0: "flux_bandpass_polangle_calibrator",
        1: "phase_calibrator",
        9: "pol_leakage_calibrator",
    }
    for tid in TARGET_FIELD_IDS:
        role_by_field[tid] = "mosaic_target"

    fields_info: list[dict[str, Any]] = []
    for fid, fname in enumerate(field_names):
        f_scans = sorted(int(s) for s in np.unique(scan_num[field_id == fid]))
        ra_rad = float(phase_dirs_rad[0, 0, fid]) % (2.0 * math.pi)
        dec_rad = float(phase_dirs_rad[1, 0, fid])
        el_min, el_max = el_bounds[fid]
        fields_info.append(
            {
                "field_id": fid,
                "name": fname,
                "scans": f_scans,
                "n_scans": len(f_scans),
                "n_rows": int(np.sum(field_id == fid)),
                "raw_intents": field_intents[fid],
                "true_role": role_by_field[fid],
                "ra_deg": math.degrees(ra_rad),
                "dec_deg": math.degrees(dec_rad),
                "el_min_deg": round(el_min, 4),
                "el_max_deg": round(el_max, 4),
            }
        )

    all_intents_missing = all(len(f["raw_intents"]) == 0 for f in fields_info)
    obs_info = {
        "n_rows": n_rows,
        "n_scans": int(len(np.unique(scan_num))),
        "n_antennas": len(ant_names),
        "antenna_names": ant_names,
        "missing_vla_pads": missing_pads,
        "dead_antenna": DEAD_ANTENNA_NAME,
        "dead_antenna_id": DEAD_ANTENNA_ID,
        "refant": REFANT_NAME,
        "j1331_median_parallel_amp_by_antenna_jy": ant_median_amp_jy,
        "n_spw": 1,
        "n_chan": int(len(chan_freq_hz)),
        "freq_min_ghz": round(float(np.min(chan_freq_hz)) / 1e9, 6),
        "freq_max_ghz": round(float(np.max(chan_freq_hz)) / 1e9, 6),
        "chan_width_mhz": round(float(np.median(np.abs(chan_width_hz))) / 1e6, 6),
        "correlations": correlations,
        "raw_flagged_vis_fraction": float(np.mean(raw_flag)),
        "raw_flagged_row_fraction": float(np.mean(raw_flag_row)),
        "all_intents_missing_in_raw": all_intents_missing,
        "intent_delimiters": list(INTENT_DELIMITERS),
        "fields": fields_info,
    }
    arrays = {
        "ant1": ant1,
        "ant2": ant2,
        "field_id": field_id,
        "scan_num": scan_num,
        "raw_flag": raw_flag,
        "raw_flag_row": raw_flag_row,
        "phase_dirs_rad": phase_dirs_rad,
    }
    return obs_info, arrays


def format_casa_phasecenter_j2000(
    phase_dirs_rad: np.ndarray, target_ids: list[int]
) -> tuple[int, str]:
    """Select central mosaic pointing closest to target centroid and format J2000 string."""
    ras = np.array([float(phase_dirs_rad[0, 0, fid]) % (2.0 * math.pi) for fid in target_ids])
    decs = np.array([float(phase_dirs_rad[1, 0, fid]) for fid in target_ids])
    mean_ra = float(np.mean(ras))
    mean_dec = float(np.mean(decs))
    dists = np.hypot((ras - mean_ra) * math.cos(mean_dec), decs - mean_dec)
    best_idx = int(np.argmin(dists))
    central_fid = target_ids[best_idx]
    ra_deg = math.degrees(ras[best_idx])
    dec_deg = math.degrees(decs[best_idx])
    return central_fid, f"J2000 {ra_deg:.8f}deg {dec_deg:.8f}deg"


def build_post_precal_checkpoint(
    raw_ms: Path,
    out_dir: Path,
    work_dir: Path,
    raw_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build golden post-precal checkpoint (`input.ms`, `calibrators.ms`, `precal_masks.npz`)."""
    post_precal_dir = out_dir / "post_precal"
    reset_dir(post_precal_dir)
    post_precal_ms = post_precal_dir / "input.ms"
    calibrators_ms = post_precal_dir / "calibrators.ms"
    masks_npz = post_precal_dir / "precal_masks.npz"

    # 1. Populate scan intents on all fields into post_precal_ms (archival 3C391 .tgz has empty STATE)
    casatasks.defintent(
        vis=str(raw_ms),
        intent="CALIBRATE_BANDPASS#ON_SOURCE,CALIBRATE_FLUX#ON_SOURCE,CALIBRATE_POL_ANGLE#ON_SOURCE",
        mode="set",
        outputvis=str(post_precal_ms),
        field=FLUX_CAL_FIELD,
    )
    casatasks.defintent(
        vis=str(post_precal_ms),
        intent="CALIBRATE_PHASE#ON_SOURCE",
        mode="set",
        outputvis=str(post_precal_ms),
        field=PHASE_CAL_FIELD,
    )
    casatasks.defintent(
        vis=str(post_precal_ms),
        intent="OBSERVE_TARGET#ON_SOURCE",
        mode="set",
        outputvis=str(post_precal_ms),
        field=TARGET_FIELD_SEL,
    )
    casatasks.defintent(
        vis=str(post_precal_ms),
        intent="CALIBRATE_POL_LEAKAGE#ON_SOURCE",
        mode="set",
        outputvis=str(post_precal_ms),
        field=POL_CAL_FIELD,
    )
    with open_msmd(post_precal_ms) as msmd:
        f0_intents = sorted(str(x) for x in msmd.intentsforfield(0))
        f1_intents = sorted(str(x) for x in msmd.intentsforfield(1))
        f2_intents = sorted(str(x) for x in msmd.intentsforfield(2))
    if (
        "CALIBRATE_BANDPASS#ON_SOURCE" not in f0_intents
        or "CALIBRATE_FLUX#ON_SOURCE" not in f0_intents
    ):
        raise RuntimeError(f"Failed to set flux/bandpass intents on Field 0: {f0_intents}")
    if "CALIBRATE_PHASE#ON_SOURCE" not in f1_intents:
        raise RuntimeError(f"Failed to set CALIBRATE_PHASE#ON_SOURCE on Field 1: {f1_intents}")
    if "CALIBRATE_PHASE#ON_SOURCE" in f2_intents or "OBSERVE_TARGET#ON_SOURCE" not in f2_intents:
        raise RuntimeError(f"Unexpected intents on mosaic target Field 2: {f2_intents}")

    # 2. Deterministic preflagging (scan 1, quack 10s, shadow, clipzeros on !ea13)
    casatasks.flagdata(vis=str(post_precal_ms), mode="manual", scan="1", flagbackup=False)
    casatasks.flagdata(
        vis=str(post_precal_ms),
        mode="quack",
        quackinterval=10.0,
        quackmode="beg",
        flagbackup=False,
    )
    casatasks.flagdata(vis=str(post_precal_ms), mode="shadow", tolerance=0.0, flagbackup=False)
    casatasks.flagdata(
        vis=str(post_precal_ms),
        mode="clip",
        clipzeros=True,
        antenna=f"!{DEAD_ANTENNA_NAME}",
        flagbackup=False,
    )

    with open_table(post_precal_ms, nomodify=True) as tb:
        deterministic_preflag = tb.getcol("FLAG")

    # 3. Populate MODEL_DATA on Field 0 (J1331+3030) with Perley-Butler 2017
    casatasks.setjy(
        vis=str(post_precal_ms),
        field=FLUX_CAL_FIELD,
        standard="Perley-Butler 2017",
        usescratch=True,
    )

    # 4. Derive initial-bandpass + rflag RFI masks on a scratch copy so post_precal/input.ms
    #    never carries a pre-existing CORRECTED_DATA column into r1_3c391_calibrate.
    rfi_work_dir = work_dir / "precal_rfi"
    reset_dir(rfi_work_dir)
    try:
        rfi_ms = rfi_work_dir / "work.ms"
        shutil.copytree(post_precal_ms, rfi_ms)

        g_init = rfi_work_dir / "G_init.cal"
        b_init = rfi_work_dir / "B_init.cal"
        casatasks.gaincal(
            vis=str(rfi_ms),
            caltable=str(g_init),
            field=FLUX_CAL_FIELD,
            spw="0:26~37",
            solint="int",
            refant=REFANT_NAME,
            calmode="p",
            antenna=f"!{DEAD_ANTENNA_NAME}",
            minsnr=3.0,
        )
        casatasks.bandpass(
            vis=str(rfi_ms),
            caltable=str(b_init),
            field=FLUX_CAL_FIELD,
            solint="inf",
            combine="scan",
            refant=REFANT_NAME,
            solnorm=True,
            bandtype="B",
            antenna=f"!{DEAD_ANTENNA_NAME}",
            gaintable=[str(g_init)],
            minsnr=3.0,
        )
        casatasks.applycal(
            vis=str(rfi_ms),
            field=FLUX_CAL_FIELD,
            antenna=f"!{DEAD_ANTENNA_NAME}",
            gaintable=[str(g_init), str(b_init)],
            interp=["nearest", "nearest,linear"],
            calwt=False,
            flagbackup=False,
        )
        casatasks.applycal(
            vis=str(rfi_ms),
            field="1~9",
            antenna=f"!{DEAD_ANTENNA_NAME}",
            gaintable=[str(b_init)],
            gainfield=[FLUX_CAL_FIELD],
            interp=["nearest,linear"],
            calwt=False,
            flagbackup=False,
        )

        # Read B_init-flattened amplitudes before rflag to classify strong (>=8 MAD) vs borderline (>=3.5 MAD) RFI
        with open_table(rfi_ms, nomodify=True) as tb:
            binit_corr = tb.getcol("CORRECTED_DATA")
            binit_flag = tb.getcol("FLAG")

        # Standard 5-sigma rflag on calibrators & target (!ea13) for golden post_precal checkpoint
        casatasks.flagdata(
            vis=str(rfi_ms),
            mode="rflag",
            datacolumn="corrected",
            antenna=f"!{DEAD_ANTENNA_NAME}",
            timedevscale=5.0,
            freqdevscale=5.0,
            flagbackup=False,
        )
        with open_table(rfi_ms, nomodify=True) as tb:
            golden_precal_flag = tb.getcol("FLAG")
    finally:
        remove_path(rfi_work_dir)

    ant1 = raw_arrays["ant1"]
    ant2 = raw_arrays["ant2"]
    field_id = raw_arrays["field_id"]
    scan_num = raw_arrays["scan_num"]
    raw_flag = raw_arrays["raw_flag"]

    ea13_rows = (ant1 == DEAD_ANTENNA_ID) | (ant2 == DEAD_ANTENNA_ID)
    ea13_non_scan1_rows = ea13_rows & (scan_num != 1)

    # Compute strong (>=8 MAD) and borderline (>=3.5 MAD) amplitude outlier masks per field & channel
    par_amp = 0.5 * (np.abs(binit_corr[0]) + np.abs(binit_corr[3]))
    pre_valid = (
        (~np.any(binit_flag, axis=0)) & (~ea13_rows[np.newaxis, :]) & (ant1 != ant2)[np.newaxis, :]
    )
    strong_amp_outlier = np.zeros_like(par_amp, dtype=bool)
    mild_amp_outlier = np.zeros_like(par_amp, dtype=bool)
    for fid in np.unique(field_id):
        f_rows = np.flatnonzero(field_id == fid)
        f_amp = par_amp[:, f_rows]
        f_val = pre_valid[:, f_rows]
        for ch in range(par_amp.shape[0]):
            v_ch = f_amp[ch, f_val[ch]]
            if v_ch.size < 16:
                continue
            med = float(np.median(v_ch))
            mad = float(1.4826 * np.median(np.abs(v_ch - med)))
            if mad <= 0.0:
                continue
            dev = np.abs(f_amp[ch] - med)
            strong_amp_outlier[ch, f_rows] = dev >= 8.0 * mad
            mild_amp_outlier[ch, f_rows] = dev >= 3.5 * mad
    del binit_corr, binit_flag, par_amp

    # Ensure ea13 remains unflagged on scans 2..103 in golden post_precal/input.ms
    # (barring quack/shadow already in deterministic_preflag) so r1_3c391_calibrate tests ea13 flagging.
    final_precal_flag = golden_precal_flag.copy()
    final_precal_flag[:, :, ea13_non_scan1_rows] = deterministic_preflag[:, :, ea13_non_scan1_rows]
    final_precal_flag_row = np.all(final_precal_flag, axis=(0, 1))

    # Write final precal flags to post_precal/input.ms and strip CORRECTED_DATA if setjy added it
    with open_table(post_precal_ms, nomodify=False) as tb:
        tb.putcol("FLAG", final_precal_flag)
        tb.putcol("FLAG_ROW", final_precal_flag_row)
        if "CORRECTED_DATA" in tb.colnames():
            tb.removecols(["CORRECTED_DATA"])
        j1331_rows = (field_id == 0) & (~final_precal_flag_row)
        model_data = tb.getcol("MODEL_DATA")

    # Verify J1331 MODEL_DATA amplitude
    j1331_rr = np.abs(model_data[0, :, :][:, j1331_rows])
    j1331_ll = np.abs(model_data[3, :, :][:, j1331_rows])
    j1331_parallel = 0.5 * (j1331_rr + j1331_ll)
    j1331_mean_amp_jy = float(np.mean(j1331_parallel))
    j1331_per_chan_amp_jy = [float(x) for x in np.mean(j1331_parallel, axis=1)]

    # Split calibrators.ms (Fields 0, 1, 9 with all present columns: DATA + MODEL_DATA)
    casatasks.mstransform(
        vis=str(post_precal_ms),
        outputvis=str(calibrators_ms),
        field=CALIBRATOR_FIELD_SEL,
        datacolumn="all",
    )

    # Purge leaked builder HISTORY rows (!ea13, Perley-Butler, host scratch paths)
    sanitize_ms_history(post_precal_ms, keep_rows=ARCHIVAL_HISTORY_ROWS)
    sanitize_ms_history(calibrators_ms, keep_rows=0)

    # Build precal evaluation masks (64, n_rows)
    raw_chan_flag = np.any(raw_flag, axis=0)
    det_chan_flag = np.any(deterministic_preflag, axis=0)
    golden_rfi_chan_flag = np.any(golden_precal_flag, axis=0)
    strong_chan_flag = golden_rfi_chan_flag & strong_amp_outlier
    border_chan_flag = golden_rfi_chan_flag | mild_amp_outlier

    ea13_mask_2d = np.broadcast_to(ea13_rows[np.newaxis, :], raw_chan_flag.shape)
    edge_chan_mask_2d = np.zeros_like(raw_chan_flag, dtype=bool)
    edge_chan_mask_2d[:2, :] = True
    edge_chan_mask_2d[-2:, :] = True

    must_flag_2d = (~raw_chan_flag) & (det_chan_flag | strong_chan_flag) & (~ea13_mask_2d)
    neutral_2d = (
        raw_chan_flag | ea13_mask_2d | edge_chan_mask_2d | (border_chan_flag & (~must_flag_2d))
    )
    clean_2d = (~raw_chan_flag) & (~must_flag_2d) & (~neutral_2d)

    # Verify golden post_precal_ms against its own masks
    golden_chan_flag = np.any(final_precal_flag, axis=0)
    golden_recall = float(np.mean(golden_chan_flag[must_flag_2d]))
    golden_fp_rate = float(np.mean(golden_chan_flag[clean_2d]))

    np.savez_compressed(
        masks_npz,
        must_flag_2d=np.packbits(must_flag_2d, axis=1),
        neutral_2d=np.packbits(neutral_2d, axis=1),
        clean_2d=np.packbits(clean_2d, axis=1),
        ea13_rows=ea13_rows,
        shape_2d=np.array(must_flag_2d.shape, dtype=np.int64),
    )

    assert_unlocked(post_precal_ms)
    assert_unlocked(calibrators_ms)

    with open_table(calibrators_ms, nomodify=True) as tb_cal:
        cal_nrows = int(tb_cal.nrows())

    ea13_post_precal_flag_frac = float(np.mean(final_precal_flag[:, :, ea13_rows]))
    return {
        "post_precal_ms": str(post_precal_ms),
        "calibrators_ms": str(calibrators_ms),
        "calibrators_nrows": cal_nrows,
        "masks_npz": str(masks_npz),
        "field_1_intents": f1_intents,
        "j1331_model_parallel_mean_jy": round(j1331_mean_amp_jy, 6),
        "j1331_model_parallel_min_jy": round(float(min(j1331_per_chan_amp_jy)), 6),
        "j1331_model_parallel_max_jy": round(float(max(j1331_per_chan_amp_jy)), 6),
        "j1331_model_parallel_per_chan_jy": [round(x, 6) for x in j1331_per_chan_amp_jy],
        "post_precal_flagged_vis_fraction": round(float(np.mean(final_precal_flag)), 6),
        "post_precal_flagged_row_fraction": round(float(np.mean(final_precal_flag_row)), 6),
        "ea13_left_unflagged_in_checkpoint": ea13_post_precal_flag_frac < 0.50,
        "ea13_flagged_vis_fraction_in_post_precal": round(ea13_post_precal_flag_frac, 6),
        "must_flag_chan_row_count": int(np.sum(must_flag_2d)),
        "neutral_chan_row_count": int(np.sum(neutral_2d)),
        "clean_chan_row_count": int(np.sum(clean_2d)),
        "golden_recall_on_must_flag": round(golden_recall, 6),
        "golden_false_positive_on_clean": round(golden_fp_rate, 6),
    }


def measure_j1331_corrected_flatness(ms_path: Path) -> dict[str, float]:
    """Measure channel phase RMS and cross-channel phase slope on J1331+3030 CORRECTED_DATA."""
    with open_table(ms_path, nomodify=True) as tb:
        field_id = tb.getcol("FIELD_ID")
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        flag = tb.getcol("FLAG")
        corrected = tb.getcol("CORRECTED_DATA")

    with open_table(ms_path / "SPECTRAL_WINDOW", nomodify=True) as tb_spw:
        freqs_mhz = tb_spw.getcol("CHAN_FREQ")[:, 0] / 1e6

    j1331_cross = (field_id == 0) & (ant1 != ant2)
    corr_sub = corrected[:, :, j1331_cross]
    flag_sub = flag[:, :, j1331_cross]

    # Parallel hands RR (0) and LL (3)
    vis_list: list[np.ndarray] = []
    chan_sums = np.zeros(len(freqs_mhz), dtype=np.complex128)
    chan_counts = np.zeros(len(freqs_mhz), dtype=np.int64)
    per_row_chan_rms: list[float] = []

    for pol_idx in (0, 3):
        v = corr_sub[pol_idx]
        f = flag_sub[pol_idx]
        good = ~f
        if np.any(good):
            vis_list.append(v[good])
        for ch in range(len(freqs_mhz)):
            g_ch = good[ch]
            if np.any(g_ch):
                unit_phasors = np.exp(1j * np.angle(v[ch, g_ch]))
                chan_sums[ch] += np.sum(unit_phasors)
                chan_counts[ch] += int(np.sum(g_ch))
        # Per-integration, per-baseline channel phase std across unflagged channels
        valid_rows = np.sum(good, axis=0) >= 16
        for r_idx in np.flatnonzero(valid_rows):
            ph_deg = np.rad2deg(np.angle(v[good[:, r_idx], r_idx]))
            # Circular wrap around mean phasor
            mean_ph = np.angle(np.mean(np.exp(1j * np.deg2rad(ph_deg))))
            wrapped_deg = np.rad2deg(np.angle(np.exp(1j * (np.deg2rad(ph_deg) - mean_ph))))
            per_row_chan_rms.append(float(np.sqrt(np.mean(wrapped_deg**2))))

    all_vis = np.concatenate(vis_list)
    all_ph_deg = np.rad2deg(np.angle(all_vis))
    visibility_phase_rms_deg = float(np.sqrt(np.mean(all_ph_deg**2)))
    median_row_chan_phase_rms_deg = float(np.median(per_row_chan_rms))

    valid_ch = chan_counts > 0
    coherent_ch_phase_deg = np.unwrap(np.angle(chan_sums[valid_ch])) * (180.0 / math.pi)
    coherent_ch_phase_deg = coherent_ch_phase_deg - np.mean(coherent_ch_phase_deg)
    coherent_channel_phase_rms_deg = float(np.sqrt(np.mean(coherent_ch_phase_deg**2)))

    slope_per_mhz, _ = np.polyfit(freqs_mhz[valid_ch], coherent_ch_phase_deg, 1)
    phase_slope_deg_per_100mhz = float(abs(slope_per_mhz) * 100.0)

    return {
        "visibility_phase_rms_deg": round(visibility_phase_rms_deg, 4),
        "median_baseline_channel_phase_rms_deg": round(median_row_chan_phase_rms_deg, 4),
        "coherent_channel_phase_rms_deg": round(coherent_channel_phase_rms_deg, 4),
        "phase_slope_deg_per_100mhz": round(phase_slope_deg_per_100mhz, 4),
    }


def build_post_calibrate_checkpoint(
    post_precal_ms: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Build golden post-calibrate checkpoint (`post_calibrate/input.ms` + `caltables/*`)."""
    post_cal_dir = out_dir / "post_calibrate"
    reset_dir(post_cal_dir)
    cal_ms = post_cal_dir / "input.ms"
    caltables_dir = post_cal_dir / "caltables"
    caltables_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(post_precal_ms, cal_ms)
    make_tree_writable(cal_ms)

    # 1. Flag dead antenna ea13
    casatasks.flagdata(
        vis=str(cal_ms),
        mode="manual",
        antenna=DEAD_ANTENNA_NAME,
        flagbackup=False,
    )

    g0_cal = caltables_dir / "G0.cal"
    k_cal = caltables_dir / "K.cal"
    b_cal = caltables_dir / "B.cal"
    g_cal = caltables_dir / "G.cal"
    flux_cal = caltables_dir / "fluxscale.cal"

    # 2. G0: integration-level phase on center channels for calibrators
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(g0_cal),
        field=f"{FLUX_CAL_FIELD},{PHASE_CAL_FIELD},{POL_CAL_FIELD}",
        spw="0:26~37",
        solint="int",
        refant=REFANT_NAME,
        calmode="p",
        minsnr=3.0,
    )

    # 3. K: single-band antenna delays on J1331+3030
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(k_cal),
        field=FLUX_CAL_FIELD,
        solint="inf",
        combine="scan",
        refant=REFANT_NAME,
        gaintype="K",
        gaintable=[str(g0_cal)],
        gainfield=[FLUX_CAL_FIELD],
        minsnr=3.0,
    )

    # 4. B: normalized complex bandpass on J1331+3030
    casatasks.bandpass(
        vis=str(cal_ms),
        caltable=str(b_cal),
        field=FLUX_CAL_FIELD,
        solint="inf",
        combine="scan",
        refant=REFANT_NAME,
        solnorm=True,
        bandtype="B",
        gaintable=[str(g0_cal), str(k_cal)],
        gainfield=[FLUX_CAL_FIELD, FLUX_CAL_FIELD],
        minsnr=3.0,
    )

    # 5. G: scan-averaged complex amplitude+phase gains on all calibrators
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(g_cal),
        field=f"{FLUX_CAL_FIELD},{PHASE_CAL_FIELD},{POL_CAL_FIELD}",
        spw="0:4~59",
        solint="inf",
        refant=REFANT_NAME,
        calmode="ap",
        gaintable=[str(k_cal), str(b_cal)],
        gainfield=[FLUX_CAL_FIELD, FLUX_CAL_FIELD],
        minsnr=3.0,
    )

    # 6. fluxscale: bootstrap flux scale from J1331+3030 to J1822-0938 and J0319+4130
    flux_res = casatasks.fluxscale(
        vis=str(cal_ms),
        caltable=str(g_cal),
        fluxtable=str(flux_cal),
        reference=[FLUX_CAL_FIELD],
        transfer=[PHASE_CAL_FIELD, POL_CAL_FIELD],
    )
    j1822_flux_jy = float(flux_res["1"]["0"]["fluxd"][0])
    j1822_flux_err_jy = float(flux_res["1"]["0"]["fluxdErr"][0])
    j0319_flux_jy = float(flux_res["9"]["0"]["fluxd"][0])

    # 7. applycal to calibrators and mosaic target fields 2..8
    casatasks.applycal(
        vis=str(cal_ms),
        field=FLUX_CAL_FIELD,
        gaintable=[str(k_cal), str(b_cal), str(flux_cal)],
        gainfield=[FLUX_CAL_FIELD, FLUX_CAL_FIELD, FLUX_CAL_FIELD],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(cal_ms),
        field=PHASE_CAL_FIELD,
        gaintable=[str(k_cal), str(b_cal), str(flux_cal)],
        gainfield=[FLUX_CAL_FIELD, FLUX_CAL_FIELD, PHASE_CAL_FIELD],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(cal_ms),
        field=POL_CAL_FIELD,
        gaintable=[str(k_cal), str(b_cal), str(flux_cal)],
        gainfield=[FLUX_CAL_FIELD, FLUX_CAL_FIELD, POL_CAL_FIELD],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(cal_ms),
        field=TARGET_FIELD_SEL,
        gaintable=[str(k_cal), str(b_cal), str(flux_cal)],
        gainfield=[FLUX_CAL_FIELD, FLUX_CAL_FIELD, PHASE_CAL_FIELD],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )

    # 8. Mild post-cal rflag on target CORRECTED_DATA
    casatasks.flagdata(
        vis=str(cal_ms),
        mode="rflag",
        field=TARGET_FIELD_SEL,
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )

    sanitize_ms_history(cal_ms, keep_rows=ARCHIVAL_HISTORY_ROWS)

    with open_table(cal_ms, nomodify=True) as tb:
        has_corrected = "CORRECTED_DATA" in tb.colnames()
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        field_id = tb.getcol("FIELD_ID")
        flag = tb.getcol("FLAG")
        flag_row = tb.getcol("FLAG_ROW")

    ea13_mask = (ant1 == DEAD_ANTENNA_ID) | (ant2 == DEAD_ANTENNA_ID)
    ea13_flagged_fraction = float(np.mean(flag[:, :, ea13_mask]))

    target_rows = np.isin(field_id, TARGET_FIELD_IDS) & (ant1 != ant2)
    target_row_fully_flagged = flag_row[target_rows] | np.all(flag[:, :, target_rows], axis=(0, 1))
    target_unflagged_row_fraction = float(1.0 - np.mean(target_row_fully_flagged))
    target_unflagged_vis_fraction = float(1.0 - np.mean(flag[:, :, target_rows]))

    flatness = measure_j1331_corrected_flatness(cal_ms)
    for tbl_path in (cal_ms, g0_cal, k_cal, b_cal, g_cal, flux_cal):
        assert_unlocked(tbl_path)

    return {
        "post_calibrate_ms": str(cal_ms),
        "caltables_dir": str(caltables_dir),
        "caltables": {
            "G0": str(g0_cal),
            "K": str(k_cal),
            "B": str(b_cal),
            "G": str(g_cal),
            "fluxscale": str(flux_cal),
        },
        "has_corrected_data": has_corrected,
        "ea13_flagged_fraction": round(ea13_flagged_fraction, 6),
        "target_retention_excludes_dead_antenna": False,
        "target_retention_excludes_autocorr": True,
        "target_unflagged_row_fraction": round(target_unflagged_row_fraction, 6),
        "target_unflagged_vis_fraction": round(target_unflagged_vis_fraction, 6),
        "j1822_fluxscale_jy": round(j1822_flux_jy, 6),
        "j1822_fluxscale_err_jy": round(j1822_flux_err_jy, 6),
        "j1822_expected_flux_jy": EXPECTED_J1822_FLUX_JY,
        "j1822_rel_error": round(
            abs(j1822_flux_jy - EXPECTED_J1822_FLUX_JY) / EXPECTED_J1822_FLUX_JY, 6
        ),
        "j0319_fluxscale_jy": round(j0319_flux_jy, 6),
        **flatness,
    }


def build_reference_mosaic_image(
    post_calibrate_ms: Path,
    out_dir: Path,
    phasecenter: str,
) -> dict[str, Any]:
    """Run deterministic multiscale mosaic tclean (`pbcor=True`) and measure image quality."""
    ref_img_dir = out_dir / "reference_image"
    reset_dir(ref_img_dir)
    img_prefix = ref_img_dir / "3c391_mosaic"

    cell_arcsec = 2.5
    imsize = 640
    scales = [0, 5, 15, 30]
    smallscalebias = 0.6
    niter = 18000
    cycleniter = 4500
    nmajor = 12
    threshold = "0.30mJy"
    usemask = "auto-multithresh"
    pbmask = 0.2
    sidelobethreshold = 1.25
    noisethreshold = 2.75
    lownoisethreshold = 1.35
    negativethreshold = 0.0
    minbeamfrac = 0.1
    growiterations = 25
    dogrowprune = False

    casatasks.tclean(
        vis=str(post_calibrate_ms),
        imagename=str(img_prefix),
        field=TARGET_FIELD_SEL,
        spw="0:4~59",
        phasecenter=phasecenter,
        specmode="mfs",
        gridder="mosaic",
        deconvolver="multiscale",
        scales=scales,
        smallscalebias=smallscalebias,
        imsize=[imsize, imsize],
        cell=[f"{cell_arcsec}arcsec", f"{cell_arcsec}arcsec"],
        weighting="briggs",
        robust=0.5,
        pblimit=PB_CUTOFF,
        niter=niter,
        cycleniter=cycleniter,
        nmajor=nmajor,
        threshold=threshold,
        usemask=usemask,
        pbmask=pbmask,
        sidelobethreshold=sidelobethreshold,
        noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold,
        negativethreshold=negativethreshold,
        minbeamfrac=minbeamfrac,
        growiterations=growiterations,
        dogrowprune=dogrowprune,
        pbcor=True,
        interactive=False,
    )

    sanitize_ms_history(post_calibrate_ms, keep_rows=ARCHIVAL_HISTORY_ROWS)

    with open_image(f"{img_prefix}.image") as ia_img:
        img_arr = np.squeeze(ia_img.getchunk())
        beam = ia_img.restoringbeam()

    if "beams" in beam:
        beam = beam["beams"]["*0"]["*0"]

    with open_image(f"{img_prefix}.pb") as ia_pb:
        pb_arr = np.squeeze(ia_pb.getchunk())

    with open_image(f"{img_prefix}.image.pbcor") as ia_pbcor:
        pbcor_arr = np.squeeze(ia_pbcor.getchunk())
        pbcor_mask = np.squeeze(ia_pbcor.getchunk(getmask=True))

    with open_image(f"{img_prefix}.residual") as ia_res:
        res_arr = np.squeeze(ia_res.getchunk())

    with open_image(f"{img_prefix}.model") as ia_mod:
        mod_arr = np.squeeze(ia_mod.getchunk())

    qa = casatools.quanta()
    try:
        b_major_arcsec = float(qa.convert(beam["major"], "arcsec")["value"])
        b_minor_arcsec = float(qa.convert(beam["minor"], "arcsec")["value"])
        b_pa_deg = float(qa.convert(beam["positionangle"], "deg")["value"])
    finally:
        with contextlib.suppress(Exception):
            qa.done()

    pixels_per_beam = b_minor_arcsec / cell_arcsec

    pb_valid = (
        np.isfinite(pb_arr) & (pb_arr >= PB_CUTOFF) & np.isfinite(res_arr) & np.isfinite(img_arr)
    )

    # Off-source mask inside pb >= 0.2: outside 3C391 SNR shell (> 220 arcsec from image center)
    # AND outside clean model support.
    nx, ny = img_arr.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx)
    radius_arcsec = np.hypot(xx, yy) * cell_arcsec

    off_source_mask = pb_valid & (radius_arcsec >= 220.0) & (np.abs(mod_arr) == 0.0)
    if int(np.sum(off_source_mask)) < 1000:
        off_source_mask = pb_valid & (np.abs(mod_arr) == 0.0)

    off_source_res = res_arr[off_source_mask]
    off_source_rms_jy = float(np.sqrt(np.mean(off_source_res**2)))
    off_source_mad_jy = float(
        1.4826 * np.median(np.abs(off_source_res - np.median(off_source_res)))
    )
    all_pb02_res_rms_jy = float(np.sqrt(np.mean(res_arr[pb_valid] ** 2)))

    peak_image_jy = float(np.nanmax(img_arr[pb_valid]))
    valid_pbcor = pbcor_mask & pb_valid & np.isfinite(pbcor_arr)
    peak_pbcor_jy = float(np.nanmax(pbcor_arr[valid_pbcor]))

    dynamic_range = peak_image_jy / off_source_rms_jy
    dynamic_range_pbcor = peak_pbcor_jy / off_source_rms_jy

    pos_mod_sum = float(np.sum(mod_arr[mod_arr > 0.0]))
    neg_mod_abs_sum = float(np.abs(np.sum(mod_arr[mod_arr < 0.0])))
    neg_bowl_ratio = neg_mod_abs_sum / pos_mod_sum if pos_mod_sum > 0.0 else 1.0

    assert_unlocked(post_calibrate_ms)
    for suffix in ("image", "image.pbcor", "pb", "residual", "model"):
        assert_unlocked(Path(f"{img_prefix}.{suffix}"))

    return {
        "reference_image_dir": str(ref_img_dir),
        "image_prefix": str(img_prefix),
        "pbcor_image": f"{img_prefix}.image.pbcor",
        "phasecenter": phasecenter,
        "imsize": [imsize, imsize],
        "cell_arcsec": cell_arcsec,
        "scales": scales,
        "smallscalebias": smallscalebias,
        "niter": niter,
        "cycleniter": cycleniter,
        "nmajor": nmajor,
        "threshold": threshold,
        "usemask": usemask,
        "pbmask": pbmask,
        "sidelobethreshold": sidelobethreshold,
        "noisethreshold": noisethreshold,
        "lownoisethreshold": lownoisethreshold,
        "negativethreshold": negativethreshold,
        "minbeamfrac": minbeamfrac,
        "growiterations": growiterations,
        "dogrowprune": dogrowprune,
        "pblimit": PB_CUTOFF,
        "pbcor": True,
        "b_major_arcsec": round(b_major_arcsec, 4),
        "b_minor_arcsec": round(b_minor_arcsec, 4),
        "b_pa_deg": round(b_pa_deg, 4),
        "pixels_per_beam": round(pixels_per_beam, 4),
        "peak_image_jy_per_beam": round(peak_image_jy, 6),
        "peak_pbcor_jy_per_beam": round(peak_pbcor_jy, 6),
        "off_source_pixel_count": int(np.sum(off_source_mask)),
        "off_source_residual_rms_jy_per_beam": round(off_source_rms_jy, 8),
        "off_source_residual_rms_mjy_per_beam": round(off_source_rms_jy * 1e3, 5),
        "off_source_residual_mad_mjy_per_beam": round(off_source_mad_jy * 1e3, 5),
        "all_pb02_residual_rms_mjy_per_beam": round(all_pb02_res_rms_jy * 1e3, 5),
        "dynamic_range": round(dynamic_range, 2),
        "dynamic_range_pbcor_over_flat_rms": round(dynamic_range_pbcor, 2),
        "positive_model_flux_jy": round(pos_mod_sum, 4),
        "negative_model_flux_abs_jy": round(neg_mod_abs_sum, 6),
        "negative_model_bowl_ratio": round(neg_bowl_ratio, 6),
    }


def _strip_host_paths_for_task(obj: Any, out_dir: Path) -> Any:
    """Convert host checkpoint paths to relative labels inside per-task ground_truth.json."""
    out_prefix = str(out_dir.resolve()) + "/"
    out_raw = str(out_dir) + "/"
    if isinstance(obj, str):
        if obj.startswith(out_prefix):
            return obj[len(out_prefix) :]
        if obj.startswith(out_raw):
            return obj[len(out_raw) :]
        return obj
    if isinstance(obj, list):
        return [_strip_host_paths_for_task(x, out_dir) for x in obj]
    if isinstance(obj, dict):
        return {
            k: _strip_host_paths_for_task(v, out_dir)
            for k, v in obj.items()
            if k not in ("staged_from", "source_ms_path", "masks_npz", "masks_npz_path")
        }
    return obj


def validate_spec_thresholds(
    obs_info: dict[str, Any],
    precal_info: dict[str, Any],
    cal_info: dict[str, Any],
    img_info: dict[str, Any],
) -> None:
    """Assert every golden checkpoint satisfies SPEC.md Section 4 tolerances with headroom."""
    j1331_expected_jy = 7.60
    j1331_rel_err = (
        abs(precal_info["j1331_model_parallel_mean_jy"] - j1331_expected_jy) / j1331_expected_jy
    )
    checks: list[tuple[str, bool, Any]] = [
        ("n_rows == 845379", obs_info["n_rows"] == 845379, obs_info["n_rows"]),
        ("n_scans == 103", obs_info["n_scans"] == 103, obs_info["n_scans"]),
        ("n_antennas == 26", obs_info["n_antennas"] == 26, obs_info["n_antennas"]),
        (
            "All intents missing in raw 3C391 MS",
            bool(obs_info["all_intents_missing_in_raw"]),
            obs_info["all_intents_missing_in_raw"],
        ),
        (
            f"J1331 MODEL_DATA within +-{MODEL_AMP_REL_TOL * 100:.1f}% of {j1331_expected_jy} Jy",
            j1331_rel_err <= MODEL_AMP_REL_TOL,
            precal_info["j1331_model_parallel_mean_jy"],
        ),
        (
            f"Golden precal recall >= {FLAG_RECALL_MIN}",
            precal_info["golden_recall_on_must_flag"] >= FLAG_RECALL_MIN,
            precal_info["golden_recall_on_must_flag"],
        ),
        (
            f"Golden precal clean false-positive < {CLEAN_FALSE_POSITIVE_MAX}",
            precal_info["golden_false_positive_on_clean"] < CLEAN_FALSE_POSITIVE_MAX,
            precal_info["golden_false_positive_on_clean"],
        ),
        (
            "ea13 left unflagged in post_precal checkpoint",
            bool(precal_info["ea13_left_unflagged_in_checkpoint"]),
            precal_info["ea13_flagged_vis_fraction_in_post_precal"],
        ),
        (
            "CORRECTED_DATA present in post_calibrate",
            bool(cal_info["has_corrected_data"]),
            cal_info["has_corrected_data"],
        ),
        (
            f"ea13 flagged fraction >= {DEAD_ANTENNA_FLAGGED_MIN}",
            cal_info["ea13_flagged_fraction"] >= DEAD_ANTENNA_FLAGGED_MIN,
            cal_info["ea13_flagged_fraction"],
        ),
        (
            f"Target unflagged row fraction in [{TARGET_UNFLAGGED_ROWS_MIN}, {TARGET_UNFLAGGED_ROWS_MAX}]",
            TARGET_UNFLAGGED_ROWS_MIN
            <= cal_info["target_unflagged_row_fraction"]
            <= TARGET_UNFLAGGED_ROWS_MAX,
            cal_info["target_unflagged_row_fraction"],
        ),
        (
            f"Target unflagged vis fraction in [{TARGET_UNFLAGGED_ROWS_MIN}, {TARGET_UNFLAGGED_ROWS_MAX}]",
            TARGET_UNFLAGGED_ROWS_MIN
            <= cal_info["target_unflagged_vis_fraction"]
            <= TARGET_UNFLAGGED_ROWS_MAX,
            cal_info["target_unflagged_vis_fraction"],
        ),
        (
            f"J1331 visibility phase RMS < {J1331_CHANNEL_PHASE_RMS_MAX_DEG} deg",
            cal_info["visibility_phase_rms_deg"] < J1331_CHANNEL_PHASE_RMS_MAX_DEG,
            cal_info["visibility_phase_rms_deg"],
        ),
        (
            f"J1331 baseline channel phase RMS < {J1331_CHANNEL_PHASE_RMS_MAX_DEG} deg",
            cal_info["median_baseline_channel_phase_rms_deg"] < J1331_CHANNEL_PHASE_RMS_MAX_DEG,
            cal_info["median_baseline_channel_phase_rms_deg"],
        ),
        (
            f"J1822 fluxscale within +-{J1822_FLUX_REL_TOL * 100:.1f}% of {EXPECTED_J1822_FLUX_JY} Jy",
            cal_info["j1822_rel_error"] <= J1822_FLUX_REL_TOL,
            cal_info["j1822_fluxscale_jy"],
        ),
        (
            f"Beam sampling b_minor/cell in [{PIXELS_PER_BEAM_MIN}, {PIXELS_PER_BEAM_MAX}]",
            PIXELS_PER_BEAM_MIN <= img_info["pixels_per_beam"] <= PIXELS_PER_BEAM_MAX,
            img_info["pixels_per_beam"],
        ),
        (
            f"Pre-selfcal off-source residual RMS < {OFF_SOURCE_RESIDUAL_RMS_MAX_MJY} mJy/beam "
            f"(golden headroom < {GOLDEN_OFF_SOURCE_RESIDUAL_RMS_MAX_MJY} mJy/beam)",
            0.0
            < img_info["off_source_residual_rms_mjy_per_beam"]
            < min(OFF_SOURCE_RESIDUAL_RMS_MAX_MJY, GOLDEN_OFF_SOURCE_RESIDUAL_RMS_MAX_MJY)
            and img_info["off_source_residual_rms_jy_per_beam"] < OFF_SOURCE_RESIDUAL_RMS_MAX_JY,
            img_info["off_source_residual_rms_mjy_per_beam"],
        ),
        (
            f"Pre-selfcal dynamic range >= {DYNAMIC_RANGE_MIN} & >= {VERIFIER_TCLEAN_DR_MIN} "
            f"(golden headroom >= {GOLDEN_DYNAMIC_RANGE_MIN})",
            img_info["dynamic_range"]
            >= max(VERIFIER_TCLEAN_DR_MIN, DYNAMIC_RANGE_MIN, GOLDEN_DYNAMIC_RANGE_MIN),
            img_info["dynamic_range"],
        ),
        (
            f"Negative model bowl ratio < {NEGATIVE_MODEL_BOWL_RATIO_MAX} "
            f"(golden headroom <= {GOLDEN_NEGATIVE_MODEL_BOWL_RATIO_MAX})",
            img_info["negative_model_bowl_ratio"]
            <= min(NEGATIVE_MODEL_BOWL_RATIO_MAX, GOLDEN_NEGATIVE_MODEL_BOWL_RATIO_MAX),
            img_info["negative_model_bowl_ratio"],
        ),
    ]
    failures = [f"{label} (got {val!r})" for label, ok, val in checks if not ok]
    if failures:
        raise RuntimeError(
            "3C391 checkpoint SPEC validation failed:\n  - " + "\n  - ".join(failures)
        )


def write_ground_truth_files(
    out_dir: Path,
    source_ms: Path,
    raw_ms: Path,
    obs_info: dict[str, Any],
    precal_info: dict[str, Any],
    cal_info: dict[str, Any],
    img_info: dict[str, Any],
    timings_sec: dict[str, float],
) -> None:
    """Write master ground_truth.json, per-task ground_truth.json files, and staged .npz masks."""
    j1331_el_min = next(f["el_min_deg"] for f in obs_info["fields"] if f["field_id"] == 0)
    j1822_el_min = next(f["el_min_deg"] for f in obs_info["fields"] if f["field_id"] == 1)
    j0319_el_min = next(f["el_min_deg"] for f in obs_info["fields"] if f["field_id"] == 9)
    mosaic_names = [f["name"] for f in obs_info["fields"] if f["field_id"] in TARGET_FIELD_IDS]

    clean_obs_info = _strip_host_paths_for_task(obs_info, out_dir)
    clean_precal_info = _strip_host_paths_for_task(precal_info, out_dir)
    clean_cal_info = _strip_host_paths_for_task(cal_info, out_dir)
    clean_img_info = _strip_host_paths_for_task(img_info, out_dir)

    inspect_gt = {
        "task_id": "r1_3c391_inspect",
        "suite": "R",
        "pass_reward_min": PASS_REWARD_MIN,
        "all_intents_missing_in_raw": True,
        "intent_delimiters": list(INTENT_DELIMITERS),
        "flux_calibrator": FLUX_CAL_FIELD,
        "bandpass_calibrator": FLUX_CAL_FIELD,
        "phase_calibrator": PHASE_CAL_FIELD,
        "phase_calibrator_missing_intent": True,
        "pol_leakage_calibrator": POL_CAL_FIELD,
        "pol_leakage_single_scan": 103,
        "mosaic_target_fields": mosaic_names,
        "mosaic_target_field_ids": TARGET_FIELD_IDS,
        "dead_antenna": DEAD_ANTENNA_NAME,
        "dead_antenna_id": DEAD_ANTENNA_ID,
        "missing_vla_pads": obs_info["missing_vla_pads"],
        "bandpass_min_elevation_deg": j1331_el_min,
        "phase_cal_min_elevation_deg": j1822_el_min,
        "pol_cal_min_elevation_deg": j0319_el_min,
        "observation": clean_obs_info,
        "weights": {
            "valid_json": 0.20,
            "flux_bandpass_calibrator": 0.20,
            "phase_vs_target_separation": 0.25,
            "dead_antenna": 0.15,
            "bandpass_min_elevation": 0.20,
        },
        "tolerances": {
            "elevation_abs_deg": ELEVATION_ABS_TOL_DEG,
            "pass_reward_min": PASS_REWARD_MIN,
        },
    }

    precal_gt = {
        "task_id": "r1_3c391_precal",
        "suite": "R",
        "pass_reward_min": PASS_REWARD_MIN,
        "precal_masks_file": "precal_masks.npz",
        "all_intents_missing_in_raw": True,
        "intent_delimiters": list(INTENT_DELIMITERS),
        "flux_calibrator": FLUX_CAL_FIELD,
        "flux_calibrator_field_id": 0,
        "phase_calibrator": PHASE_CAL_FIELD,
        "phase_calibrator_field_id": 1,
        "calibrator_field_ids": [0, 1, 9],
        "dead_antenna": DEAD_ANTENNA_NAME,
        "dead_antenna_id": DEAD_ANTENNA_ID,
        "dead_antenna_policy": "neutral_allowed",
        "j1331_model_parallel_mean_jy": precal_info["j1331_model_parallel_mean_jy"],
        "j1331_model_parallel_per_chan_jy": precal_info["j1331_model_parallel_per_chan_jy"],
        "precal_reference": clean_precal_info,
        "weights": {
            "calibrators_ms_valid": 0.20,
            "model_data_accuracy": 0.25,
            "rfi_preflag_recall": 0.30,
            "clean_false_positive_guard": 0.25,
        },
        "tolerances": {
            "model_amp_rel_tol": MODEL_AMP_REL_TOL,
            "flag_recall_min": FLAG_RECALL_MIN,
            "clean_false_positive_max": CLEAN_FALSE_POSITIVE_MAX,
            "pass_reward_min": PASS_REWARD_MIN,
        },
    }

    verifier_tclean_recipe = {
        "field": TARGET_FIELD_SEL,
        "spw": "0:4~59",
        "phasecenter": img_info["phasecenter"],
        "specmode": "mfs",
        "gridder": "mosaic",
        "deconvolver": "multiscale",
        "scales": list(img_info["scales"]),
        "smallscalebias": img_info["smallscalebias"],
        "imsize": list(img_info["imsize"]),
        "cell_arcsec": img_info["cell_arcsec"],
        "weighting": "briggs",
        "robust": 0.5,
        "pblimit": PB_CUTOFF,
        "usemask": img_info["usemask"],
        "pbmask": img_info["pbmask"],
        "sidelobethreshold": img_info["sidelobethreshold"],
        "noisethreshold": img_info["noisethreshold"],
        "lownoisethreshold": img_info["lownoisethreshold"],
        "negativethreshold": img_info["negativethreshold"],
        "minbeamfrac": img_info["minbeamfrac"],
        "growiterations": img_info["growiterations"],
        "dogrowprune": img_info["dogrowprune"],
        "niter": img_info["niter"],
        "cycleniter": img_info["cycleniter"],
        "nmajor": img_info["nmajor"],
        "threshold": img_info["threshold"],
        "pbcor": True,
        "off_source_min_radius_arcsec": 220.0,
    }

    calibrate_gt = {
        "task_id": "r1_3c391_calibrate",
        "suite": "R",
        "pass_reward_min": PASS_REWARD_MIN,
        "target_retention_excludes_dead_antenna": False,
        "target_retention_excludes_autocorr": True,
        "intent_delimiters": list(INTENT_DELIMITERS),
        "flux_calibrator": FLUX_CAL_FIELD,
        "flux_calibrator_field_id": 0,
        "phase_calibrator": PHASE_CAL_FIELD,
        "phase_calibrator_field_id": 1,
        "target_field_ids": TARGET_FIELD_IDS,
        "dead_antenna": DEAD_ANTENNA_NAME,
        "dead_antenna_id": DEAD_ANTENNA_ID,
        "refant": REFANT_NAME,
        "j1822_expected_flux_jy": EXPECTED_J1822_FLUX_JY,
        "calibrate_reference": clean_cal_info,
        "verifier_tclean_recipe": verifier_tclean_recipe,
        "reference_image_summary": clean_img_info,
        "weights": {
            "corrected_dead_ant_target_retention": 0.20,
            "bandpass_delay_flatness": 0.25,
            "fluxscale_accuracy": 0.25,
            "target_verifier_tclean_dr": 0.30,
        },
        "tolerances": {
            "target_unflagged_rows_min": TARGET_UNFLAGGED_ROWS_MIN,
            "target_unflagged_rows_max": TARGET_UNFLAGGED_ROWS_MAX,
            "dead_antenna_flagged_min": DEAD_ANTENNA_FLAGGED_MIN,
            "j1331_channel_phase_rms_max_deg": J1331_CHANNEL_PHASE_RMS_MAX_DEG,
            "j1822_flux_rel_tol": J1822_FLUX_REL_TOL,
            "verifier_tclean_dr_min": VERIFIER_TCLEAN_DR_MIN,
            "pass_reward_min": PASS_REWARD_MIN,
        },
    }

    image_gt = {
        "task_id": "r1_3c391_image",
        "suite": "R",
        "pass_reward_min": PASS_REWARD_MIN,
        "target_field_ids": TARGET_FIELD_IDS,
        "phasecenter": img_info["phasecenter"],
        "off_source_min_radius_arcsec": 220.0,
        "image_reference": clean_img_info,
        "weights": {
            "products_valid": 0.15,
            "beam_sampling": 0.20,
            "off_source_residual_rms": 0.25,
            "dynamic_range_and_bowl_guard": 0.40,
        },
        "tolerances": {
            "pixels_per_beam_min": PIXELS_PER_BEAM_MIN,
            "pixels_per_beam_max": PIXELS_PER_BEAM_MAX,
            "pb_cutoff": PB_CUTOFF,
            "off_source_residual_rms_max_mjy_per_beam": OFF_SOURCE_RESIDUAL_RMS_MAX_MJY,
            "off_source_residual_rms_max_jy_per_beam": OFF_SOURCE_RESIDUAL_RMS_MAX_JY,
            "dynamic_range_min": DYNAMIC_RANGE_MIN,
            "negative_model_bowl_ratio_max": NEGATIVE_MODEL_BOWL_RATIO_MAX,
            "pass_reward_min": PASS_REWARD_MIN,
        },
    }

    e2e_gt = {
        "task_id": "r2_3c391_e2e",
        "suite": "R",
        "pass_reward_min": PASS_REWARD_MIN,
        "precal_masks_file": "precal_masks.npz",
        "target_retention_excludes_dead_antenna": False,
        "target_retention_excludes_autocorr": True,
        "all_intents_missing_in_raw": True,
        "intent_delimiters": list(INTENT_DELIMITERS),
        "precal_ground_truth": precal_gt,
        "calibrate_ground_truth": calibrate_gt,
        "image_ground_truth": image_gt,
        "weights": {
            "calibration_score": 0.60,
            "delivered_image_score": 0.40,
        },
        "tolerances": {
            **precal_gt["tolerances"],
            **calibrate_gt["tolerances"],
            **image_gt["tolerances"],
            "pass_reward_min": PASS_REWARD_MIN,
        },
    }

    master_gt = {
        "suite": "R",
        "dataset": "3c391_ctm_mosaic_10s_spw0",
        "source_ms_path": str(source_ms),
        "pass_reward_min": PASS_REWARD_MIN,
        "target_retention_excludes_dead_antenna": False,
        "target_retention_excludes_autocorr": True,
        "all_intents_missing_in_raw": True,
        "intent_delimiters": list(INTENT_DELIMITERS),
        "checkpoints": {
            "raw_ms": str(raw_ms),
            "post_precal_ms": precal_info["post_precal_ms"],
            "post_precal_calibrators_ms": precal_info["calibrators_ms"],
            "post_precal_masks_npz": precal_info["masks_npz"],
            "post_calibrate_ms": cal_info["post_calibrate_ms"],
            "caltables_dir": cal_info["caltables_dir"],
            "reference_image_dir": img_info["reference_image_dir"],
            "reference_image_prefix": img_info["image_prefix"],
        },
        "observation": obs_info,
        "post_precal": precal_info,
        "post_calibrate": cal_info,
        "reference_image": img_info,
        "tasks": {
            "r1_3c391_inspect": inspect_gt,
            "r1_3c391_precal": precal_gt,
            "r1_3c391_calibrate": calibrate_gt,
            "r1_3c391_image": image_gt,
            "r2_3c391_e2e": e2e_gt,
        },
        "runtime_seconds": timings_sec,
    }

    masks_src = Path(precal_info["masks_npz"])
    if not masks_src.is_file():
        raise FileNotFoundError(f"Missing precal_masks.npz sidecar: {masks_src}")

    for mask_task_id in ("r1_3c391_precal", "r2_3c391_e2e"):
        stage_file_atomic(masks_src, out_dir / mask_task_id / "precal_masks.npz")

    write_json_atomic(out_dir / "ground_truth.json", master_gt)
    for task_id, payload in master_gt["tasks"].items():
        task_dir = out_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(task_dir / "ground_truth.json", payload)


def refresh_existing_3c391_checkpoints(
    source_ms: Path | None = None,
    out_dir: Path | None = None,
    *,
    cfg: BenchConfig | None = None,
) -> dict[str, Any]:
    """Sanitize existing MS HISTORY tables in place, rebuild reference_image, and rewrite JSONs."""
    resolved_cfg = configure_builder_casa_env(cfg)
    if source_ms is None:
        source_ms, _ = resolve_default_3c391_paths(resolved_cfg)
    if out_dir is None:
        out_dir = resolved_cfg.data_dir / "checkpoints" / "3c391"
    make_tree_writable(out_dir)

    raw_ms = out_dir / "raw" / "input.ms"
    post_precal_ms = out_dir / "post_precal" / "input.ms"
    calibrators_ms = out_dir / "post_precal" / "calibrators.ms"
    masks_npz = out_dir / "post_precal" / "precal_masks.npz"
    post_cal_ms = out_dir / "post_calibrate" / "input.ms"
    caltables_dir = out_dir / "post_calibrate" / "caltables"

    for required in (raw_ms, post_precal_ms, calibrators_ms, post_cal_ms, caltables_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing existing checkpoint path: {required}")
    if not masks_npz.is_file():
        raise FileNotFoundError(f"Missing existing precal masks sidecar: {masks_npz}")

    existing_gt_path = out_dir / "ground_truth.json"
    existing_gt = (
        json.loads(existing_gt_path.read_text(encoding="utf-8"))
        if existing_gt_path.is_file()
        else {}
    )
    existing_timings = dict(existing_gt.get("runtime_seconds", {}))
    precal_info = dict(existing_gt["post_precal"])
    cal_info = dict(existing_gt["post_calibrate"])
    cal_info["target_retention_excludes_dead_antenna"] = False
    cal_info["target_retention_excludes_autocorr"] = True

    archive = resolve_source_archive(source_ms)
    staged_source = archive if archive is not None else source_ms

    work_dir = resolved_cfg.tmp_dir / f"refresh_3c391_{os.getpid()}"
    reset_dir(work_dir)
    prev_cwd = Path.cwd()
    t_total0 = time.perf_counter()

    try:
        os.chdir(work_dir)

        t0 = time.perf_counter()
        sanitize_ms_history(raw_ms, keep_rows=ARCHIVAL_HISTORY_ROWS)
        assert_clean_raw_ms(raw_ms)
        sanitize_ms_history(post_precal_ms, keep_rows=ARCHIVAL_HISTORY_ROWS)
        sanitize_ms_history(calibrators_ms, keep_rows=0)
        sanitize_ms_history(post_cal_ms, keep_rows=ARCHIVAL_HISTORY_ROWS)

        obs_info, raw_arrays = inspect_pristine_ms(raw_ms)
        for tbl_path in (
            raw_ms,
            post_precal_ms,
            calibrators_ms,
            post_cal_ms,
            caltables_dir / "G0.cal",
            caltables_dir / "K.cal",
            caltables_dir / "B.cal",
            caltables_dir / "G.cal",
            caltables_dir / "fluxscale.cal",
        ):
            assert_unlocked(tbl_path)
        existing_timings["history_sanitize_and_inspect"] = round(time.perf_counter() - t0, 3)

        _, phasecenter = format_casa_phasecenter_j2000(
            raw_arrays["phase_dirs_rad"], TARGET_FIELD_IDS
        )

        t0 = time.perf_counter()
        img_info = build_reference_mosaic_image(post_cal_ms, out_dir, phasecenter)
        existing_timings["reference_image"] = round(time.perf_counter() - t0, 3)
        existing_timings["refresh_total"] = round(time.perf_counter() - t_total0, 3)

        validate_spec_thresholds(obs_info, precal_info, cal_info, img_info)
        write_ground_truth_files(
            out_dir=out_dir,
            source_ms=staged_source,
            raw_ms=raw_ms,
            obs_info=obs_info,
            precal_info=precal_info,
            cal_info=cal_info,
            img_info=img_info,
            timings_sec=existing_timings,
        )
    finally:
        os.chdir(prev_cwd)
        casatasks.clearstat()
        remove_path(work_dir)

    return json.loads((out_dir / "ground_truth.json").read_text(encoding="utf-8"))


def build_all_3c391_checkpoints(
    source_ms: Path | None = None,
    out_dir: Path | None = None,
    *,
    cfg: BenchConfig | None = None,
) -> dict[str, Any]:
    """Run full deterministic 3C391 checkpoint build and return summary metrics."""
    resolved_cfg = configure_builder_casa_env(cfg)
    if source_ms is None:
        source_ms, _ = resolve_default_3c391_paths(resolved_cfg)
    if out_dir is None:
        out_dir = resolved_cfg.data_dir / "checkpoints" / "3c391"
    archive = resolve_source_archive(source_ms)
    if archive is None and not source_ms.exists():
        raise FileNotFoundError(f"Source archive/MS not found: {source_ms}")

    watched_paths = [p for p in (source_ms, archive) if p is not None and p.exists()]
    mtimes_before = {p: p.stat().st_mtime_ns for p in watched_paths}
    t_total0 = time.perf_counter()
    timings: dict[str, float] = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    make_tree_writable(out_dir)
    work_dir = out_dir / "_work"
    reset_dir(work_dir)
    prev_cwd = Path.cwd()

    try:
        os.chdir(work_dir)

        # Step 1: Stage clean raw/input.ms from untouched .tgz (or state-equivalent fallback .ms)
        t0 = time.perf_counter()
        raw_dir = out_dir / "raw"
        reset_dir(raw_dir)
        raw_ms = raw_dir / "input.ms"
        staged_source = stage_clean_raw_ms(source_ms, raw_ms)
        obs_info, raw_arrays = inspect_pristine_ms(raw_ms)
        assert_unlocked(raw_ms)
        timings["raw_stage_and_inspect"] = round(time.perf_counter() - t0, 3)

        _, phasecenter = format_casa_phasecenter_j2000(
            raw_arrays["phase_dirs_rad"], TARGET_FIELD_IDS
        )

        # Step 2: Golden post-precal checkpoint
        t0 = time.perf_counter()
        precal_info = build_post_precal_checkpoint(raw_ms, out_dir, work_dir, raw_arrays)
        timings["post_precal"] = round(time.perf_counter() - t0, 3)

        # Step 3: Golden post-calibrate checkpoint
        t0 = time.perf_counter()
        cal_info = build_post_calibrate_checkpoint(Path(precal_info["post_precal_ms"]), out_dir)
        timings["post_calibrate"] = round(time.perf_counter() - t0, 3)

        # Step 4: Reference multiscale mosaic image
        t0 = time.perf_counter()
        img_info = build_reference_mosaic_image(
            Path(cal_info["post_calibrate_ms"]), out_dir, phasecenter
        )
        timings["reference_image"] = round(time.perf_counter() - t0, 3)
        timings["total"] = round(time.perf_counter() - t_total0, 3)

        # Step 5: Validate SPEC tolerances BEFORE writing ground truth JSONs
        validate_spec_thresholds(obs_info, precal_info, cal_info, img_info)
        write_ground_truth_files(
            out_dir=out_dir,
            source_ms=staged_source,
            raw_ms=raw_ms,
            obs_info=obs_info,
            precal_info=precal_info,
            cal_info=cal_info,
            img_info=img_info,
            timings_sec=timings,
        )
    finally:
        os.chdir(prev_cwd)
        casatasks.clearstat()
        remove_path(work_dir)

    for p, mtime_before in mtimes_before.items():
        if p.stat().st_mtime_ns != mtime_before:
            raise RuntimeError(f"External source path mtime changed during build: {p}")

    return json.loads((out_dir / "ground_truth.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build golden 3C391 Suite R checkpoints.")
    add_config_cli_args(parser)
    parser.add_argument("--source-ms", type=Path, default=None)
    parser.add_argument("--source-tgz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--refresh-imaging-only",
        action="store_true",
        help="Sanitize existing MS HISTORY tables, rebuild reference_image, and rewrite JSONs.",
    )
    args = parser.parse_args(argv)
    cfg = config_from_cli_args(args)
    default_ms, _ = resolve_default_3c391_paths(cfg)
    eff_source = args.source_tgz or args.source_ms or default_ms

    if args.refresh_imaging_only:
        summary = refresh_existing_3c391_checkpoints(
            source_ms=eff_source, out_dir=args.out_dir, cfg=cfg
        )
    else:
        summary = build_all_3c391_checkpoints(source_ms=eff_source, out_dir=args.out_dir, cfg=cfg)

    print(
        json.dumps(
            {
                "post_precal": {
                    k: v
                    for k, v in summary["post_precal"].items()
                    if k != "j1331_model_parallel_per_chan_jy"
                },
                "post_calibrate": summary["post_calibrate"],
                "reference_image": summary["reference_image"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
