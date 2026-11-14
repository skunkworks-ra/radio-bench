"""Integration and physics verification tests for Suite S synthetic MSs and Suite R 3C391 checkpoints."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Configure CASA and scratch paths inside /var/mnt/fast/radio-bench before importing casatools.
FAST_BENCH_ROOT = Path("/var/mnt/fast/radio-bench")


def _ensure_non_tmp_env(var_name: str, fallback: Path) -> None:
    raw_val = os.environ.get(var_name, "").strip()
    is_under_tmp = False
    if raw_val:
        if raw_val.startswith("/tmp"):
            is_under_tmp = True
        else:
            with contextlib.suppress(OSError):
                is_under_tmp = Path(raw_val).resolve().is_relative_to(Path("/tmp").resolve())
    if not raw_val or is_under_tmp:
        os.environ[var_name] = str(fallback)
    Path(os.environ[var_name]).mkdir(parents=True, exist_ok=True)


_ensure_non_tmp_env("TMPDIR", FAST_BENCH_ROOT / "tmp")
_ensure_non_tmp_env("MPLCONFIGDIR", FAST_BENCH_ROOT / "mplconfig")
os.environ.setdefault("CASASITECONFIG", str(FAST_BENCH_ROOT / "casaconfig_local.py"))
if not os.access(os.environ.get("HOME", "/"), os.W_OK):
    os.environ["HOME"] = str(FAST_BENCH_ROOT / "tmp")

import astropy.units as u  # noqa: E402
import casatasks  # noqa: E402
import casatools  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_builders.sim_common import (  # noqa: E402
    DEAD_ANTENNA_ID as SYNTH_DEAD_ANTENNA_ID,
)
from data_builders.sim_common import (  # noqa: E402
    DEAD_ANTENNA_NAME as SYNTH_DEAD_ANTENNA_NAME,
)
from data_builders.sim_common import (  # noqa: E402
    FLUX_CAL_NAME,
    PHASE_CAL_NAME,
    REFANT_NAME,
    RFI_CHANNELS,
    SCAN_SCHEDULE,
    SPW_CHAN_WIDTH_HZ,
    SPW_NCHAN,
    SPW_START_HZ,
    SYNTH_ANTENNA_NAMES,
    TARGET_P1_NAME,
    TARGET_P2_NAME,
    TRUE_DELAYS_NS,
    compute_theoretical_mosaic_sigma_jy,
    load_vla_d_subset,
)

PRISTINE_3C391_MS = Path("/var/mnt/fast/datasets/3c391_fresh/3c391_ctm_mosaic_10s_spw0.ms")
PRISTINE_3C391_TGZ = Path("/var/mnt/fast/datasets/3c391_fresh/3c391_ctm_mosaic_10s_spw0.ms.tgz")
SYNTH_ROOT = Path("/var/mnt/fast/radio_bench_data/synth")
CHECKPOINTS_3C391_ROOT = Path("/var/mnt/fast/radio_bench_data/checkpoints/3c391")

SUITE_S_TASKS = (
    "s1_inspect",
    "s1_precal",
    "s1_calibrate",
    "s1_image",
    "s2_e2e_synth",
)
SUITE_S_RAW_TASKS = ("s1_inspect", "s1_precal", "s2_e2e_synth")
SUITE_S_FIXED_INTENT_TASKS = ("s1_calibrate", "s1_image")

SUITE_R_TASKS = (
    "r1_3c391_inspect",
    "r1_3c391_precal",
    "r1_3c391_calibrate",
    "r1_3c391_image",
    "r2_3c391_e2e",
)

SUITE_R_MS_PATHS = (
    CHECKPOINTS_3C391_ROOT / "raw" / "input.ms",
    CHECKPOINTS_3C391_ROOT / "post_precal" / "input.ms",
    CHECKPOINTS_3C391_ROOT / "post_precal" / "calibrators.ms",
    CHECKPOINTS_3C391_ROOT / "post_calibrate" / "input.ms",
)

FORBIDDEN_HISTORY_SUBSTRINGS = (
    "Perley-Butler",
    "phase_cal.cl",
    "!ea13",
    "/var/mnt/fast/",
    "cal.K",
    "cal.B",
    "3c391_run",
)

VLA_J1331_CATALOG = SkyCoord("13h31m08.288s", "+30d30m32.959s", frame="icrs")
VLA_J1822_CATALOG = SkyCoord("18h22m28.704s", "-09d38m56.835s", frame="icrs")


@contextlib.contextmanager
def _open_table(path: Path) -> Iterator[Any]:
    """Open a casacore table strictly read-only and guarantee close + done."""
    tb = casatools.table()
    opened = False
    try:
        tb.open(str(path), nomodify=True)
        opened = True
        yield tb
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()


@contextlib.contextmanager
def _query_table(tb: Any, taql: str) -> Iterator[Any]:
    """Run a read-only TaQL query and guarantee close + done on the subtable."""
    sub = tb.query(taql)
    try:
        yield sub
    finally:
        try:
            sub.close()
        finally:
            sub.done()


@contextlib.contextmanager
def _open_msmd(path: Path) -> Iterator[Any]:
    """Open casatools.msmetadata and guarantee close + done."""
    msmd = casatools.msmetadata()
    opened = False
    try:
        msmd.open(str(path))
        opened = True
        yield msmd
    finally:
        try:
            if opened:
                msmd.close()
        finally:
            msmd.done()


@contextlib.contextmanager
def _open_image(path: Path) -> Iterator[Any]:
    """Open casatools.image read-only and guarantee close + done."""
    ia = casatools.image()
    opened = False
    try:
        ia.open(str(path))
        opened = True
        yield ia
    finally:
        try:
            if opened:
                ia.close()
        finally:
            ia.done()


def _assert_unlocked(table_path: Path) -> None:
    """Assert a CASA table/MS/caltable/image directory holds zero active casacore locks."""
    assert table_path.is_dir(), f"Expected CASA table directory to exist: {table_path}"
    casatasks.clearstat()
    with _open_table(table_path) as tb:
        assert not tb.ismultiused(checksubtables=True), (
            f"Active casacore lock detected on {table_path}"
        )


def _assert_no_history_leaks(ms_path: Path) -> None:
    """Assert MS HISTORY subtable contains zero leaked builder or calibration strings."""
    hist_path = ms_path / "HISTORY"
    assert hist_path.is_dir(), f"Missing HISTORY subtable: {hist_path}"
    with _open_table(hist_path) as tb_hist:
        hist_cols = set(tb_hist.colnames())
        hist_nrows = int(tb_hist.nrows())
        texts: list[str] = []
        if hist_nrows > 0:
            for col in ("MESSAGE", "CLI_COMMAND", "APP_PARAMS", "APPLICATION", "ORIGIN"):
                if col in hist_cols:
                    texts.extend(str(x) for x in np.ravel(tb_hist.getcol(col)))
    for bad in FORBIDDEN_HISTORY_SUBSTRINGS:
        leaked = [t for t in texts if bad in t]
        assert not leaked, (
            f"{ms_path}/HISTORY leaked forbidden string {bad!r} "
            f"({len(leaked)} entries; first: {leaked[0]!r})"
        )


def _assert_all_finite(obj: Any, *, path: str = "$") -> None:
    """Recursively verify zero NaN/Inf values in decoded JSON structures."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        assert math.isfinite(obj), f"Non-finite numeric value {obj!r} at {path}"
        return
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_all_finite(item, path=f"{path}[{idx}]")
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            _assert_all_finite(val, path=f"{path}.{key}")
        return
    raise TypeError(f"Unexpected JSON value type {type(obj).__name__} at {path}")


def _load_finite_json(json_path: Path) -> dict[str, Any]:
    """Parse JSON rejecting non-standard NaN/Infinity constants and recursively check finiteness."""
    assert json_path.is_file(), f"Missing JSON file: {json_path}"

    def _reject_constant(token: str) -> Any:
        raise ValueError(f"Non-finite JSON token {token!r} in {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    assert isinstance(payload, dict), f"Expected top-level JSON object in {json_path}"
    _assert_all_finite(payload)
    return payload


def _load_finite_npz(npz_path: Path) -> dict[str, np.ndarray]:
    """Load compressed .npz without pickle and verify all arrays are non-empty and finite."""
    assert npz_path.is_file(), f"Missing .npz sidecar: {npz_path}"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert arrays, f"Empty .npz archive: {npz_path}"
    for key, arr in arrays.items():
        assert arr.size > 0, f"Empty array {key!r} in {npz_path}"
        assert np.all(np.isfinite(arr)), f"Non-finite values in {npz_path}[{key!r}]"
    return arrays


def _quantity_to_arcsec(q: dict[str, Any]) -> float:
    val = float(q["value"])
    unit = str(q["unit"]).lower()
    if unit in ("arcsec", "''", '"'):
        return val
    if unit in ("arcmin", "'"):
        return val * 60.0
    if unit == "deg":
        return val * 3600.0
    if unit == "rad":
        return math.degrees(val) * 3600.0
    raise ValueError(f"Unsupported angular unit: {q}")


class TestSimCommonUnit:
    """Fast unit checks on deterministic Suite S simulation parameters."""

    def test_vla_d_subset_and_schedule_invariants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not os.environ["TMPDIR"].startswith("/tmp")
        assert Path(os.environ["TMPDIR"]).is_dir()

        saved_tmpdir = os.environ["TMPDIR"]
        monkeypatch.setenv("TMPDIR", "/tmp/should_never_be_used")
        _ensure_non_tmp_env("TMPDIR", FAST_BENCH_ROOT / "tmp")
        assert not os.environ["TMPDIR"].startswith("/tmp")
        monkeypatch.setenv("TMPDIR", saved_tmpdir)

        xs, ys, zs, diams, antnames = load_vla_d_subset()
        assert antnames == [f"ea{i:02d}" for i in range(1, 11)]
        assert antnames == list(SYNTH_ANTENNA_NAMES)
        assert len(xs) == len(ys) == len(zs) == len(diams) == 10
        assert all(math.isfinite(v) for coords in (xs, ys, zs) for v in coords)
        assert all(d == pytest.approx(25.0) for d in diams)

        assert SYNTH_DEAD_ANTENNA_NAME == "ea07"
        assert SYNTH_DEAD_ANTENNA_ID == 6
        assert REFANT_NAME == "ea01"
        assert RFI_CHANNELS == [29, 30, 31]

        assert TRUE_DELAYS_NS["ea01"] == 0.0
        assert TRUE_DELAYS_NS["ea02"] == pytest.approx(14.0)
        assert TRUE_DELAYS_NS["ea10"] == pytest.approx(-14.0)
        assert all(-14.0 <= tau <= 14.0 for tau in TRUE_DELAYS_NS.values())

        assert [s.scan_number for s in SCAN_SCHEDULE] == list(range(1, 14))
        pcal_scans = [s for s in SCAN_SCHEDULE if s.field_name == PHASE_CAL_NAME]
        assert [s.scan_number for s in pcal_scans] == [3, 6, 9, 12]
        assert all("CALIBRATE_PHASE" not in s.intent for s in pcal_scans)
        populated_fields = {s.field_name for s in SCAN_SCHEDULE if s.intent.strip()}
        assert len(populated_fields) / 4.0 < 0.50

    def test_theoretical_mosaic_sigma_positive_and_finite(self) -> None:
        info = compute_theoretical_mosaic_sigma_jy(0.0527, include_slew_quack=True)
        for key, val in info.items():
            assert math.isfinite(val) and val > 0.0, f"Invalid {key}={val!r}"
        assert info["clean_baselines"] == 36.0
        assert info["clean_channels"] == 61.0
        assert info["sigma_th_jy_per_beam"] > info["sigma_th_natural_jy_per_beam"]


class TestPristine3C391Source:
    """Read-only non-modification checks on external `/var/mnt/fast/datasets/3c391_fresh/`."""

    def test_pristine_3c391_source_integrity(self) -> None:
        assert PRISTINE_3C391_TGZ.is_file(), f"Missing 3C391 archival tarball: {PRISTINE_3C391_TGZ}"
        assert PRISTINE_3C391_TGZ.stat().st_size > 0, f"Empty tarball: {PRISTINE_3C391_TGZ}"

        assert PRISTINE_3C391_MS.is_dir(), f"Missing external 3C391 MS: {PRISTINE_3C391_MS}"
        _assert_unlocked(PRISTINE_3C391_MS)

        with _open_table(PRISTINE_3C391_MS) as tb:
            n_rows = int(tb.nrows())
            colnames = set(tb.colnames())
            scans = tb.getcol("SCAN_NUMBER")

        assert n_rows == 845_379
        assert "DATA" in colnames
        assert len(np.unique(scans)) == 103

        with _open_table(PRISTINE_3C391_MS / "ANTENNA") as tb_ant:
            ant_names = [str(x) for x in tb_ant.getcol("NAME")]

        assert len(ant_names) == 26
        assert ant_names[10] == "ea13"
        missing_pads = sorted({f"ea{i:02d}" for i in range(1, 29)} - set(ant_names))
        assert missing_pads == ["ea06", "ea10"]

        with _open_table(PRISTINE_3C391_MS / "FIELD") as tb_fld:
            field_names = [str(x) for x in tb_fld.getcol("NAME")]

        assert len(field_names) == 10
        assert field_names[0] == "J1331+3030"
        assert field_names[1] == "J1822-0938"
        assert field_names[9] == "J0319+4130"

        _assert_unlocked(PRISTINE_3C391_MS)


class TestSuiteSSynthetic:
    """Verification of Suite S synthetic Measurement Sets, physics, and ground truth."""

    def test_suite_s_files_finite_json_npz_history_and_locks(self) -> None:
        for task_id in SUITE_S_TASKS:
            task_dir = SYNTH_ROOT / task_id
            ms_path = task_dir / "input.ms"
            gt_path = task_dir / "ground_truth.json"

            _assert_unlocked(ms_path)
            _assert_no_history_leaks(ms_path)
            gt = _load_finite_json(gt_path)
            assert gt["task_id"] == task_id
            assert gt["suite"] == "synth"
            assert gt["n_antennas"] == 10
            assert gt["dead_antenna"] == "ea07"

        for task_id in ("s1_precal", "s2_e2e_synth"):
            tdir = SYNTH_ROOT / task_id
            tgt = _load_finite_json(tdir / "ground_truth.json")
            assert tgt["corruption_masks_file"] == "corruption_masks.npz"
            masks = _load_finite_npz(tdir / tgt["corruption_masks_file"])
            assert set(masks.keys()) >= {
                "corrupted_chan_row",
                "quack_rows",
                "scan1_quack_rows",
                "burst_rows",
                "dead_rows",
                "rfi_channels",
            }

        for task_id in ("s1_calibrate", "s2_e2e_synth"):
            tdir = SYNTH_ROOT / task_id
            tgt = _load_finite_json(tdir / "ground_truth.json")
            assert tgt["true_target_vis_file"] == "true_target_vis.npz"
            tvis = _load_finite_npz(tdir / tgt["true_target_vis_file"])
            assert set(tvis.keys()) >= {
                "target_row_indices",
                "clean_target_row_mask",
                "clean_chan_mask",
                "true_rr",
                "true_ll",
            }

        s1_img_gt = _load_finite_json(SYNTH_ROOT / "s1_image" / "ground_truth.json")
        img_ref = s1_img_gt["reference_verification"]
        assert img_ref["negative_bowl_ratio"] <= 0.08
        assert img_ref["pt_snr_image"] >= 7.0

    def test_suite_s_antennas_scans_spw_coords_obs_and_intents(self) -> None:
        expected_xs, expected_ys, expected_zs, expected_diams, expected_names = load_vla_d_subset()

        for task_id in SUITE_S_TASKS:
            ms_path = SYNTH_ROOT / task_id / "input.ms"

            with _open_table(ms_path / "ANTENNA") as tb_ant:
                ant_names = [str(x) for x in tb_ant.getcol("NAME")]
                pos = tb_ant.getcol("POSITION")
                diams = tb_ant.getcol("DISH_DIAMETER")

            assert ant_names == expected_names
            assert np.allclose(pos[0], expected_xs, rtol=0.0, atol=1e-2)
            assert np.allclose(pos[1], expected_ys, rtol=0.0, atol=1e-2)
            assert np.allclose(pos[2], expected_zs, rtol=0.0, atol=1e-2)
            assert np.allclose(diams, expected_diams)

            with _open_table(ms_path / "SPECTRAL_WINDOW") as tb_spw:
                freqs_hz = tb_spw.getcol("CHAN_FREQ")[:, 0]
                widths_hz = tb_spw.getcol("CHAN_WIDTH")[:, 0]

            assert len(freqs_hz) == SPW_NCHAN == 64
            assert freqs_hz[0] == pytest.approx(SPW_START_HZ, rel=1e-9)
            assert np.allclose(widths_hz, SPW_CHAN_WIDTH_HZ)

            with _open_table(ms_path / "OBSERVATION") as tb_obs:
                time_range = tb_obs.getcol("TIME_RANGE")
            obs_span_sec = float(time_range[1, 0] - time_range[0, 0])
            assert obs_span_sec > 8000.0, (
                f"{task_id} OBSERVATION.TIME_RANGE span {obs_span_sec:.1f}s <= 8000s"
            )

            with _open_table(ms_path / "FIELD") as tb_fld:
                field_names = [str(x) for x in tb_fld.getcol("NAME")]
                phase_dir = tb_fld.getcol("PHASE_DIR")
            assert field_names == [
                FLUX_CAL_NAME,
                PHASE_CAL_NAME,
                TARGET_P1_NAME,
                TARGET_P2_NAME,
            ]
            c_j1331 = SkyCoord(
                ra=float(phase_dir[0, 0, 0]) * u.rad,
                dec=float(phase_dir[1, 0, 0]) * u.rad,
                frame="icrs",
            )
            c_j1822 = SkyCoord(
                ra=float(phase_dir[0, 0, 1]) * u.rad,
                dec=float(phase_dir[1, 0, 1]) * u.rad,
                frame="icrs",
            )
            assert float(c_j1331.separation(VLA_J1331_CATALOG).arcsec) < 1.0
            assert float(c_j1822.separation(VLA_J1822_CATALOG).arcsec) < 1.0

            with _open_table(ms_path) as tb:
                scans = tb.getcol("SCAN_NUMBER")
                ant1 = tb.getcol("ANTENNA1")
                ant2 = tb.getcol("ANTENNA2")

            assert sorted(int(s) for s in np.unique(scans)) == list(range(1, 14))
            assert np.all(ant1 != ant2)
            assert len(scans) == 5400

            with _open_msmd(ms_path) as msmd:
                f0_intents = sorted(str(x) for x in msmd.intentsforfield(0))
                f1_intents = sorted(str(x) for x in msmd.intentsforfield(1))
                f2_intents = sorted(str(x) for x in msmd.intentsforfield(2))
                f3_intents = sorted(str(x) for x in msmd.intentsforfield(3))

            assert {"CALIBRATE_BANDPASS#ON_SOURCE", "CALIBRATE_FLUX#ON_SOURCE"} <= set(f0_intents)
            if task_id in SUITE_S_RAW_TASKS:
                all_field_intents = [f0_intents, f1_intents, f2_intents, f3_intents]
                intent_coverage = sum(1 for ints in all_field_intents if len(ints) > 0) / len(
                    all_field_intents
                )
                assert intent_coverage < 0.50, (
                    f"{task_id} raw field intent coverage {intent_coverage:.2f} >= 0.50 "
                    f"(breaks ms_set_intents)"
                )
                assert "CALIBRATE_PHASE#ON_SOURCE" not in f1_intents, (
                    f"{task_id} Field 1 must lack CALIBRATE_PHASE intent"
                )
            else:
                assert "CALIBRATE_PHASE#ON_SOURCE" in f1_intents, (
                    f"{task_id} Field 1 must have fixed CALIBRATE_PHASE intent"
                )
                assert f2_intents == ["OBSERVE_TARGET#ON_SOURCE"]
                assert f3_intents == ["OBSERVE_TARGET#ON_SOURCE"]

            _assert_unlocked(ms_path)

    def test_suite_s_corruptions_and_dead_antenna(self) -> None:
        masks = _load_finite_npz(SYNTH_ROOT / "s1_precal" / "corruption_masks.npz")
        corrupted_chan_row = masks["corrupted_chan_row"].astype(bool)
        quack_rows = masks["quack_rows"].astype(bool)
        scan1_quack_rows = masks["scan1_quack_rows"].astype(bool)
        burst_rows = masks["burst_rows"].astype(bool)
        dead_rows = masks["dead_rows"].astype(bool)
        rfi_channels = masks["rfi_channels"].astype(int).tolist()

        assert rfi_channels == [29, 30, 31]
        assert corrupted_chan_row.shape == (64, 5400)
        assert np.all(corrupted_chan_row[rfi_channels, :])
        assert np.all(corrupted_chan_row[:, quack_rows])
        assert np.all(corrupted_chan_row[:, burst_rows])
        assert np.all(corrupted_chan_row[:, dead_rows])

        for task_id in SUITE_S_RAW_TASKS:
            ms_path = SYNTH_ROOT / task_id / "input.ms"
            with _open_table(ms_path) as tb:
                data = tb.getcol("DATA")
                flag = tb.getcol("FLAG")
                scans = tb.getcol("SCAN_NUMBER")
                ant1 = tb.getcol("ANTENNA1")
                ant2 = tb.getcol("ANTENNA2")

            ms_dead_rows = (ant1 == SYNTH_DEAD_ANTENNA_ID) | (ant2 == SYNTH_DEAD_ANTENNA_ID)
            assert np.array_equal(ms_dead_rows, dead_rows)
            # Raw MSs have unflagged zeroed visibilities on dead antenna ea07
            assert float(np.max(np.abs(data[:, :, dead_rows]))) == 0.0
            assert not np.any(flag)

            # Scan-1 settling quack and Scan-5 broadband burst align with exact scan numbers
            assert np.unique(scans[scan1_quack_rows]).tolist() == [1]
            assert int(np.sum(scan1_quack_rows)) == 90
            assert np.unique(scans[burst_rows]).tolist() == [5]
            assert int(np.sum(burst_rows)) == 45

            # Verify narrowband RFI spike on channels [29, 30, 31] and broadband burst on Scan 5
            clean_live_rows = (~dead_rows) & (~quack_rows) & (~burst_rows)
            clean_ch = [c for c in range(64) if c not in rfi_channels]
            clean_median_amp = float(np.median(np.abs(data[0, clean_ch, :][:, clean_live_rows])))
            rfi_median_amp = float(np.median(np.abs(data[0, rfi_channels, :][:, clean_live_rows])))
            burst_median_amp = float(
                np.median(np.abs(data[0, clean_ch, :][:, burst_rows & (~dead_rows)]))
            )
            scan1_quack_amp = float(
                np.median(np.abs(data[0, clean_ch, :][:, scan1_quack_rows & (~dead_rows)]))
            )
            scan2_clean_amp = float(
                np.median(np.abs(data[0, clean_ch, :][:, (scans == 2) & clean_live_rows]))
            )
            assert rfi_median_amp > 10.0
            assert clean_median_amp < 2.0
            assert burst_median_amp > 15.0
            assert scan1_quack_amp > 5.0 * scan2_clean_amp

        # s1_calibrate leaves ea07 unflagged with near-zero micro-noise while preflagging RFI/quack
        cal_ms = SYNTH_ROOT / "s1_calibrate" / "input.ms"
        with _open_table(cal_ms) as tb:
            cal_data = tb.getcol("DATA")
            cal_flag = tb.getcol("FLAG")
        ea07_cal_rms = float(np.sqrt(np.mean(np.abs(cal_data[:, :, dead_rows]) ** 2)))
        assert 0.0 < ea07_cal_rms < 1.0e-3
        assert float(np.mean(~cal_flag[:, :, dead_rows])) > 0.80
        assert np.all(cal_flag[:, rfi_channels, :])
        assert np.all(cal_flag[:, :, quack_rows])
        assert np.all(cal_flag[:, :, burst_rows])

        # s1_image has ea07 100% flagged
        img_ms = SYNTH_ROOT / "s1_image" / "input.ms"
        with _open_table(img_ms) as tb:
            img_flag = tb.getcol("FLAG")
        assert float(np.mean(img_flag[:, :, dead_rows])) == pytest.approx(1.0)

    def test_suite_s_delay_phase_slope_raw_vs_calibrated(self) -> None:
        cal_ms = SYNTH_ROOT / "s1_calibrate" / "input.ms"
        img_ms = SYNTH_ROOT / "s1_image" / "input.ms"

        with _open_table(cal_ms / "SPECTRAL_WINDOW") as tb_spw:
            freqs_hz = tb_spw.getcol("CHAN_FREQ")[:, 0]

        # Raw DATA in s1_calibrate carries +-14 ns antenna delays (ea02=+14ns, ea10=-14ns vs ea01=0ns)
        with _open_table(cal_ms) as tb:
            raw_data = tb.getcol("DATA")
            raw_flag = tb.getcol("FLAG")
            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")
            fids = tb.getcol("FIELD_ID")

        for ant_idx, expected_tau_ns in ((1, 14.0), (9, -14.0)):
            bl_rows = (
                (fids == 0) & (ant1 == 0) & (ant2 == ant_idx) & (~np.all(raw_flag, axis=(0, 1)))
            )
            assert int(np.sum(bl_rows)) >= 10
            unflagged_ch = ~np.any(raw_flag[0, :, :][:, bl_rows], axis=1)
            vec = np.mean(raw_data[0, unflagged_ch, :][:, bl_rows], axis=1)
            phs_deg = np.rad2deg(np.unwrap(np.angle(vec)))
            slope_deg_per_hz, _ = np.polyfit(
                freqs_hz[unflagged_ch] - np.mean(freqs_hz[unflagged_ch]),
                phs_deg,
                1,
            )
            slope_deg_per_100mhz = float(slope_deg_per_hz * 1.0e8)
            implied_tau_ns = float(slope_deg_per_hz / 360.0 * 1.0e9)
            assert abs(slope_deg_per_100mhz) > 450.0
            assert implied_tau_ns == pytest.approx(expected_tau_ns, abs=0.75)

        # Calibrated CORRECTED_DATA in s1_image has flat phase across frequency (< 5.0 deg / 100 MHz)
        with _open_table(img_ms) as tb:
            corr = tb.getcol("CORRECTED_DATA")
            img_flag = tb.getcol("FLAG")
            ant1_i = tb.getcol("ANTENNA1")
            ant2_i = tb.getcol("ANTENNA2")
            fids_i = tb.getcol("FIELD_ID")

        dead_i = (ant1_i == SYNTH_DEAD_ANTENNA_ID) | (ant2_i == SYNTH_DEAD_ANTENNA_ID)
        j1331_live = (fids_i == 0) & (~dead_i)
        valid_ch: list[int] = []
        chan_vec: list[complex] = []
        for ch in range(len(freqs_hz)):
            m_rr = ~img_flag[0, ch, j1331_live]
            m_ll = ~img_flag[3, ch, j1331_live]
            vals = []
            if np.any(m_rr):
                vals.append(np.mean(corr[0, ch, j1331_live][m_rr]))
            if np.any(m_ll):
                vals.append(np.mean(corr[3, ch, j1331_live][m_ll]))
            if vals:
                valid_ch.append(ch)
                chan_vec.append(complex(np.mean(vals)))

        phs_corr_deg = np.rad2deg(np.unwrap(np.angle(np.array(chan_vec))))
        slope_corr_hz, _ = np.polyfit(
            freqs_hz[valid_ch] - np.mean(freqs_hz[valid_ch]),
            phs_corr_deg,
            1,
        )
        assert abs(float(slope_corr_hz) * 1.0e8) < 5.0

        for ant_idx in (1, 9):
            bl_rows = (
                (fids_i == 0)
                & (ant1_i == 0)
                & (ant2_i == ant_idx)
                & (~np.all(img_flag, axis=(0, 1)))
            )
            unflagged_ch = ~np.any(img_flag[0, :, :][:, bl_rows], axis=1)
            vec = np.mean(corr[0, unflagged_ch, :][:, bl_rows], axis=1)
            phs_b = np.rad2deg(np.unwrap(np.angle(vec)))
            sl_b, _ = np.polyfit(
                freqs_hz[unflagged_ch] - np.mean(freqs_hz[unflagged_ch]),
                phs_b,
                1,
            )
            assert abs(float(sl_b) * 1.0e8) < 5.0

    def test_suite_s_column_and_stage_invariants(self) -> None:
        for task_id in SUITE_S_TASKS:
            task_dir = SYNTH_ROOT / task_id
            # calibrators.ms is produced by the agent during precal, never pre-staged in task input dirs
            assert not (task_dir / "calibrators.ms").exists(), (
                f"Unexpected pre-staged calibrators.ms in {task_dir}"
            )

            ms_path = task_dir / "input.ms"
            with _open_table(ms_path) as tb:
                cols = set(tb.colnames())
                fids = tb.getcol("FIELD_ID")
                model = tb.getcol("MODEL_DATA") if "MODEL_DATA" in cols else None

            assert "DATA" in cols
            if task_id in ("s1_inspect", "s1_precal", "s1_calibrate", "s2_e2e_synth"):
                assert "CORRECTED_DATA" not in cols, (
                    f"{task_id}/input.ms must not carry pre-existing CORRECTED_DATA"
                )
            else:
                assert "CORRECTED_DATA" in cols, "s1_image/input.ms must carry CORRECTED_DATA"

            if task_id in SUITE_S_RAW_TASKS:
                if model is not None:
                    # Uninitialized raw MS MODEL_DATA must be pure parallel-hand unity (no setjy/sky leak)
                    assert np.allclose(model[0], 1.0 + 0.0j)
                    assert np.allclose(model[3], 1.0 + 0.0j)
                    assert np.allclose(model[1:3], 0.0 + 0.0j)
            elif task_id in SUITE_S_FIXED_INTENT_TASKS:
                assert model is not None, f"{task_id}/input.ms must carry MODEL_DATA"
                j1331_par = 0.5 * (np.abs(model[0, :, fids == 0]) + np.abs(model[3, :, fids == 0]))
                j1331_mean_jy = float(np.mean(j1331_par))
                assert abs(j1331_mean_jy - 7.60) / 7.60 <= 0.015
                tgt_mask = np.isin(fids, [2, 3])
                assert np.allclose(model[0, :, tgt_mask], 1.0 + 0.0j)
                assert np.allclose(model[3, :, tgt_mask], 1.0 + 0.0j)
                assert np.allclose(model[1:3, :, tgt_mask], 0.0 + 0.0j)


class TestSuiteR3C391Checkpoints:
    """Verification of Suite R golden 3C391 checkpoints, caltables, images, and ground truth."""

    def test_suite_r_files_finite_json_npz_history_and_locks(self) -> None:
        casa_tables = [
            *SUITE_R_MS_PATHS,
            CHECKPOINTS_3C391_ROOT / "post_calibrate" / "caltables" / "G0.cal",
            CHECKPOINTS_3C391_ROOT / "post_calibrate" / "caltables" / "K.cal",
            CHECKPOINTS_3C391_ROOT / "post_calibrate" / "caltables" / "B.cal",
            CHECKPOINTS_3C391_ROOT / "post_calibrate" / "caltables" / "G.cal",
            CHECKPOINTS_3C391_ROOT / "post_calibrate" / "caltables" / "fluxscale.cal",
            CHECKPOINTS_3C391_ROOT / "reference_image" / "3c391_mosaic.image",
            CHECKPOINTS_3C391_ROOT / "reference_image" / "3c391_mosaic.image.pbcor",
            CHECKPOINTS_3C391_ROOT / "reference_image" / "3c391_mosaic.pb",
            CHECKPOINTS_3C391_ROOT / "reference_image" / "3c391_mosaic.residual",
            CHECKPOINTS_3C391_ROOT / "reference_image" / "3c391_mosaic.model",
        ]
        for table_dir in casa_tables:
            _assert_unlocked(table_dir)
            with _open_table(table_dir) as tb:
                assert tb.nrows() > 0, f"Empty CASA table: {table_dir}"

        assert not (CHECKPOINTS_3C391_ROOT / "reference_image" / "3c391_mosaic.pbcor").exists(), (
            "Legacy 3c391_mosaic.pbcor must be replaced by 3c391_mosaic.image.pbcor"
        )

        # Verify zero builder HISTORY leaks across all 4 Suite R MSs
        for ms_path in SUITE_R_MS_PATHS:
            _assert_no_history_leaks(ms_path)

        _load_finite_npz(CHECKPOINTS_3C391_ROOT / "post_precal" / "precal_masks.npz")

        master_gt = _load_finite_json(CHECKPOINTS_3C391_ROOT / "ground_truth.json")
        assert master_gt["suite"] == "R"
        assert master_gt["pass_reward_min"] == pytest.approx(0.85)
        assert master_gt["target_retention_excludes_dead_antenna"] is False
        assert master_gt["target_retention_excludes_autocorr"] is True
        assert master_gt["all_intents_missing_in_raw"] is True
        assert master_gt["intent_delimiters"] == [",", ";"]
        assert set(master_gt["tasks"].keys()) == set(SUITE_R_TASKS)

        for task_id in SUITE_R_TASKS:
            task_gt = _load_finite_json(CHECKPOINTS_3C391_ROOT / task_id / "ground_truth.json")
            assert task_gt["task_id"] == task_id
            assert task_gt["suite"] == "R"
            assert task_gt["pass_reward_min"] == pytest.approx(0.85)
            assert task_gt["tolerances"]["pass_reward_min"] == pytest.approx(0.85)

        # Verify staged precal_masks.npz alongside ground_truth.json with relative filename
        for mask_task_id in ("r1_3c391_precal", "r2_3c391_e2e"):
            task_dir = CHECKPOINTS_3C391_ROOT / mask_task_id
            task_gt = _load_finite_json(task_dir / "ground_truth.json")
            masks_rel = task_gt["precal_masks_file"]
            assert masks_rel == "precal_masks.npz"
            assert not Path(masks_rel).is_absolute()
            assert "/" not in masks_rel
            _load_finite_npz(task_dir / masks_rel)

    def test_suite_r_raw_checkpoint_matches_pristine(self) -> None:
        raw_ms = CHECKPOINTS_3C391_ROOT / "raw" / "input.ms"
        with _open_table(raw_ms) as tb:
            assert int(tb.nrows()) == 845_379
            cols = set(tb.colnames())
            scans = tb.getcol("SCAN_NUMBER")
            raw_flag = tb.getcol("FLAG")
            raw_flag_row = tb.getcol("FLAG_ROW")
        assert "DATA" in cols
        assert "CORRECTED_DATA" not in cols
        assert "MODEL_DATA" not in cols
        assert "IMAGING_WEIGHT" not in cols
        assert len(np.unique(scans)) == 103
        assert float(np.mean(raw_flag)) == 0.0
        assert float(np.mean(raw_flag_row)) == 0.0

        with _open_table(raw_ms / "STATE") as tb_state:
            assert int(tb_state.nrows()) == 0

        with _open_table(raw_ms / "ANTENNA") as tb_ant:
            ant_names = [str(x) for x in tb_ant.getcol("NAME")]
            assert len(ant_names) == 26
            assert ant_names[10] == "ea13"

        _assert_no_history_leaks(raw_ms)

        with _open_msmd(raw_ms) as msmd:
            assert "CALIBRATE_PHASE#ON_SOURCE" not in set(msmd.intentsforfield(1))

        _assert_unlocked(raw_ms)

    def test_suite_r_post_precal_checkpoint(self) -> None:
        master_gt = _load_finite_json(CHECKPOINTS_3C391_ROOT / "ground_truth.json")
        post_precal_ms = CHECKPOINTS_3C391_ROOT / "post_precal" / "input.ms"
        calibrators_ms = CHECKPOINTS_3C391_ROOT / "post_precal" / "calibrators.ms"
        masks_npz = CHECKPOINTS_3C391_ROOT / "post_precal" / "precal_masks.npz"

        with _open_msmd(post_precal_ms) as msmd:
            f1_intents = sorted(str(x) for x in msmd.intentsforfield(1))
            f2_intents = sorted(str(x) for x in msmd.intentsforfield(2))
        assert "CALIBRATE_PHASE#ON_SOURCE" in f1_intents
        assert "CALIBRATE_PHASE#ON_SOURCE" not in f2_intents

        with _open_table(post_precal_ms) as tb:
            cols = set(tb.colnames())
            assert int(tb.nrows()) == 845_379
            assert "DATA" in cols
            assert "MODEL_DATA" in cols
            assert "CORRECTED_DATA" not in cols

            flag = tb.getcol("FLAG")
            flag_row = tb.getcol("FLAG_ROW")
            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")

            with _query_table(tb, "FIELD_ID == 0 && !FLAG_ROW") as sub_j1331:
                j1331_model = sub_j1331.getcol("MODEL_DATA")

        j1331_par = 0.5 * (np.abs(j1331_model[0]) + np.abs(j1331_model[3]))
        j1331_mean_jy = float(np.mean(j1331_par))
        assert abs(j1331_mean_jy - 7.60) / 7.60 <= 0.015
        assert j1331_mean_jy == pytest.approx(
            master_gt["post_precal"]["j1331_model_parallel_mean_jy"], abs=1e-4
        )

        # calibrators.ms has 80,465 rows and no CORRECTED_DATA
        with _open_table(calibrators_ms) as tb_cal:
            cal_cols = set(tb_cal.colnames())
            cal_nrows = int(tb_cal.nrows())
            cal_fids = {int(x) for x in np.unique(tb_cal.getcol("FIELD_ID"))}
        with _open_table(calibrators_ms / "FIELD") as tb_cal_fld:
            cal_field_names = [str(x) for x in tb_cal_fld.getcol("NAME")]
        assert cal_nrows == 80_465
        assert "DATA" in cal_cols
        assert "MODEL_DATA" in cal_cols
        assert "CORRECTED_DATA" not in cal_cols
        assert cal_fids == {0, 1, 2}
        assert cal_field_names == ["J1331+3030", "J1822-0938", "J0319+4130"]

        # Verify packed precal masks and ea13 left unflagged for r1_3c391_calibrate
        masks = _load_finite_npz(masks_npz)
        shape_2d = tuple(int(x) for x in masks["shape_2d"])
        assert shape_2d == (64, 845_379)
        must_flag_2d = np.unpackbits(masks["must_flag_2d"], axis=1)[:, : shape_2d[1]].astype(bool)
        clean_2d = np.unpackbits(masks["clean_2d"], axis=1)[:, : shape_2d[1]].astype(bool)
        assert not np.any(must_flag_2d & clean_2d)

        chan_flagged = np.any(flag, axis=0) | flag_row[np.newaxis, :]
        assert float(np.mean(chan_flagged[must_flag_2d])) >= 0.95
        assert float(np.mean(chan_flagged[clean_2d])) < 0.05

        ea13_rows = (ant1 == 10) | (ant2 == 10)
        assert float(np.mean(flag[:, :, ea13_rows])) < 0.50

        _assert_unlocked(post_precal_ms)
        _assert_unlocked(calibrators_ms)

    def test_suite_r_post_calibrate_checkpoint(self) -> None:
        master_gt = _load_finite_json(CHECKPOINTS_3C391_ROOT / "ground_truth.json")
        post_cal_ms = CHECKPOINTS_3C391_ROOT / "post_calibrate" / "input.ms"
        caltables_dir = CHECKPOINTS_3C391_ROOT / "post_calibrate" / "caltables"

        with _open_table(post_cal_ms / "SPECTRAL_WINDOW") as tb_spw:
            freqs_mhz = tb_spw.getcol("CHAN_FREQ")[:, 0] / 1.0e6

        with _open_table(post_cal_ms) as tb:
            cols = set(tb.colnames())
            assert "CORRECTED_DATA" in cols

            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")
            field_id = tb.getcol("FIELD_ID")
            flag = tb.getcol("FLAG")
            flag_row = tb.getcol("FLAG_ROW")

            with _query_table(tb, "FIELD_ID in [0, 1] && ANTENNA1 != ANTENNA2") as sub_cal:
                sub_fids = sub_cal.getcol("FIELD_ID")
                sub_corr = sub_cal.getcol("CORRECTED_DATA")
                sub_flag = sub_cal.getcol("FLAG")

        # Dead antenna ea13 (index 10) must be 100% flagged
        ea13_mask = (ant1 == 10) | (ant2 == 10)
        assert float(np.mean(flag[:, :, ea13_mask])) == pytest.approx(1.0)

        # Target unflagged row and visibility fractions in [0.70, 0.90]
        target_rows = np.isin(field_id, [2, 3, 4, 5, 6, 7, 8]) & (ant1 != ant2)
        target_row_flagged = flag_row[target_rows] | np.all(flag[:, :, target_rows], axis=(0, 1))
        target_unflagged_row_frac = float(1.0 - np.mean(target_row_flagged))
        target_unflagged_vis_frac = float(1.0 - np.mean(flag[:, :, target_rows]))
        assert 0.70 <= target_unflagged_row_frac <= 0.90
        assert 0.70 <= target_unflagged_vis_frac <= 0.90

        # J1331+3030 (Field 0) channel phase RMS < 8.0 deg
        j1331_sel = sub_fids == 0
        j1331_corr = sub_corr[:, :, j1331_sel]
        j1331_flag = sub_flag[:, :, j1331_sel]

        vis_phases_deg: list[np.ndarray] = []
        chan_sums = np.zeros(len(freqs_mhz), dtype=np.complex128)
        chan_counts = np.zeros(len(freqs_mhz), dtype=np.int64)
        for pol_idx in (0, 3):
            good = ~j1331_flag[pol_idx]
            vis_phases_deg.append(np.rad2deg(np.angle(j1331_corr[pol_idx][good])))
            for ch in range(len(freqs_mhz)):
                g_ch = good[ch]
                if np.any(g_ch):
                    chan_sums[ch] += np.sum(np.exp(1j * np.angle(j1331_corr[pol_idx, ch, g_ch])))
                    chan_counts[ch] += int(np.sum(g_ch))

        all_ph_deg = np.concatenate(vis_phases_deg)
        vis_phase_rms_deg = float(np.sqrt(np.mean(all_ph_deg**2)))
        valid_ch = chan_counts > 0
        coh_ph_deg = np.rad2deg(np.unwrap(np.angle(chan_sums[valid_ch])))
        coh_ph_deg = coh_ph_deg - np.mean(coh_ph_deg)
        coh_chan_rms_deg = float(np.sqrt(np.mean(coh_ph_deg**2)))
        slope_per_mhz, _ = np.polyfit(freqs_mhz[valid_ch], coh_ph_deg, 1)

        assert vis_phase_rms_deg < 8.0
        assert coh_chan_rms_deg < 8.0
        assert abs(float(slope_per_mhz) * 100.0) < 5.0

        # J1822-0938 (Field 1) fluxscale within +-8.0% of 2.30 Jy (from caltables + CORRECTED_DATA)
        with _open_table(caltables_dir / "G.cal") as tb_g:
            g_fids = tb_g.getcol("FIELD_ID")
            g_cparam = tb_g.getcol("CPARAM")
            g_flag = tb_g.getcol("FLAG")
        with _open_table(caltables_dir / "fluxscale.cal") as tb_fs:
            fs_fids = tb_fs.getcol("FIELD_ID")
            fs_cparam = tb_fs.getcol("CPARAM")
            fs_flag = tb_fs.getcol("FLAG")

        f1_unfl = (g_fids == 1) & (fs_fids == 1)
        valid_gains = (~g_flag[:, 0, f1_unfl]) & (~fs_flag[:, 0, f1_unfl])
        gain_ratios = np.abs(g_cparam[:, 0, f1_unfl][valid_gains]) / np.abs(
            fs_cparam[:, 0, f1_unfl][valid_gains]
        )
        fluxscale_j1822_jy = float(np.median(gain_ratios**2))
        assert abs(fluxscale_j1822_jy - 2.30) / 2.30 <= 0.08
        assert abs(master_gt["post_calibrate"]["j1822_fluxscale_jy"] - 2.30) / 2.30 <= 0.08

        j1822_sel = sub_fids == 1
        j1822_rr = sub_corr[0, :, :][:, j1822_sel][~sub_flag[0, :, :][:, j1822_sel]]
        j1822_ll = sub_corr[3, :, :][:, j1822_sel][~sub_flag[3, :, :][:, j1822_sel]]
        j1822_corr_jy = float(0.5 * (abs(np.mean(j1822_rr)) + abs(np.mean(j1822_ll))))
        assert abs(j1822_corr_jy - 2.30) / 2.30 <= 0.08

        _assert_unlocked(post_cal_ms)

    def test_suite_r_reference_image_checkpoint(self) -> None:
        master_gt = _load_finite_json(CHECKPOINTS_3C391_ROOT / "ground_truth.json")
        img_gt = _load_finite_json(CHECKPOINTS_3C391_ROOT / "r1_3c391_image" / "ground_truth.json")
        cal_gt = _load_finite_json(
            CHECKPOINTS_3C391_ROOT / "r1_3c391_calibrate" / "ground_truth.json"
        )
        e2e_gt = _load_finite_json(CHECKPOINTS_3C391_ROOT / "r2_3c391_e2e" / "ground_truth.json")
        tolerances = img_gt["tolerances"]

        assert tolerances["off_source_residual_rms_max_mjy_per_beam"] == pytest.approx(0.75)
        assert tolerances["off_source_residual_rms_max_jy_per_beam"] == pytest.approx(0.00075)
        assert tolerances["dynamic_range_min"] == pytest.approx(200.0)
        assert tolerances["negative_model_bowl_ratio_max"] == pytest.approx(0.12)
        assert tolerances["pixels_per_beam_min"] == pytest.approx(3.0)
        assert tolerances["pixels_per_beam_max"] == pytest.approx(7.0)
        assert cal_gt["tolerances"]["verifier_tclean_dr_min"] == pytest.approx(180.0)
        assert e2e_gt["tolerances"]["off_source_residual_rms_max_mjy_per_beam"] == pytest.approx(
            0.75
        )
        assert e2e_gt["tolerances"]["off_source_residual_rms_max_jy_per_beam"] == pytest.approx(
            0.00075
        )
        assert e2e_gt["tolerances"]["dynamic_range_min"] == pytest.approx(200.0)
        assert e2e_gt["tolerances"]["verifier_tclean_dr_min"] == pytest.approx(180.0)

        img_prefix = CHECKPOINTS_3C391_ROOT / "reference_image" / "3c391_mosaic"
        pbcor_dir = Path(f"{img_prefix}.image.pbcor")
        assert pbcor_dir.is_dir(), f"Missing CASA .image.pbcor directory: {pbcor_dir}"

        with _open_image(Path(f"{img_prefix}.image")) as ia_img:
            img_arr = np.squeeze(ia_img.getchunk())
            rb = ia_img.restoringbeam()
            if "beams" in rb:
                rb = rb["beams"]["*0"]["*0"]
            cs = ia_img.coordsys()
            try:
                inc_rad = cs.increment()["numeric"]
            finally:
                cs.done()

        with _open_image(Path(f"{img_prefix}.pb")) as ia_pb:
            pb_arr = np.squeeze(ia_pb.getchunk())

        with _open_image(pbcor_dir) as ia_pbcor:
            pbcor_arr = np.squeeze(ia_pbcor.getchunk())
            pbcor_mask = np.squeeze(ia_pbcor.getchunk(getmask=True))

        with _open_image(Path(f"{img_prefix}.residual")) as ia_res:
            res_arr = np.squeeze(ia_res.getchunk())

        with _open_image(Path(f"{img_prefix}.model")) as ia_mod:
            mod_arr = np.squeeze(ia_mod.getchunk())

        assert master_gt["reference_image"]["scales"] == [0, 5, 15, 30]
        cell_arcsec = abs(math.degrees(float(inc_rad[0])) * 3600.0)
        b_minor_arcsec = _quantity_to_arcsec(rb["minor"])
        pixels_per_beam = b_minor_arcsec / cell_arcsec
        assert (
            tolerances["pixels_per_beam_min"]
            <= pixels_per_beam
            <= tolerances["pixels_per_beam_max"]
        )
        assert pixels_per_beam == pytest.approx(
            master_gt["reference_image"]["pixels_per_beam"], abs=1e-3
        )

        pos_mod_sum = float(np.sum(mod_arr[mod_arr > 0.0]))
        neg_mod_abs_sum = float(np.abs(np.sum(mod_arr[mod_arr < 0.0])))
        assert pos_mod_sum > 0.0
        bowl_ratio = neg_mod_abs_sum / pos_mod_sum
        assert bowl_ratio <= 0.08
        assert bowl_ratio < tolerances["negative_model_bowl_ratio_max"]
        assert bowl_ratio == pytest.approx(
            master_gt["reference_image"]["negative_model_bowl_ratio"], abs=1e-5
        )

        pb_valid = (
            np.isfinite(pb_arr)
            & (pb_arr >= tolerances["pb_cutoff"])
            & np.isfinite(img_arr)
            & np.isfinite(res_arr)
        )
        peak_img_jy = float(np.max(img_arr[pb_valid]))
        valid_pbcor = pbcor_mask & pb_valid & np.isfinite(pbcor_arr)
        peak_pbcor_jy = float(np.max(pbcor_arr[valid_pbcor]))
        assert math.isfinite(peak_img_jy) and peak_img_jy > 0.05
        assert math.isfinite(peak_pbcor_jy) and peak_pbcor_jy > 0.05

        nx, ny = img_arr.shape
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx)
        radius_arcsec = np.hypot(xx, yy) * cell_arcsec
        off_source_mask = pb_valid & (radius_arcsec >= 220.0) & (np.abs(mod_arr) == 0.0)
        assert int(np.sum(off_source_mask)) >= 1000

        off_source_rms_jy = float(np.sqrt(np.mean(res_arr[off_source_mask] ** 2)))
        off_source_rms_mjy = off_source_rms_jy * 1.0e3
        dynamic_range = peak_img_jy / off_source_rms_jy

        assert math.isfinite(off_source_rms_jy) and off_source_rms_jy > 0.0
        assert off_source_rms_jy < tolerances["off_source_residual_rms_max_jy_per_beam"]
        assert off_source_rms_mjy < tolerances["off_source_residual_rms_max_mjy_per_beam"]
        assert (
            master_gt["reference_image"]["off_source_residual_rms_mjy_per_beam"]
            < tolerances["off_source_residual_rms_max_mjy_per_beam"]
        )
        assert dynamic_range >= tolerances["dynamic_range_min"]
        assert dynamic_range >= cal_gt["tolerances"]["verifier_tclean_dr_min"]
        assert master_gt["reference_image"]["dynamic_range"] >= tolerances["dynamic_range_min"]

        assert off_source_rms_jy == pytest.approx(
            master_gt["reference_image"]["off_source_residual_rms_jy_per_beam"], abs=1e-6
        )

        for suffix in ("image", "image.pbcor", "pb", "residual", "model"):
            _assert_unlocked(Path(f"{img_prefix}.{suffix}"))
