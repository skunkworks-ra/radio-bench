"""Seeded VLA D-config C-band casatools.simulator helpers for Suite S."""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import astropy.units as u  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import AltAz, EarthLocation, SkyCoord  # noqa: E402
from astropy.time import Time  # noqa: E402

from radio_bench.config import (  # noqa: E402
    BenchConfig,
    is_under_slash_tmp,
    load_bench_config,
    write_casaconfig_local,
)

_CASA_ENV_CONFIGURED_FOR: Path | None = None


def configure_builder_casa_env(cfg: BenchConfig | None = None) -> BenchConfig:
    """Configure offline CASA, TMPDIR, and MPLCONFIGDIR lazily before importing casatools."""
    global _CASA_ENV_CONFIGURED_FOR
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    if resolved_cfg.work_dir == _CASA_ENV_CONFIGURED_FOR:
        return resolved_cfg

    cur_tmpdir = os.environ.get("TMPDIR", "").strip()
    if not cur_tmpdir or is_under_slash_tmp(cur_tmpdir) or cfg is not None:
        os.environ["TMPDIR"] = str(resolved_cfg.tmp_dir)

    cur_mpldir = os.environ.get("MPLCONFIGDIR", "").strip()
    if not cur_mpldir or is_under_slash_tmp(cur_mpldir) or cfg is not None:
        os.environ["MPLCONFIGDIR"] = str(resolved_cfg.mplconfig_dir)

    casa_cfg = write_casaconfig_local(resolved_cfg)
    if cfg is not None:
        os.environ["CASASITECONFIG"] = str(casa_cfg)
    else:
        os.environ.setdefault("CASASITECONFIG", str(casa_cfg))

    if not os.access(os.environ.get("HOME", "/"), os.W_OK):
        os.environ["HOME"] = str(resolved_cfg.tmp_dir)

    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    _CASA_ENV_CONFIGURED_FOR = resolved_cfg.work_dir
    return resolved_cfg


SYNTH_ANTENNA_INDICES = [18, 3, 12, 19, 21, 8, 5, 24, 17, 26]
SYNTH_ANTENNA_NAMES = [
    "ea01",
    "ea02",
    "ea03",
    "ea04",
    "ea05",
    "ea06",
    "ea07",
    "ea08",
    "ea09",
    "ea10",
]
DEAD_ANTENNA_NAME = "ea07"
DEAD_ANTENNA_ID = SYNTH_ANTENNA_NAMES.index(DEAD_ANTENNA_NAME)
REFANT_NAME = "ea01"

SPW_START_HZ = 4.536e9
SPW_CHAN_WIDTH_HZ = 2.0e6
SPW_NCHAN = 64

# Exact VLA calibrator manual positions (<0.01 arcsec from 1331+305 and 1822-096).
FLUX_CAL_NAME = "J1331+3030"
FLUX_CAL_RA = "13h31m08.288s"
FLUX_CAL_DEC = "+30d30m32.959s"

PHASE_CAL_NAME = "J1822-0938"
PHASE_CAL_RA = "18h22m28.704s"
PHASE_CAL_DEC = "-09d38m56.835s"
PHASE_CAL_FLUX_JY = 2.15
PHASE_CAL_SPIX = -0.70

TARGET_P1_NAME = "SYNTH_MOSAIC_P1"
TARGET_P1_RA = "18h23m54.000s"
TARGET_P1_DEC = "-09d35m30.000s"

TARGET_P2_NAME = "SYNTH_MOSAIC_P2"
TARGET_P2_RA = "18h24m06.000s"
TARGET_P2_DEC = "-09d34m30.000s"

MOSAIC_PHASECENTER = "J2000 18h24m00.000s -09d35m00.000s"

EXT_COMP_RA = "18h24m00.000s"
EXT_COMP_DEC = "-09d35m00.000s"
EXT_COMP_FLUX_JY = 0.075
EXT_COMP_MAJOR = "16arcsec"
EXT_COMP_MINOR = "12arcsec"
EXT_COMP_PA = "25deg"

PT_COMP_RA = "18h24m08.200s"
PT_COMP_DEC = "-09d34m20.000s"

# True antenna delays in nanoseconds (within +-14 ns, refant ea01 = 0.0 ns).
TRUE_DELAYS_NS: dict[str, float] = {
    "ea01": 0.0,
    "ea02": 14.0,
    "ea03": -2.2,
    "ea04": 1.8,
    "ea05": 2.6,
    "ea06": -2.8,
    "ea07": 0.0,
    "ea08": 2.1,
    "ea09": -1.5,
    "ea10": -14.0,
}

RFI_CHANNELS = [29, 30, 31]


@dataclass(frozen=True)
class ScanSpec:
    scan_number: int
    field_name: str
    start_sec: float
    stop_sec: float
    intent: str


# Reference epoch chosen so both J1331+3030 (transit ~13.5h LST) and J1822-0938 / target (~18.4h LST)
# stay above 20 deg elevation across a 2.5h Earth-rotation synthesis track at VLA.
REF_EPOCH_UTC = "2026-04-15/10:15:00"
INTEGRATION_SEC = 10.0

# Raw schedule: only J1331+3030 (Scans 1, 2, 13) carries populated STATE intents;
# Scans 3..12 (J1822-0938 + SYNTH_MOSAIC_P1/P2) have unassigned OBS_MODE = "" (STATE_ID = 1)
# so 1/4 fields (25% < 50%) have intents in raw MSs, allowing ms_set_intents while
# requiring phase-calibrator vs mosaic-target separation in s1_inspect / s1_precal.
SCAN_SCHEDULE: list[ScanSpec] = [
    # Scan 1: short 30s settling scan on J1331+3030 (first 20s quack-corrupted)
    ScanSpec(
        1,
        FLUX_CAL_NAME,
        0.0,
        30.0,
        "CALIBRATE_FLUX#ON_SOURCE,CALIBRATE_BANDPASS#ON_SOURCE",
    ),
    # Scan 2: main 120s bandpass/flux scan on J1331+3030
    ScanSpec(
        2,
        FLUX_CAL_NAME,
        40.0,
        160.0,
        "CALIBRATE_FLUX#ON_SOURCE,CALIBRATE_BANDPASS#ON_SOURCE",
    ),
    # Scans 3..12: 3 Earth-rotation mosaic blocks over ~2.4h (4 phase-cal scans + 6 mosaic scans,
    # all missing intents -> OBS_MODE = "" in raw MSs).
    ScanSpec(3, PHASE_CAL_NAME, 200.0, 260.0, ""),
    ScanSpec(4, TARGET_P1_NAME, 270.0, 390.0, ""),
    ScanSpec(5, TARGET_P2_NAME, 400.0, 520.0, ""),
    ScanSpec(6, PHASE_CAL_NAME, 4200.0, 4260.0, ""),
    ScanSpec(7, TARGET_P1_NAME, 4270.0, 4390.0, ""),
    ScanSpec(8, TARGET_P2_NAME, 4400.0, 4520.0, ""),
    ScanSpec(9, PHASE_CAL_NAME, 8200.0, 8260.0, ""),
    ScanSpec(10, TARGET_P1_NAME, 8270.0, 8390.0, ""),
    ScanSpec(11, TARGET_P2_NAME, 8400.0, 8520.0, ""),
    ScanSpec(12, PHASE_CAL_NAME, 8530.0, 8590.0, ""),
    # Scan 13: late J1331+3030 scan at +2.5h when J1331 is setting (~20.5 deg min elevation)
    ScanSpec(
        13,
        FLUX_CAL_NAME,
        9000.0,
        9090.0,
        "CALIBRATE_FLUX#ON_SOURCE,CALIBRATE_BANDPASS#ON_SOURCE",
    ),
]


def fixed_scan_intents() -> dict[int, str]:
    """Return full scan-intent map for post-inspect checkpoints (s1_calibrate, s1_image)."""
    intents: dict[int, str] = {}
    for spec in SCAN_SCHEDULE:
        if spec.field_name == FLUX_CAL_NAME:
            intents[spec.scan_number] = "CALIBRATE_FLUX#ON_SOURCE,CALIBRATE_BANDPASS#ON_SOURCE"
        elif spec.field_name == PHASE_CAL_NAME:
            intents[spec.scan_number] = "CALIBRATE_PHASE#ON_SOURCE"
        else:
            intents[spec.scan_number] = "OBSERVE_TARGET#ON_SOURCE"
    return intents


def clear_ms_pointing(ms_path: Path) -> None:
    """Clear POINTING rows (matching VLA importasdm MSs) so CASA MosaicFT uses FIELD.PHASE_DIR cleanly."""
    configure_builder_casa_env()
    from casatools import table

    ptg_path = ms_path / "POINTING"
    if not ptg_path.is_dir():
        return

    tb = table()
    opened = False
    try:
        tb.open(str(ptg_path), nomodify=False)
        opened = True
        nrows = int(tb.nrows())
        if nrows > 0:
            tb.removerows(list(range(nrows)))
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()


def clear_ms_history(ms_path: Path) -> None:
    """Remove all rows from ms_path/HISTORY so zero builder provenance leaks into staged MSs."""
    configure_builder_casa_env()
    from casatools import table

    hist_path = ms_path / "HISTORY"
    if not hist_path.is_dir():
        return

    tb = table()
    opened = False
    try:
        tb.open(str(hist_path), nomodify=False)
        opened = True
        nrows = int(tb.nrows())
        if nrows > 0:
            tb.removerows(list(range(nrows)))
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()


def verify_ms_unlocked(ms_path: Path) -> None:
    """Assert that an MS and all its subtables hold zero active casacore locks."""
    configure_builder_casa_env()
    from casatasks import clearstat
    from casatools import table

    clearstat()
    tb = table()
    opened = False
    try:
        tb.open(str(ms_path), nomodify=True)
        opened = True
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"MS has active casacore table lock(s): {ms_path}")
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()


def load_vla_d_subset() -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    """Load 10 well-distributed antennas from CASA's shipped vla.d.cfg."""
    configure_builder_casa_env()
    from casatools import ctsys

    cfg_path = Path(ctsys.resolve("alma/simmos")) / "vla.d.cfg"
    rows: list[tuple[float, float, float, float]] = []
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))

    xs, ys, zs, diams = [], [], [], []
    for idx in SYNTH_ANTENNA_INDICES:
        rx, ry, rz, rd = rows[idx]
        xs.append(rx)
        ys.append(ry)
        zs.append(rz)
        diams.append(rd)
    return xs, ys, zs, diams, list(SYNTH_ANTENNA_NAMES)


def compute_theoretical_mosaic_sigma_jy(
    sigma_vis_jy: float,
    *,
    include_slew_quack: bool = True,
) -> dict[str, float]:
    """Compute exact theoretical natural-weighted Stokes I mosaic thermal noise (Jy/beam)."""
    n_good_ant = len(SYNTH_ANTENNA_NAMES) - 1
    n_bl = (n_good_ant * (n_good_ant - 1)) // 2
    n_chan_clean = SPW_NCHAN - len(RFI_CHANNELS)

    p1_scans = [s for s in SCAN_SCHEDULE if s.field_name == TARGET_P1_NAME]
    p2_scans = [s for s in SCAN_SCHEDULE if s.field_name == TARGET_P2_NAME]
    n_int_p1 = sum(int(round((s.stop_sec - s.start_sec) / INTEGRATION_SEC)) for s in p1_scans)
    n_int_p2 = sum(int(round((s.stop_sec - s.start_sec) / INTEGRATION_SEC)) for s in p2_scans)
    if include_slew_quack:
        n_int_p1 -= len(p1_scans)
        n_int_p2 -= len(p2_scans)
    # Subtract the single Scan 5 broadband burst integration on TARGET_P2
    n_int_p2 -= 1

    m1 = n_bl * n_int_p1 * n_chan_clean
    m2 = n_bl * n_int_p2 * n_chan_clean

    p1_coord = SkyCoord(TARGET_P1_RA, TARGET_P1_DEC, frame="icrs")
    p2_coord = SkyCoord(TARGET_P2_RA, TARGET_P2_DEC, frame="icrs")
    nu_center_ghz = (SPW_START_HZ + 0.5 * (SPW_NCHAN - 1) * SPW_CHAN_WIDTH_HZ) / 1e9

    # Sample along the P1-P2 axis to find exact peak mosaic PB^2 weight sum (EVLA Perley polynomial)
    fracs = np.linspace(0.0, 1.0, 201)
    sep_arcmin = p1_coord.separation(p2_coord).arcmin
    max_eff_vis = 0.0
    for f in fracs:
        r1 = f * sep_arcmin
        r2 = (1.0 - f) * sep_arcmin
        a1 = _evla_pb(r1, nu_center_ghz)
        a2 = _evla_pb(r2, nu_center_ghz)
        eff = m1 * (a1**2) + m2 * (a2**2)
        if eff > max_eff_vis:
            max_eff_vis = float(eff)

    sigma_nat_peak = float(sigma_vis_jy / math.sqrt(2.0 * max_eff_vis))
    sigma_single_p1 = float(sigma_vis_jy / math.sqrt(2.0 * m1))
    m_mean = 0.5 * (m1 + m2)
    # Mosaic flatnoise thermal RMS across pb >= 0.2 footprint:
    # natural weighting has ~1.18 * m_mean effective visibilities, and Briggs robust=0.5
    # applies the standard VLA OSS 1.16..1.20 weighting noise multiplier (~148 uJy at 0.0527 Jy).
    sigma_nat_fov = float(sigma_vis_jy / math.sqrt(2.0 * 1.18 * m_mean))
    sigma_briggs_r05 = float(1.16 * sigma_nat_fov)

    pt_coord = SkyCoord(PT_COMP_RA, PT_COMP_DEC, frame="icrs")
    a1_pt = _evla_pb(pt_coord.separation(p1_coord).arcmin, nu_center_ghz)
    a2_pt = _evla_pb(pt_coord.separation(p2_coord).arcmin, nu_center_ghz)
    pt_mosaic_pb = math.sqrt((m1 * (a1_pt**2) + m2 * (a2_pt**2)) / max(max_eff_vis, 1e-12))
    pt_flatnoise_pb = (m1 * (a1_pt**2) + m2 * (a2_pt**2)) / max(m1 + m2, 1e-12)
    return {
        "sigma_th_jy_per_beam": sigma_briggs_r05,
        "sigma_th_natural_jy_per_beam": sigma_nat_fov,
        "sigma_th_natural_peak_jy_per_beam": sigma_nat_peak,
        "sigma_th_single_pointing_jy_per_beam": sigma_single_p1,
        "pt_mosaic_pb_response": float(pt_mosaic_pb),
        "pt_mosaic_flatnoise_response": float(pt_flatnoise_pb),
        "clean_target_p1_integrations": float(n_int_p1),
        "clean_target_p2_integrations": float(n_int_p2),
        "clean_baselines": float(n_bl),
        "clean_channels": float(n_chan_clean),
    }


def _evla_pb(theta_arcmin: float, freq_ghz: float) -> float:
    """EVLA C-band primary beam power response (Perley polynomial)."""
    x2 = (theta_arcmin * freq_ghz) ** 2
    pb = 1.0 - 1.343e-3 * x2 + 6.579e-7 * (x2**2) - 1.186e-10 * (x2**3)
    return float(max(pb, 0.0))


def compute_field_min_elevation_deg(ms_path: Path, field_id: int) -> float:
    """Compute minimum elevation (deg) across rows of a field using astropy AltAz."""
    configure_builder_casa_env()
    from casatools import table

    tb_ant = table()
    opened = False
    try:
        tb_ant.open(str(ms_path / "ANTENNA"), nomodify=True)
        opened = True
        pos = tb_ant.getcol("POSITION")
    finally:
        try:
            if opened:
                tb_ant.close()
        finally:
            tb_ant.done()

    mean_xyz = np.mean(pos, axis=1)
    loc = EarthLocation.from_geocentric(mean_xyz[0] * u.m, mean_xyz[1] * u.m, mean_xyz[2] * u.m)

    tb_fld = table()
    opened = False
    try:
        tb_fld.open(str(ms_path / "FIELD"), nomodify=True)
        opened = True
        phase_dir = tb_fld.getcol("PHASE_DIR")
        ra_rad = float(phase_dir[0, 0, field_id])
        dec_rad = float(phase_dir[1, 0, field_id])
    finally:
        try:
            if opened:
                tb_fld.close()
        finally:
            tb_fld.done()

    tb_main = table()
    opened = False
    try:
        tb_main.open(str(ms_path), nomodify=True)
        opened = True
        fids = tb_main.getcol("FIELD_ID")
        times_mjd_s = tb_main.getcol("TIME")[fids == field_id]
    finally:
        try:
            if opened:
                tb_main.close()
        finally:
            tb_main.done()

    unique_mjd_days = np.unique(np.round(times_mjd_s, 3)) / 86400.0
    obstime = Time(unique_mjd_days, format="mjd", scale="utc")
    sky = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad, frame="icrs")
    altaz = sky.transform_to(AltAz(obstime=obstime, location=loc))
    return float(np.min(altaz.alt.deg))


def compute_field_min_elevation_casa_deg(ms_path: Path, field_id: int) -> float:
    """Cross-check minimum field elevation (deg) using casatools.measures AZELGEO."""
    configure_builder_casa_env()
    from casatools import measures, table

    tb_ant = table()
    opened = False
    try:
        tb_ant.open(str(ms_path / "ANTENNA"), nomodify=True)
        opened = True
        pos = tb_ant.getcol("POSITION")
    finally:
        try:
            if opened:
                tb_ant.close()
        finally:
            tb_ant.done()
    mean_xyz = np.mean(pos, axis=1)

    tb_fld = table()
    opened = False
    try:
        tb_fld.open(str(ms_path / "FIELD"), nomodify=True)
        opened = True
        phase_dir = tb_fld.getcol("PHASE_DIR")
        ra_rad = float(phase_dir[0, 0, field_id])
        dec_rad = float(phase_dir[1, 0, field_id])
    finally:
        try:
            if opened:
                tb_fld.close()
        finally:
            tb_fld.done()

    tb_main = table()
    opened = False
    try:
        tb_main.open(str(ms_path), nomodify=True)
        opened = True
        fids = tb_main.getcol("FIELD_ID")
        times_mjd_s = tb_main.getcol("TIME")[fids == field_id]
    finally:
        try:
            if opened:
                tb_main.close()
        finally:
            tb_main.done()

    unique_mjd_days = np.unique(np.round(times_mjd_s, 3)) / 86400.0
    me = measures()
    try:
        ant_pos = me.position(
            "ITRF",
            f"{mean_xyz[0]}m",
            f"{mean_xyz[1]}m",
            f"{mean_xyz[2]}m",
        )
        src_dir = me.direction("J2000", f"{ra_rad}rad", f"{dec_rad}rad")
        me.doframe(ant_pos)
        els_deg: list[float] = []
        for mjd_d in unique_mjd_days:
            me.doframe(me.epoch("UTC", f"{float(mjd_d)}d"))
            azel = me.measure(src_dir, "AZELGEO")
            els_deg.append(math.degrees(float(azel["m1"]["value"])))
        return float(min(els_deg))
    finally:
        me.done()


def assign_scan_intents(ms_path: Path, scan_intents: dict[int, str]) -> None:
    """Populate STATE subtable and MAIN.STATE_ID.

    Note: casacore C++ MSMetaData aborts with std::length_error if STATE.nrows() > 0
    while MAIN.STATE_ID contains -1. Unassigned scans ("") therefore point to a valid
    STATE row with OBS_MODE = "" (giving msmd.intentsforfield(fid) == [] cleanly).
    """
    configure_builder_casa_env()
    from casatools import table

    non_empty = sorted({intent for intent in scan_intents.values() if intent})
    has_unassigned = any(not intent for intent in scan_intents.values())
    unique_intents = non_empty + ([""] if has_unassigned else [])
    intent_to_state_id = {intent: idx for idx, intent in enumerate(unique_intents)}

    state_tb = table()
    opened = False
    try:
        state_tb.open(str(ms_path / "STATE"), nomodify=False)
        opened = True
        nrows = int(state_tb.nrows())
        if nrows < len(unique_intents):
            state_tb.addrows(len(unique_intents) - nrows)
        elif nrows > len(unique_intents):
            state_tb.removerows(list(range(len(unique_intents), nrows)))
        cols = set(state_tb.colnames())
        for idx, intent in enumerate(unique_intents):
            state_tb.putcell("OBS_MODE", idx, intent)
            if "CAL" in cols:
                has_cal = "CALIBRATE_" in intent
                state_tb.putcell("CAL", idx, 1.0 if has_cal else 0.0)
            if "LOAD" in cols:
                state_tb.putcell("LOAD", idx, 0.0)
            if "REF" in cols:
                state_tb.putcell("REF", idx, True)
            if "SIG" in cols:
                state_tb.putcell("SIG", idx, True)
            if "SUB_SCAN" in cols:
                state_tb.putcell("SUB_SCAN", idx, 0)
            if "FLAG_ROW" in cols:
                state_tb.putcell("FLAG_ROW", idx, False)
    finally:
        try:
            if opened:
                state_tb.close()
        finally:
            state_tb.done()

    main_tb = table()
    opened = False
    try:
        main_tb.open(str(ms_path), nomodify=False)
        opened = True
        scans = main_tb.getcol("SCAN_NUMBER")
        state_ids = np.array(
            [intent_to_state_id[scan_intents.get(int(s), "")] for s in scans],
            dtype=np.int32,
        )
        main_tb.putcol("STATE_ID", state_ids)
    finally:
        try:
            if opened:
                main_tb.close()
        finally:
            main_tb.done()


def create_base_noiseless_ms(
    ms_path: Path,
    *,
    point_source_flux_jy: float,
    work_dir: Path,
) -> dict[str, Any]:
    """Create a 10-antenna VLA-D MS with exact Perley-Butler 3C286 + phase cal + mosaic sky."""
    configure_builder_casa_env()
    from casatasks import clearcal, ft, setjy
    from casatools import componentlist, measures, simulator, table

    if ms_path.exists():
        shutil.rmtree(ms_path)
    ms_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, zs, diams, antnames = load_vla_d_subset()
    me = measures()
    try:
        vla_pos = me.observatory("VLA")

        sm = simulator()
        sm_opened = False
        try:
            sm.open(ms=str(ms_path))
            sm_opened = True
            sm.setconfig(
                telescopename="VLA",
                x=xs,
                y=ys,
                z=zs,
                dishdiameter=diams,
                mount=["ALT-AZ"] * len(xs),
                antname=antnames,
                coordsystem="global",
                referencelocation=vla_pos,
            )
            sm.setspwindow(
                spwname="Cband_spw0",
                freq=f"{SPW_START_HZ / 1e9:.6f}GHz",
                deltafreq=f"{SPW_CHAN_WIDTH_HZ / 1e6:.3f}MHz",
                freqresolution=f"{SPW_CHAN_WIDTH_HZ / 1e6:.3f}MHz",
                nchannels=SPW_NCHAN,
                stokes="RR RL LR LL",
            )
            sm.setfeed(mode="perfect R L", pol=[""])
            sm.setfield(
                sourcename=FLUX_CAL_NAME,
                sourcedirection=me.direction("J2000", FLUX_CAL_RA, FLUX_CAL_DEC),
            )
            sm.setfield(
                sourcename=PHASE_CAL_NAME,
                sourcedirection=me.direction("J2000", PHASE_CAL_RA, PHASE_CAL_DEC),
            )
            sm.setfield(
                sourcename=TARGET_P1_NAME,
                sourcedirection=me.direction("J2000", TARGET_P1_RA, TARGET_P1_DEC),
            )
            sm.setfield(
                sourcename=TARGET_P2_NAME,
                sourcedirection=me.direction("J2000", TARGET_P2_RA, TARGET_P2_DEC),
            )
            sm.setauto(autocorrwt=0.0)
            sm.settimes(
                integrationtime=f"{INTEGRATION_SEC:.1f}s",
                usehourangle=False,
                referencetime=me.epoch("UTC", REF_EPOCH_UTC),
            )
            for spec in SCAN_SCHEDULE:
                sm.observe(
                    sourcename=spec.field_name,
                    spwname="Cband_spw0",
                    starttime=f"{spec.start_sec:.1f}s",
                    stoptime=f"{spec.stop_sec:.1f}s",
                )
        finally:
            try:
                if sm_opened:
                    sm.close()
            finally:
                sm.done()

        # Assign scan numbers 1..13 explicitly according to time windows and compute full track span.
        tb_scan = table()
        opened = False
        try:
            tb_scan.open(str(ms_path), nomodify=False)
            opened = True
            times = tb_scan.getcol("TIME")
            intervals = tb_scan.getcol("INTERVAL")
            t0 = float(np.min(times))
            rel_t = times - t0
            scan_col = np.ones(len(times), dtype=np.int32)
            for spec in SCAN_SCHEDULE:
                mask = (rel_t >= spec.start_sec - 1.0) & (rel_t <= spec.stop_sec + 1.0)
                scan_col[mask] = spec.scan_number
            tb_scan.putcol("SCAN_NUMBER", scan_col)
            obs_t_min = float(np.min(times - 0.5 * intervals))
            obs_t_max = float(np.max(times + 0.5 * intervals))
        finally:
            try:
                if opened:
                    tb_scan.close()
            finally:
                tb_scan.done()

        # Update OBSERVATION.TIME_RANGE (shape (2, 1)) to span the full ~9090s track instead of Scan 1.
        tb_obs = table()
        opened = False
        try:
            tb_obs.open(str(ms_path / "OBSERVATION"), nomodify=False)
            opened = True
            tb_obs.putcol(
                "TIME_RANGE",
                np.array([[obs_t_min], [obs_t_max]], dtype=np.float64),
            )
        finally:
            try:
                if opened:
                    tb_obs.close()
            finally:
                tb_obs.done()

        assign_scan_intents(ms_path, {s.scan_number: s.intent for s in SCAN_SCHEDULE})

        # Initialize scratch columns and populate exact Perley-Butler 2017 MODEL_DATA for J1331+3030.
        clearcal(vis=str(ms_path), addmodel=True)
        setjy(
            vis=str(ms_path),
            field=FLUX_CAL_NAME,
            standard="Perley-Butler 2017",
            usescratch=True,
        )

        # Predict phase calibrator J1822-0938 into MODEL_DATA via ft().
        pcal_cl = work_dir / "phase_cal.cl"
        shutil.rmtree(pcal_cl, ignore_errors=True)
        cl = componentlist()
        try:
            cl.done()
            cl.addcomponent(
                dir=me.direction("J2000", PHASE_CAL_RA, PHASE_CAL_DEC),
                flux=PHASE_CAL_FLUX_JY,
                fluxunit="Jy",
                freq="4.600GHz",
                shape="point",
                spectrumtype="spectral index",
                index=PHASE_CAL_SPIX,
            )
            cl.rename(str(pcal_cl))
        finally:
            try:
                cl.close()
            finally:
                cl.done()
        ft(vis=str(ms_path), field=PHASE_CAL_NAME, complist=str(pcal_cl), usescratch=True)

        # Predict target mosaic sky (extended Gaussian + point source) WITH VLA primary beam via sm.setvp().
        tgt_cl = work_dir / "target_sky.cl"
        shutil.rmtree(tgt_cl, ignore_errors=True)
        cl2 = componentlist()
        try:
            cl2.done()
            cl2.addcomponent(
                dir=me.direction("J2000", EXT_COMP_RA, EXT_COMP_DEC),
                flux=EXT_COMP_FLUX_JY,
                fluxunit="Jy",
                freq="4.600GHz",
                shape="Gaussian",
                majoraxis=EXT_COMP_MAJOR,
                minoraxis=EXT_COMP_MINOR,
                positionangle=EXT_COMP_PA,
                spectrumtype="constant",
            )
            cl2.addcomponent(
                dir=me.direction("J2000", PT_COMP_RA, PT_COMP_DEC),
                flux=point_source_flux_jy,
                fluxunit="Jy",
                freq="4.600GHz",
                shape="point",
                spectrumtype="constant",
            )
            cl2.rename(str(tgt_cl))
        finally:
            try:
                cl2.close()
            finally:
                cl2.done()
    finally:
        me.done()

    # Save calibrator MODEL_DATA before sm.predict() (which overwrites MODEL_DATA across rows).
    tb_bak = table()
    opened = False
    try:
        tb_bak.open(str(ms_path), nomodify=True)
        opened = True
        fids = tb_bak.getcol("FIELD_ID")
        cal_model_backup = tb_bak.getcol("MODEL_DATA").copy()
    finally:
        try:
            if opened:
                tb_bak.close()
        finally:
            tb_bak.done()

    sm2 = simulator()
    sm2_opened = False
    try:
        sm2.openfromms(str(ms_path))
        sm2_opened = True
        sm2.setvp(dovp=True, usedefaultvp=True, dosquint=False)
        sm2.predict(complist=str(tgt_cl), incremental=False)
    finally:
        try:
            if sm2_opened:
                sm2.close()
        finally:
            sm2.done()

    tb_out = table()
    opened = False
    try:
        tb_out.open(str(ms_path), nomodify=False)
        opened = True
        # sm2.predict(incremental=False) writes predicted target visibilities into DATA!
        true_vis = tb_out.getcol("DATA")
        cal_mask = np.isin(fids, [0, 1])
        true_vis[:, :, cal_mask] = cal_model_backup[:, :, cal_mask]
        tb_out.putcol("DATA", true_vis)
        # Remove scratch columns so raw archival-like MSs carry only DATA (no leakage).
        scratch_cols = [c for c in ("CORRECTED_DATA", "MODEL_DATA") if c in tb_out.colnames()]
        if scratch_cols:
            tb_out.removecols(scratch_cols)

        j1331_par_spec = np.mean(
            0.5
            * (
                np.abs(cal_model_backup[0, :, fids == 0])
                + np.abs(cal_model_backup[3, :, fids == 0])
            ),
            axis=0,
        )
        j1331_par_amp = float(np.mean(j1331_par_spec))
        clean_ch = [c for c in range(SPW_NCHAN) if c not in RFI_CHANNELS]
        pcal_par_spec = np.mean(
            0.5
            * (
                np.abs(cal_model_backup[0, :, fids == 1])
                + np.abs(cal_model_backup[3, :, fids == 1])
            ),
            axis=0,
        )
        pcal_spw_mean_amp = float(np.mean(pcal_par_spec[clean_ch]))
    finally:
        try:
            if opened:
                tb_out.close()
        finally:
            tb_out.done()

    shutil.rmtree(pcal_cl, ignore_errors=True)
    shutil.rmtree(tgt_cl, ignore_errors=True)

    clear_ms_pointing(ms_path)
    clear_ms_history(ms_path)
    bp_min_el = compute_field_min_elevation_deg(ms_path, field_id=0)
    bp_min_el_casa = compute_field_min_elevation_casa_deg(ms_path, field_id=0)
    verify_ms_unlocked(ms_path)
    return {
        "j1331_pb2017_mean_amp_jy": round(j1331_par_amp, 6),
        "j1331_pb2017_per_chan_amp_jy": [round(float(v), 6) for v in j1331_par_spec],
        "pcal_spw_mean_amp_jy": round(pcal_spw_mean_amp, 6),
        "bp_min_elevation_deg": round(bp_min_el, 4),
        "bp_min_elevation_casa_azelgeo_deg": round(bp_min_el_casa, 4),
        "obs_start_mjd_s": round(obs_t_min, 3),
        "obs_end_mjd_s": round(obs_t_max, 3),
        "obs_duration_s": round(obs_t_max - obs_t_min, 2),
    }


def apply_corruptions_to_ms(
    ms_path: Path,
    *,
    seed: int = 20261113,
    sigma_vis_jy: float = 0.035,
    dead_ant_mode: str = "zero",
    include_slew_quack: bool = False,
    masks_out_npz: Path | None = None,
    target_vis_out_npz: Path | None = None,
) -> dict[str, Any]:
    """Apply seeded delays, bandpass, time-varying gains, thermal noise, quack, RFI, and dead ant."""
    configure_builder_casa_env()
    from casatools import table

    rng = np.random.default_rng(seed)
    nant = len(SYNTH_ANTENNA_NAMES)

    tb_spw = table()
    opened = False
    try:
        tb_spw.open(str(ms_path / "SPECTRAL_WINDOW"), nomodify=True)
        opened = True
        freqs_hz = tb_spw.getcol("CHAN_FREQ")[:, 0]
    finally:
        try:
            if opened:
                tb_spw.close()
        finally:
            tb_spw.done()

    nu0_hz = float(np.mean(freqs_hz))
    dnu_hz = freqs_hz - nu0_hz
    norm_ch = np.linspace(-1.0, 1.0, len(freqs_hz))

    # Build per-antenna frequency response: delay phase + smooth complex bandpass.
    bp_jones = np.ones((nant, 2, len(freqs_hz)), dtype=np.complex128)
    for ant_id, ant_name in enumerate(SYNTH_ANTENNA_NAMES):
        tau_s = TRUE_DELAYS_NS[ant_name] * 1e-9
        delay_phase = np.exp(-2j * math.pi * dnu_hz * tau_s)
        for pol_idx in range(2):
            if ant_name == REFANT_NAME:
                amp_poly = 1.0 + 0.04 * (norm_ch**2 - 0.33)
                phs_deg = np.zeros_like(norm_ch)
            else:
                c1 = float(rng.uniform(-0.06, 0.06))
                c2 = float(rng.uniform(-0.05, 0.05))
                amp_poly = 1.0 + c1 * norm_ch + c2 * (norm_ch**2)
                p1 = float(rng.uniform(-8.0, 8.0))
                p2 = float(rng.uniform(-10.0, 10.0))
                phs_deg = p1 * norm_ch + p2 * (norm_ch**2 - 0.33)
            bp_jones[ant_id, pol_idx, :] = amp_poly * np.exp(1j * np.deg2rad(phs_deg)) * delay_phase

    tb = table()
    opened = False
    try:
        tb.open(str(ms_path), nomodify=False)
        opened = True
        true_vis = tb.getcol("DATA").copy()  # shape: (4, nchan, nrow)
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        times = tb.getcol("TIME")
        scans = tb.getcol("SCAN_NUMBER")
        fids = tb.getcol("FIELD_ID")

        npol, nchan, nrow = true_vis.shape
        t_min, t_max = float(np.min(times)), float(np.max(times))
        t_norm = (times - t_min) / max(t_max - t_min, 1.0)

        # Raw instrument scale (~0.42 -> g_i g_j^* ~ 0.18, realistic VLA uncalibrated raw scale)
        raw_amp_base = 0.42 * rng.uniform(0.92, 1.08, size=(nant, 2))
        phs_offset_deg = rng.uniform(-35.0, 35.0, size=(nant, 2))
        phs_slope_deg = rng.uniform(-22.0, 22.0, size=(nant, 2))
        phs_offset_deg[0, :] = 0.0
        phs_slope_deg[0, :] = 0.0

        obs_vis = np.zeros_like(true_vis)
        # Corr order for circular feeds: 0=RR (R1*R2^H), 1=RL (R1*L2^H), 2=LR (L1*R2^H), 3=LL (L1*L2^H)
        corr_pol_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # Precompute time-dependent complex gains shape (nant, 2, nrow)
        g_time = np.empty((nant, 2, nrow), dtype=np.complex128)
        for a in range(nant):
            for p in range(2):
                amp_t = raw_amp_base[a, p] * (1.0 + 0.015 * np.sin(2.0 * math.pi * t_norm))
                phs_t = phs_offset_deg[a, p] + phs_slope_deg[a, p] * t_norm
                g_time[a, p, :] = amp_t * np.exp(1j * np.deg2rad(phs_t))

        row_idx = np.arange(nrow)
        for corr_idx, (p1, p2) in enumerate(corr_pol_pairs):
            bp_pair = bp_jones[ant1, p1, :].T * np.conj(bp_jones[ant2, p2, :].T)
            g_pair = (g_time[ant1, p1, row_idx] * np.conj(g_time[ant2, p2, row_idx]))[np.newaxis, :]
            obs_vis[corr_idx, :, :] = true_vis[corr_idx, :, :] * bp_pair * g_pair

        # Add complex Gaussian visibility noise (per-real/imag RMS = sigma_vis_jy after calibration)
        raw_flux_scale = float(np.mean(raw_amp_base) ** 2)
        sigma_raw = sigma_vis_jy * raw_flux_scale
        noise = rng.normal(0.0, sigma_raw, size=obs_vis.shape) + 1j * rng.normal(
            0.0, sigma_raw, size=obs_vis.shape
        )
        obs_vis += noise

        # Build exact ground-truth corruption mask (shape: (nchan, nrow), True where corrupted)
        corrupted_chan_row = np.zeros((nchan, nrow), dtype=bool)

        # 1. Scan-start quack corruption:
        #    - First 20s of Scan 1 has wild jagged settling spikes & random phase
        #    - Optionally first 10s integration of Scans 2..13 has source-slew decorrelation
        scan1_times = np.unique(times[scans == 1])
        scan1_quack_rows = (scans == 1) & np.isin(times, scan1_times[:2])
        n_s1_quack = int(np.sum(scan1_quack_rows))
        obs_vis[:, :, scan1_quack_rows] *= rng.uniform(
            12.0, 32.0, size=(npol, nchan, n_s1_quack)
        ) * np.exp(1j * rng.uniform(-math.pi, math.pi, size=(npol, nchan, n_s1_quack)))

        quack_rows = scan1_quack_rows.copy()
        if include_slew_quack:
            for s_num in np.unique(scans):
                if int(s_num) == 1:
                    continue
                s_times = np.unique(times[scans == s_num])
                slew_mask = (scans == s_num) & (np.abs(times - s_times[0]) < 0.5)
                n_slew = int(np.sum(slew_mask))
                obs_vis[:, :, slew_mask] *= 0.15 * np.exp(
                    1j * rng.uniform(-math.pi, math.pi, size=(npol, nchan, n_slew))
                )
                quack_rows |= slew_mask
        corrupted_chan_row[:, quack_rows] = True

        # 2. Narrowband persistent RFI in channels 29..31 across all scans
        for ch in RFI_CHANNELS:
            obs_vis[:, ch, :] += 18.0 * np.exp(
                1j * rng.uniform(-math.pi, math.pi, size=(npol, nrow))
            )
            corrupted_chan_row[ch, :] = True

        # 3. Time-localized broadband RFI burst in Scan 5 middle integration
        scan5_times = np.unique(times[scans == 5])
        burst_time = scan5_times[len(scan5_times) // 2]
        burst_rows = (scans == 5) & (np.abs(times - burst_time) < 0.5)
        n_burst = int(np.sum(burst_rows))
        obs_vis[:, :, burst_rows] += rng.uniform(22.0, 38.0, size=(npol, nchan, n_burst)) * np.exp(
            1j * rng.uniform(-math.pi, math.pi, size=(npol, nchan, n_burst))
        )
        corrupted_chan_row[:, burst_rows] = True

        # 4. Dead antenna ea07
        dead_rows = (ant1 == DEAD_ANTENNA_ID) | (ant2 == DEAD_ANTENNA_ID)
        n_dead = int(np.sum(dead_rows))
        if dead_ant_mode == "zero":
            obs_vis[:, :, dead_rows] = 0.0 + 0.0j
        elif dead_ant_mode == "near_zero":
            # Near-zero dead frontend micro-noise (~1.4e-4 Jy): non-zero, SNR ~ 0.001 in gaincal!
            obs_vis[:, :, dead_rows] = rng.normal(
                0.0, 1.0e-4, size=(npol, nchan, n_dead)
            ) + 1j * rng.normal(0.0, 1.0e-4, size=(npol, nchan, n_dead))
        else:
            obs_vis[:, :, dead_rows] = rng.normal(
                0.0, sigma_raw * 2.0, size=(npol, nchan, n_dead)
            ) + 1j * rng.normal(0.0, sigma_raw * 2.0, size=(npol, nchan, n_dead))
        corrupted_chan_row[:, dead_rows] = True

        tb.putcol("DATA", obs_vis)
        weights = np.full((npol, nrow), 1.0 / (sigma_vis_jy**2), dtype=np.float32)
        tb.putcol("WEIGHT", weights)
        tb.putcol("SIGMA", np.full((npol, nrow), sigma_vis_jy, dtype=np.float32))
        tb.putcol("FLAG", np.zeros((npol, nchan, nrow), dtype=bool))
        tb.putcol("FLAG_ROW", np.zeros(nrow, dtype=bool))

        clean_ch_mask = np.ones(nchan, dtype=bool)
        clean_ch_mask[RFI_CHANNELS] = False
        target_rows_mask = np.isin(fids, [2, 3])
        target_clean_rows = target_rows_mask & (~dead_rows) & (~burst_rows) & (~quack_rows)
        true_target_vis_mean_amp = float(
            np.mean(
                0.5
                * (
                    np.abs(true_vis[0, clean_ch_mask, :][:, target_clean_rows])
                    + np.abs(true_vis[3, clean_ch_mask, :][:, target_clean_rows])
                )
            )
        )

        if masks_out_npz is not None:
            masks_out_npz.parent.mkdir(parents=True, exist_ok=True)
            tmp_masks = masks_out_npz.with_name(f"{masks_out_npz.stem}.tmp.npz")
            np.savez_compressed(
                tmp_masks,
                corrupted_chan_row=corrupted_chan_row,
                quack_rows=quack_rows,
                scan1_quack_rows=scan1_quack_rows,
                burst_rows=burst_rows,
                dead_rows=dead_rows,
                rfi_channels=np.array(RFI_CHANNELS, dtype=np.int32),
            )
            os.replace(tmp_masks, masks_out_npz)

        if target_vis_out_npz is not None:
            target_vis_out_npz.parent.mkdir(parents=True, exist_ok=True)
            tgt_row_idx = np.flatnonzero(target_rows_mask).astype(np.int32)
            tmp_tvis = target_vis_out_npz.with_name(f"{target_vis_out_npz.stem}.tmp.npz")
            np.savez_compressed(
                tmp_tvis,
                target_row_indices=tgt_row_idx,
                clean_target_row_mask=target_clean_rows[target_rows_mask],
                clean_chan_mask=clean_ch_mask,
                true_rr=true_vis[0, :, :][:, target_rows_mask].astype(np.complex64),
                true_ll=true_vis[3, :, :][:, target_rows_mask].astype(np.complex64),
            )
            os.replace(tmp_tvis, target_vis_out_npz)
    finally:
        try:
            if opened:
                tb.close()
        finally:
            tb.done()

    clear_ms_history(ms_path)
    verify_ms_unlocked(ms_path)
    return {
        "n_rows": int(nrow),
        "n_chan": int(nchan),
        "n_corrupted_chan_rows": int(np.sum(corrupted_chan_row)),
        "n_clean_chan_rows": int(np.sum(~corrupted_chan_row)),
        "quack_row_indices": np.flatnonzero(quack_rows).tolist(),
        "scan1_quack_row_indices": np.flatnonzero(scan1_quack_rows).tolist(),
        "burst_row_indices": np.flatnonzero(burst_rows).tolist(),
        "dead_row_indices": np.flatnonzero(dead_rows).tolist(),
        "rfi_channels": list(RFI_CHANNELS),
        "sigma_vis_jy": sigma_vis_jy,
        "true_target_vis_mean_amp_jy": round(true_target_vis_mean_amp, 6),
    }
