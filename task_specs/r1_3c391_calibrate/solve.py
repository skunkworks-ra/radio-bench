"""Standalone CASA 6.7.6.14 oracle solver for r1_3c391_calibrate (calibrate)."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _configure_temp_env(workspace: Path) -> tuple[Path, bool]:
    """Configure non-/tmp scratch directories before importing CASA."""
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur_tmp = os.environ.get("TMPDIR", "").strip()
    if not cur_tmp or cur_tmp == "/tmp" or cur_tmp.startswith("/tmp/"):
        fast_root = Path("/mnt/fast/radio-bench/tmp")
        var_tmp = Path("/var/tmp")
        if fast_root.parent.exists() and os.access(fast_root.parent, os.W_OK):
            scratch = fast_root / f"solver_{os.getpid()}"
            os.environ.setdefault("CASASITECONFIG", "/mnt/fast/radio-bench/casaconfig_local.py")
        elif var_tmp.exists() and os.access(var_tmp, os.W_OK):
            scratch = var_tmp / f"solver_{os.getpid()}"
        else:
            scratch = workspace / f".solver_scratch_{os.getpid()}"
    else:
        scratch = Path(cur_tmp) / f"solver_{os.getpid()}"

    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)

    cur_mpl = os.environ.get("MPLCONFIGDIR", "").strip()
    if not cur_mpl or cur_mpl == "/tmp" or cur_mpl.startswith("/tmp/"):
        mpl_dir = scratch / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    if not os.access(os.environ.get("HOME", "/"), os.W_OK):
        os.environ["HOME"] = str(scratch)

    return scratch, True


# Known VLA primary flux density calibrators (J2000 radians): 3C286, 3C48, 3C147, 3C138.
_KNOWN_VLA_FLUX_CALS_RAD: tuple[tuple[float, float], ...] = (
    (math.radians(202.784533), math.radians(30.509155)),
    (math.radians(24.422081), math.radians(33.159759)),
    (math.radians(85.650575), math.radians(49.852009)),
    (math.radians(80.291192), math.radians(16.639458)),
)


def _ang_sep_rad(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    cos_d = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    return math.acos(max(-1.0, min(1.0, cos_d)))


def _next_fft_size(n: int) -> int:
    cand = max(64, int(math.ceil(n / 8.0) * 8))
    while True:
        rem = cand
        for p in (2, 3, 5):
            while rem % p == 0:
                rem //= p
        if rem == 1 and cand % 8 == 0:
            return cand
        cand += 8


@contextlib.contextmanager
def _open_table(path: Path | str, *, nomodify: bool = True) -> Iterator[Any]:
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


def _verify_unlocked(path: Path) -> None:
    import casatasks

    casatasks.clearstat()
    with _open_table(path, nomodify=True) as tb:
        if tb.ismultiused(checksubtables=True):
            raise RuntimeError(f"Active casacore table lock remains on {path}")


def _inspect_ms(ms_path: Path) -> dict[str, Any]:
    """Dynamically determine field roles, dead antennas, refant, and bandpass min elevation."""
    import numpy as np
    from casatools import measures, quanta

    with _open_table(ms_path / "FIELD", nomodify=True) as tb_f:
        field_names = [str(x) for x in tb_f.getcol("NAME")]
        phase_dir = tb_f.getcol("PHASE_DIR")

    with _open_table(ms_path / "ANTENNA", nomodify=True) as tb_a:
        ant_names = [str(x) for x in tb_a.getcol("NAME")]
        ant_pos = tb_a.getcol("POSITION")
        dish_diam = float(np.median(tb_a.getcol("DISH_DIAMETER")))

    with _open_table(ms_path / "SPECTRAL_WINDOW", nomodify=True) as tb_s:
        chan_freqs_hz = tb_s.getcol("CHAN_FREQ")[:, 0].astype(float)

    with _open_table(ms_path, nomodify=True) as tb_m:
        fids = tb_m.getcol("FIELD_ID")
        scans = tb_m.getcol("SCAN_NUMBER")
        times = tb_m.getcol("TIME")
        ant1 = tb_m.getcol("ANTENNA1")
        ant2 = tb_m.getcol("ANTENNA2")
        data = tb_m.getcol("DATA")

    coords = [
        (float(phase_dir[0, 0, i]) % (2.0 * math.pi), float(phase_dir[1, 0, i]))
        for i in range(len(field_names))
    ]
    nu_center_hz = float(np.mean(chan_freqs_hz))
    pb_fwhm_rad = 1.02 * (299792458.0 / nu_center_hz) / dish_diam
    mosaic_sep_limit = 1.25 * pb_fwhm_rad

    flux_fid: int | None = None
    for fid, (ra_r, dec_r) in enumerate(coords):
        for kra, kdec in _KNOWN_VLA_FLUX_CALS_RAD:
            if _ang_sep_rad(ra_r, dec_r, kra, kdec) < math.radians(2.0 / 60.0):
                flux_fid = fid
                break
        if flux_fid is not None:
            break
    if flux_fid is None:
        flux_fid = 0
    bp_fid = flux_fid

    remaining = [fid for fid in range(len(field_names)) if fid != flux_fid]
    target_fids = sorted(
        fid
        for fid in remaining
        if any(
            other != fid and _ang_sep_rad(*coords[fid], *coords[other]) < mosaic_sep_limit
            for other in remaining
        )
    )
    if not target_fids:
        target_fids = [remaining[-1]]

    sin_ra = float(np.mean([math.sin(coords[i][0]) for i in target_fids]))
    cos_ra = float(np.mean([math.cos(coords[i][0]) for i in target_fids]))
    tgt_mean_ra = math.atan2(sin_ra, cos_ra) % (2.0 * math.pi)
    tgt_mean_dec = float(np.mean([coords[i][1] for i in target_fids]))

    other_cals = [fid for fid in remaining if fid not in target_fids]
    multi_scan_cals = [fid for fid in other_cals if len(np.unique(scans[fids == fid])) >= 2]
    phase_candidates = multi_scan_cals if multi_scan_cals else other_cals
    phase_fid = min(
        phase_candidates,
        key=lambda fid: _ang_sep_rad(*coords[fid], tgt_mean_ra, tgt_mean_dec),
    )
    aux_cal_fids = [fid for fid in other_cals if fid != phase_fid]

    bp_scans = sorted(int(s) for s in np.unique(scans[fids == bp_fid]))
    good_bp_scans = [s for s in bp_scans if s != 1] if len(bp_scans) > 1 else bp_scans
    bp_rows = (fids == bp_fid) & np.isin(scans, good_bp_scans) & (ant1 != ant2)
    par_amp = 0.5 * (np.abs(data[0, :, :]) + np.abs(data[3, :, :]))
    ant_meds: dict[str, float] = {}
    for aid, aname in enumerate(ant_names):
        mask_a = bp_rows & ((ant1 == aid) | (ant2 == aid))
        vals = par_amp[:, mask_a]
        ant_meds[aname] = float(np.median(vals)) if vals.size > 0 else 0.0
    arr_med = float(np.median(list(ant_meds.values())))
    dead_antennas = sorted(aname for aname, med_v in ant_meds.items() if med_v < 0.05 * arr_med)
    dead_ids = [ant_names.index(a) for a in dead_antennas]

    live_ids = [i for i, a in enumerate(ant_names) if a not in dead_antennas]
    strong_live_ids = [i for i in live_ids if ant_meds[ant_names[i]] >= 0.88 * arr_med] or live_ids
    med_xyz = np.median(ant_pos[:, live_ids], axis=1)
    best_aid = min(
        strong_live_ids,
        key=lambda i: float(np.linalg.norm(ant_pos[:, i] - med_xyz)),
    )
    refant = ant_names[best_aid]

    me = measures()
    qa = quanta()
    try:
        me.doframe(me.observatory("VLA"))
        sky_dir = me.direction(
            "J2000",
            qa.quantity(coords[bp_fid][0], "rad"),
            qa.quantity(coords[bp_fid][1], "rad"),
        )
        bp_times = np.unique(np.round(times[fids == bp_fid], 3))
        els_deg: list[float] = []
        for t_sec in bp_times:
            me.doframe(me.epoch("UTC", qa.quantity(float(t_sec) / 86400.0, "d")))
            azel = me.measure(sky_dir, "AZELGEO")
            els_deg.append(float(qa.convert(azel["m1"], "deg")["value"]))
        bp_min_el_deg = float(min(els_deg))
    finally:
        with contextlib.suppress(Exception):
            qa.done()
        with contextlib.suppress(Exception):
            me.done()

    unique_times = np.sort(np.unique(np.round(times, 3)))
    dt = np.diff(unique_times)
    pos_dt = dt[dt > 0.1]
    integration_sec = float(np.median(pos_dt)) if pos_dt.size > 0 else 10.0

    scan1_times = np.unique(np.round(times[scans == 1], 3))
    scan2_fids = np.unique(fids[scans == 2])
    scan1_fids = np.unique(fids[scans == 1])
    scan1_is_short_setup = (
        len(scan1_times) <= 2
        and len(scan1_fids) == 1
        and len(scan2_fids) == 1
        and int(scan1_fids[0]) == int(scan2_fids[0])
    )

    dists_to_center = [_ang_sep_rad(*coords[fid], tgt_mean_ra, tgt_mean_dec) for fid in target_fids]
    min_dist_idx = int(np.argmin(dists_to_center))
    if dists_to_center[min_dist_idx] < math.radians(5.0 / 3600.0):
        pc_ra, pc_dec = coords[target_fids[min_dist_idx]]
    else:
        pc_ra, pc_dec = tgt_mean_ra, tgt_mean_dec
    phasecenter = f"J2000 {math.degrees(pc_ra):.8f}deg {math.degrees(pc_dec):.8f}deg"

    live_bp_rows = bp_rows & (~np.isin(ant1, dead_ids)) & (~np.isin(ant2, dead_ids))
    chan_meds = np.median(par_amp[:, live_bp_rows], axis=1)
    overall_chan_med = float(np.median(chan_meds))
    nchan = len(chan_freqs_hz)
    c_lo = max(2, nchan // 2 - 6)
    c_hi = min(nchan - 2, nchan // 2 + 6)
    clean_center_chans = [
        ch for ch in range(c_lo, c_hi) if float(chan_meds[ch]) < 2.0 * overall_chan_med
    ]
    if clean_center_chans:
        center_spw = "0:" + ";".join(str(ch) for ch in clean_center_chans)
    else:
        center_spw = f"0:{c_lo}~{c_hi - 1}"

    return {
        "field_names": field_names,
        "ant_names": ant_names,
        "chan_freqs_hz": chan_freqs_hz,
        "nchan": nchan,
        "flux_fid": flux_fid,
        "bp_fid": bp_fid,
        "phase_fid": phase_fid,
        "aux_cal_fids": aux_cal_fids,
        "target_fids": target_fids,
        "flux_calibrator": field_names[flux_fid],
        "bandpass_calibrator": field_names[bp_fid],
        "phase_calibrator": field_names[phase_fid],
        "aux_calibrators": [field_names[i] for i in aux_cal_fids],
        "target_fields": [field_names[i] for i in target_fids],
        "dead_antennas": dead_antennas,
        "dead_ids": dead_ids,
        "refant": refant,
        "bandpass_min_elevation_deg": round(bp_min_el_deg, 4),
        "integration_sec": integration_sec,
        "scan1_is_short_setup": scan1_is_short_setup,
        "bp_median_raw_amp_jy": arr_med,
        "center_spw": center_spw,
        "phasecenter": phasecenter,
        "dish_diam_m": dish_diam,
    }


def _assign_scan_intents(ms_path: Path, info: dict[str, Any]) -> None:
    """Populate STATE subtable and MAIN.STATE_ID cleanly via casatools.table."""
    import numpy as np

    flux_intent = (
        "CALIBRATE_BANDPASS#ON_SOURCE,CALIBRATE_FLUX#ON_SOURCE,CALIBRATE_POL_ANGLE#ON_SOURCE"
        if info["aux_cal_fids"]
        else "CALIBRATE_BANDPASS#ON_SOURCE,CALIBRATE_FLUX#ON_SOURCE"
    )
    field_intent_map: dict[int, str] = {
        int(info["flux_fid"]): flux_intent,
        int(info["phase_fid"]): "CALIBRATE_PHASE#ON_SOURCE",
    }
    for afid in info["aux_cal_fids"]:
        field_intent_map[int(afid)] = "CALIBRATE_POL_LEAKAGE#ON_SOURCE"
    for tfid in info["target_fids"]:
        field_intent_map[int(tfid)] = "OBSERVE_TARGET#ON_SOURCE"

    unique_intents = sorted(set(field_intent_map.values()))
    intent_to_state_id = {intent: idx for idx, intent in enumerate(unique_intents)}

    with _open_table(ms_path / "STATE", nomodify=False) as state_tb:
        nrows = int(state_tb.nrows())
        if nrows < len(unique_intents):
            state_tb.addrows(len(unique_intents) - nrows)
        elif nrows > len(unique_intents):
            state_tb.removerows(list(range(len(unique_intents), nrows)))
        cols = set(state_tb.colnames())
        for idx, intent in enumerate(unique_intents):
            state_tb.putcell("OBS_MODE", idx, intent)
            if "CAL" in cols:
                state_tb.putcell("CAL", idx, 1.0 if "CALIBRATE_" in intent else 0.0)
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

    with _open_table(ms_path, nomodify=False) as main_tb:
        fids = main_tb.getcol("FIELD_ID")
        state_ids = np.array(
            [intent_to_state_id[field_intent_map[int(fid)]] for fid in fids],
            dtype=np.int32,
        )
        main_tb.putcol("STATE_ID", state_ids)


def _run_calibrate_stage(workspace: Path, info: dict[str, Any]) -> None:
    import numpy as np
    from casatasks import applycal, bandpass, flagdata, fluxscale, gaincal

    ms_path = workspace / "input.ms"
    cal_ms_path = workspace / "calibrators.ms"
    has_cal_ms = (cal_ms_path / "table.dat").is_file()

    if info["dead_antennas"]:
        with contextlib.suppress(RuntimeError):
            flagdata(
                vis=str(ms_path),
                mode="manual",
                antenna=",".join(info["dead_antennas"]),
                flagbackup=False,
            )
        if has_cal_ms:
            with contextlib.suppress(RuntimeError):
                flagdata(
                    vis=str(cal_ms_path),
                    mode="manual",
                    antenna=",".join(info["dead_antennas"]),
                    flagbackup=False,
                )

    solve_vis = ms_path
    if has_cal_ms:
        with _open_table(cal_ms_path / "FIELD", nomodify=True) as tb_cf:
            cf_names = [str(x).strip().upper() for x in tb_cf.getcol("NAME")]
        flux_up = str(info["flux_calibrator"]).strip().upper()
        if flux_up in cf_names:
            cf_fid = cf_names.index(flux_up)
            with _open_table(cal_ms_path, nomodify=True) as tb_cm:
                if "MODEL_DATA" in tb_cm.colnames():
                    sub_m = tb_cm.query(f"FIELD_ID == {cf_fid}")
                    try:
                        if int(sub_m.nrows()) > 0:
                            mod_arr = sub_m.getcol("MODEL_DATA")
                            if float(np.max(np.abs(mod_arr))) > 1e-6:
                                solve_vis = cal_ms_path
                    finally:
                        with contextlib.suppress(Exception):
                            sub_m.close()
                        with contextlib.suppress(Exception):
                            sub_m.done()

    cal_dir = workspace / "caltables"
    shutil.rmtree(cal_dir, ignore_errors=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    g0_tab = cal_dir / "G0.cal"
    k_tab = cal_dir / "K.cal"
    b_tab = cal_dir / "B.cal"
    g_tab = cal_dir / "G.cal"
    flux_tab = cal_dir / "fluxscale.cal"

    all_cals = [info["flux_calibrator"], info["phase_calibrator"], *info["aux_calibrators"]]
    all_cal_sel = ",".join(all_cals)
    transfer_cals = [info["phase_calibrator"], *info["aux_calibrators"]]
    bp_cal = info["bandpass_calibrator"]
    nchan = int(info["nchan"])
    edge = max(2, nchan // 16)
    gain_spw = f"0:{edge}~{nchan - 1 - edge}"

    gaincal(
        vis=str(solve_vis),
        caltable=str(g0_tab),
        field=all_cal_sel,
        spw=info["center_spw"],
        solint="int",
        refant=info["refant"],
        minsnr=3.0,
        calmode="p",
    )
    gaincal(
        vis=str(solve_vis),
        caltable=str(k_tab),
        field=bp_cal,
        solint="inf",
        combine="scan",
        refant=info["refant"],
        minsnr=3.0,
        gaintype="K",
        gaintable=[str(g0_tab)],
        gainfield=[bp_cal],
    )
    bandpass(
        vis=str(solve_vis),
        caltable=str(b_tab),
        field=bp_cal,
        solint="inf",
        combine="scan",
        refant=info["refant"],
        minsnr=3.0,
        bandtype="B",
        fillgaps=3,
        solnorm=True,
        gaintable=[str(g0_tab), str(k_tab)],
        gainfield=[bp_cal, bp_cal],
    )
    gaincal(
        vis=str(solve_vis),
        caltable=str(g_tab),
        field=all_cal_sel,
        spw=gain_spw,
        solint="inf",
        refant=info["refant"],
        minsnr=3.0,
        calmode="ap",
        gaintable=[str(k_tab), str(b_tab)],
        gainfield=[bp_cal, bp_cal],
    )
    fluxscale(
        vis=str(solve_vis),
        caltable=str(g_tab),
        fluxtable=str(flux_tab),
        reference=[info["flux_calibrator"]],
        transfer=transfer_cals,
        incremental=False,
    )

    applycal(
        vis=str(ms_path),
        field=info["flux_calibrator"],
        gaintable=[str(k_tab), str(b_tab), str(flux_tab)],
        gainfield=[bp_cal, bp_cal, info["flux_calibrator"]],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )
    applycal(
        vis=str(ms_path),
        field=info["phase_calibrator"],
        gaintable=[str(k_tab), str(b_tab), str(flux_tab)],
        gainfield=[bp_cal, bp_cal, info["phase_calibrator"]],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )
    for aux_cal in info["aux_calibrators"]:
        applycal(
            vis=str(ms_path),
            field=aux_cal,
            gaintable=[str(k_tab), str(b_tab), str(flux_tab)],
            gainfield=[bp_cal, bp_cal, aux_cal],
            interp=["nearest", "nearest,linear", "linear"],
            calwt=False,
            flagbackup=False,
        )

    target_sel = ",".join(str(fid) for fid in info["target_fids"])
    applycal(
        vis=str(ms_path),
        field=target_sel,
        gaintable=[str(k_tab), str(b_tab), str(flux_tab)],
        gainfield=[bp_cal, bp_cal, info["phase_calibrator"]],
        interp=["nearest", "nearest,linear", "linear"],
        calwt=False,
        flagbackup=False,
    )

    flagdata(
        vis=str(ms_path),
        mode="rflag",
        field=target_sel,
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )

    with _open_table(ms_path, nomodify=False) as tb_ms:
        if "MODEL_DATA" in tb_ms.colnames():
            fids = tb_ms.getcol("FIELD_ID")
            model = tb_ms.getcol("MODEL_DATA")
            tgt_arr = np.isin(fids, list(info["target_fids"]))
            model[:, :, tgt_arr] = 0.0 + 0.0j
            model[0, :, tgt_arr] = 1.0 + 0.0j
            model[3, :, tgt_arr] = 1.0 + 0.0j
            tb_ms.putcol("MODEL_DATA", model)

    if has_cal_ms:
        for cal_fld in (
            info["flux_calibrator"],
            info["phase_calibrator"],
            *info["aux_calibrators"],
        ):
            applycal(
                vis=str(cal_ms_path),
                field=cal_fld,
                gaintable=[str(k_tab), str(b_tab), str(flux_tab)],
                gainfield=[bp_cal, bp_cal, cal_fld],
                interp=["nearest", "nearest,linear", "linear"],
                calwt=False,
                flagbackup=False,
            )

    for p in (ms_path, g0_tab, k_tab, b_tab, g_tab, flux_tab):
        _verify_unlocked(p)
    if has_cal_ms:
        _verify_unlocked(cal_ms_path)


def run_solver(workspace: Path) -> None:
    ms_path = workspace / "input.ms"
    info = _inspect_ms(ms_path)
    _run_calibrate_stage(workspace, info)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone CASA oracle solver for r1_3c391_calibrate."
    )
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    scratch_dir, created_scratch = _configure_temp_env(workspace)
    prev_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        run_solver(workspace)
    finally:
        with contextlib.suppress(Exception):
            import casatasks

            casatasks.clearstat()
        for temp_lat in workspace.glob("TempLattice*"):
            shutil.rmtree(temp_lat, ignore_errors=True)
        with contextlib.suppress(Exception):
            os.chdir(prev_cwd)
        if created_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
