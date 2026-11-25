"""Standalone CASA 6.7.6.14 oracle solver for r2_24a376_e2e."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path

CAL_FIELDS = "J0336+3218,0542+498=3C147,0521+166=3C138"
FLUX_CAL = "0542+498=3C147"
PHASE_CAL = "J0336+3218"
POL_CAL = "0521+166=3C138"
TARGET_FIELDS = "PER_FIELD_71,PER_FIELD_75,PER_FIELD_77,PER_FIELD_81"
REFANT = "ea16,ea04,ea07"
MOSAIC_PHASECENTER = "J2000 52.11772deg 30.44509deg"


def _configure_temp_env(workspace: Path) -> tuple[Path, bool]:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cur_tmp = os.environ.get("TMPDIR", "").strip()
    if not cur_tmp or cur_tmp == "/tmp" or cur_tmp.startswith("/tmp/"):
        var_tmp = Path("/var/tmp")
        scratch = (
            var_tmp if (var_tmp.exists() and os.access(var_tmp, os.W_OK)) else workspace
        ) / f".solver_scratch_{os.getpid()}"
    else:
        scratch = Path(cur_tmp) / f"solver_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["MPLCONFIGDIR"] = str(scratch / "mplconfig")
    (scratch / "mplconfig").mkdir(parents=True, exist_ok=True)
    return scratch, True


def run_solver(workspace: Path) -> None:
    import casatasks

    ms_path = workspace / "input.ms"
    flagonline = workspace / "input.flagonline.txt"
    cal_ms = workspace / "calibrators.ms"

    # 1. Precal
    if flagonline.is_file():
        casatasks.flagdata(vis=str(ms_path), mode="list", inpfile=str(flagonline), flagbackup=False)
    casatasks.flagdata(
        vis=str(ms_path), mode="quack", quackinterval=5.0, quackmode="beg", flagbackup=False
    )
    casatasks.flagdata(vis=str(ms_path), mode="shadow", tolerance=0.0, flagbackup=False)
    casatasks.flagdata(vis=str(ms_path), mode="clip", clipzeros=True, flagbackup=False)
    casatasks.flagdata(
        vis=str(ms_path), mode="tfcrop", timecutoff=3.0, freqcutoff=3.0, flagbackup=False
    )
    casatasks.flagdata(vis=str(ms_path), mode="extend", extendpols=True, flagbackup=False)
    casatasks.setjy(
        vis=str(ms_path), field=FLUX_CAL, standard="Perley-Butler 2017", usescratch=True
    )

    if cal_ms.exists():
        shutil.rmtree(cal_ms)
    casatasks.mstransform(
        vis=str(ms_path), outputvis=str(cal_ms), field=CAL_FIELDS, datacolumn="all", keepflags=False
    )

    # 2. Priorcals & Calibration
    gc_tab = workspace / "gain_curves.gc"
    opac_tab = workspace / "opacities.opac"
    rq_tab = workspace / "requantizer.rq"
    casatasks.gencal(vis=str(cal_ms), caltable=str(gc_tab), caltype="gc")
    taus = casatasks.plotweather(vis=str(cal_ms), doPlot=False)
    casatasks.gencal(
        vis=str(cal_ms), caltable=str(opac_tab), caltype="opac", spw="0,1,2,3", parameter=taus
    )
    casatasks.gencal(vis=str(cal_ms), caltable=str(rq_tab), caltype="rq")
    priorcals = [str(gc_tab), str(opac_tab), str(rq_tab)]

    g0 = workspace / "G0.cal"
    k_tab = workspace / "K.cal"
    b_tab = workspace / "B.cal"
    g_tab = workspace / "G.cal"
    flux_tab = workspace / "fluxscale.cal"

    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(g0),
        field=FLUX_CAL,
        spw="*:6~25",
        refant=REFANT,
        calmode="p",
        solint="int",
        minsnr=3.0,
        gaintable=priorcals,
    )
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(k_tab),
        field=FLUX_CAL,
        refant=REFANT,
        gaintype="K",
        solint="inf",
        combine="scan",
        minsnr=3.0,
        gaintable=[*priorcals, str(g0)],
    )
    casatasks.bandpass(
        vis=str(cal_ms),
        caltable=str(b_tab),
        field=FLUX_CAL,
        refant=REFANT,
        solint="inf",
        combine="scan",
        solnorm=True,
        minsnr=3.0,
        gaintable=[*priorcals, str(g0), str(k_tab)],
    )
    casatasks.gaincal(
        vis=str(cal_ms),
        caltable=str(g_tab),
        field=CAL_FIELDS,
        refant=REFANT,
        gaintype="G",
        calmode="ap",
        solint="inf",
        minsnr=3.0,
        gaintable=[*priorcals, str(k_tab), str(b_tab)],
    )
    casatasks.fluxscale(
        vis=str(cal_ms),
        caltable=str(g_tab),
        fluxtable=str(flux_tab),
        reference=[FLUX_CAL],
        transfer=[PHASE_CAL, POL_CAL],
        incremental=False,
    )

    full_tables = [*priorcals, str(k_tab), str(b_tab), str(flux_tab)]
    gf_flux = [""] * len(priorcals) + [FLUX_CAL, FLUX_CAL, FLUX_CAL]
    gf_phase = [""] * len(priorcals) + [FLUX_CAL, FLUX_CAL, PHASE_CAL]
    gf_pol = [""] * len(priorcals) + [FLUX_CAL, FLUX_CAL, POL_CAL]

    casatasks.applycal(
        vis=str(ms_path),
        field=FLUX_CAL,
        gaintable=full_tables,
        gainfield=gf_flux,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(ms_path),
        field=PHASE_CAL,
        gaintable=full_tables,
        gainfield=gf_phase,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(ms_path),
        field=POL_CAL,
        gaintable=full_tables,
        gainfield=gf_pol,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.applycal(
        vis=str(ms_path),
        field=TARGET_FIELDS,
        gaintable=full_tables,
        gainfield=gf_phase,
        applymode="calflag",
        flagbackup=False,
    )
    casatasks.flagdata(
        vis=str(ms_path),
        field=TARGET_FIELDS,
        mode="rflag",
        datacolumn="corrected",
        timedevscale=5.0,
        freqdevscale=5.0,
        flagbackup=False,
    )

    # 3. 4096x4096 wideband mtmfs mosaic imaging
    img_prefix = workspace / "mosaic_target"
    casatasks.tclean(
        vis=str(ms_path),
        imagename=str(img_prefix),
        field=TARGET_FIELDS,
        phasecenter=MOSAIC_PHASECENTER,
        datacolumn="corrected",
        stokes="I",
        specmode="mfs",
        deconvolver="mtmfs",
        nterms=2,
        scales=[0, 5, 15],
        gridder="mosaic",
        cell="2.5arcsec",
        imsize=[4096, 4096],
        weighting="briggs",
        robust=0.5,
        niter=10000,
        nsigma=3.0,
        usemask="auto-multithresh",
        sidelobethreshold=2.0,
        noisethreshold=4.25,
        lownoisethreshold=1.5,
        minbeamfrac=0.3,
        pblimit=0.2,
        pbcor=True,
        interactive=False,
    )
    casatasks.clearstat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    ws = args.workspace.resolve()
    scratch, created = _configure_temp_env(ws)
    prev = Path.cwd()
    try:
        os.chdir(ws)
        run_solver(ws)
    finally:
        with contextlib.suppress(Exception):
            import casatasks

            casatasks.clearstat()
        for temp_lat in ws.glob("TempLattice*"):
            shutil.rmtree(temp_lat, ignore_errors=True)
        with contextlib.suppress(Exception):
            os.chdir(prev)
        if created:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
