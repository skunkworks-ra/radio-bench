"""Shared read-only CASA table helpers, multi-DDID safe streaming, and VisCal discovery."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radio_bench.grading.schema import (
    ObservableRecord,
    StageOutputManifest,
    loads_strict_json,
)

VISCAL_ROLES: dict[str, frozenset[str]] = {
    "K": frozenset({"K Jones"}),
    "Kcross": frozenset({"Kcross Jones"}),
    "B": frozenset({"B Jones", "Mf Mueller"}),
    "G": frozenset({"G Jones", "T Jones", "GlinXph Jones"}),
    "D": frozenset({"D Jones", "Df Jones", "Dlls Jones", "Dflls Jones", "Df+QU Jones"}),
    "Xf": frozenset({"Xf Jones", "Xfparang+QU Jones"}),
}

ALL_VISCAL_TYPES: frozenset[str] = frozenset().union(*VISCAL_ROLES.values())

_NON_CALTABLE_DIR_SUFFIXES: tuple[str, ...] = (
    ".ms",
    ".image",
    ".pb",
    ".residual",
    ".model",
    ".pbcor",
    ".psf",
    ".sumwt",
    ".weight",
    ".mask",
    ".tt0",
    ".tt1",
    ".tt2",
    ".alpha",
    ".flagversions",
    ".cl",
)

_MS_SUBTABLE_NAMES: frozenset[str] = frozenset(
    {
        "ANTENNA",
        "DATA_DESCRIPTION",
        "FEED",
        "FIELD",
        "FLAG_CMD",
        "HISTORY",
        "OBSERVATION",
        "POINTING",
        "POLARIZATION",
        "PROCESSOR",
        "SOURCE",
        "SPECTRAL_WINDOW",
        "STATE",
        "SYSCAL",
        "SYSPOWER",
        "WEATHER",
    }
)


def resolve_scratch_root(scratch_root: Path | None = None) -> Path:
    """Resolve a writable scratch root strictly outside `/tmp`."""
    if scratch_root is not None:
        resolved = Path(scratch_root).resolve()
        if str(resolved) == "/tmp" or str(resolved).startswith("/tmp/"):
            raise ValueError(f"scratch_root must never use /tmp: {resolved}")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    env_tmp = os.environ.get("TMPDIR", "").strip()
    if env_tmp and env_tmp != "/tmp" and not env_tmp.startswith("/tmp/"):
        cand = Path(env_tmp).resolve()
        if str(cand) != "/tmp" and not str(cand).startswith("/tmp/"):
            cand.mkdir(parents=True, exist_ok=True)
            return cand

    var_tmp = Path("/var/tmp/radio-bench-verifier")
    if Path("/var/tmp").exists() and os.access("/var/tmp", os.W_OK):
        var_tmp.mkdir(parents=True, exist_ok=True)
        return var_tmp

    from radio_bench.config import load_bench_config

    cfg_tmp = load_bench_config().tmp_dir
    cfg_tmp.mkdir(parents=True, exist_ok=True)
    return cfg_tmp


def parse_sky_coord_rad(coord_str_ra: str, coord_str_dec: str | None = None) -> tuple[float, float]:
    """Parse J2000/ICRS sky coordinate string(s) into `(ra_rad, dec_rad)`."""
    from astropy.coordinates import SkyCoord

    if coord_str_dec is None:
        cleaned = str(coord_str_ra).strip()
        for prefix in ("J2000", "ICRS"):
            if cleaned.upper().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        parts = cleaned.split()
        if len(parts) < 2:
            raise ValueError(f"Cannot parse sky coordinate string: {coord_str_ra!r}")
        ra_tok, dec_tok = parts[0], parts[1]
    else:
        ra_tok, dec_tok = str(coord_str_ra).strip(), str(coord_str_dec).strip()

    sc = SkyCoord(ra_tok, dec_tok, frame="icrs")
    return float(sc.ra.rad), float(sc.dec.rad)


def has_active_table_lock(table_dir: Path) -> bool:
    """Check whether `table_dir` or any immediate subtable holds an active POSIX fcntl lock."""
    if not table_dir.is_dir():
        return False
    lock_files = [table_dir / "table.lock"]
    try:
        children = list(table_dir.iterdir())
    except OSError:
        children = []
    for child in children:
        if child.is_dir() and (child / "table.lock").is_file():
            lock_files.append(child / "table.lock")

    for lf in lock_files:
        if not lf.is_file():
            continue
        try:
            fd = os.open(str(lf), os.O_RDONLY)
        except OSError:
            continue
        try:
            # Linux x86_64 struct flock: l_type(h), l_whence(h), l_start(q), l_len(q), l_pid(i)
            probe = struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0)
            res = fcntl.fcntl(fd, fcntl.F_GETLK, probe)
            l_type = struct.unpack("hhqqi", res)[0]
            if l_type != fcntl.F_UNLCK:
                return True
        except OSError:
            continue
        finally:
            os.close(fd)
    return False


@contextlib.contextmanager
def open_ro_table(table_path: Path) -> Iterator[Any]:
    """Open a CASA table read-only with `usernoread` lock options inside `try / finally`."""
    import casatools

    tb = casatools.table()
    opened = False
    try:
        opened = bool(
            tb.open(
                str(table_path),
                nomodify=True,
                lockoptions={"option": "usernoread"},
            )
        )
        if not opened:
            raise RuntimeError(f"casatools.table.open returned False for {table_path}")
        yield tb
    finally:
        if opened:
            with contextlib.suppress(Exception):
                tb.close()
        with contextlib.suppress(Exception):
            tb.done()


def check_table_multiused(table_dir: Path) -> bool:
    """Return True if POSIX `fcntl` or casacore `ismultiused(checksubtables=True)` is active."""
    if not table_dir.is_dir():
        return False
    if has_active_table_lock(table_dir):
        return True
    if not (table_dir / "table.dat").is_file():
        return False
    try:
        with open_ro_table(table_dir) as tb:
            return bool(tb.ismultiused(checksubtables=True))
    except Exception:
        return False


def scan_workspace_table_locks(workspace_dir: Path, max_depth: int = 3) -> list[str]:
    """Scan `workspace_dir` up to `max_depth` for any locked CASA MS, caltable, or image."""
    locked_paths: list[str] = []
    visited_inodes: set[tuple[int, int]] = set()

    def _walk(cur: Path, depth: int) -> None:
        if depth > max_depth or not cur.is_dir():
            return
        try:
            st = os.stat(cur)
            ikey = (int(st.st_dev), int(st.st_ino))
        except OSError:
            return
        if ikey in visited_inodes:
            return
        visited_inodes.add(ikey)

        if cur != workspace_dir and (
            (cur / "table.dat").is_file() or (cur / "table.lock").is_file()
        ):
            if check_table_multiused(cur):
                locked_paths.append(str(cur))
            return

        try:
            entries = sorted(cur.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.name.startswith("TempLattice"):
                continue
            if entry.is_dir():
                _walk(entry, depth + 1)

    _walk(workspace_dir, 0)
    return locked_paths


@dataclass(frozen=True)
class DDIDChunk:
    """Single streamed `DATA_DESC_ID` row chunk with global row indices."""

    ddid: int
    startrow: int
    nrow: int
    rownumbers: np.ndarray
    columns: dict[str, np.ndarray]


def iter_ddid_chunks(
    tb: Any,
    col_names: tuple[str, ...] | list[str],
    *,
    query_expr: str | None = None,
    chunk_size: int = 50_000,
    include_rownumbers: bool = False,
) -> Iterator[DDIDChunk]:
    """Stream MS columns safely per `DATA_DESC_ID` in `chunk_size` slices.

    Avoids CASA 6.7.6.14 C++ `SIGABRT` heap buffer overflow when `getcol` is invoked
    across multiple `DATA_DESC_ID`s with heterogeneous `nchan` shapes.
    """
    total_nrows = int(tb.nrows())
    if total_nrows == 0:
        return

    ddids_1d = np.asarray(tb.getcol("DATA_DESC_ID"), dtype=np.int64)
    unique_ddids = sorted(int(x) for x in np.unique(ddids_1d))

    for ddid in unique_ddids:
        q = f"DATA_DESC_ID == {ddid} && ({query_expr})" if query_expr else f"DATA_DESC_ID == {ddid}"
        sub = tb.query(q)
        try:
            sub_nrows = int(sub.nrows())
            if sub_nrows == 0:
                continue
            all_rnums = (
                np.asarray(sub.rownumbers(), dtype=np.int64)
                if include_rownumbers
                else np.empty(0, dtype=np.int64)
            )
            for sr in range(0, sub_nrows, chunk_size):
                nr = min(chunk_size, sub_nrows - sr)
                col_data = {
                    col: np.asarray(sub.getcol(col, startrow=sr, nrow=nr)) for col in col_names
                }
                chunk_rnums = all_rnums[sr : sr + nr] if include_rownumbers else all_rnums
                yield DDIDChunk(
                    ddid=ddid,
                    startrow=sr,
                    nrow=nr,
                    rownumbers=chunk_rnums,
                    columns=col_data,
                )
        finally:
            with contextlib.suppress(Exception):
                sub.close()
            with contextlib.suppress(Exception):
                sub.done()


def read_ddid_to_spw_map(ms_path: Path) -> dict[int, int]:
    """Map `DATA_DESC_ID -> SPECTRAL_WINDOW_ID` via scalar 1D `DATA_DESCRIPTION` lookup."""
    dd_path = ms_path / "DATA_DESCRIPTION"
    if not (dd_path / "table.dat").is_file():
        return {0: 0}
    with open_ro_table(dd_path) as tb_dd:
        if int(tb_dd.nrows()) == 0 or "SPECTRAL_WINDOW_ID" not in tb_dd.colnames():
            return {0: 0}
        spw_ids = np.asarray(tb_dd.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
    return {int(ddid): int(spw_id) for ddid, spw_id in enumerate(spw_ids.tolist())}


def read_spw_chan_freqs(ms_path: Path, spw_id: int = 0) -> np.ndarray:
    """Read 1D channel frequencies (`Hz`) for `spw_id` safely via `getcell` (never `getcol`)."""
    spw_path = ms_path / "SPECTRAL_WINDOW"
    with open_ro_table(spw_path) as tb_spw:
        nrows = int(tb_spw.nrows())
        eff_spw = int(spw_id) if 0 <= int(spw_id) < nrows else 0
        raw_freq = tb_spw.getcell("CHAN_FREQ", eff_spw)
    return np.asarray(raw_freq, dtype=np.float64).ravel().copy()


def compute_raw_data_blake2b(ms_path: Path, *, chunk_size: int = 50_000) -> str:
    """Compute streaming BLAKE2b digest of raw `DATA` and immutable row keys per `DATA_DESC_ID`.

    Includes `(TIME, ANTENNA1, ANTENNA2, FIELD_ID, DATA_DESC_ID, DATA)` and explicitly
    excludes mutable calibration/flag columns (`STATE_ID`, `FLAG`, `FLAG_ROW`, `MODEL_DATA`,
    `CORRECTED_DATA`, `WEIGHT`, `SIGMA`).
    """
    immutable_cols = ("TIME", "ANTENNA1", "ANTENNA2", "FIELD_ID", "DATA_DESC_ID", "DATA")
    hasher = hashlib.blake2b(digest_size=32)
    with open_ro_table(ms_path) as tb:
        cols = set(tb.colnames())
        missing = [c for c in immutable_cols if c not in cols]
        if missing:
            raise KeyError(f"Missing immutable MS columns in {ms_path.name}: {missing}")
        hasher.update(f"total_nrows={int(tb.nrows())};".encode())
        for chunk in iter_ddid_chunks(tb, immutable_cols, chunk_size=chunk_size):
            hasher.update(f"ddid={chunk.ddid}:sr={chunk.startrow}:nr={chunk.nrow};".encode())
            for col in immutable_cols:
                arr = np.ascontiguousarray(chunk.columns[col])
                hasher.update(f"{col}:{arr.dtype.str}:{arr.shape};".encode())
                hasher.update(memoryview(arr))
    return hasher.hexdigest()


def split_intents(raw_intents: list[str] | tuple[str, ...]) -> set[str]:
    """Split CASA scan intent strings on both `,` and `;`."""
    tokens: set[str] = set()
    for entry in raw_intents:
        for part in re.split(r"[,;]", str(entry)):
            tok = part.strip()
            if tok:
                tokens.add(tok)
    return tokens


def read_field_intents(ms_path: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Read field names and split intents safely (`msmetadata` with `STATE`+`MAIN` fallback)."""
    import casatools

    with open_ro_table(ms_path / "FIELD") as tb_f:
        field_names = [str(x).strip() for x in tb_f.getcol("NAME")]

    intents_by_field: dict[str, set[str]] = {fn: set() for fn in field_names}

    msmd = casatools.msmetadata()
    msmd_opened = False
    try:
        msmd_opened = bool(msmd.open(str(ms_path)))
        if msmd_opened:
            for fid, fname in enumerate(field_names):
                raw_list = [str(x) for x in msmd.intentsforfield(fid)]
                intents_by_field[fname] = split_intents(raw_list)
            return field_names, intents_by_field
    except Exception:
        pass
    finally:
        if msmd_opened:
            with contextlib.suppress(Exception):
                msmd.close()
        with contextlib.suppress(Exception):
            msmd.done()

    state_path = ms_path / "STATE"
    if not (state_path / "table.dat").is_file():
        return field_names, intents_by_field

    with open_ro_table(state_path) as tb_s:
        if int(tb_s.nrows()) == 0 or "OBS_MODE" not in tb_s.colnames():
            return field_names, intents_by_field
        obs_modes = [str(x) for x in tb_s.getcol("OBS_MODE")]

    with open_ro_table(ms_path) as tb_m:
        if int(tb_m.nrows()) == 0:
            return field_names, intents_by_field
        fids = np.asarray(tb_m.getcol("FIELD_ID"), dtype=int)
        sids = np.asarray(tb_m.getcol("STATE_ID"), dtype=int)

    for fid, fname in enumerate(field_names):
        u_sids = np.unique(sids[fids == fid])
        modes = [obs_modes[int(s)] for s in u_sids if 0 <= int(s) < len(obs_modes)]
        intents_by_field[fname] = split_intents(modes)
    return field_names, intents_by_field


def resolve_field_id_by_name(
    ms_path: Path,
    field_name: str,
    *,
    fallback_fid: int | None = None,
) -> int | None:
    """Resolve a field's numeric `FIELD_ID` inside `ms_path/FIELD` strictly by name."""
    if not field_name or not (ms_path / "FIELD" / "table.dat").is_file():
        return fallback_fid
    with open_ro_table(ms_path / "FIELD") as tb_f:
        fnames = [str(x).strip() for x in tb_f.getcol("NAME")]
    target_upper = field_name.strip().upper()
    for idx, fn in enumerate(fnames):
        if fn.upper() == target_upper:
            return idx
    if fallback_fid is not None and 0 <= fallback_fid < len(fnames):
        return fallback_fid
    return None


def resolve_field_ids_by_names(ms_path: Path, field_names: list[str]) -> list[int]:
    """Resolve a list of field names to numeric `FIELD_ID`s inside `ms_path/FIELD`."""
    if not field_names or not (ms_path / "FIELD" / "table.dat").is_file():
        return []
    with open_ro_table(ms_path / "FIELD") as tb_f:
        ms_names = [str(x).strip().upper() for x in tb_f.getcol("NAME")]
    wanted = {n.strip().upper() for n in field_names if n.strip()}
    return [idx for idx, nm in enumerate(ms_names) if nm in wanted]


def resolve_ant_ids_by_names(ms_path: Path, ant_names: list[str]) -> list[int]:
    """Resolve antenna names to numeric `ANTENNA_ID`s inside `ms_path/ANTENNA`."""
    if not ant_names or not (ms_path / "ANTENNA" / "table.dat").is_file():
        return []
    with open_ro_table(ms_path / "ANTENNA") as tb_a:
        ms_ants = [str(x).strip().lower() for x in tb_a.getcol("NAME")]
    wanted = {a.strip().lower() for a in ant_names if a.strip()}
    return [idx for idx, nm in enumerate(ms_ants) if nm in wanted]


@dataclass(frozen=True)
class DiscoveredCaltable:
    """Caltable discovered via casacore `VisCal` table keyword (never filename globs)."""

    path: Path
    role: str
    viscal: str


def discover_caltables_by_viscal(
    workspace_dir: Path,
    *,
    max_depth: int = 3,
) -> list[DiscoveredCaltable]:
    """Discover all CASA calibration tables under `workspace_dir` via `VisCal` keyword."""
    visited_inodes: set[tuple[int, int]] = set()
    table_dirs: list[Path] = []

    def _scan(cur_dir: Path, depth: int) -> None:
        if depth > max_depth or not cur_dir.is_dir():
            return
        try:
            st = os.stat(cur_dir)
            inode_key = (int(st.st_dev), int(st.st_ino))
        except OSError:
            return
        if inode_key in visited_inodes:
            return
        visited_inodes.add(inode_key)

        if cur_dir != workspace_dir and (cur_dir / "table.dat").is_file():
            if cur_dir.name not in _MS_SUBTABLE_NAMES:
                table_dirs.append(cur_dir)
            return

        try:
            entries = sorted(cur_dir.iterdir(), key=lambda x: x.name)
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if name.startswith(".") or name.startswith("TempLattice"):
                continue
            low = name.lower()
            if any(low.endswith(sfx) for sfx in _NON_CALTABLE_DIR_SUFFIXES):
                continue
            if entry.is_dir():
                _scan(entry, depth + 1)

    _scan(workspace_dir, 0)

    discovered: list[DiscoveredCaltable] = []
    for p in table_dirs:
        if has_active_table_lock(p):
            continue
        try:
            with open_ro_table(p) as tb:
                viscal_raw = tb.getkeywords().get("VisCal")
                viscal = str(viscal_raw).strip() if viscal_raw is not None else ""
        except Exception:
            continue
        if not viscal:
            continue
        matched_role: str | None = None
        for role, vc_set in VISCAL_ROLES.items():
            if viscal in vc_set:
                matched_role = role
                break
        if matched_role is not None:
            discovered.append(DiscoveredCaltable(path=p, role=matched_role, viscal=viscal))

    return discovered


def _compute_gain_tr_ref_log_jump(
    cparam: np.ndarray,
    flag: np.ndarray,
    fids: np.ndarray,
    ant1: np.ndarray,
    ref_fid: int,
    tr_fid: int,
) -> float:
    """Compute `|median_{pol, ant}(log(|T(tr)| / |T(ref)|))|` across live antennas.

    In CASA `fluxscale`, transfer-field gains are scaled by `1 / sqrt(S_tr)` so each
    antenna's voltage gain `|F_a(tr)|` matches its system gain `|F_a(ref)|` (`log_jump ~ 0`).
    In unscaled `gaincal` (solved against a 1 Jy model), `|G_a(tr)| ~ sqrt(S_tr) * |G_a(ref)|`
    (`log_jump ~ 0.5 * |log(S_tr)|`).
    """
    npol = cparam.shape[0]
    p_indices = [0, npol - 1] if npol >= 2 else [0]
    log_ratios: list[float] = []
    uniq_ants = np.unique(ant1)

    for pol in p_indices:
        for aid in uniq_ants:
            ref_rows = (fids == ref_fid) & (ant1 == aid) & (~flag[pol, 0, :])
            tr_rows = (fids == tr_fid) & (ant1 == aid) & (~flag[pol, 0, :])
            if not np.any(ref_rows) or not np.any(tr_rows):
                continue
            r_amps = np.abs(cparam[pol, 0, ref_rows])
            t_amps = np.abs(cparam[pol, 0, tr_rows])
            r_amps = r_amps[np.isfinite(r_amps) & (r_amps > 1e-12)]
            t_amps = t_amps[np.isfinite(t_amps) & (t_amps > 1e-12)]
            if r_amps.size == 0 or t_amps.size == 0:
                continue
            med_r = float(np.median(r_amps))
            med_t = float(np.median(t_amps))
            if med_r > 1e-12 and med_t > 1e-12:
                log_ratios.append(math.log(med_t / med_r))

    if not log_ratios:
        return float("inf")
    return abs(float(np.median(log_ratios)))


def discover_fluxscale_from_caltables(
    workspace_dir: Path,
    *,
    flux_cal_name: str,
    phase_cal_name: str,
    fallback_flux_fid: int | None = None,
    fallback_phase_fid: int | None = None,
    max_depth: int = 3,
) -> tuple[float | None, str | None]:
    """Recover bootstrapped `phase_cal` flux (`Jy`) and provenance from `VisCal` `G`/`T` tables.

    Pairs candidate `(G_unscaled, F_fluxscaled)` tables with matching `(shape, FIELD_ID,
    ANTENNA1, TIME)`, identical reference-field amplitudes, and identical transfer-field phases,
    then physically orients `F` as the table minimizing `|median_ant(log(|T(tr)| / |T(ref)|))|`
    and computes `S_tr = median((|G(tr)| / |F(tr)|)^2)`.
    """
    caltables = [
        c for c in discover_caltables_by_viscal(workspace_dir, max_depth=max_depth) if c.role == "G"
    ]
    if len(caltables) < 2:
        return None, None

    flux_upper = flux_cal_name.strip().upper()
    phase_upper = phase_cal_name.strip().upper()

    # Tuple: (path, cparam, flag, fids, ant1, time, ref_id, tr_id)
    loaded: list[
        tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]
    ] = []

    for ct in caltables:
        p = ct.path
        try:
            ct_flux_id: int | None = None
            ct_phase_id: int | None = None
            if (p / "FIELD" / "table.dat").is_file():
                with open_ro_table(p / "FIELD") as tb_cf:
                    if int(tb_cf.nrows()) > 0 and "NAME" in tb_cf.colnames():
                        cf_names = [str(x).strip().upper() for x in tb_cf.getcol("NAME")]
                        if any(cf_names):
                            for idx, nm in enumerate(cf_names):
                                if nm == flux_upper and ct_flux_id is None:
                                    ct_flux_id = idx
                                if nm == phase_upper and ct_phase_id is None:
                                    ct_phase_id = idx
                            if ct_flux_id is None or ct_phase_id is None:
                                continue
            if ct_flux_id is None:
                ct_flux_id = fallback_flux_fid
            if ct_phase_id is None:
                ct_phase_id = fallback_phase_fid
            if ct_flux_id is None or ct_phase_id is None:
                continue

            with open_ro_table(p) as tb:
                cols = set(tb.colnames())
                if not {"CPARAM", "FIELD_ID", "FLAG", "ANTENNA1", "TIME"}.issubset(cols):
                    continue
                if int(tb.nrows()) == 0:
                    continue
                cparam = np.asarray(tb.getcol("CPARAM"))
                if cparam.ndim != 3 or cparam.shape[1] != 1:
                    continue
                flg = np.asarray(tb.getcol("FLAG"), dtype=bool)
                fids = np.asarray(tb.getcol("FIELD_ID"), dtype=np.int64)
                ant1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int64)
                t_col = np.asarray(tb.getcol("TIME"), dtype=np.float64)
                if ct_flux_id in fids and ct_phase_id in fids:
                    loaded.append(
                        (p, cparam, flg, fids, ant1, t_col, int(ct_flux_id), int(ct_phase_id))
                    )
        except Exception:
            continue

    n = len(loaded)
    for i in range(n):
        pa, cpa, flga, fida, anta, ta, ref_a, tr_a = loaded[i]
        for j in range(i + 1, n):
            pb, cpb, flgb, fidb, antb, tb_arr, ref_b, tr_b = loaded[j]
            if (
                cpa.shape != cpb.shape
                or ref_a != ref_b
                or tr_a != tr_b
                or not np.array_equal(fida, fidb)
                or not np.array_equal(anta, antb)
                or not np.allclose(ta, tb_arr, rtol=0.0, atol=1e-3)
            ):
                continue

            valid_ab = (
                (~flga)
                & (~flgb)
                & np.isfinite(cpa)
                & np.isfinite(cpb)
                & (np.abs(cpa) > 1e-12)
                & (np.abs(cpb) > 1e-12)
            )
            ref_m = (fida == ref_a)[np.newaxis, np.newaxis, :] & valid_ab
            tr_m = (fida == tr_a)[np.newaxis, np.newaxis, :] & valid_ab
            if not np.any(ref_m) or not np.any(tr_m):
                continue

            # 1. Identical reference-field gain amplitudes & phases
            ref_amp_diff = float(np.median(np.abs(np.abs(cpa[ref_m]) - np.abs(cpb[ref_m]))))
            if ref_amp_diff > 1e-4:
                continue

            # 2. Identical transfer-field phases (fluxscale scales amplitudes only)
            tr_phs_diff_rad = np.abs(np.angle(cpa[tr_m] * np.conj(cpb[tr_m])))
            if float(np.median(tr_phs_diff_rad)) > 1e-3:
                continue

            # 3. Uniform transfer amplitude ratio across antennas
            raw_ratios = np.abs(cpa[tr_m]) / np.abs(cpb[tr_m])
            med_ratio = float(np.median(raw_ratios))
            if not math.isfinite(med_ratio) or med_ratio <= 1e-12:
                continue
            rel_mad_ratio = float(np.median(np.abs(raw_ratios - med_ratio))) / med_ratio
            if rel_mad_ratio > 0.02:
                continue

            # 4. Orient (G_unscaled, F_fluxscaled) physically by system-gain continuity
            jump_a = _compute_gain_tr_ref_log_jump(cpa, flga, fida, anta, ref_a, tr_a)
            jump_b = _compute_gain_tr_ref_log_jump(cpb, flgb, fidb, antb, ref_b, tr_b)

            if jump_a <= jump_b:
                p_flux, cp_flux = pa, cpa
                p_unsc, cp_unsc = pb, cpb
            else:
                p_flux, cp_flux = pb, cpb
                p_unsc, cp_unsc = pa, cpa

            s_ratios = (np.abs(cp_unsc[tr_m]) / np.abs(cp_flux[tr_m])) ** 2
            s_est = float(np.median(s_ratios))
            if math.isfinite(s_est) and 0.01 <= s_est <= 1000.0:
                # Reject trivial identity self-copies (S == 1.0 within numerical noise)
                # unless both jump_a and jump_b are near-zero AND amplitudes genuinely differ
                if abs(s_est - 1.0) <= 1e-4:
                    continue
                prov = f"{p_unsc.name}->{p_flux.name}"
                return s_est, prov

    return None, None


def verify_pb_taper(
    pb_arr: np.ndarray,
    pb_gate: np.ndarray,
    *,
    inc_x_rad: float | None = None,
    inc_y_rad: float | None = None,
    freq_hz: float | None = None,
    dish_diameter_m: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """Verify physical primary-beam taper for both wide-FoV mosaics and narrow single pointings."""
    if int(np.sum(pb_gate)) < 50:
        return False, {}

    pb_in_gate = pb_arr[pb_gate]
    pb_max_20 = float(np.max(pb_in_gate))
    pb_min_20 = float(np.min(pb_in_gate))
    pb_fin_mask = np.isfinite(pb_arr) & (pb_arr > 0.0)
    pb_fin_all = pb_arr[pb_fin_mask]
    pb_min_full = float(np.min(pb_fin_all)) if pb_fin_all.size > 0 else 1.0
    pb_std_20 = float(np.std(pb_in_gate))

    stats = {
        "pb_max_pb20": pb_max_20,
        "pb_min_pb20": pb_min_20,
        "pb_min_full": pb_min_full,
        "pb_std_pb20": pb_std_20,
    }

    if not (0.92 <= pb_max_20 <= 1.05):
        return False, stats

    # Standard wide-FoV / mosaic primary-beam taper check
    if (pb_min_20 <= 0.65 or pb_min_full < 0.85) and pb_std_20 >= 0.05:
        return True, stats

    # Single-pointing narrow-FoV analytical Airy/Gaussian primary-beam check:
    # Rejects constant pb == 1.0 fakes while permitting narrow fields of view where pb_min >= 0.85.
    if (
        inc_x_rad is not None
        and inc_y_rad is not None
        and freq_hz is not None
        and freq_hz > 0.0
        and dish_diameter_m is not None
        and dish_diameter_m > 0.0
        and int(np.sum(pb_fin_mask)) >= 50
    ):
        pb_std_full = float(np.std(pb_fin_all))
        if pb_std_full <= 1e-5:
            return False, stats

        masked_pb = np.where(pb_gate, pb_arr, -np.inf)
        peak_flat = int(np.argmax(masked_pb))
        px_peak, py_peak = np.unravel_index(peak_flat, pb_arr.shape)
        nx, ny = pb_arr.shape
        ix, iy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
        r_rad = np.hypot(
            (ix - float(px_peak)) * abs(inc_x_rad), (iy - float(py_peak)) * abs(inc_y_rad)
        )
        r_vals = r_rad[pb_fin_mask]
        p_vals = pb_arr[pb_fin_mask]
        r_p95 = float(np.percentile(r_vals, 95.0))
        if r_p95 <= 0.0:
            return False, stats

        theta_fwhm_rad = 1.13 * (299792458.0 / float(freq_hz)) / float(dish_diameter_m)
        expected_drop = pb_max_20 * (
            1.0 - math.exp(-4.0 * math.log(2.0) * (r_p95 / max(theta_fwhm_rad, 1e-12)) ** 2)
        )
        outer_sel = pb_fin_mask & (r_rad >= 0.90 * r_p95)
        if not np.any(outer_sel):
            return False, stats
        measured_drop = pb_max_20 - float(np.median(pb_arr[outer_sel]))
        if measured_drop <= 1e-4 or expected_drop <= 1e-5:
            return False, stats

        r2 = r_vals**2
        if float(np.std(r2)) <= 1e-24 or float(np.std(p_vals)) <= 1e-12:
            return False, stats
        corr_r2_pb = float(np.corrcoef(r2, p_vals)[0, 1])
        pb_analytic = pb_max_20 * np.exp(
            -4.0 * math.log(2.0) * (r_vals / max(theta_fwhm_rad, 1e-12)) ** 2
        )
        med_analytic_err = float(np.median(np.abs(p_vals - pb_analytic)))

        analytic_ok = (
            corr_r2_pb <= -0.90
            and 0.35 * expected_drop <= measured_drop <= 2.50 * max(expected_drop, 1e-3)
            and med_analytic_err <= 0.05
        )
        return analytic_ok, stats

    return False, stats


def parse_stage_manifest(
    workspace_dir: Path,
    *,
    stages: list[str],
) -> tuple[
    StageOutputManifest | None, dict[str, Any] | None, dict[str, ObservableRecord], list[str]
]:
    """Parse `/workspace/stage_output.json` (if present) and emit Layer B `proc.manifest_*` observables."""
    obs: dict[str, ObservableRecord] = {}
    errors: list[str] = []
    manifest_path = workspace_dir / "stage_output.json"

    if manifest_path.is_file():
        raw_obj: dict[str, Any] | None = None
        raw_text = ""
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
            raw_parsed = loads_strict_json(raw_text)
            if isinstance(raw_parsed, dict):
                raw_obj = raw_parsed
            manifest = StageOutputManifest.model_validate(raw_parsed)
            obs["proc.manifest_schema_valid"] = ObservableRecord(
                value=True, status="measured", provenance="stage_output.json"
            )
            obs["proc.manifest_status"] = ObservableRecord(
                value=manifest.status, status="measured", provenance="stage_output.json"
            )
            obs["proc.manifest_halt_code"] = ObservableRecord(
                value=manifest.halt_code,
                status="measured" if manifest.halt_code is not None else "not_applicable",
                provenance="stage_output.json",
            )
            return manifest, raw_obj, obs, errors
        except Exception as exc:
            errors.append(f"Invalid stage_output.json: {exc}")
            if raw_obj is None and raw_text:
                with contextlib.suppress(Exception):
                    fallback_parsed = json.loads(raw_text)
                    if isinstance(fallback_parsed, dict):
                        raw_obj = fallback_parsed
            obs["proc.manifest_schema_valid"] = ObservableRecord(
                value=False, status="measured", provenance="stage_output.json"
            )
            raw_status = (
                str(raw_obj.get("status")).strip()
                if isinstance(raw_obj, dict) and isinstance(raw_obj.get("status"), str)
                else None
            )
            obs["proc.manifest_status"] = ObservableRecord(
                value=raw_status,
                status="measured" if raw_status else "extractor_error",
                provenance="stage_output.json",
            )
            obs["proc.manifest_halt_code"] = ObservableRecord(
                value=None, status="extractor_error", provenance="stage_output.json"
            )
            return None, raw_obj, obs, errors

    # Backward-compatible inspect_summary.json check on pure inspect stage
    if stages == ["inspect"]:
        insp_path = workspace_dir / "inspect_summary.json"
        if insp_path.is_file():
            try:
                parsed = loads_strict_json(insp_path.read_text(encoding="utf-8"))
                valid_dict = isinstance(parsed, dict) and len(parsed) > 0
                obs["proc.manifest_schema_valid"] = ObservableRecord(
                    value=valid_dict, status="measured", provenance="inspect_summary.json"
                )
                obs["proc.manifest_status"] = ObservableRecord(
                    value="completed" if valid_dict else None,
                    status="measured" if valid_dict else "extractor_error",
                    provenance="inspect_summary.json",
                )
                obs["proc.manifest_halt_code"] = ObservableRecord(
                    value=None, status="not_applicable", provenance="inspect_summary.json"
                )
                return None, parsed if isinstance(parsed, dict) else None, obs, errors
            except Exception as exc:
                errors.append(f"Invalid inspect_summary.json: {exc}")
                obs["proc.manifest_schema_valid"] = ObservableRecord(
                    value=False, status="measured", provenance="inspect_summary.json"
                )
                obs["proc.manifest_status"] = ObservableRecord(
                    value=None, status="extractor_error", provenance="inspect_summary.json"
                )
                obs["proc.manifest_halt_code"] = ObservableRecord(
                    value=None, status="extractor_error", provenance="inspect_summary.json"
                )
                return None, None, obs, errors

    obs["proc.manifest_schema_valid"] = ObservableRecord(
        value=None, status="missing_artifact", provenance="stage_output.json"
    )
    obs["proc.manifest_status"] = ObservableRecord(
        value=None, status="missing_artifact", provenance="stage_output.json"
    )
    obs["proc.manifest_halt_code"] = ObservableRecord(
        value=None, status="missing_artifact", provenance="stage_output.json"
    )
    return None, None, obs, errors
