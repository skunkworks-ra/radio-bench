"""Export `raw`, `orchestrated_skills`, and CASA-native (`casa`) skill trees + stage subagents.

Reads strictly read-only via `git archive` from the configured `radio-analyst` pack source
(`bench.toml` / `bench.local.toml`), never modifying the source repository and never touching `/tmp`.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import shutil
import subprocess
import tarfile
import textwrap
from pathlib import Path
from typing import Any

from radio_bench.config import (
    BenchConfig,
    add_config_cli_args,
    config_from_cli_args,
    load_bench_config,
    materialize_pack_source,
    reject_slash_tmp,
    resolve_git_ref,
)


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_EXPORT_DIR":
        return load_bench_config().exported_skills_dir
    if name == "RADIO_ANALYST_SHA":
        return load_bench_config().get_pack().analyst_ref
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


MS_TOKEN_RE = re.compile(r"\bms_[a-z0-9_]+\b")
PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
FORBIDDEN_DASHES = ("\u2014", "\u2013")
_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*(\r?\n)(.*?)^---[ \t]*(?:\r?\n|\Z)", re.DOTALL | re.MULTILINE
)
_FRONTMATTER_NAME_RE = re.compile(r"^name[ \t]*:", re.MULTILINE)

# Skills excluded from all single-agent skill directories (`01_skills_raw`, `01_skills_casa`, `11_full`).
SINGLE_AGENT_EXCLUDED_SKILLS: frozenset[str] = frozenset(
    {
        "orchestrate",
        "ms-simulator",
        "simulate",
    }
)
# Skills excluded even in `11_orchestrated` (`orchestrate` is retained there; simulators excluded).
ORCHESTRATED_EXCLUDED_SKILLS: frozenset[str] = frozenset(
    {
        "ms-simulator",
        "simulate",
    }
)

# Strictly forbidden dataset-specific / eval-specific tokens in `casa/` skills (case-insensitive).
CASA_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "ea13",
    "ea07",
    "j1822-0938",
    "j1822",
    "1822-096",
    "3c391",
    "j0319+4130",
    "3c84",
    "0319+415",
    "2.30 jy",
    "2.15 jy",
    "scan 1",
    "synthetic",
    "synth_mosaic",
    "[320, 320]",
    "[384, 384]",
    "[640, 640]",
    "0.75 mjy",
    "[0.70, 0.90]",
    "0.12",
    "bowl_ratio",
    "ground_truth",
    "reward.json",
    "metrics.json",
    "mcp__plugin_radio-analyst_",
)

STAGE_NAMES: tuple[str, ...] = (
    "inspect",
    "precal",
    "calibrate",
    "postcal",
    "selfcal",
    "image",
    "polcal",
)

RAW_TRANSLATION_NOTE = (
    "Translation note: If `ms_*` MCP tools are unavailable in this environment, "
    "translate every procedural step, metadata check, flagging pass, calibration solve, "
    "and imaging gate in the skills directly to `casatools` and `casatasks` using "
    "`/opt/casa-venv/bin/python`."
)


def _reject_slash_tmp(path: Path, label: str) -> Path:
    reject_slash_tmp(path, label=label)
    return path


def _sanitize_dashes(text: str) -> str:
    return text.replace("\u2014", "--").replace("\u2013", "--")


def _strip_plugin_mcp_prefixes(text: str) -> str:
    return text.replace("mcp__plugin_radio-analyst_", "mcp__")


def ensure_skill_frontmatter_name(content: str, skill_name: str) -> str:
    """Ensure a ``SKILL.md`` file has a YAML frontmatter ``name: <skill_name>`` line."""
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return f"---\nname: {skill_name}\n---\n\n{content}"
    nl = match.group(1)
    fm_body = match.group(2)
    if _FRONTMATTER_NAME_RE.search(fm_body) is not None:
        return content
    insert_pos = match.end(1)
    return f"{content[:insert_pos]}name: {skill_name}{nl}{content[insert_pos:]}"


def _verify_pinned_commit(repo: Path, sha: str) -> str:
    full_sha, _ = resolve_git_ref(repo, sha)
    return full_sha


def _git_archive_bytes(repo: Path, sha: str, paths: list[str]) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", sha, "--", *paths],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return proc.stdout


def _extract_filtered_skills(
    tar_bytes: bytes,
    prefix_to_strip: str,
    dest_dir: Path,
    *,
    excluded_top_dirs: frozenset[str],
    adapt_orchestrate_stage_names: bool = False,
) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    norm_prefix = prefix_to_strip.rstrip("/") + "/"
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.name.startswith(norm_prefix):
                continue
            rel = member.name[len(norm_prefix) :]
            if not rel:
                continue
            rel_path = Path(rel)
            if ".." in rel_path.parts or rel_path.is_absolute():
                raise ValueError(f"Unsafe archive entry: {member.name}")
            top_part = rel_path.parts[0]
            if top_part in excluded_top_dirs:
                continue
            out_path = dest_dir / rel_path
            if member.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw_bytes = fobj.read()
                if out_path.suffix == ".md":
                    txt = raw_bytes.decode("utf-8")
                    txt = _strip_plugin_mcp_prefixes(txt)
                    if adapt_orchestrate_stage_names and top_part == "orchestrate":
                        for stage in STAGE_NAMES:
                            txt = txt.replace(f"radio-analyst:{stage}-stage", f"{stage}-stage")
                        txt = txt.replace(
                            "radio-analyst:radio-interferometry", "radio-interferometry"
                        )
                    if rel_path.name == "SKILL.md" and len(rel_path.parts) >= 2:
                        txt = ensure_skill_frontmatter_name(txt, rel_path.parent.name)
                    out_path.write_text(txt, encoding="utf-8")
                else:
                    out_path.write_bytes(raw_bytes)
                out_path.chmod(0o644)
                written += 1
    return written


def _read_tar_text_map(tar_bytes: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            entries[member.name] = fobj.read().decode("utf-8")
    return entries


def _adapt_claude_command_markdown(raw_cmd: str) -> str:
    text = _sanitize_dashes(raw_cmd)
    text = _strip_plugin_mcp_prefixes(text)
    text = text.replace("radio-analyst:radio-interferometry", "radio-interferometry")
    return text


def _adapt_claude_stage_agent(raw_agent: str, stage: str, adapted_cmd: str) -> str:
    text = _sanitize_dashes(raw_agent)
    text = _strip_plugin_mcp_prefixes(text)
    text = text.replace(f"`radio-analyst:{stage}`", f"`{stage}`")
    lines = text.splitlines()
    out_lines: list[str] = []
    in_frontmatter = False
    fm_count = 0
    for line in lines:
        if line.strip() == "---":
            fm_count += 1
            in_frontmatter = fm_count == 1
            out_lines.append(line)
            continue
        if in_frontmatter and line.lstrip().startswith("model:"):
            continue
        out_lines.append(line)
    cmd_body = _strip_yaml_frontmatter(adapted_cmd)
    out_lines.extend(
        [
            "",
            "Stage command specification (follow directly if Skill lookup is unavailable):",
            "",
            cmd_body.strip(),
            "",
        ]
    )
    return "\n".join(out_lines)


def _adapt_opencode_stage_agent(raw_agent: str, adapted_cmd: str) -> str:
    text = _sanitize_dashes(raw_agent)
    text = _strip_plugin_mcp_prefixes(text)
    lines = text.splitlines()
    out_lines: list[str] = []
    in_frontmatter = False
    fm_count = 0
    cmd_body = _strip_yaml_frontmatter(adapted_cmd).strip()
    for line in lines:
        if line.strip() == "---":
            fm_count += 1
            in_frontmatter = fm_count == 1
            out_lines.append(line)
            continue
        if in_frontmatter and line.lstrip().startswith("model:"):
            continue
        if line.strip().startswith("@.claude/commands/"):
            out_lines.append(cmd_body)
            continue
        out_lines.append(line)
    out_lines.append("")
    return "\n".join(out_lines)


def _strip_yaml_frontmatter(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return markdown
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :]).lstrip("\n")
    return markdown


def _dedent_python_fences(markdown: str) -> str:
    """Dedent fenced ```python blocks so every standalone block parses cleanly."""

    def _repl(match: re.Match[str]) -> str:
        body = textwrap.dedent(match.group(1)).strip("\n")
        return f"```python\n{body}\n```"

    return PYTHON_FENCE_RE.sub(_repl, markdown)


def _build_workspace_docs() -> dict[str, str]:
    raw_doc = (
        "# CASA Radio Reduction Workspace\n\n"
        '- **First action (mandatory):** Call `skill(name="radio-interferometry")` before running '
        "any inspection, flagging, calibration, or imaging step, and follow its workflow files for each stage.\n"
        "- Working directory: `/workspace`\n"
        "- CASA 6.7.6.14 Python interpreter (`casatools`, `casatasks`, `numpy`, `astropy`, `scipy`): "
        "`/opt/casa-venv/bin/python`\n"
        "- Bounded-wait script runner: `/usr/local/bin/run-step.sh` "
        "(also available at `/opt/radio-tools/scripts/plugin/run-step.sh`)\n"
        "- Offline CASA measures data: `/opt/casa-data` (preconfigured via `/etc/casa/config.py`)\n\n"
        f"{RAW_TRANSLATION_NOTE}\n"
    )
    casa_doc = (
        "# CASA Radio Reduction Workspace\n\n"
        '- **First action (mandatory):** Call `skill(name="radio-interferometry")` before running '
        "any inspection, flagging, calibration, or imaging step, and follow its CASA workflow files for each stage.\n"
        "- Working directory: `/workspace`\n"
        "- CASA 6.7.6.14 Python interpreter (`casatools`, `casatasks`, `numpy`, `astropy`, `scipy`): "
        "`/opt/casa-venv/bin/python`\n"
        "- Bounded-wait script runner: `/usr/local/bin/run-step.sh /opt/casa-venv/bin/python <script.py> --wait 240`\n"
        "- Offline CASA measures data: `/opt/casa-data` (preconfigured via `/etc/casa/config.py`)\n"
        "- Always close and release every `casatools` handle (`table`, `msmetadata`, `image`, `ms`) "
        "inside `try / finally` blocks (`.close()` then `.done()`) so zero `table.lock` files remain.\n"
    )
    full_doc = (
        "# CASA Radio Reduction Workspace\n\n"
        '- **First action (mandatory):** Call `skill(name="radio-interferometry")` BEFORE calling '
        "any `ms_*` tool or running CASA commands. Follow the playbook and read the stage workflow file "
        "(`10-precal-workflow.md`, `07-calibration-execution.md`, `13-postcal-rfi-flagging.md`, "
        "`11-imaging.md`) before executing each stage.\n"
        "- Working directory: `/workspace`\n"
        "- CASA 6.7.6.14 Python interpreter: `/opt/casa-venv/bin/python`\n"
        "- Bounded-wait script runner for `execute=False` tool scripts: "
        "`/usr/local/bin/run-step.sh /opt/casa-venv/bin/python <script_path> --wait 240` "
        "(also available at `/opt/radio-tools/scripts/plugin/run-step.sh`)\n"
        "- MCP servers (`ms-inspect`, `ms-modify`, `ms-create`) share `/workspace`.\n"
    )
    orchestrated_doc = (
        "# Orchestrated CASA Radio Reduction Workspace\n\n"
        '- **First action (mandatory):** Call `skill(name="orchestrate")` as your VERY FIRST tool call '
        "before inspecting or modifying `/workspace/input.ms`. Do NOT ad-lib reduction stages directly with "
        "`ms_*` tools from the coordinator -- delegate each reduction stage (`inspect` -> `precal` -> "
        "`calibrate` -> `postcal` -> `image`) via `task` to its dedicated `*-stage` subagent and verify "
        "every stage gate before advancing.\n"
        "- Working directory: `/workspace`\n"
        "- Python interpreter (`python:` for stage handoffs): `/opt/casa-venv/bin/python`\n"
        "- Bounded-wait script runner: `/usr/local/bin/run-step.sh /opt/casa-venv/bin/python <script_path> --wait 240` "
        "(also available at `/opt/radio-tools/scripts/plugin/run-step.sh`)\n"
        "- Stage subagents (`inspect-stage`, `precal-stage`, `calibrate-stage`, `postcal-stage`, "
        "`selfcal-stage`, `image-stage`) are pre-staged in `.claude/agents/` and `.opencode/agents/`.\n"
    )
    return {
        "raw": raw_doc,
        "casa": casa_doc,
        "full": full_doc,
        "orchestrated": orchestrated_doc,
    }


def _build_casa_skill_files() -> dict[str, str]:
    """Return dataset-blind, eval-blind CASA-native (`01_skills_casa`) skill markdown files.

    Translates the general radio-interferometry domain methodology of `radio-analyst@00a11ea`
    into direct `casatools` and `casatasks` usage with zero `ms_*` tool tokens, zero dataset
    identifiers, zero benchmark/verifier awareness, and valid `ast.parse` Python code fences.
    """
    files: dict[str, str] = {}

    files["casa-docs/SKILL.md"] = """---
name: casa-docs
description: >
  Inspect CASA 6.7.6.14 casatasks and casatools signatures, defaults, and docstrings
  offline using /opt/casa-venv/bin/python.
---

# CASA Documentation & Signature Lookup (`casatasks` / `casatools`)

When uncertain about any CASA 6.7.6.14 parameter name, default, or return schema,
query the installed module directly in `/opt/casa-venv/bin/python` rather than guessing:

```python
import casatasks
import inspect

print(inspect.signature(casatasks.tclean))
print((casatasks.tclean.__doc__ or "")[:2000])
```

For `casatools` classes (`msmetadata`, `table`, `image`, `measures`, `quanta`, `componentlist`):

```python
import casatools
import inspect

msmd = casatools.msmetadata()
try:
    print([m for m in dir(msmd) if not m.startswith("_")])
    print(inspect.signature(msmd.chanfreqs))
finally:
    msmd.done()
```
"""

    files["report/SKILL.md"] = """---
name: report
description: >
  Record verified CASA reduction steps, calibration metrics, flagging fractions,
  and image quality checks in structured summaries.
---

# Reduction Ledger & Verification Summary

Maintain a concise, quantitative record of every stage executed in the working directory:

1. Record exact parameters passed to `defintent`, `flagdata`, `gencal`, `setjy`,
   `gaincal`, `bandpass`, `fluxscale`, `applycal`, `split`, and `tclean`.
2. Before and after each flagging or `applycal` step, measure per-field flagged fractions via
   `casatasks.flagdata(vis=..., mode="summary")` so unexpected flag jumps (`>= 15` percentage
   points) or failed calibration solves are caught immediately.
3. Verify zero active casacore `table.lock` files remain on Measurement Sets, calibration
   tables, or CASA image directories before finishing.
"""

    files["radio-interferometry/SKILL.md"] = """---
name: radio-interferometry
description: >
  Radio interferometric data analysis, calibration, RFI excision, and continuum/mosaic
  imaging for CASA Measurement Sets using casatools and casatasks.
---

# Radio Interferometry Skill (`casatools` & `casatasks`)

You are operating as a radio interferometrist reducing connected-element interferometer
data (VLA, MeerKAT, uGMRT) in CASA 6.7.6.14 (`/opt/casa-venv/bin/python`).

## Core operating principles

1. **Measure first, then decide:** Never hardcode field IDs, antenna names, channel counts,
   or imaging geometry from memory. Query the Measurement Set subtables (`OBSERVATION`,
   `FIELD`, `STATE`, `DATA_DESCRIPTION`, `SPECTRAL_WINDOW`, `POLARIZATION`, `ANTENNA`) and
   visibility arrays first, then derive parameters quantitatively.
2. **Pass field names, never numeric IDs:** `casatasks.split()` re-indexes numeric `FIELD_ID`s
   starting from `0`. Always pass explicit field name strings to `field`, `reference`,
   `transfer`, and `gainfield`.
3. **Strict casacore lock hygiene:** Every `casatools.table()`, `casatools.msmetadata()`,
   `casatools.image()`, and `casatools.ms()` handle must be closed in a `try / finally`
   block calling `.close()` followed by `.done()`. An unclosed write lock leaves a `table.lock`
   file that blocks subsequent tasks.
4. **Long-running steps via bounded runner:** Run standalone CASA Python scripts in the
   foreground through `/usr/local/bin/run-step.sh /opt/casa-venv/bin/python <script.py> --wait 240`.
   If `status=running`, invoke the exact same command again to re-attach until `status=done rc=0`.

## Supporting reference files (siblings of this `SKILL.md`)

Read `00-playbook.md`, `01-workflow.md`, and `01b-workflow-phase2.md` first, then consult
stage-specific references as you reach each stage:

- `00-playbook.md`: Stage transitions and pre-flight checks
- `01-workflow.md`: Phase 1 metadata, field-role, intent, and spectral inspection
- `01b-workflow-phase2.md`: Phase 2 array geometry, elevation, shadowing, and antenna health
- `02-orientation.md`: VLA bands, primary beams, and scan-intent repair (`defintent`)
- `03-instrument-sanity.md`: Elevation limits, scan-start settling, and `refant` scoring
- `04-diagnostic-reasoning.md`: Go/no-go triage before calibration
- `05-calibrator-science.md`: Flux density standards (`Perley-Butler 2017`) and `setjy`
- `06-failure-modes.md`: Common CASA pitfalls and recovery
- `07-calibration-execution.md`: Full continuum calibration chain (`G0 -> K -> B -> G -> fluxscale -> applycal`)
- `07b-gaincal-recovery.md`: Low-SNR and failed-solution diagnostics
- `08-pband-specifics.md`: Low-frequency VLA P-band specifics
- `09-polcal-execution.md` & `09b-polcal-reference.md`: Full-Stokes polarization calibration
- `10-precal-workflow.md`: Pre-calibration flagging, `calibrators.ms` split, `setjy`, and initial `B0 + rflag`
- `11-imaging.md`: Continuum & mosaic `tclean` parameter derivation and quality gates
- `12-selfcal.md`: Optional phase-only self-calibration
- `13-postcal-rfi-flagging.md`: Post-calibration residual and target RFI triage
"""

    files["radio-interferometry/00-playbook.md"] = """# 00 -- Stage Playbook

Inspect the live state of the Measurement Set and working directory before running any step:

| Current MS / workspace state | Next action |
|---|---|
| Uninspected MS | Run Phase 1 & 2 triage (`01-workflow.md`, `01b-workflow-phase2.md`) |
| Missing or incomplete `STATE` scan intents | Repair intents with `casatasks.defintent(..., mode="set")` (`02-orientation.md`) |
| Raw unflagged MS (`CORRECTED_DATA` absent, no `calibrators.ms`) | Execute pre-calibration (`10-precal-workflow.md`): deterministic preflags, `setjy(usescratch=True)`, split `calibrators.ms`, initial `G0 + B0`, calibrator residual `rflag` |
| Pre-calibrated MS + `calibrators.ms` ready | Execute full calibration (`07-calibration-execution.md`): dead-antenna check, `refant`, `G0 -> K -> B(solnorm=False) -> G -> fluxscale -> applycal` |
| `CORRECTED_DATA` populated on targets with residual RFI | Run conservative post-calibration RFI check (`13-postcal-rfi-flagging.md`) |
| Clean `CORRECTED_DATA` on science target(s) | Derive imaging parameters and run `tclean` (`11-imaging.md`) |
| High-SNR first image (`dynamic_range >= 50`) needing phase refinement | Optional single pass of phase-only self-calibration (`12-selfcal.md`) |
"""

    files[
        "radio-interferometry/01-workflow.md"
    ] = """# 01 -- Phase 1 Metadata, Field Role & Spectral Inspection

Open the Measurement Set read-only (`nomodify=True`) and inspect subtables before touching
visibility columns:

```python
import math
import casatools
import numpy as np


def inspect_phase1(vis_path: str) -> dict:
    tb = casatools.table()
    result = {}
    try:
        tb.open(f"{vis_path}/OBSERVATION", nomodify=True)
        result["telescope"] = str(tb.getcol("TELESCOPE_NAME")[0])
        tb.close()

        tb.open(f"{vis_path}/STATE", nomodify=True)
        obs_modes = list(tb.getcol("OBS_MODE")) if tb.nrows() > 0 else []
        tb.close()

        tb.open(f"{vis_path}/FIELD", nomodify=True)
        names = [str(x) for x in tb.getcol("NAME")]
        phase_dir = tb.getcol("PHASE_DIR")
        tb.close()

        fields = []
        for fid, name in enumerate(names):
            ra_deg = math.degrees(float(phase_dir[0, 0, fid])) % 360.0
            dec_deg = math.degrees(float(phase_dir[1, 0, fid]))
            fields.append({"field_id": fid, "name": name, "ra_deg": ra_deg, "dec_deg": dec_deg})
        result["fields"] = fields

        tb.open(f"{vis_path}/SPECTRAL_WINDOW", nomodify=True)
        spws = []
        for spw_id in range(tb.nrows()):
            freqs = np.asarray(tb.getcell("CHAN_FREQ", spw_id), dtype=float).ravel()
            widths = np.abs(np.asarray(tb.getcell("CHAN_WIDTH", spw_id), dtype=float).ravel())
            nchan = freqs.size
            c_lo = max(0, int(round(0.42 * nchan)))
            c_hi = max(c_lo, int(round(0.58 * nchan)) - 1)
            spws.append(
                {
                    "spw_id": spw_id,
                    "nchan": int(nchan),
                    "min_ghz": float(freqs.min() / 1e9),
                    "max_ghz": float(freqs.max() / 1e9),
                    "center_ghz": float(0.5 * (freqs.min() + freqs.max()) / 1e9),
                    "chan_width_mhz": float(np.median(widths) / 1e6),
                    "center_channels": f"{spw_id}:{c_lo}~{c_hi}",
                }
            )
        tb.close()
        result["spws"] = spws

        tb.open(vis_path, nomodify=True)
        scans = tb.getcol("SCAN_NUMBER")
        fids = tb.getcol("FIELD_ID")
        sids = tb.getcol("STATE_ID")
        times = tb.getcol("TIME")
        intervals = tb.getcol("INTERVAL")
        tb.close()

        scan_rows = []
        for sc in sorted(set(int(x) for x in scans)):
            m = scans == sc
            fid = int(fids[m][0])
            sid = int(sids[m][0])
            intent = str(obs_modes[sid]) if 0 <= sid < len(obs_modes) else ""
            dur_s = float(times[m].max() - times[m].min() + np.median(intervals[m]))
            scan_rows.append(
                {
                    "scan": sc,
                    "field_id": fid,
                    "field_name": names[fid],
                    "intent": intent,
                    "duration_s": round(dur_s, 2),
                    "nrows": int(np.sum(m)),
                }
            )
        result["scans"] = scan_rows
        return result
    finally:
        try:
            tb.close()
        except Exception:
            pass
        tb.done()
```

## Separating calibrators from science mosaic pointings when `STATE` is incomplete

Archival or UVFITS-imported datasets frequently have empty `STATE` tables or mislabel a
secondary calibrator with `OBSERVE_TARGET#ON_SOURCE`. Distinguish field roles using all four
physical cues together:

1. **Primary flux / bandpass calibrator:** Matches a catalogued primary standard coordinate
   and name (`3C286` / `J1331+3030` / `1331+305`, `3C48` / `J0137+3309` / `0137+331`,
   `3C147` / `J0542+4951` / `0542+498`, `3C138` / `J0521+1638` / `0521+166`), typically observed
   in long high-SNR scans at the start, middle, or end of the track.
2. **Mosaic target cluster:** Multiple pointings sharing a common base target name or tight
   sky cluster separated by roughly `0.5` to `1.0` primary-beam FWHM (`~4` to `9 arcmin` at VLA
   C-band), cycled sequentially between calibrator visits.
3. **Secondary complex-gain (phase) calibrator:** A compact extragalactic point source (`J...`
   coordinate name, coherent flat visibility amplitude across UV distance, separated by degrees
   from the mosaic cluster, typically `2` to `15 deg` away) observed repeatedly to bracket every
   target mosaic cycle.
4. **Auxiliary polarization leakage calibrator (if present):** Often observed in `1` or `2` scans
   with `CALIBRATE_POL_LEAKAGE` intent or at a distinct sky position separate from the repeating
   target-bracketing phase calibrator.
"""

    files[
        "radio-interferometry/01b-workflow-phase2.md"
    ] = """# 01b -- Phase 2 Geometry, Elevation & Dead-Antenna Inspection

## 1. Array geometry & maximum baseline `B_max`

Read `ANTENNA.POSITION` (ITRF geocentric `X, Y, Z` in meters) and `DISH_DIAMETER`, computing
pairwise 3D baseline lengths across antennas that actually appear in the main visibility table:

```python
import casatools
import numpy as np


def inspect_antennas_and_amplitudes(vis_path: str, cal_field_ids: list[int]) -> dict:
    tb = casatools.table()
    try:
        tb.open(f"{vis_path}/ANTENNA", nomodify=True)
        ant_names = [str(x) for x in tb.getcol("NAME")]
        ant_pos = np.asarray(tb.getcol("POSITION"), dtype=float)  # shape (3, n_ant)
        dish_diam = float(np.median(tb.getcol("DISH_DIAMETER")))
        tb.close()

        tb.open(vis_path, nomodify=True)
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        fids = tb.getcol("FIELD_ID")
        flag = tb.getcol("FLAG")
        data = tb.getcol("DATA")
        tb.close()

        cross = ant1 != ant2
        cal_mask = cross & np.isin(fids, cal_field_ids)
        par_idx = [0, data.shape[0] - 1] if data.shape[0] >= 2 else [0]

        per_ant = {}
        for aid, name in enumerate(ant_names):
            m = cal_mask & ((ant1 == aid) | (ant2 == aid))
            nrows = int(np.sum(m))
            if nrows == 0:
                per_ant[name] = {"nrows": 0, "flag_frac": 1.0, "median_amp": 0.0}
                continue
            sub_flag = flag[par_idx, :, :][:, :, m]
            sub_amp = np.abs(data[par_idx, :, :][:, :, m])
            unflagged = ~sub_flag
            med_amp = float(np.median(sub_amp[unflagged])) if np.any(unflagged) else 0.0
            per_ant[name] = {
                "nrows": nrows,
                "flag_frac": float(np.mean(sub_flag)),
                "median_amp": med_amp,
            }

        live_amps = [v["median_amp"] for v in per_ant.values() if v["median_amp"] > 0]
        array_med = float(np.median(live_amps)) if live_amps else 0.0
        dead_antennas = [
            name
            for name, stats in per_ant.items()
            if stats["nrows"] == 0
            or stats["flag_frac"] >= 0.99
            or (array_med > 0.0 and stats["median_amp"] < 0.05 * array_med)
        ]

        live_ids = [i for i, n in enumerate(ant_names) if n not in dead_antennas]
        b_max_m = 0.0
        for i_idx, i in enumerate(live_ids):
            for j in live_ids[i_idx + 1 :]:
                dist = float(np.linalg.norm(ant_pos[:, i] - ant_pos[:, j]))
                b_max_m = max(b_max_m, dist)

        return {
            "dish_diameter_m": dish_diam,
            "max_baseline_m": b_max_m,
            "array_median_cal_amp": array_med,
            "dead_antennas": dead_antennas,
            "per_antenna": per_ant,
        }
    finally:
        try:
            tb.close()
        except Exception:
            pass
        tb.done()
```

**Important:** Only report dead antennas from the `ANTENNA` subtable of the Measurement Set
itself (never infer missing pad numbers that are not rows in `ANTENNA`). Always check actual
cross-correlation visibility amplitudes (`ANTENNA1 != ANTENNA2`) on calibrator scans: a broken
receiver or backend often writes unflagged (`FLAG == False`) micro-noise near zero amplitude!

## 2. Exact elevation vs time across all field timestamps

Compute elevation across **every** unique `TIME` value (MJD seconds) of the field in the main
table -- elevation varies monotonically across a scan, so the true minimum occurs at a scan
boundary rather than a scan midpoint:

```python
import astropy.units as u
import casatools
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time


def field_elevation_range_deg(vis_path: str, field_id: int) -> tuple[float, float]:
    tb = casatools.table()
    try:
        tb.open(f"{vis_path}/ANTENNA", nomodify=True)
        pos = np.median(np.asarray(tb.getcol("POSITION"), dtype=float), axis=1)
        tb.close()

        tb.open(f"{vis_path}/FIELD", nomodify=True)
        phase_dir = tb.getcol("PHASE_DIR")
        ra_rad = float(phase_dir[0, 0, field_id])
        dec_rad = float(phase_dir[1, 0, field_id])
        tb.close()

        tb.open(vis_path, nomodify=True)
        fids = tb.getcol("FIELD_ID")
        times_sec = np.unique(tb.getcol("TIME")[fids == field_id])
        tb.close()

        loc = EarthLocation.from_geocentric(pos[0] * u.m, pos[1] * u.m, pos[2] * u.m)
        obstime = Time(times_sec / 86400.0, format="mjd", scale="utc")
        target = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad, frame="icrs")
        alt_deg = target.transform_to(AltAz(obstime=obstime, location=loc)).alt.deg
        return float(np.min(alt_deg)), float(np.max(alt_deg))
    finally:
        try:
            tb.close()
        except Exception:
            pass
        tb.done()
```
"""

    files[
        "radio-interferometry/02-orientation.md"
    ] = """# 02 -- VLA Bands, Primary Beams & Scan-Intent Repair

## VLA receiver bands

| Band | Frequency range | Typical primary beam FWHM (`~42 / nu_GHz` arcmin) |
|---|---|---|
| P (`4`) | 0.22 -- 0.47 GHz | ~120 arcmin |
| L | 1.0 -- 2.0 GHz | ~30 arcmin |
| S | 2.0 -- 4.0 GHz | ~15 arcmin |
| C | 4.0 -- 8.0 GHz | ~9 arcmin |
| X | 8.0 -- 12.0 GHz | ~4.5 arcmin |
| Ku | 12.0 -- 18.0 GHz | ~3.0 arcmin |
| K | 18.0 -- 26.5 GHz | ~2.0 arcmin |
| Ka | 26.5 -- 40.0 GHz | ~1.3 arcmin |
| Q | 40.0 -- 50.0 GHz | ~1.0 arcmin |

## Repairing missing or mislabelled scan intents (`casatasks.defintent`)

When `STATE` is empty or a calibrator/target field lacks its proper `OBS_MODE` intent, repair
intents on `/workspace/input.ms` using `casatasks.defintent(..., mode="set")`:

```python
import casatasks

casatasks.defintent(
    vis="/workspace/input.ms",
    intent="CALIBRATE_FLUX#ON_SOURCE,CALIBRATE_BANDPASS#ON_SOURCE",
    mode="set",
    field=flux_bp_field,
)
casatasks.defintent(
    vis="/workspace/input.ms",
    intent="CALIBRATE_PHASE#ON_SOURCE",
    mode="set",
    field=phase_cal_field,
)
casatasks.defintent(
    vis="/workspace/input.ms",
    intent="OBSERVE_TARGET#ON_SOURCE",
    mode="set",
    field=",".join(target_fields),
)
```

If an auxiliary polarization leakage calibrator is present, preserve or assign its
`CALIBRATE_POL_LEAKAGE#ON_SOURCE` intent as well.
"""

    files[
        "radio-interferometry/03-instrument-sanity.md"
    ] = """# 03 -- Instrument Sanity, Elevation & Reference Antenna Selection

## Elevation and shadowing regimes

- **Bandpass calibrator minimum elevation:**
  - `>= 20 deg`: healthy airmass and stable spillover/opacity.
  - `15 -- 20 deg`: elevated atmospheric phase scatter and ground pickup; inspect low-elevation scans.
  - `< 15 deg`: high airmass (`> 3.8`) and geometric shadowing risk in compact configurations (`C` / `D`);
    always run `casatasks.flagdata(..., mode="shadow", tolerance=0.0, flagbackup=False)`.
- **Scan-start settling (`quack`) and short setup scans:**
  - Interferometer antennas slewing onto a source frequently exhibit degraded amplitude or phase
    during the first integration (`dt = median(INTERVAL)`) of each scan, or across a brief initial
    setup/dummy scan at the very beginning of an observation.
  - Measure per-scan duration and compare parallel-hand cross-correlation amplitude in the first
    integration of each scan against interior integrations of the same scan, and compare any short
    initial calibrator setup scan against full-length scans on the same calibrator before setting
    `quackinterval` or flagging a corrupted setup scan.

## Reference antenna (`refant`) selection across all calibrators

Score candidate antennas across **every** calibrator (`flux_field` and `phase_field`), not just one:

1. **Alive and unflagged on both parallel hands:** `flag_fraction < 0.10` across calibrator scans,
   with cross-correlation parallel-hand median amplitude near the array median (never dead or
   single-pol degraded).
2. **Central array position:** Compute each live antenna's Euclidean distance `d_i` from the
   median `(X, Y, Z)` of all live antennas in `ANTENNA.POSITION`. Rank live, low-flag antennas
   by ascending `d_i` (preferring central pads over outer arm ends) and pass either the top
   antenna or a comma-separated ranked list (`"antA,antB,antC"`) to `refant`.
"""

    files[
        "radio-interferometry/04-diagnostic-reasoning.md"
    ] = """# 04 -- Diagnostic Triage & Go/No-Go Checklist

Before modifying flags or solving calibration tables, confirm and record:

1. `flux_calibrator` and `bandpass_calibrator`: exact field name(s) in `FIELD.NAME`.
2. `phase_calibrator`: exact field name(s) bracketing science target scans (cleanly separated
   from mosaic target pointings and any single-scan leakage calibrator).
3. `target_fields`: ordered list of all science target mosaic pointing names.
4. `dead_antennas`: exact antenna names from `ANTENNA.NAME` with zero rows, `>= 99%` flags, or
   near-zero cross-correlation amplitudes on calibrator scans.
5. `bandpass_min_elevation_deg`: minimum elevation in degrees across all timestamps of the
   bandpass calibrator.
"""

    files[
        "radio-interferometry/05-calibrator-science.md"
    ] = """# 05 -- Primary Flux Density Standards & `casatasks.setjy`

## Standard VLA primary flux calibrators (`Perley-Butler 2017`)

| Canonical names | J2000 RA, Dec (approx) | Validity & role |
|---|---|---|
| `3C286`, `J1331+3030`, `1331+305` | `13h31m08.29s`, `+30d30m32.96s` (`202.785 deg`, `+30.509 deg`) | Primary flux, bandpass, and polarization angle standard (`1 -- 50 GHz` in `Perley-Butler 2017`). |
| `3C48`, `J0137+3309`, `0137+331` | `01h37m41.30s`, `+33d09m35.13s` (`24.422 deg`, `+33.160 deg`) | Primary flux and bandpass standard (`Perley-Butler 2017`). |
| `3C147`, `J0542+4951`, `0542+498` | `05h42m36.14s`, `+49d51m07.23s` (`85.651 deg`, `+49.852 deg`) | Primary flux and bandpass standard (`Scaife-Heald 2012` at P-band; `Perley-Butler 2017` at L-band and above). |
| `3C138`, `J0521+1638`, `0521+166` | `05h21m09.89s`, `+16d38m22.05s` (`80.291 deg`, `+16.640 deg`) | Primary flux and polarization angle standard (`Perley-Butler 2017`). |

## Populating physical `MODEL_DATA` (`usescratch=True`)

Always pass `usescratch=True` when calling `casatasks.setjy` so CASA writes the physical
`MODEL_DATA` column on disk rather than a virtual header keyword:

```python
import os
import casatasks

cal_models_dir = "/opt/casa-data/nrao/VLA/CalModels"
model_img = f"{cal_models_dir}/3C286_C.im"

# Use the VLA component model image when baselines resolve the source and the model exists;
# in compact configurations (e.g. D-config C-band where 3C286 is < 80 klambda and > 99% point-like),
# analytical Perley-Butler 2017 (or 3C286_C.im) sets accurate parallel-hand MODEL_DATA (~7.6 Jy at 4.6 GHz).
casatasks.setjy(
    vis=vis_path,
    field=flux_field,
    standard="Perley-Butler 2017",
    model=model_img if (use_resolved_model and os.path.isdir(model_img)) else "",
    usescratch=True,
)
```

Immediately verify `MODEL_DATA` with `casatools.table` (`nomodify=True`): confirm `"MODEL_DATA"`
is in `tb.colnames()` and its parallel-hand (`RR`, `LL`) mean amplitude on `flux_field` matches
the expected `Perley-Butler 2017` spectrum (not the `1.0 Jy` uninitialized default).
"""

    files[
        "radio-interferometry/06-failure-modes.md"
    ] = """# 06 -- Common CASA Failure Modes & Prevention

1. **Stale `table.lock` files:**
   Every `casatools` open must pair with `.close()` and `.done()` inside `try / finally`. Before
   exiting, verify `[str(p) for p in Path("/workspace").rglob("table.lock")]` is empty (or confirm
   `tb.open(path, nomodify=True)` succeeds cleanly on every output table).
2. **Blanket over-flagging of clean data:**
   Never blindly flag wide edge-channel ranges (such as `10--25%` of a spectral window) or run
   low-threshold `tfcrop`/`rflag` (`< 4.5 sigma`) on uncalibrated visibilities without first checking
   per-channel median amplitude! If edge channels have normal amplitude and noise, flagging them
   destroys clean bandwidth and inflates false-positive flagging. Flag only channels, integrations,
   scans, shadowed rows, and dead antennas where the data show real corruption.
3. **Numeric `FIELD_ID` mismatch after `split()`:**
   `split()` renumbers selected fields sequentially from `0`. Passing numeric IDs derived from
   `input.ms` into solves on `calibrators.ms` (or vice versa) silently corrupts calibration transfer.
   Always select by field **name**.
4. **Negative CLEAN model bowls on extended mosaics:**
   Interferometers lack zero-spacing flux (`uv = 0`). Unmasked `hogbom` or deep unmasked
   `multiscale` cleaning on bright extended structure deconvolves negative sidelobe troughs into
   large negative `.model` components around the source. Prevent this with `gridder="mosaic"`,
   `normtype="flatnoise"`, `deconvolver="multiscale"`, `usemask="auto-multithresh"` with
   `negativethreshold=0.0`, and a physical `3 * sigma_thermal` stopping `threshold`.
"""

    files[
        "radio-interferometry/10-precal-workflow.md"
    ] = """# 10 -- Pre-Calibration Workflow (`casatools` + `casatasks`)

Goal: starting from raw `/workspace/input.ms`, repair missing scan intents, excise deterministic
and statistical RFI/hardware corruptions without over-flagging clean data, populate physical
`MODEL_DATA` (`usescratch=True`) on the primary flux calibrator, and split a clean, unlocked
`/workspace/calibrators.ms`.

## Step 1 -- Metadata triage & scan-intent repair

1. Inspect `FIELD`, `STATE`, `SPECTRAL_WINDOW`, `ANTENNA`, and scan cadence (`01-workflow.md`,
   `01b-workflow-phase2.md`).
2. Identify `flux_field` / `bp_field`, `phase_field`, `target_fields`, any auxiliary polarization
   calibrator, and any `dead_antennas`.
3. Repair missing `STATE` scan intents on `/workspace/input.ms` via `casatasks.defintent(..., mode="set")`.

## Step 2 -- Data-driven deterministic pre-flagging on `/workspace/input.ms`

Inspect visibility amplitudes on `/workspace/input.ms` before choosing flag commands:

1. **Zero-valued visibilities and dead antennas:**
   - Flag exact zeros (`mode="clip", clipzeros=True, flagbackup=False`).
   - Explicitly flag any dead/non-functional antenna (`mode="manual", antenna=",".join(dead_antennas)`).
2. **Geometric shadowing:**
   - Run `casatasks.flagdata(vis="/workspace/input.ms", mode="shadow", tolerance=0.0, flagbackup=False)`.
3. **Scan-start settling (`quack`) and anomalous setup scans:**
   - Measure integration time `dt = float(np.median(tb.getcol("INTERVAL")))`.
   - Compare cross-correlation parallel-hand amplitude of the first integration (`t - t_scan_min < 0.5 * dt`)
     against interior integrations of each scan. When scan-start integrations show amplitude drop,
     spike, or phase jump, flag the settling interval (`mode="quack", quackinterval=0.5 * dt` to `dt`,
     `quackmode="beg", flagbackup=False`).
   - Check each calibrator scan's duration and amplitude distribution: if a brief initial setup scan
     at the start of the observation has anomalous amplitude or severe dropouts compared to normal
     scans of the same calibrator, flag that setup scan (`mode="manual", scan=str(bad_scan)`).
4. **Extreme raw RFI spikes before initial bandpass:**
   - Inspect per-channel median and MAD (`1.4826 * median(|x - median(x)|)`) of cross-correlation
     amplitudes on calibrators and targets. Flag persistently corrupted narrowband RFI channels
     (`amp_chan > med + 6 * mad`) or extreme raw visibility spikes (`tfcrop` with `timecutoff=4.0,
     freqcutoff=4.0` and `extendpols=True`) without flagging clean edge channels.

## Step 3 -- Flux density model (`setjy`) and `/workspace/calibrators.ms` split

1. Populate `MODEL_DATA` on `/workspace/input.ms` for `flux_field`:

```python
import casatasks

casatasks.setjy(
    vis="/workspace/input.ms",
    field=flux_field,
    standard="Perley-Butler 2017",
    usescratch=True,
)
```

2. Split all calibrator fields by **name** into `/workspace/calibrators.ms` and populate
   `MODEL_DATA` on `/workspace/calibrators.ms` as well:

```python
import casatasks

cal_field_str = ",".join(all_calibrator_names)
casatasks.split(
    vis="/workspace/input.ms",
    outputvis="/workspace/calibrators.ms",
    field=cal_field_str,
    datacolumn="data",
    keepflags=True,
)
casatasks.setjy(
    vis="/workspace/calibrators.ms",
    field=flux_field,
    standard="Perley-Butler 2017",
    usescratch=True,
)
```

(When VLA elevation gain curves are required at high frequencies or wide elevation ranges,
generate `gain_curves.gc` via `casatasks.gencal(vis=..., caltable=..., caltype="gc")`.)

## Step 4 -- Initial `G0 + B0` bandpass flatten and residual `rflag`

Strong bandpass roll-off or delay phase slopes masquerade as spectral outliers if `rflag` is run
on raw `DATA`. Flatten the bandpass first on a temporary solve:

```python
import casatasks

casatasks.gaincal(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/precal_init.G0",
    field=bp_field,
    spw=center_chan_spw,
    solint="int",
    refant=refant,
    minsnr=3.0,
    calmode="p",
)
casatasks.bandpass(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/precal_init.B0",
    field=bp_field,
    solint="inf",
    combine="scan",
    refant=refant,
    minsnr=3.0,
    solnorm=True,
    fillgaps=3,
    gaintable=["/workspace/precal_init.G0"],
)
casatasks.applycal(
    vis="/workspace/calibrators.ms",
    field=bp_field,
    gaintable=["/workspace/precal_init.G0", "/workspace/precal_init.B0"],
    interp=["nearest", "nearest,linear"],
    calwt=False,
    applymode="calonly",
    flagbackup=False,
)
casatasks.flagdata(
    vis="/workspace/calibrators.ms",
    field=bp_field,
    mode="rflag",
    datacolumn="residual",
    timedevscale=5.0,
    freqdevscale=5.0,
    extendpols=True,
    flagbackup=False,
)
```

If transient RFI bursts also affect target or secondary calibrator scans in `/workspace/input.ms`,
apply `precal_init.B0` (`applymode="calonly"`) to `/workspace/input.ms` and run conservative
`rflag` (`datacolumn="corrected"`, `timedevscale=5.0`, `freqdevscale=5.0`, `extendpols=True`),
checking `flagdata(mode="summary")` before and after so clean visibilities remain intact, then
re-split `/workspace/calibrators.ms` (and re-run `setjy(..., usescratch=True)` on `calibrators.ms`)
so `input.ms` and `calibrators.ms` carry identical pre-calibration flags!
"""

    files[
        "radio-interferometry/07-calibration-execution.md"
    ] = """# 07 -- Full Continuum Calibration (`G0 -> K -> B -> G -> fluxscale -> applycal`)

Goal: solve parallel-hand continuum calibration on `/workspace/calibrators.ms` (or `/workspace/input.ms`
by field name if `calibrators.ms` was not pre-split), bootstrap the secondary phase calibrator
flux density scale, and apply calibration to `/workspace/input.ms` (`CORRECTED_DATA`).

## Step 0 -- Pre-flight antenna health, `refant`, and `MODEL_DATA` check

1. Inspect cross-correlation (`ANTENNA1 != ANTENNA2`) visibility amplitudes per antenna on
   calibrator scans in `/workspace/input.ms` and `/workspace/calibrators.ms` (`01b-workflow-phase2.md`).
   - Any non-functional antenna whose cross-correlation amplitude is zero or `< 5%` of the array
     median **must** be flagged on both `/workspace/input.ms` and `/workspace/calibrators.ms`
     (`flagdata(..., mode="manual", antenna=dead_str, flagbackup=False)`) before solving!
2. Score `refant` across **all** calibrators (`flux_field` and `phase_field`): pick a central,
   low-flag (`< 10%`) antenna live on both parallel hands (`RR`, `LL`).
3. Confirm `MODEL_DATA` on `flux_field` in both `/workspace/calibrators.ms` and `/workspace/input.ms`
   holds the `Perley-Butler 2017` model (`usescratch=True`), not unit `1.0 Jy` visibilities.

## Step 1 -- Initial integration phase (`G0`) on central channels

```python
import casatasks

casatasks.gaincal(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/initial_phase.G0",
    field=bp_field,
    spw=center_chan_spw,
    solint="int",
    refant=refant,
    minblperant=3,
    minsnr=3.0,
    gaintype="G",
    calmode="p",
)
```

## Step 2 -- Single-band antenna delay (`K`)

```python
import casatasks

casatasks.gaincal(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/delay.K",
    field=bp_field,
    solint="inf",
    combine="scan",
    refant=refant,
    minblperant=3,
    minsnr=3.0,
    gaintype="K",
    gaintable=["/workspace/initial_phase.G0"],
    interp=["nearest"],
)
```

## Step 3 -- Complex bandpass (`B`, `solnorm=False`)

When `MODEL_DATA` on `bp_field` is populated from `setjy(..., standard="Perley-Butler 2017", usescratch=True)`,
solve bandpass with `solnorm=False` (and `fillgaps=3` across narrow flagged RFI channels) so
the true flux scale and spectral curvature are captured directly in `bandpass.B`:

```python
import casatasks

casatasks.bandpass(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/bandpass.B",
    field=bp_field,
    solint="inf",
    combine="scan",
    refant=refant,
    minblperant=3,
    minsnr=3.0,
    bandtype="B",
    solnorm=False,
    fillgaps=3,
    gaintable=["/workspace/initial_phase.G0", "/workspace/delay.K"],
    interp=["nearest", "nearest"],
)
```

## Step 4 -- Complex gain (`G`) on primary and secondary calibrators

To prevent intra-scan atmospheric phase drift from decorrelating vector-averaged scan amplitudes
during `fluxscale`, solve short-interval phase (`phase_int.Gp`, `solint="int"`, `calmode="p"`) on
all calibrators first (or verify intra-scan phase scatter is `< 10 deg`), then solve scan-averaged
amplitude+phase (`gain.G`, `solint="inf"`, `calmode="ap"`):

```python
import casatasks

all_cals = f"{flux_field},{phase_field}"
casatasks.gaincal(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/phase_int.Gp",
    field=all_cals,
    solint="int",
    refant=refant,
    minblperant=3,
    minsnr=2.5,
    gaintype="G",
    calmode="p",
    gaintable=["/workspace/delay.K", "/workspace/bandpass.B"],
    interp=["nearest", "nearest,linear"],
)

casatasks.gaincal(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/gain.G",
    field=flux_field,
    solint="inf",
    refant=refant,
    minblperant=3,
    minsnr=3.0,
    gaintype="G",
    calmode="ap",
    gaintable=["/workspace/delay.K", "/workspace/bandpass.B", "/workspace/phase_int.Gp"],
    interp=["nearest", "nearest,linear", "nearest"],
    append=False,
)
casatasks.gaincal(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/gain.G",
    field=phase_field,
    solint="inf",
    refant=refant,
    minblperant=3,
    minsnr=3.0,
    gaintype="G",
    calmode="ap",
    gaintable=["/workspace/delay.K", "/workspace/bandpass.B", "/workspace/phase_int.Gp"],
    interp=["nearest", "nearest,linear", "nearest"],
    append=True,
)
```

## Step 5 -- Bootstrap secondary calibrator flux scale (`fluxscale`)

```python
import casatasks

flux_res = casatasks.fluxscale(
    vis="/workspace/calibrators.ms",
    caltable="/workspace/gain.G",
    fluxtable="/workspace/gain.fluxscaled",
    reference=[flux_field],
    transfer=[phase_field],
)
```

Inspect `flux_res` to verify the bootstrapped Stokes I flux density of `phase_field` is positive,
finite, and has small formal fractional uncertainty (`< 5%`).

## Step 6 -- Apply calibration (`applycal`) to `/workspace/input.ms` by field name

Apply calibration to calibrators (including `phase_int.Gp` on calibrator scans) and to all science
target fields (interpolating `gain.fluxscaled` **linearly** in time from `phase_field`):

```python
import casatasks

casatasks.applycal(
    vis="/workspace/input.ms",
    field=flux_field,
    gaintable=[
        "/workspace/delay.K",
        "/workspace/bandpass.B",
        "/workspace/phase_int.Gp",
        "/workspace/gain.fluxscaled",
    ],
    gainfield=["", "", flux_field, flux_field],
    interp=["nearest", "nearest,linear", "nearest", "nearest"],
    calwt=False,
    applymode="calflag",
    flagbackup=False,
)

casatasks.applycal(
    vis="/workspace/input.ms",
    field=phase_field,
    gaintable=[
        "/workspace/delay.K",
        "/workspace/bandpass.B",
        "/workspace/phase_int.Gp",
        "/workspace/gain.fluxscaled",
    ],
    gainfield=["", "", phase_field, phase_field],
    interp=["nearest", "nearest,linear", "nearest", "linear"],
    calwt=False,
    applymode="calflag",
    flagbackup=False,
)

casatasks.applycal(
    vis="/workspace/input.ms",
    field=",".join(target_fields),
    gaintable=[
        "/workspace/delay.K",
        "/workspace/bandpass.B",
        "/workspace/gain.fluxscaled",
    ],
    gainfield=["", "", phase_field],
    interp=["nearest", "nearest,linear", "linear"],
    calwt=False,
    applymode="calflag",
    flagbackup=False,
)
```

## Step 7 -- Post-applycal verification

Open `/workspace/input.ms` read-only with `casatools.table` and verify:
1. `"CORRECTED_DATA"` is present in `tb.colnames()`.
2. Every dead antenna is `100%` flagged.
3. On `flux_field` parallel hands (`RR`, `LL`), vector-averaged `CORRECTED_DATA` channel phase
   slope across the band is flat (`< 5 deg / 100 MHz`) and mean amplitude agrees with `MODEL_DATA`
   within `1--2%`.
4. On `phase_field` parallel hands, vector-averaged `CORRECTED_DATA` amplitude agrees with the
   `fluxscale` Stokes I flux density.
5. Inspect target `CORRECTED_DATA` amplitude distribution (`13-postcal-rfi-flagging.md`) for any
   unflagged RFI outliers while preserving clean target visibilities.
"""

    files[
        "radio-interferometry/07b-gaincal-recovery.md"
    ] = """# 07b -- Gaincal Recovery & Low-SNR Diagnostics

If `gaincal` or `bandpass` flags `> 15%` of live-antenna solutions:
1. Confirm `refant` is unflagged across all calibrator scans; if flagged in some scans, pass a
   comma-separated ranked `refant` list (`"ant1,ant2,ant3"`).
2. Confirm `MODEL_DATA` on `bp_field` is populated (`usescratch=True`) and not `0` or `1 Jy`.
3. If `solint="int"` on a faint secondary calibrator drops below `minsnr=3.0`, widen the short
   phase `solint` to `"30s"` or `"60s"` while keeping `minblperant=3`.
"""

    files["radio-interferometry/08-pband-specifics.md"] = """# 08 -- VLA P-Band Specifics

At VLA P-band (`224 -- 470 MHz`):
- Use `standard="Scaife-Heald 2012"` for `3C147` (`Perley-Butler 2017` does not cover `3C147` at P-band).
- Ionospheric phase varies rapidly (`5 -- 15 s`); keep phase `solint="int"`.
- Wide-field W-terms are always significant (`fresnel << 0.9`); use `gridder="wproject"` or `"mosaic"`.
"""

    files[
        "radio-interferometry/09-polcal-execution.md"
    ] = """# 09 -- Full-Stokes Polarization Calibration (`KCROSS -> Df -> Xf`)

When `stokes="IQUV"` is requested (skip for `stokes="I"` continuum reductions):
1. Populate full-Stokes `I, Q, U, V` model on the polarized angle calibrator (`3C286` or `3C138`).
2. Solve cross-hand delay `KCROSS` (`gaincal(..., gaintype="KCROSS", solint="inf", combine="scan")`).
3. Solve instrumental leakage `Df` (`polcal(..., poltype="Df", solint="inf", combine="scan")`) on an
   unpolarized leakage calibrator (or `Df+QU` with parallactic-angle spread).
4. Solve R-L phase angle `Xf` (`polcal(..., poltype="Xf", solint="inf", combine="scan")`) on the
   polarization angle calibrator.
5. Re-run `applycal` including `[KCROSS, Df, Xf]` with `parang=True`.
"""

    files[
        "radio-interferometry/09b-polcal-reference.md"
    ] = """# 09b -- Polarization Calibrator Reference Notes

- `3C286` (`J1331+3030`) at C-band (`4 -- 8 GHz`): fractional linear polarization `~11.2%`,
  intrinsic polarization angle `+33 deg` (`+0.576 rad`).
- `3C138` (`J0521+1638`) at C-band: fractional linear polarization `~10%`, intrinsic angle `-11 deg`.
- Compact unpolarized leakage calibrators (`p < 0.3%`): solve `poltype="Df"` assuming unit
  Stokes I and zero intrinsic `Q, U`.
"""

    files[
        "radio-interferometry/11-imaging.md"
    ] = """# 11 -- Continuum & Mosaic Imaging (`casatasks.tclean`)

Goal: derive physically principled `cell`, `imsize`, `phasecenter`, `gridder`, `deconvolver`,
`scales`, masking, and `threshold` from the calibrated Measurement Set, run `casatasks.tclean`
with `pbcor=True`, and verify the restored image products.

## Step 1 -- Select calibrated column & target fields

Open `/workspace/input.ms` read-only with `casatools.table`:
- Set `datacolumn = "corrected"` if `"CORRECTED_DATA" in tb.colnames()`, else `"data"`.
- Select only the science target fields by **name** (`field = ",".join(target_fields)`).
- Follow the output image prefix requested by the user prompt (for example `/workspace/mosaic_target`),
  or pick a clear prefix in `/workspace` if none is specified. Remove any stale `<prefix>.*`
  tables before re-running `tclean`.

## Step 2 -- Derive `cell`, `phasecenter`, and composite `imsize`

1. **Synthesized beam & `cell` (`3` to `5` pixels per beam minor axis):**
   - From maximum unflagged baseline `B_max_m` on the target fields and reference wavelength
     `lambda_m = c / nu_ref_hz`:
     `theta_syn_arcsec = (lambda_m / B_max_m) * 206265.0`
   - Target `4.0` pixels per beam (`cell_arcsec = round(theta_syn_arcsec / 4.0, 2)`), keeping
     `3.0 <= b_minor_arcsec / cell_arcsec <= 5.0` (and always inside `[3.0, 7.0]` on the actual
     restored beam from `ia.restoringbeam()`!).
2. **Mosaic `phasecenter` and `imsize`:**
   - Primary beam FWHM: `pb_fwhm_arcsec = 1.02 * (lambda_m / dish_diam_m) * 206265.0`.
   - For multi-pointing mosaics, set `phasecenter` to the J2000 centroid of the target pointing
     directions (`"J2000 <ra_hms> <dec_dms>"` or mean pointing center), compute the angular span
     `(span_ra_arcsec, span_dec_arcsec)` across all target pointings, pad by `2.5 * pb_fwhm_arcsec`
     to `3.0 * pb_fwhm_arcsec` so the full `pb >= 0.2` mosaic footprint and first sidelobe are
     covered, divide by `cell_arcsec`, and round **up** to a `2^a * 3^b * 5^c` FFT-friendly integer:

```python
import math


def next_smooth_imsize(min_pixels: int, floor_px: int = 256) -> int:
    target = max(int(math.ceil(min_pixels)), floor_px)
    candidates = []
    for a in range(1, 12):
        v2 = 2**a
        for b in range(0, 6):
            v23 = v2 * (3**b)
            for c in range(0, 4):
                val = v23 * (5**c)
                if val >= target and val % 2 == 0:
                    candidates.append(val)
    return min(candidates)
```

## Step 3 -- Derive radiometer stopping `threshold`

Estimate natural thermal noise `sigma_nat` from the VLA radiometer equation (or unflagged
visibility `WEIGHT` distribution and channel-to-channel Stokes I difference noise):
- `sigma_nat = SEFD_Jy / (eta_c * sqrt(2 * N_live * (N_live - 1) * t_per_pointing_s * bw_eff_hz))`
  (at VLA C-band: `SEFD ~ 310 Jy`, `eta_c ~ 0.93`; for Briggs `robust=0.5`, `sigma_briggs ~ 1.2 * sigma_nat`).
- Set `threshold = f"{3.0 * sigma_briggs * 1e3:.3f}mJy"` (`~3 * sigma_thermal`).

## Step 4 -- Mosaic multiscale deconvolution with positive auto-masking

For multi-pointing VLA continuum mosaics containing extended structure (or mixed point + extended
emission):
- `gridder="mosaic"`, `normtype="flatnoise"`, `pblimit=0.2`, `pbcor=True`
- `weighting="briggs"`, `robust=0.5`
- `deconvolver="multiscale"` with `scales=[0, 5, 15]` (or `[0, 5, 15, 30]` when extended emission
  spans many beams, keeping the largest scale below half the shortest-baseline maximum recoverable
  scale), `smallscalebias=0.6`
- `usemask="auto-multithresh"` with **`negativethreshold=0.0`** (`sidelobethreshold=2.0`,
  `noisethreshold=4.25`, `lownoisethreshold=1.5`, `minbeamfrac=0.3`) so CLEAN masks only positive
  signal and never digs negative `.model` bowls around extended emission!

```python
import casatasks

casatasks.tclean(
    vis="/workspace/input.ms",
    imagename=imagename_prefix,
    field=",".join(target_fields),
    datacolumn=datacolumn,
    spw="",
    stokes="I",
    specmode="mfs",
    gridder="mosaic",
    normtype="flatnoise",
    pblimit=0.2,
    deconvolver="multiscale",
    scales=[0, 5, 15],
    smallscalebias=0.6,
    cell=[f"{cell_arcsec:.2f}arcsec"],
    imsize=[imsize_px, imsize_px],
    phasecenter=mosaic_phasecenter,
    weighting="briggs",
    robust=0.5,
    niter=15000,
    threshold=threshold_str,
    usemask="auto-multithresh",
    sidelobethreshold=2.0,
    noisethreshold=4.25,
    lownoisethreshold=1.5,
    negativethreshold=0.0,
    minbeamfrac=0.3,
    pbcor=True,
    savemodel="none",
    interactive=False,
)
```

(If `auto-multithresh` is too restrictive on faint diffuse emission in a very low-SNR field, fall
back to `usemask="pb", pbmask=0.2` with a conservative `threshold = 3.5 * sigma_briggs` and monitor
the `.model` sum to ensure negative model components remain negligible relative to positive model flux.)

## Step 5 -- Post-imaging verification (`casatools.image`)

Open `.image`, `.image.pbcor`, `.pb`, `.residual`, and `.model` with `casatools.image` (closing
every handle in `try / finally` via `ia.close(); ia.done()`) and verify:
1. All five products (`.image`, `.image.pbcor`, `.pb`, `.residual`, `.model`) exist on disk with
   finite pixels and zero `table.lock` files.
2. Restoring beam minor axis `b_minor_arcsec / cell_arcsec` lies between `3.0` and `5.0` (if
   Briggs weighting yielded a slightly larger or smaller beam outside `[3.0, 7.0]`, adjust `cell`
   and `imsize` once and re-run `tclean`).
3. Off-source `.residual` RMS inside `pb >= 0.2` (excluding cleaned `.model != 0` source pixels)
   is within `~1.5x -- 2x` of the radiometer thermal noise, and peak-to-residual dynamic range
   is physical and positive.
4. Negative `.model` flux magnitude `abs(model[model < 0].sum())` is small compared to positive
   `.model` flux `model[model > 0].sum()` (confirming no divergent negative CLEAN bowl).
"""

    files[
        "radio-interferometry/12-selfcal.md"
    ] = """# 12 -- Phase-Only Self-Calibration (Optional Refinement)

Attempt a single pass of phase-only self-calibration only when the initial target image already
has high dynamic range (`>= 50`) and a clean positive `.model`:
1. Re-run or predict the clean model into `MODEL_DATA` (`savemodel="modelcolumn"` or `ft`).
2. Solve target phase-only gains (`gaincal(..., calmode="p", solint="60s", minsnr=3.0, minblperant=3)`).
3. Apply to target fields and image to a distinct prefix (`<prefix>_sc1`).
4. Retain the self-calibrated `CORRECTED_DATA` and image **only** if off-source residual RMS
   decreases and dynamic range improves by `>= 10%` without introducing negative model bowls.
"""

    files[
        "radio-interferometry/13-postcal-rfi-flagging.md"
    ] = """# 13 -- Post-Calibration Residual & Target RFI Triage

After `applycal` writes `CORRECTED_DATA` on `/workspace/input.ms`:

1. **Primary calibrator (`flux_field`):** Because physical `MODEL_DATA` holds the true sky model,
   `CORRECTED_DATA - MODEL_DATA` isolates thermal noise plus residual RFI. Never run amplitude
   or `residual` clipping on secondary calibrators or science targets whose `MODEL_DATA` is still
   the `1 Jy` placeholder!
2. **Science target (`target_fields`):** Inspect parallel-hand `CORRECTED_DATA` amplitudes across
   channels and time (`sigma = 1.4826 * MAD(Re(CORRECTED_DATA))`).
   - If target visibilities are already clean (outlier tail matches Gaussian thermal noise), do
     **not** run unnecessary target flagging!
   - If unflagged RFI bursts or bad channels inflate target visibility RMS well above the clean-channel
     thermal floor, run a conservative pass on target fields only:

```python
import casatasks

before = casatasks.flagdata(
    vis="/workspace/input.ms",
    field=",".join(target_fields),
    mode="summary",
)
casatasks.flagdata(
    vis="/workspace/input.ms",
    field=",".join(target_fields),
    mode="rflag",
    datacolumn="corrected",
    timedevscale=5.0,
    freqdevscale=5.0,
    extendpols=True,
    flagbackup=True,
)
after = casatasks.flagdata(
    vis="/workspace/input.ms",
    field=",".join(target_fields),
    mode="summary",
)
```

   - Always compare `before["flagged"] / before["total"]` vs `after["flagged"] / after["total"]`:
     if `rflag` flags an excessive jump (`> 15%` of previously clean target data), restore the
     flag backup via `casatasks.flagmanager` and flag only the specific corrupted channel/time spikes.
"""

    files["radio-interferometry/ledger.md"] = """# Reduction Ledger

Keep CASA Python scripts modular in `/workspace/` (`01_inspect.py`, `02_precal.py`,
`03_calibrate.py`, `04_image.py`), execute long steps via `/usr/local/bin/run-step.sh`, and
verify every output table is unlocked before completing the task.
"""

    files["radio-interferometry/bundles/precal.md"] = (
        files["radio-interferometry/05-calibrator-science.md"]
        + "\n\n---\n\n"
        + files["radio-interferometry/10-precal-workflow.md"]
    )
    files["radio-interferometry/bundles/calibrate.md"] = (
        files["radio-interferometry/07-calibration-execution.md"]
        + "\n\n---\n\n"
        + files["radio-interferometry/07b-gaincal-recovery.md"]
    )
    files["radio-interferometry/bundles/image.md"] = files["radio-interferometry/11-imaging.md"]
    files["radio-interferometry/bundles/image_iquv.md"] = (
        files["radio-interferometry/09-polcal-execution.md"]
        + "\n\n---\n\n"
        + files["radio-interferometry/11-imaging.md"]
    )
    files["radio-interferometry/bundles/postcal.md"] = files[
        "radio-interferometry/13-postcal-rfi-flagging.md"
    ]
    files["radio-interferometry/bundles/polcal.md"] = files[
        "radio-interferometry/09-polcal-execution.md"
    ]
    files["radio-interferometry/bundles/selfcal.md"] = files["radio-interferometry/12-selfcal.md"]

    return {rel: _dedent_python_fences(content) for rel, content in files.items()}


def validate_exported_tree(export_dir: Path) -> dict[str, object]:
    """Validate zero `ms_*` & zero ground-truth leaks in `casa/`, prefix stripping, and scoping."""
    raw_skills_dir = export_dir / "raw" / "skills"
    orch_skills_dir = export_dir / "orchestrated_skills"
    casa_skills_dir = export_dir / "casa" / "skills"
    claude_agents_dir = export_dir / "agents" / "claude"
    opencode_agents_dir = export_dir / "agents" / "opencode"

    for req_dir in (raw_skills_dir, orch_skills_dir, casa_skills_dir):
        if not req_dir.is_dir():
            raise FileNotFoundError(f"Missing exported skills directory: {req_dir}")

    # 1. Single-agent raw/skills must retain ms_* tokens, have zero plugin prefixes, and exclude orchestrate/simulators.
    for excluded in SINGLE_AGENT_EXCLUDED_SKILLS:
        if (raw_skills_dir / excluded).exists():
            raise AssertionError(f"Single-agent raw/skills must exclude {excluded!r}")
        if (casa_skills_dir / excluded).exists():
            raise AssertionError(f"Single-agent casa/skills must exclude {excluded!r}")

    if not (orch_skills_dir / "orchestrate" / "SKILL.md").is_file():
        raise AssertionError("orchestrated_skills must include orchestrate/SKILL.md")
    for excluded in ORCHESTRATED_EXCLUDED_SKILLS:
        if (orch_skills_dir / excluded).exists():
            raise AssertionError(f"orchestrated_skills must exclude {excluded!r}")

    raw_ms_matches: set[str] = set()
    raw_file_count = 0
    for path in sorted(raw_skills_dir.rglob("*")):
        if path.is_file():
            raw_file_count += 1
            if path.suffix == ".md":
                txt = path.read_text(encoding="utf-8")
                if "mcp__plugin_radio-analyst_" in txt:
                    raise AssertionError(f"Stale plugin MCP prefix in {path}")
                raw_ms_matches.update(MS_TOKEN_RE.findall(txt))
    if not raw_ms_matches:
        raise AssertionError("Expected ms_* references inside exported raw/skills/, found none")

    for path in sorted(orch_skills_dir.rglob("*.md")):
        txt = path.read_text(encoding="utf-8")
        if "mcp__plugin_radio-analyst_" in txt:
            raise AssertionError(f"Stale plugin MCP prefix in {path}")

    for skills_tree in (raw_skills_dir, orch_skills_dir, casa_skills_dir):
        for skill_md in sorted(skills_tree.rglob("SKILL.md")):
            if not skill_md.is_file():
                continue
            skill_text = skill_md.read_text(encoding="utf-8")
            fm_match = _FRONTMATTER_RE.match(skill_text)
            if fm_match is None or _FRONTMATTER_NAME_RE.search(fm_match.group(2)) is None:
                raise AssertionError(f"Missing frontmatter name: in {skill_md}")

    for doc_name in ("CLAUDE.md", "AGENTS.md"):
        raw_doc = (export_dir / "raw" / doc_name).read_text(encoding="utf-8")
        if RAW_TRANSLATION_NOTE not in raw_doc:
            raise AssertionError(f"Missing 1-line translation note in raw/{doc_name}")
        if 'skill(name="radio-interferometry")' not in raw_doc:
            raise AssertionError(f"Missing radio-interferometry skill mandate in raw/{doc_name}")
        for dash in FORBIDDEN_DASHES:
            if dash in raw_doc:
                raise AssertionError(f"Forbidden em/en dash in raw/{doc_name}")
        full_doc = (export_dir / "full" / doc_name).read_text(encoding="utf-8")
        if 'skill(name="radio-interferometry")' not in full_doc:
            raise AssertionError(f"Missing radio-interferometry skill mandate in full/{doc_name}")
        orch_doc = (export_dir / "orchestrated" / doc_name).read_text(encoding="utf-8")
        if 'skill(name="orchestrate")' not in orch_doc:
            raise AssertionError(f"Missing orchestrate skill mandate in orchestrated/{doc_name}")
        casa_doc = (export_dir / "casa" / doc_name).read_text(encoding="utf-8")
        if 'skill(name="radio-interferometry")' not in casa_doc:
            raise AssertionError(f"Missing radio-interferometry skill mandate in casa/{doc_name}")

    # 2. CASA tree must have ZERO ms_* tokens, ZERO em/en dashes, ZERO dataset/eval leaks, and valid Python blocks.
    casa_file_count = 0
    parsed_py_blocks = 0
    for path in sorted((export_dir / "casa").rglob("*")):
        if not path.is_file():
            continue
        casa_file_count += 1
        text = path.read_text(encoding="utf-8")
        leaked = MS_TOKEN_RE.findall(text)
        if leaked:
            raise AssertionError(
                f"CASA-native file {path} leaked ms_* tokens: {sorted(set(leaked))}"
            )
        for dash in FORBIDDEN_DASHES:
            if dash in text:
                raise AssertionError(f"CASA-native file {path} contains forbidden em/en dash")
        lower_text = text.lower()
        for bad_sub in CASA_FORBIDDEN_SUBSTRINGS:
            if bad_sub in lower_text:
                raise AssertionError(
                    f"CASA-native file {path} contains forbidden dataset/eval substring {bad_sub!r}"
                )
        if path.suffix == ".md":
            for block_idx, py_match in enumerate(PYTHON_FENCE_RE.finditer(text)):
                code_str = py_match.group(1)
                try:
                    ast.parse(code_str)
                except SyntaxError as exc:
                    raise AssertionError(
                        f"Invalid Python code block #{block_idx} in {path}: {exc}"
                    ) from exc
                parsed_py_blocks += 1

    # 3. Stage agents and commands must exist, have zero em/en dashes, and have zero plugin prefixes.
    for stage in STAGE_NAMES:
        for agent_dir in (claude_agents_dir, opencode_agents_dir):
            agent_file = agent_dir / f"{stage}-stage.md"
            if not agent_file.is_file():
                raise FileNotFoundError(f"Missing exported stage agent: {agent_file}")
            text = agent_file.read_text(encoding="utf-8")
            for dash in FORBIDDEN_DASHES:
                if dash in text:
                    raise AssertionError(f"Stage agent {agent_file} contains forbidden em/en dash")
            if "mcp__plugin_radio-analyst_" in text:
                raise AssertionError(f"Unmapped plugin MCP prefix in {agent_file}")

    return {
        "export_dir": str(export_dir),
        "raw_skill_files": raw_file_count,
        "raw_unique_ms_tokens": len(raw_ms_matches),
        "orchestrated_skill_files": sum(1 for p in orch_skills_dir.rglob("*") if p.is_file()),
        "casa_files": casa_file_count,
        "casa_ms_tokens": 0,
        "casa_parsed_python_blocks": parsed_py_blocks,
        "stage_agents": len(STAGE_NAMES),
    }


def _sync_default_pack_compat_root(pack_dir: Path, compat_root: Path) -> None:
    """Mirror default pack skill subtrees to `<work_dir>/exported_skills/` without touching other packs."""
    compat_root.mkdir(parents=True, exist_ok=True)
    for subtree in ("raw", "casa", "full", "orchestrated", "orchestrated_skills", "agents"):
        dst_sub = compat_root / subtree
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.copytree(pack_dir / subtree, dst_sub)
    shutil.copy2(pack_dir / "manifest.json", compat_root / "manifest.json")


def export_skills(
    *,
    radio_analyst_repo: Path | None = None,
    sha: str | None = None,
    output_dir: Path | None = None,
    pack_id: str | None = None,
    cfg: BenchConfig | None = None,
) -> dict[str, object]:
    """Export deterministic `raw`, `orchestrated_skills`, and `casa` skill trees + stage subagents."""
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    pack = resolved_cfg.get_pack(pack_id)

    if radio_analyst_repo is not None:
        eff_repo = radio_analyst_repo
        eff_ref = sha if sha is not None else pack.analyst_ref
        full_sha, short_sha = resolve_git_ref(eff_repo, eff_ref)
    else:
        eff_repo, full_sha, short_sha = materialize_pack_source(
            pack, resolved_cfg.work_dir, repo_root=resolved_cfg.repo_root
        )
        if sha is not None:
            full_sha, short_sha = resolve_git_ref(eff_repo, sha)

    sync_compat = output_dir is None and pack.pack_id == resolved_cfg.default_pack
    target_dir = (
        output_dir if output_dir is not None else (resolved_cfg.exported_skills_dir / pack.pack_id)
    )
    _reject_slash_tmp(target_dir, "output_dir")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract single-agent raw/skills (excluding orchestrate & simulators) and orchestrated_skills.
    skills_tar = _git_archive_bytes(eff_repo, full_sha, [".claude/skills"])
    _extract_filtered_skills(
        skills_tar,
        ".claude/skills",
        target_dir / "raw" / "skills",
        excluded_top_dirs=SINGLE_AGENT_EXCLUDED_SKILLS,
        adapt_orchestrate_stage_names=False,
    )
    _extract_filtered_skills(
        skills_tar,
        ".claude/skills",
        target_dir / "orchestrated_skills",
        excluded_top_dirs=ORCHESTRATED_EXCLUDED_SKILLS,
        adapt_orchestrate_stage_names=True,
    )

    # 2. Extract stage agents and commands from configured radio-analyst ref.
    agents_tar = _git_archive_bytes(
        eff_repo, full_sha, ["agents", ".opencode/agents", ".claude/commands"]
    )
    tar_texts = _read_tar_text_map(agents_tar)

    claude_agents_dir = target_dir / "agents" / "claude"
    opencode_agents_dir = target_dir / "agents" / "opencode"
    commands_dir = target_dir / "agents" / "commands"
    claude_agents_dir.mkdir(parents=True, exist_ok=True)
    opencode_agents_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)

    for stage in STAGE_NAMES:
        raw_cmd = tar_texts[f".claude/commands/{stage}.md"]
        adapted_cmd = _adapt_claude_command_markdown(raw_cmd)
        cmd_out = commands_dir / f"{stage}.md"
        cmd_out.write_text(adapted_cmd, encoding="utf-8")
        cmd_out.chmod(0o644)

        raw_claude_agent = tar_texts[f"agents/{stage}-stage.md"]
        claude_out = claude_agents_dir / f"{stage}-stage.md"
        claude_out.write_text(
            _adapt_claude_stage_agent(raw_claude_agent, stage, adapted_cmd),
            encoding="utf-8",
        )
        claude_out.chmod(0o644)

        raw_opencode_agent = tar_texts[f".opencode/agents/{stage}-stage.md"]
        opencode_out = opencode_agents_dir / f"{stage}-stage.md"
        opencode_out.write_text(
            _adapt_opencode_stage_agent(raw_opencode_agent, adapted_cmd),
            encoding="utf-8",
        )
        opencode_out.chmod(0o644)

    # 3. Emit dataset-blind CASA-native steelman skill tree under casa/skills/.
    casa_skills_root = target_dir / "casa" / "skills"
    for rel_path_str, content in _build_casa_skill_files().items():
        clean_content = _sanitize_dashes(content)
        rel_path = Path(rel_path_str)
        if rel_path.name == "SKILL.md" and len(rel_path.parts) >= 2:
            clean_content = ensure_skill_frontmatter_name(clean_content, rel_path.parent.name)
        target = casa_skills_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean_content, encoding="utf-8")
        target.chmod(0o644)

    # 4. Write CLAUDE.md and AGENTS.md workspace guides per mode.
    docs = _build_workspace_docs()
    for mode_key, doc_text in docs.items():
        clean_doc = _sanitize_dashes(doc_text)
        mode_dir = target_dir / mode_key
        mode_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("CLAUDE.md", "AGENTS.md"):
            p = mode_dir / fname
            p.write_text(clean_doc, encoding="utf-8")
            p.chmod(0o644)

    summary = validate_exported_tree(target_dir)
    summary["pack_id"] = pack.pack_id
    summary["output_dir"] = str(target_dir)
    summary["radio_analyst_sha"] = short_sha
    summary["radio_analyst_commit"] = full_sha

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_path.chmod(0o644)

    if sync_compat:
        _sync_default_pack_compat_root(target_dir, resolved_cfg.exported_skills_dir)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export raw and CASA-native skill trees from configured radio-analyst pack."
    )
    add_config_cli_args(parser)
    parser.add_argument(
        "--pack",
        type=str,
        default=None,
        help="Pack ID from bench.toml (defaults to default_pack).",
    )
    parser.add_argument(
        "--radio-analyst-repo",
        type=Path,
        default=None,
        help="Explicit read-only path to radio-analyst git repository.",
    )
    parser.add_argument(
        "--sha",
        type=str,
        default=None,
        help="Explicit git commit SHA or ref to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory under work_dir/exported_skills.",
    )
    args = parser.parse_args(argv)
    cfg = config_from_cli_args(args)
    summary = export_skills(
        radio_analyst_repo=args.radio_analyst_repo,
        sha=args.sha,
        output_dir=args.output_dir,
        pack_id=args.pack,
        cfg=cfg,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
