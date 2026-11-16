# AGENTS.md -- radio-bench Working Agreement and Roadmap

Engineering rules, document workflow, and multi-phase roadmap for contributors and AI agents working on `radio-bench`. Architectural invariants live in `design_docs/DESIGN.md`.

---

## 1. How We Work (Design -> Plan -> Spec -> Build)

1. **Design before code:** No new stage, metric, threshold, tolerance, dataset branch, or analysis plot is written before alignment with the maintainer.
2. **Document hierarchy:**
   - `design_docs/DESIGN.md`: long-term and short-term architecture and separation-of-concerns rules.
   - `PLAN-<phase>.md`: scoped implementation waves, verification gates, and rollback boundaries for one phase.
   - `SPEC-<stage_or_module>.md`: exact input checkpoint contract, output contract, physical observables, evaluation metric families, and scoring rubric for a single stage or module -- agreed upon before coding.
3. **Zero host hardcoding:**
   - Never hardcode user home directories, host mount paths, or commit SHAs in Python modules, shell scripts, or `pixi.toml`.
   - Containers hold the software (`/workspace`, `/seed_data`, `/tests`, `/skills`, `/logs`, `/var/tmp`); the host holds only configurable `work_dir`, `data_dir`, and `runs_dir` (`bench.toml` / environment variables, with repo-relative defaults on a fresh clone).
4. **Extractors measure; Rubrics and Analysis judge:**
   - Verifier extractors open CASA tables/images read-only inside `try / finally` blocks and return raw physical observables + provenance + completeness flags.
   - Zero pass/fail gates, arbitrary constants, or dataset-specific `if suite == ...` branches belong inside extractors.
   - Scoring curves, tolerances, weights, and failure tags live declaratively in data (`rubric.json` / `suite.toml`).
5. **Verification and Git discipline:**
   - Test the smallest unit first (one extractor, one rubric check, one container smoke test) before running full Harbor jobs.
   - Never use `/tmp` as a scratch area; route all temporary files through the configured `work_dir`.
   - Never start, kill, or restart a benchmark run without explicit approval.
   - Stage explicit file paths only (`never git add -A` or `git add .`), verify `git status` and run `pixi run check` + `pixi run test-unit` before committing, write terse plain commit messages (one subject line + 2 to 3 bullets, zero AI/co-author trailers), and never push without an explicitly named remote and branch.
   - Do not use em dashes or en dashes (`--`, `:`, or `()` only).

---

## 2. Repository Roadmap

### Phase 0 (`v0.1` -- Completed Baseline)
- Rootless Podman 4-image stack (`casa-base`, `agent-base`, `mcp-sidecar`, `verifier`) with offline CASA `6.7.6.14` and `61` `streamable-http` MCP tools.
- Initial 2-suite continuum baseline (`Suite S` synthetic VLA D-config + `Suite R` VLA 3C391 C-band mosaic) across `inspect`, `precal`, `calibrate`, `image`, and `e2e` (`52` Harbor `0.25.0` task variants, `10/10` non-LLM oracle tasks verified).
- Initial 2x2 factorial (`Tools x Skills`), two-stage cluster bootstrap CI, pricing, and failure-taxonomy collector.

### Phase 1 (`v0.2` -- Output Contracts, Evaluation Taxonomy, Analysis Tools & Portability)
1. **`SPEC-output-and-metrics.md` (In Design):**
   - Formal per-stage **Output Contract** (arm-neutral physical CASA products on disk + structured stage metadata/handoff).
   - Principled **Evaluation Metric Families** per stage vs. whole pipeline (Categorical Correctness, Set Precision/Recall, Numerical Accuracy/Error, Spatial/Spectral Distributional Similarity, and Process/Invariant Safety).
   - **Scientific Analysis & Comparison Toolkit**: CLI/library tools to diff an agent reduction directly against a reference checkpoint (visibility residuals, caltable gain/phase/delay comparison, image/PSF/residual pixel & aperture fidelity, error propagation across stages, and cost-quality Pareto diagnostics).
2. **`SPEC-config-and-portability.md`:**
   - Replace all host-specific paths and pinned SHAs with `bench.toml` + `configs/models.toml` + repo-relative defaults so any user can clone and run inside Podman/Docker.
3. **`SPEC-rubric-and-suite-manifest.md`:**
   - Split existing verifiers into pure Metric Extractors + runtime-mounted `rubric.json` (`python -m radio_bench.grading.runner`) and `suites/<suite_id>/suite.toml` (`radio-bench init-suite`), verified by exact parity on Suite S and Suite R.

### Phase 2 (`v0.3` -- Full `radio-analyst` Pipeline Stage DAG)
- Derive and implement individual stage specs matching the full `radio-analyst` workflow:
  - `SPEC-postcal.md`: post-calibration target `CORRECTED_DATA` RFI flagging (`rflag` residual RMS drop, flag-growth bounds, zero calibrator mutation).
  - `SPEC-polcal.md`: full-Stokes calibration (`Kcross`, `D-term` leakage, `Xf` EVPA, `parang=True`, Stokes `Q/U/V` fidelity).
  - `SPEC-selfcal.md`: phase and amp+phase self-calibration (DR progression, RMS reduction, integrated flux conservation).
  - `SPEC-image-cube.md`: continuum subtraction (`uvcontsub`) and spectral-line cube imaging (channel RMS, line flux `Jy km/s`, velocity width).
  - `SPEC-simulate-and-report.md`: synthetic visibility generation (`ms-simulator`) and provenance-verified HTML/replay reporting (`report`).
  - `SPEC-chains.md`: arbitrary multi-stage sub-DAG tasks (`stages = [...]`).

### Phase 3 (`v1.0` -- Multi-Telescope & Diagnostic Stress Benchmark)
- Onboard frozen benchmark suites across **VLA** (C/L/P/Ku), **MeerKAT** (L/UHF), **uGMRT** (Band 3/4), and **ALMA** (Band 6).
- Add diagnostic stress suites targeting real interferometric failure modes: missing/ambiguous UVFITS scan intents, resolved primary flux calibrators, low-SNR `gaincal` recovery, wideband `mtmfs` spectral curvature, and marginal parallactic-angle polarization runs.
- Open plug-in contract (`[tool_pack]`, `[skill_pack]`, `configs/models.toml`) for external radio reduction agents and models.
