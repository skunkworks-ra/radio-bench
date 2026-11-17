# radio-bench -- Design Document

**Status:** Draft for alignment (not yet implemented)
**Last revised:** 2026-11-15
**Scope:** Portable multi-user, multi-dataset, multi-stage benchmark architecture for radio interferometric data reduction.

---

## 1. Philosophy and Core Division

### 1.1 Three Strict Separations

1. **Containers hold the software; host holds only data and runs.**
   No Python module, shell script, or Pixi task may contain a user home path or host mount path. Inside containers, paths are fixed (`/workspace`, `/seed_data`, `/tests`, `/skills`, `/logs`, `/var/tmp`). On the host, all roots resolve from `bench.toml` or environment variables (`RADIO_BENCH_WORK_DIR`, `RADIO_BENCH_DATA_DIR`, `RADIO_BENCH_RUNS_DIR`), defaulting to repo-relative `./.work`, `./data`, and `./runs` for any user who clones the repo.
2. **Extractors return measurements; Rubrics compute scores.**
   Mirroring `radio-analyst` (`tools return numbers; skills do the science`), every verifier primitive in `radio-bench` is split into two layers:
   - **Metric Extractor (code, in `verifier` image):** opens CASA tables/images read-only inside `try / finally`, measures physical quantities (`flux_cal_model_amp_jy`, `bandpass_phase_rms_deg`, `fluxscale_flux_jy`, `target_flagged_fraction`, `offsource_rms_jy`, `dynamic_range`, `negative_bowl_ratio`, `dterm_rms`, `evpa_err_deg`, `selfcal_dr_ratio`), and returns raw numbers + provenance + completeness flags. Zero hardcoded pass/fail gates or dataset names live in extractors.
   - **Rubric (data, mounted at `/tests/rubric.json`):** declares which metrics apply to a task, their scoring curves, target values/bounds (set per suite), weights, and failure-taxonomy tags. Changing metrics, tolerances, or weights never requires rebuilding `verifier:latest` or editing `diagnostics.py` / `taxonomy.py`.
3. **Datasets are data (`suite.toml`), never Python branches.**
   No `if suite == "3c391"` or field-named metric keys (`j1331_*`) are permitted in library code. A dataset suite declares its telescope, polarization basis, SpW layout, semantic field roles (`flux_cal`, `bandpass_cal`, `phase_cal`, `pol_angle_cal`, `leakage_cal`, `targets`), checkpoint recipe, and ground-truth observables in `suite.toml`.

### 1.2 Document Hierarchy (How Work Proceeds)

1. **`DESIGN.md` (this document):** architectural invariants, short-term (`v0.2`) design, and long-term (`v1.0`) design.
2. **`PLAN-<phase>.md`:** scoped implementation waves for one phase of `DESIGN.md`, with verification gates and rollback boundaries.
3. **`SPEC-<stage_or_module>.md`:** exact input checkpoint contract, measurable CASA observables, rubric schema, and failure modes for one stage (`inspect`, `precal`, `calibrate`, `postcal`, `polcal`, `image`, `selfcal`, `simulate`, `report`) -- agreed upon before any code or thresholds are written.

---

## 2. Short-Term Design (`v0.2` -- Zero Hardcoding, Pluggable Datasets & Rubrics)

Goal: make the existing 4 stages + end-to-end continuum workflow (`inspect`, `precal`, `calibrate`, `image`, `e2e`) 100% portable for any contributor on any Linux host and pluggable for new continuum datasets and custom metric rubrics.

### 2.1 Zero-Hardcode Host Configuration (`bench.toml` + Environment)

| Setting | Env Override | Default (Fresh Clone) | Purpose |
| :--- | :--- | :--- | :--- |
| `work_dir` | `RADIO_BENCH_WORK_DIR` | `<repo>/.work` | Compiled Harbor `tasks/`, `exported_skills/`, scratch, `pytest` temp |
| `data_dir` | `RADIO_BENCH_DATA_DIR` | `<repo>/data` | Synthetic MS suites and extracted stage checkpoints |
| `runs_dir` | `RADIO_BENCH_RUNS_DIR` | `<repo>/runs` | Harbor trial outputs and `analysis/` reports |
| `prompts_dir` | `RADIO_BENCH_PROMPTS_DIR` | `<repo>/prompts` | External prompt/markdown pack (for example a `Crypt` vault path or separate git repo) |
| `radio_analyst_src` | `RADIO_BENCH_ANALYST_SRC` | git URL or local path | Source repo + pinned ref (`radio_analyst_ref`) for `mcp-sidecar` build and `export-skills` |
| `casa_data_dir` | `RADIO_BENCH_CASA_DATA` | `~/.casa/data` | Host CASA measures table staged once into `casa-base` |

All `pixi.toml` tasks call `python -m radio_bench.<cmd>` with relative paths and read `bench.toml` (with `bench.local.toml` gitignored for machine-specific overrides such as `/mnt/fast`).

### 2.2 Dataset Manifest (`suites/<suite_id>/suite.toml`)

Onboarding a new dataset requires adding `suites/<suite_id>/suite.toml` (no Python edits):
- **Identity:** `suite_id`, `telescope` (`VLA`, `MeerKAT`, `uGMRT`, `ALMA`), `pol_basis` (`circular` or `linear`), `raw_archive` (tarball/MS path or URL + SHA256).
- **Semantic Field Map:** maps canonical roles (`flux_cal`, `bandpass_cal`, `phase_cal`, `pol_angle_cal`, `leakage_cal`, `check_source`, `targets`) to dataset field names, replacing all `j1331_*` / `3C391` assumptions.
- **Stage Checkpoints:** declares which stages to materialize (`raw`, `post_precal`, `post_calibrate`, ...) and how each stage's `ground_truth.json` is populated from the reference reduction.
- **Resource Profile:** per-suite container `cpus`, `memory_mb`, `n_concurrent_trials`, and stage timeouts.

### 2.3 Declarative Metric Extractor & Rubric Engine

- **Extractor Registry (`radio_bench.grading.extractors`):** pure functions keyed by canonical observable ID:
  - `inspect.*`: `flux_cal_match`, `bandpass_cal_match`, `phase_cal_match`, `target_fields_jaccard`, `dead_antennas_f1`, `bandpass_min_el_err_deg`.
  - `precal.*`: `flux_cal_model_rel_err`, `rfi_must_flag_recall`, `clean_false_positive_rate`, `calibrator_overflag_fraction`.
  - `cal.*`: `dead_antenna_flagged_fraction`, `fluxscale_rel_err`, `bandpass_phase_slope_deg_per_100mhz`, `bandpass_chan_phase_rms_deg`, `target_row_retention`, `coherent_vis_amp_rel_err`, `coherent_vis_phase_err_deg`, `verifier_tclean_dr`.
  - `image.*`: `pixels_per_beam`, `offsource_residual_rms_jy`, `dynamic_range`, `negative_bowl_ratio`, `source_flux_rel_err`, `point_source_snr`, `pb_taper_consistency`.
- **Generic Verifier Runner (`python -m radio_bench.grading.runner`):**
  - `tests/test.sh` runs `runner --rubric /tests/rubric.json --truth /tests/ground_truth.json`, eliminating per-task `verify.py` scripts.
  - Emits `reward.json` (`{"reward": float, "passed": 0|1}` for Harbor) and self-describing `metrics.json` containing `measurements`, `rubric`, `subscores`, and `provenance`.
  - `diagnostics.py` and `taxonomy.py` build `truth_vs_prediction.md` and failure-mode tables dynamically from `metrics.json["rubric"]`.

### 2.4 Model & Capability Arm Config (`configs/models.toml`)

Moves all model IDs, provider endpoints (`Anthropic`, `OpenAI-compatible` / `NRP`, local `vLLM` / `sglang`), token pricing tiers, and small-vs-frontier parity pairs out of Python into `configs/models.toml`.

---

## 3. Long-Term Design (`v1.0` -- Full Pipeline DAG & Multi-Telescope Benchmark)

Goal: evaluate agents across the complete `radio-analyst` reduction lifecycle and diverse interferometric regimes (`VLA`, `MeerKAT`, `uGMRT`, `ALMA`).

### 3.1 Full `radio-analyst` Stage Coverage

| Stage | `radio-analyst` Skill / Agent | Physical Observables Extracted by Verifier |
| :--- | :--- | :--- |
| `inspect` | `inspect` (`02`, `03`) | Field roles (including missing-intent vs catalogue cases), SpW/pol geometry, dead/shadowed antennas, `refant` candidates, min elevation, PA spread |
| `precal` | `precal` (`10`) | Priorcals (`antpos`/`rq`/`gaincurve` where applicable), flux-standard `MODEL_DATA` spectrum, preflag + initial-bandpass `rflag` recall/FPR |
| `calibrate` | `calibrate` (`07`, `07b`) | Delay `K`, bandpass `B` phase/amp flatness, complex gain `G` SNR & continuity, `fluxscale` bootstrap accuracy, `CORRECTED_DATA` fidelity |
| `postcal` | `postcal` (`13`) | Target `CORRECTED_DATA` post-rflag residual RMS drop, bounded target flag growth, zero calibrator modification, no destructive `clip` |
| `polcal` | `polcal` (`09`, `09b`) | Cross-hand delay/phase (`Kcross`/`Xf`), instrumental leakage `D-term` RMS & stability, `parang=True` application, calibrator EVPA & fractional polarization, target Stokes `Q/U/V` RMS |
| `image_cont` | `image` (`11`, MFS / MTMFS) | Beam sampling, off-source thermal RMS, dynamic range, `nterms>=2` spectral index accuracy (wideband), clean-bowl guard, PB correction |
| `image_cube` | `image` (`11`, spectral cube) | `uvcontsub` continuum-free baseline RMS, velocity/frequency rest frame, per-channel noise profile, integrated line flux (`Jy km/s`) & line width |
| `selfcal` | `selfcal` (`12`) | Pre- vs post-selfcal dynamic range ratio, off-source RMS improvement, integrated flux conservation (no clean bias / flux loss), phase-only vs amp+phase stability |
| `simulate` | `simulate` / `ms-simulator` | Array layout, SpW/channelization, synthetic sky model accuracy, requested noise/corruption injection, valid MSv2 subtables |
| `report` | `report` | Structured provenance completeness (steps vs `stage_log.jsonl`), physical metric consistency against actual MS/image state, replayable script check |

### 3.2 Arbitrary Stage Chains (Sub-DAG Tasks)

Tasks are defined as `stages = [<stage_1>, ..., <stage_k>]` starting from `checkpoint_in`:
- Single-stage tasks (`len(stages) == 1`): isolate stage competence without upstream error compounding.
- Sub-chain tasks (for example `["calibrate", "postcal", "image_cont", "selfcal"]` or `["calibrate", "polcal", "image_cont"]`): test cross-stage handoffs (`INSPECT HANDOFF` / `STAGE HANDOFF`).
- Full end-to-end tasks (`reduce` vs `orchestrate`): raw archival MS (or `.ms.tgz` / SDM) to final science products and report.

### 3.3 Multi-Telescope & Stress-Regime Suite Library

- **Telescopes:** VLA (C, L, P, Ku bands), MeerKAT (L-band, UHF linear feeds), uGMRT (Band 3/4), ALMA (Band 6).
- **Diagnostic Stress Suites (testing `radio-analyst` failure-mode reasoning):**
  1. Missing or misleading scan intents (UVFITS / archival exports requiring per-field intent vs catalogue reasoning).
  2. Resolved primary flux calibrator (`uvrange` / model image requirement).
  3. Low-SNR phase calibrator requiring `07b-gaincal-recovery` (`combine='spw'`, `solint` adaptation).
  4. Marginal parallactic-angle coverage polarization run (`polcal` feasibility + posterior leakage check).
  5. Severe wideband RFI + multi-SpW spectral-index continuum (`mtmfs`, `nterms=2`) and spectral-line HI/OH cube.

### 3.4 External Agent & Tool-Pack Plug-In Contract

Any external research group can plug in:
- Custom MCP sidecar images (`[tool_pack]` in `bench.toml`),
- Custom skill directories (`[skill_pack]`), or
- Custom Harbor agent adapters / local open-weight endpoints (`configs/models.toml`),
and run the identical frozen task suites, verifiers, and 2x2 factorial analysis (`Delta_Tools`, `Delta_Skills`, synergy `I`, cost-parity).

---

## 4. Proposed Roadmap of Plans & Specs

- **Plan 1 (`PLAN-v0.2-portability-and-rubrics.md`):**
  - Spec 1A: `bench.toml` + zero-hardcoded-path refactor (`pixi.toml`, `build_images.sh`, Python modules).
  - Spec 1B: Extractor registry + declarative `rubric.json` runner + dynamic `diagnostics`/`taxonomy` (verified by exact `10/10` parity on Suite S & Suite R).
  - Spec 1C: `suite.toml` manifest + `configs/models.toml` + `radio-bench init-suite` scaffolding.
- **Plan 2 (`PLAN-v0.3-pipeline-stages.md`):**
  - Individual stage specs (`SPEC-postcal.md`, `SPEC-selfcal.md`, `SPEC-polcal.md`, `SPEC-image-cube.md`, `SPEC-chains.md`) derived one by one from the corresponding `radio-analyst` skill docs and agreed upon before implementation.
- **Plan 3 (`PLAN-v1.0-multi-telescope-suites.md`):**
  - Dataset specs for MeerKAT, uGMRT, VLA wideband/pol/line, and archival missing-metadata stress suites.
