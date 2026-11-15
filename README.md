# radio-bench

Harbor `0.25.0` 2x2 capability ablation benchmark (`Tools x Skills`, plus `skills_raw` vs `skills_casa` prompt-adaptation controls and `11_orchestrated` multi-agent end-to-end arm) for radio interferometric data reduction in CASA `6.7.6.14`.

## Architecture

- **Container Stack (rootless Podman + Compose v2)**:
  - `casa-base`: shared `debian:trixie-slim` + Python `3.12` + `casatools==6.7.6.14` + `casatasks==6.7.6.14` + offline CASA measures data (`data_auto_update = False`, `measures_auto_update = False`).
  - `agent-base`: isolated execution container (`claude`, `opencode`, `run-step.sh`) with zero bundled MCP server libraries or grading code.
  - `mcp-sidecar`: serves `ms-inspect` (`8000`, 38 tools), `ms-modify` (`8001`, 19 tools), and `ms-create` (`8002`, 4 tools) = `61` `radio-analyst` MCP tools over loopback `streamable-http` (`network_mode: "service:main"`).
  - `verifier`: isolated offline grading container (`network_mode: "none"`) computing physical interferometric metrics from visibility tables, calibration tables, and restored/residual/model CASA images.
- **Capability Arms**:
  - `00_none`: bare CASA `agent-base` container (zero skills, zero MCP sidecar).
  - `01_skills_raw`: unmodified `radio-analyst` skill guides (`ms_*` tool names retained), zero MCP sidecar.
  - `01_skills_casa`: dataset-blind CASA-native translation of the skills (`0` `ms_*` references), zero MCP sidecar.
  - `10_tools`: `mcp-sidecar` (`61` MCP tools), zero skills.
  - `11_full`: `mcp-sidecar` + `radio-analyst` skills.
  - `11_orchestrated` (end-to-end tasks): `mcp-sidecar` + `radio-analyst` skills + stage subagents (`inspect` -> `precal` -> `calibrate` -> `image`).
- **Benchmark Suites**:
  - **Suite S (Synthetic VLA D-Config Continuum Mosaic)**: `s1_inspect`, `s1_precal`, `s1_calibrate`, `s1_image`, `s2_e2e_synth`.
  - **Suite R (Real VLA 3C391 C-Band 7-Pointing Mosaic, `spw0`)**: `r1_3c391_inspect`, `r1_3c391_precal`, `r1_3c391_calibrate`, `r1_3c391_image`, `r2_3c391_e2e`.

> Note: Task prompt files (`task_specs/*/instruction.md`) and internal design/session markdown documents are versioned in a separate documentation repository and staged into `task_specs/` prior to running `build-tasks`.

## Common Commands

```bash
# Verify rootless Podman socket, Compose v2 plugin, and compose stack teardown
pixi run podman-check
pixi run podman-smoke-compose
pixi run podman-clean

# Build all 4 container images + 61-tool offline isolation smoke check
pixi run build-images
pixi run build-verifier

# Generate Suite S synthetic MS datasets and Suite R 3C391 stage checkpoints
pixi run build-synth
pixi run build-checkpoints

# Export raw and CASA-native skill trees and compile all 52 Harbor task variants
pixi run export-skills
pixi run build-tasks

# Lint, format, unit tests, and Harbor task lint
pixi run check
pixi run test-unit
pixi run harbor-lint

# Run non-LLM Harbor oracle verification gates (0 LLM tokens)
pixi run oracle-synth
pixi run oracle-3c391

# Post-run 2x2 factorial contrasts, cluster bootstrap CIs, token pricing, and failure taxonomy
pixi run analyze <harbor_job_dir>
```

## Repository Layout

- `images/`: Containerfiles (`agent-base`, `mcp-sidecar`, `verifier`), `serve_mcp_sidecar.py`, and `build_images.sh`.
- `data_builders/`: Synthetic VLA visibility simulator (`build_synthetic_suite.py`) and 3C391 checkpoint builder (`build_3c391_checkpoints.py`).
- `task_specs/`: Per-task `spec.toml`, non-LLM reference `solve.py`, and isolated `verify.py`.
- `src/radio_bench/`:
  - `podman_env.py`, `smoke_containers.py`: Podman socket, compose teardown smoke checks, and leak cleanup.
  - `export_skills.py`, `build_tasks.py`: Skill tree exporter and Harbor `0.25.0` task/job compiler.
  - `grading/`: Physical verifiers (`inspect_metrics.py`, `precal_metrics.py`, `cal_metrics.py`, `image_metrics.py`).
  - `analysis/`: Post-run collector, pricing, two-stage cluster bootstrap 2x2 contrasts, failure taxonomy, diagnostics, and CLI.
- `jobs/`: Harbor `0.25.0` job configs (`oracle_synth.yaml`, `oracle_3c391.yaml`, and Suite S / Suite R ablation configs).
- `tests/`: Unit, isolation, simulator, container smoke, verifier, and factorial analysis test suites.

## License

GNU General Public License v3.0 -- see `LICENSE`.
