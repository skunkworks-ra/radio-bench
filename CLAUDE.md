# CLAUDE.md -- radio-bench

Engineering, Harbor `0.25.0`, Podman, and isolation contract for `radio-bench`. Full benchmark specification lives in `SPEC.md`.

## Execution & Orchestration Rules

1. **Lean orchestrator + `self` subagents:** Keep the main session lean as the orchestrator running gates and git commits. Delegate every wave's implementation, independent adversarial verification, and independent code review to `self` subagents (`invoke_subagent(TypeName="self")`). Pair subagent launches with parallel work or a bounded wakeup timer.
2. **CLI & rootless Podman:** Run `pixi run <task>` directly from `PATH` inside `/var/home/pjaganna/Software/radio-bench`. Podman (`5.8.8`) runs rootless as user `pjaganna` (`uid 1000`) via `/run/user/1000/podman/podman.sock` -- never use `sudo` or root Podman.
3. **Storage & disk budget (`/var/home` is 97% full -- 32 GB free; `/var/mnt/fast` has 1.3 TB free):**
   - Never use `/tmp` as a work area.
   - Keep `.pixi` and `tasks` symlinked to `/var/mnt/fast/radio-bench/{.pixi,tasks}`.
   - Execute all tests, `TMPDIR`, `MPLCONFIGDIR`, `CASASITECONFIG` (`/var/mnt/fast/radio-bench/casaconfig_local.py`), and `pytest --basetemp` inside `/var/mnt/fast/radio-bench/`.
   - Store all synthetic MSs and 3C391 checkpoints in `/var/mnt/fast/radio_bench_data/` and all Harbor job runs in `/var/mnt/fast/radio_bench_runs/`.
4. **Zero edits to `radio-analyst`:** `/var/home/pjaganna/Software/radio-analyst` is strictly read-only (`RADIO_ANALYST_SHA = "00a11ea"`).
5. **Vault markdown rule:** Every `.md` file in `radio-bench` except `CLAUDE.md` lives in `/var/home/pjaganna/Documents/Crypt/20-Projects/radio-bench/` and is symlinked into `/var/home/pjaganna/Software/radio-bench/`. Always read and edit `.md` files via their Vault paths. Do not use em dashes or en dashes (`--` only).

## Verified Harbor `0.25.0` & Container Invariants

1. **Shared `casa-base` layer:** `agent-base`, `mcp-sidecar`, and `verifier` all inherit `localhost/radio-bench-casa-base:latest` (`debian:trixie-slim` + CPython `3.12` in `/opt/uv-python` & `/opt/casa-venv` + `casatools==6.7.6.14` + `casatasks==6.7.6.14` + `/opt/casa-data` + `CASASITECONFIG=/etc/casa/config.py` with `data_auto_update = False` and `measures_auto_update = False`).
2. **Hard `agent-base` isolation:** `agent-base` contains `run-step.sh`, `claude`, and `opencode`, and zero `ms_inspect`, `ms_modify`, `ms_create`, `radio_bench`, `verify.py`, or `ground_truth.json`.
3. **Sidecar binding:** `mcp-sidecar` must set `RADIO_MCP_HOST=0.0.0.0` (`ms_inspect` defaults to `127.0.0.1`) and serve `ms-inspect` (`8000`, 38 tools), `ms-modify` (`8001`, 19 tools), and `ms-create` (`8002`, 4 tools) = `61` tools over `streamable-http`.
4. **`harbor lint` (`0.25.0`) rules:**
   - Pass explicit task directories (`harbor lint /var/mnt/fast/radio-bench/tasks/*/`), never the parent directory alone.
   - Stage `input.ms` into `tasks/<variant>/environment/data/input.ms` via same-filesystem hardlinks (`cp -al` on `/var/mnt/fast`, regular files, zero symlinks escaping `task_root`, zero extra disk blocks), mounted `:ro,z` at `/seed_data` in `environment/docker-compose.yaml` and copied into writable named volume `workspace_vol:/workspace` (`shared` with `mcp-servers` on tool arms) with `[environment.healthcheck]` `command = "test -f /workspace/.ready"`.
   - Use isolated verifier mode (`[verifier]` `environment_mode = "separate"`, `artifacts = ["/workspace"]`, and `tests/docker-compose.yaml` mounting `.:/tests:ro,z` on `localhost/radio-bench-verifier:latest`) so `shared-verifier` lint passes with zero extra image builds per task.
5. **`reward.json` vs `metrics.json`:**
   - Harbor `0.25.0` `Verifier._parse_reward_json` (`verifier/verifier.py:112-123`) requires every value in `/logs/verifier/reward.json` to be a flat finite `int | float` (`{"reward": <float>, "passed": <0_or_1>}`); nested dicts, strings, `None`, or `NaN` raise `VerifierOutputParseError`.
   - Write all rich physical measurements, ground-truth numbers, subscores, and boolean pass flags to `/logs/verifier/metrics.json` alongside `/logs/verifier/reward.json` (`collect.py` reads both).
6. **`task.toml` & `jobs/*.yaml` (`0.25.0`):**
   - `skills_dir` and `[[environment.mcp_servers]]` (`name`, `transport = "streamable-http"`, `url = "http://mcp-servers:<port>/mcp"`) live under `[environment]`.
   - For `11_orchestrated`, stage subagent `.md` files into both `/workspace/.claude/agents/*.md` and `/workspace/.opencode/agents/*.md` (`ClaudeCode` overrides `CLAUDE_CONFIG_DIR=/logs/agent/sessions`).
   - In `jobs/*.yaml`, use top-level `n_concurrent_trials` and `retry` (not deprecated `orchestrator:`). Both `claude-code` and `opencode` accept `kwargs: { disable_web_search: true }`; never pass `reasoning_effort` or `config` to `opencode` (use `variant` / `opencode_config`).
7. **CASA lock safety:** Always open and close `casatools` (`table`, `msmetadata`, `ms`, `image`, `simulator`) inside `try / finally` blocks so zero stale `table.lock` files survive.
