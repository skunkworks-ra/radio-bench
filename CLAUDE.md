# CLAUDE.md -- radio-bench

Engineering and isolation contract for `radio-bench`. Full benchmark specification lives in `SPEC.md`.

## Core invariants -- do not violate

1. **Zero edits to `radio-analyst`:** `/var/home/pjaganna/Software/radio-analyst` is strictly read-only. Clone or export at pinned commit `RADIO_ANALYST_SHA = "00a11ea"` only.
2. **Zero leakage into `agent-base`:** The `agent-base` container image must never contain `ms_inspect`, `ms_modify`, `ms_create`, `radio_bench`, `verify.py`, or `ground_truth.json`. `python -c "import ms_inspect"` must fail with `ModuleNotFoundError`.
3. **Sidecar MCP isolation:** In `10_tools`, `11_full`, and `11_orchestrated`, the 3 FastMCP servers (`ms-inspect` port 8000, `ms-modify` port 8001, `ms-create` port 8002) run in the separate `mcp-sidecar` container sharing only `/workspace`.
4. **Tamper-proof grading:** All scoring runs inside the isolated `verifier` container after the agent container stops, writing `/logs/verifier/reward.json` (`reward` in `[0.0, 1.0]`, `passed = reward >= 0.85`, plus raw measured physical quantities and ground truth).
5. **Storage and test hygiene:** Never use `/tmp` as a work area. Run all tests, CASA logs, pytest temp directories, `.pixi`, and compiled `tasks/` inside `/var/mnt/fast/radio-bench/`. Store datasets and checkpoints in `/var/mnt/fast/radio_bench_data/` and Harbor trial runs in `/var/mnt/fast/radio_bench_runs/`.
6. **CASA table lock safety:** Always open and close `casatools` (`table`, `msmetadata`, `ms`, `image`, `simulator`) cleanly inside `try / finally` context managers so zero stale `table.lock` files survive.
7. **Vault markdown rule:** Every `.md` file in this repository except `CLAUDE.md` lives in `/var/home/pjaganna/Documents/Crypt/20-Projects/radio-bench/` and is symlinked into `/var/home/pjaganna/Software/radio-bench/`. Always edit `.md` files via their Vault paths.
8. **Verification gates:** Run `pixi run check` and `pixi run test-unit` (executing in `/var/mnt/fast/radio-bench/`) before every local commit. Stage explicit paths only (`git add <path>`).
