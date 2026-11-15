"""Compile Harbor 0.25.0 task variants across capability arms for radio-bench.

Reads the 10 canonical task specs in ``task_specs/<task_id>/`` plus the exported
skill/agent trees in ``/var/mnt/fast/radio-bench/exported_skills/`` and emits up
to 52 Harbor task directories under ``/var/mnt/fast/radio-bench/tasks/``:
- 8 stage tasks x 5 arms (00_none, 01_skills_raw, 01_skills_casa, 10_tools, 11_full) = 40
- 2 end-to-end tasks x 6 arms (+ 11_orchestrated when --include-orchestrated) = 12
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
import yaml
from harbor.models.task.config import TaskConfig

from radio_bench.export_skills import DEFAULT_EXPORT_DIR as DEFAULT_SKILLS_DIR
from radio_bench.export_skills import RADIO_ANALYST_SHA

REPO_ROOT = Path("/var/home/pjaganna/Software/radio-bench")
DEFAULT_SPECS_DIR = REPO_ROOT / "task_specs"
DEFAULT_TASKS_DIR = Path("/var/mnt/fast/radio-bench/tasks")

ARM_NONE = "00_none"
ARM_SKILLS_RAW = "01_skills_raw"
ARM_SKILLS_CASA = "01_skills_casa"
ARM_TOOLS = "10_tools"
ARM_FULL = "11_full"
ARM_ORCHESTRATED = "11_orchestrated"

SKILL_ARMS = frozenset({ARM_SKILLS_RAW, ARM_SKILLS_CASA, ARM_FULL, ARM_ORCHESTRATED})
TOOL_ARMS = frozenset({ARM_TOOLS, ARM_FULL, ARM_ORCHESTRATED})

AGENT_IMAGE = "${RADIO_BENCH_AGENT_IMAGE:-localhost/radio-bench-agent-base:latest}"
SIDECAR_IMAGE = "${RADIO_BENCH_SIDECAR_IMAGE:-localhost/radio-bench-mcp-sidecar:latest}"
VERIFIER_IMAGE = "${RADIO_BENCH_VERIFIER_IMAGE:-localhost/radio-bench-verifier:latest}"
AGENT_NETWORK_MODE = "pasta"
VERIFIER_NETWORK_MODE = "none"

MCP_SERVERS: list[dict[str, str]] = [
    {
        "name": "ms-inspect",
        "transport": "streamable-http",
        "url": "http://mcp-servers:8000/mcp",
    },
    {
        "name": "ms-modify",
        "transport": "streamable-http",
        "url": "http://mcp-servers:8001/mcp",
    },
    {
        "name": "ms-create",
        "transport": "streamable-http",
        "url": "http://mcp-servers:8002/mcp",
    },
]

REQUIRED_SPEC_KEYS = frozenset(
    {
        "task_id",
        "suite",
        "stage",
        "is_e2e",
        "difficulty",
        "category",
        "tags",
        "data_dir",
        "input_ms",
        "ground_truth",
        "extra_test_files",
        "agent_timeout_sec",
        "verifier_timeout_sec",
        "cpus",
        "memory_mb",
    }
)


@dataclass(frozen=True)
class TaskSpec:
    """Parsed and validated task_specs/<task_id>/spec.toml."""

    spec_dir: Path
    task_id: str
    suite: str
    stage: str
    is_e2e: bool
    difficulty: str
    category: str
    tags: list[str]
    data_dir: Path
    input_ms: Path
    ground_truth: Path
    extra_test_files: list[str]
    agent_timeout_sec: float
    verifier_timeout_sec: float
    cpus: int
    memory_mb: int
    input_calibrators_ms: Path | None = None
    verifier_cpus: int | None = None

    @property
    def effective_verifier_cpus(self) -> int:
        return self.verifier_cpus if self.verifier_cpus is not None else self.cpus


def load_task_spec(spec_dir: Path) -> TaskSpec:
    """Load and validate one task spec directory."""
    spec_path = spec_dir / "spec.toml"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Missing spec.toml in {spec_dir}")

    raw = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_SPEC_KEYS - raw.keys())
    if missing:
        raise ValueError(f"{spec_path} missing required keys: {missing}")

    task_id = str(raw["task_id"])
    if task_id != spec_dir.name:
        raise ValueError(f"{spec_path} task_id={task_id!r} != directory name {spec_dir.name!r}")

    suite = str(raw["suite"])
    if suite not in {"synth", "3c391"}:
        raise ValueError(f"{spec_path} invalid suite={suite!r}")

    stage = str(raw["stage"])
    if stage not in {"inspect", "precal", "calibrate", "image", "e2e"}:
        raise ValueError(f"{spec_path} invalid stage={stage!r}")

    for required_file in ("instruction.md", "solve.py", "verify.py"):
        p = spec_dir / required_file
        if not p.is_file():
            raise FileNotFoundError(f"Missing {required_file} in {spec_dir}")

    data_dir = Path(str(raw["data_dir"]))
    input_ms = Path(str(raw["input_ms"]))
    raw_cal_ms = str(raw.get("input_calibrators_ms", "")).strip()
    input_calibrators_ms = Path(raw_cal_ms) if raw_cal_ms else None
    ground_truth = Path(str(raw["ground_truth"]))
    extra_test_files = [str(x) for x in raw["extra_test_files"]]

    if not input_ms.is_dir():
        raise FileNotFoundError(f"{task_id}: input_ms directory not found: {input_ms}")
    if input_calibrators_ms is not None and not input_calibrators_ms.is_dir():
        raise FileNotFoundError(
            f"{task_id}: input_calibrators_ms directory not found: {input_calibrators_ms}"
        )
    if not ground_truth.is_file():
        raise FileNotFoundError(f"{task_id}: ground_truth file not found: {ground_truth}")
    for extra in extra_test_files:
        extra_path = data_dir / extra
        if not extra_path.is_file():
            raise FileNotFoundError(f"{task_id}: extra test file not found: {extra_path}")

    return TaskSpec(
        spec_dir=spec_dir,
        task_id=task_id,
        suite=suite,
        stage=stage,
        is_e2e=bool(raw["is_e2e"]),
        difficulty=str(raw["difficulty"]),
        category=str(raw["category"]),
        tags=[str(t) for t in raw["tags"]],
        data_dir=data_dir,
        input_ms=input_ms,
        ground_truth=ground_truth,
        extra_test_files=extra_test_files,
        agent_timeout_sec=float(raw["agent_timeout_sec"]),
        verifier_timeout_sec=float(raw["verifier_timeout_sec"]),
        cpus=int(raw["cpus"]),
        memory_mb=int(raw["memory_mb"]),
        input_calibrators_ms=input_calibrators_ms,
        verifier_cpus=int(raw["verifier_cpus"]) if "verifier_cpus" in raw else None,
    )


def arms_for_spec(
    spec: TaskSpec,
    *,
    skills_mode: str,
    include_orchestrated: bool,
) -> list[str]:
    """Return ordered capability arms to emit for ``spec``."""
    if skills_mode not in {"raw", "casa", "both"}:
        raise ValueError(f"Invalid skills_mode={skills_mode!r}")

    arms: list[str] = [ARM_NONE]
    if skills_mode in {"raw", "both"}:
        arms.append(ARM_SKILLS_RAW)
    if skills_mode in {"casa", "both"}:
        arms.append(ARM_SKILLS_CASA)
    arms.extend([ARM_TOOLS, ARM_FULL])
    if spec.is_e2e and include_orchestrated:
        arms.append(ARM_ORCHESTRATED)
    return arms


def _hardlink_file(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst`` without touching source inode permissions."""
    if src.is_symlink():
        src = src.resolve(strict=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst, follow_symlinks=True)
        os.chmod(dst, 0o644)


def _hardlink_tree(src_dir: Path, dst_dir: Path) -> None:
    """Recursively hardlink a directory tree with 0o755 dirs and zero symlinks."""
    resolved_root = src_dir.resolve(strict=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(dst_dir, 0o755)

    for root_str, dirnames, filenames in os.walk(resolved_root, followlinks=True):
        cur_src = Path(root_str)
        rel = cur_src.relative_to(resolved_root)
        cur_dst = dst_dir / rel if rel != Path(".") else dst_dir
        cur_dst.mkdir(parents=True, exist_ok=True)
        os.chmod(cur_dst, 0o755)
        dirnames.sort()
        for fname in sorted(filenames):
            _hardlink_file(cur_src / fname, cur_dst / fname)


def _copy_regular_tree(src_dir: Path, dst_dir: Path) -> None:
    """Copy a directory tree as regular 0o644 files and 0o755 dirs (no symlinks)."""
    resolved_root = src_dir.resolve(strict=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(dst_dir, 0o755)

    for root_str, dirnames, filenames in os.walk(resolved_root, followlinks=True):
        cur_src = Path(root_str)
        rel = cur_src.relative_to(resolved_root)
        cur_dst = dst_dir / rel if rel != Path(".") else dst_dir
        cur_dst.mkdir(parents=True, exist_ok=True)
        os.chmod(cur_dst, 0o755)
        dirnames.sort()
        for fname in sorted(filenames):
            src_file = (cur_src / fname).resolve(strict=True)
            dst_file = cur_dst / fname
            shutil.copyfile(src_file, dst_file)
            os.chmod(dst_file, 0o644)


def _copy_regular_file(src_file: Path, dst_file: Path, *, mode: int = 0o644) -> None:
    """Copy one file resolving symlinks and setting explicit file mode."""
    resolved = src_file.resolve(strict=True)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(dst_file.parent, 0o755)
    shutil.copyfile(resolved, dst_file)
    os.chmod(dst_file, mode)


def _write_text_file(dst_file: Path, content: str, *, mode: int = 0o644) -> None:
    """Write UTF-8 text file with explicit permissions."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(dst_file.parent, 0o755)
    dst_file.write_text(content, encoding="utf-8")
    os.chmod(dst_file, mode)


def _build_runtime_env(omp_threads: str) -> dict[str, str]:
    """Build TMPDIR + OpenMP/BLAS anti-thrashing environment dictionary."""
    return {
        "TMPDIR": "/var/tmp",
        "OMP_NUM_THREADS": omp_threads,
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _build_task_toml_dict(spec: TaskSpec, arm: str) -> dict[str, Any]:
    """Build Harbor 0.25.0 task.toml dictionary for ``(spec, arm)``."""
    variant_id = f"{spec.task_id}__{arm}"
    omp_threads = str(spec.cpus)
    ver_cpus = spec.effective_verifier_cpus
    ver_omp_threads = str(ver_cpus)

    env_cfg: dict[str, Any] = {
        "docker_image": AGENT_IMAGE,
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "workdir": "/workspace",
        "env": _build_runtime_env(omp_threads),
        "healthcheck": {
            "command": "test -f /workspace/.ready",
            "interval_sec": 2.0,
            "timeout_sec": 10.0,
            "start_period_sec": 5.0,
            "start_interval_sec": 1.0,
            "retries": 90,
        },
    }
    if arm in SKILL_ARMS:
        env_cfg["skills_dir"] = "/skills"
    if arm in TOOL_ARMS:
        env_cfg["mcp_servers"] = [dict(srv) for srv in MCP_SERVERS]

    data: dict[str, Any] = {
        "schema_version": "1.4",
        "artifacts": ["/workspace"],
        "metadata": {
            "task_id": variant_id,
            "base_task": spec.task_id,
            "suite": spec.suite,
            "stage": spec.stage,
            "arm": arm,
            "difficulty": spec.difficulty,
            "category": spec.category,
            "tags": list(spec.tags),
            "radio_analyst_sha": RADIO_ANALYST_SHA,
        },
        "agent": {
            "user": "analyst",
            "timeout_sec": spec.agent_timeout_sec,
        },
        "environment": env_cfg,
        "verifier": {
            "environment_mode": "separate",
            "timeout_sec": spec.verifier_timeout_sec,
            "user": "root",
            "env": _build_runtime_env(ver_omp_threads),
            "environment": {
                "docker_image": VERIFIER_IMAGE,
                "cpus": ver_cpus,
                "memory_mb": spec.memory_mb,
                "workdir": "/workspace",
                "env": _build_runtime_env(ver_omp_threads),
            },
        },
    }
    TaskConfig.model_validate(data)
    return data


def _build_environment_compose(
    arm: str,
    *,
    cpus: int,
    has_seed_workspace: bool,
    has_calibrators_ms: bool,
) -> dict[str, Any]:
    """Build environment/docker-compose.yaml for ``arm``."""
    has_skills = arm in SKILL_ARMS
    has_tools = arm in TOOL_ARMS
    runtime_env = _build_runtime_env(str(cpus))

    main_volumes = ["./data:/seed_data:ro,z"]
    if has_skills:
        main_volumes.append("./skills:/skills:ro,z")
    if has_seed_workspace:
        main_volumes.append("./seed_workspace:/seed_workspace:ro,z")
    main_volumes.extend(["workspace_vol:/workspace", "agent_tmp:/var/tmp"])

    startup_lines = [
        "set -euo pipefail",
        "mkdir -p /var/tmp/mplconfig",
        "chmod 1777 /var/tmp /var/tmp/mplconfig",
        "rm -rf /workspace/input.ms /workspace/calibrators.ms /workspace/.ready /workspace/.ws_owned",
        "cp -a --no-preserve=ownership /seed_data/input.ms /workspace/input.ms",
    ]
    if has_calibrators_ms:
        startup_lines.append(
            "cp -a --no-preserve=ownership /seed_data/calibrators.ms /workspace/calibrators.ms"
        )
    if has_seed_workspace:
        startup_lines.append("cp -a --no-preserve=ownership /seed_workspace/. /workspace/")
    if arm == ARM_ORCHESTRATED:
        startup_lines.extend(
            [
                "mkdir -p /home/analyst/.claude /home/analyst/.config/opencode",
                "cp -a --no-preserve=ownership /seed_workspace/.claude/. /home/analyst/.claude/",
                "cp -a --no-preserve=ownership /seed_workspace/.opencode/. /home/analyst/.config/opencode/",
                "chmod -R u+rwX,go+rX /home/analyst/.claude /home/analyst/.config",
                "chown -R 1000:1000 /home/analyst/.claude /home/analyst/.config",
            ]
        )
    startup_lines.extend(
        [
            "chmod -R u+rwX,go+rX /workspace",
            "chown -R 1000:1000 /workspace",
        ]
    )
    if has_tools:
        tcp_probe = (
            "/opt/casa-venv/bin/python -c '"
            "import socket, sys, time\n"
            "for port in (8000, 8001, 8002):\n"
            "    for _ in range(150):\n"
            "        try:\n"
            '            with socket.create_connection(("mcp-servers", port), timeout=1.0):\n'
            "                break\n"
            "        except OSError:\n"
            "            time.sleep(0.2)\n"
            "    else:\n"
            '        sys.exit(f"Timed out waiting for mcp-servers:{port}")\n'
            "'"
        )
        startup_lines.extend(
            [
                "touch /workspace/.ws_owned",
                "chown 1000:1000 /workspace/.ws_owned",
                tcp_probe,
            ]
        )

    startup_lines.extend(
        [
            "touch /workspace/.ready",
            "chown 1000:1000 /workspace/.ready",
            "exec sleep infinity",
        ]
    )

    main_service: dict[str, Any] = {
        "image": AGENT_IMAGE,
        "network_mode": AGENT_NETWORK_MODE,
        "environment": dict(runtime_env),
        "command": ["bash", "-lc", "\n".join(startup_lines)],
        "volumes": main_volumes,
    }
    if has_tools:
        main_service["extra_hosts"] = ["mcp-servers:127.0.0.1"]

    services: dict[str, Any] = {"main": main_service}

    if has_tools:
        sidecar_cmd = "\n".join(
            [
                "set -euo pipefail",
                "until [ -w /workspace ] && [ -f /workspace/.ws_owned ]; do",
                "  sleep 0.2",
                "done",
                "exec /opt/casa-venv/bin/python /usr/local/bin/serve-mcp-sidecar.py",
            ]
        )
        sidecar_env = dict(runtime_env)
        sidecar_env["RADIO_MCP_HOST"] = "127.0.0.1"
        services["mcp-servers"] = {
            "image": SIDECAR_IMAGE,
            "network_mode": "service:main",
            "environment": sidecar_env,
            "command": ["bash", "-lc", sidecar_cmd],
            "volumes": ["workspace_vol:/workspace", "agent_tmp:/var/tmp"],
        }

    return {
        "services": services,
        "volumes": {
            "workspace_vol": {},
            "agent_tmp": {},
        },
    }


def _build_verifier_compose(*, verifier_cpus: int) -> dict[str, Any]:
    """Build isolated tests/docker-compose.yaml."""
    ver_env = _build_runtime_env(str(verifier_cpus))
    return {
        "services": {
            "main": {
                "image": VERIFIER_IMAGE,
                "network_mode": VERIFIER_NETWORK_MODE,
                "environment": dict(ver_env),
                "command": [
                    "bash",
                    "-lc",
                    "mkdir -p /var/tmp/mplconfig && chmod 1777 /var/tmp /var/tmp/mplconfig && exec sleep infinity",
                ],
                "volumes": [
                    ".:/tests:ro,z",
                    "verifier_ws:/workspace",
                    "verifier_tmp:/var/tmp",
                ],
            }
        },
        "volumes": {
            "verifier_ws": {},
            "verifier_tmp": {},
        },
    }


def _stage_skills_and_workspace(
    env_dir: Path,
    *,
    arm: str,
    exported_skills_dir: Path,
) -> bool:
    """Stage arm-appropriate skills and seed_workspace files; return has_seed_workspace."""
    if arm in {ARM_NONE, ARM_TOOLS}:
        return False

    if arm == ARM_SKILLS_CASA:
        src_skills = exported_skills_dir / "casa" / "skills"
        src_guide_dir = exported_skills_dir / "casa"
    elif arm == ARM_SKILLS_RAW:
        src_skills = exported_skills_dir / "raw" / "skills"
        src_guide_dir = exported_skills_dir / "raw"
    elif arm == ARM_FULL:
        src_skills = exported_skills_dir / "raw" / "skills"
        src_guide_dir = exported_skills_dir / "full"
    elif arm == ARM_ORCHESTRATED:
        src_skills = exported_skills_dir / "orchestrated_skills"
        src_guide_dir = exported_skills_dir / "orchestrated"
    else:
        raise ValueError(f"Unsupported arm: {arm}")

    _copy_regular_tree(src_skills, env_dir / "skills")

    seed_ws = env_dir / "seed_workspace"
    seed_ws.mkdir(parents=True, exist_ok=True)
    os.chmod(seed_ws, 0o755)
    _copy_regular_file(src_guide_dir / "CLAUDE.md", seed_ws / "CLAUDE.md")
    _copy_regular_file(src_guide_dir / "AGENTS.md", seed_ws / "AGENTS.md")

    if arm == ARM_ORCHESTRATED:
        _copy_regular_tree(
            exported_skills_dir / "agents" / "claude",
            seed_ws / ".claude" / "agents",
        )
        _copy_regular_tree(
            exported_skills_dir / "agents" / "commands",
            seed_ws / ".claude" / "commands",
        )
        _copy_regular_tree(
            exported_skills_dir / "agents" / "opencode",
            seed_ws / ".opencode" / "agents",
        )

    return True


def _validate_exported_skills(exported_skills_dir: Path) -> None:
    """Verify required exported_skills subdirectories exist before compiling."""
    required_paths = [
        exported_skills_dir / "raw" / "skills",
        exported_skills_dir / "raw" / "CLAUDE.md",
        exported_skills_dir / "raw" / "AGENTS.md",
        exported_skills_dir / "orchestrated_skills",
        exported_skills_dir / "casa" / "skills",
        exported_skills_dir / "casa" / "CLAUDE.md",
        exported_skills_dir / "casa" / "AGENTS.md",
        exported_skills_dir / "full" / "CLAUDE.md",
        exported_skills_dir / "full" / "AGENTS.md",
        exported_skills_dir / "orchestrated" / "CLAUDE.md",
        exported_skills_dir / "orchestrated" / "AGENTS.md",
        exported_skills_dir / "agents" / "claude",
        exported_skills_dir / "agents" / "commands",
        exported_skills_dir / "agents" / "opencode",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing exported skill files (run `python -m radio_bench.export_skills` first): "
            + ", ".join(missing)
        )


def compile_variant(
    spec: TaskSpec,
    arm: str,
    *,
    output_dir: Path,
    exported_skills_dir: Path,
) -> Path:
    """Compile a single Harbor task directory ``<output_dir>/<task_id>__<arm>``."""
    variant_name = f"{spec.task_id}__{arm}"
    variant_dir = output_dir / variant_name
    if variant_dir.exists():
        shutil.rmtree(variant_dir)

    variant_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(variant_dir, 0o755)

    # 1. instruction.md (verbatim regular file; identical across arms of spec.task_id)
    _copy_regular_file(spec.spec_dir / "instruction.md", variant_dir / "instruction.md")

    # 2. task.toml
    toml_dict = _build_task_toml_dict(spec, arm)
    _write_text_file(variant_dir / "task.toml", tomli_w.dumps(toml_dict))

    # 3. environment/
    env_dir = variant_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(env_dir, 0o755)

    _hardlink_tree(spec.input_ms, env_dir / "data" / "input.ms")
    if spec.input_calibrators_ms is not None:
        _hardlink_tree(spec.input_calibrators_ms, env_dir / "data" / "calibrators.ms")

    has_seed_ws = _stage_skills_and_workspace(
        env_dir,
        arm=arm,
        exported_skills_dir=exported_skills_dir,
    )
    env_compose = _build_environment_compose(
        arm,
        cpus=spec.cpus,
        has_seed_workspace=has_seed_ws,
        has_calibrators_ms=spec.input_calibrators_ms is not None,
    )
    _write_text_file(
        env_dir / "docker-compose.yaml",
        yaml.safe_dump(env_compose, sort_keys=False),
    )

    # 4. solution/ (oracle only; zero ground truth or verifier files)
    sol_dir = variant_dir / "solution"
    sol_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(sol_dir, 0o755)
    _copy_regular_file(spec.spec_dir / "solve.py", sol_dir / "solve.py")
    solve_sh = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{spec.cpus}}}"\n'
        'export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"\n'
        'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"\n'
        'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"\n'
        'export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"\n'
        "cd /workspace\n"
        "/opt/casa-venv/bin/python /solution/solve.py\n"
    )
    _write_text_file(sol_dir / "solve.sh", solve_sh, mode=0o755)

    # 5. tests/ (isolated verifier bundle)
    tests_dir = variant_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(tests_dir, 0o755)
    _copy_regular_file(spec.spec_dir / "verify.py", tests_dir / "verify.py")
    _hardlink_file(spec.ground_truth, tests_dir / "ground_truth.json")
    for extra_name in spec.extra_test_files:
        _hardlink_file(spec.data_dir / extra_name, tests_dir / extra_name)

    test_sh = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{spec.effective_verifier_cpus}}}"\n'
        'export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"\n'
        'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"\n'
        'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"\n'
        'export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"\n'
        "/opt/casa-venv/bin/python /tests/verify.py "
        "--workspace /workspace --tests-dir /tests --log-dir /logs/verifier\n"
    )
    _write_text_file(tests_dir / "test.sh", test_sh, mode=0o755)
    verifier_compose = _build_verifier_compose(verifier_cpus=spec.effective_verifier_cpus)
    _write_text_file(
        tests_dir / "docker-compose.yaml",
        yaml.safe_dump(verifier_compose, sort_keys=False),
    )

    return variant_dir


def build_all_tasks(
    *,
    specs_dir: Path = DEFAULT_SPECS_DIR,
    skills_dir: Path = DEFAULT_SKILLS_DIR,
    output_dir: Path = DEFAULT_TASKS_DIR,
    skills_mode: str = "both",
    include_orchestrated: bool = True,
) -> dict[str, Any]:
    """Compile all configured Harbor task directories and prune unselected variants."""
    _validate_exported_skills(skills_dir)

    spec_dirs = sorted(p for p in specs_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not spec_dirs:
        raise FileNotFoundError(f"No task spec directories found under {specs_dir}")

    specs = [load_task_spec(d) for d in spec_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o755)

    known_prefixes = tuple(f"{s.task_id}__" for s in specs)
    emitted: list[str] = []
    by_suite: dict[str, int] = {"synth": 0, "3c391": 0}
    by_arm: dict[str, int] = {}

    for spec in specs:
        for arm in arms_for_spec(
            spec,
            skills_mode=skills_mode,
            include_orchestrated=include_orchestrated,
        ):
            variant_dir = compile_variant(
                spec,
                arm,
                output_dir=output_dir,
                exported_skills_dir=skills_dir,
            )
            emitted.append(variant_dir.name)
            by_suite[spec.suite] = by_suite.get(spec.suite, 0) + 1
            by_arm[arm] = by_arm.get(arm, 0) + 1

    emitted_set = set(emitted)
    for child in sorted(output_dir.iterdir()):
        if (
            child.is_dir()
            and child.name.startswith(known_prefixes)
            and child.name not in emitted_set
        ):
            shutil.rmtree(child)

    return {
        "output_dir": str(output_dir),
        "skills_mode": skills_mode,
        "include_orchestrated": include_orchestrated,
        "base_tasks": len(specs),
        "compiled_tasks": len(emitted),
        "by_suite": by_suite,
        "by_arm": by_arm,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile Harbor 0.25.0 task directories for radio-bench."
    )
    parser.add_argument(
        "--specs-dir",
        type=Path,
        default=DEFAULT_SPECS_DIR,
        help="Path to task_specs/ directory.",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help="Path to exported_skills/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="Destination directory for compiled Harbor tasks.",
    )
    parser.add_argument(
        "--skills-mode",
        choices=("raw", "casa", "both"),
        default="both",
        help="Which skill tree arms to emit (raw, casa, or both).",
    )
    parser.add_argument(
        "--include-orchestrated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit 11_orchestrated on end-to-end tasks (default: True).",
    )
    args = parser.parse_args(argv)

    summary = build_all_tasks(
        specs_dir=args.specs_dir,
        skills_dir=args.skills_dir,
        output_dir=args.output_dir,
        skills_mode=args.skills_mode,
        include_orchestrated=args.include_orchestrated,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
