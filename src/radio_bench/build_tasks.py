"""Compile Harbor 0.25.0 task variants across capability arms and tool/skill packs.

Loads declarative suite manifests in ``suites/<suite_id>/suite.toml`` plus exported skill/agent
trees in ``<work_dir>/exported_skills/`` and emits Harbor task directories under
``<work_dir>/tasks/``:
- Single pack (default): ``<task_id>__<arm>``
- Multi-pack (``--packs id1,id2`` or ``--force-pack-in-name``): ``<task_id>__<pack_id>__<arm>``
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import logging
import os
import re
import shutil
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
import yaml
from harbor.models.task.config import TaskConfig

from radio_bench.config import (
    DEFAULT_MCP_PORTS,
    BenchConfig,
    PackConfig,
    add_config_cli_args,
    config_from_cli_args,
    load_bench_config,
    materialize_pack_source,
    reject_slash_tmp,
    resolve_repo_root,
)
from radio_bench.grading.schema import (
    RubricSpec,
    SuiteManifest,
    loads_strict_json,
)

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    if name in {
        "REPO_ROOT",
        "DEFAULT_SPECS_DIR",
        "DEFAULT_SUITES_DIR",
        "DEFAULT_TASKS_DIR",
        "RADIO_ANALYST_SHA",
    }:
        cfg = load_bench_config()
        if name == "REPO_ROOT":
            return cfg.repo_root
        if name == "DEFAULT_SPECS_DIR":
            return cfg.repo_root / "task_specs"
        if name == "DEFAULT_SUITES_DIR":
            return cfg.repo_root / "suites"
        if name == "DEFAULT_TASKS_DIR":
            return cfg.tasks_dir
        if name == "RADIO_ANALYST_SHA":
            return cfg.get_pack().analyst_ref
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


ARM_NONE = "00_none"
ARM_SKILLS_RAW = "01_skills_raw"
ARM_SKILLS_CASA = "01_skills_casa"
ARM_TOOLS = "10_tools"
ARM_FULL = "11_full"
ARM_ORCHESTRATED = "11_orchestrated"

SKILL_ARMS = frozenset({ARM_SKILLS_RAW, ARM_SKILLS_CASA, ARM_FULL, ARM_ORCHESTRATED})
TOOL_ARMS = frozenset({ARM_TOOLS, ARM_FULL, ARM_ORCHESTRATED})

AGENT_IMAGE = "${RADIO_BENCH_AGENT_IMAGE:-localhost/radio-bench-agent-base:latest}"
SIDECAR_IMAGE = "${RADIO_BENCH_SIDECAR_IMAGE:-localhost/radio-bench-mcp-sidecar:00a11ea}"
VERIFIER_IMAGE = "${RADIO_BENCH_VERIFIER_IMAGE:-localhost/radio-bench-verifier:latest}"
AGENT_NETWORK_MODE = "pasta"
VERIFIER_NETWORK_MODE = "none"

_HEX_SHORT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_WARNED_CROSS_DEVICE_PAIRS: set[tuple[int, int]] = set()

MCP_SERVERS: list[dict[str, str]] = [
    {
        "name": "ms-inspect",
        "transport": "streamable-http",
        "url": "http://127.0.0.1:8000/mcp",
    },
    {
        "name": "ms-modify",
        "transport": "streamable-http",
        "url": "http://127.0.0.1:8001/mcp",
    },
    {
        "name": "ms-create",
        "transport": "streamable-http",
        "url": "http://127.0.0.1:8002/mcp",
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
    """Resolved task definition ready for Harbor compilation."""

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
    instruction_path: Path | None = None
    solve_path: Path | None = None
    rubric_path: Path | None = None
    extract_config_json: str | None = None
    ref_flags_file: str | None = None
    true_target_vis_file: str | None = None
    suite_id: str | None = None
    split: str = "dev"
    stages: tuple[str, ...] = ()

    @property
    def effective_verifier_cpus(self) -> int:
        return self.verifier_cpus if self.verifier_cpus is not None else self.cpus

    @property
    def effective_instruction_path(self) -> Path:
        return (
            self.instruction_path
            if self.instruction_path is not None
            else (self.spec_dir / "instruction.md")
        )

    @property
    def effective_solve_path(self) -> Path:
        return self.solve_path if self.solve_path is not None else (self.spec_dir / "solve.py")


def _resolve_spec_data_path(raw_val: str, data_root: Path) -> Path:
    p = Path(raw_val)
    return p if p.is_absolute() else (data_root / p)


def _resolve_suite_asset_path(raw_rel: str, suite_dir: Path, repo_roots: list[Path]) -> Path:
    cand = Path(raw_rel)
    if cand.is_absolute():
        return cand
    if (suite_dir / cand).exists():
        return suite_dir / cand
    for root in repo_roots:
        if (root / cand).exists():
            return root / cand
    return suite_dir / cand


def load_suite_tasks(
    *,
    suites_dir: Path | None = None,
    data_root: Path | None = None,
    prompts_dir: Path | None = None,
    only_task_ids: set[str] | None = None,
    spec_dir_overrides: dict[str, Path] | None = None,
    cfg: BenchConfig | None = None,
) -> list[TaskSpec]:
    """Load and validate all tasks across `suites/*/suite.toml`."""
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    eff_data_root = data_root if data_root is not None else resolved_cfg.data_dir
    eff_prompts_dir = prompts_dir if prompts_dir is not None else resolved_cfg.prompts_dir

    fallback_repo_root = resolve_repo_root()
    repo_roots = [resolved_cfg.repo_root]
    if fallback_repo_root.resolve() != resolved_cfg.repo_root.resolve():
        repo_roots.append(fallback_repo_root)

    if suites_dir is not None:
        eff_suites_dir = suites_dir
    elif (resolved_cfg.repo_root / "suites").is_dir():
        eff_suites_dir = resolved_cfg.repo_root / "suites"
    else:
        eff_suites_dir = fallback_repo_root / "suites"

    if not eff_suites_dir.is_dir():
        raise FileNotFoundError(f"Suites directory not found: {eff_suites_dir}")

    suite_tomls = sorted(eff_suites_dir.glob("*/suite.toml"))
    if not suite_tomls:
        raise FileNotFoundError(f"No suite.toml files found under {eff_suites_dir}")

    default_prompts = (resolved_cfg.repo_root / "task_specs").resolve()
    prompts_overridden = prompts_dir is not None or eff_prompts_dir.resolve() != default_prompts

    specs: list[TaskSpec] = []
    for toml_path in suite_tomls:
        suite_dir = toml_path.parent
        manifest = SuiteManifest.model_validate(
            tomllib.loads(toml_path.read_text(encoding="utf-8"))
        )
        suite_id = manifest.suite.suite_id
        suite_alias = manifest.suite.legacy_suite_alias or suite_id

        for task_id, tspec in manifest.tasks.items():
            if only_task_ids is not None and task_id not in only_task_ids:
                continue

            override_sdir = (spec_dir_overrides or {}).get(task_id)
            spec_dir = (
                override_sdir
                if override_sdir is not None
                else (resolved_cfg.repo_root / "task_specs" / task_id)
            )

            # Resolve instruction.md honoring prompts_dir override
            prompt_cand = eff_prompts_dir / task_id / "instruction.md"
            manifest_inst = _resolve_suite_asset_path(tspec.instruction, suite_dir, repo_roots)
            override_inst = (override_sdir / "instruction.md") if override_sdir else None

            inst_candidates: list[Path] = []
            if prompts_overridden:
                inst_candidates.append(prompt_cand)
            if override_inst is not None:
                inst_candidates.append(override_inst)
            inst_candidates.extend([manifest_inst, prompt_cand])

            instruction_path = next((c for c in inst_candidates if c.is_file()), None)
            if instruction_path is None:
                raise FileNotFoundError(
                    f"{task_id}: Missing instruction.md (checked {manifest_inst} and {prompt_cand})"
                )

            solve_candidates: list[Path] = []
            if override_sdir is not None:
                solve_candidates.append(override_sdir / "solve.py")
            solve_candidates.append(
                _resolve_suite_asset_path(tspec.oracle_script, suite_dir, repo_roots)
            )
            solve_path = next((c for c in solve_candidates if c.is_file()), None)
            if solve_path is None:
                raise FileNotFoundError(
                    f"{task_id}: Missing oracle script {tspec.oracle_script} in {suite_dir}"
                )

            rubric_path = _resolve_suite_asset_path(tspec.rubric, suite_dir, repo_roots)
            if not rubric_path.is_file():
                raise FileNotFoundError(f"{task_id}: Missing rubric file {rubric_path}")
            parsed_rubric = RubricSpec.model_validate(
                loads_strict_json(rubric_path.read_text(encoding="utf-8"))
            )
            if parsed_rubric.task_id != task_id:
                raise ValueError(
                    f"{rubric_path}: rubric task_id={parsed_rubric.task_id!r} != {task_id!r}"
                )

            extract_cfg = manifest.to_extract_config(task_id)
            extract_cfg_json = (
                json.dumps(
                    extract_cfg.model_dump(mode="json"),
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )

            # Resolve checkpoint / MS / ground_truth paths
            ckpt_entry = manifest.checkpoints.entries.get(tspec.checkpoint_in)
            raw_in_ms = tspec.input_ms or (ckpt_entry.input_ms if ckpt_entry else None)
            if not raw_in_ms:
                raise ValueError(
                    f"{task_id}: Missing input_ms or checkpoint {tspec.checkpoint_in!r}"
                )
            raw_cal_ms = tspec.input_calibrators_ms or (
                ckpt_entry.input_calibrators_ms if ckpt_entry else None
            )

            t_data_dir = _resolve_spec_data_path(tspec.data_dir, eff_data_root)
            in_ms_path = _resolve_spec_data_path(raw_in_ms, eff_data_root)
            in_cal_ms_path = (
                _resolve_spec_data_path(raw_cal_ms, eff_data_root) if raw_cal_ms else None
            )
            gt_path = _resolve_spec_data_path(tspec.ground_truth, eff_data_root)

            if not in_ms_path.is_dir():
                raise FileNotFoundError(f"{task_id}: input_ms directory not found: {in_ms_path}")
            if in_cal_ms_path is not None and not in_cal_ms_path.is_dir():
                raise FileNotFoundError(
                    f"{task_id}: input_calibrators_ms directory not found: {in_cal_ms_path}"
                )
            if not gt_path.is_file():
                raise FileNotFoundError(f"{task_id}: ground_truth file not found: {gt_path}")
            for extra in tspec.extra_test_files:
                extra_path = t_data_dir / extra
                if not extra_path.is_file():
                    raise FileNotFoundError(f"{task_id}: extra test file not found: {extra_path}")

            eff_cpus = tspec.cpus if tspec.cpus is not None else manifest.resources.cpus
            eff_ver_cpus = (
                tspec.verifier_cpus
                if tspec.verifier_cpus is not None
                else manifest.resources.verifier_cpus
            )
            eff_mem = (
                tspec.memory_mb if tspec.memory_mb is not None else manifest.resources.memory_mb
            )

            specs.append(
                TaskSpec(
                    spec_dir=spec_dir,
                    task_id=task_id,
                    suite=suite_alias,
                    stage=tspec.canonical_stage,
                    is_e2e=tspec.is_e2e,
                    difficulty=tspec.difficulty,
                    category=manifest.suite.category,
                    tags=list(manifest.suite.tags),
                    data_dir=t_data_dir,
                    input_ms=in_ms_path,
                    ground_truth=gt_path,
                    extra_test_files=list(tspec.extra_test_files),
                    agent_timeout_sec=float(tspec.agent_timeout_sec),
                    verifier_timeout_sec=float(tspec.verifier_timeout_sec),
                    cpus=int(eff_cpus),
                    memory_mb=int(eff_mem),
                    input_calibrators_ms=in_cal_ms_path,
                    verifier_cpus=int(eff_ver_cpus) if eff_ver_cpus is not None else None,
                    instruction_path=instruction_path,
                    solve_path=solve_path,
                    rubric_path=rubric_path,
                    extract_config_json=extract_cfg_json,
                    ref_flags_file=tspec.ref_flags_file,
                    true_target_vis_file=tspec.true_target_vis_file,
                    suite_id=suite_id,
                    split=manifest.suite.split,
                    stages=tuple(tspec.stages),
                )
            )

    specs.sort(key=lambda s: s.task_id)
    return specs


def load_task_spec(
    spec_dir: Path,
    *,
    data_root: Path | None = None,
    prompts_dir: Path | None = None,
    cfg: BenchConfig | None = None,
) -> TaskSpec:
    """Load one task spec by `task_id` from `suites/*/suite.toml` (with `spec.toml` fallback)."""
    resolved_cfg = cfg if cfg is not None else load_bench_config()
    task_id = spec_dir.name
    spec_path = spec_dir / "spec.toml"

    # Prefer explicit `spec.toml` when present (e.g. synthetic unit-test task fixtures),
    # otherwise load from `suites/*/suite.toml`.
    if not spec_path.is_file():
        suite_matches: list[TaskSpec] = []
        with contextlib.suppress(FileNotFoundError):
            suite_matches = load_suite_tasks(
                data_root=data_root,
                prompts_dir=prompts_dir,
                only_task_ids={task_id},
                spec_dir_overrides={task_id: spec_dir},
                cfg=resolved_cfg,
            )
        if suite_matches:
            return suite_matches[0]
        raise FileNotFoundError(
            f"Task {task_id!r} not found in suites/*/suite.toml and no spec.toml in {spec_dir}"
        )

    eff_data_root = data_root if data_root is not None else resolved_cfg.data_dir
    eff_prompts_dir = prompts_dir if prompts_dir is not None else resolved_cfg.prompts_dir

    raw = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_SPEC_KEYS - raw.keys())
    if missing:
        raise ValueError(f"{spec_path} missing required keys: {missing}")

    if str(raw["task_id"]) != task_id:
        raise ValueError(f"{spec_path} task_id={raw['task_id']!r} != directory name {task_id!r}")

    prompt_cand = eff_prompts_dir / task_id / "instruction.md"
    spec_inst = spec_dir / "instruction.md"
    default_prompts = (resolved_cfg.repo_root / "task_specs").resolve()
    prompts_overridden = prompts_dir is not None or eff_prompts_dir.resolve() != default_prompts
    candidates = (prompt_cand, spec_inst) if prompts_overridden else (spec_inst, prompt_cand)
    instruction_path = next((c for c in candidates if c.is_file()), None)
    if instruction_path is None:
        raise FileNotFoundError(f"Missing instruction.md in {spec_dir} (and {prompt_cand})")

    solve_path = spec_dir / "solve.py"
    if not solve_path.is_file():
        raise FileNotFoundError(f"Missing solve.py in {spec_dir}")

    t_data_dir = _resolve_spec_data_path(str(raw["data_dir"]), eff_data_root)
    input_ms = _resolve_spec_data_path(str(raw["input_ms"]), eff_data_root)
    raw_cal_ms = str(raw.get("input_calibrators_ms", "")).strip()
    input_calibrators_ms = (
        _resolve_spec_data_path(raw_cal_ms, eff_data_root) if raw_cal_ms else None
    )
    ground_truth = _resolve_spec_data_path(str(raw["ground_truth"]), eff_data_root)
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
        extra_path = t_data_dir / extra
        if not extra_path.is_file():
            raise FileNotFoundError(f"{task_id}: extra test file not found: {extra_path}")

    return TaskSpec(
        spec_dir=spec_dir,
        task_id=task_id,
        suite=str(raw["suite"]),
        stage=str(raw["stage"]),
        is_e2e=bool(raw["is_e2e"]),
        difficulty=str(raw["difficulty"]),
        category=str(raw["category"]),
        tags=[str(t) for t in raw["tags"]],
        data_dir=t_data_dir,
        input_ms=input_ms,
        ground_truth=ground_truth,
        extra_test_files=extra_test_files,
        agent_timeout_sec=float(raw["agent_timeout_sec"]),
        verifier_timeout_sec=float(raw["verifier_timeout_sec"]),
        cpus=int(raw["cpus"]),
        memory_mb=int(raw["memory_mb"]),
        input_calibrators_ms=input_calibrators_ms,
        verifier_cpus=int(raw["verifier_cpus"]) if "verifier_cpus" in raw else None,
        instruction_path=instruction_path,
        solve_path=solve_path,
        rubric_path=(spec_dir / "rubric.json") if (spec_dir / "rubric.json").is_file() else None,
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


def _warn_cross_device_once(src: Path, dst_parent: Path, src_dev: int, dst_dev: int) -> None:
    pair = (src_dev, dst_dev)
    if pair not in _WARNED_CROSS_DEVICE_PAIRS:
        _WARNED_CROSS_DEVICE_PAIRS.add(pair)
        logger.warning(
            "Cross-filesystem staging detected (src=%s st_dev=%d, dst=%s st_dev=%d); "
            "falling back from zero-copy hardlink (os.link) to regular 0o644 file copy.",
            src,
            src_dev,
            dst_parent,
            dst_dev,
        )


def _hardlink_file(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst`` when on the same filesystem, else warn and copy as 0o644."""
    if src.is_symlink():
        src = src.resolve(strict=True)
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_dev = src.stat().st_dev
    dst_dev = dst.parent.stat().st_dev
    if src_dev != dst_dev:
        _warn_cross_device_once(src, dst.parent, src_dev, dst_dev)
        dst.unlink(missing_ok=True)
        shutil.copy2(src, dst, follow_symlinks=True)
        os.chmod(dst, 0o644)
        return

    try:
        dst.unlink(missing_ok=True)
        os.link(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        _warn_cross_device_once(src, dst.parent, src_dev, -1)
        dst.unlink(missing_ok=True)
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


def _mcp_servers_for_ports(mcp_ports: dict[str, int]) -> list[dict[str, str]]:
    ordered_names = ["ms-inspect", "ms-modify", "ms-create"]
    extras = sorted(set(mcp_ports) - set(ordered_names))
    out: list[dict[str, str]] = []
    for srv_name in [*ordered_names, *extras]:
        if srv_name in mcp_ports:
            out.append(
                {
                    "name": srv_name,
                    "transport": "streamable-http",
                    "url": f"http://127.0.0.1:{int(mcp_ports[srv_name])}/mcp",
                }
            )
    return out


def format_variant_name(
    task_id: str,
    arm: str,
    *,
    pack_id: str | None = None,
    include_pack_in_name: bool = False,
) -> str:
    if include_pack_in_name and pack_id:
        return f"{task_id}__{pack_id}__{arm}"
    return f"{task_id}__{arm}"


def _build_task_toml_dict(
    spec: TaskSpec,
    arm: str,
    *,
    variant_id: str,
    pack_id: str,
    analyst_sha: str,
    mcp_ports: dict[str, int],
) -> dict[str, Any]:
    """Build Harbor 0.25.0 task.toml dictionary for ``(spec, pack, arm)``."""
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
        env_cfg["mcp_servers"] = _mcp_servers_for_ports(mcp_ports)

    meta_dict: dict[str, Any] = {
        "task_id": variant_id,
        "base_task": spec.task_id,
        "pack_id": pack_id,
        "analyst_sha": analyst_sha,
        "suite": spec.suite,
        "suite_id": spec.suite_id or spec.suite,
        "split": spec.split,
        "stage": spec.stage,
        "stages": list(spec.stages) if spec.stages else [spec.stage],
        "arm": arm,
        "difficulty": spec.difficulty,
        "category": spec.category,
        "tags": list(spec.tags),
        "radio_analyst_sha": analyst_sha,
    }

    data: dict[str, Any] = {
        "schema_version": "1.4",
        "artifacts": ["/workspace"],
        "metadata": meta_dict,
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
    sidecar_image: str,
    mcp_ports: dict[str, int],
) -> dict[str, Any]:
    """Build environment/docker-compose.yaml for ``arm`` and ``pack``."""
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
        ports_tuple = tuple(int(p) for p in mcp_ports.values())
        tcp_probe = (
            "/opt/casa-venv/bin/python -c '"
            "import socket, sys, time\n"
            f"for port in {ports_tuple!r}:\n"
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
        if "ms-inspect" in mcp_ports and int(mcp_ports["ms-inspect"]) != 8000:
            sidecar_env["RADIO_MCP_PORT_MS_INSPECT"] = str(int(mcp_ports["ms-inspect"]))
        if "ms-modify" in mcp_ports and int(mcp_ports["ms-modify"]) != 8001:
            sidecar_env["RADIO_MCP_PORT_MS_MODIFY"] = str(int(mcp_ports["ms-modify"]))
        if "ms-create" in mcp_ports and int(mcp_ports["ms-create"]) != 8002:
            sidecar_env["RADIO_MCP_PORT_MS_CREATE"] = str(int(mcp_ports["ms-create"]))

        sidecar_image_ref = (
            sidecar_image
            if sidecar_image.startswith("${")
            else f"${{RADIO_BENCH_SIDECAR_IMAGE:-{sidecar_image}}}"
        )
        services["mcp-servers"] = {
            "image": sidecar_image_ref,
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


def _validate_exported_skills(exported_skills_dir: Path, *, pack_id: str | None = None) -> None:
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
        hint = f" --pack {pack_id}" if pack_id else ""
        raise FileNotFoundError(
            f"Missing exported skill files (run `python -m radio_bench.export_skills{hint}` first): "
            + ", ".join(missing)
        )


def _resolve_pack_skills_and_sha(
    pack: PackConfig,
    skills_dir: Path,
    *,
    default_pack: str,
    cfg: BenchConfig,
) -> tuple[Path, str]:
    """Locate exported_skills directory and resolved short SHA for `pack`."""
    per_pack_dir = skills_dir / pack.pack_id
    if (per_pack_dir / "raw" / "skills").is_dir():
        eff_skills = per_pack_dir
    elif pack.pack_id == default_pack and (skills_dir / "raw" / "skills").is_dir():
        eff_skills = skills_dir
    else:
        eff_skills = per_pack_dir

    _validate_exported_skills(eff_skills, pack_id=pack.pack_id)

    manifest_file = eff_skills / "manifest.json"
    if manifest_file.is_file():
        try:
            mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
            sha_cand = str(mdata.get("radio_analyst_sha") or "").strip()
            commit_cand = str(mdata.get("radio_analyst_commit") or "").strip()
            if _HEX_SHORT_RE.fullmatch(sha_cand) and sha_cand.upper() != "HEAD":
                return eff_skills, sha_cand.lower()
            if _HEX_SHORT_RE.fullmatch(commit_cand):
                return eff_skills, commit_cand[:7].lower()
        except (OSError, ValueError):
            pass

    if _HEX_SHORT_RE.fullmatch(pack.analyst_ref) and pack.analyst_ref.upper() != "HEAD":
        return eff_skills, pack.analyst_ref.lower()

    _, _, short_sha = materialize_pack_source(pack, cfg.work_dir, repo_root=cfg.repo_root)
    return eff_skills, short_sha


def compile_variant(
    spec: TaskSpec,
    arm: str,
    *,
    output_dir: Path,
    exported_skills_dir: Path,
    pack: PackConfig | None = None,
    analyst_sha: str | None = None,
    include_pack_in_name: bool = False,
) -> Path:
    """Compile a single Harbor task directory under ``output_dir``."""
    reject_slash_tmp(output_dir, label="output_dir")
    eff_pack = pack or PackConfig(
        pack_id="ra_00a11ea",
        description="Frozen v0.1 radio-analyst baseline",
        analyst_src="",
        analyst_ref="00a11ea",
        sidecar_image="localhost/radio-bench-mcp-sidecar:00a11ea",
        mcp_ports=dict(DEFAULT_MCP_PORTS),
    )
    eff_sha = analyst_sha or (
        eff_pack.analyst_ref.lower() if _HEX_SHORT_RE.fullmatch(eff_pack.analyst_ref) else "00a11ea"
    )

    variant_name = format_variant_name(
        spec.task_id,
        arm,
        pack_id=eff_pack.pack_id,
        include_pack_in_name=include_pack_in_name,
    )
    variant_dir = output_dir / variant_name
    if variant_dir.exists():
        shutil.rmtree(variant_dir)

    variant_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(variant_dir, 0o755)

    # 1. instruction.md (verbatim regular file; identical across arms of spec.task_id)
    _copy_regular_file(spec.effective_instruction_path, variant_dir / "instruction.md")

    # 2. task.toml
    toml_dict = _build_task_toml_dict(
        spec,
        arm,
        variant_id=variant_name,
        pack_id=eff_pack.pack_id,
        analyst_sha=eff_sha,
        mcp_ports=eff_pack.mcp_ports,
    )
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
        sidecar_image=eff_pack.sidecar_image,
        mcp_ports=eff_pack.mcp_ports,
    )
    _write_text_file(
        env_dir / "docker-compose.yaml",
        yaml.safe_dump(env_compose, sort_keys=False),
    )

    # 4. solution/ (oracle only; zero ground truth, rubric, or verifier files)
    sol_dir = variant_dir / "solution"
    sol_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(sol_dir, 0o755)
    _copy_regular_file(spec.effective_solve_path, sol_dir / "solve.py")
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

    # 5. tests/ (isolated declarative verifier bundle: extract_config.json + rubric.json + ground_truth.json)
    tests_dir = variant_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(tests_dir, 0o755)

    if spec.extract_config_json is not None and spec.rubric_path is not None:
        _write_text_file(tests_dir / "extract_config.json", spec.extract_config_json)
        _copy_regular_file(spec.rubric_path, tests_dir / "rubric.json")
    else:
        from radio_bench.grading.inspect_metrics import resolve_task_configs

        gt_dict = loads_strict_json(spec.ground_truth.read_text(encoding="utf-8"))
        ec_fb, rb_fb = resolve_task_configs(spec.task_id, spec.stage, gt_dict)
        _write_text_file(
            tests_dir / "extract_config.json",
            json.dumps(ec_fb.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        )
        _write_text_file(
            tests_dir / "rubric.json",
            json.dumps(rb_fb.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        )

    _hardlink_file(spec.ground_truth, tests_dir / "ground_truth.json")
    for extra_name in spec.extra_test_files:
        _hardlink_file(spec.data_dir / extra_name, tests_dir / extra_name)

    if spec.ref_flags_file:
        src_flags = spec.data_dir / spec.ref_flags_file
        if src_flags.is_file() and not (tests_dir / "ref_flags.npz").exists():
            _hardlink_file(src_flags, tests_dir / "ref_flags.npz")

    if spec.true_target_vis_file:
        src_tvis = spec.data_dir / spec.true_target_vis_file
        if src_tvis.is_file() and not (tests_dir / "true_target_vis.npz").exists():
            _hardlink_file(src_tvis, tests_dir / "true_target_vis.npz")

    test_sh = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{spec.effective_verifier_cpus}}}"\n'
        'export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"\n'
        'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"\n'
        'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"\n'
        'export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"\n'
        "/opt/casa-venv/bin/python -m radio_bench.grading.runner "
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
    specs_dir: Path | None = None,
    suites_dir: Path | None = None,
    skills_dir: Path | None = None,
    output_dir: Path | None = None,
    skills_mode: str = "both",
    include_orchestrated: bool = True,
    pack_id: str | None = None,
    pack_ids: Sequence[str] | None = None,
    force_pack_in_name: bool = False,
    cfg: BenchConfig | None = None,
) -> dict[str, Any]:
    """Compile all configured Harbor task directories and prune unselected variants."""
    _WARNED_CROSS_DEVICE_PAIRS.clear()
    resolved_cfg = cfg if cfg is not None else load_bench_config()

    eff_skills_dir = skills_dir if skills_dir is not None else resolved_cfg.exported_skills_dir
    eff_output_dir = output_dir if output_dir is not None else resolved_cfg.tasks_dir
    reject_slash_tmp(eff_output_dir, label="output_dir")

    selected_pack_ids: list[str]
    if pack_ids is not None and len(pack_ids) > 0:
        selected_pack_ids = [p.strip() for p in pack_ids if p.strip()]
    elif pack_id is not None and pack_id.strip():
        selected_pack_ids = [pack_id.strip()]
    else:
        selected_pack_ids = [resolved_cfg.default_pack]

    if not selected_pack_ids:
        raise ValueError("At least one pack ID is required")

    packs = [resolved_cfg.get_pack(pid) for pid in selected_pack_ids]
    include_pack_in_name = bool(force_pack_in_name or len(packs) > 1)

    pack_bundles: list[tuple[PackConfig, Path, str]] = []
    for p in packs:
        p_skills_dir, p_sha = _resolve_pack_skills_and_sha(
            p,
            eff_skills_dir,
            default_pack=resolved_cfg.default_pack,
            cfg=resolved_cfg,
        )
        pack_bundles.append((p, p_skills_dir, p_sha))

    if specs_dir is not None:
        spec_dirs = sorted(
            p for p in specs_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        if not spec_dirs:
            raise FileNotFoundError(f"No task spec directories found under {specs_dir}")
        specs = [load_task_spec(d, cfg=resolved_cfg) for d in spec_dirs]
    else:
        specs = load_suite_tasks(suites_dir=suites_dir, cfg=resolved_cfg)
        if not specs:
            raise FileNotFoundError("No tasks loaded from suites/*/suite.toml")

    eff_output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(eff_output_dir, 0o755)

    known_prefixes = tuple(f"{s.task_id}__" for s in specs)
    emitted: list[str] = []
    by_suite: dict[str, int] = {}
    by_arm: dict[str, int] = {}
    by_pack: dict[str, int] = {}

    for spec in specs:
        arms = arms_for_spec(
            spec,
            skills_mode=skills_mode,
            include_orchestrated=include_orchestrated,
        )
        for p_cfg, p_skills_dir, p_sha in pack_bundles:
            for arm in arms:
                variant_dir = compile_variant(
                    spec,
                    arm,
                    output_dir=eff_output_dir,
                    exported_skills_dir=p_skills_dir,
                    pack=p_cfg,
                    analyst_sha=p_sha,
                    include_pack_in_name=include_pack_in_name,
                )
                emitted.append(variant_dir.name)
                by_suite[spec.suite] = by_suite.get(spec.suite, 0) + 1
                by_arm[arm] = by_arm.get(arm, 0) + 1
                by_pack[p_cfg.pack_id] = by_pack.get(p_cfg.pack_id, 0) + 1

    emitted_set = set(emitted)
    for child in sorted(eff_output_dir.iterdir()):
        if (
            child.is_dir()
            and child.name.startswith(known_prefixes)
            and child.name not in emitted_set
        ):
            shutil.rmtree(child)

    return {
        "output_dir": str(eff_output_dir),
        "skills_mode": skills_mode,
        "include_orchestrated": include_orchestrated,
        "packs": [p.pack_id for p in packs],
        "include_pack_in_name": include_pack_in_name,
        "base_tasks": len(specs),
        "compiled_tasks": len(emitted),
        "by_suite": by_suite,
        "by_arm": by_arm,
        "by_pack": by_pack,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile Harbor 0.25.0 task directories for radio-bench."
    )
    add_config_cli_args(parser)
    parser.add_argument(
        "--pack",
        type=str,
        default=None,
        help="Single pack ID to compile (emits <task_id>__<arm> unless --force-pack-in-name).",
    )
    parser.add_argument(
        "--packs",
        type=str,
        default=None,
        help="Comma-separated pack IDs (multi-pack emits <task_id>__<pack_id>__<arm>).",
    )
    parser.add_argument(
        "--force-pack-in-name",
        action="store_true",
        help="Include <pack_id> in directory names even when compiling a single pack.",
    )
    parser.add_argument(
        "--suites-dir",
        type=Path,
        default=None,
        help="Path to suites/ directory containing suite.toml manifests.",
    )
    parser.add_argument(
        "--specs-dir",
        type=Path,
        default=None,
        help="Optional subset directory of task folders.",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        help="Path to exported_skills/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
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

    cfg = config_from_cli_args(args)
    parsed_packs = [p.strip() for p in args.packs.split(",") if p.strip()] if args.packs else None
    summary = build_all_tasks(
        specs_dir=args.specs_dir,
        suites_dir=args.suites_dir,
        skills_dir=args.skills_dir,
        output_dir=args.output_dir,
        skills_mode=args.skills_mode,
        include_orchestrated=args.include_orchestrated,
        pack_id=args.pack,
        pack_ids=parsed_packs,
        force_pack_in_name=args.force_pack_in_name,
        cfg=cfg,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
