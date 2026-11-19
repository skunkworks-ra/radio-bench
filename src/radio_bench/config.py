"""Host path resolution, multi-pack source materialization, and model registry loader."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PACK_ID = "ra_00a11ea"
DEFAULT_MCP_PORTS: dict[str, int] = {
    "ms-inspect": 8000,
    "ms-modify": 8001,
    "ms-create": 8002,
}

_HEX_REF_RE = re.compile(r"^[0-9a-fA-F]{4,40}$")
_SCP_GIT_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+:.+")

_BUILTIN_PACKS: dict[str, dict[str, Any]] = {
    "ra_00a11ea": {
        "description": "Frozen v0.1 radio-analyst baseline",
        "analyst_src": "https://github.com/skunkworks-ra/radio-analyst.git",
        "analyst_ref": "00a11ea",
        "sidecar_image": "localhost/radio-bench-mcp-sidecar:00a11ea",
        "mcp_ports": dict(DEFAULT_MCP_PORTS),
    },
    "ra_rescue": {
        "description": "Conditional pre-flight + rescue-reviewer scientist loop",
        "analyst_src": "../radio-analyst-rescue",
        "analyst_ref": "HEAD",
        "sidecar_image": "localhost/radio-bench-mcp-sidecar:ra_rescue",
        "mcp_ports": dict(DEFAULT_MCP_PORTS),
    },
}

_BUILTIN_MODELS_RAW: dict[str, Any] = {
    "models": {
        "claude_haiku_4_5": {
            "harness": "claude-code",
            "model_name": "anthropic/claude-haiku-4-5",
            "api_format": "anthropic_messages",
            "api_key_env": "ANTHROPIC_API_KEY",
            "thinking_levels": ["default"],
            "context_window_tokens": 200000,
            "disable_web_search": True,
            "aliases": ["claude-haiku-4-5", "haiku-4-5", "haiku-4.5"],
            "pricing": {
                "input_per_mtok": 1.00,
                "output_per_mtok": 5.00,
                "cache_read_per_mtok": 0.10,
                "cache_write_5m_per_mtok": 1.25,
                "cache_write_1h_per_mtok": 2.00,
                "cliff_prompt_tokens": 100000,
                "cliff_multiplier": 5.0,
            },
        },
        "claude_sonnet_4_6": {
            "harness": "claude-code",
            "model_name": "anthropic/claude-sonnet-4-6",
            "api_format": "anthropic_messages",
            "api_key_env": "ANTHROPIC_API_KEY",
            "thinking_levels": ["default", "high"],
            "context_window_tokens": 200000,
            "disable_web_search": True,
            "aliases": ["claude-sonnet-4-6", "sonnet-4-6", "sonnet-4.6", "sonnet-4-5"],
            "pricing": {
                "input_per_mtok": 3.00,
                "output_per_mtok": 15.00,
                "cache_read_per_mtok": 0.30,
                "cache_write_5m_per_mtok": 3.75,
                "cache_write_1h_per_mtok": 6.00,
            },
        },
        "nrp_kimi": {
            "harness": "opencode",
            "model_name": "nrp/kimi",
            "api_format": "openai_chat_completions",
            "base_url": "https://ellm.nrp-nautilus.io/v1",
            "api_key_env": "NRP_API_KEY",
            "opencode_version": "1.19.5",
            "thinking_levels": ["default"],
            "context_window_tokens": 131072,
            "disable_web_search": True,
            "pricing": {
                "input_per_mtok": 0.60,
                "output_per_mtok": 3.00,
                "cache_read_per_mtok": 0.10,
                "cache_write_5m_per_mtok": 0.60,
                "cache_write_1h_per_mtok": 0.60,
            },
        },
        "nrp_glm5": {
            "harness": "opencode",
            "model_name": "nrp/glm-5",
            "api_format": "openai_chat_completions",
            "base_url": "https://ellm.nrp-nautilus.io/v1",
            "api_key_env": "NRP_API_KEY",
            "opencode_version": "1.19.5",
            "thinking_levels": ["default"],
            "context_window_tokens": 131072,
            "disable_web_search": True,
            "pricing": {
                "input_per_mtok": 1.00,
                "output_per_mtok": 3.20,
                "cache_read_per_mtok": 0.20,
                "cache_write_5m_per_mtok": 1.00,
                "cache_write_1h_per_mtok": 1.00,
            },
        },
        "tacc_kimi": {
            "harness": "opencode",
            "model_name": "tacc/kimi",
            "api_format": "openai_chat_completions",
            "base_url_env": "TACC_BASE_URL",
            "api_key_env": "TACC_API_KEY",
            "opencode_version": "1.19.5",
            "thinking_levels": ["default"],
            "context_window_tokens": 131072,
            "disable_web_search": True,
            "pricing": {
                "input_per_mtok": 0.60,
                "output_per_mtok": 3.00,
                "cache_read_per_mtok": 0.10,
                "cache_write_5m_per_mtok": 0.60,
                "cache_write_1h_per_mtok": 0.60,
            },
        },
        "local_llama_qwen": {
            "harness": "opencode",
            "model_name": "local/qwen",
            "api_format": "openai_chat_completions",
            "base_url": "http://host.containers.internal:8080/v1",
            "api_key_env": "LOCAL_LLM_API_KEY",
            "opencode_version": "1.19.5",
            "thinking_levels": ["default"],
            "context_window_tokens": 65536,
            "max_concurrent_trials": 1,
            "host_gateway_required": True,
            "disable_web_search": True,
            "pricing": {
                "input_per_mtok": 0.0,
                "output_per_mtok": 0.0,
                "cache_read_per_mtok": 0.0,
                "cache_write_5m_per_mtok": 0.0,
                "cache_write_1h_per_mtok": 0.0,
            },
        },
    },
    "parity_pairs": [
        {
            "pair_id": "anthropic_haiku_vs_sonnet",
            "small": {"model": "claude_haiku_4_5", "arm": "11_full", "thinking_level": "default"},
            "frontier": {
                "model": "claude_sonnet_4_6",
                "arm": "00_none",
                "thinking_level": "default",
            },
        },
        {
            "pair_id": "nrp_kimi_vs_glm5",
            "small": {"model": "nrp_kimi", "arm": "11_full", "thinking_level": "default"},
            "frontier": {"model": "nrp_glm5", "arm": "00_none", "thinking_level": "default"},
        },
    ],
}


def resolve_repo_root(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the repository root without touching sys.executable."""
    eff_env = os.environ if env is None else env
    raw_override = eff_env.get("RADIO_BENCH_REPO_ROOT", "").strip()
    if raw_override:
        return Path(os.path.expanduser(raw_override)).resolve()

    src_cand = Path(__file__).resolve().parents[2]
    if (src_cand / "bench.toml").is_file() or (src_cand / "pyproject.toml").is_file():
        return src_cand

    container_cand = Path("/opt/radio-bench")
    if (container_cand / "bench.toml").is_file() or (container_cand / "pyproject.toml").is_file():
        return container_cand

    return src_cand


def is_under_slash_tmp(path: Path | str) -> bool:
    """Return True if path is syntactically or after symlink resolution under /tmp."""
    raw = str(path).strip()
    if not raw:
        return False
    if raw == "/tmp" or raw.startswith("/tmp/"):
        return True
    p = Path(raw)
    if p == Path("/tmp") or Path("/tmp") in p.parents:
        return True
    try:
        resolved = str(p.expanduser().resolve())
    except OSError:
        return False
    return resolved == "/tmp" or resolved.startswith("/tmp/")


def reject_slash_tmp(path: Path | str, label: str = "path") -> Path:
    """Validate that path is neither syntactically nor via symlinks under /tmp."""
    raw = str(path).strip()
    if not raw:
        raise ValueError(f"{label} must not be empty")
    if raw == "/tmp" or raw.startswith("/tmp/"):
        raise ValueError(f"{label} must not use /tmp: {path}")
    p = Path(os.path.expanduser(raw))
    if p == Path("/tmp") or Path("/tmp") in p.parents:
        raise ValueError(f"{label} must not use /tmp: {path}")
    resolved = p.resolve()
    res_str = str(resolved)
    if res_str == "/tmp" or res_str.startswith("/tmp/"):
        raise ValueError(f"{label} resolves under /tmp: {path} -> {resolved}")
    return resolved


def _expand_tokens(value: str, *, uid: int | None = None) -> str:
    resolved_uid = os.getuid() if uid is None else uid
    return value.replace("${UID}", str(resolved_uid))


def _resolve_config_path(
    raw_value: str | Path,
    *,
    repo_root: Path,
    label: str,
    uid: int | None = None,
) -> Path:
    expanded = os.path.expanduser(_expand_tokens(str(raw_value).strip(), uid=uid))
    if not expanded:
        raise ValueError(f"{label} must not be empty")
    raw_path = Path(expanded)
    # Check raw syntactic /tmp before anchoring relative paths.
    if str(raw_path) == "/tmp" or str(raw_path).startswith("/tmp/"):
        raise ValueError(f"{label} must not use /tmp: {raw_value}")
    anchored = raw_path if raw_path.is_absolute() else (repo_root / raw_path)
    return reject_slash_tmp(anchored, label=label)


def is_remote_git_url(src: str) -> bool:
    """Return True when src is a remote git URL rather than a local filesystem path."""
    s = src.strip()
    if s.startswith(("https://", "http://", "ssh://", "git://")):
        return True
    return bool(_SCP_GIT_RE.match(s))


@dataclass(frozen=True)
class PackConfig:
    """Tool and skill pack definition from [packs.<pack_id>]."""

    pack_id: str
    description: str
    analyst_src: str
    analyst_ref: str
    sidecar_image: str
    mcp_ports: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MCP_PORTS))


@dataclass(frozen=True)
class BenchConfig:
    """Resolved host path and pack configuration."""

    repo_root: Path
    work_dir: Path
    data_dir: Path
    runs_dir: Path
    casaguides_dir: Path
    prompts_dir: Path
    casa_data_dir: Path
    docker_host: str
    container_tmpdir: Path
    default_pack: str
    packs: dict[str, PackConfig]

    @property
    def tasks_dir(self) -> Path:
        return self.work_dir / "tasks"

    @property
    def exported_skills_dir(self) -> Path:
        return self.work_dir / "exported_skills"

    @property
    def jobs_dir(self) -> Path:
        return self.work_dir / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.work_dir / "tmp"

    @property
    def mplconfig_dir(self) -> Path:
        return self.work_dir / "mplconfig"

    @property
    def pytest_dir(self) -> Path:
        return self.work_dir / "pytest"

    @property
    def pytest_cache_dir(self) -> Path:
        return self.work_dir / "pytest_cache"

    @property
    def docker_config_dir(self) -> Path:
        return self.work_dir / "docker_config"

    @property
    def casaconfig_path(self) -> Path:
        return self.work_dir / "casaconfig_local.py"

    def get_pack(self, pack_id: str | None = None) -> PackConfig:
        target = (pack_id or self.default_pack).strip()
        if target not in self.packs:
            available = ", ".join(sorted(self.packs))
            raise KeyError(f"Unknown pack {target!r}; configured packs: {available}")
        return self.packs[target]


@dataclass(frozen=True)
class ModelPricingConfig:
    """Declarative USD per-million-token pricing and prompt cliff settings."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_5m_per_mtok: float
    cache_write_1h_per_mtok: float
    cliff_prompt_tokens: int | None = None
    cliff_multiplier: float = 1.0


@dataclass(frozen=True)
class ModelSpec:
    """Declarative inference endpoint, harness, and pricing definition."""

    key: str
    harness: str
    model_name: str
    api_format: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    opencode_version: str | None = None
    thinking_levels: tuple[str, ...] = ("default",)
    context_window_tokens: int = 131072
    max_concurrent_trials: int | None = None
    host_gateway_required: bool = False
    disable_web_search: bool = True
    aliases: tuple[str, ...] = ()
    opencode_config: dict[str, Any] = field(default_factory=dict)
    pricing: ModelPricingConfig = field(
        default_factory=lambda: ModelPricingConfig(0.0, 0.0, 0.0, 0.0, 0.0)
    )

    @property
    def agent(self) -> str:
        return self.harness

    @property
    def model(self) -> str:
        return self.model_name


@dataclass(frozen=True)
class ParitySideSpec:
    """One side (small or frontier) of a parity comparison pair."""

    model: str
    arm: str
    thinking_level: str = "default"


@dataclass(frozen=True)
class ParityPairSpec:
    """Small + Full Stack vs Frontier Bare parity comparison definition."""

    pair_id: str
    small: ParitySideSpec
    frontier: ParitySideSpec


@dataclass(frozen=True)
class ModelsConfig:
    """Loaded models and parity pair registry from configs/models.toml."""

    models: dict[str, ModelSpec]
    parity_pairs: tuple[ParityPairSpec, ...]

    def resolve_model_name(self, key_or_name: str) -> str:
        if key_or_name in self.models:
            return self.models[key_or_name].model_name
        return key_or_name


def _extract_flat_settings(doc: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for top_key in (
        "work_dir",
        "data_dir",
        "runs_dir",
        "casaguides_dir",
        "prompts_dir",
        "casa_data_dir",
        "docker_host",
        "container_tmpdir",
        "default_pack",
    ):
        if top_key in doc and doc[top_key] is not None:
            merged[top_key] = doc[top_key]
    for section_name in ("paths", "runtime"):
        sec = doc.get(section_name)
        if isinstance(sec, Mapping):
            for k, v in sec.items():
                if v is not None:
                    merged[k] = v
    return merged


def _merge_pack_dicts(
    target: dict[str, dict[str, Any]],
    incoming: Mapping[str, Any] | None,
) -> None:
    if not isinstance(incoming, Mapping):
        return
    for pack_id, raw_pack in incoming.items():
        if not isinstance(raw_pack, Mapping):
            continue
        cur = target.setdefault(str(pack_id), {})
        for k, v in raw_pack.items():
            if k == "mcp_ports" and isinstance(v, Mapping):
                ports = dict(cur.get("mcp_ports") or DEFAULT_MCP_PORTS)
                for srv, port in v.items():
                    ports[str(srv)] = int(port)
                cur["mcp_ports"] = ports
            elif v is not None:
                cur[str(k)] = v


def load_bench_config(
    *,
    config_path: Path | str | None = None,
    repo_root: Path | None = None,
    work_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    runs_dir: Path | str | None = None,
    casaguides_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    casa_data_dir: Path | str | None = None,
    docker_host: str | None = None,
    container_tmpdir: Path | str | None = None,
    default_pack: str | None = None,
    env: Mapping[str, str] | None = None,
    uid: int | None = None,
    ignore_local_config: bool = False,
) -> BenchConfig:
    """Load BenchConfig with strict precedence: CLI > env > bench.local.toml > bench.toml.

    Side-effect-free: never creates directories or validates pack analyst_src existence.
    """
    eff_env = os.environ if env is None else env
    resolved_repo = repo_root.resolve() if repo_root is not None else resolve_repo_root(eff_env)

    # Track setting value + priority rank:
    # 0 = builtin default, 1 = bench.toml (or explicit config_path), 2 = bench.local.toml,
    # 3 = environment variables, 4 = explicit function/CLI arguments.
    values: dict[str, tuple[Any, int]] = {
        "work_dir": ("./.work", 0),
        "data_dir": ("./data", 0),
        "runs_dir": ("./runs", 0),
        "casaguides_dir": ("./data/casaguides", 0),
        "prompts_dir": ("./task_specs", 0),
        "casa_data_dir": ("~/.casa/data", 0),
        "docker_host": ("unix:///run/user/${UID}/podman/podman.sock", 0),
        "container_tmpdir": ("./.work/podman_tmp", 0),
        "default_pack": (DEFAULT_PACK_ID, 0),
    }

    raw_packs: dict[str, dict[str, Any]] = {}
    _merge_pack_dicts(raw_packs, _BUILTIN_PACKS)

    env_cfg_override = eff_env.get("RADIO_BENCH_CONFIG", "").strip()
    if config_path is not None:
        base_toml_path = Path(os.path.expanduser(str(config_path)))
        if not base_toml_path.is_file():
            raise FileNotFoundError(f"Explicit config file not found: {base_toml_path}")
        base_doc = tomllib.loads(base_toml_path.read_text(encoding="utf-8"))
        for k, v in _extract_flat_settings(base_doc).items():
            values[k] = (v, 1)
        _merge_pack_dicts(raw_packs, base_doc.get("packs"))
    elif env_cfg_override:
        env_toml_path = Path(os.path.expanduser(env_cfg_override))
        if not env_toml_path.is_file():
            raise FileNotFoundError(f"RADIO_BENCH_CONFIG file not found: {env_toml_path}")
        base_doc = tomllib.loads(env_toml_path.read_text(encoding="utf-8"))
        for k, v in _extract_flat_settings(base_doc).items():
            values[k] = (v, 1)
        _merge_pack_dicts(raw_packs, base_doc.get("packs"))
    else:
        committed_toml = resolved_repo / "bench.toml"
        if committed_toml.is_file():
            committed_doc = tomllib.loads(committed_toml.read_text(encoding="utf-8"))
            for k, v in _extract_flat_settings(committed_doc).items():
                values[k] = (v, 1)
            _merge_pack_dicts(raw_packs, committed_doc.get("packs"))

        if not ignore_local_config:
            local_toml = resolved_repo / "bench.local.toml"
            if local_toml.is_file():
                local_doc = tomllib.loads(local_toml.read_text(encoding="utf-8"))
                for k, v in _extract_flat_settings(local_doc).items():
                    values[k] = (v, 2)
                _merge_pack_dicts(raw_packs, local_doc.get("packs"))

    env_map = {
        "work_dir": "RADIO_BENCH_WORK_DIR",
        "data_dir": "RADIO_BENCH_DATA_DIR",
        "runs_dir": "RADIO_BENCH_RUNS_DIR",
        "casaguides_dir": "RADIO_BENCH_CASAGUIDES_DIR",
        "prompts_dir": "RADIO_BENCH_PROMPTS_DIR",
        "casa_data_dir": "RADIO_BENCH_CASA_DATA",
        "docker_host": "DOCKER_HOST",
        "container_tmpdir": "CONTAINER_TMPDIR",
        "default_pack": "RADIO_BENCH_DEFAULT_PACK",
    }
    for key, env_var in env_map.items():
        env_val = eff_env.get(env_var, "").strip()
        if env_val:
            values[key] = (env_val, 3)

    cli_overrides: dict[str, Any] = {
        "work_dir": work_dir,
        "data_dir": data_dir,
        "runs_dir": runs_dir,
        "casaguides_dir": casaguides_dir,
        "prompts_dir": prompts_dir,
        "casa_data_dir": casa_data_dir,
        "docker_host": docker_host,
        "container_tmpdir": container_tmpdir,
        "default_pack": default_pack,
    }
    for key, cli_val in cli_overrides.items():
        if cli_val is not None and str(cli_val).strip():
            values[key] = (cli_val, 4)

    resolved_work = _resolve_config_path(
        values["work_dir"][0], repo_root=resolved_repo, label="work_dir", uid=uid
    )
    resolved_data = _resolve_config_path(
        values["data_dir"][0], repo_root=resolved_repo, label="data_dir", uid=uid
    )
    resolved_runs = _resolve_config_path(
        values["runs_dir"][0], repo_root=resolved_repo, label="runs_dir", uid=uid
    )
    resolved_casaguides = _resolve_config_path(
        values["casaguides_dir"][0], repo_root=resolved_repo, label="casaguides_dir", uid=uid
    )
    resolved_prompts = _resolve_config_path(
        values["prompts_dir"][0], repo_root=resolved_repo, label="prompts_dir", uid=uid
    )
    resolved_casa_data = _resolve_config_path(
        values["casa_data_dir"][0], repo_root=resolved_repo, label="casa_data_dir", uid=uid
    )

    work_rank = values["work_dir"][1]
    ctmp_val, ctmp_rank = values["container_tmpdir"]
    if ctmp_rank < work_rank:
        resolved_ctmp = reject_slash_tmp(resolved_work / "podman_tmp", label="container_tmpdir")
    else:
        resolved_ctmp = _resolve_config_path(
            ctmp_val, repo_root=resolved_repo, label="container_tmpdir", uid=uid
        )

    resolved_docker_host = _expand_tokens(str(values["docker_host"][0]).strip(), uid=uid)
    if resolved_docker_host.startswith("unix://"):
        sock_raw = resolved_docker_host[len("unix://") :]
        sock_expanded = os.path.expanduser(sock_raw)
        reject_slash_tmp(sock_expanded, label="docker_host")
        resolved_docker_host = f"unix://{sock_expanded}"

    packs: dict[str, PackConfig] = {}
    for pack_id, pdata in sorted(raw_packs.items()):
        raw_src = _expand_tokens(str(pdata.get("analyst_src", "")).strip(), uid=uid)
        if not raw_src:
            raise ValueError(f"packs.{pack_id}.analyst_src must not be empty")
        if not is_remote_git_url(raw_src):
            _resolve_config_path(
                raw_src,
                repo_root=resolved_repo,
                label=f"packs.{pack_id}.analyst_src",
                uid=uid,
            )
        ports_raw = pdata.get("mcp_ports") or DEFAULT_MCP_PORTS
        ports = {str(k): int(v) for k, v in ports_raw.items()}
        packs[pack_id] = PackConfig(
            pack_id=pack_id,
            description=str(pdata.get("description", "")),
            analyst_src=raw_src,
            analyst_ref=str(pdata.get("analyst_ref", "HEAD")).strip() or "HEAD",
            sidecar_image=str(
                pdata.get("sidecar_image", f"localhost/radio-bench-mcp-sidecar:{pack_id}")
            ).strip(),
            mcp_ports=ports,
        )

    resolved_default_pack = str(values["default_pack"][0]).strip() or DEFAULT_PACK_ID
    if resolved_default_pack not in packs:
        raise ValueError(
            f"default_pack={resolved_default_pack!r} is not defined in [packs.*]: {sorted(packs)}"
        )

    return BenchConfig(
        repo_root=resolved_repo,
        work_dir=resolved_work,
        data_dir=resolved_data,
        runs_dir=resolved_runs,
        casaguides_dir=resolved_casaguides,
        prompts_dir=resolved_prompts,
        casa_data_dir=resolved_casa_data,
        docker_host=resolved_docker_host,
        container_tmpdir=resolved_ctmp,
        default_pack=resolved_default_pack,
        packs=packs,
    )


def _deep_merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {str(k): v for k, v in base.items()}
    for k, v in override.items():
        ks = str(k)
        if isinstance(v, Mapping) and isinstance(out.get(ks), Mapping):
            out[ks] = _deep_merge_mapping(out[ks], v)
        elif isinstance(v, Mapping):
            out[ks] = _deep_merge_mapping({}, v)
        else:
            out[ks] = v
    return out


def _merge_models_doc(base_doc: dict[str, Any], override_doc: Mapping[str, Any]) -> None:
    if isinstance(override_doc.get("models"), Mapping):
        models_target = base_doc.setdefault("models", {})
        for mkey, mval in override_doc["models"].items():
            if not isinstance(mval, Mapping):
                continue
            cur = dict(models_target.get(mkey) or {})
            if "base_url_env" in mval and mval["base_url_env"] and "base_url" not in mval:
                cur.pop("base_url", None)
            if "base_url" in mval and mval["base_url"] and "base_url_env" not in mval:
                cur.pop("base_url_env", None)
            for fk, fv in mval.items():
                if fk == "pricing" and isinstance(fv, Mapping):
                    merged_pricing = dict(cur.get("pricing") or {})
                    merged_pricing.update(fv)
                    cur["pricing"] = merged_pricing
                elif fk == "opencode_config" and isinstance(fv, Mapping):
                    cur["opencode_config"] = _deep_merge_mapping(
                        cur.get("opencode_config") or {}, fv
                    )
                else:
                    cur[fk] = fv
            models_target[str(mkey)] = cur
    if isinstance(override_doc.get("parity_pairs"), list):
        base_doc["parity_pairs"] = list(override_doc["parity_pairs"])


def load_models_config(
    *,
    config_path: Path | str | None = None,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    ignore_local_config: bool = False,
) -> ModelsConfig:
    """Load ModelsConfig with precedence: explicit path > env > models.local.toml > configs/models.toml."""
    eff_env = os.environ if env is None else env
    resolved_repo = repo_root.resolve() if repo_root is not None else resolve_repo_root(eff_env)

    merged_doc: dict[str, Any] = {
        "models": {k: dict(v) for k, v in _BUILTIN_MODELS_RAW["models"].items()},
        "parity_pairs": list(_BUILTIN_MODELS_RAW["parity_pairs"]),
    }

    env_models_path = eff_env.get("RADIO_BENCH_MODELS_CONFIG", "").strip()
    if config_path is not None:
        explicit = Path(os.path.expanduser(str(config_path)))
        if not explicit.is_file():
            raise FileNotFoundError(f"Explicit models config not found: {explicit}")
        merged_doc = {"models": {}, "parity_pairs": []}
        _merge_models_doc(merged_doc, tomllib.loads(explicit.read_text(encoding="utf-8")))
    elif env_models_path:
        explicit_env = Path(os.path.expanduser(env_models_path))
        if not explicit_env.is_file():
            raise FileNotFoundError(f"RADIO_BENCH_MODELS_CONFIG file not found: {explicit_env}")
        merged_doc = {"models": {}, "parity_pairs": []}
        _merge_models_doc(merged_doc, tomllib.loads(explicit_env.read_text(encoding="utf-8")))
    else:
        committed = resolved_repo / "configs" / "models.toml"
        if committed.is_file():
            merged_doc = {"models": {}, "parity_pairs": []}
            _merge_models_doc(merged_doc, tomllib.loads(committed.read_text(encoding="utf-8")))
        if not ignore_local_config:
            for local_cand in (
                resolved_repo / "configs" / "models.local.toml",
                resolved_repo / "models.local.toml",
            ):
                if local_cand.is_file():
                    _merge_models_doc(
                        merged_doc, tomllib.loads(local_cand.read_text(encoding="utf-8"))
                    )

    parsed_models: dict[str, ModelSpec] = {}
    for key, raw_m in (merged_doc.get("models") or {}).items():
        if not isinstance(raw_m, Mapping):
            continue
        p_raw = raw_m.get("pricing") or {}
        cliff_tok_raw = p_raw.get("cliff_prompt_tokens")
        pricing = ModelPricingConfig(
            input_per_mtok=float(p_raw.get("input_per_mtok", 0.0)),
            output_per_mtok=float(p_raw.get("output_per_mtok", 0.0)),
            cache_read_per_mtok=float(p_raw.get("cache_read_per_mtok", 0.0)),
            cache_write_5m_per_mtok=float(p_raw.get("cache_write_5m_per_mtok", 0.0)),
            cache_write_1h_per_mtok=float(p_raw.get("cache_write_1h_per_mtok", 0.0)),
            cliff_prompt_tokens=int(cliff_tok_raw) if cliff_tok_raw is not None else None,
            cliff_multiplier=float(p_raw.get("cliff_multiplier", 1.0)),
        )
        t_levels = tuple(str(x) for x in (raw_m.get("thinking_levels") or ["default"]))
        aliases = tuple(str(x) for x in (raw_m.get("aliases") or []))
        max_conc_raw = raw_m.get("max_concurrent_trials")
        max_conc: int | None = None
        if max_conc_raw is not None:
            max_conc = int(max_conc_raw)
            if max_conc < 1:
                raise ValueError(
                    f"models.{key}.max_concurrent_trials must be >= 1, got {max_conc_raw!r}"
                )
        raw_api_fmt = str(raw_m["api_format"]).strip() if raw_m.get("api_format") else None
        if raw_api_fmt and raw_api_fmt.lower() == "anthropic":
            norm_api_fmt: str | None = "anthropic_messages"
        else:
            norm_api_fmt = raw_api_fmt
        raw_oc_cfg = raw_m.get("opencode_config")
        oc_cfg = _deep_merge_mapping({}, raw_oc_cfg) if isinstance(raw_oc_cfg, Mapping) else {}
        parsed_models[str(key)] = ModelSpec(
            key=str(key),
            harness=str(raw_m["harness"]),
            model_name=str(raw_m["model_name"]),
            api_format=norm_api_fmt,
            base_url=str(raw_m["base_url"]) if raw_m.get("base_url") else None,
            base_url_env=str(raw_m["base_url_env"]) if raw_m.get("base_url_env") else None,
            api_key_env=str(raw_m["api_key_env"]) if raw_m.get("api_key_env") else None,
            opencode_version=str(raw_m["opencode_version"])
            if raw_m.get("opencode_version")
            else None,
            thinking_levels=t_levels or ("default",),
            context_window_tokens=int(raw_m.get("context_window_tokens", 131072)),
            max_concurrent_trials=max_conc,
            host_gateway_required=bool(raw_m.get("host_gateway_required", False)),
            disable_web_search=bool(raw_m.get("disable_web_search", True)),
            aliases=aliases,
            opencode_config=oc_cfg,
            pricing=pricing,
        )

    pairs: list[ParityPairSpec] = []
    for raw_p in merged_doc.get("parity_pairs") or []:
        if not isinstance(raw_p, Mapping):
            continue
        sm = raw_p.get("small") or {}
        fr = raw_p.get("frontier") or {}
        pairs.append(
            ParityPairSpec(
                pair_id=str(raw_p["pair_id"]),
                small=ParitySideSpec(
                    model=str(sm["model"]),
                    arm=str(sm.get("arm", "11_full")),
                    thinking_level=str(sm.get("thinking_level", "default")),
                ),
                frontier=ParitySideSpec(
                    model=str(fr["model"]),
                    arm=str(fr.get("arm", "00_none")),
                    thinking_level=str(fr.get("thinking_level", "default")),
                ),
            )
        )

    return ModelsConfig(models=parsed_models, parity_pairs=tuple(pairs))


def resolve_git_ref(repo: Path, ref: str) -> tuple[str, str]:
    """Resolve git ref in repo (supporting regular repos, git worktrees, and HEAD/branch/hex).

    Returns `(full_commit_sha, short_sha)` where `short_sha` preserves `ref.lower()` when `ref`
    is a 7..40-character hex prefix (e.g. `00a11ea`) and otherwise uses `full_commit_sha[:7]`.
    """
    if not repo.exists() or not (repo / ".git").exists():
        raise FileNotFoundError(f"Git repository or worktree not found at {repo}")

    clean_ref = ref.strip()
    if not clean_ref:
        raise ValueError("analyst_ref must not be empty")

    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{clean_ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if (
        proc.returncode != 0
        and not _HEX_REF_RE.fullmatch(clean_ref)
        and clean_ref.upper() != "HEAD"
        and not clean_ref.startswith("origin/")
    ):
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"origin/{clean_ref}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    if proc.returncode != 0:
        raise ValueError(
            f"Unable to resolve git ref {clean_ref!r} in {repo}: {proc.stderr.strip()}"
        )

    full_sha = proc.stdout.strip().lower()
    if _HEX_REF_RE.fullmatch(clean_ref) and clean_ref.upper() != "HEAD":
        expected_prefix = clean_ref.lower()
        if not full_sha.startswith(expected_prefix):
            raise ValueError(
                f"Pinned SHA mismatch in {repo}: expected prefix {clean_ref!r}, got {full_sha!r}"
            )
        short_sha = expected_prefix if len(expected_prefix) >= 7 else full_sha[:7]
    else:
        short_sha = full_sha[:7]

    return full_sha, short_sha


def materialize_pack_source(
    pack: PackConfig,
    work_dir: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[Path, str, str]:
    """Resolve or clone `pack.analyst_src` read-only and verify `pack.analyst_ref`.

    Returns `(local_git_repo_path, full_commit_sha, short_sha)`.
    """
    checked_work = reject_slash_tmp(work_dir, label="work_dir")
    eff_repo_root = repo_root.resolve() if repo_root is not None else resolve_repo_root()

    src = pack.analyst_src.strip()
    if is_remote_git_url(src):
        scratch_base = reject_slash_tmp(checked_work / "tmp", label="work_dir/tmp")
        scratch_base.mkdir(parents=True, exist_ok=True)
        clone_dir = scratch_base / f"pack_src_{pack.pack_id}"
        if (clone_dir / ".git").exists():
            origin_proc = subprocess.run(
                ["git", "-C", str(clone_dir), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if origin_proc.returncode != 0 or origin_proc.stdout.strip() != src:
                shutil.rmtree(clone_dir)
            else:
                subprocess.run(
                    ["git", "-C", str(clone_dir), "fetch", "--quiet", "--all", "--tags"],
                    check=True,
                    timeout=180,
                )
        if not (clone_dir / ".git").exists():
            subprocess.run(
                ["git", "clone", "--no-checkout", "--quiet", src, str(clone_dir)],
                check=True,
                timeout=300,
            )
        full_sha, short_sha = resolve_git_ref(clone_dir, pack.analyst_ref)
        return clone_dir, full_sha, short_sha

    expanded = Path(os.path.expanduser(src))
    if str(expanded) == "/tmp" or str(expanded).startswith("/tmp/"):
        raise ValueError(f"packs.{pack.pack_id}.analyst_src must not use /tmp: {src}")
    local_repo = expanded if expanded.is_absolute() else (eff_repo_root / expanded)
    local_repo = reject_slash_tmp(local_repo, label=f"packs.{pack.pack_id}.analyst_src")
    if not local_repo.exists() or not (local_repo / ".git").exists():
        raise FileNotFoundError(
            f"Pack {pack.pack_id!r} analyst_src git repository not found at {local_repo} "
            f"(configure [packs.{pack.pack_id}] analyst_src in bench.local.toml)"
        )
    full_sha, short_sha = resolve_git_ref(local_repo, pack.analyst_ref)
    return local_repo, full_sha, short_sha


def write_casaconfig_local(cfg: BenchConfig) -> Path:
    """Write `<work_dir>/casaconfig_local.py` pointing offline at `cfg.casa_data_dir`."""
    reject_slash_tmp(cfg.work_dir, label="work_dir")
    reject_slash_tmp(cfg.casa_data_dir, label="casa_data_dir")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    casa_data_str = str(cfg.casa_data_dir)
    casa_log_str = str(cfg.tmp_dir / "casa.log")
    content = (
        f"measurespath = {casa_data_str!r}\n"
        f"datapath = [{casa_data_str!r}]\n"
        "data_auto_update = False\n"
        "measures_auto_update = False\n"
        "nologfile = True\n"
        f"logfile = {casa_log_str!r}\n"
    )
    target = cfg.casaconfig_path
    if not target.is_file() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")
    return target


def add_config_cli_args(parser: argparse.ArgumentParser) -> None:
    """Attach canonical `--config`, `--work-dir`, `--data-dir`, and `--runs-dir` CLI options."""
    parser.add_argument("--config", type=Path, default=None, help="Explicit bench.toml path.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override work_dir (highest CLI precedence).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data_dir (highest CLI precedence).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Override runs_dir (highest CLI precedence).",
    )


def config_from_cli_args(args: argparse.Namespace) -> BenchConfig:
    """Resolve BenchConfig honoring `--config`, `--work-dir`, `--data-dir`, and `--runs-dir`."""
    return load_bench_config(
        config_path=getattr(args, "config", None),
        work_dir=getattr(args, "work_dir", None),
        data_dir=getattr(args, "data_dir", None),
        runs_dir=getattr(args, "runs_dir", None),
    )


def prepare_runtime_env(
    cfg: BenchConfig | None = None,
    *,
    base_env: Mapping[str, str] | None = None,
    tmpdir: Path | str | None = None,
    config_path: Path | str | None = None,
    mutate_os_environ: bool = False,
) -> dict[str, str]:
    """Validate non-/tmp directories, create `<work_dir>` scratch subtrees, and return child env."""
    resolved_cfg = (
        cfg if cfg is not None else load_bench_config(config_path=config_path, env=base_env)
    )

    # Validate every configured path strictly before creating any directory.
    for label, path in (
        ("work_dir", resolved_cfg.work_dir),
        ("data_dir", resolved_cfg.data_dir),
        ("runs_dir", resolved_cfg.runs_dir),
        ("casaguides_dir", resolved_cfg.casaguides_dir),
        ("prompts_dir", resolved_cfg.prompts_dir),
        ("casa_data_dir", resolved_cfg.casa_data_dir),
        ("container_tmpdir", resolved_cfg.container_tmpdir),
        ("tmp_dir", resolved_cfg.tmp_dir),
    ):
        reject_slash_tmp(path, label=label)

    eff_base = dict(os.environ if base_env is None else base_env)
    explicit_env_tmp = eff_base.get("RADIO_BENCH_TMPDIR", "").strip()
    if tmpdir is not None:
        resolved_tmp = reject_slash_tmp(tmpdir, label="TMPDIR")
    elif explicit_env_tmp:
        resolved_tmp = reject_slash_tmp(explicit_env_tmp, label="RADIO_BENCH_TMPDIR")
    else:
        resolved_tmp = resolved_cfg.tmp_dir

    pycache_dir = resolved_cfg.work_dir / "pycache"
    for subdir in (
        resolved_tmp,
        resolved_cfg.tmp_dir,
        resolved_cfg.mplconfig_dir,
        resolved_cfg.pytest_dir,
        resolved_cfg.pytest_cache_dir,
        resolved_cfg.docker_config_dir,
        resolved_cfg.container_tmpdir,
        pycache_dir,
    ):
        subdir.mkdir(parents=True, exist_ok=True)

    casa_cfg_path = write_casaconfig_local(resolved_cfg)

    exported_vars: dict[str, str] = {
        "TMPDIR": str(resolved_tmp),
        "PYTHONPYCACHEPREFIX": str(pycache_dir),
        "MPLCONFIGDIR": str(resolved_cfg.mplconfig_dir),
        "CASASITECONFIG": str(casa_cfg_path),
        "DOCKER_CONFIG": str(resolved_cfg.docker_config_dir),
        "DOCKER_HOST": resolved_cfg.docker_host,
        "CONTAINER_TMPDIR": str(resolved_cfg.container_tmpdir),
        "RADIO_BENCH_WORK_DIR": str(resolved_cfg.work_dir),
        "RADIO_BENCH_DATA_DIR": str(resolved_cfg.data_dir),
        "RADIO_BENCH_RUNS_DIR": str(resolved_cfg.runs_dir),
        "RADIO_BENCH_CASAGUIDES_DIR": str(resolved_cfg.casaguides_dir),
        "RADIO_BENCH_PROMPTS_DIR": str(resolved_cfg.prompts_dir),
        "RADIO_BENCH_CASA_DATA": str(resolved_cfg.casa_data_dir),
    }
    if config_path is not None and str(config_path).strip():
        exported_vars["RADIO_BENCH_CONFIG"] = str(
            Path(os.path.expanduser(str(config_path))).resolve()
        )

    eff_base.update(exported_vars)
    if mutate_os_environ:
        os.environ.update(exported_vars)

    return eff_base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect resolved radio-bench host and pack configuration."
    )
    add_config_cli_args(parser)
    parser.add_argument("--pack", type=str, default=None, help="Pack ID to inspect/materialize.")
    parser.add_argument(
        "--materialize-pack",
        action="store_true",
        help="Resolve or clone the selected pack's git source and print commit SHA.",
    )
    parser.add_argument(
        "--shell-env",
        action="store_true",
        help="Emit shell-safe variable assignments for images/build_images.sh.",
    )
    args = parser.parse_args(argv)

    cfg = config_from_cli_args(args)
    pack = cfg.get_pack(args.pack)

    if args.shell_env:
        env = prepare_runtime_env(cfg, config_path=args.config)
        pairs: dict[str, str] = {
            "REPO_ROOT": str(cfg.repo_root),
            "WORK_DIR": str(cfg.work_dir),
            "DATA_DIR": str(cfg.data_dir),
            "RUNS_DIR": str(cfg.runs_dir),
            "HOST_CASA_DATA": str(cfg.casa_data_dir),
            "TMPDIR": env["TMPDIR"],
            "CONTAINER_TMPDIR": env["CONTAINER_TMPDIR"],
            "MPLCONFIGDIR": env["MPLCONFIGDIR"],
            "CASASITECONFIG": env["CASASITECONFIG"],
            "DOCKER_CONFIG": env["DOCKER_CONFIG"],
            "DOCKER_HOST": env["DOCKER_HOST"],
            "DEFAULT_PACK": cfg.default_pack,
            "PACK_ID": pack.pack_id,
            "SIDECAR_IMAGE": pack.sidecar_image,
        }
        if args.materialize_pack:
            repo_path, full_sha, short_sha = materialize_pack_source(
                pack, cfg.work_dir, repo_root=cfg.repo_root
            )
            pairs["RADIO_ANALYST_REPO"] = str(repo_path)
            pairs["RADIO_ANALYST_FULL_SHA"] = full_sha
            pairs["RADIO_ANALYST_SHA"] = short_sha
        for k, v in pairs.items():
            print(f"{k}={shlex.quote(v)}")
        return 0

    payload: dict[str, Any] = {
        "repo_root": str(cfg.repo_root),
        "work_dir": str(cfg.work_dir),
        "data_dir": str(cfg.data_dir),
        "runs_dir": str(cfg.runs_dir),
        "casaguides_dir": str(cfg.casaguides_dir),
        "prompts_dir": str(cfg.prompts_dir),
        "casa_data_dir": str(cfg.casa_data_dir),
        "docker_host": cfg.docker_host,
        "container_tmpdir": str(cfg.container_tmpdir),
        "default_pack": cfg.default_pack,
        "selected_pack": {
            "pack_id": pack.pack_id,
            "description": pack.description,
            "analyst_src": pack.analyst_src,
            "analyst_ref": pack.analyst_ref,
            "sidecar_image": pack.sidecar_image,
            "mcp_ports": pack.mcp_ports,
        },
    }
    if args.materialize_pack:
        repo_path, full_sha, short_sha = materialize_pack_source(
            pack, cfg.work_dir, repo_root=cfg.repo_root
        )
        payload["materialized_pack"] = {
            "repo_path": str(repo_path),
            "full_sha": full_sha,
            "short_sha": short_sha,
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
