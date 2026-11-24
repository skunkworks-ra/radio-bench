"""Custom Harbor agent subclasses for radio-bench runtime guards and safe auto-retrigger."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harbor.agents.installed.base import (
    ApiError,
    ApiInternalServerError,
    ApiRateLimitError,
    ApiResponseStalledError,
    NonZeroAgentExitCodeError,
)
from harbor.agents.installed.opencode import OpenCode

from radio_bench.analysis._trace import extract_agent_runtime_diagnostics

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

_DEFAULT_RATE_LIMIT_MAX_RESUMES = 6
_DEFAULT_RATE_LIMIT_BASE_WAIT_SEC = 30.0
_DEFAULT_RATE_LIMIT_MAX_WAIT_SEC = 300.0
_RATE_LIMIT_RESUME_PROMPT = (
    "The previous upstream API request hit an HTTP 429 rate limit. "
    "Resume immediately from where you stopped and finish the remaining steps "
    "without restarting already completed work."
)


def _extract_last_session_and_retry_after(log_path: Path | None) -> tuple[str | None, float]:
    """Return `(last_session_id, retry_after_sec)` parsed from OpenCode JSONL `opencode.txt`."""
    if log_path is None or not log_path.is_file():
        return None, 0.0
    session_id: str | None = None
    retry_after_sec = 0.0
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, 0.0
    for raw_line in lines:
        s = raw_line.strip()
        if not s.startswith("{"):
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        sid = ev.get("sessionID")
        if isinstance(sid, str) and sid.strip():
            session_id = sid.strip()
        if str(ev.get("type") or "").strip().lower() == "error":
            err = ev.get("error") if isinstance(ev.get("error"), dict) else {}
            data = err.get("data") if isinstance(err.get("data"), dict) else {}
            headers = data.get("responseHeaders")
            if isinstance(headers, dict):
                for hk, hv in headers.items():
                    if str(hk).strip().lower() == "retry-after":
                        try:
                            val = float(str(hv).strip())
                            if val > 0.0:
                                retry_after_sec = val
                        except ValueError:
                            pass
    return session_id, retry_after_sec


class RadioBenchOpenCode(OpenCode):
    """OpenCode agent wrapper with fast pre-installed check, in-place 429 resume, and retry mapping.

    Preserves ``OpenCode.name() == "opencode"`` so ``result.json["agent_info"]["name"]``
    and downstream analysis/pricing remain identical to stock Harbor OpenCode.
    """

    def __init__(
        self,
        *args: Any,
        rate_limit_max_resumes: int = _DEFAULT_RATE_LIMIT_MAX_RESUMES,
        rate_limit_base_wait_sec: float = _DEFAULT_RATE_LIMIT_BASE_WAIT_SEC,
        rate_limit_max_wait_sec: float = _DEFAULT_RATE_LIMIT_MAX_WAIT_SEC,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_max_resumes = max(0, int(rate_limit_max_resumes))
        self._rate_limit_base_wait_sec = max(0.0, float(rate_limit_base_wait_sec))
        self._rate_limit_max_wait_sec = max(
            self._rate_limit_base_wait_sec, float(rate_limit_max_wait_sec)
        )

    async def _sync_host_log(self, environment: BaseEnvironment) -> Path:
        """Ensure host `opencode.txt` reflects the latest container output when unmounted."""
        host_log = self.logs_dir / self._OUTPUT_FILENAME
        is_path_mounted_fn = getattr(environment, "is_path_mounted", None)
        is_mounted = bool(
            callable(is_path_mounted_fn)
            and is_path_mounted_fn(self.environment_logs_dir, self.logs_dir)
        )
        if not is_mounted:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            remote_log = (self.environment_logs_dir / self._OUTPUT_FILENAME).as_posix()
            if host_log.is_file():
                refresh_log = self.logs_dir / f".{self._OUTPUT_FILENAME}.refresh"
                try:
                    with contextlib.suppress(Exception):
                        await environment.download_file(remote_log, refresh_log)
                        if refresh_log.is_file():
                            refresh_log.replace(host_log)
                finally:
                    with contextlib.suppress(OSError):
                        refresh_log.unlink(missing_ok=True)
            else:
                with contextlib.suppress(Exception):
                    await environment.download_file(remote_log, host_log)
        return host_log

    async def _resume_session_in_place(
        self,
        environment: BaseEnvironment,
        session_id: str,
    ) -> Exception | None:
        """Resume an interrupted OpenCode session in the live container, appending to `opencode.txt`."""
        env = dict(self.model_connection.env)
        env["OPENCODE_FAKE_VCS"] = "git"
        env["XDG_DATA_HOME"] = (self.environment_logs_dir / "opencode" / "xdg-data").as_posix()
        env["XDG_STATE_HOME"] = (self.environment_logs_dir / "opencode" / "xdg-state").as_posix()

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        remote_log = (self.environment_logs_dir / self._OUTPUT_FILENAME).as_posix()
        cmd = (
            "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
            f"opencode --model={shlex.quote(self._execution_model_name())} run --format=json "
            f"--session {shlex.quote(session_id)} {cli_flags_arg}--thinking "
            f"--dangerously-skip-permissions -- {shlex.quote(_RATE_LIMIT_RESUME_PROMPT)} "
            f"2>&1 </dev/null | stdbuf -oL tee -a {shlex.quote(remote_log)}"
        )
        try:
            await self.exec_as_agent(environment, command=cmd, env=env)
        except (NonZeroAgentExitCodeError, ApiError) as exc:
            return exc
        return None

    async def install(self, environment: BaseEnvironment) -> None:
        """Skip slow runtime nvm/npm installation when matching ``opencode`` and ``stdbuf`` exist."""
        check_cmd = (
            "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
            "command -v opencode >/dev/null 2>&1 && "
            "command -v stdbuf >/dev/null 2>&1 && "
            "opencode --version"
        )
        try:
            res = await self.exec_as_agent(environment, command=check_cmd)
        except (NonZeroAgentExitCodeError, RuntimeError) as exc:
            self.logger.debug(
                "Pre-installed opencode check failed (%s); falling back to Harbor install.",
                exc,
            )
            await super().install(environment)
            return

        stdout = getattr(res, "stdout", None) or ""
        non_empty = [line.strip() for line in stdout.splitlines() if line.strip()]
        installed_version = non_empty[-1] if non_empty else ""
        expected_version = str(self._version).strip().lstrip("v") if self._version else ""

        version_matched = bool(installed_version) and (
            not expected_version
            or bool(re.search(rf"(?:^|[\s/v]){re.escape(expected_version)}$", installed_version))
        )
        if version_matched:
            self.logger.info(
                "Using pre-installed opencode %s (expected %s); skipping runtime install.",
                installed_version,
                expected_version or "any",
            )
            return

        self.logger.info(
            "Pre-installed opencode version mismatch (found %r, expected %r); "
            "falling back to Harbor install.",
            installed_version,
            expected_version,
        )
        await super().install(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        caught_exc: Exception | None = None
        try:
            await super().run(instruction, environment, context)
        except (NonZeroAgentExitCodeError, ApiError) as exc:
            caught_exc = exc

        # With auto_mounting=False (Podman default), /logs/agent/opencode.txt is not
        # synced onto the host until after run() returns (and --resume runs may leave
        # a stale host copy). Refresh it when unmounted, preserving any existing host
        # copy if download_file fails.
        host_log = await self._sync_host_log(environment)

        diag = extract_agent_runtime_diagnostics(
            host_log if host_log.is_file() else None,
            exc_type=type(caught_exc).__name__ if caught_exc is not None else None,
            exc_msg=str(caught_exc) if caught_exc is not None else None,
        )

        # Wait out transient HTTP 429 rate limits by resuming the existing OpenCode
        # session inside the live container instead of tearing down and restarting cold.
        rate_limit_resumes = 0
        while (
            diag["agent_exit_status"] == "provider_api_error"
            and diag["provider_error_kind"] == "rate_limit_429"
            and rate_limit_resumes < self._rate_limit_max_resumes
        ):
            session_id, retry_after_sec = _extract_last_session_and_retry_after(host_log)
            if not session_id:
                break
            rate_limit_resumes += 1
            backoff_sec = min(
                self._rate_limit_base_wait_sec * (2 ** (rate_limit_resumes - 1)),
                self._rate_limit_max_wait_sec,
            )
            wait_sec = max(backoff_sec, retry_after_sec)
            self.logger.warning(
                "Upstream HTTP 429 rate limit on session %s (in-place resume %d/%d); "
                "waiting %.1fs before resuming existing OpenCode session.",
                session_id,
                rate_limit_resumes,
                self._rate_limit_max_resumes,
                wait_sec,
            )
            if wait_sec > 0.0:
                await asyncio.sleep(wait_sec)
            caught_exc = await self._resume_session_in_place(environment, session_id)
            host_log = await self._sync_host_log(environment)
            diag = extract_agent_runtime_diagnostics(
                host_log if host_log.is_file() else None,
                exc_type=type(caught_exc).__name__ if caught_exc is not None else None,
                exc_msg=str(caught_exc) if caught_exc is not None else None,
            )
        if diag["decoder_collapse"]:
            msg = (
                "Upstream model decoder collapse detected "
                f"({diag['decoder_collapse_detail']}) in {host_log.name}"
            )
            if caught_exc is not None:
                raise ApiInternalServerError(msg) from caught_exc
            raise ApiInternalServerError(msg)

        if diag["agent_exit_status"] == "provider_api_error":
            p_kind = diag["provider_error_kind"]
            p_code = diag["provider_status_code"]
            if p_kind == "upstream_connection_or_5xx":
                raise ApiInternalServerError(
                    f"Upstream provider 5xx/connection error (HTTP {p_code})"
                ) from caught_exc
            if p_kind == "upstream_timeout_504":
                raise ApiResponseStalledError(
                    f"Upstream provider gateway timeout (HTTP {p_code})"
                ) from caught_exc
            if p_kind == "rate_limit_429":
                raise ApiRateLimitError(
                    f"Upstream provider rate limit (HTTP {p_code})"
                ) from caught_exc
            if p_kind == "tool_call_parse_400":
                raise NonZeroAgentExitCodeError(
                    "Malformed tool call rejected by provider (HTTP 400)"
                ) from caught_exc
            code_sfx = f" (HTTP {p_code})" if p_code is not None else ""
            raise NonZeroAgentExitCodeError(
                f"Fatal upstream provider API error{code_sfx}"
            ) from caught_exc

        if caught_exc is not None and not isinstance(caught_exc, ApiError):
            if diag["benign_epipe_exit"]:
                self.logger.debug(
                    "Ignoring benign OpenCode shutdown EPIPE exit after clean completion: %s",
                    caught_exc,
                )
                return
            if (
                diag["agent_exit_status"] == "ok"
                and "opencode emitted error event(s):" in str(caught_exc).lower()
            ):
                self.logger.debug(
                    "Ignoring recovered mid-session OpenCode error event exit after clean completion: %s",
                    caught_exc,
                )
                return

        if caught_exc is not None:
            raise caught_exc
