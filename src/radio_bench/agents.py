"""Custom Harbor agent subclasses for radio-bench runtime guards and safe auto-retrigger."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

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


class RadioBenchOpenCode(OpenCode):
    """OpenCode agent wrapper with decoder-collapse detection and accurate provider retry mapping.

    Preserves ``OpenCode.name() == "opencode"`` so ``result.json["agent_info"]["name"]``
    and downstream analysis/pricing remain identical to stock Harbor OpenCode.
    """

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        caught_exc: NonZeroAgentExitCodeError | None = None
        try:
            await super().run(instruction, environment, context)
        except NonZeroAgentExitCodeError as exc:
            caught_exc = exc

        # With auto_mounting=False (Podman default), /logs/agent/opencode.txt is not
        # synced onto the host until after run() returns (and --resume runs may leave
        # a stale host copy). Refresh it when unmounted, preserving any existing host
        # copy if download_file fails.
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
