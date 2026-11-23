"""Custom Harbor agent subclasses for radio-bench runtime guards and safe auto-retrigger."""

from __future__ import annotations

import contextlib
import re
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
    """OpenCode agent wrapper with fast pre-installed check, decoder-collapse detection, and retry mapping.

    Preserves ``OpenCode.name() == "opencode"`` so ``result.json["agent_info"]["name"]``
    and downstream analysis/pricing remain identical to stock Harbor OpenCode.
    """

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
