"""Safe one-shot transport for the signed-in Claude Code CLI."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


class ClaudeCodeError(RuntimeError):
    pass


def find_claude_binary() -> str | None:
    override = os.environ.get("ARTEMIS_CLAUDE_BIN")
    if override:
        path = Path(override).expanduser()
        return str(path) if path.is_file() else None
    return shutil.which("claude")


def claude_client_status() -> tuple[bool, str]:
    binary = find_claude_binary()
    if binary is None:
        return (
            False,
            "Claude Code was not found. Install it, then run `claude auth login`, or run "
            "`claude setup-token` and export `CLAUDE_CODE_OAUTH_TOKEN`.",
        )
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return True, "Claude Code setup token is available through CLAUDE_CODE_OAUTH_TOKEN."

    import subprocess

    try:
        result = subprocess.run(
            [binary, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not query Claude Code login status: {exc}"
    try:
        status = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        status = {}
    if result.returncode == 0 and status.get("loggedIn") is True:
        method = str(status.get("authMethod") or "local login")
        return True, f"Claude Code is signed in ({method})."
    return (
        False,
        "Claude Code is installed but is not signed in. Run `claude auth login`, or run "
        "`claude setup-token` and export `CLAUDE_CODE_OAUTH_TOKEN`.",
    )


class ClaudeCodeClient:
    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or find_claude_binary()
        if self.binary is None:
            raise ClaudeCodeError(
                "Claude Code was not found. Install it, then run `claude auth login`, or run "
                "`claude setup-token` and export `CLAUDE_CODE_OAUTH_TOKEN`."
            )

    async def complete(
        self,
        *,
        model: str,
        effort: str | None,
        prompt: str,
        output_schema: dict[str, Any],
        image_paths: list[Path],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        args = [
            self.binary,
            "-p",
            "--safe-mode",
            "--restricted",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--output-format",
            "json",
            "--model",
            model,
            "--json-schema",
            json.dumps(output_schema, separators=(",", ":")),
            "--tools",
            "Read" if image_paths else "",
        ]
        if effort and effort != "none":
            args.extend(["--effort", effort])

        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(image_paths[0].parent) if image_paths else tempfile.gettempdir(),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ClaudeCodeError(f"Could not start Claude Code: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=timeout_seconds
            )
        except TimeoutError as exc:
            await _stop_process(process)
            raise ClaudeCodeError(
                f"Claude Code model call timed out after {timeout_seconds:g} seconds"
            ) from exc
        except asyncio.CancelledError:
            await _stop_process(process)
            raise

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            if token:
                detail = detail.replace(token, "[redacted]")
            raise ClaudeCodeError(detail or f"Claude Code exited with status {process.returncode}")
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ClaudeCodeError("Claude Code returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise ClaudeCodeError("Claude Code returned an unexpected JSON response")

        structured = payload.get("structured_output")
        if isinstance(structured, dict):
            text = json.dumps(structured)
        else:
            result = payload.get("result")
            text = result if isinstance(result, str) else json.dumps(result or {})
        return {
            "text": text,
            "model": payload.get("model") or model,
            "usage": payload.get("usage") or {},
        }


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except TimeoutError:
        process.kill()
        await process.wait()
