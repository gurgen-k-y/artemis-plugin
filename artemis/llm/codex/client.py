"""Codex app-server JSONL transport and local login discovery."""

from __future__ import annotations

import asyncio
import atexit
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import weakref

from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class CodexAppServerError(RuntimeError):
    pass


def find_codex_binary() -> str | None:
    configured = os.environ.get("ARTEMIS_CODEX_BIN", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return str(path) if path.is_file() else None
    return shutil.which("codex.exe" if os.name == "nt" else "codex")


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def codex_client_status() -> tuple[bool, str]:
    binary = find_codex_binary()
    if not binary:
        return False, (
            "Codex CLI was not found. Install it, then run `codex login` or "
            "`codex login --device-auth` on a headless machine."
        )
    try:
        result = subprocess.run(
            [binary, "login", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not query Codex login status: {exc}"
    output = " ".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode == 0 and "logged in" in output.lower():
        return True, output or "Logged in"
    return False, output or (
        "Codex is installed but is not signed in. Run `codex login` or `codex login --device-auth`."
    )


class CodexAppServerClient:
    def __init__(self, binary: str) -> None:
        self.binary = binary
        self.process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._thread_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        async with self._start_lock:
            if self.process and self.process.returncode is None:
                return
            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.binary,
                    "app-server",
                    "--stdio",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=_creation_flags(),
                )
            except OSError as exc:
                raise CodexAppServerError(f"Could not start Codex app-server: {exc}") from exc
            self._reader_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())
            try:
                await self.request(
                    "initialize",
                    {
                        "clientInfo": {"name": "artemis", "title": "Artemis", "version": "1.0"},
                        "capabilities": {"experimentalApi": False},
                    },
                    start=False,
                )
                await self.notify("initialized", {}, start=False)
            except Exception:
                await self.close()
                raise

    async def request(self, method: str, params: dict[str, Any], *, start: bool = True) -> Any:
        if start:
            await self.start()
        if not self.process or self.process.returncode is not None or self.process.stdin is None:
            raise CodexAppServerError("Codex app-server is not running")
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write({"id": request_id, "method": method, "params": params})
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any], *, start: bool = True) -> None:
        if start:
            await self.start()
        await self._write({"method": method, "params": params})

    async def _write(self, payload: dict[str, Any]) -> None:
        if not self.process or self.process.returncode is not None or self.process.stdin is None:
            raise CodexAppServerError("Codex app-server closed its stdio connection")
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        async with self._write_lock:
            self.process.stdin.write(data)
            await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while line := await self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                request_id = message.get("id")
                method = message.get("method")
                if request_id is not None and not method:
                    future = self._pending.get(request_id)
                    if future and not future.done():
                        if "error" in message:
                            future.set_exception(CodexAppServerError(str(message["error"])))
                        else:
                            future.set_result(message.get("result"))
                    continue
                if request_id is not None and method:
                    await self._write(
                        {
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Artemis does not expose Codex-side tools or approvals",
                            },
                        }
                    )
                    continue
                params = message.get("params") or {}
                queue = self._thread_queues.get(params.get("threadId"))
                if queue is not None:
                    queue.put_nowait(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"Codex app-server reader stopped: {exc}", exc_info=True)
        finally:
            error = CodexAppServerError("Codex app-server exited before completion")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)

    async def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        try:
            while line := await self.process.stderr.readline():
                logger.debug(f"codex app-server: {line.decode(errors='replace').rstrip()}")
        except asyncio.CancelledError:
            raise

    async def complete(
        self,
        *,
        model: str,
        effort: str | None,
        cwd: str,
        instructions: str,
        developer_instructions: str,
        inputs: list[dict[str, Any]],
        output_schema: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        await self.start()
        params: dict[str, Any] = {
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "cwd": cwd,
            "ephemeral": True,
            "serviceName": "artemis",
            "baseInstructions": instructions,
            "developerInstructions": developer_instructions,
        }
        if model.lower() not in {"auto", "default"}:
            params["model"] = model
        started = await self.request("thread/start", params)
        thread_id = ((started or {}).get("thread") or {}).get("id")
        if not thread_id:
            raise CodexAppServerError("Codex app-server did not return a thread id")
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._thread_queues[thread_id] = queue
        usage: dict[str, Any] | None = None
        turn_id: str | None = None
        try:
            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": inputs,
                "outputSchema": output_schema,
            }
            if effort and effort != "none":
                turn_params["effort"] = effort
            turn_started = await self.request("turn/start", turn_params)
            turn_id = ((turn_started or {}).get("turn") or {}).get("id")

            async def wait_for_turn() -> dict[str, Any]:
                nonlocal usage
                while True:
                    event = await queue.get()
                    event_params = event.get("params") or {}
                    if event.get("method") == "thread/tokenUsage/updated":
                        usage = event_params.get("tokenUsage")
                    if event.get("method") == "turn/completed":
                        return event_params.get("turn") or {}

            turn = await asyncio.wait_for(wait_for_turn(), timeout_seconds)
        except TimeoutError as exc:
            await self._interrupt_turn(thread_id, turn_id)
            raise CodexAppServerError(
                f"Codex model call timed out after {timeout_seconds:g} seconds"
            ) from exc
        finally:
            self._thread_queues.pop(thread_id, None)
        if turn.get("status") != "completed":
            raise CodexAppServerError(
                f"Codex turn failed: {turn.get('error') or turn.get('status')}"
            )
        text = "\n".join(
            str(item["text"])
            for item in turn.get("items") or []
            if item.get("type") == "agentMessage" and item.get("text")
        )
        if not text:
            raise CodexAppServerError("Codex turn completed without an assistant message")
        return {
            "text": text,
            "thread_id": thread_id,
            "model": (started or {}).get("model") or model,
            "usage": usage,
        }

    async def _interrupt_turn(self, thread_id: str, turn_id: str | None) -> None:
        if not turn_id:
            return
        try:
            await asyncio.wait_for(
                self.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id}), 5
            )
        except Exception as exc:
            logger.debug(f"Could not interrupt Codex turn {turn_id}: {exc}")

    async def close(self) -> None:
        process = self.process
        self.process = None
        tasks = [task for task in (self._reader_task, self._stderr_task) if task]
        for task in tasks:
            if not task.done():
                task.cancel()
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                process.kill()
                await process.wait()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def terminate(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()


_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, CodexAppServerClient] = (
    weakref.WeakKeyDictionary()
)


def client_for_running_loop() -> CodexAppServerClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is not None:
        return client
    binary = find_codex_binary()
    if not binary:
        raise CodexAppServerError(
            "Codex CLI was not found. Install it, then run `codex login` or "
            "`codex login --device-auth`"
        )
    client = CodexAppServerClient(binary)
    _clients[loop] = client
    return client


async def close_current_client() -> None:
    client = _clients.pop(asyncio.get_running_loop(), None)
    if client is not None:
        await client.close()


@atexit.register
def _terminate_clients() -> None:
    for client in list(_clients.values()):
        client.terminate()
