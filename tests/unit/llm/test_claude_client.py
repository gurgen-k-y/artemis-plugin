import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

import artemis.llm.claude.client as claude_client
from artemis.llm.claude.client import ClaudeCodeClient, ClaudeCodeError


class FakeProcess:
    def __init__(self, stdout: bytes = b"{}", stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.input: bytes | None = None
        self.terminated = False
        self.killed = False

    async def communicate(self, value: bytes):
        self.input = value
        return self.stdout, self.stderr

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_complete_uses_safe_structured_mode_and_setup_token(monkeypatch):
    process = FakeProcess(
        json.dumps(
            {
                "structured_output": {"content": "ready"},
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        ).encode()
    )
    call = {}

    async def create(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return process

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "secret-setup-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key-must-not-win")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    result = await ClaudeCodeClient("/usr/bin/claude").complete(
        model="sonnet",
        effort="medium",
        prompt="hello",
        output_schema={"type": "object"},
        image_paths=[],
        timeout_seconds=2,
    )

    assert json.loads(result["text"]) == {"content": "ready"}
    assert process.input == b"hello"
    assert "--safe-mode" in call["args"]
    assert "--restricted" in call["args"]
    assert "--no-session-persistence" in call["args"]
    assert call["args"][call["args"].index("--tools") + 1] == ""
    assert call["kwargs"]["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "secret-setup-token"
    assert "ANTHROPIC_API_KEY" not in call["kwargs"]["env"]
    assert "secret-setup-token" not in " ".join(call["args"])


@pytest.mark.asyncio
async def test_complete_allows_only_read_when_images_are_present(monkeypatch, tmp_path):
    process = FakeProcess(b'{"result":"{\\"content\\":\\"seen\\"}"}')
    call = {}

    async def create(*args, **kwargs):
        call["args"] = args
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    image = tmp_path / "screen.png"
    image.write_bytes(b"image")

    await ClaudeCodeClient("claude").complete(
        model="haiku",
        effort="low",
        prompt="inspect",
        output_schema={"type": "object"},
        image_paths=[image],
        timeout_seconds=2,
    )

    assert call["args"][call["args"].index("--tools") + 1] == "Read"


@pytest.mark.asyncio
async def test_complete_rejects_malformed_output(monkeypatch):
    async def create(*args, **kwargs):
        return FakeProcess(b"not-json")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    with pytest.raises(ClaudeCodeError, match="malformed JSON"):
        await ClaudeCodeClient("claude").complete(
            model="sonnet",
            effort=None,
            prompt="hello",
            output_schema={"type": "object"},
            image_paths=[],
            timeout_seconds=2,
        )


@pytest.mark.asyncio
async def test_complete_reports_process_start_failure(monkeypatch):
    async def create(*args, **kwargs):
        raise OSError("executable disappeared")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    with pytest.raises(ClaudeCodeError, match="Could not start Claude Code"):
        await ClaudeCodeClient("claude").complete(
            model="sonnet",
            effort=None,
            prompt="hello",
            output_schema={"type": "object"},
            image_paths=[],
            timeout_seconds=2,
        )


@pytest.mark.asyncio
async def test_complete_redacts_setup_token_from_cli_errors(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "setup-secret")

    async def create(*args, **kwargs):
        return FakeProcess(stderr=b"invalid setup-secret", returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    with pytest.raises(ClaudeCodeError, match=r"invalid \[redacted\]") as error:
        await ClaudeCodeClient("claude").complete(
            model="sonnet",
            effort=None,
            prompt="hello",
            output_schema={"type": "object"},
            image_paths=[],
            timeout_seconds=2,
        )

    assert "setup-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_complete_timeout_terminates_process(monkeypatch):
    process = FakeProcess()

    async def communicate(_value):
        await asyncio.sleep(10)

    process.communicate = communicate

    async def create(*args, **kwargs):
        process.returncode = None
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    with pytest.raises(ClaudeCodeError, match="timed out"):
        await ClaudeCodeClient("claude").complete(
            model="sonnet",
            effort=None,
            prompt="hello",
            output_schema={"type": "object"},
            image_paths=[],
            timeout_seconds=0.001,
        )

    assert process.terminated


@pytest.mark.asyncio
async def test_complete_cancellation_terminates_process(monkeypatch):
    process = FakeProcess()

    async def communicate(_value):
        await asyncio.sleep(10)

    process.communicate = communicate

    async def create(*args, **kwargs):
        process.returncode = None
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    task = asyncio.create_task(
        ClaudeCodeClient("claude").complete(
            model="sonnet",
            effort=None,
            prompt="hello",
            output_schema={"type": "object"},
            image_paths=[],
            timeout_seconds=10,
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminated


def test_status_accepts_setup_token_without_auth_probe(monkeypatch):
    monkeypatch.setattr(claude_client, "find_claude_binary", lambda: "/usr/bin/claude")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "secret")

    ready, detail = claude_client.claude_client_status()

    assert ready is True
    assert "secret" not in detail
    assert "CLAUDE_CODE_OAUTH_TOKEN" in detail


def test_status_accepts_local_oauth_login(monkeypatch):
    monkeypatch.setattr(claude_client, "find_claude_binary", lambda: "/usr/bin/claude")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"loggedIn":true,"authMethod":"oauth_token"}',
        ),
    )

    ready, detail = claude_client.claude_client_status()

    assert ready is True
    assert "oauth_token" in detail


def test_missing_claude_guides_login_and_setup_token(monkeypatch):
    monkeypatch.setattr(claude_client, "find_claude_binary", lambda: None)

    ready, detail = claude_client.claude_client_status()

    assert ready is False
    assert "claude auth login" in detail
    assert "claude setup-token" in detail


def test_old_claude_version_is_not_ready(monkeypatch):
    import artemis.llm.claude.client as client

    monkeypatch.setattr(client, "find_claude_binary", lambda: "claude")
    monkeypatch.setattr(client, "_installed_version", lambda _binary: (2, 1, 200))

    ready, detail = client.claude_client_status()

    assert ready is False
    assert "too old" in detail
