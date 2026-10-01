import pytest

import artemis.llm.codex.client as codex_client
from artemis.llm.codex.client import CodexAppServerClient, CodexAppServerError


class StubClient(CodexAppServerClient):
    def __init__(self, events: list[dict]):
        super().__init__("codex")
        self.events = events
        self.requests: list[tuple[str, dict]] = []

    async def start(self) -> None:
        return None

    async def request(self, method: str, params: dict, *, start: bool = True):
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thread-1"}, "model": "gpt-test"}
        if method == "turn/start":
            queue = self._thread_queues[params["threadId"]]
            for event in self.events:
                queue.put_nowait(event)
            return {"turn": {"id": "turn-1"}}
        raise AssertionError(method)


@pytest.mark.asyncio
async def test_completion_collects_turn_text_and_usage():
    client = StubClient(
        [
            {
                "method": "thread/tokenUsage/updated",
                "params": {"threadId": "thread-1", "tokenUsage": {"last": {"totalTokens": 5}}},
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": '{"content":"ready"}'}],
                    },
                },
            },
        ]
    )

    result = await client.complete(
        model="gpt-test",
        effort="low",
        cwd="/tmp",
        instructions="system",
        developer_instructions="contract",
        inputs=[{"type": "text", "text": "hello"}],
        output_schema={"type": "object"},
        timeout_seconds=1,
    )

    assert result["text"] == '{"content":"ready"}'
    assert result["usage"] == {"last": {"totalTokens": 5}}
    assert client.requests[0][1]["sandbox"] == "read-only"
    assert client.requests[0][1]["approvalPolicy"] == "never"
    assert "thread-1" not in client._thread_queues


@pytest.mark.asyncio
async def test_completion_timeout_cleans_thread_queue():
    client = StubClient([])

    with pytest.raises(CodexAppServerError, match="timed out"):
        await client.complete(
            model="default",
            effort=None,
            cwd="/tmp",
            instructions="system",
            developer_instructions="contract",
            inputs=[{"type": "text", "text": "hello"}],
            output_schema={"type": "object"},
            timeout_seconds=0.001,
        )

    assert "thread-1" not in client._thread_queues


def test_missing_codex_guides_device_auth(monkeypatch):
    monkeypatch.setattr(codex_client, "find_codex_binary", lambda: None)

    ready, detail = codex_client.codex_client_status()

    assert ready is False
    assert "codex login --device-auth" in detail
