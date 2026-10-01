import base64
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import pytest

import artemis.llm.codex.model as codex_model
from artemis.llm.codex.model import CodexAppServerChatModel
from artemis.llm.router import ModelEndpoint, ModelFactory


class FakeClient:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []
        self.image_paths: list[Path] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        self.image_paths = [
            Path(item["path"]) for item in kwargs["inputs"] if item["type"] == "localImage"
        ]
        assert all(path.exists() for path in self.image_paths)
        return self.response


@pytest.mark.asyncio
async def test_codex_completion_uses_structured_app_server_contract(monkeypatch):
    fake = FakeClient(
        {
            "text": json.dumps({"content": "ready"}),
            "thread_id": "thread-1",
            "model": "gpt-test",
            "usage": {"last": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}},
        }
    )
    monkeypatch.setattr(codex_model, "client_for_running_loop", lambda: fake)
    model = CodexAppServerChatModel(model_name="gpt-test", reasoning_effort="low")

    result = await model.ainvoke(
        [SystemMessage(content="Be exact"), HumanMessage(content="Return ready")]
    )

    assert result.content == "ready"
    assert result.response_metadata["provider"] == "codex"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    call = fake.calls[0]
    assert call["instructions"] == "Be exact"
    assert call["output_schema"]["required"] == ["content"]
    assert call["inputs"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_codex_returns_langchain_tool_call(monkeypatch):
    fake = FakeClient(
        {
            "text": json.dumps(
                {
                    "kind": "tool_call",
                    "content": "",
                    "tool_name": "add_numbers",
                    "tool_arguments_json": json.dumps({"a": 19, "b": 23}),
                }
            ),
            "thread_id": "thread-1",
            "model": "gpt-test",
            "usage": None,
        }
    )
    monkeypatch.setattr(codex_model, "client_for_running_loop", lambda: fake)

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    model = CodexAppServerChatModel(model_name="gpt-test").bind_tools(
        [add_numbers], tool_choice="required"
    )
    result = await model.ainvoke("Add numbers")

    assert result.tool_calls[0]["name"] == "add_numbers"
    assert result.tool_calls[0]["args"] == {"a": 19, "b": 23}
    assert fake.calls[0]["output_schema"]["properties"]["kind"]["enum"] == ["tool_call"]


@pytest.mark.asyncio
async def test_codex_stages_and_cleans_image(monkeypatch):
    fake = FakeClient(
        {
            "text": json.dumps({"content": "seen"}),
            "thread_id": "thread-1",
            "model": "gpt-test",
            "usage": None,
        }
    )
    monkeypatch.setattr(codex_model, "client_for_running_loop", lambda: fake)
    model = CodexAppServerChatModel(model_name="gpt-test")
    image = base64.b64encode(b"image-bytes").decode()

    result = await model.ainvoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Inspect"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ]
            )
        ]
    )

    assert result.content == "seen"
    assert fake.image_paths
    assert all(not path.exists() for path in fake.image_paths)


def test_factory_builds_codex_without_api_key():
    model = ModelFactory.create_model(ModelEndpoint(provider="codex", model_name="gpt-test"))

    assert isinstance(model, CodexAppServerChatModel)
