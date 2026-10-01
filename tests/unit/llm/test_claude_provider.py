import base64
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import pytest

import artemis.llm.claude.model as claude_model
from artemis.llm.claude.model import ClaudeCodeChatModel
from artemis.llm.router import ModelEndpoint, ModelFactory


class FakeClient:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []
        self.image_paths: list[Path] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        self.image_paths = list(kwargs["image_paths"])
        assert all(path.exists() for path in self.image_paths)
        return self.response


@pytest.mark.asyncio
async def test_claude_completion_uses_structured_contract(monkeypatch):
    fake = FakeClient(
        {
            "text": json.dumps({"content": "ready"}),
            "model": "sonnet",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )
    monkeypatch.setattr(claude_model, "ClaudeCodeClient", lambda: fake)
    model = ClaudeCodeChatModel(model_name="sonnet", reasoning_effort="medium")

    result = await model.ainvoke(
        [SystemMessage(content="Be exact"), HumanMessage(content="Return ready")]
    )

    assert result.content == "ready"
    assert result.response_metadata["provider"] == "claude-code"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert fake.calls[0]["output_schema"]["required"] == ["content"]
    assert "Be exact" in fake.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_claude_returns_langchain_tool_call(monkeypatch):
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
            "model": "sonnet",
            "usage": {},
        }
    )
    monkeypatch.setattr(claude_model, "ClaudeCodeClient", lambda: fake)

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    result = (
        await ClaudeCodeChatModel()
        .bind_tools([add_numbers], tool_choice="required")
        .ainvoke("Add numbers")
    )

    assert result.tool_calls[0]["name"] == "add_numbers"
    assert result.tool_calls[0]["args"] == {"a": 19, "b": 23}
    assert "Available Artemis tools" in fake.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_claude_stages_and_cleans_image(monkeypatch):
    fake = FakeClient({"text": json.dumps({"content": "seen"}), "usage": {}})
    monkeypatch.setattr(claude_model, "ClaudeCodeClient", lambda: fake)
    image = base64.b64encode(b"image-bytes").decode()

    result = await ClaudeCodeChatModel().ainvoke(
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


def test_factory_builds_claude_code_without_api_key():
    model = ModelFactory.create_model(ModelEndpoint(provider="claude-code", model_name="sonnet"))

    assert isinstance(model, ClaudeCodeChatModel)
