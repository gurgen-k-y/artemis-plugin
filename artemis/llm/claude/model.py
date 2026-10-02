"""LangChain model backed by one-shot Claude Code calls."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from artemis.llm.claude.client import ClaudeCodeClient, ClaudeCodeError
from artemis.llm.client_messages import (
    format_tools,
    message_transcript,
    parse_response,
    parse_tool_call,
    response_contract,
    system_instructions,
)
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class ClaudeCodeChatModel(BaseChatModel):
    model_name: str = Field(default="sonnet")
    reasoning_effort: str | None = Field(default=None)
    timeout_seconds: float = Field(default=180.0, gt=0)

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "transport": "claude-code-print",
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self.bind(tools=format_tools(tools), tool_choice=tool_choice, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        tools = [] if kwargs.get("tool_choice") == "none" else list(kwargs.get("tools") or [])
        output_schema, contract = response_contract(tools, kwargs.get("tool_choice"))
        transcript, images, temp_paths = message_transcript(messages)
        try:
            with tempfile.TemporaryDirectory(prefix="artemis-claude-images-") as image_dir:
                image_paths = []
                for index, image in enumerate(images):
                    source = Path(image["path"])
                    destination = Path(image_dir) / f"{index}{source.suffix or '.img'}"
                    shutil.copyfile(source, destination)
                    image_paths.append(destination)
                sections = [
                    system_instructions(messages),
                    contract,
                    "Artemis executes automation tools. Do not execute tools or modify files yourself.",
                ]
                if tools:
                    sections.append("Available Artemis tools:\n" + json.dumps(tools, default=str))
                if image_paths:
                    sections.append(
                        "Use the Read tool only to inspect these temporary screenshots:\n"
                        + "\n".join(str(path) for path in image_paths)
                    )
                sections.append("Conversation:\n" + (transcript or "Continue."))
                result = await ClaudeCodeClient().complete(
                    model=self.model_name,
                    effort=self.reasoning_effort,
                    prompt="\n\n".join(sections),
                    output_schema=output_schema,
                    image_paths=image_paths,
                    timeout_seconds=self.timeout_seconds,
                )
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.debug(f"Could not remove temporary Claude image {path}")

        parsed = parse_response(result["text"])
        usage = result.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("outputTokens") or 0)
        usage_metadata = (
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            if usage
            else None
        )
        metadata = {"provider": "claude-code", "model": result.get("model") or self.model_name}
        if tools and parsed.get("kind") == "tool_call":
            try:
                name, args = parse_tool_call(parsed)
            except ValueError as exc:
                raise ClaudeCodeError(str(exc)) from exc
            message = AIMessage(
                content=parsed.get("content") or "",
                tool_calls=[
                    {
                        "name": name,
                        "args": args,
                        "id": f"call_{uuid4().hex}",
                        "type": "tool_call",
                    }
                ],
                response_metadata=metadata,
                usage_metadata=usage_metadata,
            )
        else:
            message = AIMessage(
                content=str(parsed.get("content") or result["text"]),
                response_metadata=metadata,
                usage_metadata=usage_metadata,
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))
