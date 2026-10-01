"""LangChain model backed by the signed-in Codex app-server."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from artemis.llm.client_messages import (
    format_tools,
    message_transcript,
    parse_response,
    parse_tool_call,
    response_contract,
    system_instructions,
)
from artemis.llm.codex.client import (
    CodexAppServerError,
    client_for_running_loop,
    close_current_client,
)
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class CodexAppServerChatModel(BaseChatModel):
    model_name: str = Field(default="default")
    reasoning_effort: str | None = Field(default=None)
    timeout_seconds: float = Field(default=180.0, gt=0)
    cwd: str = Field(default_factory=os.getcwd)

    @property
    def _llm_type(self) -> str:
        return "codex-app-server"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "transport": "app-server-stdio",
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
        tools = list(kwargs.get("tools") or [])
        output_schema, contract = response_contract(tools, kwargs.get("tool_choice"))
        transcript, images, temp_paths = message_transcript(messages)
        inputs: list[dict[str, Any]] = [
            {"type": "text", "text": transcript or "Continue from the supplied instructions."}
        ]
        inputs.extend(images)
        developer_instructions = contract
        if tools:
            developer_instructions += "\n\nAvailable Artemis tools:\n" + json.dumps(
                tools, default=str
            )
        try:
            result = await client_for_running_loop().complete(
                model=self.model_name,
                effort=self.reasoning_effort,
                cwd=str(Path(self.cwd).resolve()),
                instructions=system_instructions(messages),
                developer_instructions=developer_instructions,
                inputs=inputs,
                output_schema=output_schema,
                timeout_seconds=self.timeout_seconds,
            )
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove temporary Codex image %s", path)
        parsed = parse_response(result["text"])
        response_metadata = {
            "provider": "codex",
            "model": result.get("model") or self.model_name,
            "thread_id": result.get("thread_id"),
        }
        usage = (result.get("usage") or {}).get("last") or {}
        usage_metadata = (
            {
                "input_tokens": int(usage.get("inputTokens") or 0),
                "output_tokens": int(usage.get("outputTokens") or 0),
                "total_tokens": int(usage.get("totalTokens") or 0),
            }
            if usage
            else None
        )
        if tools and parsed.get("kind") == "tool_call":
            try:
                name, args = parse_tool_call(parsed)
            except ValueError as exc:
                raise CodexAppServerError(str(exc)) from exc
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
                response_metadata=response_metadata,
                usage_metadata=usage_metadata,
            )
        else:
            message = AIMessage(
                content=str(parsed.get("content") or result["text"]),
                response_metadata=response_metadata,
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

        async def invoke() -> ChatResult:
            try:
                return await self._agenerate(messages, stop=stop, **kwargs)
            finally:
                await close_current_client()

        return asyncio.run(invoke())
