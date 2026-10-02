# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Built-in model provider adapters."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from artemis.config.settings import settings
from artemis.llm.google.provider import supports_thinking_level
from artemis.llm.registry import Endpoint, ProviderAdapter, provider_registry
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def _secret(setting: Any) -> str | None:
    return setting.get_secret_value() if setting else None


def _present(value: str | None, name: str, label: str) -> None:
    if not value:
        raise RuntimeError(f"{label} requires {name} in .env")


def _patch_google_tool_config() -> None:
    try:
        from google.genai.types import ToolConfig
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

        if getattr(ChatGoogleGenerativeAI, "_is_artemis_patched", False):
            return
        original = ChatGoogleGenerativeAI._process_tool_config

        def patched(self, tool_choice, tool_config, formatted_tools):
            config = original(self, tool_choice, tool_config, formatted_tools)
            has_builtin = False
            has_functions = False
            for tool in formatted_tools or ():
                value = tool if isinstance(tool, dict) else getattr(tool, "__dict__", {})
                has_builtin |= any(
                    value.get(key) is not None
                    for key in (
                        "google_search",
                        "code_execution",
                        "google_maps",
                        "google_search_retrieval",
                    )
                )
                has_functions |= bool(value.get("function_declarations"))
            if tool_config:
                normalized = (
                    ToolConfig.model_validate(tool_config)
                    if isinstance(tool_config, dict)
                    else tool_config
                )
                requested = getattr(normalized, "include_server_side_tool_invocations", None)
                if requested is not None:
                    config = config or ToolConfig()
                    config.include_server_side_tool_invocations = requested
            if has_builtin and has_functions:
                config = config or ToolConfig()
                config.include_server_side_tool_invocations = True
            return config

        ChatGoogleGenerativeAI._process_tool_config = patched
        ChatGoogleGenerativeAI._is_artemis_patched = True
    except Exception as exc:
        logger.warning(f"Could not patch ChatGoogleGenerativeAI tool config: {exc}")


def _google(endpoint: Endpoint) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

    _patch_google_tool_config()
    api_key = getattr(endpoint, "api_key", None) or _secret(settings.GOOGLE_API_KEY)
    api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    thinking_level = (
        getattr(endpoint, "thinking_level", None)
        if supports_thinking_level(endpoint.model_name)
        else None
    )
    kwargs = {
        "model": endpoint.model_name,
        "temperature": getattr(endpoint, "temperature", 0.0),
        "max_output_tokens": getattr(endpoint, "max_tokens", None),
        "api_key": api_key,
        "timeout": getattr(endpoint, "timeout_seconds", 60.0),
        "thinking_budget": getattr(endpoint, "thinking_budget", None),
        "thinking_level": thinking_level,
        "include_thoughts": getattr(endpoint, "include_thoughts", None),
        "safety_settings": {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    }
    return ChatGoogleGenerativeAI(
        **{key: value for key, value in kwargs.items() if value is not None}
    )


def _vertex(endpoint: Endpoint) -> BaseChatModel:
    from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory

    kwargs = {
        "model_name": endpoint.model_name,
        "temperature": getattr(endpoint, "temperature", 0.0),
        "max_output_tokens": getattr(endpoint, "max_tokens", None),
        "timeout": getattr(endpoint, "timeout_seconds", 60.0),
        "thinking_budget": getattr(endpoint, "thinking_budget", None),
        "safety_settings": {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    }
    return ChatVertexAI(**{key: value for key, value in kwargs.items() if value is not None})


def _openai_compatible(endpoint: Endpoint) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    provider = provider_registry.canonical_name(endpoint.provider)
    api_key = getattr(endpoint, "api_key", None)
    base_url = getattr(endpoint, "api_base", None)
    if provider == "openai":
        api_key = (
            api_key or _secret(settings.OPENAI_API_KEY) or os.environ.get("OPENAI_API_KEY", "EMPTY")
        )
        base_url = base_url or (str(settings.OPENAI_BASE_URL) if settings.OPENAI_BASE_URL else None)
    elif provider == "openrouter":
        api_key = (
            api_key
            or _secret(settings.OPEN_ROUTER_API_KEY)
            or os.environ.get("OPEN_ROUTER_API_KEY")
        )
        base_url = base_url or "https://openrouter.ai/api/v1"
    elif provider == "xai":
        api_key = api_key or _secret(settings.XAI_API_KEY) or os.environ.get("XAI_API_KEY")
        base_url = base_url or "https://api.x.ai/v1"
    else:
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    kwargs = {
        "model": endpoint.model_name,
        "temperature": getattr(endpoint, "temperature", 0.0),
        "max_tokens": getattr(endpoint, "max_tokens", None),
        "api_key": api_key,
        "base_url": base_url,
        "timeout": getattr(endpoint, "timeout_seconds", 60.0),
        "reasoning_effort": getattr(endpoint, "reasoning_effort", None),
    }
    return ChatOpenAI(**{key: value for key, value in kwargs.items() if value is not None})


def _anthropic(endpoint: Endpoint) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = getattr(endpoint, "api_key", None) or _secret(settings.ANTHROPIC_API_KEY)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    kwargs: dict[str, Any] = {
        "model": endpoint.model_name,
        "temperature": getattr(endpoint, "temperature", 0.0),
        "api_key": api_key,
        "timeout": getattr(endpoint, "timeout_seconds", 60.0),
    }
    budget = getattr(endpoint, "thinking_budget", None)
    effort = getattr(endpoint, "reasoning_effort", None)
    if not budget and effort:
        budget = {"low": 2048, "medium": 8192, "high": 32768}.get(effort.lower())
    if budget:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs["temperature"] = 1.0
    return ChatAnthropic(**{key: value for key, value in kwargs.items() if value is not None})


def _validate_google(label: str) -> None:
    _present(_secret(settings.GOOGLE_API_KEY), "GOOGLE_API_KEY", label)


def _validate_openai(label: str) -> None:
    _present(_secret(settings.OPENAI_API_KEY), "OPENAI_API_KEY", label)


def _validate_anthropic(label: str) -> None:
    _present(
        _secret(settings.ANTHROPIC_API_KEY) or os.environ.get("ANTHROPIC_API_KEY"),
        "ANTHROPIC_API_KEY",
        label,
    )


def _validate_openrouter(label: str) -> None:
    _present(_secret(settings.OPEN_ROUTER_API_KEY), "OPEN_ROUTER_API_KEY", label)


def _validate_xai(label: str) -> None:
    _present(_secret(settings.XAI_API_KEY), "XAI_API_KEY", label)


def _validate_vertex(_label: str) -> None:
    from third_party.mobile_use.config.llm import validate_vertex_ai_credentials

    validate_vertex_ai_credentials()


def register_builtin_providers() -> None:
    from artemis.llm.codex import CodexAppServerChatModel, codex_client_status

    def codex(endpoint: Endpoint) -> BaseChatModel:
        return CodexAppServerChatModel(
            model_name=endpoint.model_name,
            reasoning_effort=getattr(endpoint, "reasoning_effort", None),
            timeout_seconds=getattr(endpoint, "timeout_seconds", 180.0),
        )

    def validate_codex(label: str) -> None:
        ready, detail = codex_client_status()
        if not ready:
            raise RuntimeError(f"{label} requires a signed-in Codex CLI: {detail}")

    adapters = (
        ProviderAdapter("google", _google, ("gemini",), _validate_google),
        ProviderAdapter("vertexai", _vertex, ("vertex", "vertex-ai"), _validate_vertex),
        ProviderAdapter("openai", _openai_compatible, validator=_validate_openai),
        ProviderAdapter("codex", codex, ("codex-client", "chatgpt-client"), validate_codex),
        ProviderAdapter("anthropic", _anthropic, ("claude",), _validate_anthropic),
        ProviderAdapter("openrouter", _openai_compatible, ("open-router",), _validate_openrouter),
        ProviderAdapter("xai", _openai_compatible, ("grok", "x-ai"), _validate_xai),
        ProviderAdapter("ollama", _openai_compatible),
        ProviderAdapter("vllm", _openai_compatible),
        ProviderAdapter("custom", _openai_compatible),
    )
    for adapter in adapters:
        try:
            provider_registry.register(adapter)
        except ValueError:
            pass
