# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model endpoint descriptors and registry-backed model construction."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator

from artemis.llm.providers import register_builtin_providers
from artemis.llm.registry import normalize_provider_name, provider_registry
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class ModelProvider:
    """Compatibility constants for built-in providers.

    Provider names are strings rather than a closed enum so installed plugins
    can add providers without changing Artemis core.
    """

    GOOGLE = "google"
    GEMINI = "google"
    VERTEX_AI = "vertexai"
    OPENAI = "openai"
    CODEX = "codex"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    XAI = "xai"
    OLLAMA = "ollama"
    VLLM = "vllm"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, value: Any) -> str:
        register_builtin_providers()
        return provider_registry.canonical_name(value)


class ModelEndpoint(BaseModel):
    """Configuration for an LLM/VLM model endpoint."""

    provider: str = Field(default=ModelProvider.GOOGLE, description="Model provider")
    model_name: str = Field(default="gemini-2.5-flash", description="Model name identifier")
    api_key: str | None = Field(default=None, description="API Key or secret")
    api_base: str | None = Field(default=None, description="Custom API endpoint base URL")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum completion tokens")
    timeout_seconds: float = Field(default=60.0, description="Request timeout in seconds")
    is_multimodal: bool = Field(default=True, description="Whether endpoint accepts images")
    reasoning_effort: str | None = Field(default=None, description="Reasoning effort")
    thinking_budget: int | None = Field(default=None, description="Thinking budget token count")
    thinking_level: str | None = Field(default=None, description="Thinking level")
    include_thoughts: bool | None = Field(default=None, description="Include thought traces")
    enable_grounding: bool = Field(default=False, description="Enable Google Search grounding")

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: Any) -> str:
        return normalize_provider_name(value)

    def cache_key(self) -> tuple[Any, ...]:
        api_key_digest = (
            hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16] if self.api_key else None
        )
        return (
            ModelProvider.from_string(self.provider),
            self.model_name,
            self.temperature,
            self.max_tokens,
            self.timeout_seconds,
            self.thinking_budget,
            self.thinking_level,
            self.include_thoughts,
            self.reasoning_effort,
            self.enable_grounding,
            self.api_base,
            api_key_digest,
        )


class ModelFactory:
    """Instantiate and cache chat models through the provider registry."""

    _cache: dict[tuple[Any, ...], BaseChatModel] = {}

    @classmethod
    def get_model(cls, endpoint: ModelEndpoint) -> BaseChatModel:
        key = endpoint.cache_key()
        if key not in cls._cache:
            cls._cache[key] = cls.create_model(endpoint)
        return cls._cache[key]

    @classmethod
    def create_model(cls, endpoint: ModelEndpoint) -> BaseChatModel:
        if os.environ.get("ARTEMIS_FAKE_LLM") == "1":
            from artemis.llm.fake_model import FakeChatModel

            delay = float(os.environ.get("ARTEMIS_FAKE_LLM_DELAY_S", "0") or 0)
            logger.warning(
                f"ARTEMIS_FAKE_LLM=1 — returning FakeChatModel (delay={delay}s) "
                f"instead of {endpoint.provider}/{endpoint.model_name}"
            )
            return FakeChatModel(delay_s=delay)
        register_builtin_providers()
        return provider_registry.create(endpoint)
