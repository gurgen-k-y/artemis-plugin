from types import SimpleNamespace

import pytest

from artemis.config.llm import _expand_default_into_nodes
from artemis.llm.registry import ProviderAdapter, ProviderRegistry
from artemis.llm.router import ModelEndpoint, ModelProvider


def test_registry_resolves_alias_and_constructs_provider():
    registry = ProviderRegistry()
    model = object()
    registry.register(ProviderAdapter("example", lambda _endpoint: model, ("example-ai",)))

    endpoint = SimpleNamespace(provider="example-ai", model_name="test")

    assert registry.canonical_name(endpoint.provider) == "example"
    assert registry.create(endpoint) is model


def test_registry_rejects_duplicate_alias():
    registry = ProviderRegistry()
    registry.register(ProviderAdapter("one", lambda _endpoint: object(), ("shared",)))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ProviderAdapter("two", lambda _endpoint: object(), ("shared",)))


def test_model_endpoint_accepts_plugin_provider_names():
    endpoint = ModelEndpoint(provider="future-provider", model_name="future-model")

    assert endpoint.provider == "future-provider"
    assert ModelProvider.from_string("gemini") == "google"


def test_tool_choice_none_forbids_tool_calls():
    from artemis.llm.client_messages import response_contract

    tools = [{"type": "function", "function": {"name": "tap", "parameters": {}}}]

    schema, _ = response_contract(tools, "none")

    assert "tool_name" not in schema["properties"]


def test_llm_preset_env_selects_default(monkeypatch):
    from artemis.config.llm import _apply_selected_preset

    config = {"default": {"provider": "google"}, "presets": {"codex": {"provider": "codex"}}}
    monkeypatch.setenv("ARTEMIS_LLM_PRESET", "codex")

    assert _apply_selected_preset(config)["default"] == {"provider": "codex"}

    monkeypatch.setenv("ARTEMIS_LLM_PRESET", "missing")
    with pytest.raises(ValueError, match="Unknown"):
        _apply_selected_preset(config)


def test_node_provider_override_survives_preset_expansion():
    config = {
        "default": {"provider": "codex", "model": "default"},
        "nodes": {
            "object_detector": {
                "provider": "google",
                "model": "gemini-robotics-er-2-preview",
            },
            "hopper": {"provider": "google", "model": "gemini-3.5-flash-lite"},
        },
    }

    expanded = _expand_default_into_nodes(config)

    assert expanded["utils"]["object_detector"]["provider"] == "google"
    assert expanded["utils"]["hopper"]["provider"] == "google"
