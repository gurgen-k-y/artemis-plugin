from types import SimpleNamespace

import pytest

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
