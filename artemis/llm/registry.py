# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Extensible registry for Artemis model providers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from threading import Lock
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel


class Endpoint(Protocol):
    provider: str
    model_name: str


ProviderFactory = Callable[[Endpoint], BaseChatModel]
ProviderValidator = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ProviderAdapter:
    name: str
    factory: ProviderFactory
    aliases: tuple[str, ...] = ()
    validator: ProviderValidator | None = None


def normalize_provider_name(value: Any) -> str:
    if value is None:
        return "google"
    name = str(value).strip().lower().replace("_", "-")
    return name or "google"


class ProviderRegistry:
    """Maps stable provider names and aliases to focused adapters."""

    def __init__(self, entry_point_group: str = "artemis.llm_providers") -> None:
        self._entry_point_group = entry_point_group
        self._adapters: dict[str, ProviderAdapter] = {}
        self._aliases: dict[str, str] = {}
        self._plugins_loaded = False
        self._lock = Lock()

    def register(self, adapter: ProviderAdapter, *, replace: bool = False) -> None:
        name = normalize_provider_name(adapter.name)
        aliases = {normalize_provider_name(alias) for alias in adapter.aliases}
        aliases.add(name)
        if not replace and (name in self._adapters or any(a in self._aliases for a in aliases)):
            raise ValueError(f"LLM provider or alias already registered: {name}")
        if replace and name in self._adapters:
            old = self._adapters[name]
            for alias in (old.name, *old.aliases):
                self._aliases.pop(normalize_provider_name(alias), None)
        self._adapters[name] = ProviderAdapter(
            name=name,
            factory=adapter.factory,
            aliases=tuple(sorted(aliases - {name})),
            validator=adapter.validator,
        )
        for alias in aliases:
            owner = self._aliases.get(alias)
            if owner is not None and owner != name and not replace:
                raise ValueError(f"LLM provider alias already registered: {alias}")
            self._aliases[alias] = name

    def resolve(self, value: Any) -> ProviderAdapter:
        self.load_plugins()
        requested = normalize_provider_name(value)
        name = self._aliases.get(requested)
        if name is None:
            available = ", ".join(self.names())
            raise ValueError(f"Unknown LLM provider {value!r}. Available providers: {available}")
        return self._adapters[name]

    def canonical_name(self, value: Any) -> str:
        return self.resolve(value).name

    def create(self, endpoint: Endpoint) -> BaseChatModel:
        return self.resolve(endpoint.provider).factory(endpoint)

    def validate(self, provider: Any, label: str) -> None:
        validator = self.resolve(provider).validator
        if validator is not None:
            validator(label)

    def names(self) -> tuple[str, ...]:
        self.load_plugins()
        return tuple(sorted(self._adapters))

    def load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        with self._lock:
            if self._plugins_loaded:
                return
            self._plugins_loaded = True
            discovered = metadata.entry_points()
            entries: Iterable[metadata.EntryPoint]
            if hasattr(discovered, "select"):
                entries = discovered.select(group=self._entry_point_group)
            else:
                entries = discovered.get(self._entry_point_group, ())
            for entry in entries:
                loaded = entry.load()
                adapter = (
                    loaded()
                    if callable(loaded) and not isinstance(loaded, ProviderAdapter)
                    else loaded
                )
                if not isinstance(adapter, ProviderAdapter):
                    raise TypeError(
                        f"Provider entry point {entry.name!r} must return ProviderAdapter"
                    )
                self.register(adapter)


provider_registry = ProviderRegistry()
