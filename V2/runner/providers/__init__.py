"""어댑터 레지스트리."""

from __future__ import annotations

from .anthropic_adapter import AnthropicAdapter
from .base import BaseAdapter, RawResult
from .hyperclova_adapter import HyperClovaAdapter
from .openai_compat import OpenAICompatAdapter

ADAPTERS = {
    "openai_compat": OpenAICompatAdapter,
    "anthropic": AnthropicAdapter,
    "hyperclova": HyperClovaAdapter,
}


def build_adapter(spec) -> BaseAdapter:
    try:
        cls = ADAPTERS[spec.adapter]
    except KeyError:
        raise KeyError(f"알 수 없는 어댑터: {spec.adapter} (모델 {spec.key})")
    return cls(spec)


__all__ = ["ADAPTERS", "build_adapter", "BaseAdapter", "RawResult"]
