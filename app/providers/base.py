from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass
class GeneratedText:
    provider: str
    text: str


@dataclass
class StructuredResult:
    provider: str
    value: dict | None
    raw_text: str = ""
    valid: bool = False
    error: str | None = None


class LLMProvider(Protocol):
    name: str

    def stream(self, prompt: str) -> Iterable[str]:
        ...

    def generate(self, prompt: str) -> GeneratedText:
        ...

    def generate_structured(self, prompt: str) -> StructuredResult:
        ...
