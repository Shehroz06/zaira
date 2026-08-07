from __future__ import annotations

import json
import os
from typing import Iterable

from app.llm import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, ask_llm_stream, embed
from app.providers.base import GeneratedText, StructuredResult


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None, embed_model: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
        self.embed_model = embed_model or os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    def stream(self, prompt: str) -> Iterable[str]:
        yield from ask_llm_stream(prompt, model=self.model)

    def generate(self, prompt: str) -> GeneratedText:
        return GeneratedText(provider=self.name, text="".join(self.stream(prompt)))

    def generate_structured(self, prompt: str) -> StructuredResult:
        result = self.generate(prompt)
        text = result.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return StructuredResult(provider=self.name, value=payload, raw_text=result.text, valid=True)
        except json.JSONDecodeError as exc:
            return StructuredResult(provider=self.name, value=None, raw_text=result.text, valid=False, error=str(exc))
        return StructuredResult(provider=self.name, value=None, raw_text=result.text, valid=False, error="Output was not a JSON object")

    def embed(self, text: str):
        return embed(text, model=self.embed_model)
