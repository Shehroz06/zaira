from __future__ import annotations

import os
import re
from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel

from app.providers import GeminiProvider, OllamaProvider
from app.rag import QueryIntent, parse_query_intent


class IntentSchema(BaseModel):
    ingredients_include: list[str] = []
    ingredients_exclude: list[str] = []
    diet: list[str] = []
    meal_type: list[str] = []
    max_time_minutes: int | None = None
    equipment_include: list[str] = []
    equipment_exclude: list[str] = []
    dish_type: str | None = None
    prefer_quick: bool = False

    class Config:
        extra = "forbid"


@dataclass
class QueryUnderstanding:
    intent: QueryIntent
    provider: str
    fallback_used: bool = False
    raw_text: str = ""
    error: str | None = None


class LLMService:
    def __init__(self):
        self.mode = os.environ.get("LLM_PROVIDER_MODE", "auto").strip().lower()
        # The Gemini structured-intent call adds a full API round-trip before
        # retrieval can start. The local regex parser covers the same fields,
        # so the remote call is opt-in.
        self.llm_intent_enabled = os.environ.get("ZAIRA_LLM_INTENT", "false").strip().lower() == "true"
        self.gemini = GeminiProvider()
        self.ollama = OllamaProvider()
        self.last_provider = self.ollama.name
        self.last_query_provider = self.ollama.name
        self.last_metrics: dict[str, float | str | bool | None] = {}

    def is_complex_query(self, query: str) -> bool:
        intent = parse_query_intent(query)
        score = 0
        score += len(intent.ingredients_include)
        score += len(intent.ingredients_exclude)
        score += len(intent.diet)
        score += len(intent.meal_type)
        score += len(intent.equipment_include)
        score += len(intent.equipment_exclude)
        score += 1 if intent.max_time_minutes is not None else 0
        score += 1 if re.search(r"\band\b|\bwith\b|\bwithout\b|\bno\b|\bprefer\b", query.lower()) else 0
        score += 1 if len(query.split()) > 12 else 0
        return score >= 2

    def _choose_generation_provider(self, query: str):
        if self.mode == "ollama":
            return self.ollama
        if self.mode == "gemini":
            return self.gemini if self.gemini.available else self.ollama
        if self.is_complex_query(query) and self.gemini.available:
            return self.gemini
        return self.ollama

    def _structured_intent_prompt(self, user_query: str) -> str:
        return f"""
Convert the cooking request into strict JSON only.
Return exactly one JSON object and nothing else.

Allowed keys:
- ingredients_include: array of strings
- ingredients_exclude: array of strings
- diet: array of strings
- meal_type: array of strings
- max_time_minutes: integer or null
- equipment_include: array of strings
- equipment_exclude: array of strings
- dish_type: string or null
- prefer_quick: boolean

Use empty arrays when no values are present.
Use null for unknown scalar values.
Do not invent extra keys.

User request:
{user_query}
""".strip()

    def _fallback_intent(self, query: str) -> QueryIntent:
        return parse_query_intent(query)

    def _intent_from_schema(self, payload: dict) -> QueryIntent:
        schema = IntentSchema.parse_obj(payload)
        return QueryIntent(
            ingredients_include=[str(item).strip() for item in schema.ingredients_include if str(item).strip()],
            ingredients_exclude=[str(item).strip() for item in schema.ingredients_exclude if str(item).strip()],
            diet=[str(item).strip() for item in schema.diet if str(item).strip()],
            meal_type=[str(item).strip() for item in schema.meal_type if str(item).strip()],
            max_time_minutes=schema.max_time_minutes,
            equipment_include=[str(item).strip() for item in schema.equipment_include if str(item).strip()],
            equipment_exclude=[str(item).strip() for item in schema.equipment_exclude if str(item).strip()],
            dish_type=schema.dish_type,
            prefer_quick=bool(schema.prefer_quick),
        )

    def resolve_query_intent(self, user_query: str) -> QueryUnderstanding:
        started = perf_counter()
        fallback = self._fallback_intent(user_query)

        should_use_gemini = self.llm_intent_enabled and self.gemini.available and self.mode in {"auto", "gemini"}
        if should_use_gemini:
            try:
                structured = self.gemini.generate_structured(self._structured_intent_prompt(user_query))
                if structured.valid and structured.value is not None:
                    intent = self._intent_from_schema(structured.value)
                    self.last_query_provider = structured.provider
                    self.last_metrics = {
                        "query_understanding_provider": structured.provider,
                        "query_understanding_fallback": False,
                        "query_understanding_latency": perf_counter() - started,
                    }
                    return QueryUnderstanding(intent=intent, provider=structured.provider, raw_text=structured.raw_text)
                self.last_metrics = {
                    "query_understanding_provider": self.ollama.name,
                    "query_understanding_fallback": True,
                    "query_understanding_latency": perf_counter() - started,
                }
                return QueryUnderstanding(
                    intent=fallback,
                    provider=self.ollama.name,
                    fallback_used=True,
                    raw_text=structured.raw_text,
                    error=structured.error or "Gemini returned invalid structured output",
                )
            except Exception as exc:
                self.last_metrics = {
                    "query_understanding_provider": self.ollama.name,
                    "query_understanding_fallback": True,
                    "query_understanding_latency": perf_counter() - started,
                }
                return QueryUnderstanding(intent=fallback, provider=self.ollama.name, fallback_used=True, error=str(exc))

        self.last_query_provider = self.ollama.name
        self.last_metrics = {
            "query_understanding_provider": self.ollama.name,
            "query_understanding_fallback": False,
            "query_understanding_latency": perf_counter() - started,
        }
        return QueryUnderstanding(intent=fallback, provider=self.ollama.name)

    def stream_answer(self, prompt: str, query: str = ""):
        started = perf_counter()
        if self.gemini.available and self.mode in {"auto", "gemini"}:
            providers = [self.gemini, self.ollama]
        else:
            providers = [self.ollama]

        chosen = providers[0]

        last_error = None
        for provider in providers:
            try:
                self.last_provider = provider.name
                self.last_metrics.update(
                    {
                        "generation_provider": provider.name,
                        "generation_fallback": provider.name != chosen.name,
                    }
                )
                yield from provider.stream(prompt)
                self.last_metrics["generation_latency"] = perf_counter() - started
                return
            except Exception as exc:
                last_error = exc
                if provider.name != self.ollama.name and self.ollama not in providers:
                    providers.append(self.ollama)

        self.last_metrics["generation_latency"] = perf_counter() - started
        raise RuntimeError(f"No LLM provider could generate an answer: {last_error}")
