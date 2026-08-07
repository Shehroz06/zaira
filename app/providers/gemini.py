from __future__ import annotations

import json
import os
import re
from typing import Iterable

import requests

from app.providers.base import GeneratedText, StructuredResult


class GeminiProvider:
    name = "gemini"

    def __init__(self):
        self.enabled = os.environ.get("GEMINI_ENABLED", "false").lower() == "true"
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.timeout_seconds = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))

    @property
    def available(self):
        return self.enabled and bool(self.api_key)

    def _endpoint(self, suffix, sse=False):
        query = "?alt=sse" if sse else ""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:{suffix}{query}"

    def _headers(self):
        # Sent as a header rather than a `?key=` query param so the key can
        # never end up embedded in a requests.HTTPError message (raise_for_status
        # includes the request URL) or in any log line that captures that URL.
        return {"x-goog-api-key": self.api_key}

    def _payload(self, prompt, structured=False):
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            },
        }
        if structured:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        return payload

    def _extract_text(self, response_json):
        candidates = response_json.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
            combined = "".join(texts).strip()
            if combined:
                return combined
        return ""

    def _extract_chunk_text(self, response_json):
        # Like _extract_text but without strip(): streamed chunks may begin or
        # end mid-word, so surrounding whitespace must be preserved.
        candidates = response_json.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
            combined = "".join(texts)
            if combined:
                return combined
        return ""

    def _request(self, prompt, structured=False):
        if not self.available:
            raise RuntimeError("Gemini is unavailable")
        response = requests.post(
            self._endpoint("generateContent"),
            json=self._payload(prompt, structured=structured),
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return text

    def stream(self, prompt: str) -> Iterable[str]:
        if not self.available:
            raise RuntimeError("Gemini is unavailable")
        with requests.post(
            self._endpoint("streamGenerateContent", sse=True),
            json=self._payload(prompt, structured=False),
            headers=self._headers(),
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            yielded = False
            for line in response.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = self._extract_chunk_text(chunk)
                if text:
                    yielded = True
                    yield text
            if not yielded:
                raise RuntimeError("Gemini returned an empty response")

    def generate(self, prompt: str) -> GeneratedText:
        return GeneratedText(provider=self.name, text=self._request(prompt, structured=False))

    def generate_structured(self, prompt: str) -> StructuredResult:
        try:
            text = self._request(prompt, structured=True)
        except Exception as exc:
            return StructuredResult(provider=self.name, value=None, raw_text="", valid=False, error=str(exc))

        cleaned = text.strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            cleaned = match.group(0)
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                return StructuredResult(provider=self.name, value=payload, raw_text=text, valid=True)
        except json.JSONDecodeError as exc:
            return StructuredResult(provider=self.name, value=None, raw_text=text, valid=False, error=str(exc))
        return StructuredResult(provider=self.name, value=None, raw_text=text, valid=False, error="Output was not a JSON object")
