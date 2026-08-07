import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# llama.cpp defaults to 2 threads regardless of CPU size, which is slow.
# Using more speeds up generation, but competes with everything else running
# on the machine for CPU, so default conservatively and let it be tuned via
# env var (e.g. ZAIRA_NUM_THREADS=8 if the machine is otherwise idle).
def resolve_num_threads():
    value = os.environ.get("ZAIRA_NUM_THREADS")
    if value:
        try:
            resolved = int(value)
            return max(1, resolved)
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


NUM_THREADS = resolve_num_threads()

# Keep models loaded between requests; Ollama's default unloads them after
# ~5 idle minutes, which forces a slow cold reload on the next chat.
KEEP_ALIVE = os.environ.get("ZAIRA_KEEP_ALIVE", "30m")


def ask_llm_stream(prompt, model=DEFAULT_CHAT_MODEL):
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_thread": NUM_THREADS,
                "num_predict": 500,
            },
        },
        stream=True,
        timeout=None,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                break


def embed(text, model=DEFAULT_EMBED_MODEL):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": model,
            "prompt": text,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_thread": NUM_THREADS},
        },
        timeout=None,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts, model=DEFAULT_EMBED_MODEL):
    # Uses Ollama's batch endpoint (single input list -> single forward pass)
    # instead of one HTTP round-trip per text. This is what makes bulk indexing
    # of large datasets tractable — see scripts/build_index.py.
    if not texts:
        return []
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": model,
            "input": texts,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_thread": NUM_THREADS},
        },
        timeout=None,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
