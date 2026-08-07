# Zaira

A local cooking assistant. Ask for a recipe in natural language and get grounded, relevant results streamed back — no cloud dependency required.

## Features

- Natural-language recipe search with intent parsing (ingredients, diet, meal type, time, equipment)
- Hybrid retrieval: vector similarity + BM25 keyword search, fused and reranked locally
- Streams a short intro reply, then shows matches as compact, expandable recipe cards
- Rejects weak matches instead of forcing a bad answer
- Runs fully offline via Ollama; optional Gemini fallback for query understanding and generation

## Tech stack

- **Backend**: FastAPI, NumPy (hybrid retrieval index)
- **LLM**: Ollama (local), optional Gemini
- **Frontend**: single-file HTML/CSS/JS, no build step

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

## Environment variables

```bash
cp .env.example .env
```

Key variables: `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `LLM_PROVIDER_MODE`, `GEMINI_ENABLED`/`GEMINI_API_KEY`. Full reference: [docs/SETUP.md](docs/SETUP.md).

## Running locally

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/`. Without a prepared dataset, Zaira uses the bundled demo recipes — see [docs/SETUP.md](docs/SETUP.md) for preparing a full dataset.

## Project structure

```text
Zaira/
├── main.py                FastAPI app, /chat endpoint (entrypoint)
├── app/
│   ├── zaira.py            query flow, retrieval, prompt grounding
│   ├── rag.py               intent parsing, hybrid retrieval
│   ├── llm.py                Ollama HTTP client
│   ├── llm_service.py        provider selection and fallback
│   ├── recipes.py            recipe normalization helpers
│   └── providers/             Ollama and Gemini provider wrappers
├── static/index.html      browser frontend
├── scripts/                dataset prep, indexing, evaluation
└── data/                    recipe dataset and embeddings (generated)
```

## Useful commands

```bash
python scripts/prepare_data.py data/raw/recipes.csv --limit 10000  # CSV -> recipes.json
python scripts/build_index.py                                       # recipes.json -> embeddings.npy
python scripts/promote_checkpoint.py                                 # use a partial index while building
python scripts/evaluate_retrieval.py --max-cases 5 --k 2              # retrieval quality check
```

## License

MIT — see [LICENSE](LICENSE).
