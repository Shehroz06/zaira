# Setup & Operations Guide

Detailed configuration, dataset preparation, and operational notes. See the [README](../README.md) for a quick start.

## Environment variables

Copy `.env.example` to `.env` and adjust for your machine. `.env` is loaded automatically if present.

| Variable | Purpose |
|---|---|
| `OLLAMA_URL` | Ollama HTTP endpoint |
| `OLLAMA_MODEL` | Generation model used by Ollama |
| `OLLAMA_EMBED_MODEL` | Embedding model used by Ollama |
| `ZAIRA_NUM_THREADS` | CPU threads for Ollama generation — increase only if your machine has headroom |
| `ZAIRA_MIN_RELEVANCE` | Retrieval acceptance threshold — keep at default unless you've measured the effect on your dataset |
| `ZAIRA_RRF_K` | Reciprocal rank fusion constant |
| `ZAIRA_RUNTIME_EMBED_LIMIT` | Max recipes to embed at request time before requiring a precomputed index |
| `ZAIRA_DEBUG_RETRIEVAL` | Set to `1` to log retrieval debug info |
| `ZAIRA_RESULT_COUNT` | Number of recipe cards returned per query (default 4) |
| `LLM_PROVIDER_MODE` | `auto`, `gemini`, or `ollama` |
| `GEMINI_ENABLED` | Enables Gemini if `true` (requires `GEMINI_API_KEY`) |
| `GEMINI_API_KEY` | Gemini API key, server-side only |
| `GEMINI_MODEL` | Gemini model name |
| `GEMINI_TIMEOUT_SECONDS` | Gemini request timeout |

If Gemini is enabled, Zaira tries it first and falls back to Ollama automatically on failure or timeout.

## Preparing the recipe dataset

Zaira is built and tested against the [Food.com Recipes and Reviews dataset](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) on Kaggle (~520,000 recipes). Download `recipes.csv` into `data/raw/`, then:

```bash
python scripts/prepare_data.py data/raw/recipes.csv --limit 10000
python scripts/build_index.py
```

- `prepare_data.py` auto-detects the dataset's `Name` / `RecipeIngredientParts` / `RecipeInstructions` / `PrepTime` / `CookTime` / `TotalTime` columns. `--limit` defaults to 2000 if omitted. Other recipe CSVs work too — see `TITLE_KEYS` / `INGREDIENT_KEYS` / `STEP_KEYS` at the top of the script to match different headers.
- `build_index.py` embeds recipes via Ollama in batches and checkpoints progress to `data/embeddings.checkpoint.npy`, so a large dataset can safely take hours — an interrupted run resumes automatically. Flags:
  - `--batch-size` (default 128): recipes embedded per Ollama request
  - `--workers` (default 1): concurrent in-flight embedding requests. Ollama pins embedding models to a single processing slot regardless of concurrency, so raising this rarely helps and can hurt under memory pressure
  - `--restart`: ignore any existing checkpoint and start over

This produces `data/recipes.json` and `data/embeddings.npy`. Without a prepared dataset, Zaira falls back to the bundled `demo/demo_recipes.json`. For datasets larger than `ZAIRA_RUNTIME_EMBED_LIMIT` (default 32), `data/embeddings.npy` must be precomputed — Zaira won't embed a large dataset at request time.

## Using the app while a large index is still building

You don't have to wait for `build_index.py` to finish:

1. Start `build_index.py` in the background — it checkpoints after every batch.
2. Promote whatever's been embedded so far into the live index:
   ```bash
   python scripts/promote_checkpoint.py
   ```
   This copies the checkpoint over `data/embeddings.npy`. Zaira serves a partial index against the recipes already embedded (a consistent prefix, since recipes are embedded in dataset order).
3. Restart the app to pick it up. Re-run `promote_checkpoint.py` and restart again whenever you want a bigger index.

Running the app and `build_index.py` at the same time works, but both compete for the same embedding model — expect chat responses to be noticeably slower while the background build is active. Pause `build_index.py` (safe, resumes from checkpoint) for fast responses during a testing session.

## Retrieval evaluation

```bash
python scripts/evaluate_retrieval.py --max-cases 5 --k 2
```

Prints retrieved recipes, Hit@K, no-result handling, and stage timings.

## Known limitations

- CPU-only generation is slow for larger or more detailed answers.
- Gemini support is optional and only active with a valid API key.
- Retrieval is tuned for recipe-sized datasets, not very large general-purpose corpora.
- Reranking and response caching are not implemented.
- Retrieval quality depends on the source dataset's quality and consistency.
