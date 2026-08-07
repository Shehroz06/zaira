import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# This job can run for hours; without this, Python block-buffers stdout
# whenever it's redirected to a file/log instead of a terminal, so progress
# wouldn't show up until the buffer fills or the process exits.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.recipes import canonical_recipe_text  # noqa: E402
from app.llm import embed_batch  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RECIPES_PATH = DATA_DIR / "recipes.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CHECKPOINT_PATH = DATA_DIR / "embeddings.checkpoint.npy"
CHECKPOINT_META_PATH = DATA_DIR / "embeddings.checkpoint.meta.json"

DEFAULT_BATCH_SIZE = 128
# Ollama pins embedding models to a single processing slot regardless of
# OLLAMA_NUM_PARALLEL (confirmed empirically — the spawned llama-server always
# shows "-np 1" for embedding models on this Ollama version), so concurrent
# client requests don't parallelize on the server; they just queue. Under
# memory pressure, multiple large in-flight batches can also make it more
# likely Ollama evicts/reloads the model between requests. Sequential is the
# safer default; raise --workers only if you've confirmed it helps on your
# setup.
DEFAULT_WORKERS = 1
MAX_ATTEMPTS = 4


def chunk(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def embed_batch_with_retry(texts):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return embed_batch(texts)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                wait = 2 ** attempt
                print(f"  batch failed ({exc}); retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})")
                time.sleep(wait)
    raise RuntimeError(f"Batch embedding failed after {MAX_ATTEMPTS} attempts: {last_error}")


def load_checkpoint(total_recipes):
    if not CHECKPOINT_PATH.exists() or not CHECKPOINT_META_PATH.exists():
        return None

    try:
        meta = json.loads(CHECKPOINT_META_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if meta.get("total_recipes") != total_recipes:
        print("Checkpoint does not match the current dataset (recipe count changed); ignoring it.")
        return None

    embeddings = np.load(CHECKPOINT_PATH, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] > total_recipes:
        print("Checkpoint file looks corrupted; ignoring it.")
        return None

    return embeddings


def save_checkpoint(embeddings, total_recipes):
    # np.save silently appends ".npy" to whatever path it's given, so the temp
    # path must already end in ".npy" or the later rename target is wrong.
    # The PID keeps this unique if two instances of this script ever run at
    # once — two processes sharing one temp filename means whichever renames
    # first deletes the file out from under the other, crashing it with a
    # FileNotFoundError.
    tmp_path = CHECKPOINT_PATH.with_name(f"{CHECKPOINT_PATH.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp_path, embeddings)
    tmp_path.replace(CHECKPOINT_PATH)
    CHECKPOINT_META_PATH.write_text(json.dumps({"total_recipes": total_recipes}))


def clear_checkpoint():
    CHECKPOINT_PATH.unlink(missing_ok=True)
    CHECKPOINT_META_PATH.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Precompute recipe embeddings via Ollama.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Recipes embedded per Ollama request")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent in-flight embedding requests")
    parser.add_argument("--restart", action="store_true", help="Ignore any existing checkpoint and start over")
    args = parser.parse_args()

    if not RECIPES_PATH.exists():
        print(f"{RECIPES_PATH} not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    recipes = json.loads(RECIPES_PATH.read_text())
    total = len(recipes)
    texts = [canonical_recipe_text(recipe) for recipe in recipes]

    if args.restart:
        clear_checkpoint()
        done_embeddings = None
    else:
        done_embeddings = load_checkpoint(total)

    done_count = 0 if done_embeddings is None else done_embeddings.shape[0]
    if done_count:
        print(f"Resuming from checkpoint: {done_count}/{total} recipes already embedded.")

    if done_count >= total:
        print("Checkpoint already covers the full dataset; skipping embedding step.")
        all_embeddings = done_embeddings
    else:
        remaining_texts = texts[done_count:]
        batches = list(chunk(remaining_texts, args.batch_size))

        print(
            f"Embedding {len(remaining_texts)}/{total} remaining recipes via Ollama "
            f"({len(batches)} batches of up to {args.batch_size}, {args.workers} concurrent requests)..."
        )

        results = [done_embeddings] if done_embeddings is not None else []
        completed_recipes = done_count
        started = time.perf_counter()

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for batch_embeddings in executor.map(embed_batch_with_retry, batches):
                    results.append(np.asarray(batch_embeddings, dtype=np.float32))
                    completed_recipes += len(batch_embeddings)
                    save_checkpoint(np.concatenate(results, axis=0), total)

                    elapsed = time.perf_counter() - started
                    rate = (completed_recipes - done_count) / elapsed if elapsed > 0 else 0
                    remaining = total - completed_recipes
                    eta_seconds = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"  {completed_recipes}/{total} "
                        f"({rate:.1f} recipes/s, ETA {eta_seconds / 60:.1f} min)"
                    )
        except KeyboardInterrupt:
            print(f"\nInterrupted. Progress saved at {completed_recipes}/{total} recipes.")
            print("Re-run this script to resume from the checkpoint.")
            sys.exit(1)

        all_embeddings = np.concatenate(results, axis=0)

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    normalized = all_embeddings / (norms + 1e-8)
    np.save(EMBEDDINGS_PATH, normalized)
    clear_checkpoint()
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
