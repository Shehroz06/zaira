import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.rag import DATASET_EMBEDDINGS  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.checkpoint.npy"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy the in-progress build_index.py checkpoint over data/embeddings.npy, "
            "so the app can serve whatever has been embedded so far without waiting "
            "for the full dataset to finish. Safe to run while build_index.py keeps "
            "running in the background — restart the app afterward to pick it up."
        )
    )
    args = parser.parse_args()
    del args

    if not CHECKPOINT_PATH.exists():
        print(f"{CHECKPOINT_PATH} not found — is scripts/build_index.py running (or has it already finished)?")
        sys.exit(1)

    embeddings = np.load(CHECKPOINT_PATH, allow_pickle=False)
    np.save(DATASET_EMBEDDINGS, embeddings)
    print(f"Promoted {embeddings.shape[0]} embedded recipes to {DATASET_EMBEDDINGS}.")
    print("Restart the app (uvicorn main:app) to pick up the larger index.")


if __name__ == "__main__":
    main()
