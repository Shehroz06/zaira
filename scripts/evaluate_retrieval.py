import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import get_index, retrieve  # noqa: E402

COMMON_INGREDIENT_TOKENS = {
    "all-purpose",
    "and",
    "baking",
    "bread",
    "butter",
    "can",
    "chopped",
    "cup",
    "cups",
    "diced",
    "egg",
    "eggs",
    "flour",
    "fresh",
    "ground",
    "large",
    "minced",
    "oil",
    "ounce",
    "ounces",
    "piece",
    "pieces",
    "pinch",
    "pound",
    "pounds",
    "salt",
    "sliced",
    "small",
    "tablespoon",
    "tablespoons",
    "teaspoon",
    "teaspoons",
    "water",
}


def _ingredient_phrase(recipe):
    words = []
    for ingredient in recipe.get("ingredients", []):
        for token in ingredient.lower().replace("-", " ").replace(",", " ").split():
            cleaned = "".join(character for character in token if character.isalnum())
            if cleaned and cleaned not in COMMON_INGREDIENT_TOKENS and not cleaned.isdigit():
                words.append(cleaned)
        if len(words) >= 3:
            break
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    if not unique_words:
        unique_words = [recipe.get("title", "recipe").lower()]
    return " ".join(unique_words[:3])


def _is_distinctive_title(title):
    # Short, generic titles ("Biryani", "Lemonade") have many close variants
    # in a large dataset and get crowded out of a tight top-k by richer
    # recipes mentioning the same word more often — that's a ranking nuance,
    # not a retrieval bug, and it makes those titles noisy as eval cases.
    return len(title.split()) >= 3 or len(title) >= 14


def _mangle_word(word):
    # Simulates a realistic single-typo mistake (adjacent-character
    # transposition, e.g. "chicken" -> "chikcen") rather than a random edit.
    if len(word) < 4:
        return word
    mid = len(word) // 2
    chars = list(word)
    chars[mid], chars[mid + 1] = chars[mid + 1], chars[mid]
    return "".join(chars)


def _typo_variant(query):
    words = query.split()
    if not words:
        return query
    target = max(range(len(words)), key=lambda i: len(words[i]))
    words[target] = _mangle_word(words[target])
    return " ".join(words)


def build_default_cases(index, max_cases):
    # Each selected title produces two cases sharing the same expected
    # title: the exact title, and a typo-mangled version of it. This tests
    # typo tolerance directly against whatever the current dataset contains,
    # instead of hardcoding dish names that may or may not be indexed.
    cases = []
    seen_titles = set()
    for recipe in index.recipes:
        title = recipe.get("title", "").strip()
        if not title or title in seen_titles or not _is_distinctive_title(title):
            continue
        seen_titles.add(title)
        cases.append({"query": title, "expected_recipe_titles": [title], "kind": "exact"})
        typo_query = _typo_variant(title)
        if typo_query.lower() != title.lower():
            cases.append({"query": typo_query, "expected_recipe_titles": [title], "kind": "typo"})
        if len(seen_titles) >= max_cases:
            break

    # Two off-topic probes: a clean case with no shared vocabulary, and an
    # adversarial one that coincidentally shares a real ingredient word
    # ("kernel" as in corn kernel) with the dataset despite being unrelated.
    cases.append(
        {
            "query": "How do I fix a JavaScript null pointer exception?",
            "expected_recipe_titles": [],
            "expect_no_result": True,
            "kind": "no_result",
        }
    )
    cases.append(
        {
            "query": "How do I repair a Linux kernel?",
            "expected_recipe_titles": [],
            "expect_no_result": True,
            "kind": "no_result",
        }
    )
    return cases


def evaluate_case(case, k):
    result = retrieve(case["query"], k=k, debug=True)
    retrieved_titles = [hit.recipe.get("title", "") for hit in result.hits]
    expected_titles = case.get("expected_recipe_titles", [])
    hit = any(title in retrieved_titles[:k] for title in expected_titles)
    expect_no_result = case.get("expect_no_result", False)
    no_result_ok = expect_no_result and not retrieved_titles

    print("Query:")
    print(case["query"])
    print("Retrieved:")
    if retrieved_titles:
        for position, retrieved_title in enumerate(retrieved_titles, start=1):
            score = result.hits[position - 1].fused_score
            print(f"{position}. {retrieved_title}")
            print(f"   score: {score:.3f}")
    else:
        print("(none)")
    print("Expected:")
    if expected_titles:
        print(", ".join(expected_titles))
    else:
        print("(none)")
    print(f"Hit@{k}:")
    print("YES" if hit else "NO")
    print(f"No-result handling:")
    print("YES" if no_result_ok else "NO")
    print("Timings:")
    print(json.dumps({key: round(value, 4) for key, value in result.timings.items()}, indent=2, sort_keys=True))
    print()
    return {
        "hit": hit,
        "no_result_ok": no_result_ok,
        "retrieved_count": len(retrieved_titles),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Zaira retrieval quality on local recipe queries.")
    parser.add_argument("--k", type=int, default=2, help="Top-k cutoff for hit rate")
    parser.add_argument("--max-cases", type=int, default=5, help="How many auto-generated recipe cases to test")
    args = parser.parse_args()

    index = get_index()
    cases = build_default_cases(index, args.max_cases)
    results = [evaluate_case(case, args.k) for case in cases]

    def rate(values):
        return sum(1 for value in values if value) / max(1, len(values))

    hits_by_kind = {"exact": [], "typo": []}
    no_result_hits = []
    for case, result in zip(cases, results):
        kind = case.get("kind")
        if kind in hits_by_kind:
            hits_by_kind[kind].append(result["hit"])
        if case.get("expect_no_result", False):
            no_result_hits.append(result["no_result_ok"])

    print("Summary:")
    print(f"Exact-title Hit@{args.k}: {rate(hits_by_kind['exact']):.2%} (n={len(hits_by_kind['exact'])})")
    print(f"Typo-variant Hit@{args.k}: {rate(hits_by_kind['typo']):.2%} (n={len(hits_by_kind['typo'])})")
    print(f"No-result handling: {rate(no_result_hits):.2%} (n={len(no_result_hits)})")


if __name__ == "__main__":
    main()