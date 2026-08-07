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


def build_default_cases(index, max_cases):
    cases = []
    seen_titles = set()
    for recipe in index.recipes:
        title = recipe.get("title", "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        query = title
        cases.append({"query": query, "expected_recipe_titles": [title]})
        if len(cases) >= max_cases:
            break

    cases.append(
        {
            "query": "How do I repair a Linux kernel?",
            "expected_recipe_titles": [],
            "expect_no_result": True,
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

    hit_rate = sum(1 for item in results[:-1] if item["hit"]) / max(1, len(results) - 1)
    no_result_tests = [case for case in cases if case.get("expect_no_result", False)]
    no_result_rate = sum(1 for item in results if item["no_result_ok"]) / max(1, len(no_result_tests))
    print("Summary:")
    print(f"Hit@{args.k}: {hit_rate:.2%}")
    print(f"No-result handling: {no_result_rate:.2%}")


if __name__ == "__main__":
    main()