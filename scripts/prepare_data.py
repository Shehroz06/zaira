import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.recipes import infer_recipe_metadata, parse_duration_minutes  # noqa: E402

TITLE_KEYS = ["title", "name", "recipe_name", "recipename"]
INGREDIENT_KEYS = ["ingredients", "recipeingredientparts", "ingredient_list"]
STEP_KEYS = ["steps", "instructions", "directions", "recipeinstructions"]
PREP_TIME_KEYS = ["preptime", "prep_time"]
COOK_TIME_KEYS = ["cooktime", "cook_time"]
TOTAL_TIME_KEYS = ["totaltime", "total_time"]


def pick_column(fieldnames, candidates):
    lookup = {f.lower().replace("_", "").replace(" ", ""): f for f in fieldnames}
    for c in candidates:
        key = c.lower().replace("_", "").replace(" ", "")
        if key in lookup:
            return lookup[key]
    return None


def format_list_item(item):
    if isinstance(item, dict):
        quantity = str(item.get("quantity", "")).strip()
        unit = str(item.get("unit", "")).strip()
        name = str(item.get("name", "")).strip()
        return " ".join(p for p in [quantity, unit, name] if p)
    return str(item).strip()


def parse_list_field(value):
    if not value:
        return []
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            return [format_list_item(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    if value.lower().startswith("c(") and value.endswith(")"):
        inner = value[2:-1]
        # Each element is a quoted string that can itself contain commas
        # ("Add onions, garlic, and salt."), so a naive split(",") shreds a
        # single step into fragments at every internal comma. Extract the
        # quoted segments instead of splitting on commas directly.
        quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', inner)
        if quoted:
            return [q.replace('\\"', '"').strip() for q in quoted if q.strip()]
        return [p.strip().strip("\"'") for p in inner.split(",") if p.strip()]
    for sep in ["\n", ";", "|"]:
        if sep in value:
            return [p.strip() for p in value.split(sep) if p.strip()]
    return [value]


def main():
    parser = argparse.ArgumentParser(
        description="Convert a raw Kaggle recipes CSV into data/recipes.json"
    )
    parser.add_argument("csv_path", help="Path to the downloaded Kaggle CSV")
    parser.add_argument("--out", default="data/recipes.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, default=2000, help="Max number of recipes to keep")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        title_col = pick_column(fieldnames, TITLE_KEYS)
        ingredients_col = pick_column(fieldnames, INGREDIENT_KEYS)
        steps_col = pick_column(fieldnames, STEP_KEYS)
        prep_time_col = pick_column(fieldnames, PREP_TIME_KEYS)
        cook_time_col = pick_column(fieldnames, COOK_TIME_KEYS)
        total_time_col = pick_column(fieldnames, TOTAL_TIME_KEYS)

        if not (title_col and ingredients_col and steps_col):
            print("Could not confidently detect columns.")
            print(f"Columns found in CSV: {fieldnames}")
            print(f"Detected -> title: {title_col}, ingredients: {ingredients_col}, steps: {steps_col}")
            print("Edit TITLE_KEYS / INGREDIENT_KEYS / STEP_KEYS at the top of this script to match your file.")
            sys.exit(1)

        print(f"Using columns -> title: {title_col}, ingredients: {ingredients_col}, steps: {steps_col}")
        print(f"Time columns -> prep: {prep_time_col}, cook: {cook_time_col}, total: {total_time_col}")

        recipes = []
        for row in reader:
            title = (row.get(title_col) or "").strip()
            ingredients = parse_list_field(row.get(ingredients_col))
            steps = parse_list_field(row.get(steps_col))
            if not title or not ingredients or not steps:
                continue
            recipe = {
                "title": title,
                "ingredients": ingredients,
                "steps": steps,
            }
            if prep_time_col:
                prep_time = parse_duration_minutes(row.get(prep_time_col))
                if prep_time is not None:
                    recipe["prep_time"] = prep_time
            if cook_time_col:
                cook_time = parse_duration_minutes(row.get(cook_time_col))
                if cook_time is not None:
                    recipe["cook_time"] = cook_time
            if total_time_col:
                total_time = parse_duration_minutes(row.get(total_time_col))
                if total_time is not None:
                    recipe["total_time"] = total_time
            recipes.append(infer_recipe_metadata(recipe))
            if len(recipes) >= args.limit:
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(recipes, indent=2))
    print(f"Wrote {len(recipes)} recipes to {out_path}")


if __name__ == "__main__":
    main()
