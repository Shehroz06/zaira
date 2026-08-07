import re
from copy import deepcopy


MEAT_MARKERS = {
    "beef",
    "bacon",
    "broth",
    "chicken",
    "duck",
    "fish",
    "ham",
    "lamb",
    "meat",
    "pork",
    "prosciutto",
    "salami",
    "sausage",
    "shrimp",
    "turkey",
    "tuna",
}

DAIRY_MARKERS = {
    "butter",
    "cheese",
    "cream",
    "ghee",
    "milk",
    "parmesan",
    "yogurt",
}

OVEN_MARKERS = {"bake", "broil", "oven", "roast"}
STOVETOP_MARKERS = {
    "boil",
    "fry",
    "heat oil",
    "pan",
    "saute",
    "simmer",
    "skillet",
    "stovetop",
    "stir fry",
    "stir-fry",
    "wok",
}
SLOW_COOKER_MARKERS = {"crockpot", "slow cooker"}
GRILL_MARKERS = {"grill", "grilled"}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            return value
        if isinstance(value, int):
            return value
    return None


def _coerce_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


_ISO_DURATION_RE = re.compile(
    r"^P(?:\d+D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$",
    re.IGNORECASE,
)


def parse_duration_minutes(value):
    # Handles ISO-8601 durations like "PT45M" or "PT24H45M" (the format used
    # by the Food.com dataset's CookTime/PrepTime/TotalTime columns). Falls
    # back to _coerce_int for plain integers so other datasets still work.
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _ISO_DURATION_RE.match(text)
    if match and any(match.group(name) for name in ("hours", "minutes", "seconds")):
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = int(match.group("seconds") or 0)
        return hours * 60 + minutes + (1 if seconds else 0)
    return _coerce_int(value)


def normalize_recipe(recipe):
    normalized = deepcopy(recipe)
    normalized["title"] = str(recipe.get("title", "")).strip()
    normalized["ingredients"] = as_list(recipe.get("ingredients"))
    normalized["steps"] = as_list(recipe.get("steps"))

    for field in ("prep_time", "cook_time", "total_time"):
        value = _coerce_int(recipe.get(field))
        if value is not None:
            normalized[field] = value

    for field in ("meal_type", "diet", "equipment"):
        values = as_list(recipe.get(field))
        if values:
            normalized[field] = values

    return normalized


def _text_blob(recipe):
    parts = [
        recipe.get("title", ""),
        " ".join(recipe.get("ingredients", [])),
        " ".join(recipe.get("steps", [])),
        " ".join(recipe.get("meal_type", [])),
        " ".join(recipe.get("diet", [])),
        " ".join(recipe.get("equipment", [])),
    ]
    return " ".join(part for part in parts if part).lower()


def infer_recipe_metadata(recipe):
    normalized = normalize_recipe(recipe)
    text_blob = _text_blob(normalized)

    if not normalized.get("total_time"):
        # Also backfills a degenerate 0 (e.g. the source CSV's PrepTime and
        # CookTime were both blank/"PT0S"), not just a missing value — a
        # 0-minute recipe isn't real data, so treat it the same as absent.
        prep_time = normalized.get("prep_time") or 0
        cook_time = normalized.get("cook_time") or 0
        computed = prep_time + cook_time
        normalized["total_time"] = computed if computed else None

    if not normalized.get("equipment"):
        equipment = []
        if any(marker in text_blob for marker in OVEN_MARKERS):
            equipment.append("oven")
        if any(marker in text_blob for marker in SLOW_COOKER_MARKERS):
            equipment.append("slow cooker")
        if any(marker in text_blob for marker in GRILL_MARKERS):
            equipment.append("grill")
        if any(marker in text_blob for marker in STOVETOP_MARKERS):
            equipment.append("stovetop")
        if equipment:
            normalized["equipment"] = list(dict.fromkeys(equipment))

    if not normalized.get("diet"):
        ingredients_text = " ".join(normalized.get("ingredients", [])).lower()
        if any(marker in ingredients_text for marker in MEAT_MARKERS):
            normalized["diet"] = ["non-vegetarian"]

    return normalized


def canonical_recipe_text(recipe):
    normalized = infer_recipe_metadata(recipe)
    lines = [
        "Recipe Title:",
        normalized.get("title", "").strip(),
        "",
        "Ingredients:",
        ", ".join(normalized.get("ingredients", [])) or "Unknown",
        "",
        "Instructions:",
    ]

    steps = normalized.get("steps", [])
    if steps:
        lines.extend(f"{index + 1}. {step}" for index, step in enumerate(steps))
    else:
        lines.append("Unknown")

    if normalized.get("meal_type"):
        lines.extend(["", "Meal Type:", ", ".join(normalized["meal_type"])])
    if normalized.get("diet"):
        lines.extend(["", "Diet:", ", ".join(normalized["diet"])])
    if normalized.get("equipment"):
        lines.extend(["", "Equipment:", ", ".join(normalized["equipment"])])
    if normalized.get("prep_time") is not None:
        lines.extend(["", "Prep Time:", f"{normalized['prep_time']} minutes"])
    if normalized.get("cook_time") is not None:
        lines.extend(["", "Cook Time:", f"{normalized['cook_time']} minutes"])
    if normalized.get("total_time") is not None:
        lines.extend(["", "Total Time:", f"{normalized['total_time']} minutes"])

    return "\n".join(lines).strip()


def recipe_tokens(recipe):
    normalized = infer_recipe_metadata(recipe)
    token_source = " ".join(
        [
            normalized.get("title", ""),
            " ".join(normalized.get("ingredients", [])),
            " ".join(normalized.get("steps", [])),
            " ".join(normalized.get("meal_type", [])),
            " ".join(normalized.get("diet", [])),
            " ".join(normalized.get("equipment", [])),
        ]
    ).lower()
    return re.findall(r"[a-z0-9]+", token_source)