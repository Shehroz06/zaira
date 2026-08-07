import json
import logging
import os

from app.llm_service import LLMService
from app.rag import retrieve

logger = logging.getLogger(__name__)
DEBUG_RETRIEVAL = os.environ.get("ZAIRA_DEBUG_RETRIEVAL", "0") == "1"
RESULT_COUNT = int(os.environ.get("ZAIRA_RESULT_COUNT", "4"))
LLM = LLMService()

# Marks the boundary between the streamed intro text and the trailing recipe
# card JSON, so the frontend can render the intro live while it arrives and
# only parse the tail once the stream ends. Kept as plain text (rather than
# switching /chat to SSE/JSON) so the response stays a single text/plain
# stream and main.py doesn't need to change.
RECIPES_DELIMITER = "\n<<<ZAIRA_RECIPES_JSON>>>\n"


def _build_intro_prompt(user_query, retrieval_result):
    titles = [hit.recipe.get("title", "").strip() for hit in retrieval_result.hits if hit.recipe.get("title")]
    intent_summary = json.dumps(retrieval_result.intent.__dict__, sort_keys=True)

    return f"""
You are Zaira, a local cooking assistant.

Write one or two short, friendly sentences introducing the recipe options
below for the user's request. Do not list ingredients or steps — those are
shown separately as cards. Do not mention retrieval, the dataset, recipe
numbers, or internal ranking. Reply with the sentences only — no
surrounding quotation marks, no markdown, no preamble. Do not open with a
generic greeting or stock phrase like "Welcome to our kitchen",
"Let's get cooking", or similar — get straight to the point about the
request and the options found.

User request:
{user_query}

Detected constraints:
{intent_summary}

Recipe titles found:
{", ".join(titles)}
""".strip()


def _recipe_card(recipe):
    card = {"title": recipe.get("title", "").strip()}
    for field in ("ingredients", "steps", "meal_type", "diet", "equipment"):
        values = recipe.get(field)
        if values:
            card[field] = values
    for field in ("prep_time", "cook_time", "total_time"):
        value = recipe.get(field)
        if value is not None:
            card[field] = value
    return card


def zaira_response_stream(user_query):
    try:
        understanding = LLM.resolve_query_intent(user_query)
        retrieval_result = retrieve(user_query, k=RESULT_COUNT, debug=DEBUG_RETRIEVAL, intent=understanding.intent)
    except Exception as exc:
        logger.warning("Retrieval failed: %s", exc)
        yield "I could not search the local recipe index right now. Please try again in a moment."
        return

    if DEBUG_RETRIEVAL and retrieval_result.debug_lines:
        logger.info("\n".join(retrieval_result.debug_lines))
        logger.info("Retrieval timings: %s", retrieval_result.timings)
        logger.info("Query understanding provider: %s", understanding.provider)
        if understanding.fallback_used:
            logger.info("Structured intent fallback used: %s", understanding.error or "unknown error")

    if not retrieval_result.accepted or not retrieval_result.hits:
        yield (
            f"No local recipe match for '{user_query}'. "
            "Try adding an ingredient, meal type, or a time constraint."
        )
        return

    prompt = _build_intro_prompt(user_query, retrieval_result)

    try:
        yield from LLM.stream_answer(prompt, query=user_query)
    except Exception as exc:
        logger.warning("LLM generation failed: %s", exc)
        yield "Here's what I found:"

    cards = [_recipe_card(hit.recipe) for hit in retrieval_result.hits]
    yield RECIPES_DELIMITER + json.dumps(cards)
