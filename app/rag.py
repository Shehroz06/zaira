import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import snowballstemmer
from rapidfuzz import fuzz, process

from app.llm import embed
from app.recipes import (
    DAIRY_MARKERS,
    MEAT_MARKERS,
    canonical_recipe_text,
    infer_recipe_metadata,
    recipe_tokens,
)

logger = logging.getLogger(__name__)

# app/rag.py -> project root is one level up from this file's directory.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_RECIPES = DATA_DIR / "recipes.json"
DATASET_EMBEDDINGS = DATA_DIR / "embeddings.npy"
FALLBACK_RECIPES = BASE_DIR / "demo" / "demo_recipes.json"

DEFAULT_K = 2
DEFAULT_MIN_RELEVANCE = float(os.environ.get("ZAIRA_MIN_RELEVANCE", "0.50"))
RRF_K = int(os.environ.get("ZAIRA_RRF_K", "60"))
MAX_RUNTIME_EMBED_RECIPES = int(os.environ.get("ZAIRA_RUNTIME_EMBED_LIMIT", "32"))
SPELLING_CORRECTION_SCORE_CUTOFF = int(os.environ.get("ZAIRA_SPELLING_SCORE_CUTOFF", "80"))
SPELLING_CORRECTION_MIN_TOKEN_LEN = 3
# A token seen in this many or more documents is treated as correctly
# spelled and skipped by fuzzy correction — below it, the token is rare
# enough that it's worth checking whether it's actually a typo of a more
# common word (the source dataset is scraped and contains real misspellings,
# e.g. "chiken" occurring verbatim in a handful of recipes).
SPELLING_CORRECTION_TRUSTED_DF = int(os.environ.get("ZAIRA_SPELLING_TRUSTED_DF", "5"))
# Squashes BM25's unbounded raw score into a 0-1 range comparable to cosine
# similarity, so the acceptance gate isn't comparing incommensurable scales
# (see BM25_ACCEPTANCE_SATURATION usage in retrieve()).
BM25_ACCEPTANCE_SATURATION = float(os.environ.get("ZAIRA_BM25_ACCEPTANCE_SATURATION", "8.0"))
# Tokens occurring in more than this fraction of documents are excluded from
# the BM25 vocabulary entirely — a corpus-derived stopword cutoff, catching
# generic high-frequency words the fixed BM25_STOPWORDS list doesn't name
# (e.g. stemming "exception" -> "except" collides with the very common
# recipe-instruction word "except", as in "chop everything except the...").
BM25_MAX_DOCUMENT_FREQUENCY_RATIO = float(os.environ.get("ZAIRA_BM25_MAX_DF_RATIO", "0.02"))

# Stopwords are excluded from the BM25 index entirely (not just at query
# time): a query like "how do I repair a linux kernel" shares "how"/"do"/"a"
# with huge numbers of unrelated recipes, and summing their BM25 contributions
# alongside real terms can inflate the total score enough to pass the
# acceptance threshold for a completely irrelevant query.
BM25_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "without",
        "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "doing",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
        "how", "what", "when", "where", "why", "who", "which",
        "can", "could", "will", "would", "should", "shall", "may", "might", "must",
        "not", "no", "so", "as", "at", "by", "from", "up", "down", "out", "about", "into", "than", "then",
    }
)

_bm25_stemmer = snowballstemmer.stemmer("english")


def _prepare_bm25_tokens(tokens):
    filtered = [token for token in tokens if token not in BM25_STOPWORDS]
    if not filtered:
        return []
    return _bm25_stemmer.stemWords(filtered)

VEGAN_MARKERS = {"egg", "eggs", "honey"} | DAIRY_MARKERS | MEAT_MARKERS
MEAL_TYPES = {"breakfast", "brunch", "lunch", "dinner", "snack", "dessert"}
EQUIPMENT_TYPES = {"oven", "grill", "slow cooker", "stovetop"}


@dataclass
class QueryIntent:
    ingredients_include: list[str] = field(default_factory=list)
    ingredients_exclude: list[str] = field(default_factory=list)
    diet: list[str] = field(default_factory=list)
    meal_type: list[str] = field(default_factory=list)
    max_time_minutes: int | None = None
    equipment_include: list[str] = field(default_factory=list)
    equipment_exclude: list[str] = field(default_factory=list)
    dish_type: str | None = None
    prefer_quick: bool = False


@dataclass
class RetrievedRecipe:
    recipe: dict
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None


@dataclass
class RetrievalResult:
    query: str
    intent: QueryIntent
    hits: list[RetrievedRecipe]
    accepted: bool
    threshold: float
    timings: dict[str, float] = field(default_factory=dict)
    debug_lines: list[str] = field(default_factory=list)

    @property
    def recipes(self):
        return [hit.recipe for hit in self.hits]


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _split_terms(text):
    parts = re.split(r",|\band\b|\bor\b|/", text)
    terms = []
    for part in parts:
        cleaned = re.sub(r"[^a-z0-9\- ]+", " ", part.lower()).strip()
        cleaned = re.sub(r"\b(?:fresh|ripe|large|small|medium|diced|chopped|minced|thinly sliced|sliced|ground|skinless|boneless|optional|to taste|as needed)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            terms.append(cleaned)
    return terms


def _find_meal_types(text):
    return [meal for meal in MEAL_TYPES if re.search(rf"\b{re.escape(meal)}\b", text)]


def _find_equipment_exclusions(text):
    exclusions = []
    if re.search(r"\b(?:no|without|avoid|don't have|dont have|no access to)\s+oven\b", text):
        exclusions.append("oven")
    if re.search(r"\b(?:no|without|avoid)\s+grill(?:ing)?\b", text):
        exclusions.append("grill")
    if re.search(r"\b(?:no|without|avoid)\s+slow cooker\b", text):
        exclusions.append("slow cooker")
    if re.search(r"\b(?:no|without|avoid)\s+(?:stove|stovetop|pan|skillet|wok)\b", text):
        exclusions.append("stovetop")
    return list(dict.fromkeys(exclusions))


def _find_equipment_inclusions(text):
    includes = []
    if re.search(r"\b(?:need|want|have)\s+no?\s*oven\b", text) or re.search(r"\boven\b", text):
        includes.append("oven")
    if re.search(r"\bslow cooker\b", text):
        includes.append("slow cooker")
    if re.search(r"\bgrill(?:ed|ing)?\b", text):
        includes.append("grill")
    if re.search(r"\b(?:stovetop|skillet|pan|wok|simmer|saute|sauteed|boil|fry)\b", text):
        includes.append("stovetop")
    return list(dict.fromkeys(includes))


def _find_time_limit(text):
    explicit = [
        re.search(r"\b(?:under|less than|within|in)\s+(\d{1,3})\s*(?:minutes?|mins?|min)\b", text),
        re.search(r"\b(?:takes|take|takes about|about)\s+(\d{1,3})\s*(?:minutes?|mins?|min)\b", text),
    ]
    for match in explicit:
        if match:
            return int(match.group(1))
    if re.search(r"\bquick\b|\bfast\b|\bsimple\b", text):
        return 30
    return None


def _extract_included_ingredients(text):
    matches = []
    patterns = [
        r"\b(?:with|using|have|got|containing|including|made with|want|need|looking for|craving)\s+([^.?;!]+)",
        r"\bi have\s+([^.?;!]+)",
        r"\bwhat can i make with\s+([^.?;!]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            snippet = match.group(1)
            snippet = re.split(r"\b(?:without|no|avoid|except|but not|under|less than|within)\b", snippet, maxsplit=1)[0]
            matches.extend(_split_terms(snippet))
    cleaned = []
    for term in matches:
        if term not in MEAL_TYPES and term not in EQUIPMENT_TYPES:
            cleaned.append(term)
    return list(dict.fromkeys(cleaned))


def _extract_excluded_ingredients(text):
    matches = []
    patterns = [
        r"\b(?:without|avoid|no|don't have|dont have|exclude|excluding)\s+([^.?;!]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            snippet = match.group(1)
            snippet = re.split(r"\b(?:under|less than|within|for|please)\b", snippet, maxsplit=1)[0]
            matches.extend(_split_terms(snippet))

    cleaned = []
    for term in matches:
        if term in {"oven", "grill", "slow cooker", "stovetop"}:
            continue
        cleaned.append(term)

    if re.search(r"\b(?:no|without|avoid)\s+dairy\b|\bdairy-free\b", text):
        cleaned.extend(sorted(DAIRY_MARKERS))
    if re.search(r"\b(?:no|without|avoid)\s+meat\b|\bvegetarian\b", text):
        cleaned.extend(sorted(MEAT_MARKERS))

    return list(dict.fromkeys(cleaned))


def parse_query_intent(query):
    text = _normalize_text(query)
    intent = QueryIntent()
    intent.ingredients_include = _extract_included_ingredients(text)
    intent.ingredients_exclude = _extract_excluded_ingredients(text)
    intent.meal_type = _find_meal_types(text)
    intent.max_time_minutes = _find_time_limit(text)
    intent.prefer_quick = bool(re.search(r"\bquick\b|\bfast\b|\bsimple\b", text))
    intent.equipment_exclude = _find_equipment_exclusions(text)
    intent.equipment_include = _find_equipment_inclusions(text)

    if re.search(r"\bvegetarian\b", text):
        intent.diet.append("vegetarian")
    if re.search(r"\bvegan\b", text):
        intent.diet.append("vegan")
    if re.search(r"\bdairy-free\b|\bno dairy\b", text):
        intent.diet.append("dairy-free")
    if re.search(r"\bgluten-free\b|\bno gluten\b", text):
        intent.diet.append("gluten-free")

    deduped_diet = list(dict.fromkeys(intent.diet))
    intent.diet = deduped_diet
    return intent


def _intent_tokens(query, intent):
    base_tokens = re.findall(r"[a-z0-9]+", _normalize_text(query))
    extra_tokens = []
    for values in (
        intent.ingredients_include,
        intent.ingredients_exclude,
        intent.diet,
        intent.meal_type,
        intent.equipment_include,
        intent.equipment_exclude,
    ):
        extra_tokens.extend(re.findall(r"[a-z0-9]+", " ".join(values)))
    if intent.dish_type:
        extra_tokens.extend(re.findall(r"[a-z0-9]+", intent.dish_type))
    if intent.max_time_minutes is not None:
        extra_tokens.append(str(intent.max_time_minutes))
    return list(dict.fromkeys(base_tokens + extra_tokens))


def _has_any(text, markers):
    return any(marker in text for marker in markers)


def _recipe_text(normalized):
    parts = [
        normalized.get("title", ""),
        " ".join(normalized.get("ingredients", [])),
        " ".join(normalized.get("steps", [])),
        " ".join(normalized.get("meal_type", [])),
        " ".join(normalized.get("diet", [])),
        " ".join(normalized.get("equipment", [])),
    ]
    return _normalize_text(" ".join(part for part in parts if part))


@dataclass
class RecipeFilterProfile:
    # Everything a recipe needs for _recipe_satisfies, computed once per
    # recipe at index-build time instead of on every /chat request. At
    # dataset sizes in the hundreds of thousands, redoing these string joins
    # and regex substitutions per request turns a single query into a
    # multi-second (or multi-minute) filtering pass.
    meal_type: frozenset
    equipment_set: frozenset
    equipment_text: str
    text_blob: str
    total_time: int | None
    has_meat: bool
    has_dairy: bool
    has_vegan_marker: bool
    diet_is_nonveg: bool
    diet_is_veg_or_nonveg: bool


def _build_filter_profile(normalized):
    ingredient_text = _normalize_text(" ".join(normalized.get("ingredients", [])))
    step_text = _normalize_text(" ".join(normalized.get("steps", [])))
    diet_text = _normalize_text(" ".join(normalized.get("diet", [])))
    equipment_text = _normalize_text(" ".join(normalized.get("equipment", [])))
    text_blob = f"{ingredient_text} {step_text} {_recipe_text(normalized)}"
    # infer_recipe_metadata already backfills total_time from prep/cook time
    # (or leaves it None if none of the three are real data), so this is the
    # single resolved value — no need to re-derive it from the raw fields.
    total_time = normalized.get("total_time")

    return RecipeFilterProfile(
        meal_type=frozenset(normalized.get("meal_type", [])),
        equipment_set=frozenset(normalized.get("equipment", [])),
        equipment_text=equipment_text,
        text_blob=text_blob,
        total_time=total_time,
        has_meat=_has_any(ingredient_text, MEAT_MARKERS),
        has_dairy=_has_any(ingredient_text, DAIRY_MARKERS),
        has_vegan_marker=_has_any(ingredient_text, VEGAN_MARKERS),
        diet_is_nonveg=_has_any(diet_text, {"non-vegetarian"}),
        diet_is_veg_or_nonveg=_has_any(diet_text, {"vegetarian", "non-vegetarian"}),
    )


def _recipe_satisfies(profile, intent):
    # A recipe with no real prep/cook/total time data can't show a time badge,
    # which breaks the uniform look of the card list — excluded outright
    # rather than shown with a missing badge. Only ~0.4% of the dataset lacks
    # this data, so the coverage cost is negligible.
    if profile.total_time is None:
        return False

    if intent.meal_type and profile.meal_type:
        if not set(intent.meal_type) & profile.meal_type:
            return False

    if intent.equipment_exclude:
        if any(item in profile.equipment_text or item in profile.text_blob for item in intent.equipment_exclude):
            return False

    if intent.equipment_include:
        if profile.equipment_text and not set(intent.equipment_include) & profile.equipment_set:
            return False

    if intent.max_time_minutes is not None and profile.total_time is not None:
        if profile.total_time > intent.max_time_minutes:
            return False

    if "vegetarian" in intent.diet:
        if profile.has_meat or profile.diet_is_nonveg:
            return False

    if "vegan" in intent.diet:
        if profile.has_vegan_marker or profile.diet_is_veg_or_nonveg:
            return False

    if "dairy-free" in intent.diet and profile.has_dairy:
        return False

    if intent.ingredients_exclude and any(term in profile.text_blob for term in intent.ingredients_exclude):
        return False

    return True


def _intent_bonus(normalized, intent):
    # `normalized` is a recipe already normalized by RecipeIndex.__init__.
    ingredient_text = _normalize_text(" ".join(normalized.get("ingredients", [])))
    bonus = 0.0

    if intent.meal_type and set(intent.meal_type).intersection(normalized.get("meal_type", [])):
        bonus += 0.12
    if intent.diet and set(intent.diet).intersection(normalized.get("diet", [])):
        bonus += 0.12
    if intent.equipment_include and set(intent.equipment_include).intersection(normalized.get("equipment", [])):
        bonus += 0.08
    if intent.max_time_minutes is not None:
        total_time = normalized.get("total_time") or normalized.get("cook_time") or normalized.get("prep_time")
        if total_time is not None and total_time <= intent.max_time_minutes:
            bonus += 0.08

    if intent.ingredients_include:
        matches = sum(1 for term in intent.ingredients_include if term in ingredient_text)
        bonus += min(0.2, 0.08 * matches)

    return bonus


def _rrf_fuse(rankings):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, recipe_id in enumerate(ranking, start=1):
            scores[recipe_id] += 1.0 / (RRF_K + rank)
    return scores


def _normalize_embeddings(embeddings):
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-8)


def _build_runtime_embeddings(recipes):
    texts = [canonical_recipe_text(recipe) for recipe in recipes]
    embeddings = []
    for text in texts:
        embeddings.append(embed(text))
    return _normalize_embeddings(embeddings)


def _load_recipes():
    if DATASET_RECIPES.exists():
        return json.loads(DATASET_RECIPES.read_text()), DATASET_RECIPES
    return json.loads(FALLBACK_RECIPES.read_text()), FALLBACK_RECIPES


class RecipeIndex:
    def __init__(self, recipes, embeddings, source_path):
        self.recipes = [infer_recipe_metadata(recipe) for recipe in recipes]
        self.source_path = source_path
        self.filter_profiles = [_build_filter_profile(recipe) for recipe in self.recipes]
        self.tokens = [_prepare_bm25_tokens(recipe_tokens(recipe)) for recipe in self.recipes]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.doc_lengths = np.asarray([max(1, len(tokens)) for tokens in self.tokens], dtype=np.float32)
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 1.0
        self.document_frequency = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))
        self.embedding_matrix = _normalize_embeddings(embeddings)
        self._idf = self._build_idf()
        self._doc_length_norm = self.doc_lengths / self.avg_doc_length
        self._token_ids, self._postings_doc_ids, self._postings_tfs, self._postings_offsets, self._idf_array = (
            self._build_postings()
        )
        self._vocabulary = list(self._token_ids.keys())
        self._query_cache = {}
        self._spelling_cache = {}

    def _build_idf(self):
        total_documents = max(1, len(self.tokens))
        idf = {}
        for token, document_count in self.document_frequency.items():
            idf[token] = np.log(1.0 + (total_documents - document_count + 0.5) / (document_count + 0.5))
        return idf

    def _build_postings(self):
        # An inverted index (token -> the doc_ids/tfs of docs containing it),
        # built once at startup, so a query only touches the postings of its
        # own tokens instead of looping over every candidate recipe. At
        # dataset sizes in the hundreds of thousands, the naive per-candidate
        # BM25 loop this replaces was the single biggest cost of a request.
        total_documents = max(1, len(self.tokens))
        max_document_frequency = max(1, int(total_documents * BM25_MAX_DOCUMENT_FREQUENCY_RATIO))
        vocabulary = [
            token for token in self._idf.keys() if self.document_frequency.get(token, 0) <= max_document_frequency
        ]
        token_ids = {token: index for index, token in enumerate(vocabulary)}
        idf_array = np.asarray([self._idf[token] for token in vocabulary], dtype=np.float32)

        doc_id_list = []
        token_id_list = []
        tf_list = []
        for doc_id, frequencies in enumerate(self.term_frequencies):
            for token, tf in frequencies.items():
                token_id = token_ids.get(token)
                if token_id is None:
                    continue
                doc_id_list.append(doc_id)
                token_id_list.append(token_id)
                tf_list.append(tf)

        token_id_arr = np.asarray(token_id_list, dtype=np.int32)
        doc_id_arr = np.asarray(doc_id_list, dtype=np.int32)
        tf_arr = np.asarray(tf_list, dtype=np.float32)

        order = np.argsort(token_id_arr, kind="stable")
        sorted_doc_ids = doc_id_arr[order]
        sorted_tfs = tf_arr[order]

        counts = np.bincount(token_id_arr, minlength=len(vocabulary))
        offsets = np.zeros(len(vocabulary) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)

        return token_ids, sorted_doc_ids, sorted_tfs, offsets, idf_array

    def _query_embedding(self, query):
        key = _normalize_text(query)
        if key not in self._query_cache:
            self._query_cache[key] = _normalize_embeddings([embed(key)])[0]
        return self._query_cache[key]

    def _correct_token(self, token):
        # BM25 only matches tokens present in the postings, so a misspelled
        # ingredient/dish word (e.g. "chiken", "briyani") looks up nothing —
        # unlike vector search, which tolerates typos via embedding
        # similarity. Snap unknown tokens to the closest known vocabulary
        # token before lookup so the keyword channel isn't silently blind to
        # them. Short tokens are skipped: at 1-2 chars, edit-distance
        # matching is more likely to overcorrect than fix a real typo.
        #
        # A token already present in the vocabulary isn't necessarily
        # correctly spelled — the source dataset is scraped and contains its
        # own typos (e.g. "chiken" occurs verbatim in a handful of recipes).
        # Only tokens common enough to be trusted (SPELLING_CORRECTION_TRUSTED_DF)
        # skip correction outright; rare tokens still get a chance to snap to
        # a meaningfully more common near-match.
        if len(token) < SPELLING_CORRECTION_MIN_TOKEN_LEN:
            return token
        token_df = self.document_frequency.get(token, 0)
        if token_df >= SPELLING_CORRECTION_TRUSTED_DF:
            return token
        if token not in self._spelling_cache:
            match = process.extractOne(
                token, self._vocabulary, scorer=fuzz.ratio, score_cutoff=SPELLING_CORRECTION_SCORE_CUTOFF
            )
            corrected = token
            if match:
                candidate = match[0]
                if candidate != token and self.document_frequency.get(candidate, 0) > token_df:
                    corrected = candidate
            self._spelling_cache[token] = corrected
        return self._spelling_cache[token]

    def _keyword_scores(self, query_tokens, candidate_ids):
        query_tokens = _prepare_bm25_tokens(query_tokens)
        if not query_tokens:
            return np.zeros(len(candidate_ids), dtype=np.float32)

        query_tokens = [self._correct_token(token) for token in query_tokens]

        k1 = 1.5
        b = 0.75
        scores = np.zeros(len(self.recipes), dtype=np.float32)
        for token in query_tokens:
            token_id = self._token_ids.get(token)
            if token_id is None:
                continue
            start, end = self._postings_offsets[token_id], self._postings_offsets[token_id + 1]
            if start == end:
                continue
            doc_ids = self._postings_doc_ids[start:end]
            tfs = self._postings_tfs[start:end]
            idf = self._idf_array[token_id]
            denom = tfs + k1 * (1.0 - b + b * self._doc_length_norm[doc_ids])
            scores[doc_ids] += idf * (tfs * (k1 + 1.0)) / denom
        return scores[candidate_ids]

    def _candidate_ids(self, intent):
        candidate_ids = [
            index for index, profile in enumerate(self.filter_profiles) if _recipe_satisfies(profile, intent)
        ]
        return candidate_ids

    def retrieve(self, query, k=DEFAULT_K, min_relevance=DEFAULT_MIN_RELEVANCE, debug=False, intent=None):
        timings = {}
        started = perf_counter()
        if intent is None:
            intent = parse_query_intent(query)
        timings["query_parsing"] = perf_counter() - started

        started = perf_counter()
        candidate_ids = self._candidate_ids(intent)
        timings["filtering"] = perf_counter() - started

        if not candidate_ids:
            timings["total"] = sum(timings.values())
            debug_lines = []
            if debug:
                debug_lines.append(f"Query: {query}")
                debug_lines.append(f"Intent: {json.dumps(asdict(intent), sort_keys=True)}")
                debug_lines.append(f"Candidates after filter: 0 / {len(self.recipes)}")
                debug_lines.append(f"Threshold: {min_relevance:.3f}")
                debug_lines.append("Accepted: no")
            return RetrievalResult(
                query=query,
                intent=intent,
                hits=[],
                accepted=False,
                threshold=min_relevance,
                timings=timings,
                debug_lines=debug_lines,
            )

        started = perf_counter()
        query_embedding = self._query_embedding(query)
        timings["query_embedding"] = perf_counter() - started

        started = perf_counter()
        candidate_embeddings = self.embedding_matrix[candidate_ids]
        vector_scores = candidate_embeddings @ query_embedding
        vector_order = [candidate_ids[index] for index in np.argsort(-vector_scores)]
        timings["vector_retrieval"] = perf_counter() - started

        started = perf_counter()
        keyword_tokens = _intent_tokens(query, intent)
        keyword_scores = self._keyword_scores(keyword_tokens, candidate_ids)
        keyword_order = [candidate_ids[index] for index in np.argsort(-keyword_scores)]
        timings["keyword_retrieval"] = perf_counter() - started

        started = perf_counter()
        fused = _rrf_fuse([vector_order[: max(20, k * 5)], keyword_order[: max(20, k * 5)]])
        ranked_ids = sorted(fused.keys(), key=lambda recipe_id: fused[recipe_id], reverse=True)
        timings["fusion"] = perf_counter() - started

        hits = []
        for rank, recipe_id in enumerate(ranked_ids, start=1):
            vector_rank = vector_order.index(recipe_id) + 1 if recipe_id in vector_order else None
            keyword_rank = keyword_order.index(recipe_id) + 1 if recipe_id in keyword_order else None
            vector_score = float(vector_scores[candidate_ids.index(recipe_id)]) if recipe_id in candidate_ids else 0.0
            keyword_score = float(keyword_scores[candidate_ids.index(recipe_id)]) if recipe_id in candidate_ids else 0.0
            fused_score = float(fused[recipe_id] + _intent_bonus(self.recipes[recipe_id], intent))
            hits.append(
                RetrievedRecipe(
                    recipe=self.recipes[recipe_id],
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    fused_score=fused_score,
                    vector_rank=vector_rank,
                    keyword_rank=keyword_rank,
                )
            )

        hits = hits[:k]
        best_vector_score = max((hit.vector_score for hit in hits), default=0.0)
        best_keyword_score = max((hit.keyword_score for hit in hits), default=0.0)
        # BM25's raw score is unbounded (can run into double digits), unlike
        # cosine similarity's [0, 1] range — comparing it directly against
        # min_relevance let any query sharing even a weak, spurious token
        # with a recipe (e.g. "kernel" in "How do I repair a Linux kernel?"
        # matching "Whole Kernel Corn") pass the gate. Squash it onto a
        # comparable scale before gating.
        normalized_keyword_score = best_keyword_score / (best_keyword_score + BM25_ACCEPTANCE_SATURATION)
        accepted = bool(hits) and (best_vector_score >= min_relevance or normalized_keyword_score >= 0.5)

        debug_lines = []
        if debug:
            debug_lines.append(f"Query: {query}")
            debug_lines.append(f"Intent: {json.dumps(asdict(intent), sort_keys=True)}")
            debug_lines.append(f"Candidates after filter: {len(candidate_ids)} / {len(self.recipes)}")
            for position, hit in enumerate(hits, start=1):
                debug_lines.append(
                    f"{position}. {hit.recipe.get('title', 'Untitled')} | vector={hit.vector_score:.3f} | keyword={hit.keyword_score:.3f} | fused={hit.fused_score:.3f}"
                )
            debug_lines.append(f"Threshold: {min_relevance:.3f}")
            debug_lines.append(f"Accepted: {'yes' if accepted else 'no'}")

        timings["total"] = sum(timings.values())
        return RetrievalResult(
            query=query,
            intent=intent,
            hits=hits if accepted else [],
            accepted=accepted,
            threshold=min_relevance,
            timings=timings,
            debug_lines=debug_lines,
        )


def _load_index():
    recipes, source_path = _load_recipes()
    embeddings_path_exists = DATASET_EMBEDDINGS.exists()
    embeddings = None

    if embeddings_path_exists:
        embeddings = np.load(DATASET_EMBEDDINGS, allow_pickle=False)
        if embeddings.ndim != 2:
            embeddings = None
        elif embeddings.shape[0] < len(recipes):
            # A partial embeddings.npy (e.g. promoted from an in-progress
            # scripts/build_index.py checkpoint) is usable — recipes are
            # embedded in dataset order, so the first N recipes line up with
            # the first N embedding rows. Serve that prefix instead of
            # rejecting the file outright.
            logger.info(
                "Using a partial index: %d/%d recipes have precomputed embeddings.",
                embeddings.shape[0],
                len(recipes),
            )
            recipes = recipes[: embeddings.shape[0]]
        elif embeddings.shape[0] > len(recipes):
            embeddings = None

    if embeddings is None:
        if len(recipes) > MAX_RUNTIME_EMBED_RECIPES:
            raise RuntimeError(
                "No usable embeddings.npy found for a dataset larger than the runtime fallback limit. "
                "Run scripts/build_index.py to precompute embeddings before starting the app."
            )
        embeddings = _build_runtime_embeddings(recipes)
        if source_path == DATASET_RECIPES:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.save(DATASET_EMBEDDINGS, embeddings)

    return RecipeIndex(recipes, embeddings, source_path)


@lru_cache(maxsize=1)
def get_index():
    return _load_index()


def retrieve(query, k=DEFAULT_K, min_relevance=DEFAULT_MIN_RELEVANCE, debug=False, intent=None):
    return get_index().retrieve(query, k=k, min_relevance=min_relevance, debug=debug, intent=intent)
