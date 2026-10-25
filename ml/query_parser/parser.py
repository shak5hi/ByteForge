"""Rule-based query parser: regex + category gazetteer.

Handles "Raja Hotel near Bennett University", "restaurants near Bennett University",
"cafes within 2 km of Pari Chowk", "hotels in Greater Noida", "pharmacy near me",
and degrades gracefully (ambiguous=True + notes) when parts are missing.
Upgrade path: add a landmark gazetteer from the DB for fuzzy reference matching;
add a learned model only if error analysis shows the rules failing.
"""
from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

from ml.schemas import ParsedQuery

# canonical -> synonyms (extend from real data)
CATEGORIES: dict[str, list[str]] = {
    "restaurant": ["restaurant", "eatery", "dhaba", "food place"],
    "hotel": ["hotel", "lodge", "guest house", "guesthouse", "inn"],
    "cafe": ["cafe", "café", "coffee shop", "coffee"],
    "hospital": ["hospital", "clinic", "nursing home"],
    "pharmacy": ["pharmacy", "chemist", "medical store", "medicine shop"],
    "atm": ["atm", "cash machine"],
    "bank": ["bank"],
    "school": ["school"],
    "gym": ["gym", "fitness centre", "fitness center"],
    "supermarket": ["supermarket", "grocery store", "grocery", "kirana"],
    "petrol pump": ["petrol pump", "gas station", "fuel station", "petrol station"],
}

_LOOKUP: dict[str, str] = {}
for canon, syns in CATEGORIES.items():
    for s in syns:
        variants = [s, s + "s", s + "es"]
        if s.endswith("y"):  # pharmacy -> pharmacies
            variants.append(s[:-1] + "ies")
        for variant in variants:
            _LOOKUP[variant.lower()] = canon
_CAT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in sorted(_LOOKUP, key=len, reverse=True)) + r")\b",
    re.I,
)

_WITHIN_RE = re.compile(
    r"\bwithin\s+(?P<n>\d+(?:\.\d+)?)\s*(?P<u>km|kms|kilometers?|kilometres?|miles?|meters?|metres?|m)\b\s*(?:of|from)?",
    re.I,
)
_REL_RE = re.compile(r"\b(?P<rel>nearby|near to|near|close to|around|next to|beside|inside|in|at)\b", re.I)
_REL_MAP = {
    "nearby": "near", "near to": "near", "near": "near", "close to": "near",
    "around": "near", "next to": "near", "beside": "near",
    "inside": "in", "in": "in", "at": "at",
}
_UNIT_KM = {"km": 1.0, "m": 0.001, "mile": 1.609}
_FILLER_RE = re.compile(
    r"^(?:find me|find|show me|show|search for|search|get me|get|take me to|looking for|"
    r"i am looking for|i'm looking for|i need|i want|list of|best|top|good|cheap|famous|"
    r"nearest|closest|any|some|an|a|the)\s+",
    re.I,
)
_USER_LOC = {"me", "my location", "here", "my place", "my current location"}
_USER_LOC_RE = re.compile(r"\b(?:(?:near|around|close to|nearby)\s+(?:me|my\s+(?:current\s+)?(?:location|place))|nearby)\b", re.I)

# Price / opening-hours phrases we cannot search on yet: strip them so they don't become part of the name.
_NOISE_RES = [
    re.compile(
        r"\b(?:under|below|less than|upto|up to)\s*(?:rs\.?|inr|₹)?\s*\d[\d,]*(?!\d)(?!\s*(?:km|kms|kilomet|mile|met|m\b))",
        re.I,
    ),
    re.compile(r"\bopen\s+(?:now|24\s*(?:hours?|hrs?|x\s*7))\b", re.I),
    re.compile(r"\b24\s*(?:hours?|hrs?)\b|\b24\s*[x/]\s*7\b", re.I),
]

# Conversational phrasing -> category word.
_I = r"(?:i|we)"
_INTENTS = [
    (re.compile(
        rf"\b(?:where\s+(?:can|do|could|should)\s+{_I}\s+(?:eat|get\s+food|have\s+(?:lunch|dinner|breakfast))"
        rf"|where\s+to\s+eat|(?:places?|somewhere|something)\s+to\s+eat|(?:i\s+am|i['’]?m)\s+hungry)\b", re.I),
     "restaurants"),
    (re.compile(rf"\b(?:where\s+(?:can|do|could|should)\s+{_I}\s+stay|where\s+to\s+stay|(?:places?|somewhere)\s+to\s+stay)\b", re.I),
     "hotels"),
    (re.compile(rf"\b(?:where\s+(?:can|do)\s+{_I}\s+(?:withdraw|get)\s+cash|withdraw\s+cash)\b", re.I), "atm"),
    (re.compile(rf"\b(?:where\s+(?:can|do)\s+{_I}\s+(?:buy|get)\s+medicines?|buy\s+medicines?)\b", re.I), "pharmacy"),
]

# Single-word category spellings eligible for typo matching. Short words are excluded
# (too many false positives) and "dhaba" is excluded because of "Abu Dhabi".
_FUZZY_KEYS = [k for k in _LOOKUP if " " not in k and len(k) >= 5 and k != "dhaba"]


def _unit_to_km(u: str) -> float:
    u = u.lower()
    if u.startswith("k"):
        return _UNIT_KM["km"]
    if u.startswith("mi"):
        return _UNIT_KM["mile"]
    return _UNIT_KM["m"]


def _split_category(text: str) -> tuple[str | None, str | None, str]:
    m = _CAT_RE.search(text)
    if not m:
        return None, None, text.strip()
    rest = " ".join((text[: m.start()] + " " + text[m.end():]).split())
    return _LOOKUP[m.group(0).lower()], m.group(0), rest


def _fuzzy_category(text: str) -> tuple[str | None, str | None, str]:
    """Typo-tolerant category lookup: 'restarant' -> restaurant, 'Hotal' -> hotel."""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        t = tok.lower().strip(",.-")
        if len(t) < 5:
            continue
        limit = 1 if len(t) <= 6 else 2
        best: tuple[int, str] | None = None
        for key in _FUZZY_KEYS:
            if abs(len(key) - len(t)) > limit:
                continue
            d = Levenshtein.distance(t, key)
            if d <= limit and (best is None or d < best[0]):
                best = (d, key)
        if best:
            return _LOOKUP[best[1]], tok, " ".join(tokens[:i] + tokens[i + 1:])
    return None, None, text.strip()


def _trailing_category(text: str) -> tuple[str | None, str | None, str]:
    """Category at the very end of `text` ('Bennett University restaurants')."""
    text = text.strip()
    matches = list(_CAT_RE.finditer(text))
    if matches and matches[-1].end() == len(text):
        m = matches[-1]
        return _LOOKUP[m.group(0).lower()], m.group(0), text[: m.start()].strip(" ,.-")
    return None, None, text


def _clean_front(left: str) -> str:
    left = left.strip()
    while True:
        new = _FILLER_RE.sub("", left, count=1)
        if new == left:
            return left
        left = new.strip()


def parse_query(query: str) -> ParsedQuery:
    raw = (query or "").strip()
    out = ParsedQuery(raw=raw)
    if not raw:
        out.ambiguous = True
        out.notes.append("empty query")
        return out

    text = raw
    for noise in _NOISE_RES:
        for m in noise.finditer(text):
            out.notes.append(f"ignored unsupported filter: '{m.group(0).strip()}'")
        text = noise.sub(" ", text)
    for pattern, category_word in _INTENTS:
        text = pattern.sub(category_word, text)
    text = " ".join(text.split())

    radius_m = _WITHIN_RE.search(text)
    if radius_m:
        out.relation = "within"
        out.radius_km = round(float(radius_m["n"]) * _unit_to_km(radius_m["u"]), 3)
        left, right = text[: radius_m.start()], text[radius_m.end():]
        if _USER_LOC_RE.search(left):  # "hotels near me within 5 km"
            left = _USER_LOC_RE.sub(" ", left)
            out.use_user_location = True
    else:
        rel_m = _REL_RE.search(text)
        if rel_m:
            out.relation = _REL_MAP[rel_m["rel"].lower()]
            left, right = text[: rel_m.start()], text[rel_m.end():]
        else:
            left, right = text, ""

    left = _clean_front(left)
    out.category, out.category_text, name_part = _split_category(left)
    if out.category is None:
        out.category, out.category_text, name_part = _fuzzy_category(left)
        if out.category:
            out.notes.append(f"corrected category spelling: '{out.category_text}' -> '{out.category}'")
    name = name_part.strip(" ,.-")

    ref = right.strip(" ,.-")
    if ref.lower() in _USER_LOC:
        out.use_user_location, ref = True, ""

    # Landmark-first word order: "Bennett University nearby restaurants", "near Bennett University restaurants".
    if out.category is None and ref:
        if name:
            c, t, rest = _split_category(ref)
            if c and not rest.strip(" ,.-"):
                out.category, out.category_text, ref, name = c, t, name, ""
                ref = name_part.strip(" ,.-")
                out.notes.append("category found after the landmark; reordered")
        else:
            c, t, rest = _trailing_category(ref)
            if c and rest:
                out.category, out.category_text, ref = c, t, rest
                out.ambiguous = True
                out.notes.append("category taken from the end of the landmark; could be part of its name")

    out.name = name or None
    out.reference = ref or None

    # Confidence + ambiguity notes
    score = 0.0
    if out.category or out.name:
        score += 0.4
    if out.reference:
        score += 0.4
    if out.relation != "none":
        score += 0.2
    out.confidence = round(score, 2)

    if out.relation != "none" and not out.reference and not out.use_user_location:
        out.use_user_location = True
        out.notes.append("relation without a landmark; assuming the user's location")
    if out.relation == "none" and not out.reference:
        out.notes.append("no location reference; search globally or ask the user")
    if not out.category and not out.name:
        out.ambiguous = True
        out.notes.append("no category or business name found")
    if out.category and out.name and out.relation == "none":
        out.ambiguous = True
        out.notes.append("could not tell whether the name is a business or a place")
    return out
