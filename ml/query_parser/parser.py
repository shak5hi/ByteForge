"""Rule-based query parser: regex + category gazetteer.

Handles "Raja Hotel near Bennett University", "restaurants near Bennett University",
"cafes within 2 km of Pari Chowk", "hotels in Greater Noida", "pharmacy near me",
and degrades gracefully (ambiguous=True + notes) when parts are missing.
Upgrade path: add a landmark gazetteer from the DB for fuzzy reference matching;
add a learned model only if error analysis shows the rules failing.
"""
from __future__ import annotations

import re

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
        for variant in (s, s + "s", s + "es"):
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
_FILLER_RE = re.compile(r"^(?:find|show me|show|search for|search|get me|best|top|good|cheap|famous)\s+", re.I)
_USER_LOC = {"me", "my location", "here", "my place", "my current location"}


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


def parse_query(query: str) -> ParsedQuery:
    raw = (query or "").strip()
    out = ParsedQuery(raw=raw)
    if not raw:
        out.ambiguous = True
        out.notes.append("empty query")
        return out

    text = raw
    radius_m = _WITHIN_RE.search(text)
    if radius_m:
        out.relation = "within"
        out.radius_km = round(float(radius_m["n"]) * _unit_to_km(radius_m["u"]), 3)
        left, right = text[: radius_m.start()], text[radius_m.end():]
    else:
        rel_m = _REL_RE.search(text)
        if rel_m:
            out.relation = _REL_MAP[rel_m["rel"].lower()]
            left, right = text[: rel_m.start()], text[rel_m.end():]
        else:
            left, right = text, ""

    left = _FILLER_RE.sub("", left.strip())
    out.category, out.category_text, name_part = _split_category(left)
    out.name = name_part.strip(" ,.-") or None

    ref = right.strip(" ,.-")
    if ref.lower() in _USER_LOC:
        out.use_user_location, ref = True, ""
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
