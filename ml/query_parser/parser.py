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
    # Hinglish (Roman-script Hindi) synonyms are listed after the English ones.
    "restaurant": ["restaurant", "eatery", "dhaba", "food place", "khana", "khane ki jagah"],
    "hotel": ["hotel", "lodge", "guest house", "guesthouse", "inn"],
    "cafe": ["cafe", "café", "coffee shop", "coffee"],
    "hospital": ["hospital", "clinic", "nursing home", "aspatal", "haspatal"],
    "pharmacy": ["pharmacy", "chemist", "medical store", "medicine shop",
                 "dawakhana", "dawai ki dukan", "dawa ki dukan", "dawai", "dawa"],
    "atm": ["atm", "cash machine"],
    "bank": ["bank"],
    "school": ["school", "vidyalaya"],
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
    r"nearest|closest|any|some|an|a|the|"
    r"ek|koi|accha|achha|acha|badhiya|sasta|saste|sasti|sabse)\s+",
    re.I,
)

# ---- Hinglish (Roman-script Hindi) and "nearest" handling ---------------------------------
# Hindi puts the landmark first and the postposition after it: "Bennett University ke paas restaurant".
# We rewrite such queries into the English word order the rest of the parser already handles.
_HI_NEAR_ME = re.compile(
    r"\b(?:(?:mere|mera|meri|apne)\s+(?:paas|pass|aas\s?paas|aaspaas|nazdeek|nazdik|najdeek)"
    r"|(?:yahan|yaha|yahaan|yahin)\s+(?:ke\s+)?(?:paas|pass|aas\s?paas|aaspaas)(?:\s+mein)?)\b",
    re.I,
)
_HI_SUPERLATIVE = re.compile(r"\bsabse\s+(?:paas|pass|nazdeek|nazdik|najdeek)\b", re.I)
_EN_NEAREST_TO = re.compile(r"^\s*(?:nearest|closest)\s+(?P<item>.+?)\s+to\s+(?P<ref>.+)$", re.I)
_EN_SUPERLATIVE = re.compile(r"^\s*(?:nearest|closest)\b|\b(?:nearest|closest)\s*$", re.I)
_HI_REL = re.compile(
    r"\b(?:ke|ki|k)\s+(?:paas|pass|aas\s?paas|aaspaas|nazdeek|nazdik|najdeek|nazdeeki|qareeb|kareeb)\b", re.I
)
_HI_IN = re.compile(r"\bmein\b", re.I)
_HI_WITHIN = re.compile(
    r"^(?:(?P<ref>.+?)\s+se\s+)?(?P<n>\d+(?:\.\d+)?)\s*(?P<u>km|kms|kilometers?|kilometres?|meters?|metres?|m)\s+"
    r"(?:ke\s+)?(?:andar|bheetar|bhitar)\b\s*(?P<rest>.*)$",
    re.I,
)
# Request/politeness words that carry no search meaning.
_HI_STOP = re.compile(
    r"\b(?:chahiye|chahiyea|dikhao|dikhaiye|dikhana|batao|bataiye|dhundo|dhoondo|khojo|milega|milegi|"
    r"kahan|kaha|kidhar|please|pls|plz|bhai|yaar|hai|hain|mujhe|mujhko|mereko)\b",
    re.I,
)
_HI_INTENTS = [
    (re.compile(
        r"\b(?:khana\s+(?:kahan|kaha|kidhar)\s+(?:milega|milegi|milta\s+hai|mil\s+sakta\s+hai)"
        r"|(?:kahan|kaha|kidhar)\s+(?:khayein|khayen|khaye|kha\s+sakte\s+hain?))\b", re.I), "restaurants"),
    (re.compile(r"\b(?:rukne|ruknay)\s+ki\s+jagah\b|\b(?:kahan|kaha|kidhar)\s+(?:ruk\s+sakte|ruke|rukein)\b", re.I),
     "hotels"),
    (re.compile(r"\b(?:paise|paisa|cash)\s+nikal(?:ne|na|ni)(?:\s+hain?)?\b", re.I), "atm"),
    (re.compile(r"\b(?:dawai|dawa)\s+(?:kahan|kaha)\s+(?:milegi|milti)\b", re.I), "pharmacy"),
]

# Business names that contain a category word. Without protection "Cafe Coffee Day" would be
# split into category=cafe + name="Coffee Day".
_BRANDS = [
    "cafe coffee day", "coffee day", "state bank of india", "bank of baroda", "bank of india",
    "union bank of india", "central bank of india",
]
_BRAND_RE = re.compile(r"\b(?:" + "|".join(re.escape(b) for b in sorted(_BRANDS, key=len, reverse=True)) + r")\b", re.I)
_BRAND_TOKEN_RE = re.compile(r"qqbrand(\d+)qq")
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


def _split_item_landmark(text: str) -> tuple[str, str]:
    """Hindi word order: 'Raja Hotel Bennett University' -> ('Raja Hotel', 'Bennett University').

    The item is everything up to and including the first category word (or protected brand);
    whatever follows is the landmark.
    """
    text = text.strip()
    end = 0
    cat = _CAT_RE.search(text)
    brand = _BRAND_TOKEN_RE.search(text)
    if cat:
        end = cat.end()
    if brand and (not cat or brand.start() < cat.start()):
        end = brand.end()
    return text[:end].strip(), text[end:].strip()


def _hinglish_to_english(text: str) -> tuple[str, bool, bool]:
    """Rewrite Hinglish constructs into English word order.

    Returns (text, near_me, superlative): `near_me` means the query is relative to the user's
    own location ("mere paas"); `superlative` means 'nearest/closest/sabse paas'.
    """
    near_me = superlative = False

    for pattern, word in _HI_INTENTS:
        text = pattern.sub(word, text)

    m = _EN_NEAREST_TO.match(text)
    if m:  # "closest hotel to Bennett University" -> "hotel near Bennett University"
        text = f"{m['item']} near {m['ref']}"
    elif _EN_SUPERLATIVE.search(text):
        superlative = True
        text = _EN_SUPERLATIVE.sub(" ", text)
    if _HI_SUPERLATIVE.search(text):
        superlative = True
        text = _HI_SUPERLATIVE.sub(" ", text)
    if _HI_NEAR_ME.search(text):
        near_me = True
        text = _HI_NEAR_ME.sub(" ", text)

    text = _HI_STOP.sub(" ", text)
    text = " ".join(text.split())

    m = _HI_WITHIN.match(text)
    if m:  # "Bennett University se 2 km ke andar restaurant" -> "restaurant within 2 km of Bennett University"
        ref, rest = (m["ref"] or "").strip(), (m["rest"] or "").strip()
        if ref and not rest:
            rest, ref = _split_item_landmark(ref)
        text = f"{rest} within {m['n']} {m['u']} of {ref or 'me'}"
        return " ".join(text.split()), near_me, superlative

    for pattern, english in ((_HI_REL, "near"), (_HI_IN, "in")):
        m = pattern.search(text)
        if not m:
            continue
        left, right = text[: m.start()].strip(), text[m.end():].strip()
        if right:
            text = f"{right} {english} {left}"
        else:
            item, landmark = _split_item_landmark(left)
            text = f"{item} {english} {landmark}" if item else f"{english} {left}"
        break
    return " ".join(text.split()), near_me, superlative


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

    brands: list[str] = []

    def _protect(m: re.Match) -> str:  # hide brand names from category matching
        brands.append(m.group(0))
        return f"qqbrand{len(brands) - 1}qq"

    text = _BRAND_RE.sub(_protect, text)
    text, near_me, superlative = _hinglish_to_english(" ".join(text.split()))

    def _restore(s: str) -> str:
        return _BRAND_TOKEN_RE.sub(lambda m: brands[int(m.group(1))], s)

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

    out.name = _restore(name) or None
    out.reference = _restore(ref) or None

    # "mere paas", "nearest ...": relative to the user's own location unless a landmark was given.
    if (near_me or superlative) and not out.reference:
        if out.relation == "none":
            out.relation = "near"
        out.use_user_location = True

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
