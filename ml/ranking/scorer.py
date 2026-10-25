"""Transparent weighted ranking: text relevance + distance decay + category + match confidence.

Upgrade to XGBoost ranking (LambdaMART) only once you have relevance labels.
"""
from __future__ import annotations

import math
from typing import Optional

from rapidfuzz import fuzz

from ml._legacy import preprocess as _pp
from ml.entity_resolution.matcher import haversine_m
from ml.schemas import ParsedQuery, Place, RankedResult

normalize_name = _pp.normalize_name
normalize_text = _pp.normalize_text

DISTANCE_SCALE_KM = 2.0  # score halves roughly every ~1.4 km (exp decay)
WEIGHTS_WITH_NAME = {"text": 0.40, "distance": 0.30, "category": 0.20, "confidence": 0.10}
WEIGHTS_NO_NAME = {"text": 0.00, "distance": 0.55, "category": 0.35, "confidence": 0.10}


def rank(
    parsed: ParsedQuery,
    candidates: list[Place],
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
    limit: int = 20,
) -> list[RankedResult]:
    w = WEIGHTS_WITH_NAME if parsed.name else WEIGHTS_NO_NAME
    q_name = normalize_name(parsed.name)
    q_cat = normalize_text(parsed.category)
    have_ref = ref_lat is not None and ref_lon is not None
    results: list[RankedResult] = []

    for c in candidates:
        comp: dict[str, float] = {}
        comp["text"] = fuzz.token_set_ratio(q_name, normalize_name(c.name)) / 100 if q_name else 0.0

        dist_km = haversine_m(ref_lat, ref_lon, c.lat, c.lon) / 1000 if have_ref else None
        comp["distance"] = math.exp(-dist_km / DISTANCE_SCALE_KM) if dist_km is not None else 0.5
        if parsed.radius_km and dist_km is not None and dist_km > parsed.radius_km:
            continue  # hard filter for "within X km"

        c_cat = normalize_text(c.category)
        comp["category"] = (1.0 if q_cat and q_cat in c_cat else 0.0) if q_cat else 0.5
        comp["confidence"] = max(0.0, min(1.0, c.match_confidence))

        score = sum(w[k] * comp[k] for k in w)
        results.append(
            RankedResult(
                id=c.id,
                score=round(score, 4),
                distance_km=None if dist_km is None else round(dist_km, 3),
                components={k: round(v, 4) for k, v in comp.items()},
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
