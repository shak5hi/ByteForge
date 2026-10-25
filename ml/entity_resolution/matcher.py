"""Pairwise matcher: wraps the trained competition XGBoost model for single-pair scoring."""
from __future__ import annotations

import math
import warnings
from typing import Mapping, Optional

import joblib
import pandas as pd

from ml import _legacy
from ml.schemas import MatchResult

_BUNDLE: Optional[dict] = None
_THRESHOLD: Optional[float] = None
_TRIED = False
HEURISTIC_THRESHOLD = 0.85


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _load() -> Optional[dict]:
    global _BUNDLE, _THRESHOLD, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # xgboost pickle-version notice
                _BUNDLE = joblib.load(_legacy.MODEL_PATH)
            _THRESHOLD = float(_legacy.THRESHOLD_PATH.read_text().strip())
        except Exception:
            _BUNDLE = None
    return _BUNDLE


def _prep(rec: Mapping) -> dict:
    p = _legacy.preprocess
    return {
        "name_norm": p.normalize_name(rec.get("name") or rec.get("business_name")),
        "address_norm": p.normalize_address(rec.get("address") or rec.get("business_address")),
        "country_norm": p.normalize_text(rec.get("country")),
    }


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def match_pair(a: Mapping, b: Mapping) -> MatchResult:
    """a, b: dicts with `name` (or business_name), optional `address`, `country`, `lat`, `lon`.

    Uses the trained model when its artifact is present. `block_name_score` (a blocker
    score at training time) is approximated by name-token overlap for a single pair.
    Falls back to a heuristic when no model can be loaded.
    """
    pa, pb = _prep(a), _prep(b)
    feats = _legacy.features.pair_features(pa, pb, block_name_score=_token_overlap(pa["name_norm"], pb["name_norm"]))
    bundle = _load()

    if bundle is not None:
        cols = bundle["feature_columns"]
        X = pd.DataFrame([feats])[cols].fillna(0.0)
        score = float(bundle["model"].predict_proba(X)[0, 1])
        threshold, source = _THRESHOLD, "model"
    else:
        score = 0.7 * feats["name_token_set_ratio"] + 0.3 * feats["address_token_set_ratio"]
        threshold, source = HEURISTIC_THRESHOLD, "heuristic"

    # Optional geo evidence: two records with coordinates far apart are not the same place.
    try:
        d = haversine_m(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
        if d > 500:
            score *= 0.5
    except (KeyError, TypeError, ValueError):
        pass

    return MatchResult(score=round(score, 4), is_match=score >= threshold, threshold=threshold, source=source)
