import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance


FEATURE_COLUMNS = [
    "block_name_score",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_jaro_winkler",
    "name_jaccard",
    "name_length_ratio",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_sort_ratio",
    "address_jaro_winkler",
    "address_jaccard",
    "numeric_address_jaccard",
    "country_match",
    "name_x_address",
    "max_name_similarity",
]


def _tokens(x):
    if not x:
        return set()
    return set(str(x).split())


def _jaccard(a, b):
    a = _tokens(a)
    b = _tokens(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _numeric_tokens(x):
    return set(re.findall(r"\d+", str(x)))


def pair_features(a, b, block_name_score=0.0):
    name_a = a["name_norm"]
    name_b = b["name_norm"]

    addr_a = a["address_norm"]
    addr_b = b["address_norm"]

    name_ratio = fuzz.ratio(name_a, name_b) / 100.0
    name_token = fuzz.token_set_ratio(name_a, name_b) / 100.0
    name_sort = fuzz.token_sort_ratio(name_a, name_b) / 100.0
    name_jw = distance.JaroWinkler.normalized_similarity(
        name_a,
        name_b,
    )

    addr_ratio = fuzz.ratio(addr_a, addr_b) / 100.0
    addr_token = fuzz.token_set_ratio(addr_a, addr_b) / 100.0
    addr_sort = fuzz.token_sort_ratio(addr_a, addr_b) / 100.0
    addr_jw = distance.JaroWinkler.normalized_similarity(
        addr_a,
        addr_b,
    )

    addr_jaccard = _jaccard(addr_a, addr_b)

    nums_a = _numeric_tokens(addr_a)
    nums_b = _numeric_tokens(addr_b)

    numeric_jaccard = (
        len(nums_a & nums_b) / len(nums_a | nums_b)
        if nums_a or nums_b
        else 0.0
    )

    country_match = float(
        a["country_norm"] == b["country_norm"]
    )

    length_ratio = (
        min(len(name_a), len(name_b))
        / max(len(name_a), len(name_b))
        if max(len(name_a), len(name_b))
        else 1.0
    )

    return {
        "block_name_score": float(block_name_score),
        "name_ratio": name_ratio,
        "name_token_set_ratio": name_token,
        "name_token_sort_ratio": name_sort,
        "name_jaro_winkler": name_jw,
        "name_jaccard": _jaccard(name_a, name_b),
        "name_length_ratio": length_ratio,
        "address_ratio": addr_ratio,
        "address_token_set_ratio": addr_token,
        "address_token_sort_ratio": addr_sort,
        "address_jaro_winkler": addr_jw,
        "address_jaccard": addr_jaccard,
        "numeric_address_jaccard": numeric_jaccard,
        "country_match": country_match,
        "name_x_address": name_token * addr_token,
        "max_name_similarity": max(
            name_ratio,
            name_token,
            name_sort,
            name_jw,
        ),
    }


def build_feature_frame(source1, source2, candidates):
    """
    Build numerical features for candidate pairs.

    candidates must contain:
        source1_entity_id
        candidate_entity_id

    This implementation uses merges to avoid the expensive
    iterrows() + .loc[] lookup used by the original version.
    """

    if candidates.empty:
        return (
            pd.DataFrame(columns=FEATURE_COLUMNS),
            [],
        )

    s1_cols = [
        "entity_id",
        "name_norm",
        "address_norm",
        "country_norm",
    ]

    s2_cols = [
        "entity_id",
        "name_norm",
        "address_norm",
        "country_norm",
    ]

    left = source1[s1_cols].copy()
    right = source2[s2_cols].copy()

    left = left.rename(
        columns={
            "entity_id": "source1_entity_id",
            "name_norm": "name_a",
            "address_norm": "address_a",
            "country_norm": "country_a",
        }
    )

    right = right.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "name_norm": "name_b",
            "address_norm": "address_b",
            "country_norm": "country_b",
        }
    )

    # Attach Source 1 fields.
    merged = candidates[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].merge(
        left,
        on="source1_entity_id",
        how="left",
        sort=False,
    )

    # Attach candidate-source fields.
    merged = merged.merge(
        right,
        on="candidate_entity_id",
        how="left",
        sort=False,
    )

    # RapidFuzz still needs to operate pair-by-pair.
    # We use zip() over arrays rather than iterrows().
    name_a = merged["name_a"].fillna("").tolist()
    name_b = merged["name_b"].fillna("").tolist()

    address_a = merged["address_a"].fillna("").tolist()
    address_b = merged["address_b"].fillna("").tolist()

    country_a = merged["country_a"].fillna("").tolist()
    country_b = merged["country_b"].fillna("").tolist()

    name_ratio = []
    name_token = []
    name_sort = []
    name_jw = []

    address_ratio = []
    address_token = []
    address_sort = []
    address_jw = []

    name_jaccard = []
    address_jaccard = []
    numeric_jaccard = []
    name_length_ratio = []

    for na, nb, aa, ab in zip(
        name_a,
        name_b,
        address_a,
        address_b,
    ):
        nr = fuzz.ratio(na, nb) / 100.0
        nt = fuzz.token_set_ratio(na, nb) / 100.0
        ns = fuzz.token_sort_ratio(na, nb) / 100.0
        nj = distance.JaroWinkler.normalized_similarity(
            na,
            nb,
        )

        ar = fuzz.ratio(aa, ab) / 100.0
        at = fuzz.token_set_ratio(aa, ab) / 100.0
        ass = fuzz.token_sort_ratio(aa, ab) / 100.0
        aj = distance.JaroWinkler.normalized_similarity(
            aa,
            ab,
        )

        name_ratio.append(nr)
        name_token.append(nt)
        name_sort.append(ns)
        name_jw.append(nj)

        address_ratio.append(ar)
        address_token.append(at)
        address_sort.append(ass)
        address_jw.append(aj)

        name_jaccard.append(_jaccard(na, nb))
        address_jaccard.append(_jaccard(aa, ab))

        nums_a = _numeric_tokens(aa)
        nums_b = _numeric_tokens(ab)

        numeric_jaccard.append(
            len(nums_a & nums_b) / len(nums_a | nums_b)
            if nums_a or nums_b
            else 0.0
        )

        name_length_ratio.append(
            min(len(na), len(nb))
            / max(len(na), len(nb))
            if max(len(na), len(nb))
            else 1.0
        )

    X = pd.DataFrame(
        {
            "block_name_score": (
                merged["block_name_score"]
                if "block_name_score" in merged
                else 0.0
            ),
            "name_ratio": name_ratio,
            "name_token_set_ratio": name_token,
            "name_token_sort_ratio": name_sort,
            "name_jaro_winkler": name_jw,
            "name_jaccard": name_jaccard,
            "name_length_ratio": name_length_ratio,
            "address_ratio": address_ratio,
            "address_token_set_ratio": address_token,
            "address_token_sort_ratio": address_sort,
            "address_jaro_winkler": address_jw,
            "address_jaccard": address_jaccard,
            "numeric_address_jaccard": numeric_jaccard,
            "country_match": [
                float(a == b)
                for a, b in zip(country_a, country_b)
            ],
            "name_x_address": [
                a * b
                for a, b in zip(name_token, address_token)
            ],
            "max_name_similarity": [
                max(a, b, c, d)
                for a, b, c, d in zip(
                    name_ratio,
                    name_token,
                    name_sort,
                    name_jw,
                )
            ],
        }
    )

    X = X[FEATURE_COLUMNS].fillna(0.0)

    keys = list(
        zip(
            merged["source1_entity_id"],
            merged["candidate_entity_id"],
        )
    )

    return X, keys