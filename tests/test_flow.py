"""Tests for ml.flow using fake places and fake database functions (no PostGIS needed)."""
from rapidfuzz import fuzz

from ml.entity_resolution.matcher import haversine_m
from ml.flow import run_search

BENNETT = {"id": 900, "name": "Bennett University", "lat": 28.4503, "lon": 77.5845, "category": "university", "is_landmark": True}
PARI = {"id": 901, "name": "Pari Chowk", "lat": 28.4744, "lon": 77.5040, "category": "junction", "is_landmark": True}
LANDMARKS = [BENNETT, PARI]

PLACES = [
    {"id": 1, "name": "Raja Hotel", "lat": 28.4510, "lon": 77.5850, "category": "hotel"},
    {"id": 2, "name": "Raja Hotel", "lat": 28.5500, "lon": 77.7000, "category": "hotel"},  # ~17 km away
    {"id": 3, "name": "Sunrise Inn", "lat": 28.4505, "lon": 77.5846, "category": "hotel"},
    {"id": 4, "name": "Spice Garden", "lat": 28.4590, "lon": 77.5845, "category": "restaurant"},
    {"id": 5, "name": "Tandoor Hut", "lat": 28.4520, "lon": 77.5850, "category": "restaurant"},
    {"id": 6, "name": "Brew Lab", "lat": 28.4600, "lon": 77.5845, "category": "cafe"},   # ~1.7 km
    {"id": 7, "name": "Bean There", "lat": 28.4515, "lon": 77.5845, "category": "cafe"},  # ~0.15 km
    {"id": 8, "name": "Apollo Pharmacy", "lat": 28.4744, "lon": 77.5040, "category": "pharmacy"},
    {"id": 9, "name": "Raja Palace", "lat": 28.4512, "lon": 77.5850, "category": None},   # category missing in data
]


def _dist_km(center, p):
    return haversine_m(center[0], center[1], p["lat"], p["lon"]) / 1000


def fake_resolve(text, near):
    best = max(LANDMARKS, key=lambda p: fuzz.WRatio(text.lower(), p["name"].lower()))
    return best if fuzz.WRatio(text.lower(), best["name"].lower()) >= 70 else None


def fake_nearby(lat, lon, radius_km, category, limit):
    hits = [p for p in PLACES if (not category or p["category"] == category) and _dist_km((lat, lon), p) <= radius_km]
    return sorted(hits, key=lambda p: _dist_km((lat, lon), p))[:limit]


def fake_by_name(text, center, radius_km, category, limit):
    hits = [
        p for p in PLACES
        if fuzz.token_set_ratio(text.lower(), p["name"].lower()) >= 70
        and (not category or p["category"] == category)
        and (center is None or radius_km is None or _dist_km(center, p) <= radius_km)
    ]
    return hits[:limit]


def search(q, **kw):
    return run_search(q, resolve_landmark=fake_resolve, find_nearby=fake_nearby, find_by_name=fake_by_name, **kw)


def test_named_place_near_landmark():
    r = search("Raja Hotel near Bennett University")
    assert r.status == "ok"
    assert r.reference["name"] == "Bennett University"
    ids = [x["id"] for x in r.results]
    assert ids[0] == 1            # exact name, closest
    assert 2 not in ids           # outside the default radius
    assert isinstance(r.results[0]["id"], int)  # backend ids come back unchanged
    assert r.results[0]["score"] > 0 and "score_components" in r.results[0]


def test_category_search_uses_nearby():
    r = search("restaurants near Bennett University")
    assert [x["id"] for x in r.results][:2] == [5, 4]  # closest restaurant first
    assert all(x["category"] == "restaurant" for x in r.results)


def test_within_radius_is_respected():
    r = search("cafes within 1 km of Bennett University")
    assert r.radius_km == 1.0
    assert [x["id"] for x in r.results] == [7]  # Brew Lab is ~1.7 km away


def test_near_me_uses_user_location():
    r = search("pharmacy near me", user_location=(28.4744, 77.5040))
    assert r.status == "ok" and r.center == (28.4744, 77.5040)
    assert [x["id"] for x in r.results] == [8]


def test_near_me_without_location_asks_for_it():
    r = search("pharmacy near me")
    assert r.status == "needs_user_location"
    assert r.results == []


def test_category_alone_needs_a_location():
    assert search("hotels").status == "needs_user_location"


def test_unknown_landmark():
    r = search("hotels near Zzzzzz Qqqqq Institute")
    assert r.status == "landmark_not_found"


def test_category_filter_is_relaxed_when_data_has_no_category():
    r = search("Raja Palace hotel")  # stored category is None
    assert [x["id"] for x in r.results] == [9]
    assert any("relaxed" in n for n in r.notes)


def test_overrides_win_over_parser():
    r = search("Raja Hotel", overrides={"reference": "Pari Chowk", "radius_km": 50})
    assert r.reference["name"] == "Pari Chowk"
    assert r.radius_km == 50


def test_empty_query_does_not_crash():
    r = search("")
    assert r.status == "ok" and r.results == []
