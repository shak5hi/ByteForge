from ml.api import match_entities, parse_location_query, rank_results

BENNETT = (28.4503, 77.5845)


def _cands():
    return [
        {"id": "far", "name": "Raja Hotel", "lat": 28.55, "lon": 77.70, "category": "hotel"},
        {"id": "near", "name": "Raja Hotel", "lat": 28.4510, "lon": 77.5850, "category": "hotel"},
        {"id": "wrong_name", "name": "Sunrise Inn", "lat": 28.4505, "lon": 77.5846, "category": "hotel"},
    ]


def test_nearest_exact_name_wins():
    parsed = parse_location_query("Raja Hotel near Bennett University")
    out = rank_results(parsed, _cands(), *BENNETT)
    assert out[0]["id"] == "near"
    assert out[0]["score"] > out[-1]["score"]


def test_radius_filter():
    parsed = parse_location_query("hotels within 1 km of Bennett University")
    ids = {r["id"] for r in rank_results(parsed, _cands(), *BENNETT)}
    assert "far" not in ids


def test_match_uses_trained_model():
    a = {"name": "Raja Hotel Pvt Ltd", "address": "12 MG Road, Noida", "country": "India"}
    b = {"name": "RAJA HOTEL PRIVATE LIMITED", "address": "12 M.G. Rd Noida", "country": "India"}
    r = match_entities(a, b)
    assert r["source"] == "model"
    assert r["is_match"] is True


def test_match_rejects_different_businesses():
    a = {"name": "Raja Hotel", "address": "12 MG Road, Noida", "country": "India"}
    c = {"name": "Sunrise Pharmacy", "address": "5 Park Street, Kolkata", "country": "India"}
    assert match_entities(a, c)["is_match"] is False


def test_far_apart_coordinates_penalize_match():
    base = {"name": "Raja Hotel", "address": "12 MG Road", "country": "India"}
    near = match_entities({**base, "lat": 28.45, "lon": 77.58}, {**base, "lat": 28.4501, "lon": 77.5801})
    far = match_entities({**base, "lat": 28.45, "lon": 77.58}, {**base, "lat": 19.07, "lon": 72.87})
    assert far["score"] < near["score"]
