from ml.api import parse_location_query


def test_name_category_reference():
    r = parse_location_query("Raja Hotel near Bennett University")
    assert r["name"] == "Raja"
    assert r["category"] == "hotel"
    assert r["reference"] == "Bennett University"
    assert r["relation"] == "near"
    assert not r["ambiguous"]


def test_category_only():
    r = parse_location_query("restaurants near Bennett University")
    assert r["category"] == "restaurant"
    assert r["name"] is None
    assert r["reference"] == "Bennett University"


def test_within_radius():
    r = parse_location_query("cafes within 2 km of Pari Chowk")
    assert r["relation"] == "within"
    assert r["radius_km"] == 2.0
    assert r["reference"] == "Pari Chowk"
    assert r["category"] == "cafe"


def test_near_me_uses_user_location():
    r = parse_location_query("pharmacy near me")
    assert r["use_user_location"] is True
    assert r["reference"] is None


def test_bare_nearby():
    r = parse_location_query("hotels nearby")
    assert r["category"] == "hotel"
    assert r["use_user_location"] is True


def test_empty_and_garbage_do_not_raise():
    assert parse_location_query("")["ambiguous"] is True
    assert parse_location_query("   ")["ambiguous"] is True
    assert isinstance(parse_location_query("???"), dict)


def test_ambiguous_name_without_relation():
    r = parse_location_query("Raja Hotel")
    assert r["category"] == "hotel"
    assert r["ambiguous"] is True
