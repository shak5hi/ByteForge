"""Regression guard: the core tier must stay fully correct; ranking cases must all pass.

The hard tier documents known limits and is intentionally not asserted.
"""
import json

from ml.evaluation.run_eval import HERE, eval_parser, eval_ranking, load_queries


def test_core_parser_queries_all_correct():
    core = [r for r in load_queries() if r["tier"] == "core"]
    result = eval_parser(core)
    assert result["failures"] == [], result["failures"]


def test_ranking_cases_all_top1():
    cases = json.loads((HERE / "ranking_cases.json").read_text())
    result = eval_ranking(cases)
    assert result["failures"] == [], result["failures"]


def test_plural_y_categories():
    from ml.api import parse_location_query

    assert parse_location_query("pharmacies near me")["category"] == "pharmacy"


def test_hard_tier_stays_fixed():
    hard = [r for r in load_queries() if r["tier"] == "hard"]
    assert eval_parser(hard)["failures"] == []


def test_heldout_floor():
    """Held-out queries are never tuned on; guard against regressions below the first measured score."""
    held = [r for r in load_queries() if r["tier"] == "heldout"]
    assert eval_parser(held)["exact_acc"] >= 0.90


def test_typos_and_word_order():
    from ml.api import parse_location_query

    assert parse_location_query("restarant near Bennett University")["category"] == "restaurant"
    r = parse_location_query("Bennett University nearby restaurants")
    assert (r["category"], r["reference"]) == ("restaurant", "Bennett University")
    r = parse_location_query("hotels under 2000 near Bennett University")
    assert r["name"] is None and any("ignored" in n for n in r["notes"])


def test_abu_dhabi_is_not_a_dhaba():
    from ml.api import parse_location_query

    assert parse_location_query("hotels in Abu Dhabi")["category"] == "hotel"
    assert parse_location_query("Abu Dhabi near me")["category"] is None
