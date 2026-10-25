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
