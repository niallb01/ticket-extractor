"""Checks the scoring rules without calling the model."""

import pytest
from pydantic import ValidationError

from extract import TicketExtraction, score_extraction

EXPECTED = {
    "category": "authentication",
    "priority": "high",
    "summary": "Users cannot access accounts",
    "suggested_team": "backend",
    "needs_human_review": True,
}


def test_exact_fields_pass():
    actual = TicketExtraction(**EXPECTED)
    result = score_extraction(EXPECTED, actual)
    assert result["passed"]
    assert all(result["fields"].values())


def test_wrong_category_fails():
    actual = TicketExtraction(**{**EXPECTED, "category": "bug"})
    result = score_extraction(EXPECTED, actual)
    assert result["fields"]["category"] is False
    assert result["passed"] is False


def test_empty_summary_fails():
    actual = TicketExtraction(**{**EXPECTED, "summary": "   "})
    result = score_extraction(EXPECTED, actual)
    assert result["summary_ok"] is False
    assert result["passed"] is False


def test_schema_rejects_unknown_category():
    with pytest.raises(ValidationError):
        TicketExtraction(**{**EXPECTED, "category": "urgent"})
