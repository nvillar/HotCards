"""Tests for document-level validation."""

import pytest
from pydantic import ValidationError

from hypergen.domain.models import Card, Stack
from hypergen.domain.validation import normalize_card_name


def test_normalize_card_name_is_case_and_unicode_insensitive() -> None:
    assert normalize_card_name("  CAFÉ ") == normalize_card_name("cafe\u0301")


def test_stack_rejects_case_insensitive_duplicate_card_names() -> None:
    with pytest.raises(ValidationError, match="unique ignoring case"):
        Stack(
            name="Castle",
            cards=(Card(name="Courtyard"), Card(name=" courtyard ")),
        )


def test_distinct_card_names_are_accepted() -> None:
    stack = Stack(
        name="Castle",
        cards=(Card(name="Courtyard"), Card(name="Garden")),
    )

    assert len(stack.cards) == 2
