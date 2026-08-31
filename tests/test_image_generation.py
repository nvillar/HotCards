"""Tests for direct Description prompt composition."""

from uuid import uuid4

import pytest

from hotcards.domain.models import (
    GenerateInputs,
    ImageReferenceSnapshot,
    StyleSnapshot,
)
from hotcards.generation.image_generation import compose_generation_prompt


def test_description_is_delivered_without_hidden_rewriting() -> None:
    description = "A stone courtyard at dusk in detailed ink and watercolor."

    inputs = GenerateInputs(description=description)

    assert compose_generation_prompt(inputs) == description


def test_reference_does_not_add_role_instructions() -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    description = "A red fox in crisp monochrome halftone linework using image 1."

    result = compose_generation_prompt(
        GenerateInputs(
            description=description,
            references=(snapshot,),
        )
    )

    assert result == description
    assert "Reference" not in result


def test_selected_style_is_appended_after_the_description() -> None:
    description = "A red fox beneath a tree"
    style = StyleSnapshot(
        style_id=uuid4(),
        name="Ink",
        prompt_text="Rendered with bold black ink contours.",
    )

    result = compose_generation_prompt(
        GenerateInputs(
            description=description,
            style=style,
        )
    )

    assert result == ("A red fox beneath a tree.\n\nRendered with bold black ink contours.")


def test_blank_selected_style_does_not_change_delivery() -> None:
    description = "A red fox beneath a tree."

    result = compose_generation_prompt(
        GenerateInputs(
            description=description,
            style=StyleSnapshot(
                style_id=uuid4(),
                name="Draft Style",
                prompt_text="",
            ),
        )
    )

    assert result == description


@pytest.mark.parametrize("value", ("", "   "))
def test_description_must_be_non_empty_for_composition(value: str) -> None:
    with pytest.raises(ValueError, match="Description"):
        compose_generation_prompt(
            GenerateInputs(
                description=value,
            )
        )
