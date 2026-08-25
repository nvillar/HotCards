"""Tests for reviewed Image Prompt delivery."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from hypergen.domain.models import (
    ImageGenerationInputs,
    ImageReferenceSnapshot,
    StyleSnapshot,
)
from hypergen.generation.image_prompts import compose_image_prompt


def test_image_prompt_is_delivered_without_hidden_rewriting() -> None:
    prompt = "A stone courtyard at dusk in detailed ink and watercolor."

    inputs = ImageGenerationInputs(
        description="A courtyard",
        image_prompt=prompt,
    )

    assert compose_image_prompt(inputs) == prompt


def test_reference_does_not_add_role_instructions() -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    prompt = "A red fox in crisp monochrome halftone linework."

    result = compose_image_prompt(
        ImageGenerationInputs(
            description="A red fox using the same visual style.",
            image_prompt=prompt,
            reference=snapshot,
        )
    )

    assert result == prompt
    assert "Use image" not in result
    assert "Reference" not in result


def test_selected_style_is_appended_after_the_reviewed_image_prompt() -> None:
    prompt = "A red fox beneath a tree"
    style = StyleSnapshot(
        style_id=uuid4(),
        name="Ink",
        prompt_text="Rendered with bold black ink contours.",
    )

    result = compose_image_prompt(
        ImageGenerationInputs(
            description="A fox",
            image_prompt=prompt,
            style=style,
        )
    )

    assert result == ("A red fox beneath a tree.\n\nRendered with bold black ink contours.")


def test_blank_selected_style_does_not_change_delivery() -> None:
    prompt = "A red fox beneath a tree."

    result = compose_image_prompt(
        ImageGenerationInputs(
            description="A fox",
            image_prompt=prompt,
            style=StyleSnapshot(
                style_id=uuid4(),
                name="Draft Style",
                prompt_text="",
            ),
        )
    )

    assert result == prompt


@pytest.mark.parametrize("value", ("", "   "))
def test_image_prompt_must_be_non_empty(value: str) -> None:
    with pytest.raises(ValidationError):
        ImageGenerationInputs(
            description="A courtyard",
            image_prompt=value,
        )
