"""Tests for reviewed Image Prompt delivery."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from hypergen.domain.models import ImageGenerationInputs, ImageReferenceSnapshot
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


@pytest.mark.parametrize("value", ("", "   "))
def test_image_prompt_must_be_non_empty(value: str) -> None:
    with pytest.raises(ValidationError):
        ImageGenerationInputs(
            description="A courtyard",
            image_prompt=value,
        )
