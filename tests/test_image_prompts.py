"""Tests for deterministic author-controlled image prompts."""

import pytest

from hypergen.domain.models import ImageGenerationInputs
from hypergen.generation.image_prompts import compose_image_prompt


def test_prompt_uses_the_authored_description_without_rewriting() -> None:
    description = (
        "A stone courtyard at dusk in detailed ink and watercolor."
    )

    assert compose_image_prompt(
        ImageGenerationInputs(description=description)
    ) == description


def test_empty_or_whitespace_description_is_rejected() -> None:
    for description in ("", "   "):
        with pytest.raises(ValueError, match="Description"):
            compose_image_prompt(
                ImageGenerationInputs(description=description)
            )
