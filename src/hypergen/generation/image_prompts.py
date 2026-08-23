"""Deterministic author-controlled MFLUX prompt composition."""

from __future__ import annotations

from hypergen.domain.models import ImageGenerationInputs

IMAGE_PROMPT_VERSION = "image-prompt-v3"


def compose_image_prompt(inputs: ImageGenerationInputs) -> str:
    """Return the authored Description as the stable generation prompt."""
    prompt = inputs.description
    if not prompt.strip():
        raise ValueError("enter a Description before generating a background")
    return prompt


__all__ = ["IMAGE_PROMPT_VERSION", "compose_image_prompt"]
