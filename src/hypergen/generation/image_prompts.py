"""Deterministic author-controlled MFLUX prompt composition."""

from __future__ import annotations

from hypergen.domain.models import ImageGenerationInputs

IMAGE_PROMPT_VERSION = "image-prompt-v2"


def compose_image_prompt(inputs: ImageGenerationInputs) -> str:
    """Join Description and selected Style in a stable, inspectable order."""
    prompt = "\n\n".join(
        part
        for part in (inputs.description, inputs.style_prompt)
        if part.strip()
    )
    if not prompt.strip():
        raise ValueError(
            "enter a Description or Style before generating a background"
        )
    return prompt


__all__ = ["IMAGE_PROMPT_VERSION", "compose_image_prompt"]
