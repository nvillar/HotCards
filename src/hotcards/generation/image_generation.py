"""Deterministic prompt composition for direct image generation."""

from __future__ import annotations

from hotcards.domain.models import ImageGenerationInputs

DIRECT_GENERATION_PROMPT_VERSION = "direct-generation-v1"


def _with_sentence_boundary(value: str) -> str:
    sentence_endings = (".", "!", "?", "。", "！", "？")
    if value.endswith(sentence_endings):
        return value
    if len(value) >= 2 and value[-1] in {'"', "”", "’", "»"} and value[-2] in sentence_endings:
        return value
    return f"{value}."


def compose_generation_prompt(inputs: ImageGenerationInputs) -> str:
    """Append the exact selected Style to the authored Description."""
    description = inputs.description.strip()
    if not description:
        raise ValueError("enter a Description before generating")
    style = inputs.style
    if style is None or not style.prompt_text.strip():
        return description
    return "\n\n".join(
        (
            _with_sentence_boundary(description),
            style.prompt_text.strip(),
        )
    )


__all__ = [
    "DIRECT_GENERATION_PROMPT_VERSION",
    "compose_generation_prompt",
]
