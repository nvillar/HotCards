"""Deterministic prompt composition for direct image generation."""

from __future__ import annotations

from hotcards.domain.models import (
    GenerateInputs,
    StyleSnapshot,
)

DIRECT_GENERATION_PROMPT_VERSION = "direct-generation-v1"
EDIT_PROMPT_TOKEN_BUDGET = 512


def _with_sentence_boundary(value: str) -> str:
    sentence_endings = (".", "!", "?", "。", "！", "？")
    if value.endswith(sentence_endings):
        return value
    if len(value) >= 2 and value[-1] in {'"', "”", "’", "»"} and value[-2] in sentence_endings:
        return value
    return f"{value}."


def compose_generation_prompt(inputs: GenerateInputs) -> str:
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


def compose_edit_prompt(
    instruction: str,
    style: StyleSnapshot | None = None,
) -> str:
    """Build the exact authored Flux Edit prompt with optional Style continuity."""
    normalized_instruction = instruction.strip()
    if not normalized_instruction:
        raise ValueError("enter an Edit Instruction before editing")
    parts = [normalized_instruction]
    if style is not None and style.prompt_text.strip():
        parts.append(
            "Unless the Edit Instruction explicitly changes the visual treatment, "
            "keep the result consistent with this selected Style:\n\n"
            f"{style.prompt_text.strip()}"
        )
    return "\n\n".join(parts)


__all__ = [
    "DIRECT_GENERATION_PROMPT_VERSION",
    "EDIT_PROMPT_TOKEN_BUDGET",
    "compose_edit_prompt",
    "compose_generation_prompt",
]
