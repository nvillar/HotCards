"""Deterministic prompt composition for direct image generation."""

from __future__ import annotations

from collections.abc import Sequence

from hotcards.domain.models import (
    AcceptedEdit,
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


def compose_refine_prompt(
    description: str,
    style: StyleSnapshot | None,
    edit_lineage: Sequence[AcceptedEdit],
) -> str:
    """Compose Refine from current authored intent and accepted Edit lineage."""
    authored_description = description.strip()
    if not authored_description:
        raise ValueError("enter a Description before refining")
    parts = [authored_description]
    if style is not None and style.prompt_text.strip():
        parts[0] = _with_sentence_boundary(parts[0])
        parts.append(style.prompt_text.strip())
    if edit_lineage:
        instructions = "\n".join(
            f"{index}. {edit.instruction.strip()}"
            for index, edit in enumerate(edit_lineage, start=1)
        )
        parts.append(
            "The current Description is authoritative. The source image already "
            "includes these accepted edits; preserve them unless they conflict "
            f"with the current Description:\n{instructions}"
        )
    return "\n\n".join(parts)


def compose_edit_prompt(
    instruction: str,
    style: StyleSnapshot | None = None,
) -> str:
    """Build the exact deterministic minimal-change Flux Edit prompt."""
    normalized_instruction = instruction.strip()
    if not normalized_instruction:
        raise ValueError("enter an Edit Instruction before editing")
    parts = [
        f"Edit the provided image according to this instruction:\n\n{normalized_instruction}",
        "The source image is authoritative for everything not explicitly "
        "changed by the instruction. Make only the requested change and the "
        "minimum accompanying changes necessary for visual coherence. Preserve "
        "all other subjects, identities, objects, text, composition, framing, "
        "background, lighting, colors, and visual style. Do not add, remove, or "
        "reinterpret unrelated details.",
    ]
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
    "compose_refine_prompt",
]
