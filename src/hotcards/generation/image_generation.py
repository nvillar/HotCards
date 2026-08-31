"""Deterministic prompt composition for direct image generation."""

from __future__ import annotations

from collections.abc import Sequence

from hotcards.domain.models import (
    AcceptedEdit,
    EditPreserveOptions,
    GenerateInputs,
    StyleSnapshot,
)

DIRECT_GENERATION_PROMPT_VERSION = "direct-generation-v1"
EDIT_PROMPT_TOKEN_BUDGET = 512

_EDIT_PRESERVE_CLAUSES = (
    (
        "subject_identity",
        "Preserve the recognizable identity and appearance of existing subjects.",
    ),
    (
        "pose_and_expression",
        "Preserve poses, gestures, gaze, and facial expressions.",
    ),
    (
        "composition_and_framing",
        "Preserve viewpoint, perspective, placement, crop, and framing.",
    ),
    (
        "background",
        "Preserve the existing background and environment.",
    ),
    (
        "lighting_and_color",
        "Preserve lighting, shadows, contrast, and overall color treatment.",
    ),
    (
        "existing_text_and_logos",
        "Preserve visible text, lettering, symbols, and logos.",
    ),
)


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
    preserve: EditPreserveOptions,
) -> str:
    """Build the exact deterministic Flux Edit prompt in canonical order."""
    normalized_instruction = instruction.strip()
    if not normalized_instruction:
        raise ValueError("enter an Edit Instruction before editing")
    sections = [
        "Edit the provided image according to this instruction:"
        f"\n\n{normalized_instruction}"
    ]
    constraints = [
        clause
        for field_name, clause in _EDIT_PRESERVE_CLAUSES
        if getattr(preserve, field_name)
    ]
    if constraints:
        sections.append(
            "Preserve the following properties except where changing them is "
            "explicitly required by the edit instruction:\n"
            + "\n".join(f"- {constraint}" for constraint in constraints)
        )
    sections.append(
        "Make only the changes required by the edit instruction. Preserve all "
        "other details."
    )
    return "\n\n".join(sections)


__all__ = [
    "DIRECT_GENERATION_PROMPT_VERSION",
    "EDIT_PROMPT_TOKEN_BUDGET",
    "compose_edit_prompt",
    "compose_generation_prompt",
    "compose_refine_prompt",
]
