"""Deterministic author-controlled MFLUX prompt composition."""

from __future__ import annotations

from hypergen.domain.models import ImageGenerationInputs

IMAGE_PROMPT_VERSION = "image-prompt-v4"

_REFERENCE_INSTRUCTIONS = (
    (
        "identity_reference",
        "IDENTITY",
        "Preserve the recognizable appearance and identity of the referenced "
        "subject, object, person, or place. Construct everything else from the "
        "Description.",
    ),
    (
        "visual_style_reference",
        "VISUAL STYLE",
        "Use only the referenced medium, linework, texture, palette, lighting, "
        "and rendering treatment. Ignore its depicted content and composition.",
    ),
    (
        "setting_reference",
        "SETTING",
        "Preserve the referenced environment, architecture, materials, and "
        "location vocabulary. Do not copy unrelated subjects from it.",
    ),
)


def compose_image_prompt(inputs: ImageGenerationInputs) -> str:
    """Compile Description and assigned reference roles in deterministic order."""
    if not inputs.description.strip():
        raise ValueError("enter a Description before generating a background")
    reference_sections = [
        f"REFERENCE IMAGE {index}\n{role}\n{instruction}"
        for index, (_field, role, instruction) in enumerate(
            (
                definition
                for definition in _REFERENCE_INSTRUCTIONS
                if getattr(inputs, definition[0]) is not None
            ),
            start=1,
        )
    ]
    return "\n\n".join(
        (*reference_sections, f"SCENE\n{inputs.description}")
        if reference_sections
        else (inputs.description,)
    )


__all__ = ["IMAGE_PROMPT_VERSION", "compose_image_prompt"]
