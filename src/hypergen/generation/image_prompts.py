"""Deterministic author-controlled MFLUX prompt composition."""

from __future__ import annotations

from hypergen.domain.models import ImageGenerationInputs, ReferenceRole

IMAGE_PROMPT_VERSION = "image-prompt-v5"

_REFERENCE_INSTRUCTIONS = {
    ReferenceRole.IDENTITY: (
        "IDENTITY",
        "Preserve the recognizable appearance and identity of the referenced "
        "subject, object, person, or place. Construct everything else from the "
        "Description.",
    ),
    ReferenceRole.VISUAL_STYLE: (
        "VISUAL STYLE",
        "Use only the referenced medium, linework, texture, palette, lighting, "
        "and rendering treatment. Ignore its depicted content and composition.",
    ),
    ReferenceRole.SETTING: (
        "SETTING",
        "Preserve the referenced environment, architecture, materials, and "
        "location vocabulary. Do not copy unrelated subjects from it.",
    ),
}


def compose_image_prompt(inputs: ImageGenerationInputs) -> str:
    """Compile Description and assigned reference roles in deterministic order."""
    if not inputs.description.strip():
        raise ValueError("enter a Description before generating a background")
    reference_sections = []
    for index, (_reference, roles) in enumerate(
        inputs.grouped_references(),
        start=1,
    ):
        labels = " + ".join(_REFERENCE_INSTRUCTIONS[role][0] for role in roles)
        instructions = "\n".join(
            _REFERENCE_INSTRUCTIONS[role][1] for role in roles
        )
        reference_sections.append(
            f"REFERENCE IMAGE {index}\n{labels}\n{instructions}"
        )
    return "\n\n".join(
        (*reference_sections, f"SCENE\n{inputs.description}")
        if reference_sections
        else (inputs.description,)
    )


__all__ = ["IMAGE_PROMPT_VERSION", "compose_image_prompt"]
