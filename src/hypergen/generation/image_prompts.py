"""Deterministic author-controlled MFLUX prompt composition."""

from __future__ import annotations

from hypergen.domain.models import ImageGenerationInputs, ReferenceRole

IMAGE_PROMPT_VERSION = "image-prompt-v8"

_SCENE_AUTHORITY = (
    "SCENE AUTHORITY\n"
    "SCENE supplies actions, poses, object states, time, weather, camera, mood, "
    "composition, quoted visible text, and every property not assigned to a "
    "labeled reference role."
)

_REFERENCE_INSTRUCTIONS = {
    ReferenceRole.SUBJECT: (
        "SUBJECT",
        "Derive the recognizable appearance and identity of the subject, "
        "object, person, or place from this image.",
    ),
    ReferenceRole.STYLE: (
        "STYLE",
        "Derive medium, linework, texture, palette, lighting, and rendering "
        "treatment from this image.",
    ),
    ReferenceRole.SETTING: (
        "SETTING",
        "Derive environment, architecture, materials, terrain, and location "
        "character from this image.",
    ),
}


def compose_image_prompt(inputs: ImageGenerationInputs) -> str:
    """Compile Description and assigned reference roles in deterministic order."""
    description = inputs.effective_description.strip()
    if not description:
        raise ValueError(
            "enter a Description or Enriched Description before generating"
        )
    reference_sections = []
    for index, (_reference, roles) in enumerate(
        inputs.grouped_references(),
        start=1,
    ):
        labels = " + ".join(_REFERENCE_INSTRUCTIONS[role][0] for role in roles)
        instructions = "\n".join(_REFERENCE_INSTRUCTIONS[role][1] for role in roles)
        reference_sections.append(f"REFERENCE IMAGE {index}\n{labels}\n{instructions}")
    if not reference_sections:
        return description
    return "\n\n".join(
        (_SCENE_AUTHORITY, *reference_sections, f"SCENE\n{description}")
    )


__all__ = ["IMAGE_PROMPT_VERSION", "compose_image_prompt"]
