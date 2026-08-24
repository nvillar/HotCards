"""Deterministic author-controlled MFLUX prompt composition."""

from __future__ import annotations

from hypergen.domain.models import ImageGenerationInputs, ReferenceRole

IMAGE_PROMPT_VERSION = "image-prompt-v7"

_COMBINED_AUTHORITY = (
    "AUTHOR INTENT AUTHORITY\n"
    "DESCRIPTION is authoritative for actions, poses, object states, time, "
    "weather, camera, mood, composition, and quoted visible text. ENRICHED "
    "VISUAL DETAIL augments DESCRIPTION with appearance, style, setting, "
    "materials, lighting, and atmosphere. When they conflict, render the state "
    "specified by DESCRIPTION."
)
_DESCRIPTION_AUTHORITY = (
    "DESCRIPTION AUTHORITY\n"
    "DESCRIPTION supplies the desired scene, including actions, poses, object "
    "states, time, weather, camera, mood, composition, and quoted visible text."
)
_ENRICHED_AUTHORITY = (
    "ENRICHED DESCRIPTION AUTHORITY\n"
    "ENRICHED DESCRIPTION supplies the complete desired scene. Assigned "
    "references add visual detail only within their labeled roles."
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
    description = inputs.description.strip()
    enriched_description = (
        inputs.enriched_description.strip()
        if inputs.enriched_description is not None
        else ""
    )
    if not description and not enriched_description:
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
        if description and not enriched_description:
            return description
        if enriched_description and not description:
            return enriched_description
    description_sections = []
    if description:
        description_sections.append(f"DESCRIPTION\n{description}")
    if enriched_description:
        description_sections.append(
            (
                "ENRICHED VISUAL DETAIL"
                if description
                else "ENRICHED DESCRIPTION"
            )
            + f"\n{enriched_description}"
        )
    authority = (
        _COMBINED_AUTHORITY
        if description and enriched_description
        else (
            _DESCRIPTION_AUTHORITY
            if description
            else _ENRICHED_AUTHORITY
        )
    )
    return "\n\n".join(
        (authority, *reference_sections, *description_sections)
    )


__all__ = ["IMAGE_PROMPT_VERSION", "compose_image_prompt"]
