"""Deterministic author-controlled MFLUX prompt composition."""

from __future__ import annotations

from hypergen.domain.models import ImageGenerationInputs, ReferenceRole

IMAGE_PROMPT_VERSION = "image-prompt-v9"

_REFERENCE_ATTRIBUTES = {
    ReferenceRole.SUBJECT: (
        "the subject's recognizable identity and appearance, including its "
        "defining features, body, and clothing"
    ),
    ReferenceRole.STYLE: (
        "the medium, linework, texture, palette, lighting, and rendering treatment"
    ),
    ReferenceRole.SETTING: (
        "the environment, architecture, intrinsic materials, terrain, and "
        "spatial character"
    ),
}

_SCENE_PRESERVATION = (
    "Preserve the described subjects, actions, poses, object states, time, "
    "weather, viewpoint, framing, and composition."
)


def _join_attributes(attributes: tuple[str, ...]) -> str:
    if len(attributes) == 1:
        return attributes[0]
    if len(attributes) == 2:
        return " and ".join(attributes)
    return f"{', '.join(attributes[:-1])}, and {attributes[-1]}"


def _with_sentence_boundary(value: str) -> str:
    sentence_endings = (".", "!", "?", "。", "！", "？")
    if value.endswith(sentence_endings):
        return value
    if (
        len(value) >= 2
        and value[-1] in {'"', "”", "’", "»"}
        and value[-2] in sentence_endings
    ):
        return value
    return f"{value}."


def compose_reference_prompt(
    description: str,
    reference_instructions: tuple[str, ...],
) -> str:
    """Place scene content before natural-language reference instructions."""
    scene = description.strip()
    if not scene:
        raise ValueError(
            "enter a Description or Enriched Description before generating"
        )
    instructions = tuple(
        instruction.strip()
        for instruction in reference_instructions
        if instruction.strip()
    )
    if not instructions:
        return scene
    return " ".join(
        (_with_sentence_boundary(scene), *instructions, _SCENE_PRESERVATION)
    )


def compose_image_prompt(inputs: ImageGenerationInputs) -> str:
    """Compile Description and assigned reference roles in deterministic order."""
    description = inputs.effective_description.strip()
    if not description:
        raise ValueError(
            "enter a Description or Enriched Description before generating"
        )
    reference_instructions = []
    for index, (_reference, roles) in enumerate(
        inputs.grouped_references(),
        start=1,
    ):
        attributes = tuple(_REFERENCE_ATTRIBUTES[role] for role in roles)
        reference_instructions.append(
            f"Use image {index} for {_join_attributes(attributes)}."
        )
    return compose_reference_prompt(description, tuple(reference_instructions))


__all__ = [
    "IMAGE_PROMPT_VERSION",
    "compose_image_prompt",
    "compose_reference_prompt",
]
