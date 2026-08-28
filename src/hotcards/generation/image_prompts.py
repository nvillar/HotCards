"""Validated Image Prompt delivery to MFLUX."""

from __future__ import annotations

from hotcards.domain.models import ImageGenerationInputs

IMAGE_PROMPT_VERSION = "image-prompt-v11"

_SCENE_PRESERVATION = (
    "Preserve the described subjects, actions, poses, object states, time, "
    "weather, viewpoint, framing, and composition."
)


def _with_sentence_boundary(value: str) -> str:
    sentence_endings = (".", "!", "?", "。", "！", "？")
    if value.endswith(sentence_endings):
        return value
    if len(value) >= 2 and value[-1] in {'"', "”", "’", "»"} and value[-2] in sentence_endings:
        return value
    return f"{value}."


def compose_reference_prompt(
    description: str,
    reference_instructions: tuple[str, ...],
) -> str:
    """Place scene content before natural-language reference instructions."""
    scene = description.strip()
    if not scene:
        raise ValueError("enter an Image Prompt before generating")
    instructions = tuple(
        instruction.strip() for instruction in reference_instructions if instruction.strip()
    )
    if not instructions:
        return scene
    return " ".join((_with_sentence_boundary(scene), *instructions, _SCENE_PRESERVATION))


def compose_image_prompt(inputs: ImageGenerationInputs) -> str:
    """Append the exact selected Style to the reviewed Image Prompt."""
    style = inputs.style
    if style is None or not style.prompt_text.strip():
        return inputs.image_prompt
    return "\n\n".join(
        (
            _with_sentence_boundary(inputs.image_prompt),
            style.prompt_text.strip(),
        )
    )


__all__ = [
    "IMAGE_PROMPT_VERSION",
    "compose_image_prompt",
    "compose_reference_prompt",
]
