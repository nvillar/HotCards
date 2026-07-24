"""Vision contract for reconstructing a generation-ready Scene from an image."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from hypergen.domain.models import (
    DomainModel,
    NonEmptyString,
    NonNegativeFiniteFloat,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime

IMAGE_DESCRIPTION_PROMPT_VERSION = "image-description-v1"


class ImageDescriptionRequest(DomainModel):
    """One active revision image to describe without authoring metadata."""

    image_path: Path
    prompt_version: NonEmptyString = IMAGE_DESCRIPTION_PROMPT_VERSION


class ImageDescriptionModelOutput(DomainModel):
    """Structured response requested from vision-capable Ollama."""

    scene: NonEmptyString


class ImageDescriptionResult(DomainModel):
    """Generation-ready Scene description with debugging metadata."""

    scene: NonEmptyString
    raw_response: NonEmptyString
    model_identifier: NonEmptyString
    prompt_version: NonEmptyString
    duration_seconds: NonNegativeFiniteFloat
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None


def build_image_description_prompt(request: ImageDescriptionRequest) -> str:
    """Build a bounded image-to-Scene prompt."""
    return f"""\
Describe the supplied card image as a detailed text-to-image Scene prompt that could generate a
close visual version of it.

Return JSON matching the supplied schema.
- Describe only visible visual content: subjects, setting, composition, camera/viewpoint, spatial
  relationships, lighting, color, materials, atmosphere, and rendering style.
- Be concrete and concise enough to remain directly editable by an author.
- Do not infer interactions, navigation, hotspots, hidden story facts, or destinations.
- Mention visible written text only when it is visually important.
- Do not mention that you are analyzing an image.
- Return only the description in the scene field.

Prompt contract: {request.prompt_version}
"""


class OllamaImageDescriber:
    """Describe one image through the shared structured Ollama runtime."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def describe(self, request: ImageDescriptionRequest) -> ImageDescriptionResult:
        if not request.image_path.is_file():
            raise ModelResponseError(
                f"image description input does not exist: {request.image_path}"
            )
        call = self._runtime.chat_structured(
            prompt=build_image_description_prompt(request),
            schema=ImageDescriptionModelOutput.model_json_schema(),
            image_path=request.image_path,
        )
        try:
            output = ImageDescriptionModelOutput.model_validate_json(call.content)
        except ValidationError as error:
            raise ModelResponseError(
                "Ollama returned an invalid image description response for "
                f"{request.prompt_version}: {error}",
                raw_response=call.content,
                response_metadata={
                    "elapsed_seconds": call.elapsed_seconds,
                    "total_duration_ns": call.total_duration_ns,
                    "load_duration_ns": call.load_duration_ns,
                    "prompt_eval_count": call.prompt_eval_count,
                    "eval_count": call.eval_count,
                    "done_reason": call.done_reason,
                },
            ) from error
        return ImageDescriptionResult(
            scene=output.scene,
            raw_response=call.content,
            model_identifier=self._runtime.settings.model,
            prompt_version=request.prompt_version,
            duration_seconds=call.elapsed_seconds,
            total_duration_ns=call.total_duration_ns,
            load_duration_ns=call.load_duration_ns,
            prompt_eval_count=call.prompt_eval_count,
            eval_count=call.eval_count,
            done_reason=call.done_reason,
        )


__all__ = [
    "IMAGE_DESCRIPTION_PROMPT_VERSION",
    "ImageDescriptionRequest",
    "ImageDescriptionResult",
    "OllamaImageDescriber",
    "build_image_description_prompt",
]
