"""Opt-in Ollama contract for enriching an author-visible Description."""

from __future__ import annotations

import json

from pydantic import ValidationError, model_validator

from hypergen.domain.models import (
    DomainModel,
    NonEmptyString,
    NonNegativeFiniteFloat,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime
from hypergen.generation.structured_output import structured_json_content

SCENE_ENRICHMENT_PROMPT_VERSION = "scene-enrichment-v2"


class SceneEnrichmentRequest(DomainModel):
    """Author Description plus optional visible image and Style context."""

    scene: str = ""
    effective_style: str = ""
    image_description: str | None = None
    prompt_version: NonEmptyString = SCENE_ENRICHMENT_PROMPT_VERSION

    @model_validator(mode="after")
    def require_description_or_image(self) -> SceneEnrichmentRequest:
        """Require at least one source of visual authoring context."""
        if not self.scene.strip() and not (
            self.image_description is not None and self.image_description.strip()
        ):
            raise ValueError("Description or image description must not be empty")
        return self


class SceneEnrichmentModelOutput(DomainModel):
    """Structured response requested from Ollama."""

    scene: NonEmptyString


class SceneEnrichmentResult(DomainModel):
    """Reviewable Description rewrite with debugging metadata."""

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


def build_scene_enrichment_prompt(request: SceneEnrichmentRequest) -> str:
    """Build a bounded rewrite prompt without interaction intent."""
    source = json.dumps(
        {
            "authored_description": request.scene,
            "effective_style": request.effective_style,
            "visible_image_description": request.image_description,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Write a richer text-to-image Description for one illustrated card.

Return JSON matching the supplied schema.
- Preserve the authored Description's subject, setting, mood, and factual intent when it is
  present. It is authoritative if it conflicts with the visible image description.
- Treat visible_image_description only as evidence of visible details in the active accepted
  background. Incorporate compatible composition, viewpoint, spatial relationships, lighting,
  color, materials, atmosphere, and rendering characteristics.
- If the authored Description is empty, construct the result from visible_image_description.
- Treat effective_style as read-only visual context. Do not repeat it mechanically.
- Do not add interactions, navigation instructions, hotspots, captions, labels, signs, or
  interface elements unless the authored Description explicitly requests them.
- Do not add hidden story facts, destinations, resolution or dimensions, step counts, model
  parameters, or other technical generation settings.
- Return only the rewritten Description in the scene field.

Prompt contract: {request.prompt_version}
Input:
{source}
"""


class OllamaSceneEnricher:
    """Enrich a Description through the shared structured Ollama runtime."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult:
        call = self._runtime.chat_structured(
            prompt=build_scene_enrichment_prompt(request),
            schema=SceneEnrichmentModelOutput.model_json_schema(),
        )
        try:
            output = SceneEnrichmentModelOutput.model_validate_json(
                structured_json_content(call.content)
            )
        except ValidationError as error:
            raise ModelResponseError(
                "Ollama returned an invalid Description enrichment response for "
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
        return SceneEnrichmentResult(
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
    "OllamaSceneEnricher",
    "SCENE_ENRICHMENT_PROMPT_VERSION",
    "SceneEnrichmentRequest",
    "SceneEnrichmentResult",
    "build_scene_enrichment_prompt",
]
