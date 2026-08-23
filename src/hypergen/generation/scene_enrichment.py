"""Opt-in Ollama contract for enriching an author-visible Description."""

from __future__ import annotations

import json

from pydantic import ValidationError

from hypergen.domain.models import (
    DomainModel,
    NonEmptyString,
    NonNegativeFiniteFloat,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime
from hypergen.generation.structured_output import structured_json_content

SCENE_ENRICHMENT_PROMPT_VERSION = "scene-enrichment-v3"


class SceneEnrichmentRequest(DomainModel):
    """One non-empty author-written Description to expand."""

    scene: NonEmptyString
    prompt_version: NonEmptyString = SCENE_ENRICHMENT_PROMPT_VERSION


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
    """Build a bounded FLUX-oriented rewrite prompt from authored text."""
    source = json.dumps(
        {"authored_description": request.scene},
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Write a richer text-to-image Description for one illustrated card.

Return JSON matching the supplied schema.
- Preserve the authored subject, setting, mood, visible text, and factual intent.
- Add concrete form, scale, texture, materials, lighting quality and direction, shadows,
  spatial relationships, environment, atmosphere, and camera or composition details.
- Keep requested visible text in quotation marks.
- Make abstract qualities visually concrete without inventing new story facts.
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
