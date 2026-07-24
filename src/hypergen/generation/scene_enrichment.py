"""Opt-in Ollama contract for rewriting an author-visible Scene."""

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

SCENE_ENRICHMENT_PROMPT_VERSION = "scene-enrichment-v1"


class SceneEnrichmentRequest(DomainModel):
    """Author-controlled Scene plus read-only effective Style context."""

    scene: NonEmptyString
    effective_style: str = ""
    prompt_version: NonEmptyString = SCENE_ENRICHMENT_PROMPT_VERSION


class SceneEnrichmentModelOutput(DomainModel):
    """Structured response requested from Ollama."""

    scene: NonEmptyString


class SceneEnrichmentResult(DomainModel):
    """Reviewable Scene rewrite with debugging metadata."""

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
            "scene": request.scene,
            "effective_style": request.effective_style,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Rewrite the author's Scene into a richer text-to-image description for one illustrated card.

Return JSON matching the supplied schema.
- Preserve the author's subject, setting, mood, and factual intent.
- Add concrete visual composition, lighting, materials, atmosphere, and spatial details only when
  they support the original Scene.
- Treat effective_style as read-only visual context. Do not repeat it mechanically.
- Do not add interactions, navigation instructions, hotspots, captions, labels, signs, or
  interface elements unless the Scene explicitly requests them.
- Return only the rewritten Scene in the scene field.

Prompt contract: {request.prompt_version}
Author-controlled input:
{source}
"""


class OllamaSceneEnricher:
    """Rewrite a Scene through the shared structured Ollama runtime."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult:
        call = self._runtime.chat_structured(
            prompt=build_scene_enrichment_prompt(request),
            schema=SceneEnrichmentModelOutput.model_json_schema(),
        )
        try:
            output = SceneEnrichmentModelOutput.model_validate_json(call.content)
        except ValidationError as error:
            raise ModelResponseError(
                "Ollama returned an invalid Scene enrichment response for "
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
