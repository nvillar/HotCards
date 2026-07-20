"""Prompt contract for deriving MFLUX render prompts with Ollama."""

from __future__ import annotations

import json

from pydantic import Field, ValidationError

from hypergen.domain.models import (
    DomainModel,
    ImageGenerationInputs,
    NonEmptyString,
    NonNegativeFiniteFloat,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime

IMAGE_PROMPT_VERSION = "image-prompt-v1"


class RenderPromptModelOutput(DomainModel):
    """Structured response requested from the configured Ollama model."""

    render_prompt: NonEmptyString
    interactive_subjects: tuple[NonEmptyString, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )


class DerivedRenderPrompt(DomainModel):
    """Inspectible render prompt and its derivation metadata."""

    prompt: NonEmptyString
    interactive_subjects: tuple[NonEmptyString, ...]
    raw_response: NonEmptyString
    model_identifier: NonEmptyString
    prompt_version: NonEmptyString = IMAGE_PROMPT_VERSION
    duration_seconds: NonNegativeFiniteFloat
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None


def build_image_prompt_request(inputs: ImageGenerationInputs) -> str:
    """Build the versioned instruction used to derive a render prompt."""
    source = json.dumps(inputs.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return f"""\
You derive a text-to-image render prompt for an illustrated interactive card.

Return JSON matching the supplied schema. The render prompt must:
- preserve the scene description;
- apply the stack art direction and optional card style;
- identify visually relevant interactive subjects from the interaction description;
- make those subjects clearly visible and spatially distinct;
- return at most 16 interactive subjects;
- avoid written labels, captions, signs, and interface-like elements unless the scene explicitly
  requests them.

Do not copy navigation instructions verbatim. Translate interaction semantics into visible scene
composition only.

Prompt contract: {IMAGE_PROMPT_VERSION}
Author-controlled inputs:
{source}
"""


class OllamaImagePromptDeriver:
    """Derive production MFLUX prompts through the shared Ollama runtime."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def derive(self, inputs: ImageGenerationInputs) -> DerivedRenderPrompt:
        """Derive and strictly parse one render prompt."""
        call = self._runtime.chat_structured(
            prompt=build_image_prompt_request(inputs),
            schema=RenderPromptModelOutput.model_json_schema(),
        )
        try:
            output = RenderPromptModelOutput.model_validate_json(call.content)
        except ValidationError as error:
            raise ModelResponseError(
                "Ollama returned an invalid image-prompt response for "
                f"{IMAGE_PROMPT_VERSION}: {error}"
            ) from error
        return DerivedRenderPrompt(
            prompt=output.render_prompt,
            interactive_subjects=output.interactive_subjects,
            raw_response=call.content,
            model_identifier=self._runtime.settings.model,
            duration_seconds=call.elapsed_seconds,
            total_duration_ns=call.total_duration_ns,
            load_duration_ns=call.load_duration_ns,
        )
