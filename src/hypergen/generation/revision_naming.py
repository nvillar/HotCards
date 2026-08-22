"""Structured vision contract for naming a generated image revision."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError, field_validator

from hypergen.domain.models import (
    MAX_REVISION_NAME_LENGTH,
    DomainModel,
    NonEmptyString,
    NonNegativeFiniteFloat,
    RevisionName,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime

REVISION_NAMING_PROMPT_VERSION = "revision-naming-v1"


class RevisionNamingRequest(DomainModel):
    """Generated image and author context used to propose a short title."""

    image_path: Path
    render_prompt: NonEmptyString
    existing_names: tuple[RevisionName, ...] = ()
    prompt_version: NonEmptyString = REVISION_NAMING_PROMPT_VERSION


class RevisionNamingModelOutput(DomainModel):
    """Structured title returned by Ollama."""

    name: RevisionName

    @field_validator("name")
    @classmethod
    def reject_generic_name(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("revision name must be a single line")
        if value.casefold() in {
            "background",
            "generated",
            "image",
            "revision",
            "untitled",
        }:
            raise ValueError("revision name must describe the generated image")
        return value


class RevisionNamingResult(DomainModel):
    """Validated revision title with debugging metadata."""

    name: RevisionName
    raw_response: NonEmptyString
    model_identifier: NonEmptyString
    prompt_version: NonEmptyString
    duration_seconds: NonNegativeFiniteFloat
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None


def build_revision_naming_prompt(request: RevisionNamingRequest) -> str:
    """Build a bounded image-title prompt with existing names as context."""
    context = json.dumps(
        {
            "render_prompt": request.render_prompt,
            "existing_revision_names": request.existing_names,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Give the supplied generated card background a short, descriptive revision name.

Return JSON matching the supplied schema.
- Base the name primarily on visible subjects, setting, mood, or composition in the image.
- Use the render prompt only as supporting context.
- Use two to five words and no more than 48 characters.
- Do not use generic names such as Generated, Background, Image, Revision, or Untitled.
- Do not add numbering, quotation marks, or terminal punctuation.
- Prefer a name unlike the existing revision names, but return only the name field.

Prompt contract: {request.prompt_version}
Context:
{context}
"""


def unique_revision_name(proposed_name: str, existing_names: Iterable[str]) -> str:
    """Append the first available numeric suffix for a case-insensitive collision."""
    name = proposed_name.strip()
    used = {existing.strip().casefold() for existing in existing_names}
    if name.casefold() not in used:
        return name
    index = 2
    while True:
        suffix = f" {index}"
        candidate = f"{name[: MAX_REVISION_NAME_LENGTH - len(suffix)].rstrip()}{suffix}"
        if candidate.casefold() not in used:
            return candidate
        index += 1


class OllamaRevisionNamer:
    """Name one generated image through the shared structured Ollama runtime."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def name(self, request: RevisionNamingRequest) -> RevisionNamingResult:
        if not request.image_path.is_file():
            raise ModelResponseError(f"revision naming input does not exist: {request.image_path}")
        call = self._runtime.chat_structured(
            prompt=build_revision_naming_prompt(request),
            schema=RevisionNamingModelOutput.model_json_schema(),
            image_path=request.image_path,
        )
        try:
            content = call.content.strip()
            output = (
                RevisionNamingModelOutput.model_validate_json(content)
                if content.startswith("{")
                else RevisionNamingModelOutput.model_validate({"name": content})
            )
        except ValidationError as error:
            raise ModelResponseError(
                "Ollama returned an invalid revision naming response for "
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
        return RevisionNamingResult(
            name=output.name,
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
    "OllamaRevisionNamer",
    "REVISION_NAMING_PROMPT_VERSION",
    "RevisionNamingRequest",
    "RevisionNamingResult",
    "build_revision_naming_prompt",
    "unique_revision_name",
]
