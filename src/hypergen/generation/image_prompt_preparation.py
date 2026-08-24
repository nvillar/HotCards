"""Optional-reference Ollama contract for preparing an Image Prompt."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from hypergen.domain.models import (
    DomainModel,
    NonEmptyString,
    NonNegativeFiniteFloat,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import (
    VISION_CAPABILITY,
    OllamaRuntime,
)
from hypergen.generation.structured_output import structured_json_content

IMAGE_PROMPT_PREPARATION_VERSION = "image-prompt-preparation-v1"


class ImagePromptPreparationRequest(DomainModel):
    """One authoritative Description and whether a Reference is attached."""

    description: NonEmptyString
    has_reference: bool = False
    prompt_version: NonEmptyString = IMAGE_PROMPT_PREPARATION_VERSION


class ImagePromptModelOutput(DomainModel):
    """Structured response with private deliberation and one visible result."""

    subject_traits: str | None = None
    setting_traits: str | None = None
    visual_treatment: str | None = None
    target_overrides: str | None = None
    image_prompt: NonEmptyString


class _ImagePromptRepairOutput(DomainModel):
    image_prompt: NonEmptyString


class ImagePromptPreparationResult(DomainModel):
    """Reviewable Image Prompt with debugging metadata."""

    image_prompt: NonEmptyString
    raw_response: NonEmptyString
    model_identifier: NonEmptyString
    prompt_version: NonEmptyString
    duration_seconds: NonNegativeFiniteFloat
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None


def build_image_prompt_preparation_prompt(
    request: ImagePromptPreparationRequest,
) -> str:
    """Build the bounded preparation prompt for text-only or image-aware use."""
    source = json.dumps(
        {
            "authored_description": request.description,
            "reference_attached": request.has_reference,
        },
        ensure_ascii=False,
        indent=2,
    )
    reference_contract = (
        """\
REFERENCE INTERPRETATION
- Inspect the attached Reference image directly. The authored Description is authoritative.
- Definite continuity language such as "the computer", "the person", or "the room" combined
  with now, still, changed, or another delta identifies that visible entity as continuing.
  "Same [entity]" also requests continuity.
- For a continuing entity, retain unmentioned stable identity and construction plus reusable
  visual treatment unless the Description explicitly overrides them.
- A request for the same visual style transfers only reusable medium, linework, texture,
  palette, shading, and rendering. It must not copy the Reference subject, setting, objects,
  screen contents, composition, or narrative.
- An explicit target medium, rendering style, palette, identity, setting, state, viewpoint,
  crop, framing, or composition overrides the corresponding Reference characteristic.
- Screen or sign contents, object state, pose, action, weather, time, lighting state,
  viewpoint, crop, framing, and surrounding composition are transient unless explicitly
  retained.
- A vague requested change may be resolved as a plausible concrete proposal. Do not ask a
  question or discuss ambiguity; the author will review and edit the Image Prompt.
"""
        if request.has_reference
        else """\
NO REFERENCE
- Prepare the Image Prompt only from the authored Description.
- Add concrete visual detail where it helps image generation, but do not invent a medium,
  rendering technology, camera specification, visible text, interaction, or story fact.
- A vague requested change may be resolved as a plausible concrete proposal. Do not ask a
  question or discuss ambiguity; the author will review and edit the Image Prompt.
"""
    )
    return f"""\
Prepare a production-ready FLUX.2 Image Prompt.

OUTPUT
- Return exactly one JSON object with exactly these keys: subject_traits, setting_traits,
  visual_treatment, target_overrides, image_prompt.
- The first four fields are private deliberation. Each must be one scalar string or null.
- image_prompt must be one non-empty natural-language paragraph with no headings or lists.
- Use 30 to 100 words by default. Expand only when needed to preserve explicit detail.

AUTHORITY
- Preserve every explicit authored subject, action, pose, object state, time, weather,
  viewpoint, crop, framing, composition, mood, color, and visible object.
- Preserve close-ups, limited fields of view, and statements that a subject fills the frame.
- Preserve exact authored visible text in quotation marks and on its intended object.
- When no quoted text is authored, introduce no visible words, lettering, signs, captions,
  labels, or interface text.
- Any color explicitly assigned in the Description overrides the Reference palette for that
  subject, object, text, or region. Retain compatible treatment and make that color visible.

{reference_contract}
PRIVATE DELIBERATION
- subject_traits: stable identity, silhouette, proportions, construction, materials,
  controls, and distinguishing components only when subject continuity is requested;
  otherwise null. Exclude pose, action, state, viewpoint, and surface contents.
- setting_traits: stable architecture and environment only when setting continuity is
  requested; otherwise null.
- visual_treatment: reusable medium, linework, texture, palette, shading, and rendering only
  when continuity or style transfer is requested; otherwise null.
- target_overrides: concrete requested changes only; otherwise null.

IMAGE PROMPT
- Synthesize one concrete, positive, standalone description of only the desired final image.
- Put the main subject, action, and critical authored changes first.
- Add concrete form, scale, materials, lighting, spatial relationships, atmosphere, and
  composition only where they support the authored result.
- Never retain old screen or sign contents when the Description changes that surface.
- Resolve every explicit override consistently. Do not blend an overridden medium or style
  back into the result.
- Never mention a reference image, source image, original image, previous version,
  comparison, transfer, replacement, omission, inference, instruction, ambiguity,
  or alternative.
- Never use process language such as "same style", "inspired by", "instead of", "replacing",
  "unchanged", or "has changed".
- Include no hotspots, navigation, destinations, dimensions, model settings, steps, seeds,
  or other machine parameters.

Prompt contract: {request.prompt_version}
Input:
{source}
"""


_WORD_PATTERN = re.compile(r"[^\W_]+(?:-[^\W_]+)*", flags=re.UNICODE)
_QUOTE_PAIRS = {'"': '"', "“": "”", "«": "»", "„": "“"}
_STATE_OPPOSITES = {
    "open": frozenset({"closed", "sealed", "shut"}),
    "opened": frozenset({"closed", "sealed", "shut"}),
    "closed": frozenset({"open", "opened"}),
    "sealed": frozenset({"open", "opened"}),
    "shut": frozenset({"open", "opened"}),
    "intact": frozenset({"broken", "collapsed", "damaged", "destroyed"}),
    "broken": frozenset({"intact"}),
    "collapsed": frozenset({"intact", "standing", "upright"}),
    "standing": frozenset({"collapsed", "fallen"}),
    "upright": frozenset({"collapsed", "fallen"}),
    "fallen": frozenset({"standing", "upright"}),
    "empty": frozenset({"crowded", "occupied"}),
    "unoccupied": frozenset({"crowded", "occupied"}),
    "vacant": frozenset({"crowded", "occupied"}),
    "occupied": frozenset({"empty", "unoccupied", "vacant"}),
    "lit": frozenset({"dark", "unlit"}),
    "illuminated": frozenset({"dark", "unlit"}),
    "dark": frozenset({"illuminated", "lit"}),
    "unlit": frozenset({"illuminated", "lit"}),
}
_STATE_PHRASE_BOUNDARIES = frozenset(
    {
        "a",
        "above",
        "after",
        "an",
        "and",
        "at",
        "before",
        "behind",
        "below",
        "beside",
        "beyond",
        "but",
        "contains",
        "featuring",
        "for",
        "from",
        "in",
        "into",
        "leading",
        "leads",
        "marked",
        "near",
        "next",
        "of",
        "on",
        "or",
        "past",
        "revealing",
        "reveals",
        "shows",
        "sits",
        "stands",
        "the",
        "through",
        "to",
        "under",
        "with",
    }
)
_PROCESS_LANGUAGE = re.compile(
    r"\b(?:reference image|source image|original image|previous (?:image|version)|"
    r"prior (?:image|version)|comparison|transfer|replacement|replacing|unchanged|"
    r"inference|ambiguity|alternative)\b"
    r"|\binstead of\b|\bsame style\b|\binspired by\b|\bhas changed\b",
    flags=re.IGNORECASE,
)


def _words(value: str) -> tuple[str, ...]:
    return tuple(_WORD_PATTERN.findall(value.casefold()))


def _quoted_text(value: str) -> tuple[set[str], bool]:
    matches: set[str] = set()
    closing: str | None = None
    is_balanced = True
    start = 0
    for index, character in enumerate(value):
        if closing is None:
            closing = _QUOTE_PAIRS.get(character)
            if closing is not None:
                start = index + 1
            elif character in {"”", "»"}:
                is_balanced = False
        elif character == closing:
            matches.add(value[start:index])
            closing = None
    return matches, is_balanced and closing is None


def _without_quoted_text(value: str) -> str:
    characters = list(value)
    closing: str | None = None
    start = 0
    for index, character in enumerate(value):
        if closing is None:
            closing = _QUOTE_PAIRS.get(character)
            if closing is not None:
                start = index
        elif character == closing:
            characters[start : index + 1] = " " * (index + 1 - start)
            closing = None
    return "".join(characters)


def _authored_state_assertions(value: str) -> set[tuple[str, str]]:
    words = _words(value)
    assertions: set[tuple[str, str]] = set()
    copulas = {"are", "is", "remain", "remains", "stand", "stands"}
    for index, word in enumerate(words):
        if word not in _STATE_OPPOSITES:
            continue
        copula_index = next(
            (
                candidate
                for candidate in range(index - 1, max(-1, index - 4), -1)
                if words[candidate] in copulas
            ),
            None,
        )
        if copula_index is not None and copula_index >= 1:
            assertions.add((words[copula_index - 1], word))
            continue
        phrase: list[str] = []
        for noun_index in range(index + 1, min(len(words), index + 5)):
            noun = words[noun_index]
            if (
                noun in _STATE_OPPOSITES
                or noun in copulas
                or noun in _STATE_PHRASE_BOUNDARIES
            ):
                break
            phrase.append(noun)
        if phrase:
            assertions.add((phrase[-1], word))
    return assertions


def _validate_image_prompt(authored: str, image_prompt: str) -> None:
    output_assertions = _authored_state_assertions(image_prompt)
    for noun, state in _authored_state_assertions(authored):
        for opposite in _STATE_OPPOSITES[state]:
            if (noun, opposite) in output_assertions:
                raise ValueError(
                    f"authored {noun!r} is {state!r}, but the Image Prompt makes "
                    f"it {opposite!r}"
                )
    authored_quotes, _ = _quoted_text(authored)
    output_quotes, output_quotes_are_balanced = _quoted_text(image_prompt)
    if not output_quotes_are_balanced:
        raise ValueError("Image Prompt contains unclosed visible text")
    invented_quotes = output_quotes - authored_quotes
    if invented_quotes:
        values = ", ".join(sorted(repr(value) for value in invented_quotes))
        raise ValueError(f"Image Prompt invented visible text: {values}")
    missing_quotes = authored_quotes - output_quotes
    if missing_quotes:
        values = ", ".join(sorted(repr(value) for value in missing_quotes))
        raise ValueError(f"Image Prompt omitted or changed visible text: {values}")
    process_language = _PROCESS_LANGUAGE.search(
        _without_quoted_text(image_prompt)
    )
    if process_language is not None:
        raise ValueError(
            "Image Prompt contains comparison or process language: "
            f"{process_language.group(0)!r}"
        )


def _build_repair_prompt(
    request: ImagePromptPreparationRequest,
    image_prompt: str,
    conflict: str,
) -> str:
    return f"""\
Repair an Image Prompt that violated its final-state contract.

Return exactly {{"image_prompt": "<corrected Image Prompt>"}} with no surrounding text.
Change only what is necessary to resolve the detected conflict. Preserve all explicit
authored subjects, actions, object states, colors, visible text, viewpoint, framing,
composition, and requested medium. Describe only the desired final image.

Authored Description:
{request.description}

Invalid Image Prompt:
{image_prompt}

Detected conflict:
{conflict}
"""


class OllamaImagePromptPreparer:
    """Prepare one Image Prompt through the shared structured Ollama runtime."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def prepare(
        self,
        request: ImagePromptPreparationRequest,
        *,
        reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult:
        if request.has_reference != (reference_image_path is not None):
            raise ValueError(
                "request reference state must match the attached image"
            )
        self._runtime.require_model(
            capabilities=frozenset({VISION_CAPABILITY})
        )
        call = self._runtime.chat_structured(
            prompt=build_image_prompt_preparation_prompt(request),
            schema=ImagePromptModelOutput.model_json_schema(),
            image_path=reference_image_path,
        )
        try:
            output = ImagePromptModelOutput.model_validate_json(
                structured_json_content(call.content)
            )
        except ValidationError as error:
            raise _model_response_error(request, call, error) from error
        try:
            _validate_image_prompt(request.description, output.image_prompt)
        except ValueError as error:
            call = self._runtime.chat_structured(
                prompt=_build_repair_prompt(
                    request,
                    output.image_prompt,
                    str(error),
                ),
                schema=_ImagePromptRepairOutput.model_json_schema(),
            )
            try:
                repaired = _ImagePromptRepairOutput.model_validate_json(
                    structured_json_content(call.content)
                )
                _validate_image_prompt(
                    request.description,
                    repaired.image_prompt,
                )
                output = output.model_copy(
                    update={"image_prompt": repaired.image_prompt}
                )
            except (ValidationError, ValueError) as repair_error:
                raise _model_response_error(
                    request,
                    call,
                    repair_error,
                ) from repair_error
        return ImagePromptPreparationResult(
            image_prompt=output.image_prompt,
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


def _model_response_error(
    request: ImagePromptPreparationRequest,
    call: object,
    error: Exception,
) -> ModelResponseError:
    return ModelResponseError(
        "Ollama returned an invalid Image Prompt "
        f"for {request.prompt_version}: {error}",
        raw_response=call.content,
        response_metadata={
            "elapsed_seconds": call.elapsed_seconds,
            "total_duration_ns": call.total_duration_ns,
            "load_duration_ns": call.load_duration_ns,
            "prompt_eval_count": call.prompt_eval_count,
            "eval_count": call.eval_count,
            "done_reason": call.done_reason,
        },
    )


__all__ = [
    "IMAGE_PROMPT_PREPARATION_VERSION",
    "ImagePromptModelOutput",
    "ImagePromptPreparationRequest",
    "ImagePromptPreparationResult",
    "OllamaImagePromptPreparer",
    "build_image_prompt_preparation_prompt",
]
