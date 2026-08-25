"""Optional-reference Ollama contract for preparing an Image Prompt."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from hypergen.domain.models import (
    DomainModel,
    NonEmptyString,
    NonNegativeFiniteFloat,
)
from hypergen.generation.errors import GenerationError, ModelResponseError
from hypergen.generation.ollama_client import (
    VISION_CAPABILITY,
    OllamaRuntime,
)
from hypergen.generation.structured_output import structured_json_content

IMAGE_PROMPT_PREPARATION_VERSION = "image-prompt-preparation-v6"


class ImagePromptPreparationRequest(DomainModel):
    """One authoritative Description and whether a Reference is attached."""

    description: NonEmptyString
    has_reference: bool = False
    reference_description: str | None = None
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


class ImagePromptPreparationAttempt(DomainModel):
    """One initial or repair model call retained for evaluation and diagnostics."""

    phase: Literal["initial", "repair"]
    raw_response: NonEmptyString
    duration_seconds: NonNegativeFiniteFloat
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None


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
    repair_applied: bool = False
    attempts: tuple[ImagePromptPreparationAttempt, ...] = ()


def build_image_prompt_preparation_prompt(
    request: ImagePromptPreparationRequest,
) -> str:
    """Build the bounded preparation prompt for text-only or image-aware use."""
    source = json.dumps(
        {
            "authored_description": request.description,
            "reference_attached": request.has_reference,
            "reference_generation_description": (
                request.reference_description
                if request.has_reference
                else None
            ),
        },
        ensure_ascii=False,
        indent=2,
    )
    reference_contract = (
        """\
REFERENCE INTERPRETATION
- Inspect the attached Reference image directly. The authored Description is authoritative.
- reference_generation_description is the effective authored prompt captured when this exact
  Reference background was generated, including any reviewed Image Prompt or legacy enriched
  text that generation actually used. Treat its explicit identity and visual-style language as
  the primary semantic interpretation of the image whenever it applies to the requested
  continuity or transfer.
- Preserve applicable explicit treatment terms from reference_generation_description. Do not
  relabel "early Mac and HyperCard", pixel art, watercolor, engraving, collage, or another
  authored treatment as a merely similar-looking medium such as newspaper print or comic art.
- Use the pixels to confirm visible appearance, supply concrete visible details, and resolve
  omissions. Do not let a new visual guess override compatible authored Reference semantics.
- reference_generation_description is context, not a second target prompt. Do not copy its
  transient action, object state, screen or sign contents, viewpoint, framing, composition,
  weather, time, or setting unless the target Description requests that continuity.
- The Reference was selected because the author intends some visible continuity or transfer.
- A definite phrase identifying a visible entity, such as "the computer", "the monitor",
  "the person", or "the room", means that entity continues from the Reference even without
  words such as same, now, still, or changed. Pronouns referring to visible entities and
  "same [entity]" also request continuity.
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
- Treat the visible rendered appearance as truth. Never infer a conventional real-world
  color, material, age, era, or technology that is not visible or authored. In particular,
  do not turn a monochrome Reference into a beige, cream, tan, or otherwise colored object.
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
  viewpoint, crop, framing, composition, mood, color, visible object, medium, palette,
  linework, texture, shading, rendering technique, and visual style.
- Preserve the meaning of explicit visual properties without requiring the same wording.
  Do not omit, weaken, or reinterpret them as a merely similar treatment.
- Preserve close-ups, limited fields of view, and statements that a subject fills the frame.
- Preserve exact affirmatively authored visible text in quotation marks and on its intended
  object. Quoted words in an explicit exclusion such as `no "EXIT" text` are forbidden,
  not requested; preserve the text-free result without rendering those words.
- When no affirmative quoted text is authored, introduce no visible words, lettering, signs,
  captions, labels, or interface text.
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
  when continuity or style transfer is requested; otherwise null. Capture every dominant
  visible treatment trait, including color mode, edge or line character, dither or halftone
  pattern, tonal strategy, apparent medium or rendering technology, and detail level. Do not
  reduce a distinctive treatment to a generic era or mood label.
- target_overrides: concrete requested changes plus every explicit target-authored medium,
  palette, linework, texture, shading, rendering technique, and visual style; otherwise null.

IMAGE PROMPT
- Synthesize one concrete, positive, standalone description of only the desired final image.
- Put the main subject, action, and critical authored changes first.
- Incorporate every applicable non-null subject_traits, setting_traits, visual_treatment, and
  target_overrides detail. Private deliberation is a checklist for the final Image Prompt,
  not optional notes.
- Describe continuing entities through their observed distinguishing construction and
  treatment rather than generic stereotypes such as "old", "vintage", or "beige".
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
_ACHROMATIC_LANGUAGE = re.compile(
    r"\b(?:black(?:-|\s+)and(?:-|\s+)white|grayscale|greyscale)\b",
    flags=re.IGNORECASE,
)
_CHROMATIC_COLOR = re.compile(
    r"\b(?:beige|brown|tan|cream|red|orange|yellow|green|blue|purple|violet|"
    r"pink|cyan|magenta|teal|turquoise)\b",
    flags=re.IGNORECASE,
)
_QUOTE_CLAUSE_BOUNDARY = re.compile(
    r"[.!?;,]|\b(?:but|except|however)\b",
    flags=re.IGNORECASE,
)
_FORBIDDEN_QUOTE_PREFIX = re.compile(
    r"(?:"
    r"\b(?:no|without)\s+"
    r"(?:(?:using|showing|displaying|including|rendering)\s+)?"
    r"(?:(?:the|any)\s+)?"
    r"(?:(?:(?:visible|legible)\s+)?"
    r"(?:text|words?|lettering|labels?|captions?)\s+)?"
    r"|"
    r"\bno\s+(?:(?:visible|legible)\s+)?"
    r"(?:text|words?|lettering|labels?|captions?)\s+"
    r"(?:reading|saying|spelling)\s*"
    r"|"
    r"\b(?:do not|don't|does not|doesn't|did not|didn't|must not|"
    r"should not|never)\s+"
    r"(?:show|display|include|contain|render|write|read|feature)"
    r"(?:\s+(?:the\s+)?(?:text|words?|lettering|labels?|captions?))?\s*"
    r"|"
    r"\b(?:(?:must|should|does|do|did)\s+)?"
    r"(?:omit|omits|omitted|exclude|excludes|excluded|"
    r"avoid|avoids|avoided)\s+"
    r"(?:(?:using|showing|displaying|including|rendering)\s+)?"
    r"(?:the\s+)?(?:(?:text|word|words|lettering|label|labels|"
    r"caption|captions)\s+)?"
    r")$",
    flags=re.IGNORECASE,
)
_FORBIDDEN_QUOTE_SUFFIX = re.compile(
    r"^\s*(?:(?:text|words?|lettering|labels?|captions?)\s+)?(?:"
    r"(?:(?:is|are|must be|should be)\s+)?(?:not|never)\s+"
    r"(?:shown|displayed|included|rendered|written|visible)"
    r"|"
    r"(?:(?:is|are|was|were|must be|should be)\s+)"
    r"(?:omitted|excluded|avoided)"
    r")\b",
    flags=re.IGNORECASE,
)
_QUOTED_LIST_CONNECTOR = re.compile(
    r"\s*(?:,\s*(?:(?:and|or)\s*)?|(?:and|or)\s+)\s*",
    flags=re.IGNORECASE,
)


def _words(value: str) -> tuple[str, ...]:
    return tuple(_WORD_PATTERN.findall(value.casefold()))


def _quoted_text_spans(
    value: str,
) -> tuple[tuple[tuple[str, int, int], ...], bool]:
    matches: list[tuple[str, int, int]] = []
    closing: str | None = None
    is_balanced = True
    start = 0
    opening = 0
    for index, character in enumerate(value):
        if closing is None:
            closing = _QUOTE_PAIRS.get(character)
            if closing is not None:
                opening = index
                start = index + 1
            elif character in {"”", "»"}:
                is_balanced = False
        elif character == closing:
            matches.append((value[start:index], opening, index + 1))
            closing = None
    return tuple(matches), is_balanced and closing is None


def _quoted_text(value: str) -> tuple[set[str], bool]:
    spans, is_balanced = _quoted_text_spans(value)
    return {text for text, _start, _end in spans}, is_balanced


def _authored_quoted_text(value: str) -> tuple[set[str], set[str]]:
    spans, _is_balanced = _quoted_text_spans(value)
    forbidden = [False] * len(spans)
    for index, (_text, start, end) in enumerate(spans):
        prefix_start = max(
            (
                boundary.end()
                for boundary in _QUOTE_CLAUSE_BOUNDARY.finditer(
                    value,
                    0,
                    start,
                )
            ),
            default=0,
        )
        suffix_boundary = _QUOTE_CLAUSE_BOUNDARY.search(value, end)
        suffix_end = (
            suffix_boundary.start()
            if suffix_boundary is not None
            else len(value)
        )
        forbidden[index] = bool(
            _FORBIDDEN_QUOTE_PREFIX.search(value[prefix_start:start])
            or _FORBIDDEN_QUOTE_SUFFIX.search(value[end:suffix_end])
        )
    changed = True
    while changed:
        changed = False
        for index in range(1, len(spans)):
            connector = value[spans[index - 1][2] : spans[index][1]]
            if (
                _QUOTED_LIST_CONNECTOR.fullmatch(connector)
                and forbidden[index - 1] != forbidden[index]
            ):
                forbidden[index - 1] = True
                forbidden[index] = True
                changed = True
    required = {
        text
        for (text, _start, _end), is_forbidden in zip(
            spans,
            forbidden,
            strict=True,
        )
        if not is_forbidden
    }
    prohibited = {
        text
        for (text, _start, _end), is_forbidden in zip(
            spans,
            forbidden,
            strict=True,
        )
        if is_forbidden
    }
    return required, prohibited


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


def _validate_image_prompt(
    authored: str,
    image_prompt: str,
    *,
    visual_treatment: str | None = None,
) -> None:
    output_assertions = _authored_state_assertions(image_prompt)
    for noun, state in _authored_state_assertions(authored):
        for opposite in _STATE_OPPOSITES[state]:
            if (noun, opposite) in output_assertions:
                raise ValueError(
                    f"authored {noun!r} is {state!r}, but the Image Prompt makes "
                    f"it {opposite!r}"
                )
    required_quotes, forbidden_quotes = _authored_quoted_text(authored)
    output_quotes, output_quotes_are_balanced = _quoted_text(image_prompt)
    if not output_quotes_are_balanced:
        raise ValueError("Image Prompt contains unclosed visible text")
    included_forbidden_quotes = output_quotes & (
        forbidden_quotes - required_quotes
    )
    if included_forbidden_quotes:
        values = ", ".join(
            sorted(repr(value) for value in included_forbidden_quotes)
        )
        raise ValueError(f"Image Prompt included forbidden visible text: {values}")
    invented_quotes = output_quotes - required_quotes
    if invented_quotes:
        values = ", ".join(sorted(repr(value) for value in invented_quotes))
        raise ValueError(f"Image Prompt invented visible text: {values}")
    missing_quotes = required_quotes - output_quotes
    if missing_quotes:
        values = ", ".join(sorted(repr(value) for value in missing_quotes))
        raise ValueError(f"Image Prompt omitted or changed visible text: {values}")
    if (
        visual_treatment is not None
        and _ACHROMATIC_LANGUAGE.search(visual_treatment)
    ):
        authored_colors = {
            match.group(0).casefold()
            for match in _CHROMATIC_COLOR.finditer(authored)
        }
        invented_colors = {
            match.group(0).casefold()
            for match in _CHROMATIC_COLOR.finditer(image_prompt)
        } - authored_colors
        if invented_colors:
            values = ", ".join(sorted(invented_colors))
            raise ValueError(
                "Image Prompt invented chromatic color under an achromatic "
                f"Reference treatment: {values}"
            )
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
authored subjects, actions, object states, colors, affirmatively requested visible text,
visible-text exclusions, viewpoint, framing, composition, and requested medium. Describe
only the desired final image.

Authored Description:
{request.description}

Invalid Image Prompt:
{image_prompt}

Detected conflict:
{conflict}
"""


def _attempt(
    phase: Literal["initial", "repair"],
    call: object,
) -> ImagePromptPreparationAttempt:
    return ImagePromptPreparationAttempt(
        phase=phase,
        raw_response=call.content,
        duration_seconds=call.elapsed_seconds,
        total_duration_ns=call.total_duration_ns,
        load_duration_ns=call.load_duration_ns,
        prompt_eval_count=call.prompt_eval_count,
        eval_count=call.eval_count,
        done_reason=call.done_reason,
    )


def _sum_optional_int(
    attempts: tuple[ImagePromptPreparationAttempt, ...],
    field: Literal[
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "eval_count",
    ],
) -> int | None:
    values = [getattr(attempt, field) for attempt in attempts]
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _failed_call_attempt(
    phase: Literal["initial", "repair"],
    error: ModelResponseError,
) -> dict[str, object]:
    return {
        "phase": phase,
        "raw_response": error.raw_response,
        "duration_seconds": error.response_metadata.get("elapsed_seconds"),
        "total_duration_ns": error.response_metadata.get("total_duration_ns"),
        "load_duration_ns": error.response_metadata.get("load_duration_ns"),
        "prompt_eval_count": error.response_metadata.get("prompt_eval_count"),
        "eval_count": error.response_metadata.get("eval_count"),
        "done_reason": error.response_metadata.get("done_reason"),
    }


def _aggregate_attempt_metadata(
    attempts: tuple[dict[str, object], ...],
) -> dict[str, int | str | float | None]:
    totals: dict[str, int | float | None] = {}
    for field in (
        "duration_seconds",
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "eval_count",
    ):
        values = [attempt.get(field) for attempt in attempts]
        totals[field] = (
            sum(value for value in values if isinstance(value, (int, float)))
            if values and all(isinstance(value, (int, float)) for value in values)
            else None
        )
    return {
        "elapsed_seconds": totals["duration_seconds"],
        "total_duration_ns": totals["total_duration_ns"],
        "load_duration_ns": totals["load_duration_ns"],
        "prompt_eval_count": totals["prompt_eval_count"],
        "eval_count": totals["eval_count"],
        "done_reason": attempts[-1].get("done_reason"),
    }


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
        try:
            call = self._runtime.chat_structured(
                prompt=build_image_prompt_preparation_prompt(request),
                schema=ImagePromptModelOutput.model_json_schema(),
                image_path=reference_image_path,
            )
        except ModelResponseError as initial_call_error:
            initial_call_error.response_attempts = (
                _failed_call_attempt("initial", initial_call_error),
            )
            raise
        attempts = [_attempt("initial", call)]
        try:
            output = ImagePromptModelOutput.model_validate_json(
                structured_json_content(call.content)
            )
        except ValidationError as error:
            raise _model_response_error(request, tuple(attempts), error) from error
        try:
            _validate_image_prompt(
                request.description,
                output.image_prompt,
                visual_treatment=output.visual_treatment,
            )
        except ValueError as error:
            try:
                call = self._runtime.chat_structured(
                    prompt=_build_repair_prompt(
                        request,
                        output.image_prompt,
                        str(error),
                    ),
                    schema=_ImagePromptRepairOutput.model_json_schema(),
                )
            except GenerationError as repair_call_error:
                prior_attempts = tuple(
                    attempt.model_dump(mode="json") for attempt in attempts
                )
                failed_attempt = (
                    (_failed_call_attempt("repair", repair_call_error),)
                    if isinstance(repair_call_error, ModelResponseError)
                    else ({"phase": "repair", "raw_response": None},)
                )
                response_attempts = (
                    *prior_attempts,
                    *failed_attempt,
                )
                repair_call_error.response_attempts = response_attempts
                if isinstance(repair_call_error, ModelResponseError):
                    repair_call_error.response_metadata = (
                        _aggregate_attempt_metadata(response_attempts)
                    )
                raise
            attempts.append(_attempt("repair", call))
            try:
                repaired = _ImagePromptRepairOutput.model_validate_json(
                    structured_json_content(call.content)
                )
                _validate_image_prompt(
                    request.description,
                    repaired.image_prompt,
                    visual_treatment=output.visual_treatment,
                )
                output = output.model_copy(
                    update={"image_prompt": repaired.image_prompt}
                )
            except (ValidationError, ValueError) as repair_error:
                raise _model_response_error(
                    request,
                    tuple(attempts),
                    repair_error,
                ) from repair_error
        retained_attempts = tuple(attempts)
        return ImagePromptPreparationResult(
            image_prompt=output.image_prompt,
            raw_response=call.content,
            model_identifier=self._runtime.settings.model,
            prompt_version=request.prompt_version,
            duration_seconds=sum(
                attempt.duration_seconds for attempt in retained_attempts
            ),
            total_duration_ns=_sum_optional_int(
                retained_attempts,
                "total_duration_ns",
            ),
            load_duration_ns=_sum_optional_int(
                retained_attempts,
                "load_duration_ns",
            ),
            prompt_eval_count=_sum_optional_int(
                retained_attempts,
                "prompt_eval_count",
            ),
            eval_count=_sum_optional_int(retained_attempts, "eval_count"),
            done_reason=call.done_reason,
            repair_applied=len(retained_attempts) > 1,
            attempts=retained_attempts,
        )


def _model_response_error(
    request: ImagePromptPreparationRequest,
    attempts: tuple[ImagePromptPreparationAttempt, ...],
    error: Exception,
) -> ModelResponseError:
    last_attempt = attempts[-1]
    return ModelResponseError(
        "Ollama returned an invalid Image Prompt "
        f"for {request.prompt_version}: {error}",
        raw_response=last_attempt.raw_response,
        response_metadata={
            "elapsed_seconds": sum(
                attempt.duration_seconds for attempt in attempts
            ),
            "total_duration_ns": _sum_optional_int(
                attempts,
                "total_duration_ns",
            ),
            "load_duration_ns": _sum_optional_int(
                attempts,
                "load_duration_ns",
            ),
            "prompt_eval_count": _sum_optional_int(
                attempts,
                "prompt_eval_count",
            ),
            "eval_count": _sum_optional_int(attempts, "eval_count"),
            "done_reason": last_attempt.done_reason,
        },
        response_attempts=tuple(
            attempt.model_dump(mode="json") for attempt in attempts
        ),
    )


__all__ = [
    "IMAGE_PROMPT_PREPARATION_VERSION",
    "ImagePromptPreparationAttempt",
    "ImagePromptModelOutput",
    "ImagePromptPreparationRequest",
    "ImagePromptPreparationResult",
    "OllamaImagePromptPreparer",
    "build_image_prompt_preparation_prompt",
]
