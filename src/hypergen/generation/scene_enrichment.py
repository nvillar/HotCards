"""Opt-in Ollama contract for enriching an author-visible Description."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from hypergen.domain.models import (
    DomainModel,
    NonEmptyString,
    NonNegativeFiniteFloat,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime
from hypergen.generation.structured_output import structured_json_content

SCENE_ENRICHMENT_PROMPT_VERSION = "scene-enrichment-v8"


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
        {
            "authored_description": request.scene,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Rewrite the authored Description as a production-ready FLUX.2 image prompt.

OUTPUT
- Return exactly {{"scene": "<final Description>"}} with no surrounding text.
- Write one natural-language paragraph with no headings or lists.
- Use 30 to 80 words by default. Expand only as needed to preserve explicit input details in
  a genuinely complex scene, and remove repetition before adding length.
- Order content by visual priority: main subject, key action or pose, critical visual style,
  essential setting and context, then secondary details. Important elements come first.
- Describe only the desired visible result in direct, positive language. Replace exclusions
  with positive states such as "an empty courtyard" or "the gate is firmly closed."

AUTHORITY
- The authored Description is the complete source of scene content. Preserve every explicit
  subject, action, pose, object state, time, weather, viewpoint, crop, framing, composition,
  mood, color, and visible object.
- Begin from the authored visible event and restate its defining composition before adding
  concrete visual detail. A short Description is not permission to widen the view, restore
  omitted surroundings, introduce new subjects or objects, or invent a broader scene.
- Preserve close-ups, limited fields of view, and statements that the subject fills the
  frame. Convert exclusions into equivalent positive composition constraints without
  weakening them.
- Produce one synthesis without discussing instructions, constraints, or the rewrite
  process.

FLUX.2 DETAIL GUIDANCE
- Add concrete form, scale, texture, materials, lighting quality and direction, shadows,
  spatial relationships, atmosphere, framing, and composition when they improve the image.
- Associate each color, material, and spatial detail with its specific object. Preserve exact
  authored color names and hex codes on their intended objects.
- Add camera bodies, lenses, film stocks, aperture, or depth of field only when the authored
  Description is explicitly photographic.
- Preserve authored visible text exactly in quotation marks and on its intended object.
  Describe placement and typography only when authored. When no quoted text is authored,
  introduce no visible words, lettering, signs, captions, or labels.
- Make abstract qualities visually concrete without inventing story facts, interactions,
  navigation, hotspots, destinations, dimensions, model settings, or technical parameters.

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


class _AuthoredIntentConflict(ValueError):
    """A rewritten Description contradicts an explicit authored object state."""


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


def _validate_authored_object_states(authored: str, scene: str) -> None:
    output_assertions = _authored_state_assertions(scene)
    for noun, state in _authored_state_assertions(authored):
        for opposite in _STATE_OPPOSITES[state]:
            if (noun, opposite) in output_assertions:
                raise _AuthoredIntentConflict(
                    f"authored {noun!r} is {state!r}, but the rewrite makes it "
                    f"{opposite!r}"
                )


def _validate_authored_fidelity(
    request: SceneEnrichmentRequest,
    output: SceneEnrichmentModelOutput,
) -> None:
    _validate_authored_object_states(request.scene, output.scene)
    authored_quotes, _ = _quoted_text(request.scene)
    output_quotes, output_quotes_are_balanced = _quoted_text(output.scene)
    if not output_quotes_are_balanced:
        raise ValueError("rewritten Description contains unclosed visible text")
    invented_quotes = output_quotes - authored_quotes
    if invented_quotes:
        values = ", ".join(sorted(repr(value) for value in invented_quotes))
        raise ValueError(f"rewritten Description invented visible text: {values}")
    missing_quotes = authored_quotes - output_quotes
    if missing_quotes:
        values = ", ".join(sorted(repr(value) for value in missing_quotes))
        raise ValueError(f"rewritten Description omitted or changed visible text: {values}")


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
            _validate_authored_fidelity(request, output)
        except _AuthoredIntentConflict as error:
            call = self._runtime.chat_structured(
                prompt=_build_intent_repair_prompt(request, output, str(error)),
                schema=SceneEnrichmentModelOutput.model_json_schema(),
            )
            try:
                output = SceneEnrichmentModelOutput.model_validate_json(
                    structured_json_content(call.content)
                )
                _validate_authored_fidelity(request, output)
            except (ValidationError, ValueError) as repair_error:
                raise _model_response_error(
                    request,
                    call,
                    repair_error,
                ) from repair_error
        except ValidationError as error:
            raise _model_response_error(request, call, error) from error
        except ValueError as error:
            raise _model_response_error(request, call, error) from error
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


def _build_intent_repair_prompt(
    request: SceneEnrichmentRequest,
    output: SceneEnrichmentModelOutput,
    conflict: str,
) -> str:
    return f"""\
Repair a FLUX.2 Description that contradicted explicit authored intent.

Return exactly {{"scene": "<corrected Description>"}} with no surrounding text.
Change only what is necessary to resolve the conflict. The authored Description
is authoritative for every subject, action, pose, object state, viewpoint,
framing, and composition.

Authored Description:
{request.scene}

Invalid candidate:
{output.scene}

Detected conflict:
{conflict}
"""


def _model_response_error(
    request: SceneEnrichmentRequest,
    call: object,
    error: Exception,
) -> ModelResponseError:
    return ModelResponseError(
        "Ollama returned a Description that did not preserve authored intent "
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
    "OllamaSceneEnricher",
    "SCENE_ENRICHMENT_PROMPT_VERSION",
    "SceneEnrichmentRequest",
    "SceneEnrichmentResult",
    "build_scene_enrichment_prompt",
]
