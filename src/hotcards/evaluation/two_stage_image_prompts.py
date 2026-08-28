"""Evaluation-only two-stage Reference-account Image Prompt candidate."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

import hotcards.domain.models as domain_models_module
import hotcards.generation.image_prompt_preparation as image_prompt_preparation_module
import hotcards.generation.ollama_client as ollama_client_module
import hotcards.generation.structured_output as structured_output_module
from hotcards.domain.models import DomainModel, NonEmptyString
from hotcards.evaluation.image_prompts import (
    ImagePromptBenchmarkSettings,
    run_image_prompt_benchmark,
)
from hotcards.evaluation.manifest import contract_digest
from hotcards.generation.errors import GenerationError, ModelResponseError
from hotcards.generation.image_prompt_preparation import (
    ImagePromptModelOutput,
    ImagePromptPreparationAttempt,
    ImagePromptPreparationRequest,
    ImagePromptPreparationResult,
    ImagePromptRepairOutput,
    build_image_prompt_repair_prompt,
    validate_image_prompt,
)
from hotcards.generation.ollama_client import (
    VISION_CAPABILITY,
    OllamaCallResult,
    OllamaRuntime,
    OllamaSettings,
)
from hotcards.generation.structured_output import structured_json_content

TWO_STAGE_IMAGE_PROMPT_VERSION = "image-prompt-two-stage-reference-account-v1"
TWO_STAGE_IMAGE_PROMPT_CANDIDATE_ID = "two-stage-reference-account"
EVIDENCE_GATE_IMAGE_PROMPT_VERSION = "image-prompt-evidence-gate-v1"
EVIDENCE_GATE_IMAGE_PROMPT_CANDIDATE_ID = "target-conditioned-evidence-gate"


class ReferenceAccountOutput(DomainModel):
    """One target-independent natural-language account of a Reference."""

    reference_account: NonEmptyString


class ApplicableReferenceEvidenceOutput(DomainModel):
    """Only Reference evidence authorized to influence one target."""

    applicable_reference_evidence: NonEmptyString | None


def build_reference_account_prompt(
    request: ImagePromptPreparationRequest,
) -> str:
    """Build the target-independent visual account stage."""
    if not request.has_reference or request.reference_description is None:
        raise ValueError("a Reference account requires Reference provenance")
    provenance = json.dumps(
        {"reference_generation_description": request.reference_description},
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Document the attached Reference image before any future target is known.

Return exactly one JSON object with exactly one key, reference_account. Its value must be one
concise natural-language paragraph with no headings or lists.

Write a neutral visual account that:
- Identifies visible subjects and their distinguishing construction, proportions, materials,
  controls, and other stable identity details.
- Describes the visible setting, spatial relationships, composition, viewpoint, framing,
  object state, pose, action, lighting, weather, time, and surface or screen contents.
- Describes reusable visual treatment precisely: color mode and palette, medium, rendering
  technology, line or edge character, texture, dither or halftone pattern, shading, tonal
  strategy, and detail level.
- Naturally distinguishes stable construction and treatment from current scene state,
  viewpoint, framing, contents, and other transient observations.

The generation description is authored semantic provenance for this exact image. Preserve its
explicit identity and treatment terms whenever compatible with visible pixels. Use the pixels
to confirm appearance and add concrete visible detail, not to replace compatible authored
semantics with a generic guess.

Do not infer hidden story facts or conventional real-world colors, materials, age, era, or
technology that are neither visible nor authored. Do not propose a future image, make transfer
recommendations, decide what should be preserved, or refer to an authored target.

Prompt contract: {request.prompt_version}
Input:
{provenance}
"""


def build_two_stage_synthesis_prompt(
    request: ImagePromptPreparationRequest,
    *,
    reference_account: str | None,
) -> str:
    """Build the text-only target synthesis stage."""
    source = json.dumps(
        {
            "authored_description": request.description,
            "reference_attached": request.has_reference,
            "reference_generation_description": (
                request.reference_description if request.has_reference else None
            ),
            "reference_account": reference_account,
        },
        ensure_ascii=False,
        indent=2,
    )
    reference_contract = (
        """\
REFERENCE INTERPRETATION
- The authored Description is authoritative. The Reference account is neutral evidence about
  one selected image, not a second target prompt.
- reference_generation_description is the effective authored prompt used to generate the
  Reference. Treat its explicit identity and visual-treatment language as the primary
  semantic interpretation whenever it applies to requested continuity or transfer.
- A definite visible entity such as "the computer", "the monitor", "the person", or "the
  room", a pronoun referring to one, or "same [entity]" requests continuity with that entity.
- For a continuing entity, retain unmentioned stable identity, construction, and reusable
  treatment. Do not retain transient state, pose, action, viewpoint, framing, composition,
  weather, time, lighting state, screen or sign contents unless the Description requests it.
- A request for visual style alone transfers only medium, linework, texture, palette, shading,
  and rendering. It does not transfer subjects, setting, objects, composition, or narrative.
- Explicit target identity, setting, state, viewpoint, crop, framing, composition, medium,
  palette, linework, texture, shading, rendering technique, and style override the Reference.
- Treat visible rendered appearance as truth. Do not infer conventional real-world color,
  material, age, era, or technology that is not visible or authored.
"""
        if request.has_reference
        else """\
NO REFERENCE
- Prepare the Image Prompt only from the authored Description.
- Add concrete visible detail where useful, but do not invent a medium, rendering technology,
  camera specification, visible text, interaction, or hidden story fact.
"""
    )
    return f"""\
Prepare a production-ready FLUX.2 Image Prompt from the supplied textual evidence.

OUTPUT
- Return exactly one JSON object with exactly these keys: subject_traits, setting_traits,
  visual_treatment, target_overrides, image_prompt.
- The first four fields are private deliberation. Each is one scalar string or null.
- image_prompt is one non-empty natural-language paragraph with no headings or lists.
- Use 30 to 100 words by default. Expand only to preserve explicit detail.

AUTHORITY
- Preserve every explicit authored subject, action, pose, object state, time, weather,
  viewpoint, crop, framing, composition, mood, color, visible object, medium, palette,
  linework, texture, shading, rendering technique, and visual style.
- Preserve close-ups, exact counts and arrangements, spatial relationships, limited fields of
  view, and statements that a subject fills the frame.
- Preserve exact affirmatively authored visible text in quotation marks on its intended
  object. Quoted words in explicit exclusions are forbidden rather than requested.
- When no affirmative quoted text is authored, introduce no visible words, lettering, signs,
  captions, labels, or interface text.

{reference_contract}
PRIVATE DELIBERATION
- subject_traits: stable identity and construction only when subject continuity is requested;
  otherwise null. Exclude state, pose, action, viewpoint, and surface contents.
- setting_traits: stable architecture and environment only when setting continuity is
  requested; otherwise null.
- visual_treatment: reusable medium, linework, texture, palette, shading, and rendering only
  when continuity or style transfer is requested; otherwise null.
- target_overrides: concrete requested changes and every explicit target-authored visual
  treatment; otherwise null.
- Use these fields as a checklist. Every applicable detail must appear in image_prompt.

IMAGE PROMPT
- Synthesize one concrete, positive, standalone description of only the desired final image.
- Put the main subject, action, exact arrangement, and critical authored changes first.
- Describe continuing entities through distinguishing visible construction and treatment, not
  generic stereotypes.
- Resolve every explicit override consistently. Never blend an overridden treatment back in.
- Never mention a Reference, source image, original image, previous version, comparison,
  transfer, replacement, omission, inference, instruction, ambiguity, or alternative.
- Never use process language such as "same style", "inspired by", "instead of", "replacing",
  "unchanged", or "has changed".
- Include no hotspots, navigation, destinations, dimensions, model settings, steps, or seeds.

Prompt contract: {request.prompt_version}
Input:
{source}
"""


def build_applicable_reference_evidence_prompt(
    request: ImagePromptPreparationRequest,
) -> str:
    """Build the target-conditioned lossy Reference-selection stage."""
    if not request.has_reference or request.reference_description is None:
        raise ValueError("Reference evidence selection requires Reference provenance")
    source = json.dumps(
        {
            "authored_description": request.description,
            "reference_generation_description": request.reference_description,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Select only the attached Reference evidence that is authorized to influence the target.

Return exactly one JSON object with exactly one key, applicable_reference_evidence. Its value
must be one concise natural-language paragraph or null. This is a lossy allowlist: the final
Image Prompt writer will receive this value but cannot see the image, provenance, or any
discarded details.

AUTHORITY
- The authored Description is the complete target authority.
- reference_generation_description is authored provenance for this exact Reference. Preserve
  its explicit identity and treatment terminology when applicable and compatible with pixels.
- The Reference was selected because some visible continuity or treatment may be intended,
  unless the Description explicitly replaces every relevant characteristic.

SELECTION
- A definite visible entity such as "the computer", "the monitor", "the person", or "the
  room", a pronoun referring to one, or "same [entity]" requests continuity with that entity.
- For a continuing entity, include only stable identity, silhouette, proportions,
  construction, materials, controls, distinguishing components, and reusable treatment.
- Include stable architecture only when setting continuity is requested.
- For style transfer, include only reusable medium, linework, texture, palette, shading,
  rendering technology, tonal strategy, and detail level.
- Include no Reference state, pose, action, viewpoint, crop, framing, composition, weather,
  time, lighting state, screen or sign contents unless the Description explicitly retains it.
- Explicit target identity, setting, state, viewpoint, crop, framing, composition, medium,
  palette, linework, texture, shading, rendering technique, and style override corresponding
  Reference details.
- Preserve only facts that should appear positively in the final target. Do not include
  rejected details, exclusions, comparisons, alternatives, explanations, or instructions.
- Return null only when the Description explicitly supersedes every visible Reference
  characteristic, including treatment.

Do not restate the target Description. Do not invent hidden facts or conventional real-world
colors, materials, age, era, or technology that are neither visible nor authored.

Prompt contract: {request.prompt_version}
Input:
{source}
"""


def build_evidence_gate_synthesis_prompt(
    request: ImagePromptPreparationRequest,
    *,
    applicable_reference_evidence: str | None,
) -> str:
    """Build synthesis with no access to discarded Reference context."""
    source = json.dumps(
        {
            "authored_description": request.description,
            "applicable_reference_evidence": applicable_reference_evidence,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Prepare a production-ready FLUX.2 Image Prompt from the supplied target and authorized visual
evidence.

OUTPUT
- Return exactly one JSON object with exactly these keys: subject_traits, setting_traits,
  visual_treatment, target_overrides, image_prompt.
- The first four fields are private deliberation. Each is one scalar string or null.
- image_prompt is one non-empty natural-language paragraph with no headings or lists.
- Use 30 to 100 words by default. Expand only to preserve explicit detail.

AUTHORITY
- authored_description is the complete target authority.
- applicable_reference_evidence contains only facts already authorized for this target. Use
  all compatible evidence, but never let it weaken or replace an explicit target detail.
- Preserve every explicit subject, action, pose, object state, count, arrangement, spatial
  relationship, time, weather, viewpoint, crop, framing, composition, mood, color, visible
  object, medium, palette, linework, texture, shading, rendering technique, and visual style.
- Preserve close-ups, limited fields of view, and statements that a subject fills the frame.
- Preserve exact affirmatively authored visible text in quotation marks on its intended
  object. Quoted words in explicit exclusions are forbidden rather than requested.
- When no affirmative quoted text is authored, introduce no visible words, lettering, signs,
  captions, labels, or interface text.

PRIVATE DELIBERATION
- subject_traits: authorized stable entity identity and construction; otherwise null.
- setting_traits: authorized stable architecture and environment; otherwise null.
- visual_treatment: authorized or target-authored medium, linework, texture, palette, shading,
  and rendering; otherwise null.
- target_overrides: every explicit target-authored change; otherwise null.
- Use these fields as a checklist. Every applicable detail must appear in image_prompt.

IMAGE PROMPT
- Synthesize one concrete, positive, standalone description of only the desired final image.
- Put the main subject, action, exact arrangement, and critical authored changes first.
- Resolve every explicit override consistently.
- Never mention evidence selection, a Reference, source image, original image, previous
  version, comparison, transfer, replacement, omission, inference, instruction, ambiguity,
  or alternative.
- Never use process language such as "same style", "inspired by", "instead of", "replacing",
  "unchanged", or "has changed".
- Include no hotspots, navigation, destinations, dimensions, model settings, steps, or seeds.

Prompt contract: {request.prompt_version}
Input:
{source}
"""


def _attempt(
    phase: Literal[
        "reference_account",
        "evidence_selection",
        "synthesis",
        "repair",
    ],
    call: OllamaCallResult,
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


def _raise_model_response_error(
    request: ImagePromptPreparationRequest,
    attempts: tuple[ImagePromptPreparationAttempt, ...],
    error: Exception,
) -> ModelResponseError:
    last = attempts[-1]
    raise ModelResponseError(
        f"Ollama returned an invalid two-stage Image Prompt for "
        f"{request.prompt_version}: {error}",
        raw_response=last.raw_response,
        response_metadata={
            "elapsed_seconds": sum(attempt.duration_seconds for attempt in attempts),
            "total_duration_ns": _sum_optional_int(attempts, "total_duration_ns"),
            "load_duration_ns": _sum_optional_int(attempts, "load_duration_ns"),
            "prompt_eval_count": _sum_optional_int(attempts, "prompt_eval_count"),
            "eval_count": _sum_optional_int(attempts, "eval_count"),
            "done_reason": last.done_reason,
        },
        response_attempts=tuple(
            attempt.model_dump(mode="json") for attempt in attempts
        ),
    ) from error


def _failed_call_attempt(
    phase: Literal[
        "reference_account",
        "evidence_selection",
        "synthesis",
        "repair",
    ],
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


class TwoStageImagePromptPreparer:
    """Prepare an Image Prompt by separating Reference observation and synthesis."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def prepare(
        self,
        request: ImagePromptPreparationRequest,
        *,
        reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult:
        if request.has_reference != (reference_image_path is not None):
            raise ValueError("request reference state must match the attached image")
        self._runtime.require_model(capabilities=frozenset({VISION_CAPABILITY}))
        attempts: list[ImagePromptPreparationAttempt] = []
        reference_account: str | None = None

        if reference_image_path is not None:
            try:
                account_call = self._runtime.chat_structured(
                    prompt=build_reference_account_prompt(request),
                    schema=ReferenceAccountOutput.model_json_schema(),
                    image_path=reference_image_path,
                )
            except GenerationError as error:
                if isinstance(error, ModelResponseError):
                    failed_attempts = (
                        _failed_call_attempt("reference_account", error),
                    )
                    error.response_metadata = _aggregate_attempt_metadata(
                        failed_attempts
                    )
                else:
                    failed_attempts = (
                        {"phase": "reference_account", "raw_response": None},
                    )
                error.response_attempts = failed_attempts
                raise
            attempts.append(_attempt("reference_account", account_call))
            try:
                account_output = ReferenceAccountOutput.model_validate_json(
                    structured_json_content(account_call.content)
                )
            except ValidationError as error:
                _raise_model_response_error(request, tuple(attempts), error)
            reference_account = account_output.reference_account

        try:
            synthesis_call = self._runtime.chat_structured(
                prompt=build_two_stage_synthesis_prompt(
                    request,
                    reference_account=reference_account,
                ),
                schema=ImagePromptModelOutput.model_json_schema(),
            )
        except GenerationError as error:
            prior_attempts = tuple(
                attempt.model_dump(mode="json") for attempt in attempts
            )
            if isinstance(error, ModelResponseError):
                failed_attempts = (
                    *prior_attempts,
                    _failed_call_attempt("synthesis", error),
                )
                error.response_metadata = _aggregate_attempt_metadata(
                    failed_attempts
                )
            else:
                failed_attempts = (
                    *prior_attempts,
                    {"phase": "synthesis", "raw_response": None},
                )
            error.response_attempts = failed_attempts
            raise
        attempts.append(_attempt("synthesis", synthesis_call))
        try:
            output = ImagePromptModelOutput.model_validate_json(
                structured_json_content(synthesis_call.content)
            )
        except ValidationError as error:
            _raise_model_response_error(request, tuple(attempts), error)

        final_call = synthesis_call
        try:
            validate_image_prompt(
                request.description,
                output.image_prompt,
                visual_treatment=output.visual_treatment,
            )
        except ValueError as conflict:
            try:
                repair_call = self._runtime.chat_structured(
                    prompt=build_image_prompt_repair_prompt(
                        request,
                        output.image_prompt,
                        str(conflict),
                        reference_backed_output=False,
                    ),
                    schema=ImagePromptRepairOutput.model_json_schema(),
                )
            except GenerationError as error:
                prior_attempts = tuple(
                    attempt.model_dump(mode="json") for attempt in attempts
                )
                if isinstance(error, ModelResponseError):
                    failed_attempts = (
                        *prior_attempts,
                        _failed_call_attempt("repair", error),
                    )
                    error.response_metadata = _aggregate_attempt_metadata(
                        failed_attempts
                    )
                else:
                    failed_attempts = (
                        *prior_attempts,
                        {"phase": "repair", "raw_response": None},
                    )
                error.response_attempts = failed_attempts
                raise
            attempts.append(_attempt("repair", repair_call))
            try:
                repaired = ImagePromptRepairOutput.model_validate_json(
                    structured_json_content(repair_call.content)
                )
                validate_image_prompt(
                    request.description,
                    repaired.image_prompt,
                    visual_treatment=output.visual_treatment,
                )
            except (ValidationError, ValueError) as error:
                _raise_model_response_error(request, tuple(attempts), error)
            output = output.model_copy(
                update={"image_prompt": repaired.image_prompt}
            )
            final_call = repair_call

        retained_attempts = tuple(attempts)
        return ImagePromptPreparationResult(
            image_prompt=output.image_prompt,
            raw_response=final_call.content,
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
            done_reason=final_call.done_reason,
            repair_applied=retained_attempts[-1].phase == "repair",
            attempts=retained_attempts,
        )


class EvidenceGateImagePromptPreparer:
    """Prepare an Image Prompt from target-filtered Reference evidence."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def prepare(
        self,
        request: ImagePromptPreparationRequest,
        *,
        reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult:
        if request.has_reference != (reference_image_path is not None):
            raise ValueError("request reference state must match the attached image")
        self._runtime.require_model(capabilities=frozenset({VISION_CAPABILITY}))
        attempts: list[ImagePromptPreparationAttempt] = []
        applicable_evidence: str | None = None

        if reference_image_path is not None:
            try:
                evidence_call = self._runtime.chat_structured(
                    prompt=build_applicable_reference_evidence_prompt(request),
                    schema=ApplicableReferenceEvidenceOutput.model_json_schema(),
                    image_path=reference_image_path,
                )
            except GenerationError as error:
                if isinstance(error, ModelResponseError):
                    failed_attempts = (
                        _failed_call_attempt("evidence_selection", error),
                    )
                    error.response_metadata = _aggregate_attempt_metadata(
                        failed_attempts
                    )
                else:
                    failed_attempts = (
                        {"phase": "evidence_selection", "raw_response": None},
                    )
                error.response_attempts = failed_attempts
                raise
            attempts.append(_attempt("evidence_selection", evidence_call))
            try:
                selected = ApplicableReferenceEvidenceOutput.model_validate_json(
                    structured_json_content(evidence_call.content)
                )
            except ValidationError as error:
                _raise_model_response_error(request, tuple(attempts), error)
            applicable_evidence = selected.applicable_reference_evidence

        try:
            synthesis_call = self._runtime.chat_structured(
                prompt=build_evidence_gate_synthesis_prompt(
                    request,
                    applicable_reference_evidence=applicable_evidence,
                ),
                schema=ImagePromptModelOutput.model_json_schema(),
            )
        except GenerationError as error:
            prior_attempts = tuple(
                attempt.model_dump(mode="json") for attempt in attempts
            )
            if isinstance(error, ModelResponseError):
                failed_attempts = (
                    *prior_attempts,
                    _failed_call_attempt("synthesis", error),
                )
                error.response_metadata = _aggregate_attempt_metadata(
                    failed_attempts
                )
            else:
                failed_attempts = (
                    *prior_attempts,
                    {"phase": "synthesis", "raw_response": None},
                )
            error.response_attempts = failed_attempts
            raise
        attempts.append(_attempt("synthesis", synthesis_call))
        try:
            output = ImagePromptModelOutput.model_validate_json(
                structured_json_content(synthesis_call.content)
            )
        except ValidationError as error:
            _raise_model_response_error(request, tuple(attempts), error)

        final_call = synthesis_call
        try:
            validate_image_prompt(
                request.description,
                output.image_prompt,
                visual_treatment=output.visual_treatment,
            )
        except ValueError as conflict:
            try:
                repair_call = self._runtime.chat_structured(
                    prompt=build_image_prompt_repair_prompt(
                        request,
                        output.image_prompt,
                        str(conflict),
                        reference_backed_output=False,
                    ),
                    schema=ImagePromptRepairOutput.model_json_schema(),
                )
            except GenerationError as error:
                prior_attempts = tuple(
                    attempt.model_dump(mode="json") for attempt in attempts
                )
                if isinstance(error, ModelResponseError):
                    failed_attempts = (
                        *prior_attempts,
                        _failed_call_attempt("repair", error),
                    )
                    error.response_metadata = _aggregate_attempt_metadata(
                        failed_attempts
                    )
                else:
                    failed_attempts = (
                        *prior_attempts,
                        {"phase": "repair", "raw_response": None},
                    )
                error.response_attempts = failed_attempts
                raise
            attempts.append(_attempt("repair", repair_call))
            try:
                repaired = ImagePromptRepairOutput.model_validate_json(
                    structured_json_content(repair_call.content)
                )
                validate_image_prompt(
                    request.description,
                    repaired.image_prompt,
                    visual_treatment=output.visual_treatment,
                )
            except (ValidationError, ValueError) as error:
                _raise_model_response_error(request, tuple(attempts), error)
            output = output.model_copy(
                update={"image_prompt": repaired.image_prompt}
            )
            final_call = repair_call

        retained_attempts = tuple(attempts)
        return ImagePromptPreparationResult(
            image_prompt=output.image_prompt,
            raw_response=final_call.content,
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
            done_reason=final_call.done_reason,
            repair_applied=retained_attempts[-1].phase == "repair",
            attempts=retained_attempts,
        )


def run_evidence_gate_image_prompt_benchmark(
    settings: ImagePromptBenchmarkSettings,
) -> Path:
    """Run the target-conditioned evidence-gate candidate."""
    if settings.candidate_id != EVIDENCE_GATE_IMAGE_PROMPT_CANDIDATE_ID:
        raise ValueError("evidence-gate settings use the wrong candidate ID")
    if settings.candidate_prompt_version != EVIDENCE_GATE_IMAGE_PROMPT_VERSION:
        raise ValueError("evidence-gate settings use the wrong prompt version")
    return run_image_prompt_benchmark(
        settings,
        preparer_factory=lambda ollama_settings: EvidenceGateImagePromptPreparer(
            OllamaRuntime(ollama_settings)
        ),
        candidate_contract={
            "version": EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
            "sha256": contract_digest(
                EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
                inspect.getsource(sys.modules[__name__]),
                inspect.getsource(image_prompt_preparation_module),
                inspect.getsource(ollama_client_module),
                inspect.getsource(structured_output_module),
                inspect.getsource(domain_models_module),
            ),
        },
    )


def run_two_stage_image_prompt_benchmark(
    settings: ImagePromptBenchmarkSettings,
) -> Path:
    """Run the evaluation-only two-stage candidate on the maintained benchmark."""
    if settings.candidate_id != TWO_STAGE_IMAGE_PROMPT_CANDIDATE_ID:
        raise ValueError("two-stage benchmark settings use the wrong candidate ID")
    if settings.candidate_prompt_version != TWO_STAGE_IMAGE_PROMPT_VERSION:
        raise ValueError("two-stage benchmark settings use the wrong prompt version")
    return run_image_prompt_benchmark(
        settings,
        preparer_factory=lambda ollama_settings: TwoStageImagePromptPreparer(
            OllamaRuntime(ollama_settings)
        ),
        candidate_contract={
            "version": TWO_STAGE_IMAGE_PROMPT_VERSION,
            "sha256": contract_digest(
                TWO_STAGE_IMAGE_PROMPT_VERSION,
                inspect.getsource(sys.modules[__name__]),
                inspect.getsource(image_prompt_preparation_module),
                inspect.getsource(ollama_client_module),
                inspect.getsource(structured_output_module),
                inspect.getsource(domain_models_module),
            ),
        },
    )


def two_stage_preparer_factory(
    settings: OllamaSettings,
) -> TwoStageImagePromptPreparer:
    """Construct the two-stage candidate for tests and alternate runners."""
    return TwoStageImagePromptPreparer(OllamaRuntime(settings))


__all__ = [
    "EVIDENCE_GATE_IMAGE_PROMPT_CANDIDATE_ID",
    "EVIDENCE_GATE_IMAGE_PROMPT_VERSION",
    "TWO_STAGE_IMAGE_PROMPT_CANDIDATE_ID",
    "TWO_STAGE_IMAGE_PROMPT_VERSION",
    "ApplicableReferenceEvidenceOutput",
    "EvidenceGateImagePromptPreparer",
    "ReferenceAccountOutput",
    "TwoStageImagePromptPreparer",
    "build_applicable_reference_evidence_prompt",
    "build_evidence_gate_synthesis_prompt",
    "build_reference_account_prompt",
    "build_two_stage_synthesis_prompt",
    "run_evidence_gate_image_prompt_benchmark",
    "run_two_stage_image_prompt_benchmark",
    "two_stage_preparer_factory",
]
