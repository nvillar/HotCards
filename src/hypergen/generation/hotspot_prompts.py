"""Structured hotspot-generation contract and Ollama adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, StringConstraints, ValidationError, model_validator

from hypergen.domain.geometry import (
    DEFAULT_MODEL_COORDINATE_EXTENT,
    model_point_to_document,
    validate_polygon,
)
from hypergen.domain.models import (
    CardReference,
    DomainModel,
    Interaction,
    NavigateAction,
    NonEmptyString,
    NonNegativeFiniteFloat,
    Point,
    Polygon,
    ResolvedCardReference,
    UnresolvedCardReference,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaCallResult, OllamaRuntime

HOTSPOT_PROMPT_VERSION = "hotspot-prompt-v2"
HOTSPOT_SCHEMA_VERSION = "hotspot-schema-v2"
UNRESOLVED_DESTINATION_TOKEN = "UNRESOLVED"
MAX_INTERACTIONS = 4
MAX_COMPONENTS_PER_INTERACTION = 2
MAX_POINTS_PER_COMPONENT = 12
CardToken = Annotated[str, StringConstraints(pattern=r"^C[1-9][0-9]*$")]


class CardCatalogueEntry(DomainModel):
    """Request-local author-facing card information without UUIDs."""

    token: CardToken
    name: NonEmptyString
    description: str = ""


class HotspotGenerationRequest(DomainModel):
    """UI-independent request for hotspot proposals."""

    image_path: Path
    interaction_description: str
    card_catalogue: tuple[CardCatalogueEntry, ...]
    coordinate_extent: int = Field(default=DEFAULT_MODEL_COORDINATE_EXTENT, gt=0)
    prompt_version: Literal[HOTSPOT_PROMPT_VERSION] = HOTSPOT_PROMPT_VERSION
    schema_version: Literal[HOTSPOT_SCHEMA_VERSION] = HOTSPOT_SCHEMA_VERSION

    @model_validator(mode="after")
    def require_unique_card_tokens(self) -> HotspotGenerationRequest:
        """Keep every request-local destination token unambiguous."""
        tokens = [card.token for card in self.card_catalogue]
        if len(tokens) != len(set(tokens)):
            raise ValueError("card catalogue tokens must be unique")
        return self


class ModelPoint(DomainModel):
    """Integer point emitted in the request coordinate extent."""

    x: int
    y: int


class ModelPolygon(DomainModel):
    """One model-emitted polygon component."""

    points: tuple[ModelPoint, ...] = Field(
        min_length=3,
        max_length=MAX_POINTS_PER_COMPONENT,
    )


class ModelInteractionOutput(DomainModel):
    """One structured interaction emitted by Ollama."""

    source_interaction_index: int = Field(ge=1, le=MAX_INTERACTIONS)
    label: NonEmptyString
    destination_token: CardToken | Literal["UNRESOLVED"]
    polygons: tuple[ModelPolygon, ...] = Field(
        min_length=1,
        max_length=MAX_COMPONENTS_PER_INTERACTION,
    )


class ModelUnlocatedInteractionOutput(DomainModel):
    """One author-described subject the model cannot locate in the image."""

    source_interaction_index: int = Field(ge=1, le=MAX_INTERACTIONS)
    label: NonEmptyString
    reason: NonEmptyString


class HotspotModelOutput(DomainModel):
    """Complete structured response requested from Ollama."""

    interactions: tuple[ModelInteractionOutput, ...] = Field(
        default_factory=tuple,
        max_length=MAX_INTERACTIONS,
    )
    unlocated_interactions: tuple[ModelUnlocatedInteractionOutput, ...] = Field(
        default_factory=tuple,
        max_length=MAX_INTERACTIONS,
    )


def build_hotspot_response_schema(request: HotspotGenerationRequest) -> dict[str, Any]:
    """Constrain destination output to this request's tokens plus safe abstention."""
    schema = HotspotModelOutput.model_json_schema()
    interaction_properties = schema["$defs"]["ModelInteractionOutput"]["properties"]
    interaction_properties["destination_token"] = {
        "title": "Destination Token",
        "type": "string",
        "enum": [
            *(card.token for card in request.card_catalogue),
            UNRESOLVED_DESTINATION_TOKEN,
        ],
    }
    return schema


class ExistingCandidateTarget(DomainModel):
    """Candidate resolved to a supplied request-local card."""

    type: Literal["existing"] = "existing"
    card_token: CardToken
    card_name: NonEmptyString


class NewCandidateTarget(DomainModel):
    """Candidate proposes an explicit new-card action for review."""

    type: Literal["new"] = "new"
    proposed_name: NonEmptyString


class UnresolvedCandidateTarget(DomainModel):
    """Candidate remains unresolved for author review."""

    type: Literal["unresolved"] = "unresolved"
    description: str = ""


CandidateTarget = Annotated[
    ExistingCandidateTarget | NewCandidateTarget | UnresolvedCandidateTarget,
    Field(discriminator="type"),
]


class CandidatePolygon(DomainModel):
    """Clamped candidate geometry plus validation warnings."""

    points: tuple[Point, ...]
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class HotspotProposal(DomainModel):
    """One reviewable candidate interaction."""

    source_interaction_index: int = Field(ge=1, le=MAX_INTERACTIONS)
    label: NonEmptyString
    target: CandidateTarget
    polygons: tuple[CandidatePolygon, ...]


class HotspotReconciliationWarning(DomainModel):
    """Actionable warning for an interaction subject absent from the image."""

    source_interaction_index: int = Field(ge=1, le=MAX_INTERACTIONS)
    label: NonEmptyString
    reason: NonEmptyString

    @property
    def message(self) -> str:
        return (
            f'Could not locate "{self.label}" in the image: {self.reason}. '
            "Edit Scene and regenerate the background, draw a manual hotspot, "
            "or adjust/remove the interaction."
        )


class HotspotGenerationResult(DomainModel):
    """Strict candidate output with diagnostics and raw response access."""

    proposals: tuple[HotspotProposal, ...]
    warnings: tuple[str, ...]
    reconciliation_warnings: tuple[HotspotReconciliationWarning, ...] = Field(
        default_factory=tuple
    )
    raw_response: NonEmptyString
    model_identifier: NonEmptyString
    prompt_version: NonEmptyString
    schema_version: NonEmptyString
    duration_seconds: NonNegativeFiniteFloat
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None


def build_hotspot_prompt(request: HotspotGenerationRequest) -> str:
    """Build the versioned vision prompt without exposing stable UUIDs."""
    catalogue = [
        {
            "token": card.token,
            "name": card.name,
            "description": card.description,
        }
        for card in request.card_catalogue
    ]
    payload = {
        "interaction_description": request.interaction_description,
        "card_catalogue": catalogue,
        "coordinate_extent": request.coordinate_extent,
    }
    return f"""\
Analyze the supplied card image and propose clickable polygon hotspots for the author's interaction
description.

Return JSON matching the supplied schema.
- Use integer coordinates from 0 through {request.coordinate_extent}.
- Use one interaction per semantic action and one or more polygon components per interaction.
- The interaction description is authoritative. Return each described interaction exactly once.
  Do not add interactions merely because another object is visible.
- If a described subject cannot be located in the image, do not invent geometry. Omit it from
  interactions and return it once in unlocated_interactions with its source index, short label,
  and a concrete reason.
- Number the semantic actions in the interaction description from 1 in textual order. Return that
  number as source_interaction_index so every result remains tied to its author-described action.
- Return only the interactions needed, not the maximum allowed. The safety bounds are at most
  {MAX_INTERACTIONS} interactions, {MAX_COMPONENTS_PER_INTERACTION} polygon components per
  interaction, and {MAX_POINTS_PER_COMPONENT} points per component.
- Keep polygons simple and reasonably editable; do not create holes.
- Match destination names in the interaction description to the supplied card catalogue. Return the
  exact opaque token of the intended supplied card as destination_token. Return "UNRESOLVED" only
  when no supplied destination can be inferred.
- Destination resolution comes from the author's text and catalogue, not from visible image
  content. Do not mark a destination unresolved merely because the destination card is not pictured.
- Never invent or return UUIDs.
- Do not report confidence scores.

Prompt contract: {request.prompt_version}
Schema contract: {request.schema_version}
Request:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _candidate_target(
    destination_token: CardToken | Literal["UNRESOLVED"],
    catalogue: dict[str, CardCatalogueEntry],
    warnings: list[str],
) -> CandidateTarget:
    if destination_token == UNRESOLVED_DESTINATION_TOKEN:
        return UnresolvedCandidateTarget()
    card = catalogue.get(destination_token)
    if card is None:
        warning = f"model selected unknown card token {destination_token!r}"
        warnings.append(warning)
        return UnresolvedCandidateTarget(description=warning)
    return ExistingCandidateTarget(card_token=card.token, card_name=card.name)


def _candidate_polygon(
    output: ModelPolygon,
    *,
    extent: int,
    warnings: list[str],
) -> CandidatePolygon:
    polygon_warnings: list[str] = []
    if any(
        point.x < 0 or point.x > extent or point.y < 0 or point.y > extent
        for point in output.points
    ):
        polygon_warnings.append("model coordinates were clamped to the canvas")
    points = tuple(
        model_point_to_document(point.x, point.y, extent=extent) for point in output.points
    )
    polygon_warnings.extend(issue.message for issue in validate_polygon(points))
    warnings.extend(polygon_warnings)
    return CandidatePolygon(points=points, warnings=tuple(polygon_warnings))


def candidate_target_to_reference(
    target: CandidateTarget,
    *,
    card_ids_by_token: dict[str, UUID],
) -> CardReference:
    """Convert a reviewed request-local target into a persisted reference."""
    if isinstance(target, ExistingCandidateTarget):
        card_id = card_ids_by_token.get(target.card_token)
        if card_id is None:
            raise ValueError(f"no card ID is mapped for candidate token {target.card_token!r}")
        return ResolvedCardReference(target_card_id=card_id)
    if isinstance(target, NewCandidateTarget):
        return UnresolvedCardReference(target_name=target.proposed_name)
    return UnresolvedCardReference(target_name=target.description or None)


def apply_hotspot_proposal(
    proposal: HotspotProposal,
    *,
    card_ids_by_token: dict[str, UUID],
) -> Interaction:
    """Cross the Apply boundary, rejecting invalid candidate geometry."""
    return Interaction(
        label=proposal.label,
        action=NavigateAction(
            target=candidate_target_to_reference(
                proposal.target,
                card_ids_by_token=card_ids_by_token,
            )
        ),
        polygons=tuple(Polygon(points=polygon.points) for polygon in proposal.polygons),
    )


class OllamaHotspotGenerator:
    """Generate and normalize reviewable hotspot candidates."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def generate(self, request: HotspotGenerationRequest) -> HotspotGenerationResult:
        """Run one strict multimodal hotspot request."""
        if not request.image_path.is_file():
            raise ModelResponseError(f"hotspot input image does not exist: {request.image_path}")
        call = self._runtime.chat_structured(
            prompt=build_hotspot_prompt(request),
            schema=build_hotspot_response_schema(request),
            image_path=request.image_path,
        )
        return normalize_hotspot_response(
            request=request,
            call=call,
            model_identifier=self._runtime.settings.model,
        )


def normalize_hotspot_response(
    *,
    request: HotspotGenerationRequest,
    call: OllamaCallResult,
    model_identifier: str,
) -> HotspotGenerationResult:
    """Strictly parse and normalize one raw response using the production boundary."""
    try:
        output = HotspotModelOutput.model_validate_json(call.content)
    except ValidationError as error:
        raise ModelResponseError(
            f"Ollama returned an invalid hotspot response for {request.schema_version}: {error}",
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

    warnings: list[str] = []
    catalogue = {card.token: card for card in request.card_catalogue}
    proposals = tuple(
        HotspotProposal(
            source_interaction_index=interaction.source_interaction_index,
            label=interaction.label,
            target=_candidate_target(interaction.destination_token, catalogue, warnings),
            polygons=tuple(
                _candidate_polygon(
                    polygon,
                    extent=request.coordinate_extent,
                    warnings=warnings,
                )
                for polygon in interaction.polygons
            ),
        )
        for interaction in output.interactions
    )
    seen_source_interactions: set[int] = set()
    for proposal in proposals:
        if proposal.source_interaction_index in seen_source_interactions:
            warnings.append(
                "model repeated source interaction "
                f"{proposal.source_interaction_index}: {proposal.label!r}"
            )
        seen_source_interactions.add(proposal.source_interaction_index)
    reconciliation_warnings: list[HotspotReconciliationWarning] = []
    seen_unlocated_interactions: set[int] = set()
    for warning in output.unlocated_interactions:
        if warning.source_interaction_index in seen_source_interactions:
            warnings.append(
                "model both located and marked source interaction "
                f"{warning.source_interaction_index} as unlocated; "
                "the contradictory unlocated warning was ignored"
            )
            continue
        if warning.source_interaction_index in seen_unlocated_interactions:
            warnings.append(
                "model repeated unlocated source interaction "
                f"{warning.source_interaction_index}: {warning.label!r}"
            )
            continue
        seen_unlocated_interactions.add(warning.source_interaction_index)
        reconciliation_warnings.append(
            HotspotReconciliationWarning(
                source_interaction_index=warning.source_interaction_index,
                label=warning.label,
                reason=warning.reason,
            )
        )
    if len(output.interactions) == MAX_INTERACTIONS:
        warnings.append(
            f"model response reached the {MAX_INTERACTIONS}-interaction safety limit; "
            "output may be repetitive or truncated"
        )
    return HotspotGenerationResult(
        proposals=proposals,
        warnings=tuple(warnings),
        reconciliation_warnings=tuple(reconciliation_warnings),
        raw_response=call.content,
        model_identifier=model_identifier,
        prompt_version=request.prompt_version,
        schema_version=request.schema_version,
        duration_seconds=call.elapsed_seconds,
        total_duration_ns=call.total_duration_ns,
        load_duration_ns=call.load_duration_ns,
        prompt_eval_count=call.prompt_eval_count,
        eval_count=call.eval_count,
        done_reason=call.done_reason,
    )
