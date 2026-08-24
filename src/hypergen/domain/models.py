"""Serialized domain models for HyperGen stack documents."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

CURRENT_SCHEMA_VERSION = 5

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFiniteFloat = Annotated[FiniteFloat, Field(ge=0.0)]


def _reject_nonfinite_json_numbers(value: JsonValue) -> JsonValue:
    """Reject values that cannot survive a standards-compliant JSON round trip."""
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON metadata numbers must be finite")
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_json_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_json_numbers(item)
    return value


class DomainModel(BaseModel):
    """Base configuration shared by serialized domain models."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def _validate_for_serialization(self) -> None:
        """Revalidate recursively before crossing a serialized boundary."""
        values = BaseModel.model_dump(self, mode="python", round_trip=True)
        type(self).model_validate(values)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize only a recursively valid snapshot."""
        self._validate_for_serialization()
        return BaseModel.model_dump(self, **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize to JSON only after recursive revalidation."""
        self._validate_for_serialization()
        return BaseModel.model_dump_json(self, **kwargs)


class RunOverlayMode(StrEnum):
    """Hotspot presentation in Run mode."""

    HIDDEN = "hidden"
    HOVER = "hover"
    VISIBLE = "visible"


class CanvasSize(DomainModel):
    """Fixed logical size shared by every card in a stack."""

    width: PositiveInt = 1024
    height: PositiveInt = 768


class Point(DomainModel):
    """A point in normalized document coordinates."""

    x: NormalizedCoordinate
    y: NormalizedCoordinate


class Polygon(DomainModel):
    """A simple polygon component without holes."""

    points: tuple[Point, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def require_valid_geometry(self) -> Polygon:
        """Reject invalid polygons at the document boundary."""
        from hypergen.domain.geometry import validate_polygon

        issues = validate_polygon(self.points)
        if issues:
            messages = "; ".join(issue.message for issue in issues)
            raise ValueError(messages)
        return self


class ResolvedCardReference(DomainModel):
    """A runtime reference to an existing card."""

    type: Literal["resolved"] = "resolved"
    target_card_id: UUID


class UnresolvedCardReference(DomainModel):
    """A runtime reference whose target is not currently available."""

    type: Literal["unresolved"] = "unresolved"
    target_name: str | None = None


CardReference = Annotated[
    ResolvedCardReference | UnresolvedCardReference,
    Field(discriminator="type"),
]


class ReferenceRole(StrEnum):
    """One deterministic image-reference role."""

    SUBJECT = "subject"
    STYLE = "style"
    SETTING = "setting"
    IDENTITY = SUBJECT
    VISUAL_STYLE = STYLE


class NavigateAction(DomainModel):
    """Navigate to another card or retain an unresolved destination."""

    type: Literal["navigate"] = "navigate"
    target: CardReference


class Interaction(DomainModel):
    """One destination-derived interaction with one or more polygon components."""

    id: UUID = Field(default_factory=uuid4)
    label: NonEmptyString = "Unresolved"
    action: NavigateAction
    polygons: tuple[Polygon, ...] = Field(default_factory=tuple)


class ImageReferenceSnapshot(DomainModel):
    """Exact source state used for one historical image generation."""

    card_id: UUID
    revision_id: UUID
    background_id: UUID


class ImageGenerationInputs(DomainModel):
    """Author-controlled inputs captured for a generated image."""

    description: str
    enriched_description: str | None = None
    subject_reference: ImageReferenceSnapshot | None = None
    style_reference: ImageReferenceSnapshot | None = None
    setting_reference: ImageReferenceSnapshot | None = None

    @property
    def enrichment_context(self) -> str:
        """Return exact generation-time text context for later enrichment."""
        if not self.enriched_description:
            return self.description
        if not self.description:
            return self.enriched_description
        return f"{self.description}\n{self.enriched_description}"

    def references_by_role(
        self,
    ) -> tuple[tuple[ReferenceRole, ImageReferenceSnapshot], ...]:
        """Return assigned references in deterministic role order."""
        return tuple(
            (role, reference)
            for role in ReferenceRole
            if (
                reference := getattr(self, f"{role.value}_reference")
            )
            is not None
        )

    def grouped_references(
        self,
    ) -> tuple[
        tuple[ImageReferenceSnapshot, tuple[ReferenceRole, ...]],
        ...,
    ]:
        """Group roles that use the same unique source background."""
        groups: list[
            tuple[ImageReferenceSnapshot, list[ReferenceRole]]
        ] = []
        group_indexes: dict[tuple[UUID, UUID, UUID], int] = {}
        for role, reference in self.references_by_role():
            key = (
                reference.card_id,
                reference.revision_id,
                reference.background_id,
            )
            index = group_indexes.get(key)
            if index is None:
                group_indexes[key] = len(groups)
                groups.append((reference, [role]))
            else:
                groups[index][1].append(role)
        return tuple(
            (reference, tuple(roles))
            for reference, roles in groups
        )


class ImageGenerationMetadata(DomainModel):
    """Reproducibility metadata for an accepted generated image."""

    inputs: ImageGenerationInputs
    render_prompt: NonEmptyString
    model_identifier: NonEmptyString
    mflux_version: NonEmptyString
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    seed: int
    width: PositiveInt
    height: PositiveInt
    step_count: PositiveInt
    quantization: str | None = None
    effective_settings: dict[str, JsonValue] = Field(default_factory=dict)
    generated_at: AwareDatetime
    duration_seconds: NonNegativeFiniteFloat

    @field_validator("effective_settings")
    @classmethod
    def require_json_safe_settings(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Ensure effective settings have a lossless JSON representation."""
        _reject_nonfinite_json_numbers(value)
        return value


class HotspotSet(DomainModel):
    """The complete applied hotspot set for one card revision."""

    interactions: tuple[Interaction, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_unique_interaction_ids(self) -> HotspotSet:
        """Keep interaction identity unambiguous within an applied set."""
        interaction_ids = [interaction.id for interaction in self.interactions]
        if len(interaction_ids) != len(set(interaction_ids)):
            raise ValueError("interaction IDs must be unique within a hotspot set")
        return self


class GeneratedBackground(DomainModel):
    """One immutable generated image asset and its provenance."""

    id: UUID = Field(default_factory=uuid4)
    type: Literal["generated"] = "generated"
    image_path: NonEmptyString
    generation_metadata: ImageGenerationMetadata
    created_at: AwareDatetime


Background = GeneratedBackground


class EnrichmentReferenceSnapshot(DomainModel):
    """Exact reference image used to produce an Enriched Description."""

    role: ReferenceRole
    card_id: UUID
    revision_id: UUID
    background_id: UUID


class EnrichedDescription(DomainModel):
    """One derived Description and the inputs that established its freshness."""

    text: NonEmptyString
    source_description: str
    references: tuple[EnrichmentReferenceSnapshot, ...] = ()
    model_identifier: NonEmptyString | None = None
    prompt_version: NonEmptyString | None = None

    def is_current(
        self,
        *,
        source_description: str,
        references: tuple[EnrichmentReferenceSnapshot, ...],
    ) -> bool:
        """Return whether the derived text still matches its upstream inputs."""
        return (
            self.source_description == source_description
            and self.references == references
        )


class CardRevision(DomainModel):
    """One complete revision of a card's authored content."""

    id: UUID = Field(default_factory=uuid4)
    description: str = ""
    enriched_description: EnrichedDescription | None = None
    background: Background | None = None
    hotspot_set: HotspotSet | None = None
    subject: CardReference | None = None
    style: CardReference | None = None
    setting: CardReference | None = None

    @field_validator("hotspot_set")
    @classmethod
    def copy_attached_hotspot_set(cls, value: HotspotSet | None) -> HotspotSet | None:
        """Prevent one mutable object graph from being attached to two revisions."""
        return value.model_copy(deep=True) if value is not None else None

    @property
    def image_path(self) -> str | None:
        """Return the active background path for transitional callers."""
        return self.background.image_path if self.background is not None else None

    @property
    def generation_metadata(self) -> ImageGenerationMetadata | None:
        """Return generated-image provenance when available."""
        return (
            self.background.generation_metadata
            if self.background is not None
            else None
        )

    @property
    def created_at(self) -> datetime:
        """Return background creation time, or a neutral value for blank revisions."""
        if self.background is not None:
            return self.background.created_at
        return datetime.min.replace(tzinfo=UTC)


class Card(DomainModel):
    """One named card and its ordered complete revisions."""

    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString
    revisions: tuple[CardRevision, ...] = Field(default_factory=tuple)
    active_revision_id: UUID | None = None

    @model_validator(mode="after")
    def require_known_active_revision(self) -> Card:
        """Reject active revision IDs that do not belong to this card."""
        if not self.revisions:
            first_revision = CardRevision()
            object.__setattr__(self, "revisions", (first_revision,))
            object.__setattr__(self, "active_revision_id", first_revision.id)
        elif self.active_revision_id is None:
            object.__setattr__(self, "active_revision_id", self.revisions[0].id)
        revision_ids = [revision.id for revision in self.revisions]
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("revision IDs must be unique within a card")
        if self.active_revision_id not in set(revision_ids):
            raise ValueError("active_revision_id must identify a revision on this card")
        return self

    @property
    def active_revision(self) -> CardRevision:
        """Return the card's validated active revision."""
        assert self.active_revision_id is not None
        return next(
            revision
            for revision in self.revisions
            if revision.id == self.active_revision_id
        )

class Stack(DomainModel):
    """The authoritative portable HyperGen stack document."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, strict=True)
    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString
    canvas: CanvasSize = Field(default_factory=CanvasSize)
    run_overlay_mode: RunOverlayMode = RunOverlayMode.HIDDEN
    start_card_id: UUID | None = None
    cards: tuple[Card, ...] = Field(default_factory=tuple)

    @field_validator("schema_version")
    @classmethod
    def require_current_schema_version(cls, value: int) -> int:
        """Accept only the schema implemented by the current domain model."""
        if value != CURRENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CURRENT_SCHEMA_VERSION}")
        return value

    @model_validator(mode="after")
    def require_known_start_card(self) -> Stack:
        """Reject start-card IDs that do not belong to this stack."""
        from hypergen.domain.validation import require_unique_card_names

        card_ids = [card.id for card in self.cards]
        known_card_ids = set(card_ids)
        card_names = {card.id: card.name for card in self.cards}
        if len(card_ids) != len(known_card_ids):
            raise ValueError("card IDs must be unique within a stack")
        require_unique_card_names(self.cards)
        if self.start_card_id is not None and self.start_card_id not in known_card_ids:
            raise ValueError("start_card_id must identify a card in this stack")
        revision_ids = [revision.id for card in self.cards for revision in card.revisions]
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("revision IDs must be unique within a stack")
        for card in self.cards:
            for revision in card.revisions:
                for reference in (
                    revision.subject,
                    revision.style,
                    revision.setting,
                ):
                    if not isinstance(reference, ResolvedCardReference):
                        continue
                    if reference.target_card_id not in known_card_ids:
                        raise ValueError(
                            "resolved card references must identify a card in this stack"
                        )
                    if reference.target_card_id == card.id:
                        raise ValueError("a card revision cannot reference its own card")
                if revision.hotspot_set is None:
                    continue
                for interaction in revision.hotspot_set.interactions:
                    target = interaction.action.target
                    if (
                        isinstance(target, ResolvedCardReference)
                        and target.target_card_id not in known_card_ids
                    ):
                        raise ValueError(
                            "resolved card references must identify a card in this stack"
                        )
                    derived_label = (
                        card_names[target.target_card_id]
                        if isinstance(target, ResolvedCardReference)
                        else "Unresolved"
                    )
                    if interaction.label != derived_label:
                        object.__setattr__(
                            interaction,
                            "label",
                            derived_label,
                        )
        return self
