"""Serialized domain models for HyperGen stack documents."""

from __future__ import annotations

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

CURRENT_SCHEMA_VERSION = 1

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


class ImageOrigin(StrEnum):
    """How an accepted image revision entered a stack."""

    GENERATED = "generated"
    IMPORTED = "imported"


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


class NavigateAction(DomainModel):
    """Navigate to another card or retain an unresolved destination."""

    type: Literal["navigate"] = "navigate"
    target: CardReference


class Interaction(DomainModel):
    """One semantic interaction with one or more polygon components."""

    id: UUID = Field(default_factory=uuid4)
    label: NonEmptyString
    action: NavigateAction
    polygons: tuple[Polygon, ...] = Field(min_length=1)


class ImageGenerationInputs(DomainModel):
    """Author-controlled inputs captured for a generated image."""

    scene_description: str
    global_style: str
    card_style: str | None = None

    @property
    def effective_style(self) -> str:
        """Return the per-card replacement or the stack-wide default."""
        return self.card_style if self.card_style is not None else self.global_style


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


class HotspotGenerationProvenance(DomainModel):
    """Reproducibility data for an applied generated hotspot set."""

    model_identifier: NonEmptyString
    ollama_version: str | None = None
    prompt_version: NonEmptyString
    schema_version: NonEmptyString
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
    """The complete applied hotspot set for one image revision."""

    interactions: tuple[Interaction, ...] = Field(default_factory=tuple)
    generation_provenance: HotspotGenerationProvenance | None = None

    @model_validator(mode="after")
    def require_unique_interaction_ids(self) -> HotspotSet:
        """Keep interaction identity unambiguous within an applied set."""
        interaction_ids = [interaction.id for interaction in self.interactions]
        if len(interaction_ids) != len(set(interaction_ids)):
            raise ValueError("interaction IDs must be unique within a hotspot set")
        return self


class ImageRevision(DomainModel):
    """An immutable accepted image artifact and its associated hotspot set."""

    id: UUID = Field(default_factory=uuid4)
    image_path: NonEmptyString
    origin: ImageOrigin
    source_filename: str | None = None
    generation_metadata: ImageGenerationMetadata | None = None
    hotspot_set: HotspotSet | None = None
    created_at: AwareDatetime

    @field_validator("hotspot_set")
    @classmethod
    def copy_attached_hotspot_set(cls, value: HotspotSet | None) -> HotspotSet | None:
        """Prevent one mutable object graph from being attached to two revisions."""
        return value.model_copy(deep=True) if value is not None else None

    @model_validator(mode="after")
    def require_generated_image_metadata(self) -> ImageRevision:
        """Ensure generated images always retain their reproducibility metadata."""
        if self.origin is ImageOrigin.GENERATED and self.generation_metadata is None:
            raise ValueError("generated image revisions require generation_metadata")
        return self


class Card(DomainModel):
    """One authored card and all of its accepted image revisions."""

    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString
    scene_description: str = ""
    interaction_description: str = ""
    card_style: str | None = None
    image_revisions: tuple[ImageRevision, ...] = Field(default_factory=tuple)
    active_revision_id: UUID | None = None

    @model_validator(mode="after")
    def require_known_active_revision(self) -> Card:
        """Reject active revision IDs that do not belong to this card."""
        revision_ids = [revision.id for revision in self.image_revisions]
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("image revision IDs must be unique within a card")
        if self.active_revision_id is not None and self.active_revision_id not in set(revision_ids):
            raise ValueError("active_revision_id must identify an image revision on this card")
        return self


class Stack(DomainModel):
    """The authoritative portable HyperGen stack document."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, strict=True)
    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString
    global_style: str = ""
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
        if len(card_ids) != len(known_card_ids):
            raise ValueError("card IDs must be unique within a stack")
        require_unique_card_names(self.cards)
        if self.start_card_id is not None and self.start_card_id not in known_card_ids:
            raise ValueError("start_card_id must identify a card in this stack")
        revision_ids = [revision.id for card in self.cards for revision in card.image_revisions]
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("image revision IDs must be unique within a stack")
        for card in self.cards:
            for revision in card.image_revisions:
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
        return self
