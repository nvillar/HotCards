"""Serialized domain models for HyperGen stack documents."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

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

CURRENT_SCHEMA_VERSION = 9

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
    target_name: NonEmptyString | None = None

    @field_validator("target_name", mode="before")
    @classmethod
    def blank_target_has_no_retained_name(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


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


class KeyDefinition(DomainModel):
    """One stack-owned binary state token."""

    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString


class HotspotConditions(DomainModel):
    """The complete all-of condition that gates one hotspot."""

    requires: tuple[UUID, ...] = Field(default_factory=tuple)
    forbids: tuple[UUID, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_unambiguous_keys(self) -> HotspotConditions:
        if len(self.requires) != len(set(self.requires)):
            raise ValueError("required keys must be unique")
        if len(self.forbids) != len(set(self.forbids)):
            raise ValueError("forbidden keys must be unique")
        if set(self.requires) & set(self.forbids):
            raise ValueError("a key cannot be both required and forbidden")
        return self


class HotspotKeyChanges(DomainModel):
    """One atomic key transition applied before optional navigation."""

    remove: tuple[UUID, ...] = Field(default_factory=tuple)
    grant: tuple[UUID, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_unambiguous_changes(self) -> HotspotKeyChanges:
        if len(self.remove) != len(set(self.remove)):
            raise ValueError("removed keys must be unique")
        if len(self.grant) != len(set(self.grant)):
            raise ValueError("granted keys must be unique")
        if set(self.remove) & set(self.grant):
            raise ValueError("a key cannot be both removed and granted")
        return self


class Interaction(DomainModel):
    """One conditional interaction with key changes, navigation, and geometry."""

    id: UUID = Field(default_factory=uuid4)
    label: NonEmptyString = "New Hotspot"
    conditions: HotspotConditions = Field(default_factory=HotspotConditions)
    key_changes: HotspotKeyChanges = Field(default_factory=HotspotKeyChanges)
    action: NavigateAction | None = None
    polygons: tuple[Polygon, ...] = Field(default_factory=tuple)


class ImageReferenceSnapshot(DomainModel):
    """Exact source state used for one historical image generation."""

    card_id: UUID
    revision_id: UUID
    background_id: UUID


class StyleDefinition(DomainModel):
    """One stack-owned named rendering treatment."""

    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString
    prompt_text: str = ""

    @field_validator("prompt_text")
    @classmethod
    def trim_prompt_text(cls, value: str) -> str:
        return value.strip()


def _built_in_style_id(slug: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"https://hypergen.app/styles/{slug}")


HYPERCARD_STYLE_ID = _built_in_style_id("hypercard")

BUILT_IN_STYLES = (
    StyleDefinition(
        id=HYPERCARD_STYLE_ID,
        name="HyperCard",
        prompt_text=(
            "Render the entire image using pure black and pure white pixels. "
            "Translate every described color into a distinct 1-bit dither pattern. "
            "Use early Macintosh HyperCard bitmap artwork with hard pixel edges, "
            "sparse high-contrast linework, and ordered dithering for every midtone."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("cinematic-film"),
        name="Cinematic Film",
        prompt_text=(
            "Rendered as a cinematic live-action film still with naturalistic lens "
            "detail, nuanced color grading, rich shadow separation, motivated "
            "directional lighting, and subtle organic film grain."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("isometric-game"),
        name="Isometric Game",
        prompt_text=(
            "Rendered as stylized isometric game art with a consistent top-down "
            "isometric projection, clean geometric forms, simplified surfaces, "
            "bright readable colors, soft ambient shadows, and even illumination."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("pixel-art"),
        name="Pixel Art",
        prompt_text=(
            "Rendered as crisp 16-bit pixel art with grid-aligned hard-edged pixels, "
            "a limited color palette, flat color clusters, deliberate dithering, "
            "and pixel-scale shading."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("watercolor-painting"),
        name="Watercolor Painting",
        prompt_text=(
            "Rendered as a watercolor painting with translucent layered washes, "
            "visible pigment blooms, softly bleeding edges, luminous white paper, "
            "natural fiber texture, and loose expressive brushwork."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("color-pencil"),
        name="Color Pencil",
        prompt_text=(
            "Rendered as a colored-pencil drawing with visible layered strokes, "
            "waxy pigment texture, burnished highlights, fine hatching and "
            "crosshatching, vibrant color, and natural paper grain."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("pencil-sketch"),
        name="Pencil Sketch",
        prompt_text=(
            "Render the entire image as monochrome graphite on white paper. "
            "Translate every described color into graphite value and texture. "
            "Use expressive pencil linework, crosshatched and softly smudged tonal "
            "shading, erased highlights, and visible paper grain."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("glazed-ceramic"),
        name="Glazed Ceramic",
        prompt_text=(
            "Depicted as a hand-painted glazed ceramic diorama: smooth sculpted clay "
            "forms, saturated underglaze color, vitreous gloss, soft specular "
            "reflections, subtle glaze pooling, and handcrafted irregularities."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("graphic-novel"),
        name="Graphic Novel",
        prompt_text=(
            "Rendered as a graphic novel illustration with bold black ink contours, "
            "clean expressive linework, flat cel-shaded color, a controlled palette, "
            "hard-edged shadows, and dramatic tonal contrast."
        ),
    ),
    StyleDefinition(
        id=_built_in_style_id("miniature-toy"),
        name="Miniature Toy",
        prompt_text=(
            "Depicted as a handcrafted miniature toy diorama with painted resin "
            "figures and scenery, simplified tactile forms, fine model-making "
            "detail, soft studio illumination, and shallow macro-lens depth of field."
        ),
    ),
)


class StyleSnapshot(DomainModel):
    """Exact selected Style state used for one image generation."""

    style_id: UUID
    name: NonEmptyString
    prompt_text: str


class ImageGenerationInputs(DomainModel):
    """Author-controlled inputs captured for a generated image."""

    description: str
    image_prompt: NonEmptyString
    reference: ImageReferenceSnapshot | None = None
    style: StyleSnapshot | None = None

    @property
    def effective_description(self) -> str:
        """Return the prepared Image Prompt sent to image generation."""
        return self.image_prompt

    def references(self) -> tuple[ImageReferenceSnapshot, ...]:
        """Return the optional generation reference as a uniform tuple."""
        return (self.reference,) if self.reference is not None else ()


class LegacyImageGenerationInputs(DomainModel):
    """Exact schema-v5 inputs retained in historical generation metadata."""

    description: str
    enriched_description: str | None = None
    subject_reference: ImageReferenceSnapshot | None = None
    style_reference: ImageReferenceSnapshot | None = None
    setting_reference: ImageReferenceSnapshot | None = None

    @property
    def effective_description(self) -> str:
        """Return the one Description sent to image generation."""
        return self.enriched_description or self.description

    def references_by_role(
        self,
    ) -> tuple[tuple[ReferenceRole, ImageReferenceSnapshot], ...]:
        """Return assigned references in deterministic role order."""
        return tuple(
            (role, reference)
            for role in ReferenceRole
            if (reference := getattr(self, f"{role.value}_reference")) is not None
        )

    def grouped_references(
        self,
    ) -> tuple[
        tuple[ImageReferenceSnapshot, tuple[ReferenceRole, ...]],
        ...,
    ]:
        """Group roles that use the same unique source background."""
        groups: list[tuple[ImageReferenceSnapshot, list[ReferenceRole]]] = []
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
        return tuple((reference, tuple(roles)) for reference, roles in groups)


class ImageGenerationMetadata(DomainModel):
    """Reproducibility metadata for an accepted generated image."""

    inputs: ImageGenerationInputs | LegacyImageGenerationInputs
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


class ImagePrompt(DomainModel):
    """One prepared Image Prompt and the inputs that established its freshness."""

    text: NonEmptyString
    source_description: str
    reference: ImageReferenceSnapshot | None = None
    model_identifier: NonEmptyString | None = None
    prompt_version: NonEmptyString | None = None

    def is_current(
        self,
        *,
        source_description: str,
        reference: ImageReferenceSnapshot | None = None,
        model_identifier: str | None = None,
        prompt_version: str | None = None,
    ) -> bool:
        """Return whether the derived text still matches its upstream inputs."""
        return (
            self.source_description == source_description
            and self.reference == reference
            and (model_identifier is None or self.model_identifier == model_identifier)
            and (prompt_version is None or self.prompt_version == prompt_version)
        )


class CardRevision(DomainModel):
    """One complete revision of a card's authored content."""

    id: UUID = Field(default_factory=uuid4)
    description: str = ""
    image_prompt: ImagePrompt | None = None
    background: Background | None = None
    hotspot_set: HotspotSet | None = None
    reference: CardReference | None = None
    style_id: UUID | None = None

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
        return self.background.generation_metadata if self.background is not None else None

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
            revision for revision in self.revisions if revision.id == self.active_revision_id
        )


def _automatic_interaction_label(
    interaction: Interaction,
    *,
    card_names: dict[UUID, str],
    key_names: dict[UUID, str],
) -> str:
    changes: list[str] = []
    if interaction.key_changes.remove:
        changes.append(
            f"Remove {key_names[interaction.key_changes.remove[0]]}"
            if len(interaction.key_changes.remove) == 1
            else "Remove keys"
        )
    if interaction.key_changes.grant:
        changes.append(
            f"Grant {key_names[interaction.key_changes.grant[0]]}"
            if len(interaction.key_changes.grant) == 1
            else "Grant keys"
        )
    destination: str | None = None
    action = interaction.action
    if action is not None:
        target = action.target
        destination = (
            card_names[target.target_card_id]
            if isinstance(target, ResolvedCardReference)
            else target.target_name or "Unresolved destination"
        )
    if changes and destination is not None:
        return f"{' · '.join(changes)} → {destination}"
    if changes:
        return " · ".join(changes)
    if destination is not None:
        return destination
    return "New Hotspot"


class Stack(DomainModel):
    """The authoritative portable HyperGen stack document."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, strict=True)
    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString
    canvas: CanvasSize = Field(default_factory=CanvasSize)
    run_overlay_mode: RunOverlayMode = RunOverlayMode.HIDDEN
    start_card_id: UUID | None = None
    styles: tuple[StyleDefinition, ...] = Field(default_factory=lambda: BUILT_IN_STYLES)
    new_card_style_id: UUID | None = HYPERCARD_STYLE_ID
    keys: tuple[KeyDefinition, ...] = Field(default_factory=tuple)
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
        style_ids = [style.id for style in self.styles]
        known_style_ids = set(style_ids)
        if len(style_ids) != len(known_style_ids):
            raise ValueError("Style IDs must be unique within a stack")
        style_names = [style.name.casefold() for style in self.styles]
        if len(style_names) != len(set(style_names)):
            raise ValueError("Style names must be unique within a stack")
        if self.new_card_style_id is not None and self.new_card_style_id not in known_style_ids:
            raise ValueError("new_card_style_id must identify a Style in this stack")
        key_ids = [key.id for key in self.keys]
        known_key_ids = set(key_ids)
        if len(key_ids) != len(known_key_ids):
            raise ValueError("Key IDs must be unique within a stack")
        key_names = [key.name.casefold() for key in self.keys]
        if len(key_names) != len(set(key_names)):
            raise ValueError("Key names must be unique within a stack")
        key_names_by_id = {key.id: key.name for key in self.keys}
        for card in self.cards:
            for revision in card.revisions:
                if revision.style_id is not None and revision.style_id not in known_style_ids:
                    raise ValueError("revision style_id must identify a Style in this stack")
                reference = revision.reference
                if isinstance(reference, ResolvedCardReference):
                    if reference.target_card_id not in known_card_ids:
                        raise ValueError(
                            "resolved card references must identify a card in this stack"
                        )
                    if reference.target_card_id == card.id:
                        raise ValueError("a card revision cannot reference its own card")
                if revision.hotspot_set is None:
                    continue
                for interaction in revision.hotspot_set.interactions:
                    referenced_key_ids = (
                        *interaction.conditions.requires,
                        *interaction.conditions.forbids,
                        *interaction.key_changes.remove,
                        *interaction.key_changes.grant,
                    )
                    if any(key_id not in known_key_ids for key_id in referenced_key_ids):
                        raise ValueError(
                            "hotspot key references must identify Keys in this stack"
                        )
                    if interaction.action is not None:
                        target = interaction.action.target
                        if (
                            isinstance(target, ResolvedCardReference)
                            and target.target_card_id not in known_card_ids
                        ):
                            raise ValueError(
                                "resolved card references must identify a card in this stack"
                            )
                    derived_label = _automatic_interaction_label(
                        interaction,
                        card_names=card_names,
                        key_names=key_names_by_id,
                    )
                    if interaction.label != derived_label:
                        object.__setattr__(
                            interaction,
                            "label",
                            derived_label,
                        )
        return self

    def style_by_id(self, style_id: UUID | None) -> StyleDefinition | None:
        """Return a selected Style definition, or No Style."""
        if style_id is None:
            return None
        return next(style for style in self.styles if style.id == style_id)

    def key_by_id(self, key_id: UUID) -> KeyDefinition:
        """Return one validated stack-owned Key definition."""
        return next(key for key in self.keys if key.id == key_id)
