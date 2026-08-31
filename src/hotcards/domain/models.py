"""Serialized domain models for HotCards stack documents."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    output_dimensions,
    validate_aligned_output_dimensions,
    validate_exact_output_dimensions,
)

CURRENT_SCHEMA_VERSION = 11

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFiniteFloat = Annotated[FiniteFloat, Field(ge=0.0)]


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
        from hotcards.domain.geometry import validate_polygon

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
            raise ValueError("each key can appear in Has only once")
        if len(self.forbids) != len(set(self.forbids)):
            raise ValueError("each key can appear in Lacks only once")
        if set(self.requires) & set(self.forbids):
            raise ValueError("the runner cannot both have and lack the same key")
        return self


class HotspotKeyChanges(DomainModel):
    """One atomic key transition applied before optional navigation."""

    remove: tuple[UUID, ...] = Field(default_factory=tuple)
    grant: tuple[UUID, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_unambiguous_changes(self) -> HotspotKeyChanges:
        if len(self.remove) != len(set(self.remove)):
            raise ValueError("each key can appear in Lose only once")
        if len(self.grant) != len(set(self.grant)):
            raise ValueError("each key can appear in Gain only once")
        if set(self.remove) & set(self.grant):
            raise ValueError("a key cannot be both gained and lost")
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


_BUILT_IN_STYLE_IDS = {
    "hypercard": UUID("2372dddb-99e0-5e62-b740-405d0f7dab2a"),
    "cinematic-film": UUID("a23ac4cf-0358-500a-a4fb-1f120f1f9e47"),
    "isometric-game": UUID("7baf1057-786a-587e-a52f-c63b513519c6"),
    "pixel-art": UUID("72b678e8-49b0-50d8-af4b-735b31def4c0"),
    "watercolor-painting": UUID("5f602dd8-d459-5633-bd1a-f9b1c86fe473"),
    "color-pencil": UUID("61a9ca01-5458-5607-9940-8088955640a2"),
    "pencil-sketch": UUID("1e0aeb8e-7112-5ddc-becf-f6053ce31ffb"),
    "glazed-ceramic": UUID("a73faa84-f09b-5832-b8a7-9af17fcb7c86"),
    "graphic-novel": UUID("ea72f569-5ac1-5906-88b8-a0b7171b4d15"),
    "miniature-toy": UUID("eda5058a-d463-5fd7-82dc-f50e7a571efa"),
}


def _built_in_style_id(slug: str) -> UUID:
    return _BUILT_IN_STYLE_IDS[slug]


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


class ImageOperationSettings(DomainModel):
    """Exact shared execution settings and measured result facts."""

    model_identifier: NonEmptyString
    mflux_version: NonEmptyString
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    seed: int
    width: PositiveInt
    height: PositiveInt
    step_count: PositiveInt
    quantization: int | None = None
    guidance: FiniteFloat = Field(default=1.0, gt=0.0)
    scheduler: NonEmptyString = "flow_match_euler_discrete"
    use_kv_cache: bool = False
    generated_at: AwareDatetime
    duration_seconds: NonNegativeFiniteFloat


class ImageSourceSnapshot(DomainModel):
    """Exact source revision and background used by a derived operation."""

    card_id: UUID
    revision_id: UUID
    background_id: UUID


class EditPreserveOptions(DomainModel):
    """Six deterministic preservation controls captured for Edit."""

    subject_identity: bool = False
    pose_and_expression: bool = False
    composition_and_framing: bool = False
    background: bool = False
    lighting_and_color: bool = False
    existing_text_and_logos: bool = False


class CurrentSourceSize(DomainModel):
    """Use a source image's exact decoded dimensions for derived output."""

    mode: Literal["current"] = "current"
    width: PositiveInt
    height: PositiveInt

    @model_validator(mode="after")
    def require_aligned_dimensions(self) -> CurrentSourceSize:
        validate_aligned_output_dimensions(self.width, self.height)
        return self


class ExactOutputSize(DomainModel):
    """Use an exact author-selected size for direct Generate output."""

    mode: Literal["exact"] = "exact"
    width: PositiveInt
    height: PositiveInt

    @model_validator(mode="after")
    def require_aligned_dimensions(self) -> ExactOutputSize:
        validate_aligned_output_dimensions(self.width, self.height)
        return self


class PresetOutputSize(DomainModel):
    """Use one named long-edge tier at the stack aspect ratio."""

    mode: Literal["preset"] = "preset"
    tier: ResolutionTier


GenerateOutputSize = Annotated[
    PresetOutputSize | ExactOutputSize,
    Field(discriminator="mode"),
]
RefineOutputSize = Annotated[
    CurrentSourceSize | PresetOutputSize,
    Field(discriminator="mode"),
]
EditOutputSize = Annotated[
    CurrentSourceSize | PresetOutputSize,
    Field(discriminator="mode"),
]


def selected_output_dimensions(
    output_size: GenerateOutputSize | RefineOutputSize | EditOutputSize,
    aspect_ratio: AspectRatio,
) -> tuple[int, int]:
    """Resolve exact output pixels from one strict size selection."""
    if isinstance(output_size, (CurrentSourceSize, ExactOutputSize)):
        return output_size.width, output_size.height
    return output_dimensions(output_size.tier, aspect_ratio)


class GenerateInputs(DomainModel):
    """Exact author-controlled inputs accepted by direct Generate."""

    description: str
    references: tuple[ImageReferenceSnapshot, ...] = Field(
        default_factory=tuple,
        max_length=2,
    )
    style: StyleSnapshot | None = None
    output_size: GenerateOutputSize = PresetOutputSize(tier=ResolutionTier.MEDIUM)

    @field_validator("description")
    @classmethod
    def require_authored_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Generate requires a nonempty Description")
        return value


class AcceptedEdit(DomainModel):
    """One accepted Edit instruction retained in chronological order."""

    instruction: NonEmptyString
    preserve: EditPreserveOptions
    expanded_prompt: NonEmptyString


class RefineTransformation(StrEnum):
    """Named Refine transformations with fixed production strengths."""

    REIMAGINE = "reimagine"
    BALANCED = "balanced"
    PRESERVE = "preserve"

    @property
    def strength(self) -> float:
        """Return the fixed MFLUX img2img strength for this transformation."""
        return {
            RefineTransformation.REIMAGINE: 0.25,
            RefineTransformation.BALANCED: 0.50,
            RefineTransformation.PRESERVE: 0.75,
        }[self]


class DirectGenerateProvenance(DomainModel):
    """Provenance for current direct Description generation."""

    operation: Literal["generate"] = "generate"
    inputs: GenerateInputs
    render_prompt: NonEmptyString
    settings: ImageOperationSettings


class LegacyGenerateProvenance(DomainModel):
    """Current-schema preservation of an externally patched historical Generate."""

    operation: Literal["legacy_generate"] = "legacy_generate"
    render_prompt: NonEmptyString
    references: tuple[ImageReferenceSnapshot, ...] = Field(default_factory=tuple)
    settings: ImageOperationSettings


class RefineProvenance(DomainModel):
    """Provenance for an accepted Refine derived from one source revision."""

    operation: Literal["refine"] = "refine"
    source: ImageSourceSnapshot
    description: str
    style: StyleSnapshot | None = None
    edit_lineage: tuple[AcceptedEdit, ...] = Field(default_factory=tuple)
    render_prompt: NonEmptyString
    output_size: RefineOutputSize
    transformation: RefineTransformation
    strength: FiniteFloat = Field(ge=0.0, le=1.0)
    settings: ImageOperationSettings

    @model_validator(mode="after")
    def require_transformation_strength(self) -> RefineProvenance:
        if self.strength != self.transformation.strength:
            raise ValueError(
                f"{self.transformation.value} Refine strength must be "
                f"{self.transformation.strength:.2f}"
            )
        return self


class EditProvenance(DomainModel):
    """Provenance for an accepted Edit derived from one source revision."""

    operation: Literal["edit"] = "edit"
    source: ImageSourceSnapshot
    instruction: NonEmptyString
    preserve: EditPreserveOptions
    expanded_prompt: NonEmptyString
    output_size: EditOutputSize
    edit_lineage: tuple[AcceptedEdit, ...] = Field(min_length=1)
    prompt_token_count: PositiveInt
    prompt_token_budget: Literal[512] = 512
    settings: ImageOperationSettings

    @property
    def accepted_edit(self) -> AcceptedEdit:
        """Return the accepted Edit represented by this operation."""
        return AcceptedEdit(
            instruction=self.instruction,
            preserve=self.preserve,
            expanded_prompt=self.expanded_prompt,
        )

    @model_validator(mode="after")
    def require_current_edit_at_lineage_end(self) -> EditProvenance:
        if self.edit_lineage[-1] != self.accepted_edit:
            raise ValueError("Edit lineage must end with the accepted current Edit")
        if self.prompt_token_count > self.prompt_token_budget:
            raise ValueError("Edit prompt token count exceeds its 512-token budget")
        return self


OriginalImageProvenance = Annotated[
    DirectGenerateProvenance | LegacyGenerateProvenance | RefineProvenance | EditProvenance,
    Field(discriminator="operation"),
]


class DuplicateProvenance(DomainModel):
    """Independent copy attribution without a live source dependency."""

    operation: Literal["duplicate"] = "duplicate"
    source: ImageSourceSnapshot
    original_provenance: OriginalImageProvenance

    @property
    def settings(self) -> ImageOperationSettings:
        """Expose the copied operation settings to provenance consumers."""
        return self.original_provenance.settings


ImageProvenance = Annotated[
    DirectGenerateProvenance
    | LegacyGenerateProvenance
    | RefineProvenance
    | EditProvenance
    | DuplicateProvenance,
    Field(discriminator="operation"),
]


def original_image_provenance(
    provenance: ImageProvenance,
) -> OriginalImageProvenance:
    """Flatten duplicate attribution to the exact original image operation."""
    if isinstance(provenance, DuplicateProvenance):
        return provenance.original_provenance
    return provenance


def image_edit_lineage(provenance: ImageProvenance) -> tuple[AcceptedEdit, ...]:
    """Return the accepted Edit lineage inherited by a derived operation."""
    original = original_image_provenance(provenance)
    if isinstance(original, (RefineProvenance, EditProvenance)):
        return original.edit_lineage
    return ()


def image_operation_settings(provenance: ImageProvenance) -> ImageOperationSettings:
    """Return the exact operation settings behind an image, flattening duplicates."""
    return original_image_provenance(provenance).settings


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
    provenance: ImageProvenance
    created_at: AwareDatetime


Background = GeneratedBackground


class CardRevision(DomainModel):
    """One complete revision of a card's authored content."""

    id: UUID = Field(default_factory=uuid4)
    description: str = ""
    background: Background | None = None
    hotspot_set: HotspotSet | None = None
    references: tuple[CardReference, ...] = Field(
        default_factory=tuple,
        max_length=2,
    )
    style_id: UUID | None = None
    generate_output_size: GenerateOutputSize = PresetOutputSize(tier=ResolutionTier.MEDIUM)

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
    def provenance(self) -> ImageProvenance | None:
        """Return generated-image provenance when available."""
        return self.background.provenance if self.background is not None else None

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
            f"Lose {key_names[interaction.key_changes.remove[0]]}"
            if len(interaction.key_changes.remove) == 1
            else "Lose keys"
        )
    if interaction.key_changes.grant:
        changes.append(
            f"Gain {key_names[interaction.key_changes.grant[0]]}"
            if len(interaction.key_changes.grant) == 1
            else "Gain keys"
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
    """The authoritative portable HotCards stack document."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, strict=True)
    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyString
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
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
        from hotcards.domain.validation import require_unique_card_names

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
        revisions_by_id = {
            revision.id: revision for card in self.cards for revision in card.revisions
        }
        revision_card_ids = {
            revision.id: card.id for card in self.cards for revision in card.revisions
        }
        derived_sources: dict[UUID, UUID] = {}
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
                if isinstance(revision.generate_output_size, ExactOutputSize):
                    validate_exact_output_dimensions(
                        revision.generate_output_size.width,
                        revision.generate_output_size.height,
                        self.aspect_ratio,
                    )
                provenance = revision.provenance
                original_provenance = (
                    original_image_provenance(provenance) if provenance is not None else None
                )
                if isinstance(original_provenance, DirectGenerateProvenance):
                    if isinstance(
                        original_provenance.inputs.output_size,
                        ExactOutputSize,
                    ):
                        validate_exact_output_dimensions(
                            original_provenance.inputs.output_size.width,
                            original_provenance.inputs.output_size.height,
                            self.aspect_ratio,
                        )
                    expected_dimensions = selected_output_dimensions(
                        original_provenance.inputs.output_size,
                        self.aspect_ratio,
                    )
                    if (
                        original_provenance.settings.width,
                        original_provenance.settings.height,
                    ) != expected_dimensions:
                        raise ValueError(
                            "direct Generate dimensions must match its selected output size"
                        )
                elif isinstance(original_provenance, RefineProvenance):
                    if isinstance(
                        original_provenance.output_size,
                        CurrentSourceSize,
                    ):
                        validate_exact_output_dimensions(
                            original_provenance.output_size.width,
                            original_provenance.output_size.height,
                            self.aspect_ratio,
                        )
                    expected_dimensions = selected_output_dimensions(
                        original_provenance.output_size,
                        self.aspect_ratio,
                    )
                    if (
                        original_provenance.settings.width,
                        original_provenance.settings.height,
                    ) != expected_dimensions:
                        raise ValueError(
                            f"{original_provenance.operation.title()} dimensions must match "
                            "its selected output size"
                        )
                elif isinstance(original_provenance, EditProvenance):
                    if isinstance(
                        original_provenance.output_size,
                        CurrentSourceSize,
                    ):
                        validate_exact_output_dimensions(
                            original_provenance.output_size.width,
                            original_provenance.output_size.height,
                            self.aspect_ratio,
                        )
                    expected_dimensions = selected_output_dimensions(
                        original_provenance.output_size,
                        self.aspect_ratio,
                    )
                    if (
                        original_provenance.settings.width,
                        original_provenance.settings.height,
                    ) != expected_dimensions:
                        raise ValueError("Edit dimensions must match its selected output size")
                if isinstance(provenance, (RefineProvenance, EditProvenance)):
                    source = provenance.source
                    if source.revision_id not in revision_card_ids:
                        raise ValueError(
                            "derived image sources must identify a revision in this stack"
                        )
                    if revision_card_ids[source.revision_id] != source.card_id:
                        raise ValueError("derived image source card must own the source revision")
                    if source.revision_id == revision.id:
                        raise ValueError("a revision cannot derive from itself")
                    source_background = revisions_by_id[source.revision_id].background
                    if source_background is None or source_background.id != source.background_id:
                        raise ValueError(
                            "derived image source background must match the source revision"
                        )
                    if isinstance(
                        provenance,
                        (RefineProvenance, EditProvenance),
                    ) and isinstance(
                        provenance.output_size,
                        CurrentSourceSize,
                    ):
                        source_settings = image_operation_settings(source_background.provenance)
                        if (
                            provenance.output_size.width,
                            provenance.output_size.height,
                        ) != (
                            source_settings.width,
                            source_settings.height,
                        ):
                            raise ValueError(
                                "current-size derived output must match its source "
                                "background dimensions"
                            )
                    if isinstance(
                        provenance,
                        EditProvenance,
                    ) and isinstance(
                        provenance.output_size,
                        PresetOutputSize,
                    ):
                        source_settings = image_operation_settings(source_background.provenance)
                        if (
                            provenance.settings.width * provenance.settings.height
                            <= source_settings.width * source_settings.height
                        ):
                            raise ValueError(
                                "preset Edit output must have more pixels than "
                                "its source background"
                            )
                    derived_sources[revision.id] = source.revision_id
                resolved_reference_ids: list[UUID] = []
                for reference in revision.references:
                    if not isinstance(reference, ResolvedCardReference):
                        continue
                    if reference.target_card_id not in known_card_ids:
                        raise ValueError(
                            "resolved card references must identify a card in this stack"
                        )
                    if reference.target_card_id == card.id:
                        raise ValueError("a card revision cannot reference its own card")
                    resolved_reference_ids.append(reference.target_card_id)
                if len(resolved_reference_ids) != len(set(resolved_reference_ids)):
                    raise ValueError("a card revision cannot reference the same card twice")
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
                        raise ValueError("hotspot key references must identify Keys in this stack")
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
        for revision_id in derived_sources:
            visited: set[UUID] = set()
            current_revision_id = revision_id
            while current_revision_id in derived_sources:
                if current_revision_id in visited:
                    raise ValueError("derived image source lineage cannot contain cycles")
                visited.add(current_revision_id)
                current_revision_id = derived_sources[current_revision_id]
        for revision_id, source_revision_id in derived_sources.items():
            provenance = revisions_by_id[revision_id].provenance
            source_provenance = revisions_by_id[source_revision_id].provenance
            assert isinstance(provenance, (RefineProvenance, EditProvenance))
            assert source_provenance is not None
            source_lineage = image_edit_lineage(source_provenance)
            if isinstance(provenance, RefineProvenance):
                if provenance.edit_lineage != source_lineage:
                    raise ValueError("Refine lineage must equal its source revision lineage")
            elif provenance.edit_lineage != (
                *source_lineage,
                provenance.accepted_edit,
            ):
                raise ValueError(
                    "Edit lineage must equal its source revision lineage plus "
                    "the accepted current Edit"
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

    @property
    def canvas(self) -> CanvasSize:
        """Return a fixed non-serialized logical canvas for UI geometry."""
        width, height = {
            AspectRatio.SQUARE: (1024, 1024),
            AspectRatio.LANDSCAPE: (1024, 768),
            AspectRatio.PORTRAIT: (768, 1024),
            AspectRatio.WIDESCREEN: (1024, 576),
        }[self.aspect_ratio]
        return CanvasSize(width=width, height=height)
