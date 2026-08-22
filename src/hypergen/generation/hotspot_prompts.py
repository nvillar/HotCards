"""Strict geometry-only hotspot remapping contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, ValidationError, model_validator

from hypergen.domain.geometry import (
    DEFAULT_MODEL_COORDINATE_EXTENT,
    model_point_to_document,
)
from hypergen.domain.models import (
    DomainModel,
    HotspotRemapProvenance,
    NonEmptyString,
    Polygon,
)
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaCallResult, OllamaRuntime
from hypergen.generation.structured_output import structured_json_content

HOTSPOT_REMAP_PROMPT_VERSION = "hotspot-remap-prompt-v1"
HOTSPOT_REMAP_SCHEMA_VERSION = "hotspot-remap-schema-v1"
MAX_INTERACTIONS_PER_CALL = 2
MAX_COMPONENTS_PER_INTERACTION = 2
MAX_POINTS_PER_COMPONENT = 12
HotspotToken = Annotated[str, StringConstraints(pattern=r"^H[1-9][0-9]*$")]


class RemapHotspotInput(DomainModel):
    """One existing hotspot identified by a request-local token."""

    token: HotspotToken
    label: NonEmptyString


class HotspotRemapRequest(DomainModel):
    """Complete geometry-only remap request."""

    image_path: Path
    hotspots: tuple[RemapHotspotInput, ...] = Field(min_length=1)
    coordinate_extent: int = Field(default=DEFAULT_MODEL_COORDINATE_EXTENT, gt=0)
    prompt_version: Literal[HOTSPOT_REMAP_PROMPT_VERSION] = (
        HOTSPOT_REMAP_PROMPT_VERSION
    )
    schema_version: Literal[HOTSPOT_REMAP_SCHEMA_VERSION] = (
        HOTSPOT_REMAP_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def require_unique_tokens(self) -> HotspotRemapRequest:
        tokens = [hotspot.token for hotspot in self.hotspots]
        if len(tokens) != len(set(tokens)):
            raise ValueError("hotspot remap tokens must be unique")
        return self


class ModelPoint(DomainModel):
    x: int
    y: int


class ModelPolygon(DomainModel):
    points: tuple[ModelPoint, ...] = Field(
        min_length=3,
        max_length=MAX_POINTS_PER_COMPONENT,
    )


class ModelMappedHotspot(DomainModel):
    token: HotspotToken
    polygons: tuple[ModelPolygon, ...] = Field(
        min_length=1,
        max_length=MAX_COMPONENTS_PER_INTERACTION,
    )


class ModelUnlocatedHotspot(DomainModel):
    token: HotspotToken
    reason: NonEmptyString


class HotspotRemapModelOutput(DomainModel):
    mapped: tuple[ModelMappedHotspot, ...] = Field(default_factory=tuple)
    unlocated: tuple[ModelUnlocatedHotspot, ...] = Field(default_factory=tuple)


class UnlocatedHotspot(DomainModel):
    token: HotspotToken
    reason: NonEmptyString


class HotspotRemapResult(DomainModel):
    """Validated geometry keyed to existing request-local hotspot tokens."""

    polygons_by_token: dict[str, tuple[Polygon, ...]]
    unlocated: tuple[UnlocatedHotspot, ...]
    warnings: tuple[str, ...]
    raw_responses: tuple[NonEmptyString, ...]
    provenance: HotspotRemapProvenance
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reasons: tuple[str, ...] = Field(default_factory=tuple)


def build_hotspot_remap_schema(
    hotspots: tuple[RemapHotspotInput, ...],
) -> dict[str, Any]:
    """Constrain every model token to the current request batch."""
    schema = HotspotRemapModelOutput.model_json_schema()
    tokens = [hotspot.token for hotspot in hotspots]
    for definition in ("ModelMappedHotspot", "ModelUnlocatedHotspot"):
        schema["$defs"][definition]["properties"]["token"] = {
            "title": "Hotspot Token",
            "type": "string",
            "enum": tokens,
        }
    return schema


def build_hotspot_remap_prompt(
    request: HotspotRemapRequest,
    hotspots: tuple[RemapHotspotInput, ...],
) -> str:
    """Build a versioned prompt that cannot redefine hotspot semantics."""
    payload = {
        "hotspots": [
            {"token": hotspot.token, "label": hotspot.label}
            for hotspot in hotspots
        ],
        "coordinate_extent": request.coordinate_extent,
    }
    response_shape = {
        "mapped": [
            {
                "token": hotspots[0].token,
                "polygons": [
                    {
                        "points": [
                            {"x": 100, "y": 100},
                            {"x": 200, "y": 100},
                            {"x": 150, "y": 200},
                        ]
                    }
                ],
            }
        ],
        "unlocated": [
            {
                "token": hotspot.token,
                "reason": "the labeled subject is not visible",
            }
            for hotspot in hotspots[1:]
        ],
    }
    return f"""\
Locate each supplied existing hotspot subject in the current card image.

Return JSON matching the supplied schema.
- Return every supplied token exactly once, in either mapped or unlocated.
- Never invent a token or hotspot.
- Do not return labels, destinations, actions, ordering, IDs, or prose.
- Use exactly the response keys and nesting shown here:
{json.dumps(response_shape, ensure_ascii=False, indent=2)}
- The geometry key must be "polygons", never "polygon_components".
- Each polygon point must be an object with integer x and y fields.
- Use coordinates from 0 through {request.coordinate_extent}.
- Use at most {MAX_COMPONENTS_PER_INTERACTION} polygon components and
  {MAX_POINTS_PER_COMPONENT} points per component.
- Keep polygons simple and editable. Do not create holes.
- If a subject cannot be located, return it in unlocated instead of inventing geometry.

Prompt contract: {request.prompt_version}
Schema contract: {request.schema_version}
Request:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _normalize_batch(
    request: HotspotRemapRequest,
    hotspots: tuple[RemapHotspotInput, ...],
    call: OllamaCallResult,
) -> tuple[dict[str, tuple[Polygon, ...]], list[UnlocatedHotspot], list[str]]:
    try:
        output = HotspotRemapModelOutput.model_validate_json(
            structured_json_content(call.content)
        )
    except ValidationError as error:
        raise ModelResponseError(
            f"Ollama returned an invalid remap response for "
            f"{request.schema_version}: {error}",
            raw_response=call.content,
        ) from error
    expected_tokens = {hotspot.token for hotspot in hotspots}
    returned_tokens = [
        *(item.token for item in output.mapped),
        *(item.token for item in output.unlocated),
    ]
    if len(returned_tokens) != len(set(returned_tokens)):
        raise ModelResponseError(
            "Ollama returned a hotspot token more than once",
            raw_response=call.content,
        )
    unknown = set(returned_tokens) - expected_tokens
    if unknown:
        raise ModelResponseError(
            f"Ollama returned unknown hotspot tokens: {sorted(unknown)}",
            raw_response=call.content,
        )

    polygons_by_token: dict[str, tuple[Polygon, ...]] = {}
    unlocated = [
        UnlocatedHotspot(token=item.token, reason=item.reason)
        for item in output.unlocated
    ]
    warnings: list[str] = []
    for mapped in output.mapped:
        if any(
            point.x < 0
            or point.x > request.coordinate_extent
            or point.y < 0
            or point.y > request.coordinate_extent
            for polygon in mapped.polygons
            for point in polygon.points
        ):
            warnings.append(
                f"{mapped.token} coordinates were clamped to the canvas"
            )
        try:
            polygons_by_token[mapped.token] = tuple(
                Polygon(
                    points=tuple(
                        model_point_to_document(
                            point.x,
                            point.y,
                            extent=request.coordinate_extent,
                        )
                        for point in polygon.points
                    )
                )
                for polygon in mapped.polygons
            )
        except ValidationError:
            unlocated.append(
                UnlocatedHotspot(
                    token=mapped.token,
                    reason="the returned polygon geometry was invalid",
                )
            )
            warnings.append(
                f"{mapped.token} kept its existing geometry because the "
                "returned polygon was invalid"
            )
    for hotspot in hotspots:
        if (
            hotspot.token not in polygons_by_token
            and all(item.token != hotspot.token for item in unlocated)
        ):
            unlocated.append(
                UnlocatedHotspot(
                    token=hotspot.token,
                    reason="the model omitted this hotspot",
                )
            )
    return polygons_by_token, unlocated, warnings


class OllamaHotspotRemapper:
    """Remap all supplied hotspots in bounded model calls."""

    def __init__(self, runtime: OllamaRuntime) -> None:
        self._runtime = runtime

    def remap(self, request: HotspotRemapRequest) -> HotspotRemapResult:
        if not request.image_path.is_file():
            raise ModelResponseError(
                f"hotspot remap image does not exist: {request.image_path}"
            )
        polygons_by_token: dict[str, tuple[Polygon, ...]] = {}
        unlocated: list[UnlocatedHotspot] = []
        warnings: list[str] = []
        raw_responses: list[str] = []
        total_duration = 0.0
        prompt_eval_count = 0
        eval_count = 0
        has_prompt_count = True
        has_eval_count = True
        done_reasons: list[str] = []
        for start in range(0, len(request.hotspots), MAX_INTERACTIONS_PER_CALL):
            batch = request.hotspots[start : start + MAX_INTERACTIONS_PER_CALL]
            call = self._runtime.chat_structured(
                prompt=build_hotspot_remap_prompt(request, batch),
                schema=build_hotspot_remap_schema(batch),
                image_path=request.image_path,
            )
            mapped, missing, batch_warnings = _normalize_batch(
                request,
                batch,
                call,
            )
            polygons_by_token.update(mapped)
            unlocated.extend(missing)
            warnings.extend(batch_warnings)
            raw_responses.append(call.content)
            total_duration += call.elapsed_seconds
            if call.prompt_eval_count is None:
                has_prompt_count = False
            else:
                prompt_eval_count += call.prompt_eval_count
            if call.eval_count is None:
                has_eval_count = False
            else:
                eval_count += call.eval_count
            if call.done_reason is not None:
                done_reasons.append(call.done_reason)
        return HotspotRemapResult(
            polygons_by_token=polygons_by_token,
            unlocated=tuple(unlocated),
            warnings=tuple(warnings),
            raw_responses=tuple(raw_responses),
            provenance=HotspotRemapProvenance(
                model_identifier=self._runtime.settings.model,
                prompt_version=request.prompt_version,
                schema_version=request.schema_version,
                generated_at=datetime.now(UTC),
                duration_seconds=total_duration,
            ),
            prompt_eval_count=prompt_eval_count if has_prompt_count else None,
            eval_count=eval_count if has_eval_count else None,
            done_reasons=tuple(done_reasons),
        )


__all__ = [
    "HOTSPOT_REMAP_PROMPT_VERSION",
    "HOTSPOT_REMAP_SCHEMA_VERSION",
    "HotspotRemapRequest",
    "HotspotRemapResult",
    "HotspotToken",
    "MAX_COMPONENTS_PER_INTERACTION",
    "MAX_INTERACTIONS_PER_CALL",
    "MAX_POINTS_PER_COMPONENT",
    "OllamaHotspotRemapper",
    "RemapHotspotInput",
    "UnlocatedHotspot",
    "build_hotspot_remap_prompt",
    "build_hotspot_remap_schema",
]
