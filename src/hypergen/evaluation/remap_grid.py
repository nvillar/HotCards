"""Grid-overlay experiments for hotspot coordinate grounding."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev

from PIL import Image, ImageChops, ImageDraw
from pydantic import Field

from hypergen.domain.models import DomainModel, Polygon
from hypergen.evaluation.manifest import atomic_write_json
from hypergen.evaluation.reports import create_contact_sheet
from hypergen.generation.errors import GenerationError, ModelResponseError
from hypergen.generation.hotspot_prompts import (
    HotspotRemapRequest,
    HotspotRemapResult,
    OllamaHotspotRemapper,
    RemapHotspotInput,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.storage.stack_store import StackStore

DEFAULT_GRID_MODELS = ("qwen3.5:9b-mlx", "qwen3.8:27b-mlx")
DEFAULT_GRID_DIVISIONS = (0, 5, 10, 20)


class RemapGridExperimentSettings(DomainModel):
    """Inputs for one repeated model/grid comparison."""

    output_dir: Path
    stack_bundle: Path
    card_name: str
    revision_number: int = Field(gt=0)
    models: tuple[str, ...] = Field(default=DEFAULT_GRID_MODELS, min_length=1)
    grid_divisions: tuple[int, ...] = Field(
        default=DEFAULT_GRID_DIVISIONS,
        min_length=1,
    )
    trials: int = Field(default=3, gt=0)
    ollama_endpoint: str = "http://localhost:11434"
    ollama_timeout_seconds: float = Field(default=180.0, gt=0)
    ollama_num_predict: int = Field(default=2048, gt=0)
    ollama_context_length: int = Field(default=8192, gt=0)


@dataclass(frozen=True, slots=True)
class _ReferenceHotspot:
    token: str
    label: str
    polygons: tuple[Polygon, ...]


def default_remap_grid_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"remap-grid-{timestamp}"


def _model_artifact_id(model: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


def render_coordinate_grid(
    source_path: Path,
    output_path: Path,
    *,
    divisions: int,
    extent: int = 1000,
) -> Path:
    """Overlay a labeled normalized coordinate grid without resizing the image."""
    if divisions < 2 or divisions > 20:
        raise ValueError("grid divisions must be between 2 and 20")
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    label_every = max(1, divisions // 10)
    line_width = max(1, image.width // 512)
    for index in range(1, divisions):
        x = round(image.width * index / divisions)
        y = round(image.height * index / divisions)
        major = index % label_every == 0
        alpha = 150 if major else 80
        draw.line((x, 0, x, image.height), fill=(0, 220, 255, alpha), width=line_width)
        draw.line((0, y, image.width, y), fill=(255, 0, 180, alpha), width=line_width)
        if major:
            coordinate = round(extent * index / divisions)
            draw.text(
                (x + 2, 2),
                f"x={coordinate}",
                fill=(0, 0, 0, 255),
                stroke_width=2,
                stroke_fill=(255, 255, 255, 230),
            )
            draw.text(
                (2, y + 2),
                f"y={coordinate}",
                fill=(0, 0, 0, 255),
                stroke_width=2,
                stroke_fill=(255, 255, 255, 230),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def _polygon_mask(
    polygons: tuple[Polygon, ...],
    size: tuple[int, int],
) -> Image.Image:
    mask = Image.new("1", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(
            [
                (
                    round(point.x * (size[0] - 1)),
                    round(point.y * (size[1] - 1)),
                )
                for point in polygon.points
            ],
            fill=1,
        )
    return mask


def _polygon_iou(
    reference: tuple[Polygon, ...],
    predicted: tuple[Polygon, ...],
    size: tuple[int, int],
) -> float:
    reference_mask = _polygon_mask(reference, size)
    predicted_mask = _polygon_mask(predicted, size)
    intersection = ImageChops.logical_and(reference_mask, predicted_mask)
    union = ImageChops.logical_or(reference_mask, predicted_mask)
    intersection_count = intersection.histogram()[255]
    union_count = union.histogram()[255]
    return intersection_count / union_count if union_count else 0.0


def _bounds(polygons: tuple[Polygon, ...]) -> tuple[float, float, float, float]:
    points = [point for polygon in polygons for point in polygon.points]
    return (
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _bbox_iou(
    reference: tuple[Polygon, ...],
    predicted: tuple[Polygon, ...],
) -> float:
    left_a, top_a, right_a, bottom_a = _bounds(reference)
    left_b, top_b, right_b, bottom_b = _bounds(predicted)
    intersection_width = max(0.0, min(right_a, right_b) - max(left_a, left_b))
    intersection_height = max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))
    intersection = intersection_width * intersection_height
    area_a = (right_a - left_a) * (bottom_a - top_a)
    area_b = (right_b - left_b) * (bottom_b - top_b)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _annotate_comparison(
    source_path: Path,
    output_path: Path,
    references: tuple[_ReferenceHotspot, ...],
    predictions: dict[str, tuple[Polygon, ...]],
) -> Path:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)

    def draw_polygons(
        polygons: tuple[Polygon, ...],
        *,
        color: str,
        width: int,
    ) -> None:
        for polygon in polygons:
            points = [
                (round(point.x * image.width), round(point.y * image.height))
                for point in polygon.points
            ]
            draw.line(points + [points[0]], fill=color, width=width)

    for reference in references:
        draw_polygons(reference.polygons, color="#34c759", width=4)
        draw_polygons(predictions.get(reference.token, ()), color="#ff3b30", width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def _score_result(
    references: tuple[_ReferenceHotspot, ...],
    result: HotspotRemapResult,
    size: tuple[int, int],
) -> dict[str, object]:
    hotspot_metrics: list[dict[str, object]] = []
    for reference in references:
        predicted = result.polygons_by_token.get(reference.token, ())
        hotspot_metrics.append(
            {
                "token": reference.token,
                "label": reference.label,
                "mapped": bool(predicted),
                "polygon_iou": (
                    _polygon_iou(reference.polygons, predicted, size)
                    if predicted
                    else 0.0
                ),
                "bbox_iou": (
                    _bbox_iou(reference.polygons, predicted)
                    if predicted
                    else 0.0
                ),
            }
        )
    return {
        "mapping_rate": sum(bool(item["mapped"]) for item in hotspot_metrics)
        / len(hotspot_metrics),
        "mean_polygon_iou": mean(
            float(item["polygon_iou"]) for item in hotspot_metrics
        ),
        "mean_bbox_iou": mean(
            float(item["bbox_iou"]) for item in hotspot_metrics
        ),
        "hotspots": hotspot_metrics,
        "warnings": list(result.warnings),
        "duration_seconds": result.provenance.duration_seconds,
        "prompt_tokens": result.prompt_eval_count,
        "output_tokens": result.eval_count,
    }


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    combinations = sorted(
        {(str(row["model"]), int(row["grid_divisions"])) for row in rows}
    )
    for model, divisions in combinations:
        trials = [
            row
            for row in rows
            if row["model"] == model
            and row["grid_divisions"] == divisions
        ]
        successes = [row for row in trials if row["status"] == "success"]
        polygon_values = [
            float(row["mean_polygon_iou"]) if row["status"] == "success" else 0.0
            for row in trials
        ]
        successful_polygon_values = [
            float(row["mean_polygon_iou"]) for row in successes
        ]
        bbox_values = [
            float(row["mean_bbox_iou"]) if row["status"] == "success" else 0.0
            for row in trials
        ]
        mapping_values = [
            float(row["mapping_rate"]) if row["status"] == "success" else 0.0
            for row in trials
        ]
        summaries.append(
            {
                "model": model,
                "grid_divisions": divisions,
                "successful_trials": len(successes),
                "trial_count": len(trials),
                "success_rate": len(successes) / len(trials),
                "mean_polygon_iou": mean(polygon_values) if polygon_values else 0.0,
                "min_polygon_iou": min(polygon_values) if polygon_values else 0.0,
                "successful_mean_polygon_iou": (
                    mean(successful_polygon_values)
                    if successful_polygon_values
                    else 0.0
                ),
                "polygon_iou_stddev": (
                    pstdev(polygon_values) if len(polygon_values) > 1 else 0.0
                ),
                "mean_bbox_iou": mean(bbox_values) if bbox_values else 0.0,
                "mean_mapping_rate": mean(mapping_values) if mapping_values else 0.0,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -float(item["mean_polygon_iou"]),
            -float(item["min_polygon_iou"]),
        ),
    )


def _write_outputs(
    settings: RemapGridExperimentSettings,
    source: dict[str, object],
    rows: list[dict[str, object]],
    contact_entries: list[tuple[Path, str]],
) -> Path:
    summaries = _summaries(rows)
    result = {
        "experiment_version": "remap-grid-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "settings": {
            **settings.model_dump(mode="json"),
            "stack_bundle": settings.stack_bundle.name,
        },
        "ollama_defaults": asdict(
            OllamaSettings(endpoint=settings.ollama_endpoint)
        ),
        "summaries": summaries,
        "trials": rows,
    }
    result_path = settings.output_dir / "remap-grid-results.json"
    atomic_write_json(result_path, result)
    with (settings.output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    create_contact_sheet(
        contact_entries,
        settings.output_dir / "contact-sheet.png",
        columns=3,
    )
    return result_path


def run_remap_grid_experiment(
    settings: RemapGridExperimentSettings,
) -> Path:
    """Run repeated plain/grid Remap calls against one authored revision."""
    if any(value != 0 and not 2 <= value <= 20 for value in settings.grid_divisions):
        raise ValueError("grid divisions must be zero or between 2 and 20")
    store = StackStore(settings.stack_bundle)
    stack = store.load()
    card = next(
        (candidate for candidate in stack.cards if candidate.name == settings.card_name),
        None,
    )
    if card is None:
        raise ValueError(f"card {settings.card_name!r} was not found")
    if settings.revision_number > len(card.revisions):
        raise ValueError(
            f"revision {settings.revision_number} does not exist on card "
            f"{settings.card_name!r}"
        )
    revision = card.revisions[settings.revision_number - 1]
    if revision.background is None:
        raise ValueError("the selected revision has no background")
    if revision.hotspot_set is None or not revision.hotspot_set.interactions:
        raise ValueError("the selected revision has no hotspot reference geometry")
    if any(not interaction.polygons for interaction in revision.hotspot_set.interactions):
        raise ValueError("every reference hotspot must contain geometry")

    source_path = store.asset_path(revision.background.image_path)
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    source_copy = settings.output_dir / "source.png"
    shutil.copy2(source_path, source_copy)
    with Image.open(source_copy) as image:
        image_size = image.size
    references = tuple(
        _ReferenceHotspot(
            token=f"H{index}",
            label=interaction.label,
            polygons=interaction.polygons,
        )
        for index, interaction in enumerate(
            revision.hotspot_set.interactions,
            start=1,
        )
    )
    source_metadata: dict[str, object] = {
        "stack_id": str(stack.id),
        "stack_name": stack.name,
        "stack_document_sha256": hashlib.sha256(
            stack.model_dump_json().encode("utf-8")
        ).hexdigest(),
        "card_id": str(card.id),
        "card_name": card.name,
        "revision_id": str(revision.id),
        "revision_number": settings.revision_number,
        "image_sha256": hashlib.sha256(source_copy.read_bytes()).hexdigest(),
        "image_size": list(image_size),
        "hotspots": [
            {
                "token": item.token,
                "label": item.label,
                "polygons": [
                    polygon.model_dump(mode="json")
                    for polygon in item.polygons
                ],
            }
            for item in references
        ],
    }
    grid_paths: dict[int, Path] = {0: source_copy}
    for divisions in settings.grid_divisions:
        if divisions:
            grid_paths[divisions] = render_coordinate_grid(
                source_copy,
                settings.output_dir / "grids" / f"grid-{divisions}.png",
                divisions=divisions,
            )

    rows: list[dict[str, object]] = []
    contact_entries: list[tuple[Path, str]] = []
    for model in settings.models:
        model_artifact_id = _model_artifact_id(model)
        runtime = OllamaRuntime(
            OllamaSettings(
                endpoint=settings.ollama_endpoint,
                model=model,
                request_timeout_seconds=settings.ollama_timeout_seconds,
                num_predict=settings.ollama_num_predict,
                context_length=settings.ollama_context_length,
            )
        )
        try:
            runtime.require_model(capabilities=frozenset({"vision"}))
        except GenerationError as error:
            rows.extend(
                {
                    "model": model,
                    "grid_divisions": divisions,
                    "trial": trial,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "stage": "model_setup",
                }
                for divisions in settings.grid_divisions
                for trial in range(1, settings.trials + 1)
            )
            _write_outputs(
                settings,
                source_metadata,
                rows,
                contact_entries,
            )
            continue
        remapper = OllamaHotspotRemapper(runtime)
        for divisions in settings.grid_divisions:
            for trial in range(1, settings.trials + 1):
                request = HotspotRemapRequest(
                    image_path=grid_paths[divisions],
                    hotspots=tuple(
                        RemapHotspotInput(
                            token=reference.token,
                            label=reference.label,
                        )
                        for reference in references
                    ),
                    coordinate_grid_divisions=divisions or None,
                )
                row: dict[str, object] = {
                    "model": model,
                    "grid_divisions": divisions,
                    "trial": trial,
                }
                try:
                    result = remapper.remap(request)
                except GenerationError as error:
                    if (
                        isinstance(error, ModelResponseError)
                        and error.raw_response is not None
                    ):
                        raw_dir = (
                            settings.output_dir
                            / "raw"
                            / model_artifact_id
                        )
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        (
                            raw_dir
                            / f"grid-{divisions}-trial-{trial}-failed.txt"
                        ).write_text(error.raw_response, encoding="utf-8")
                    row.update(
                        {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                else:
                    raw_dir = settings.output_dir / "raw" / model_artifact_id
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    (raw_dir / f"grid-{divisions}-trial-{trial}.json").write_text(
                        json.dumps(list(result.raw_responses), indent=2),
                        encoding="utf-8",
                    )
                    row.update(
                        {
                            "status": "success",
                            **_score_result(references, result, image_size),
                        }
                    )
                    overlay_path = _annotate_comparison(
                        source_copy,
                        settings.output_dir
                        / "overlays"
                        / model_artifact_id
                        / f"grid-{divisions}-trial-{trial}.png",
                        references,
                        result.polygons_by_token,
                    )
                    contact_entries.append(
                        (
                            overlay_path,
                            f"{model} grid={divisions} trial={trial} "
                            f"IoU={row['mean_polygon_iou']:.3f}",
                        )
                    )
                rows.append(row)
                _write_outputs(
                    settings,
                    source_metadata,
                    rows,
                    contact_entries,
                )

    return _write_outputs(
        settings,
        source_metadata,
        rows,
        contact_entries,
    )


__all__ = [
    "DEFAULT_GRID_DIVISIONS",
    "DEFAULT_GRID_MODELS",
    "RemapGridExperimentSettings",
    "default_remap_grid_output_dir",
    "render_coordinate_grid",
    "run_remap_grid_experiment",
]
