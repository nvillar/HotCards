"""Controlled FLUX.2 evaluation for proposed deterministic Style suffixes."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator

from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    output_dimensions,
)
from hotcards.domain.models import (
    DomainModel,
    GenerateInputs,
    NonEmptyString,
    PositiveInt,
    PresetOutputSize,
    StyleSnapshot,
)
from hotcards.evaluation.contracts import SafeCaseId
from hotcards.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hotcards.evaluation.reports import create_contact_sheet
from hotcards.generation.errors import ImageGenerationError, ModelLoadError
from hotcards.generation.image_generation import compose_generation_prompt
from hotcards.generation.mflux_generator import (
    MfluxGenerateRequest,
    MfluxGenerateResult,
    MfluxGenerator,
)

STYLE_PRESET_EXPERIMENT_VERSION = "style-preset-experiment-v1"
STYLE_PRESET_RESULT_VERSION = "style-preset-result-v1"
STYLE_PRESET_REVIEW_VERSION = "style-preset-review-v1"
DEFAULT_STYLE_PRESET_EXPERIMENT = Path("evals/cases/styles/presets.json")
DEFAULT_STYLE_PRESET_SEEDS = (42,)


class StylePresetCondition(DomainModel):
    """One proposed Style suffix or the unstyled control."""

    style_id: SafeCaseId
    name: NonEmptyString
    prompt_text: NonEmptyString | None = None
    expected_characteristics: tuple[NonEmptyString, ...] = Field(default_factory=tuple)


class StylePresetScene(DomainModel):
    """One style-neutral scene rendered under every condition."""

    case_id: SafeCaseId
    title: NonEmptyString
    description: NonEmptyString
    required_visual_elements: tuple[NonEmptyString, ...] = Field(min_length=1)
    composition_requirements: tuple[NonEmptyString, ...] = Field(min_length=1)


class StylePresetExperiment(DomainModel):
    """Tracked matrix of scenes and proposed Style suffixes."""

    experiment_version: Literal["style-preset-experiment-v1"] = STYLE_PRESET_EXPERIMENT_VERSION
    experiment_id: SafeCaseId
    title: NonEmptyString
    description: NonEmptyString
    styles: tuple[StylePresetCondition, ...] = Field(min_length=2)
    scenes: tuple[StylePresetScene, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _validate_matrix(self) -> StylePresetExperiment:
        style_ids = [style.style_id for style in self.styles]
        if len(style_ids) != len(set(style_ids)):
            raise ValueError("Style preset IDs must be unique")
        style_names = [style.name.casefold() for style in self.styles]
        if len(style_names) != len(set(style_names)):
            raise ValueError("Style preset names must be unique")
        scene_ids = [scene.case_id for scene in self.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("Style preset scene IDs must be unique")
        controls = [style for style in self.styles if style.prompt_text is None]
        if len(controls) != 1 or controls[0].style_id != "no-style":
            raise ValueError("Style preset experiment requires one 'no-style' control")
        if self.styles[0].style_id != "no-style":
            raise ValueError("the 'no-style' control must be the first condition")
        if controls[0].expected_characteristics:
            raise ValueError("the 'no-style' control must not define Style characteristics")
        if any(
            not style.expected_characteristics
            for style in self.styles
            if style.prompt_text is not None
        ):
            raise ValueError("each Style preset requires expected characteristics")
        return self


class StylePresetSettings(DomainModel):
    """Controlled MFLUX settings for one Style matrix run."""

    output_dir: Path
    experiment_path: Path = DEFAULT_STYLE_PRESET_EXPERIMENT
    model_identifier: NonEmptyString = "flux2-klein-9b"
    seeds: tuple[int, ...] = Field(default=DEFAULT_STYLE_PRESET_SEEDS, min_length=1)
    tier: ResolutionTier = ResolutionTier.FULL
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
    step_count: PositiveInt = 4
    quantization: int | None = None

    @model_validator(mode="after")
    def _unique_seeds(self) -> StylePresetSettings:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Style preset seeds must be unique")
        return self


class ImageGeneratorProtocol(Protocol):
    """Production MFLUX surface consumed by the Style runner."""

    def generate(self, request: MfluxGenerateRequest) -> MfluxGenerateResult: ...


ImageGeneratorFactory = Callable[[], ImageGeneratorProtocol]


def default_style_preset_output_dir() -> Path:
    """Return a unique immutable output directory for one Style run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"style-presets-{timestamp}"


def load_style_preset_experiment(
    experiment_path: Path = DEFAULT_STYLE_PRESET_EXPERIMENT,
) -> StylePresetExperiment:
    """Load and strictly validate the tracked Style matrix."""
    return StylePresetExperiment.model_validate_json(experiment_path.read_text(encoding="utf-8"))


def compose_style_preset_prompt(description: str, prompt_text: str | None) -> str:
    """Append one Style treatment without changing the authored Description."""
    return compose_generation_prompt(
        GenerateInputs(
            description=description,
            style=(
                StyleSnapshot(
                    style_id=uuid5(
                        NAMESPACE_URL,
                        "https://hotcards.app/evaluation/style-preset",
                    ),
                    name="Evaluation Style",
                    prompt_text=prompt_text,
                )
                if prompt_text is not None
                else None
            ),
        )
    )


def _generation_record(
    generated: MfluxGenerateResult,
    *,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "status": "success",
        "artifact_path": generated.output_path.relative_to(output_dir).as_posix(),
        "load_duration_seconds": generated.load_duration_seconds,
        "inference_duration_seconds": generated.generation_duration_seconds,
        "serialization_duration_seconds": generated.serialization_duration_seconds,
        "total_duration_seconds": generated.provenance.settings.duration_seconds,
        "metadata": generated.provenance.model_dump(mode="json"),
    }


def _failure_record(error: BaseException) -> dict[str, object]:
    return {
        "status": "failed",
        "failure": {
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        },
    }


def _scorecard_criteria(
    scene: StylePresetScene,
    style: StylePresetCondition,
) -> list[dict[str, object]]:
    if style.prompt_text is None:
        style_description = (
            "The control image is coherent and does not exhibit a strong unintended "
            "preset treatment."
        )
    else:
        characteristics = "; ".join(style.expected_characteristics)
        style_description = (
            f"The image is recognizably {style.name} and exhibits: {characteristics}."
        )
    return [
        {
            "criterion_id": "style-fidelity",
            "description": style_description,
            "score": None,
            "notes": None,
        },
        {
            "criterion_id": "subject-preservation",
            "description": (
                "All required subjects remain clearly recognizable: "
                + "; ".join(scene.required_visual_elements)
                + "."
            ),
            "score": None,
            "notes": None,
        },
        {
            "criterion_id": "composition-preservation",
            "description": (
                "The authored spatial relationships remain clear: "
                + "; ".join(scene.composition_requirements)
                + "."
            ),
            "score": None,
            "notes": None,
        },
        {
            "criterion_id": "artifact-control",
            "description": (
                "The image contains no unrequested visible text, UI chrome, panel "
                "layout, malformed geometry, or duplicated major subjects."
            ),
            "score": None,
            "notes": None,
        },
    ]


def _write_review_html(
    review_path: Path,
    *,
    experiment: StylePresetExperiment,
    scene_sheets: list[dict[str, str | None]],
    style_sheets: list[dict[str, str | None]],
    full_resolution_outputs: list[dict[str, str]],
) -> None:
    def sections(items: list[dict[str, str | None]]) -> str:
        rendered = []
        for item in items:
            path = item["path"]
            content = (
                f'<a href="{html.escape(quote(path))}">'
                f'<img src="{html.escape(quote(path))}" '
                f'alt="{html.escape(str(item["title"]))}"></a>'
                if path is not None
                else "<p>No successful outputs were available for this group.</p>"
            )
            rendered.append(
                f"<section><h2>{html.escape(str(item['title']))}</h2>{content}</section>"
            )
        return "".join(rendered)

    gallery = "".join(
        "<figure>"
        f'<a href="{html.escape(quote(item["path"]))}">'
        f'<img src="{html.escape(quote(item["path"]))}" '
        f'alt="{html.escape(item["title"])}"></a>'
        f"<figcaption>{html.escape(item['title'])}</figcaption>"
        "</figure>"
        for item in full_resolution_outputs
    )

    review_path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(experiment.title)}</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem;color:#18202a;max-width:1500px}}
section{{border-top:2px solid #ccd3dc;margin-top:2rem;padding-top:1rem}}
img{{max-width:100%;height:auto;border:1px solid #ccd3dc}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1rem}}
.gallery figure{{margin:0}} .gallery img{{width:100%;height:auto}}
</style></head><body>
<h1>{html.escape(experiment.title)}</h1>
<p>{html.escape(experiment.description)}</p>
<p>Score every output for Style fidelity, subject preservation, composition
preservation, and artifact control. Then compare each Style across scenes for
consistency and content leakage.</p>
<h1>Scene comparisons</h1>
{sections(scene_sheets)}
<h1>Cross-scene Style comparisons</h1>
{sections(style_sheets)}
<h1>Full-resolution outputs</h1>
<p>Open any image to inspect pixel edges, grain, paper texture, hatching, and
other details lost in the contact-sheet thumbnails.</p>
<div class="gallery">{gallery}</div>
</body></html>""",
        encoding="utf-8",
    )


def _write_result(output_dir: Path, result: Mapping[str, object]) -> Path:
    result_path = output_dir / "style-preset-results.json"
    atomic_write_json(result_path, result)
    return result_path


def _execute_style_preset_evaluation(
    settings: StylePresetSettings,
    *,
    experiment: StylePresetExperiment,
    generator: ImageGeneratorProtocol,
    lifecycle: RunLifecycle,
) -> Path:
    outputs_dir = settings.output_dir / "outputs"
    review_dir = settings.output_dir / "review"
    outputs_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    result: dict[str, object] = {
        "result_version": STYLE_PRESET_RESULT_VERSION,
        "suite": "style-presets",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "experiment": experiment.model_dump(mode="json"),
        "results": records,
        "warnings": [],
    }
    _write_result(settings.output_dir, result)
    width, height = output_dimensions(
        settings.tier,
        settings.aspect_ratio,
    )

    for scene in experiment.scenes:
        for seed in settings.seeds:
            for style in experiment.styles:
                render_prompt = compose_style_preset_prompt(
                    scene.description,
                    style.prompt_text,
                )
                style_snapshot = (
                    StyleSnapshot(
                        style_id=uuid5(
                            NAMESPACE_URL,
                            f"https://hotcards.app/evaluation/styles/{style.style_id}",
                        ),
                        name=style.name,
                        prompt_text=style.prompt_text,
                    )
                    if style.prompt_text is not None
                    else None
                )
                stage = f"render:{scene.case_id}:seed-{seed}:{style.style_id}"
                lifecycle.set_stage(stage)
                output_path = outputs_dir / scene.case_id / f"seed-{seed}" / f"{style.style_id}.png"
                request = MfluxGenerateRequest(
                    inputs=GenerateInputs(
                        description=scene.description,
                        style=style_snapshot,
                        output_size=PresetOutputSize(tier=settings.tier),
                    ),
                    render_prompt=render_prompt,
                    output_path=output_path,
                    model_identifier=settings.model_identifier,
                    seed=seed,
                    aspect_ratio=settings.aspect_ratio,
                    width=width,
                    height=height,
                    step_count=settings.step_count,
                    quantization=settings.quantization,
                )
                try:
                    generated = generator.generate(request)
                    generation = _generation_record(
                        generated,
                        output_dir=settings.output_dir,
                    )
                    lifecycle.complete_stage(stage)
                except ModelLoadError:
                    raise
                except Exception as error:
                    generation = _failure_record(error)
                records.append(
                    {
                        "case_id": scene.case_id,
                        "seed": seed,
                        "style_id": style.style_id,
                        "style_name": style.name,
                        "render_prompt": render_prompt,
                        "generation": generation,
                    }
                )
                _write_result(settings.output_dir, result)

    records_by_key = {
        (str(record["case_id"]), int(record["seed"]), str(record["style_id"])): record
        for record in records
    }
    scene_sheets: list[dict[str, str | None]] = []
    for scene in experiment.scenes:
        for seed in settings.seeds:
            entries = []
            for style in experiment.styles:
                generation = records_by_key[(scene.case_id, seed, style.style_id)]["generation"]
                assert isinstance(generation, dict)
                if artifact_path := generation.get("artifact_path"):
                    entries.append((settings.output_dir / str(artifact_path), style.name))
            sheet_path = review_dir / f"scene-{scene.case_id}-seed-{seed}.png"
            created_sheet = create_contact_sheet(
                entries,
                sheet_path,
                cell_size=(400, 300),
                columns=3,
            )
            scene_sheets.append(
                {
                    "title": f"{scene.title} - seed {seed}",
                    "path": (
                        created_sheet.relative_to(review_dir).as_posix()
                        if created_sheet is not None
                        else None
                    ),
                }
            )

    style_sheets: list[dict[str, str | None]] = []
    for style in experiment.styles:
        entries = []
        for scene in experiment.scenes:
            for seed in settings.seeds:
                generation = records_by_key[(scene.case_id, seed, style.style_id)]["generation"]
                assert isinstance(generation, dict)
                if artifact_path := generation.get("artifact_path"):
                    entries.append(
                        (
                            settings.output_dir / str(artifact_path),
                            f"{scene.title} - seed {seed}",
                        )
                    )
        sheet_path = review_dir / f"style-{style.style_id}.png"
        created_sheet = create_contact_sheet(
            entries,
            sheet_path,
            cell_size=(520, 390),
            columns=2,
        )
        style_sheets.append(
            {
                "title": style.name,
                "path": (
                    created_sheet.relative_to(review_dir).as_posix()
                    if created_sheet is not None
                    else None
                ),
            }
        )

    scorecard_outputs = []
    for scene in experiment.scenes:
        for seed in settings.seeds:
            for style in experiment.styles:
                record = records_by_key[(scene.case_id, seed, style.style_id)]
                generation = record["generation"]
                assert isinstance(generation, dict)
                scorecard_outputs.append(
                    {
                        "case_id": scene.case_id,
                        "seed": seed,
                        "style_id": style.style_id,
                        "style_name": style.name,
                        "status": generation["status"],
                        "artifact_path": generation.get("artifact_path"),
                        "criteria": _scorecard_criteria(scene, style),
                        "overall_notes": None,
                    }
                )
    scorecard = {
        "review_version": STYLE_PRESET_REVIEW_VERSION,
        "experiment_id": experiment.experiment_id,
        "reviewer": None,
        "scoring_values": ["pass", "fail", "uncertain"],
        "instructions": (
            "Score each criterion independently, then compare each Style across "
            "scenes for consistency and content leakage."
        ),
        "outputs": scorecard_outputs,
        "cross_scene_style_notes": {style.style_id: None for style in experiment.styles},
    }
    atomic_write_json(review_dir / "scorecard.json", scorecard)
    full_resolution_outputs = [
        {
            "title": (f"{row['style_name']} - {row['case_id']} - seed {row['seed']}"),
            "path": (Path("..") / str(row["generation"]["artifact_path"])).as_posix(),
        }
        for row in records
        if row["generation"].get("artifact_path")  # type: ignore[union-attr]
    ]
    _write_review_html(
        review_dir / "index.html",
        experiment=experiment,
        scene_sheets=scene_sheets,
        style_sheets=style_sheets,
        full_resolution_outputs=full_resolution_outputs,
    )
    lifecycle.complete_stage("review-artifacts")

    failures = [
        record
        for record in records
        if record["generation"]["status"] == "failed"  # type: ignore[index]
    ]
    successes = len(records) - len(failures)
    if successes == 0:
        result["status"] = "failed"
    else:
        result["status"] = "completed_with_failures" if failures else "success"
    result["review_path"] = "review/index.html"
    result["scorecard_path"] = "review/scorecard.json"
    return _write_result(settings.output_dir, result)


def run_style_preset_evaluation(
    settings: StylePresetSettings,
    *,
    generator_factory: ImageGeneratorFactory = MfluxGenerator,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Render every tracked scene under every proposed Style condition."""
    experiment = load_style_preset_experiment(settings.experiment_path)
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="style-presets",
        settings=settings.model_dump(mode="json"),
        models={"mflux": settings.model_identifier},
        contracts={
            "experiment": {
                "version": experiment.experiment_version,
                "sha256": contract_digest(experiment.model_dump(mode="json")),
            }
        },
        environment_provider=environment_provider,
    )
    try:
        result_path = _execute_style_preset_evaluation(
            settings,
            experiment=experiment,
            generator=generator_factory(),
            lifecycle=lifecycle,
        )
    except Exception as error:
        result_path = settings.output_dir / "style-preset-results.json"
        failure = {
            "stage": lifecycle.manifest["stage"],
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = "failed"
            result["failure"] = failure
            _write_result(settings.output_dir, result)
        lifecycle.finalize(status="failed", failure=failure)
        raise

    completed = json.loads(result_path.read_text(encoding="utf-8"))
    failed_rows = [
        {
            "case_id": row["case_id"],
            "seed": row["seed"],
            "style_id": row["style_id"],
            **row["generation"]["failure"],
        }
        for row in completed["results"]
        if row["generation"]["status"] == "failed"
    ]
    lifecycle.finalize(
        status=completed["status"],
        failure=(
            {"classification": "stage_failures", "stages": failed_rows} if failed_rows else None
        ),
    )
    if completed["status"] == "failed":
        raise ImageGenerationError(
            "Style preset evaluation produced no images; "
            f"diagnostics were retained at {result_path}"
        )
    return result_path


__all__ = [
    "DEFAULT_STYLE_PRESET_EXPERIMENT",
    "DEFAULT_STYLE_PRESET_SEEDS",
    "STYLE_PRESET_EXPERIMENT_VERSION",
    "STYLE_PRESET_RESULT_VERSION",
    "StylePresetExperiment",
    "StylePresetSettings",
    "compose_style_preset_prompt",
    "default_style_preset_output_dir",
    "load_style_preset_experiment",
    "run_style_preset_evaluation",
]
