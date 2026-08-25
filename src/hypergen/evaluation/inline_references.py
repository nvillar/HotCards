"""Paired FLUX evaluation for explicit inline Reference language."""

from __future__ import annotations

import html
import json
import random
import secrets
import shutil
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

from PIL import Image
from pydantic import Field, model_validator

from hypergen.domain.models import (
    DomainModel,
    ImageGenerationInputs,
    ImageReferenceSnapshot,
    NonEmptyString,
    PositiveInt,
)
from hypergen.evaluation.contracts import SafeCaseId
from hypergen.evaluation.image_prompts import (
    DEFAULT_IMAGE_PROMPT_BENCHMARK,
    ImagePromptBenchmark,
    ImagePromptBenchmarkCase,
    RubricCriterion,
    benchmark_asset_path,
    load_image_prompt_benchmark,
)
from hypergen.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hypergen.evaluation.reports import create_contact_sheet
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerationResult,
    MfluxGenerator,
)

INLINE_REFERENCE_EXPERIMENT_VERSION = "inline-reference-language-v1"
INLINE_REFERENCE_RESULT_VERSION = "inline-reference-render-result-v1"
INLINE_REFERENCE_REVIEW_VERSION = "inline-reference-review-v1"
DEFAULT_INLINE_REFERENCE_EXPERIMENT = Path(
    "evals/cases/image_prompts/inline_reference_language.json"
)
DEFAULT_INLINE_REFERENCE_SEEDS = (42, 314159, 271828)


class InlineReferenceCase(DomainModel):
    """One paired prompt condition tied to a maintained benchmark case."""

    case_id: SafeCaseId
    standalone_prompt: NonEmptyString
    inline_prompt: NonEmptyString


class InlineReferenceExperiment(DomainModel):
    """Tracked prompt pairs for one image-level Reference experiment."""

    experiment_version: Literal["inline-reference-language-v1"] = (
        INLINE_REFERENCE_EXPERIMENT_VERSION
    )
    experiment_id: SafeCaseId
    title: NonEmptyString
    description: NonEmptyString
    cases: tuple[InlineReferenceCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> InlineReferenceExperiment:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("inline Reference experiment case IDs must be unique")
        return self


class InlineReferenceSettings(DomainModel):
    """Controlled MFLUX and blinding settings for one paired run."""

    output_dir: Path
    experiment_path: Path = DEFAULT_INLINE_REFERENCE_EXPERIMENT
    benchmark_path: Path = DEFAULT_IMAGE_PROMPT_BENCHMARK
    model_identifier: NonEmptyString = "flux2-klein-9b-kv"
    seeds: tuple[int, ...] = Field(default=DEFAULT_INLINE_REFERENCE_SEEDS, min_length=1)
    width: PositiveInt = 1024
    height: PositiveInt = 768
    step_count: PositiveInt = 4
    quantization: int | None = None
    blinding_seed: int | None = None

    @model_validator(mode="after")
    def _unique_seeds(self) -> InlineReferenceSettings:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("inline Reference experiment seeds must be unique")
        return self


class ImageGeneratorProtocol(Protocol):
    """Production MFLUX surface consumed by the paired runner."""

    def generate(self, request: MfluxGenerationRequest) -> MfluxGenerationResult: ...


ImageGeneratorFactory = Callable[[], ImageGeneratorProtocol]


def default_inline_reference_output_dir() -> Path:
    """Return a unique immutable output directory for one paired run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"inline-references-{timestamp}"


def load_inline_reference_experiment(
    experiment_path: Path,
    benchmark_path: Path = DEFAULT_IMAGE_PROMPT_BENCHMARK,
) -> tuple[InlineReferenceExperiment, ImagePromptBenchmark]:
    """Load the prompt pairs and verify their frozen benchmark dependencies."""
    experiment = InlineReferenceExperiment.model_validate_json(
        experiment_path.read_text(encoding="utf-8")
    )
    benchmark = load_image_prompt_benchmark(benchmark_path)
    if not any(
        criterion.criterion_id == "no-unrequested-visible-text"
        for criterion in benchmark.common_criteria
    ):
        raise ValueError(
            "inline Reference benchmark requires common criterion "
            "'no-unrequested-visible-text'"
        )
    benchmark_cases = {case.case_id: case for case in benchmark.cases}
    for case in experiment.cases:
        benchmark_case = benchmark_cases.get(case.case_id)
        if benchmark_case is None:
            raise ValueError(
                f"inline Reference case {case.case_id!r} is absent from the benchmark"
            )
        if benchmark_case.reference is None:
            raise ValueError(
                f"inline Reference case {case.case_id!r} has no Reference image"
            )
        if case.standalone_prompt == case.inline_prompt:
            raise ValueError(
                f"inline Reference case {case.case_id!r} prompt conditions must differ"
            )
    return experiment, benchmark


def _balanced_assignments(
    pair_ids: tuple[str, ...],
    *,
    blinding_seed: int,
) -> dict[str, dict[str, str]]:
    shuffled = list(pair_ids)
    random.Random(blinding_seed).shuffle(shuffled)
    inline_as_a = set(shuffled[: len(shuffled) // 2])
    return {
        pair_id: (
            {"A": "inline", "B": "standalone"}
            if pair_id in inline_as_a
            else {"A": "standalone", "B": "inline"}
        )
        for pair_id in pair_ids
    }


def _reference_snapshot(asset_id: str) -> ImageReferenceSnapshot:
    return ImageReferenceSnapshot(
        card_id=uuid5(NAMESPACE_URL, f"hypergen-eval:{asset_id}:card"),
        revision_id=uuid5(NAMESPACE_URL, f"hypergen-eval:{asset_id}:revision"),
        background_id=uuid5(NAMESPACE_URL, f"hypergen-eval:{asset_id}:background"),
    )


def _generation_record(
    generated: MfluxGenerationResult,
    *,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "status": "success",
        "artifact_path": generated.output_path.relative_to(output_dir).as_posix(),
        "load_duration_seconds": generated.load_duration_seconds,
        "inference_duration_seconds": generated.generation_duration_seconds,
        "serialization_duration_seconds": generated.serialization_duration_seconds,
        "total_duration_seconds": generated.metadata.duration_seconds,
        "metadata": generated.metadata.model_dump(mode="json"),
    }


def _public_settings(settings: InlineReferenceSettings) -> dict[str, object]:
    return settings.model_dump(mode="json", exclude={"blinding_seed"})


def _sanitize_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        sanitized = image.copy()
    sanitized.save(destination, format="PNG")


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
    benchmark: ImagePromptBenchmark,
    case: ImagePromptBenchmarkCase,
) -> list[dict[str, object]]:
    visible_text = next(
        criterion
        for criterion in benchmark.common_criteria
        if criterion.criterion_id == "no-unrequested-visible-text"
    )
    integrity = RubricCriterion(
        criterion_id="no-obvious-generation-artifacts",
        category="target_fidelity",
        severity="major",
        description=(
            "The image has no obvious malformed geometry, duplicated subject, "
            "unreadable structure, or other generation artifact that prevents review."
        ),
    )
    return [
        {
            **criterion.model_dump(mode="json"),
            "score": None,
            "notes": None,
        }
        for criterion in (visible_text, *case.criteria, integrity)
    ]


def _write_review_html(review_path: Path, scorecard: Mapping[str, object]) -> None:
    sections: list[str] = []
    for pair in scorecard["pairs"]:  # type: ignore[index]
        criteria = "".join(
            "<li>"
            f"<strong>{html.escape(str(item['severity']))}</strong> "
            f"{html.escape(str(item['criterion_id']))}: "
            f"{html.escape(str(item['description']))}</li>"
            for item in pair["candidates"][0]["criteria"]
        )
        sheet = html.escape(quote(Path(pair["comparison_sheet"]).name))
        sections.append(
            "<section>"
            f"<h2>{html.escape(str(pair['title']))} · seed {pair['seed']}</h2>"
            f"<p>{html.escape(str(pair['target_description']))}</p>"
            f'<a href="{sheet}"><img src="{sheet}" '
            f'alt="{html.escape(str(pair["pair_id"]))} comparison"></a>'
            f"<h3>Criteria for A and B</h3><ul>{criteria}</ul>"
            "</section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>HyperGen blind inline Reference review</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem;color:#18202a;max-width:1200px}}
section{{border-top:2px solid #ccd3dc;margin-top:2rem;padding-top:1rem}}
img{{max-width:100%;height:auto;border:1px solid #ccd3dc}}
li{{margin:.35rem 0}}
</style></head><body>
<h1>Blind inline Reference review</h1>
<p>Score A and B independently as pass, fail, or uncertain for every criterion.
Do not inspect files under <code>audit/</code> until all scores are complete.
Any technical failure or failed critical criterion is a hard failure.</p>
{''.join(sections)}
</body></html>"""
    review_path.write_text(document, encoding="utf-8")


def _write_result(output_dir: Path, result: Mapping[str, object]) -> Path:
    result_path = output_dir / "inline-reference-results.json"
    atomic_write_json(result_path, result)
    return result_path


def _execute_inline_reference_evaluation(
    settings: InlineReferenceSettings,
    *,
    experiment: InlineReferenceExperiment,
    benchmark: ImagePromptBenchmark,
    generator: ImageGeneratorProtocol,
    lifecycle: RunLifecycle,
    blinding_seed: int,
) -> Path:
    benchmark_cases = {case.case_id: case for case in benchmark.cases}
    benchmark_assets = {asset.asset_id: asset for asset in benchmark.assets}
    pair_ids = tuple(
        f"{case.case_id}-seed-{seed}"
        for case in experiment.cases
        for seed in settings.seeds
    )
    assignments = _balanced_assignments(
        pair_ids,
        blinding_seed=blinding_seed,
    )
    inputs_dir = settings.output_dir / "inputs"
    outputs_dir = settings.output_dir / "outputs"
    review_dir = settings.output_dir / "review"
    audit_dir = settings.output_dir / "audit"
    for directory in (inputs_dir, outputs_dir, review_dir, audit_dir):
        directory.mkdir(parents=True)

    referenced_asset_ids = {
        benchmark_cases[case.case_id].reference.asset_id  # type: ignore[union-attr]
        for case in experiment.cases
    }
    copied_assets: dict[str, Path] = {}
    for asset_id in sorted(referenced_asset_ids):
        asset = benchmark_assets[asset_id]
        source = benchmark_asset_path(settings.benchmark_path, asset)
        destination = inputs_dir / Path(asset.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied_assets[asset_id] = destination

    condition_key = {
        "experiment_version": experiment.experiment_version,
        "warning": "Do not inspect until blind scoring is complete.",
        "blinding_seed": blinding_seed,
        "pairs": [
            {
                "pair_id": pair_id,
                "case_id": experiment_case.case_id,
                "seed": seed,
                "candidates": {
                    label: {
                        "condition_id": assignments[pair_id][label],
                        "render_prompt": (
                            experiment_case.inline_prompt
                            if assignments[pair_id][label] == "inline"
                            else experiment_case.standalone_prompt
                        ),
                    }
                    for label in ("A", "B")
                },
            }
            for experiment_case in experiment.cases
            for seed in settings.seeds
            if (pair_id := f"{experiment_case.case_id}-seed-{seed}")
        ],
    }
    atomic_write_json(audit_dir / "condition-key.json", condition_key)
    audit_records: list[dict[str, object]] = []
    atomic_write_json(audit_dir / "generation-records.json", audit_records)
    result: dict[str, object] = {
        "result_version": INLINE_REFERENCE_RESULT_VERSION,
        "suite": "inline-references",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": _public_settings(settings),
        "experiment": {
            "experiment_version": experiment.experiment_version,
            "experiment_id": experiment.experiment_id,
            "title": experiment.title,
        },
        "results": [],
        "warnings": [],
    }
    _write_result(settings.output_dir, result)
    records: list[dict[str, object]] = result["results"]  # type: ignore[assignment]

    for experiment_case in experiment.cases:
        benchmark_case = benchmark_cases[experiment_case.case_id]
        reference = benchmark_case.reference
        assert reference is not None
        reference_path = copied_assets[reference.asset_id]
        prompts = {
            "standalone": experiment_case.standalone_prompt,
            "inline": experiment_case.inline_prompt,
        }
        for seed in settings.seeds:
            pair_id = f"{experiment_case.case_id}-seed-{seed}"
            for label in ("A", "B"):
                condition_id = assignments[pair_id][label]
                prompt = prompts[condition_id]
                stage = f"pair:{pair_id}:candidate-{label.lower()}"
                lifecycle.set_stage(stage)
                public_output_path = (
                    outputs_dir
                    / experiment_case.case_id
                    / f"seed-{seed}"
                    / f"candidate-{label.lower()}.png"
                )
                audit_output_path = (
                    audit_dir
                    / "outputs"
                    / experiment_case.case_id
                    / f"seed-{seed}"
                    / f"candidate-{label.lower()}.png"
                )
                request = MfluxGenerationRequest(
                    inputs=ImageGenerationInputs(
                        description=benchmark_case.description,
                        image_prompt=prompt,
                        reference=_reference_snapshot(reference.asset_id),
                    ),
                    render_prompt=prompt,
                    output_path=audit_output_path,
                    model_identifier=settings.model_identifier,
                    seed=seed,
                    width=settings.width,
                    height=settings.height,
                    step_count=settings.step_count,
                    quantization=settings.quantization,
                    reference_image_paths=(reference_path,),
                )
                try:
                    generated = generator.generate(request)
                    audited_generation = _generation_record(
                        generated,
                        output_dir=settings.output_dir,
                    )
                    _sanitize_png(generated.output_path, public_output_path)
                    generation = {
                        key: value
                        for key, value in audited_generation.items()
                        if key != "metadata"
                    }
                    generation["artifact_path"] = public_output_path.relative_to(
                        settings.output_dir
                    ).as_posix()
                    lifecycle.complete_stage(stage)
                except Exception as error:
                    generation = _failure_record(error)
                    audited_generation = generation
                record = {
                    "pair_id": pair_id,
                    "case_id": experiment_case.case_id,
                    "seed": seed,
                    "candidate_label": label,
                    "reference_asset_id": reference.asset_id,
                    "generation": generation,
                }
                records.append(record)
                audit_records.append(
                    {
                        **record,
                        "condition_id": condition_id,
                        "render_prompt": prompt,
                        "generation": audited_generation,
                    }
                )
                atomic_write_json(
                    audit_dir / "generation-records.json",
                    audit_records,
                )
                _write_result(settings.output_dir, result)
    scorecard_pairs: list[dict[str, object]] = []
    records_by_pair = {
        (str(record["pair_id"]), str(record["candidate_label"])): record
        for record in records
    }
    for experiment_case in experiment.cases:
        benchmark_case = benchmark_cases[experiment_case.case_id]
        reference = benchmark_case.reference
        assert reference is not None
        reference_path = copied_assets[reference.asset_id]
        for seed in settings.seeds:
            pair_id = f"{experiment_case.case_id}-seed-{seed}"
            candidates: list[dict[str, object]] = []
            entries = [(reference_path, "Reference")]
            for label in ("A", "B"):
                record = records_by_pair[(pair_id, label)]
                generation = record["generation"]
                assert isinstance(generation, dict)
                artifact_path = generation.get("artifact_path")
                if artifact_path:
                    entries.append(
                        (settings.output_dir / str(artifact_path), f"Candidate {label}")
                    )
                candidates.append(
                    {
                        "label": label,
                        "status": generation["status"],
                        "artifact_path": artifact_path,
                        "criteria": _scorecard_criteria(benchmark, benchmark_case),
                        "hard_failure": (
                            True if generation["status"] == "failed" else None
                        ),
                        "overall_notes": None,
                    }
                )
            sheet_path = review_dir / f"{pair_id}.png"
            create_contact_sheet(
                entries,
                sheet_path,
                cell_size=(420, 320),
                columns=3,
            )
            scorecard_pairs.append(
                {
                    "pair_id": pair_id,
                    "case_id": experiment_case.case_id,
                    "title": benchmark_case.title,
                    "seed": seed,
                    "target_description": benchmark_case.description,
                    "reference_path": reference_path.relative_to(
                        settings.output_dir
                    ).as_posix(),
                    "comparison_sheet": sheet_path.relative_to(review_dir).as_posix(),
                    "candidates": candidates,
                    "pair_preference": None,
                    "pair_notes": None,
                }
            )
    scorecard = {
        "review_version": INLINE_REFERENCE_REVIEW_VERSION,
        "experiment_id": experiment.experiment_id,
        "reviewer": None,
        "scoring_values": ["pass", "fail", "uncertain"],
        "instructions": (
            "Score A and B independently. Treat technical failure or any failed "
            "critical criterion as a hard failure. Do not inspect audit files "
            "until every pair is scored."
        ),
        "pairs": scorecard_pairs,
    }
    atomic_write_json(review_dir / "scorecard.json", scorecard)
    _write_review_html(review_dir / "index.html", scorecard)
    lifecycle.complete_stage("review-artifacts")

    failures = [
        record
        for record in records
        if record["generation"]["status"] == "failed"  # type: ignore[index]
    ]
    result["status"] = "completed_with_failures" if failures else "success"
    result["review_path"] = "review/index.html"
    result["scorecard_path"] = "review/scorecard.json"
    result["condition_key_path"] = "audit/condition-key.json"
    return _write_result(settings.output_dir, result)


def run_inline_reference_evaluation(
    settings: InlineReferenceSettings,
    *,
    generator_factory: ImageGeneratorFactory = MfluxGenerator,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Run matched standalone and inline-reference prompts through MFLUX."""
    experiment, benchmark = load_inline_reference_experiment(
        settings.experiment_path,
        settings.benchmark_path,
    )
    blinding_seed = (
        settings.blinding_seed
        if settings.blinding_seed is not None
        else secrets.randbits(128)
    )
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="inline-references",
        settings=_public_settings(settings),
        models={"mflux_edit": settings.model_identifier},
        contracts={
            "experiment": {
                "version": experiment.experiment_version,
                "sha256": contract_digest(experiment.model_dump(mode="json")),
            },
            "benchmark": {
                "version": benchmark.benchmark_version,
                "sha256": contract_digest(benchmark.model_dump(mode="json")),
            },
        },
        environment_provider=environment_provider,
    )
    try:
        result_path = _execute_inline_reference_evaluation(
            settings,
            experiment=experiment,
            benchmark=benchmark,
            generator=generator_factory(),
            lifecycle=lifecycle,
            blinding_seed=blinding_seed,
        )
    except Exception as error:
        result_path = settings.output_dir / "inline-reference-results.json"
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
            "pair_id": row["pair_id"],
            "candidate_label": row["candidate_label"],
            **row["generation"]["failure"],
        }
        for row in completed["results"]
        if row["generation"]["status"] == "failed"
    ]
    lifecycle.finalize(
        status=completed["status"],
        failure=(
            {"classification": "stage_failures", "stages": failed_rows}
            if failed_rows
            else None
        ),
    )
    return result_path


__all__ = [
    "DEFAULT_INLINE_REFERENCE_EXPERIMENT",
    "DEFAULT_INLINE_REFERENCE_SEEDS",
    "INLINE_REFERENCE_EXPERIMENT_VERSION",
    "INLINE_REFERENCE_RESULT_VERSION",
    "InlineReferenceCase",
    "InlineReferenceExperiment",
    "InlineReferenceSettings",
    "default_inline_reference_output_dir",
    "load_inline_reference_experiment",
    "run_inline_reference_evaluation",
]
