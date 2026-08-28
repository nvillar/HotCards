"""Maintained Image Prompt benchmark and production-baseline runner."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import Field, field_validator, model_validator

import hotcards.domain.models as domain_models_module
import hotcards.generation.image_prompt_preparation as image_prompt_preparation_module
import hotcards.generation.ollama_client as ollama_client_module
import hotcards.generation.structured_output as structured_output_module
from hotcards.domain.models import DomainModel, NonEmptyString, PositiveInt
from hotcards.evaluation.contracts import SafeCaseId
from hotcards.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hotcards.evaluation.reports import render_reports, render_reports_checked
from hotcards.generation.errors import ModelResponseError
from hotcards.generation.image_prompt_preparation import (
    IMAGE_PROMPT_PREPARATION_VERSION,
    ImagePromptPreparationRequest,
    ImagePromptPreparationResult,
    OllamaImagePromptPreparer,
)
from hotcards.generation.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaRuntime,
    OllamaSettings,
)

IMAGE_PROMPT_BENCHMARK_VERSION = "image-prompt-benchmark-v2"
IMAGE_PROMPT_BENCHMARK_RESULT_VERSION = "image-prompt-benchmark-result-v1"
DEFAULT_IMAGE_PROMPT_BENCHMARK = Path(
    "evals/cases/image_prompts/benchmark.json"
)


class BenchmarkProvenance(DomainModel):
    """Human-readable origin and reuse terms for one benchmark input."""

    source_kind: Literal["project-generated", "authored"]
    description: NonEmptyString
    reuse_terms: NonEmptyString


class BenchmarkAsset(DomainModel):
    """One immutable Reference image owned by the benchmark."""

    asset_id: SafeCaseId
    path: NonEmptyString
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: BenchmarkProvenance

    @field_validator("path")
    @classmethod
    def _relative_safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("benchmark asset path must be safe relative data")
        if path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("benchmark asset must use a supported image extension")
        return path.as_posix()


class BenchmarkReference(DomainModel):
    """Frozen Reference image and its generation-time semantic provenance."""

    asset_id: SafeCaseId
    generation_description: NonEmptyString


class RubricCriterion(DomainModel):
    """One observable pass/fail judgment for a generated Image Prompt."""

    criterion_id: SafeCaseId
    category: Literal[
        "target_fidelity",
        "continuity",
        "reference_restraint",
        "override",
        "visible_text",
        "prompt_quality",
    ]
    severity: Literal["critical", "major", "minor"]
    description: NonEmptyString


class ImagePromptBenchmarkCase(DomainModel):
    """One self-contained target request and its human-authored rubric."""

    case_id: SafeCaseId
    title: NonEmptyString
    cohort: Literal["development", "regression", "edge"]
    tags: tuple[SafeCaseId, ...] = Field(min_length=1)
    description: NonEmptyString
    reference: BenchmarkReference | None = None
    criteria: tuple[RubricCriterion, ...] = Field(min_length=1)
    provenance: BenchmarkProvenance

    @model_validator(mode="after")
    def _unique_criterion_ids(self) -> ImagePromptBenchmarkCase:
        criterion_ids = [criterion.criterion_id for criterion in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(
                f"benchmark case {self.case_id!r} criterion IDs must be unique"
            )
        if len(self.tags) != len(set(self.tags)):
            raise ValueError(f"benchmark case {self.case_id!r} tags must be unique")
        return self


class ImagePromptBenchmark(DomainModel):
    """Version-controlled Image Prompt inputs, assets, and scoring criteria."""

    benchmark_version: Literal["image-prompt-benchmark-v2"] = (
        IMAGE_PROMPT_BENCHMARK_VERSION
    )
    benchmark_id: SafeCaseId
    title: NonEmptyString
    description: NonEmptyString
    common_criteria: tuple[RubricCriterion, ...] = Field(min_length=1)
    assets: tuple[BenchmarkAsset, ...] = Field(default_factory=tuple)
    cases: tuple[ImagePromptBenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_cross_references(self) -> ImagePromptBenchmark:
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("benchmark asset IDs must be unique")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")
        common_ids = [criterion.criterion_id for criterion in self.common_criteria]
        if len(common_ids) != len(set(common_ids)):
            raise ValueError("benchmark common criterion IDs must be unique")
        known_assets = set(asset_ids)
        referenced_assets = {
            case.reference.asset_id
            for case in self.cases
            if case.reference is not None
        }
        missing = sorted(referenced_assets - known_assets)
        if missing:
            raise ValueError(
                "benchmark cases reference unknown assets: " + ", ".join(missing)
            )
        unused = sorted(known_assets - referenced_assets)
        if unused:
            raise ValueError("benchmark assets are unused: " + ", ".join(unused))
        for case in self.cases:
            duplicate_ids = set(common_ids) & {
                criterion.criterion_id for criterion in case.criteria
            }
            if duplicate_ids:
                values = ", ".join(sorted(duplicate_ids))
                raise ValueError(
                    f"benchmark case {case.case_id!r} duplicates common criteria: "
                    f"{values}"
                )
        return self


class ImagePromptBenchmarkSettings(DomainModel):
    """Controlled settings for one live production-baseline benchmark run."""

    output_dir: Path
    benchmark_path: Path = DEFAULT_IMAGE_PROMPT_BENCHMARK
    ollama_models: tuple[NonEmptyString, ...] = Field(
        default=(DEFAULT_OLLAMA_MODEL,),
        min_length=1,
    )
    endpoint: NonEmptyString = "http://localhost:11434"
    repetitions: PositiveInt = 1
    candidate_id: SafeCaseId = "production"
    candidate_prompt_version: NonEmptyString = IMAGE_PROMPT_PREPARATION_VERSION

    @model_validator(mode="after")
    def _unique_models(self) -> ImagePromptBenchmarkSettings:
        if len(self.ollama_models) != len(set(self.ollama_models)):
            raise ValueError("Image Prompt benchmark model identifiers must be unique")
        return self


class ImagePromptPreparerProtocol(Protocol):
    """Candidate surface consumed by the benchmark runner."""

    def prepare(
        self,
        request: ImagePromptPreparationRequest,
        *,
        reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult: ...


ImagePromptPreparerFactory = Callable[
    [OllamaSettings],
    ImagePromptPreparerProtocol,
]


def _default_preparer_factory(settings: OllamaSettings) -> ImagePromptPreparerProtocol:
    return OllamaImagePromptPreparer(OllamaRuntime(settings))


def default_image_prompt_benchmark_output_dir() -> Path:
    """Return a unique output directory for one live benchmark run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"image-prompts-{timestamp}"


def benchmark_asset_path(benchmark_path: Path, asset: BenchmarkAsset) -> Path:
    """Resolve one validated benchmark asset within the benchmark directory."""
    root = benchmark_path.parent.resolve()
    path = (benchmark_path.parent / PurePosixPath(asset.path)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"benchmark asset path escapes its directory: {asset.path}"
        ) from error
    return path


def load_image_prompt_benchmark(path: Path) -> ImagePromptBenchmark:
    """Load a strict benchmark and verify every frozen asset checksum."""
    benchmark = ImagePromptBenchmark.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    for asset in benchmark.assets:
        asset_path = benchmark_asset_path(path, asset)
        if not asset_path.is_file():
            raise ValueError(
                f"benchmark asset {asset.asset_id!r} is missing: {asset.path}"
            )
        digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        if digest != asset.sha256:
            raise ValueError(
                f"benchmark asset {asset.asset_id!r} checksum mismatch: "
                f"expected {asset.sha256}, got {digest}"
            )
        try:
            with Image.open(asset_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(
                f"benchmark asset {asset.asset_id!r} is not a readable image"
            ) from error
    return benchmark


def _scorecard(
    benchmark: ImagePromptBenchmark,
    case: ImagePromptBenchmarkCase,
) -> dict[str, object]:
    criteria = (*benchmark.common_criteria, *case.criteria)
    return {
        "criteria": [
            {
                **criterion.model_dump(mode="json"),
                "score": None,
                "notes": None,
            }
            for criterion in criteria
        ],
        "hard_failure": None,
        "overall_notes": None,
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def _model_path_key(model: str) -> str:
    display = _safe_name(model).strip("._-") or "model"
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()
    return f"{display[:48]}-{digest}"


def _write_result(output_dir: Path, result: Mapping[str, object]) -> Path:
    path = output_dir / "image-prompt-results.json"
    atomic_write_json(path, result)
    return path


def _raw_response_path(
    output_dir: Path,
    *,
    model: str,
    case_id: str,
    repetition: int,
    attempt: int,
    phase: str,
) -> Path:
    return (
        output_dir
        / "raw"
        / _model_path_key(model)
        / (
            f"{case_id}-repetition-{repetition}-"
            f"attempt-{attempt}-{_safe_name(phase)}.txt"
        )
    )


def _write_raw_response(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.relative_to(path.parents[2]).as_posix()


def _write_attempt_responses(
    output_dir: Path,
    *,
    model: str,
    case_id: str,
    repetition: int,
    responses: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    return tuple(
        _write_raw_response(
            _raw_response_path(
                output_dir,
                model=model,
                case_id=case_id,
                repetition=repetition,
                attempt=index,
                phase=phase,
            ),
            content,
        )
        for index, (phase, content) in enumerate(responses, start=1)
    )


def _success_record(
    result: ImagePromptPreparationResult,
    *,
    raw_response_paths: tuple[str, ...],
) -> dict[str, object]:
    return {
        "status": "success",
        "image_prompt": result.image_prompt,
        "raw_response_paths": raw_response_paths,
        "attempt_count": len(raw_response_paths),
        "repair_applied": result.repair_applied,
        "model_identifier": result.model_identifier,
        "prompt_version": result.prompt_version,
        "elapsed_seconds": result.duration_seconds,
        "total_duration_ns": result.total_duration_ns,
        "load_duration_ns": result.load_duration_ns,
        "prompt_eval_count": result.prompt_eval_count,
        "eval_count": result.eval_count,
        "done_reason": result.done_reason,
    }


def _failure_record(
    error: BaseException,
    *,
    raw_response_paths: tuple[str, ...],
    attempt_count: int,
    repair_attempted: bool,
) -> dict[str, object]:
    metadata = (
        error.response_metadata
        if isinstance(error, ModelResponseError)
        else {}
    )
    return {
        "status": "failed",
        "failure": {
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        },
        "raw_response_paths": raw_response_paths,
        "attempt_count": attempt_count,
        "repair_attempted": repair_attempted,
        **metadata,
    }


def _execute_image_prompt_benchmark(
    settings: ImagePromptBenchmarkSettings,
    *,
    preparer_factory: ImagePromptPreparerFactory,
    lifecycle: RunLifecycle,
) -> Path:
    benchmark = load_image_prompt_benchmark(settings.benchmark_path)
    benchmark_payload = benchmark.model_dump(mode="json")
    lifecycle.update_contract(
        "image_prompt_benchmark",
        {
            "version": benchmark.benchmark_version,
            "sha256": contract_digest(benchmark_payload),
        },
    )
    asset_by_id = {asset.asset_id: asset for asset in benchmark.assets}
    rows: list[dict[str, object]] = []
    result: dict[str, object] = {
        "result_version": IMAGE_PROMPT_BENCHMARK_RESULT_VERSION,
        "suite": "image_prompts",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "benchmark": benchmark_payload,
        "candidate_id": settings.candidate_id,
        "results": rows,
    }
    _write_result(settings.output_dir, result)

    for model in settings.ollama_models:
        preparer = preparer_factory(
            OllamaSettings(endpoint=settings.endpoint, model=model)
        )
        for case in benchmark.cases:
            reference_path = (
                benchmark_asset_path(
                    settings.benchmark_path,
                    asset_by_id[case.reference.asset_id],
                )
                if case.reference is not None
                else None
            )
            request = ImagePromptPreparationRequest(
                description=case.description,
                has_reference=case.reference is not None,
                reference_description=(
                    case.reference.generation_description
                    if case.reference is not None
                    else None
                ),
                prompt_version=settings.candidate_prompt_version,
            )
            for repetition in range(1, settings.repetitions + 1):
                stage = (
                    f"image_prompt:{settings.candidate_id}:{model}:"
                    f"{case.case_id}:{repetition}"
                )
                lifecycle.set_stage(stage)
                try:
                    prepared = preparer.prepare(
                        request,
                        reference_image_path=reference_path,
                    )
                except Exception as error:
                    error_attempts = tuple(
                        getattr(error, "response_attempts", ())
                    )
                    responses = (
                        tuple(
                            (
                                str(attempt.get("phase", "attempt")),
                                str(attempt["raw_response"]),
                            )
                            for attempt in error_attempts
                            if attempt.get("raw_response")
                        )
                        if error_attempts
                        else ()
                    )
                    if (
                        not responses
                        and isinstance(error, ModelResponseError)
                        and error.raw_response is not None
                    ):
                        responses = (("final", error.raw_response),)
                    attempt_count = (
                        len(error_attempts)
                        if error_attempts
                        else len(responses)
                    )
                    raw_relatives = _write_attempt_responses(
                        settings.output_dir,
                        model=model,
                        case_id=case.case_id,
                        repetition=repetition,
                        responses=responses,
                    )
                    record = _failure_record(
                        error,
                        raw_response_paths=raw_relatives,
                        attempt_count=attempt_count,
                        repair_attempted=any(
                            attempt.get("phase") == "repair"
                            for attempt in error_attempts
                        ),
                    )
                else:
                    responses = (
                        tuple(
                            (attempt.phase, attempt.raw_response)
                            for attempt in prepared.attempts
                        )
                        if prepared.attempts
                        else (("final", prepared.raw_response),)
                    )
                    raw_relatives = _write_attempt_responses(
                        settings.output_dir,
                        model=model,
                        case_id=case.case_id,
                        repetition=repetition,
                        responses=responses,
                    )
                    record = _success_record(
                        prepared,
                        raw_response_paths=raw_relatives,
                    )
                    lifecycle.complete_stage(stage)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "cohort": case.cohort,
                        "tags": case.tags,
                        "candidate_id": settings.candidate_id,
                        "model": model,
                        "repetition": repetition,
                        **record,
                        "rubric": _scorecard(benchmark, case),
                    }
                )
                _write_result(settings.output_dir, result)

    failures = [row for row in rows if row["status"] == "failed"]
    result["status"] = "completed_with_failures" if failures else "success"
    result_path = _write_result(settings.output_dir, result)
    lifecycle.set_stage("reports")
    render_reports_checked(result_path)
    lifecycle.complete_stage("reports")
    return result_path


def run_image_prompt_benchmark(
    settings: ImagePromptBenchmarkSettings,
    *,
    preparer_factory: ImagePromptPreparerFactory = _default_preparer_factory,
    candidate_contract: Mapping[str, object] | None = None,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Run one candidate against the maintained benchmark."""
    contract = dict(
        candidate_contract
        or {
            "version": settings.candidate_prompt_version,
            "sha256": contract_digest(
                settings.candidate_prompt_version,
                inspect.getsource(image_prompt_preparation_module),
                inspect.getsource(ollama_client_module),
                inspect.getsource(structured_output_module),
                inspect.getsource(domain_models_module),
            ),
        }
    )
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="image_prompts",
        settings=settings.model_dump(mode="json"),
        models={"ollama_candidates": list(settings.ollama_models)},
        contracts={
            "image_prompt_candidate": contract,
            "image_prompt_benchmark": {
                "version": IMAGE_PROMPT_BENCHMARK_VERSION
            },
        },
        environment_provider=environment_provider,
    )
    try:
        result_path = _execute_image_prompt_benchmark(
            settings,
            preparer_factory=preparer_factory,
            lifecycle=lifecycle,
        )
    except Exception as error:
        result_path = settings.output_dir / "image-prompt-results.json"
        report_failure = lifecycle.manifest["stage"] == "reports"
        failure = {
            "stage": lifecycle.manifest["stage"],
            "classification": (
                "report_rendering" if report_failure else classify_failure(error)
            ),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        status = (
            "completed_with_report_failure"
            if report_failure
            else "failed"
        )
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = status
            result["failure"] = failure
            _write_result(settings.output_dir, result)
            if not report_failure:
                try:
                    render_reports(result_path)
                except Exception as report_error:
                    lifecycle.add_warning(
                        f"failure report could not be rendered: {report_error}"
                    )
        lifecycle.finalize(status=status, failure=failure)
        raise

    completed = json.loads(result_path.read_text(encoding="utf-8"))
    failed_rows = [
        {
            "case_id": row["case_id"],
            "model": row["model"],
            "repetition": row["repetition"],
            **row["failure"],
        }
        for row in completed["results"]
        if row["status"] == "failed"
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
    "DEFAULT_IMAGE_PROMPT_BENCHMARK",
    "IMAGE_PROMPT_BENCHMARK_RESULT_VERSION",
    "IMAGE_PROMPT_BENCHMARK_VERSION",
    "BenchmarkAsset",
    "BenchmarkProvenance",
    "BenchmarkReference",
    "ImagePromptBenchmark",
    "ImagePromptBenchmarkCase",
    "ImagePromptBenchmarkSettings",
    "RubricCriterion",
    "benchmark_asset_path",
    "default_image_prompt_benchmark_output_dir",
    "load_image_prompt_benchmark",
    "run_image_prompt_benchmark",
]
