"""End-to-end production-adapter evaluation with one fixed MFLUX axis."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from hypergen.domain.models import DomainModel, ImageGenerationInputs, NonEmptyString, PositiveInt
from hypergen.evaluation.contracts import SafeCaseId
from hypergen.evaluation.hotspots import (
    HUMAN_RUBRIC_FIELDS as HOTSPOT_RUBRIC_FIELDS,
)
from hypergen.evaluation.hotspots import HotspotEvaluationCase, _success_record
from hypergen.evaluation.images import HUMAN_RUBRIC_FIELDS as IMAGE_RUBRIC_FIELDS
from hypergen.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hypergen.evaluation.reports import render_reports, render_reports_checked
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.hotspot_prompts import (
    HOTSPOT_REMAP_PROMPT_VERSION,
    HOTSPOT_REMAP_SCHEMA_VERSION,
    HotspotRemapModelOutput,
    HotspotRemapRequest,
    OllamaHotspotRemapper,
    RemapHotspotInput,
    build_hotspot_remap_prompt,
    build_hotspot_remap_schema,
)
from hypergen.generation.image_prompts import IMAGE_PROMPT_VERSION, compose_image_prompt
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerator,
)
from hypergen.generation.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaRuntime,
    OllamaSettings,
)

E2E_CASE_VERSION = "e2e-case-v2"
E2E_RESULT_VERSION = "e2e-result-v2"
E2E_OLLAMA_MODELS = ("qwen3.5:4b", DEFAULT_OLLAMA_MODEL, "qwen3.6:35b")
E2E_MFLUX_MODEL = "flux2-klein-4b"


class E2EEvaluationCase(DomainModel):
    """Tracked author input, destination contract, and provenance."""

    case_version: Literal["e2e-case-v2"] = E2E_CASE_VERSION
    case_id: SafeCaseId
    inputs: ImageGenerationInputs
    hotspots: tuple[RemapHotspotInput, ...] = Field(min_length=1)
    provenance: NonEmptyString
    reuse_terms: NonEmptyString


class E2EEvaluationSettings(DomainModel):
    """Fixed-axis end-to-end settings."""

    output_dir: Path
    case_dir: Path = Path("evals/cases/e2e")
    ollama_endpoint: NonEmptyString = "http://localhost:11434"
    seed: int = 42
    width: PositiveInt = 1024
    height: PositiveInt = 768
    step_count: PositiveInt = 4
    quantization: int | None = None
    coordinate_extent: PositiveInt = 1000
    ollama_timeout_seconds: float = Field(default=120.0, gt=0.0, allow_inf_nan=False)
    ollama_num_predict: PositiveInt = 1024
    ollama_context_length: PositiveInt = 8192


OllamaRuntimeFactory = Callable[[OllamaSettings], OllamaRuntime]
MfluxGeneratorFactory = Callable[[], MfluxGenerator]


def default_e2e_output_dir() -> Path:
    """Return a unique output directory for an end-to-end run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"e2e-{timestamp}"


def load_e2e_cases(case_dir: Path) -> tuple[E2EEvaluationCase, ...]:
    """Load tracked end-to-end cases in deterministic filename order."""
    paths = sorted(case_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no end-to-end evaluation cases found in {case_dir}")
    cases = tuple(
        E2EEvaluationCase.model_validate_json(path.read_text(encoding="utf-8")) for path in paths
    )
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("end-to-end evaluation case IDs must be unique")
    return cases


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def _write_result(output_dir: Path, result: dict[str, object]) -> Path:
    path = output_dir / "e2e-results.json"
    atomic_write_json(path, result)
    return path


def _failure_stage(name: str, error: BaseException, elapsed: float) -> dict[str, object]:
    rubric = None
    if name == "image_generation":
        rubric = {field: None for field in IMAGE_RUBRIC_FIELDS}
    elif name == "hotspot_remap":
        rubric = {field: None for field in HOTSPOT_RUBRIC_FIELDS}
    return {
        "name": name,
        "status": "failed",
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(UTC).isoformat(),
        "failure": {
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        },
        "warnings": [],
        "human_rubric": rubric,
    }


def _collect_stage_warnings(result: dict[str, object]) -> list[str]:
    return [
        f"{candidate['case_id']}:{candidate['ollama_model']}:{stage['name']}: {warning}"
        for candidate in result.get("candidates", [])  # type: ignore[union-attr]
        for stage in candidate.get("stages", [])
        for warning in stage.get("warnings", [])
    ]


def _execute_e2e(
    settings: E2EEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory,
    mflux_factory: MfluxGeneratorFactory,
    lifecycle: RunLifecycle,
    cases: tuple[E2EEvaluationCase, ...],
) -> Path:
    raw_dir = settings.output_dir / "raw"
    raw_dir.mkdir()
    images_dir = settings.output_dir / "images"
    images_dir.mkdir()
    candidates: list[dict[str, object]] = []
    result: dict[str, object] = {
        "result_version": E2E_RESULT_VERSION,
        "suite": "e2e",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "models": {
            "ollama_candidates": list(E2E_OLLAMA_MODELS),
            "fixed_mflux": E2E_MFLUX_MODEL,
        },
        "cases": [case.model_dump(mode="json") for case in cases],
        "candidates": candidates,
        "warnings": [],
    }
    _write_result(settings.output_dir, result)
    mflux = mflux_factory()
    for case in cases:
        render_prompt = compose_image_prompt(case.inputs)
        stage_name = "image_generation"
        lifecycle.set_stage(f"{case.case_id}:{stage_name}")
        image_path = images_dir / case.case_id / "background.png"
        started = perf_counter()
        try:
            generated = mflux.generate(
                MfluxGenerationRequest(
                    inputs=case.inputs,
                    render_prompt=render_prompt,
                    output_path=image_path,
                    model_identifier=E2E_MFLUX_MODEL,
                    seed=settings.seed,
                    width=settings.width,
                    height=settings.height,
                    step_count=settings.step_count,
                    quantization=settings.quantization,
                )
            )
        except Exception as error:
            artifact_path = None
            image_stage = _failure_stage(
                stage_name,
                error,
                perf_counter() - started,
            )
        else:
            artifact_path = image_path.relative_to(settings.output_dir).as_posix()
            image_stage = {
                "name": stage_name,
                "status": "success",
                "elapsed_seconds": generated.metadata.duration_seconds,
                "completed_at": datetime.now(UTC).isoformat(),
                "artifact_path": artifact_path,
                "timings": {
                    "load_seconds": generated.load_duration_seconds,
                    "inference_seconds": generated.generation_duration_seconds,
                    "serialization_seconds": generated.serialization_duration_seconds,
                },
                "metadata": generated.metadata.model_dump(mode="json"),
                "warnings": [],
                "human_rubric": {field: None for field in IMAGE_RUBRIC_FIELDS},
            }
            lifecycle.complete_stage(f"{case.case_id}:{stage_name}")
        for model in E2E_OLLAMA_MODELS:
            model_name = _safe_name(model)
            raw_case_dir = raw_dir / case.case_id / model_name
            raw_case_dir.mkdir(parents=True)
            candidate: dict[str, object] = {
                "case_id": case.case_id,
                "ollama_model": model,
                "mflux_model": E2E_MFLUX_MODEL,
                "prompt_version": IMAGE_PROMPT_VERSION,
                "render_prompt": render_prompt,
                "stages": [image_stage],
                "artifact_path": artifact_path,
                "hotspot_proposals": None,
                "image_human_rubric": {field: None for field in IMAGE_RUBRIC_FIELDS},
                "hotspot_human_rubric": {field: None for field in HOTSPOT_RUBRIC_FIELDS},
            }
            candidates.append(candidate)
            if artifact_path is None:
                _write_result(settings.output_dir, result)
                continue
            runtime = runtime_factory(
                OllamaSettings(
                    endpoint=settings.ollama_endpoint,
                    model=model,
                    think=False,
                    temperature=0.0,
                    request_timeout_seconds=settings.ollama_timeout_seconds,
                    num_predict=settings.ollama_num_predict,
                    context_length=settings.ollama_context_length,
                )
            )
            candidate["ollama_effective_settings"] = asdict(runtime.settings)
            stage_name = "ollama_diagnostics"
            lifecycle.set_stage(f"{case.case_id}:{model}:{stage_name}")
            started = perf_counter()
            try:
                runtime.require_model(capabilities=frozenset({"vision"}))
            except Exception as error:
                candidate["stages"].append(  # type: ignore[union-attr]
                    _failure_stage(stage_name, error, perf_counter() - started)
                )
                _write_result(settings.output_dir, result)
                continue
            candidate["stages"].append(  # type: ignore[union-attr]
                {
                    "name": stage_name,
                    "status": "success",
                    "elapsed_seconds": perf_counter() - started,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "warnings": [],
                    "human_rubric": None,
                }
            )
            _write_result(settings.output_dir, result)
            lifecycle.complete_stage(f"{case.case_id}:{model}:{stage_name}")

            stage_name = "hotspot_remap"
            lifecycle.set_stage(f"{case.case_id}:{model}:{stage_name}")
            started = perf_counter()
            try:
                hotspots = OllamaHotspotRemapper(runtime).remap(
                    HotspotRemapRequest(
                        image_path=image_path,
                        hotspots=case.hotspots,
                        coordinate_extent=settings.coordinate_extent,
                    )
                )
            except Exception as error:
                if isinstance(error, ModelResponseError) and error.raw_response is not None:
                    (raw_case_dir / "partial-hotspots.json").write_text(
                        error.raw_response, encoding="utf-8"
                    )
                candidate["stages"].append(  # type: ignore[union-attr]
                    _failure_stage(stage_name, error, perf_counter() - started)
                )
                _write_result(settings.output_dir, result)
                continue
            (raw_case_dir / "hotspots.json").write_text(
                json.dumps(list(hotspots.raw_responses), indent=2),
                encoding="utf-8",
            )
            scored = _success_record(
                HotspotEvaluationCase(
                    case_id=case.case_id,
                    image_path=image_path.name,
                    hotspots=case.hotspots,
                ),
                hotspots,
            )
            candidate["hotspot_proposals"] = scored["result"]["proposals"]  # type: ignore[index]
            candidate["stages"].append(  # type: ignore[union-attr]
                {
                    "name": stage_name,
                    "status": "success",
                    "elapsed_seconds": hotspots.provenance.duration_seconds,
                    "completed_at": datetime.now(UTC).isoformat(),
                    **scored,
                    "human_rubric": candidate["hotspot_human_rubric"],
                }
            )
            _write_result(settings.output_dir, result)
            lifecycle.complete_stage(f"{case.case_id}:{model}:{stage_name}")

    failures = [
        stage
        for candidate in candidates
        for stage in candidate["stages"]  # type: ignore[union-attr]
        if stage["status"] == "failed"
    ]
    result["warnings"] = _collect_stage_warnings(result)
    result["status"] = "completed_with_failures" if failures else "success"
    result_path = _write_result(settings.output_dir, result)
    lifecycle.set_stage("reports")
    render_reports_checked(result_path)
    lifecycle.complete_stage("reports")
    return result_path


def run_e2e_evaluation(
    settings: E2EEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory = OllamaRuntime,
    mflux_factory: MfluxGeneratorFactory = MfluxGenerator,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Run the exact Ollama comparison against one fixed MFLUX model."""
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="e2e",
        settings=settings.model_dump(mode="json"),
        models={
            "ollama_candidates": list(E2E_OLLAMA_MODELS),
            "fixed_mflux": E2E_MFLUX_MODEL,
        },
        contracts={
            "image_prompt": {
                "version": IMAGE_PROMPT_VERSION,
                "sha256": contract_digest(
                    IMAGE_PROMPT_VERSION,
                    inspect.getsource(compose_image_prompt),
                ),
            },
            "hotspot_prompt": {
                "version": HOTSPOT_REMAP_PROMPT_VERSION,
                "sha256": contract_digest(
                    HOTSPOT_REMAP_PROMPT_VERSION,
                    inspect.getsource(build_hotspot_remap_prompt),
                ),
            },
            "hotspot_schema": {
                "version": HOTSPOT_REMAP_SCHEMA_VERSION,
                "sha256": contract_digest(
                    HotspotRemapModelOutput.model_json_schema(),
                    inspect.getsource(build_hotspot_remap_schema),
                ),
            },
            "case": {
                "version": E2E_CASE_VERSION,
                "sha256": None,
            },
        },
        environment_provider=environment_provider,
    )
    try:
        cases = load_e2e_cases(settings.case_dir)
        lifecycle.update_contract(
            "case",
            {
                "version": E2E_CASE_VERSION,
                "sha256": contract_digest([case.model_dump(mode="json") for case in cases]),
            },
        )
        result_path = _execute_e2e(
            settings,
            runtime_factory=runtime_factory,
            mflux_factory=mflux_factory,
            lifecycle=lifecycle,
            cases=cases,
        )
    except Exception as error:
        result_path = settings.output_dir / "e2e-results.json"
        report_failure = lifecycle.manifest["stage"] == "reports"
        failure = {
            "stage": lifecycle.manifest["stage"],
            "classification": ("report_rendering" if report_failure else classify_failure(error)),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        final_status = "completed_with_report_failure" if report_failure else "failed"
        warnings: list[str] = []
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = final_status
            result["failure"] = failure
            warnings = _collect_stage_warnings(result)
            result["warnings"] = warnings
            _write_result(settings.output_dir, result)
            if not report_failure:
                try:
                    render_reports(result_path)
                except Exception as report_error:
                    lifecycle.add_warning(f"failure report could not be rendered: {report_error}")
        lifecycle.finalize(
            status=final_status,
            failure=failure,
            warnings=warnings,
        )
        raise
    completed_result = json.loads(result_path.read_text(encoding="utf-8"))
    status = completed_result["status"]
    stage_failures = [
        {
            "case_id": candidate["case_id"],
            "ollama_model": candidate["ollama_model"],
            "stage": stage["name"],
            **stage["failure"],
        }
        for candidate in completed_result["candidates"]
        for stage in candidate["stages"]
        if stage["status"] == "failed"
    ]
    lifecycle.finalize(
        status=status,
        failure=(
            {"classification": "stage_failures", "stages": stage_failures}
            if stage_failures
            else None
        ),
        warnings=completed_result["warnings"],
    )
    return result_path
