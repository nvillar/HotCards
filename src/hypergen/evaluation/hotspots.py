"""Controlled multimodal evaluation of existing-hotspot remapping."""

from __future__ import annotations

import inspect
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from PIL import Image, ImageDraw
from pydantic import Field

from hypergen.domain.geometry import validate_polygon
from hypergen.domain.models import DomainModel, NonEmptyString, PositiveInt
from hypergen.evaluation.contracts import SafeCaseId
from hypergen.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hypergen.evaluation.reports import render_reports, render_reports_checked
from hypergen.generation.errors import GenerationError, ModelResponseError
from hypergen.generation.hotspot_prompts import (
    HOTSPOT_REMAP_PROMPT_VERSION,
    HOTSPOT_REMAP_SCHEMA_VERSION,
    MAX_COMPONENTS_PER_INTERACTION,
    MAX_INTERACTIONS_PER_CALL,
    MAX_POINTS_PER_COMPONENT,
    HotspotRemapModelOutput,
    HotspotRemapRequest,
    HotspotRemapResult,
    OllamaHotspotRemapper,
    RemapHotspotInput,
    build_hotspot_remap_prompt,
    build_hotspot_remap_schema,
)
from hypergen.generation.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaRuntime,
    OllamaSettings,
)

HOTSPOT_CASE_VERSION = "hotspot-remap-case-v1"
HOTSPOT_RESULT_VERSION = "hotspot-result-v2"
DEFAULT_OLLAMA_MODELS = ("qwen3.5:4b", DEFAULT_OLLAMA_MODEL, "qwen3.6:35b")
HUMAN_RUBRIC_FIELDS = (
    "missing_geometry",
    "misplaced_geometry",
    "component_quality",
    "edit_cost",
)


class HotspotEvaluationCase(DomainModel):
    """One frozen image and its existing hotspot subjects."""

    case_version: Literal["hotspot-remap-case-v1"] = HOTSPOT_CASE_VERSION
    case_id: SafeCaseId
    image_path: NonEmptyString
    hotspots: tuple[RemapHotspotInput, ...] = Field(min_length=1)
    regression_tags: tuple[NonEmptyString, ...] = Field(default_factory=tuple)


class HotspotEvaluationSettings(DomainModel):
    """Identical production remap settings for a model comparison."""

    output_dir: Path
    case_dir: Path = Path("evals/cases/hotspots")
    fixture_root: Path = Path("evals/cases")
    ollama_endpoint: NonEmptyString = "http://localhost:11434"
    ollama_models: tuple[NonEmptyString, ...] = DEFAULT_OLLAMA_MODELS
    coordinate_extent: PositiveInt = 1000
    ollama_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        allow_inf_nan=False,
    )
    ollama_num_predict: PositiveInt = 1024
    ollama_context_length: PositiveInt = 8192
    include_ablations: bool = True
    ablation_model: NonEmptyString = DEFAULT_OLLAMA_MODEL


OllamaRuntimeFactory = Callable[[OllamaSettings], OllamaRuntime]


def default_hotspot_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"hotspots-{timestamp}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def load_hotspot_cases(
    case_dir: Path,
    *,
    fixture_root: Path,
) -> tuple[HotspotEvaluationCase, ...]:
    """Load remap cases and constrain every image to the fixture root."""
    paths = sorted(case_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no hotspot evaluation cases found in {case_dir}")
    cases = tuple(
        HotspotEvaluationCase.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("hotspot evaluation case IDs must be unique")
    root = fixture_root.resolve()
    for case in cases:
        image_path = (fixture_root / case.image_path).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"hotspot case {case.case_id!r} image escapes fixture root"
            ) from error
        if not image_path.is_file():
            raise ValueError(
                f"hotspot case {case.case_id!r} image does not exist: {image_path}"
            )
    return cases


def _image_path(
    case: HotspotEvaluationCase,
    settings: HotspotEvaluationSettings,
) -> Path:
    return settings.fixture_root / case.image_path


def _request(
    case: HotspotEvaluationCase,
    settings: HotspotEvaluationSettings,
    *,
    image_path: Path | None = None,
) -> HotspotRemapRequest:
    return HotspotRemapRequest(
        image_path=image_path or _image_path(case, settings),
        hotspots=case.hotspots,
        coordinate_extent=settings.coordinate_extent,
    )


def _proposals(
    case: HotspotEvaluationCase,
    result: HotspotRemapResult,
) -> list[dict[str, object]]:
    labels = {hotspot.token: hotspot.label for hotspot in case.hotspots}
    return [
        {
            "token": token,
            "label": labels[token],
            "polygons": [
                polygon.model_dump(mode="json")
                for polygon in polygons
            ],
        }
        for token, polygons in result.polygons_by_token.items()
    ]


def _success_record(
    case: HotspotEvaluationCase,
    result: HotspotRemapResult,
) -> dict[str, object]:
    proposals = _proposals(case, result)
    polygons = [
        polygon
        for mapped in result.polygons_by_token.values()
        for polygon in mapped
    ]
    mapped_count = len(result.polygons_by_token)
    expected_count = len(case.hotspots)
    return {
        "status": "success",
        "result": {
            **result.model_dump(
                mode="json",
                exclude={"raw_responses"},
            ),
            "proposals": proposals,
        },
        "structured_valid": True,
        "repetition": {"duplicate_count": 0, "rate": 0.0},
        "schema_limits": {
            "batch_count": len(result.raw_responses),
            "component_limit_reached": any(
                len(mapped) == MAX_COMPONENTS_PER_INTERACTION
                for mapped in result.polygons_by_token.values()
            ),
            "point_limit_reached": any(
                len(polygon.points) == MAX_POINTS_PER_COMPONENT
                for polygon in polygons
            ),
            "token_limit_reached": "length" in result.done_reasons,
        },
        "token_usage": {
            "prompt_tokens": result.prompt_eval_count,
            "output_tokens": result.eval_count,
        },
        "geometry": {
            "polygon_count": len(polygons),
            "valid_after_cleanup": sum(
                not validate_polygon(polygon.points)
                for polygon in polygons
            ),
            "validity_rate_after_cleanup": (
                sum(
                    not validate_polygon(polygon.points)
                    for polygon in polygons
                )
                / len(polygons)
                if polygons
                else None
            ),
        },
        "remap": {
            "expected_count": expected_count,
            "mapped_count": mapped_count,
            "unlocated_count": len(result.unlocated),
            "mapping_rate": mapped_count / expected_count,
            "semantic_change_count": 0,
        },
        "warnings": result.warnings,
        "human_rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
    }


def _is_timeout(error: GenerationError) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, httpx.TimeoutException):
            return True
        cause = cause.__cause__
    return False


def _failure_record(error: BaseException) -> dict[str, object]:
    raw_response = (
        error.raw_response if isinstance(error, ModelResponseError) else None
    )
    response_metadata = (
        error.response_metadata
        if isinstance(error, ModelResponseError)
        else {}
    )
    return {
        "status": "failed",
        "structured_valid": False,
        "error_type": type(error).__name__,
        "message": str(error),
        "timeout": (
            _is_timeout(error)
            if isinstance(error, GenerationError)
            else False
        ),
        "failure": {
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        },
        "raw_response_retained": raw_response is not None,
        "response_metadata": response_metadata,
        "token_usage": {
            "prompt_tokens": response_metadata.get("prompt_eval_count"),
            "output_tokens": response_metadata.get("eval_count"),
        },
        "schema_limits": {
            "token_limit_reached": (
                response_metadata.get("done_reason") == "length"
            ),
        },
        "human_rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
    }


def _run_phase(
    *,
    remapper: OllamaHotspotRemapper,
    request: HotspotRemapRequest,
    case: HotspotEvaluationCase,
    raw_path: Path,
) -> dict[str, object]:
    try:
        result = remapper.remap(request)
    except GenerationError as error:
        if isinstance(error, ModelResponseError) and error.raw_response is not None:
            raw_path.write_text(error.raw_response, encoding="utf-8")
        return _failure_record(error)
    raw_path.write_text(
        json.dumps(list(result.raw_responses), indent=2),
        encoding="utf-8",
    )
    return _success_record(case, result)


def _grid_image(source_path: Path, output_path: Path) -> Path:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index in range(1, 10):
        x = round(image.width * index / 10)
        y = round(image.height * index / 10)
        draw.line((x, 0, x, image.height), fill=(255, 255, 255), width=1)
        draw.line((0, y, image.width, y), fill=(255, 255, 255), width=1)
    image.save(output_path, format="PNG")
    return output_path


def _model_summary(
    rows: list[dict[str, object]],
    model: str,
) -> dict[str, object]:
    model_rows = [row for row in rows if row["model"] == model]
    phases = [
        phase
        for row in model_rows
        for phase in (row["cold"], row["warm"])
    ]
    successes = [phase for phase in phases if phase["status"] == "success"]

    def average(path: tuple[str, ...]) -> float | None:
        values: list[float] = []
        for phase in successes:
            value: object = phase
            for key in path:
                value = value[key]  # type: ignore[index]
            if isinstance(value, (int, float)):
                values.append(float(value))
        return sum(values) / len(values) if values else None

    return {
        "model": model,
        "phase_count": len(phases),
        "success_rate": len(successes) / len(phases) if phases else 0.0,
        "average_mapping_rate": average(("remap", "mapping_rate")),
        "average_geometry_validity": average(
            ("geometry", "validity_rate_after_cleanup")
        ),
        "average_output_tokens": average(("token_usage", "output_tokens")),
    }


def _write_result(output_dir: Path, result: dict[str, object]) -> Path:
    result_path = output_dir / "hotspot-results.json"
    atomic_write_json(result_path, result)
    return result_path


def _collect_result_warnings(result: dict[str, object]) -> list[str]:
    warnings = [
        f"{row['case_id']}:{row['model']}:{phase}: {warning}"
        for row in result.get("models", [])  # type: ignore[union-attr]
        for phase in ("cold", "warm")
        for warning in row[phase].get("warnings", [])
    ]
    warnings.extend(
        f"ablation:{item['name']}:{item['model']}: {warning}"
        for item in result.get("ablations", [])  # type: ignore[union-attr]
        for warning in item["result"].get("warnings", [])
    )
    return sorted(set(warnings))


def _execute_hotspot_evaluation(
    settings: HotspotEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory,
    lifecycle: RunLifecycle,
) -> Path:
    cases = load_hotspot_cases(
        settings.case_dir,
        fixture_root=settings.fixture_root,
    )
    lifecycle.update_contract(
        "hotspot_case",
        {
            "version": HOTSPOT_CASE_VERSION,
            "sha256": contract_digest(
                [case.model_dump(mode="json") for case in cases]
            ),
        },
    )
    raw_dir = settings.output_dir / "raw"
    raw_dir.mkdir()
    source_dir = settings.output_dir / "artifacts" / "sources"
    source_dir.mkdir(parents=True)
    serialized_cases: list[dict[str, object]] = []
    for case in cases:
        source_path = (
            source_dir / f"{case.case_id}{_image_path(case, settings).suffix}"
        )
        shutil.copyfile(_image_path(case, settings), source_path)
        serialized = case.model_dump(mode="json")
        serialized["artifact_image_path"] = source_path.relative_to(
            settings.output_dir
        ).as_posix()
        serialized_cases.append(serialized)

    rows: list[dict[str, object]] = []
    ablations: list[dict[str, object]] = []
    ollama_settings: dict[str, object] = {}
    result: dict[str, object] = {
        "result_version": HOTSPOT_RESULT_VERSION,
        "suite": "hotspots",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "contract": {
            "prompt_version": HOTSPOT_REMAP_PROMPT_VERSION,
            "schema_version": HOTSPOT_REMAP_SCHEMA_VERSION,
            "selected_limits": {
                "interactions_per_call": MAX_INTERACTIONS_PER_CALL,
                "components_per_interaction": MAX_COMPONENTS_PER_INTERACTION,
                "points_per_component": MAX_POINTS_PER_COMPONENT,
            },
        },
        "ollama_effective_settings": ollama_settings,
        "cases": serialized_cases,
        "models": rows,
        "summaries": [],
        "ablations": ablations,
        "recorded_regressions": [],
        "warnings": [],
    }
    _write_result(settings.output_dir, result)

    for model in settings.ollama_models:
        lifecycle.set_stage(f"models:{model}:setup")
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
        ollama_settings[model] = asdict(runtime.settings)
        try:
            runtime.require_model(capabilities=frozenset({"vision"}))
        except Exception as error:
            failure = _failure_record(error)
            rows.extend(
                {
                    "case_id": case.case_id,
                    "model": model,
                    "cold": failure,
                    "warm": failure,
                }
                for case in cases
            )
            _write_result(settings.output_dir, result)
            continue
        remapper = OllamaHotspotRemapper(runtime)
        for case in cases:
            lifecycle.set_stage(f"models:{case.case_id}:{model}")
            case_raw_dir = raw_dir / case.case_id
            case_raw_dir.mkdir(exist_ok=True)
            request = _request(case, settings)
            model_name = _safe_name(model)
            try:
                runtime.unload_model()
            except Exception as error:
                cold = _failure_record(error)
            else:
                cold = _run_phase(
                    remapper=remapper,
                    request=request,
                    case=case,
                    raw_path=case_raw_dir / f"{model_name}-cold.json",
                )
            warm = _run_phase(
                remapper=remapper,
                request=request,
                case=case,
                raw_path=case_raw_dir / f"{model_name}-warm.json",
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "model": model,
                    "cold": cold,
                    "warm": warm,
                }
            )
            _write_result(settings.output_dir, result)
            if cold["status"] == "success" and warm["status"] == "success":
                lifecycle.complete_stage(f"models:{case.case_id}:{model}")

    if settings.include_ablations:
        lifecycle.set_stage("ablations")
        case = cases[0]
        artifact_dir = settings.output_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        grid_path = _grid_image(
            _image_path(case, settings),
            artifact_dir / f"{case.case_id}-grid.png",
        )
        for name, think, image_path in (
            ("grid", False, grid_path),
            ("thinking", True, _image_path(case, settings)),
        ):
            runtime = runtime_factory(
                OllamaSettings(
                    endpoint=settings.ollama_endpoint,
                    model=settings.ablation_model,
                    think=think,
                    temperature=0.0,
                    request_timeout_seconds=settings.ollama_timeout_seconds,
                    num_predict=settings.ollama_num_predict,
                    context_length=settings.ollama_context_length,
                )
            )
            try:
                runtime.require_model(capabilities=frozenset({"vision"}))
                runtime.unload_model()
            except Exception as error:
                ablation_result = _failure_record(error)
            else:
                ablation_result = _run_phase(
                    remapper=OllamaHotspotRemapper(runtime),
                    request=_request(case, settings, image_path=image_path),
                    case=case,
                    raw_path=raw_dir / f"ablation-{name}.json",
                )
            ablations.append(
                {
                    "name": name,
                    "model": settings.ablation_model,
                    "think": think,
                    "image_variant": "grid" if name == "grid" else "original",
                    "result": ablation_result,
                }
            )
            _write_result(settings.output_dir, result)
        if all(item["result"]["status"] == "success" for item in ablations):
            lifecycle.complete_stage("ablations")

    result.update(
        {
            "status": (
                "completed_with_failures"
                if any(
                    row[phase]["status"] == "failed"
                    for row in rows
                    for phase in ("cold", "warm")
                )
                or any(
                    item["result"]["status"] == "failed"  # type: ignore[index]
                    for item in ablations
                )
                else "success"
            ),
            "summaries": [
                _model_summary(rows, model)
                for model in settings.ollama_models
            ],
        }
    )
    result["warnings"] = _collect_result_warnings(result)
    result_path = _write_result(settings.output_dir, result)
    lifecycle.set_stage("reports")
    render_reports_checked(result_path)
    lifecycle.complete_stage("reports")
    return result_path


def run_hotspot_evaluation(
    settings: HotspotEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory = OllamaRuntime,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Run remap evaluation with a retained exception-safe manifest."""
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="hotspots",
        settings=settings.model_dump(mode="json"),
        models={
            "ollama_candidates": list(settings.ollama_models),
            "mflux": None,
        },
        contracts={
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
            "hotspot_case": {"version": HOTSPOT_CASE_VERSION},
        },
        environment_provider=environment_provider,
    )
    try:
        result_path = _execute_hotspot_evaluation(
            settings,
            runtime_factory=runtime_factory,
            lifecycle=lifecycle,
        )
    except Exception as error:
        result_path = settings.output_dir / "hotspot-results.json"
        report_failure = lifecycle.manifest["stage"] == "reports"
        failure = {
            "stage": lifecycle.manifest["stage"],
            "classification": (
                "report_rendering"
                if report_failure
                else classify_failure(error)
            ),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        final_status = (
            "completed_with_report_failure" if report_failure else "failed"
        )
        warnings: list[str] = []
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = final_status
            result["failure"] = failure
            warnings = _collect_result_warnings(result)
            result["warnings"] = warnings
            _write_result(settings.output_dir, result)
            if not report_failure:
                try:
                    render_reports(result_path)
                except Exception as report_error:
                    lifecycle.add_warning(
                        f"failure report could not be rendered: {report_error}"
                    )
        lifecycle.finalize(
            status=final_status,
            failure=failure,
            warnings=warnings,
        )
        raise

    completed = json.loads(result_path.read_text(encoding="utf-8"))
    stage_failures = [
        {
            "case_id": row["case_id"],
            "model": row["model"],
            "phase": phase,
            **row[phase]["failure"],
        }
        for row in completed["models"]
        for phase in ("cold", "warm")
        if row[phase]["status"] == "failed"
    ]
    stage_failures.extend(
        {
            "case_id": completed["cases"][0]["case_id"],
            "model": item["model"],
            "phase": f"ablation:{item['name']}",
            **item["result"]["failure"],
        }
        for item in completed["ablations"]
        if item["result"]["status"] == "failed"
    )
    lifecycle.finalize(
        status=completed["status"],
        failure=(
            {"classification": "stage_failures", "stages": stage_failures}
            if stage_failures
            else None
        ),
        warnings=completed["warnings"],
    )
    return result_path


__all__ = [
    "DEFAULT_OLLAMA_MODELS",
    "HOTSPOT_CASE_VERSION",
    "HOTSPOT_RESULT_VERSION",
    "HUMAN_RUBRIC_FIELDS",
    "HotspotEvaluationCase",
    "HotspotEvaluationSettings",
    "default_hotspot_output_dir",
    "load_hotspot_cases",
    "run_hotspot_evaluation",
]
