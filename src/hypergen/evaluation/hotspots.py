"""Controlled multimodal hotspot evaluation across Ollama candidates."""

from __future__ import annotations

import inspect
import json
import re
import shutil
from collections import Counter
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
    HOTSPOT_PROMPT_VERSION,
    HOTSPOT_SCHEMA_VERSION,
    MAX_COMPONENTS_PER_INTERACTION,
    MAX_INTERACTIONS,
    MAX_POINTS_PER_COMPONENT,
    CardCatalogueEntry,
    ExistingCandidateTarget,
    HotspotGenerationRequest,
    HotspotGenerationResult,
    HotspotModelOutput,
    HotspotProposal,
    OllamaHotspotGenerator,
    build_hotspot_prompt,
    build_hotspot_response_schema,
    normalize_hotspot_response,
)
from hypergen.generation.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaCallResult,
    OllamaRuntime,
    OllamaSettings,
)
from hypergen.generation.structured_output import structured_json_content

HOTSPOT_CASE_VERSION = "hotspot-case-v1"
HOTSPOT_RESULT_VERSION = "hotspot-result-v1"
DEFAULT_OLLAMA_MODELS = ("qwen3.5:4b", DEFAULT_OLLAMA_MODEL, "qwen3.6:35b")
HUMAN_RUBRIC_FIELDS = (
    "missing_hotspots",
    "invented_hotspots",
    "destination_resolution",
    "component_quality",
    "edit_cost",
)


class ExpectedHotspot(DomainModel):
    """Expected semantic destination used for deterministic scoring."""

    label: NonEmptyString
    target_token: NonEmptyString


class HotspotEvaluationCase(DomainModel):
    """One frozen image and expected hotspot behavior."""

    case_version: Literal["hotspot-case-v1"] = HOTSPOT_CASE_VERSION
    case_id: SafeCaseId
    image_path: NonEmptyString
    interaction_description: NonEmptyString
    card_catalogue: tuple[CardCatalogueEntry, ...]
    expected_hotspots: tuple[ExpectedHotspot, ...] = Field(min_length=1)
    regression_tags: tuple[NonEmptyString, ...] = Field(default_factory=tuple)


class HotspotEvaluationSettings(DomainModel):
    """Identical production contract settings for a model comparison."""

    output_dir: Path
    case_dir: Path = Path("evals/cases/hotspots")
    fixture_root: Path = Path("evals/cases")
    ollama_endpoint: NonEmptyString = "http://localhost:11434"
    ollama_models: tuple[NonEmptyString, ...] = DEFAULT_OLLAMA_MODELS
    coordinate_extent: PositiveInt = 1000
    ollama_timeout_seconds: float = Field(default=120.0, gt=0.0, allow_inf_nan=False)
    ollama_num_predict: PositiveInt = 1024
    ollama_context_length: PositiveInt = 8192
    include_ablations: bool = True
    ablation_model: NonEmptyString = DEFAULT_OLLAMA_MODEL


OllamaRuntimeFactory = Callable[[OllamaSettings], OllamaRuntime]


def default_hotspot_output_dir() -> Path:
    """Return a unique output directory for a live hotspot-suite run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"hotspots-{timestamp}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def load_hotspot_cases(
    case_dir: Path,
    *,
    fixture_root: Path,
) -> tuple[HotspotEvaluationCase, ...]:
    """Load cases and verify every frozen image remains inside the fixture root."""
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
            raise ValueError(f"hotspot case {case.case_id!r} image escapes fixture root") from error
        if not image_path.is_file():
            raise ValueError(f"hotspot case {case.case_id!r} image does not exist: {image_path}")
    return cases


def _image_path(case: HotspotEvaluationCase, settings: HotspotEvaluationSettings) -> Path:
    return settings.fixture_root / case.image_path


def _request(
    case: HotspotEvaluationCase,
    settings: HotspotEvaluationSettings,
    *,
    image_path: Path | None = None,
) -> HotspotGenerationRequest:
    return HotspotGenerationRequest(
        image_path=image_path or _image_path(case, settings),
        interaction_description=case.interaction_description,
        card_catalogue=case.card_catalogue,
        coordinate_extent=settings.coordinate_extent,
    )


def _geometry_metrics(
    raw: HotspotModelOutput,
    result: HotspotGenerationResult,
    *,
    extent: int,
) -> dict[str, int | float | None]:
    raw_polygons = [polygon for interaction in raw.interactions for polygon in interaction.polygons]
    before_valid = sum(
        not validate_polygon(
            tuple((point.x / extent, point.y / extent) for point in polygon.points)
        )
        for polygon in raw_polygons
    )
    cleaned_polygons = [polygon for proposal in result.proposals for polygon in proposal.polygons]
    after_valid = sum(not validate_polygon(polygon.points) for polygon in cleaned_polygons)
    return {
        "polygon_count": len(raw_polygons),
        "valid_before_cleanup": before_valid,
        "valid_after_cleanup": after_valid,
        "validity_rate_before_cleanup": (
            before_valid / len(raw_polygons) if raw_polygons else None
        ),
        "validity_rate_after_cleanup": (
            after_valid / len(cleaned_polygons) if cleaned_polygons else None
        ),
    }


def _destination_metrics(
    case: HotspotEvaluationCase,
    result: HotspotGenerationResult,
) -> dict[str, int | float]:
    expected = {
        index: item.target_token for index, item in enumerate(case.expected_hotspots, start=1)
    }
    proposals_by_index: dict[int, list[HotspotProposal]] = {}
    for proposal in result.proposals:
        proposals_by_index.setdefault(proposal.source_interaction_index, []).append(proposal)
    correct = sum(
        1
        for index, token in expected.items()
        if len(proposals_by_index.get(index, ())) == 1
        and (proposal := proposals_by_index[index][0]) is not None
        and isinstance(proposal.target, ExistingCandidateTarget)
        and proposal.target.card_token == token
    )
    expected_count = len(expected)
    known_indexes = expected.keys() & proposals_by_index.keys()
    return {
        "expected_count": expected_count,
        "proposal_count": len(result.proposals),
        "correct_destination_count": correct,
        "destination_accuracy": correct / expected_count,
        "missing_interaction_count": len(expected.keys() - proposals_by_index.keys()),
        "invented_interaction_count": len(proposals_by_index.keys() - expected.keys()),
        "ambiguous_duplicate_count": sum(
            len(proposals) - 1
            for index, proposals in proposals_by_index.items()
            if index in expected and len(proposals) > 1
        ),
        "unresolved_or_new_count": sum(
            not isinstance(proposal.target, ExistingCandidateTarget)
            for index in known_indexes
            for proposal in proposals_by_index[index]
        ),
    }


def _success_record(
    case: HotspotEvaluationCase,
    result: HotspotGenerationResult,
    *,
    extent: int,
) -> dict[str, object]:
    raw = HotspotModelOutput.model_validate_json(
        structured_json_content(result.raw_response)
    )
    source_indexes = [interaction.source_interaction_index for interaction in raw.interactions]
    duplicates = sum(count - 1 for count in Counter(source_indexes).values())
    total = len(source_indexes)
    return {
        "status": "success",
        "result": result.model_dump(mode="json", exclude={"raw_response"}),
        "structured_valid": True,
        "repetition": {
            "duplicate_count": duplicates,
            "rate": duplicates / total if total else 0.0,
        },
        "schema_limits": {
            "interaction_limit_reached": len(raw.interactions) == MAX_INTERACTIONS,
            "component_limit_reached": any(
                len(interaction.polygons) == MAX_COMPONENTS_PER_INTERACTION
                for interaction in raw.interactions
            ),
            "point_limit_reached": any(
                len(polygon.points) == MAX_POINTS_PER_COMPONENT
                for interaction in raw.interactions
                for polygon in interaction.polygons
            ),
            "token_limit_reached": result.done_reason == "length",
        },
        "token_usage": {
            "prompt_tokens": result.prompt_eval_count,
            "output_tokens": result.eval_count,
        },
        "geometry": _geometry_metrics(raw, result, extent=extent),
        "destinations": _destination_metrics(case, result),
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
    raw_response = error.raw_response if isinstance(error, ModelResponseError) else None
    response_metadata = error.response_metadata if isinstance(error, ModelResponseError) else {}
    return {
        "status": "failed",
        "structured_valid": False,
        "error_type": type(error).__name__,
        "message": str(error),
        "timeout": _is_timeout(error) if isinstance(error, GenerationError) else False,
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
            "token_limit_reached": response_metadata.get("done_reason") == "length",
        },
        "human_rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
    }


def _run_phase(
    *,
    generator: OllamaHotspotGenerator,
    request: HotspotGenerationRequest,
    case: HotspotEvaluationCase,
    extent: int,
    raw_path: Path,
) -> dict[str, object]:
    try:
        result = generator.generate(request)
    except GenerationError as error:
        raw_response = error.raw_response if isinstance(error, ModelResponseError) else None
        if raw_response is not None:
            raw_path.write_text(raw_response, encoding="utf-8")
        return _failure_record(error)
    raw_path.write_text(result.raw_response, encoding="utf-8")
    return _success_record(case, result, extent=extent)


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


def _model_summary(rows: list[dict[str, object]], model: str) -> dict[str, object]:
    model_rows = [row for row in rows if row["model"] == model]
    phases = [phase for row in model_rows for phase in (row["cold"], row["warm"])]
    successes = [phase for phase in phases if phase["status"] == "success"]
    failures = [phase for phase in phases if phase["status"] == "failed"]

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
        "timeout_rate": sum(bool(phase.get("timeout")) for phase in failures) / len(phases)
        if phases
        else 0.0,
        "average_repetition_rate": average(("repetition", "rate")),
        "average_destination_accuracy": average(("destinations", "destination_accuracy")),
        "average_geometry_validity_before": average(("geometry", "validity_rate_before_cleanup")),
        "average_geometry_validity_after": average(("geometry", "validity_rate_after_cleanup")),
        "average_output_tokens": average(("token_usage", "output_tokens")),
        "interaction_limit_hits": sum(
            bool(phase["schema_limits"].get("interaction_limit_reached"))  # type: ignore[union-attr]
            for phase in phases
        ),
        "token_limit_hits": sum(
            bool(phase["schema_limits"].get("token_limit_reached"))  # type: ignore[union-attr]
            for phase in phases
        ),
    }


def _recorded_regressions(
    *,
    case: HotspotEvaluationCase,
    settings: HotspotEvaluationSettings,
) -> list[dict[str, object]]:
    regression_dir = settings.case_dir / "recorded"
    rows: list[dict[str, object]] = []
    request = _request(case, settings)
    paths = [
        path
        for path in sorted(regression_dir.glob("*.json"))
        if not path.name.endswith(".metadata.json")
    ]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        name = path.name.removesuffix(".response.json").removesuffix(".json")
        metadata_path = path.with_name(f"{name}.metadata.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        call = OllamaCallResult(
            content=content,
            elapsed_seconds=0.0,
            total_duration_ns=0,
            load_duration_ns=0,
            prompt_eval_count=metadata.get("prompt_eval_count", 0),
            eval_count=metadata.get("eval_count", 0),
            done_reason=metadata.get("done_reason", "stop"),
        )
        try:
            result = normalize_hotspot_response(
                request=request,
                call=call,
                model_identifier="recorded-regression",
            )
        except ModelResponseError as error:
            rows.append(
                {
                    "name": name,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "response_metadata": error.response_metadata,
                    "schema_limits": {
                        "token_limit_reached": (
                            error.response_metadata.get("done_reason") == "length"
                        )
                    },
                }
            )
            continue
        rows.append(
            {
                "name": name,
                **_success_record(
                    case,
                    result,
                    extent=settings.coordinate_extent,
                ),
            }
        )
    return rows


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
    warnings.extend(
        f"recorded_regression:{item['name']}: {warning}"
        for item in result.get("recorded_regressions", [])  # type: ignore[union-attr]
        for warning in item.get("warnings", [])
    )
    return sorted(set(warnings))


def _execute_hotspot_evaluation(
    settings: HotspotEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory,
    lifecycle: RunLifecycle,
) -> Path:
    """Execute identical frozen cases inside an initialized run."""
    cases = load_hotspot_cases(settings.case_dir, fixture_root=settings.fixture_root)
    lifecycle.update_contract(
        "hotspot_case",
        {
            "version": HOTSPOT_CASE_VERSION,
            "sha256": contract_digest([case.model_dump(mode="json") for case in cases]),
        },
    )
    raw_dir = settings.output_dir / "raw"
    raw_dir.mkdir()
    source_dir = settings.output_dir / "artifacts" / "sources"
    source_dir.mkdir(parents=True)
    serialized_cases: list[dict[str, object]] = []
    for case in cases:
        source_path = source_dir / f"{case.case_id}{_image_path(case, settings).suffix}"
        shutil.copyfile(_image_path(case, settings), source_path)
        serialized_case = case.model_dump(mode="json")
        serialized_case["artifact_image_path"] = source_path.relative_to(
            settings.output_dir
        ).as_posix()
        serialized_cases.append(serialized_case)
    rows: list[dict[str, object]] = []
    ollama_settings: dict[str, object] = {}
    ablations: list[dict[str, object]] = []
    result: dict[str, object] = {
        "result_version": HOTSPOT_RESULT_VERSION,
        "suite": "hotspots",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "contract": {
            "prompt_version": HOTSPOT_PROMPT_VERSION,
            "schema_version": HOTSPOT_SCHEMA_VERSION,
            "selected_limits": {
                "interactions": MAX_INTERACTIONS,
                "components_per_interaction": MAX_COMPONENTS_PER_INTERACTION,
                "points_per_component": MAX_POINTS_PER_COMPONENT,
                "output_tokens": settings.ollama_num_predict,
                "context_tokens": settings.ollama_context_length,
                "timeout_seconds": settings.ollama_timeout_seconds,
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
            for case in cases:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "model": model,
                        "cold": failure,
                        "warm": failure,
                    }
                )
            _write_result(settings.output_dir, result)
            continue
        generator = OllamaHotspotGenerator(runtime)
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
                    generator=generator,
                    request=request,
                    case=case,
                    extent=settings.coordinate_extent,
                    raw_path=case_raw_dir / f"{model_name}-cold.json",
                )
            warm = _run_phase(
                generator=generator,
                request=request,
                case=case,
                extent=settings.coordinate_extent,
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
        wording_case = case.model_copy(
            update={
                "interaction_description": (
                    "Create exactly two interactions. First, the arched gate leads to "
                    "Moonlit Garden. Second, the red travel chest opens Treasure Room."
                )
            }
        )
        grid_dir = settings.output_dir / "artifacts"
        grid_dir.mkdir(exist_ok=True)
        grid_path = _grid_image(
            _image_path(case, settings),
            grid_dir / f"{case.case_id}-grid.png",
        )
        for name, think, image_path, request_case in (
            ("grid", False, grid_path, case),
            ("thinking", True, _image_path(case, settings), case),
            ("wording", False, _image_path(case, settings), wording_case),
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
                    generator=OllamaHotspotGenerator(runtime),
                    request=_request(request_case, settings, image_path=image_path),
                    case=case,
                    extent=settings.coordinate_extent,
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
                    row[phase]["status"] == "failed" for row in rows for phase in ("cold", "warm")
                )
                or any(
                    item["result"]["status"] == "failed"  # type: ignore[index]
                    for item in ablations
                )
                else "success"
            ),
            "summaries": [_model_summary(rows, model) for model in settings.ollama_models],
            "recorded_regressions": _recorded_regressions(
                case=cases[0],
                settings=settings,
            ),
            "contract_comparisons": {
                "image_input": "original model rows versus grid ablation",
                "thinking": "production false versus thinking ablation",
                "prompt_wording": "production prose versus explicit numbered wording ablation",
                "schema_and_token_behavior": (
                    "schema-saturation and supplied-token recorded regressions"
                ),
                "polygon_complexity": "selected limits plus invalid-geometry recorded regression",
                "geometry_cleanup": (
                    "validity before versus after deterministic coordinate clamping"
                ),
            },
        },
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
    """Run hotspot evaluation with a retained exception-safe manifest."""
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="hotspots",
        settings=settings.model_dump(mode="json"),
        models={"ollama_candidates": list(settings.ollama_models), "mflux": None},
        contracts={
            "hotspot_prompt": {
                "version": HOTSPOT_PROMPT_VERSION,
                "sha256": contract_digest(
                    HOTSPOT_PROMPT_VERSION,
                    inspect.getsource(build_hotspot_prompt),
                ),
            },
            "hotspot_schema": {
                "version": HOTSPOT_SCHEMA_VERSION,
                "sha256": contract_digest(
                    HotspotModelOutput.model_json_schema(),
                    inspect.getsource(build_hotspot_response_schema),
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
            warnings = _collect_result_warnings(result)
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
            "case_id": row["case_id"],
            "model": row["model"],
            "phase": phase,
            **row[phase]["failure"],
        }
        for row in completed_result["models"]
        for phase in ("cold", "warm")
        if row[phase]["status"] == "failed"
    ]
    stage_failures.extend(
        {
            "case_id": completed_result["cases"][0]["case_id"],
            "model": item["model"],
            "phase": f"ablation:{item['name']}",
            **item["result"]["failure"],
        }
        for item in completed_result["ablations"]
        if item["result"]["status"] == "failed"
    )
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
