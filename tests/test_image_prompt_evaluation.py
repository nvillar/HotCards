"""Tests for the maintained Image Prompt benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from hotcards.evaluation.cli import build_parser, run_cli
from hotcards.evaluation.image_prompts import (
    ImagePromptBenchmark,
    ImagePromptBenchmarkSettings,
    load_image_prompt_benchmark,
    run_image_prompt_benchmark,
)
from hotcards.generation.errors import ModelResponseError
from hotcards.generation.image_prompt_preparation import (
    ImagePromptPreparationAttempt,
    ImagePromptPreparationResult,
)
from hotcards.generation.ollama_client import OllamaSettings


def _criterion(
    criterion_id: str,
    *,
    category: str = "target_fidelity",
) -> dict[str, str]:
    return {
        "criterion_id": criterion_id,
        "category": category,
        "severity": "critical",
        "description": f"Observable requirement for {criterion_id}.",
    }


def _write_benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "benchmark"
    references = root / "references"
    references.mkdir(parents=True)
    image = references / "reference.png"
    Image.new("RGB", (8, 6), "navy").save(image, format="PNG")
    payload = {
        "benchmark_version": "image-prompt-benchmark-v2",
        "benchmark_id": "test-benchmark",
        "title": "Test benchmark",
        "description": "A self-contained benchmark fixture.",
        "common_criteria": [
            _criterion("standalone", category="prompt_quality")
        ],
        "assets": [
            {
                "asset_id": "reference",
                "path": "references/reference.png",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "provenance": {
                    "source_kind": "project-generated",
                    "description": "Generated for this test.",
                    "reuse_terms": "Reusable with HotCards.",
                },
            }
        ],
        "cases": [
            {
                "case_id": "referenced",
                "title": "Referenced case",
                "cohort": "development",
                "tags": ["continuity"],
                "description": "The panel is open.",
                "reference": {
                    "asset_id": "reference",
                    "generation_description": "A closed circular panel.",
                },
                "criteria": [_criterion("panel-open")],
                "provenance": {
                    "source_kind": "authored",
                    "description": "Written for this test.",
                    "reuse_terms": "Reusable with HotCards.",
                },
            },
            {
                "case_id": "text-only",
                "title": "Text-only case",
                "cohort": "regression",
                "tags": ["no-reference"],
                "description": "A moonlit courtyard.",
                "reference": None,
                "criteria": [_criterion("courtyard")],
                "provenance": {
                    "source_kind": "authored",
                    "description": "Written for this test.",
                    "reuse_terms": "Reusable with HotCards.",
                },
            },
        ],
    }
    path = root / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FakePreparer:
    def __init__(
        self,
        settings: OllamaSettings,
        calls: list[dict[str, object]],
    ) -> None:
        self.settings = settings
        self.calls = calls

    def prepare(
        self,
        request: object,
        *,
        reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult:
        self.calls.append(
            {
                "request": request,
                "reference_image_path": reference_image_path,
                "model": self.settings.model,
            }
        )
        return ImagePromptPreparationResult(
            image_prompt=f"Prepared {request.description}",  # type: ignore[attr-defined]
            raw_response='{"image_prompt": "fixture"}',
            model_identifier=self.settings.model,
            prompt_version=request.prompt_version,  # type: ignore[attr-defined]
            duration_seconds=0.25,
        )


def test_tracked_image_prompt_benchmark_is_self_contained() -> None:
    path = Path("evals/cases/image_prompts/benchmark.json")
    benchmark = load_image_prompt_benchmark(path)

    assert benchmark.benchmark_id == "hotcards-image-prompts"
    assert {case.case_id for case in benchmark.cases} == {
        "closed-hatch-text-only",
        "open-hatch-state-change",
        "new-setting-style-transfer",
        "window-view-reframe",
        "computer-closeup",
        "computer-front-viewpoint",
        "computer-error-screen",
        "computer-exit-symbol",
        "laboratory-hidden-entry",
        "stairs-through-panel",
        "style-only-forest",
        "watercolor-style-override",
        "local-color-override",
        "exact-visible-text",
        "cut-paper-city-style-only",
        "blue-cabinet-open",
    }
    assert {case.cohort for case in benchmark.cases} == {
        "development",
        "regression",
        "edge",
    }
    assert any(case.reference is None for case in benchmark.cases)
    assert all(case.criteria for case in benchmark.cases)
    assert all(asset.provenance.source_kind == "project-generated" for asset in benchmark.assets)


def test_benchmark_rejects_path_traversal_and_unknown_assets() -> None:
    with pytest.raises(ValidationError, match="safe relative data"):
        ImagePromptBenchmark.model_validate(
            {
                "benchmark_version": "image-prompt-benchmark-v2",
                "benchmark_id": "bad",
                "title": "Bad benchmark",
                "description": "Invalid path fixture.",
                "common_criteria": (_criterion("common"),),
                "assets": (
                    {
                        "asset_id": "bad",
                        "path": "../outside.png",
                        "sha256": "0" * 64,
                        "provenance": {
                            "source_kind": "authored",
                            "description": "Fixture.",
                            "reuse_terms": "Fixture.",
                        },
                    },
                ),
                "cases": (
                    {
                        "case_id": "one",
                        "title": "One",
                        "cohort": "edge",
                        "tags": ("one",),
                        "description": "One case.",
                        "criteria": (_criterion("case"),),
                        "provenance": {
                            "source_kind": "authored",
                            "description": "Fixture.",
                            "reuse_terms": "Fixture.",
                        },
                    },
                ),
            }
        )


def test_benchmark_loader_rejects_changed_frozen_asset(
    tmp_path: Path,
) -> None:
    path = _write_benchmark(tmp_path)
    benchmark = load_image_prompt_benchmark(path)
    asset_path = path.parent / benchmark.assets[0].path
    asset_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_image_prompt_benchmark(path)


def test_image_prompt_runner_records_outputs_and_blank_scorecards(
    tmp_path: Path,
) -> None:
    benchmark_path = _write_benchmark(tmp_path)
    calls: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_image_prompt_benchmark(
        ImagePromptBenchmarkSettings(
            output_dir=output_dir,
            benchmark_path=benchmark_path,
            ollama_models=("small", "large"),
            repetitions=2,
        ),
        preparer_factory=lambda settings: FakePreparer(settings, calls),
        candidate_contract={"version": "candidate-v1", "sha256": "fixture"},
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert len(result["results"]) == 8
    assert len(calls) == 8
    referenced_calls = [
        call for call in calls if call["reference_image_path"] is not None
    ]
    assert len(referenced_calls) == 4
    assert all(
        row["rubric"]["hard_failure"] is None for row in result["results"]
    )
    assert all(
        criterion["score"] is None
        for row in result["results"]
        for criterion in row["rubric"]["criteria"]
    )
    assert len(list((output_dir / "raw").rglob("*.txt"))) == 8
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "report.html").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["contracts"]["image_prompt_candidate"]["version"] == (
        "candidate-v1"
    )


def test_image_prompt_runner_isolates_candidate_failure(
    tmp_path: Path,
) -> None:
    benchmark_path = _write_benchmark(tmp_path)
    output_dir = tmp_path / "run"

    class PartlyFailingPreparer(FakePreparer):
        def prepare(
            self,
            request: object,
            *,
            reference_image_path: Path | None = None,
        ) -> ImagePromptPreparationResult:
            if "moonlit" in request.description:  # type: ignore[attr-defined]
                raise ModelResponseError(
                    "candidate failed",
                    raw_response='{"invalid": true}',
                )
            return super().prepare(
                request,
                reference_image_path=reference_image_path,
            )

    result_path = run_image_prompt_benchmark(
        ImagePromptBenchmarkSettings(
            output_dir=output_dir,
            benchmark_path=benchmark_path,
        ),
        preparer_factory=lambda settings: PartlyFailingPreparer(settings, []),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed_with_failures"
    assert [row["status"] for row in result["results"]] == [
        "success",
        "failed",
    ]
    assert result["results"][1]["failure"]["classification"] == (
        "structured_output_validation"
    )
    assert len(result["results"][1]["raw_response_paths"]) == 1
    assert (
        output_dir / result["results"][1]["raw_response_paths"][0]
    ).is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"


def test_runner_distinguishes_multistage_failure_from_repair(
    tmp_path: Path,
) -> None:
    benchmark_path = _write_benchmark(tmp_path)
    output_dir = tmp_path / "run"

    class TwoStageFailingPreparer(FakePreparer):
        def prepare(
            self,
            request: object,
            *,
            reference_image_path: Path | None = None,
        ) -> ImagePromptPreparationResult:
            raise ModelResponseError(
                "synthesis failed",
                raw_response='{"invalid": true}',
                response_attempts=(
                    {
                        "phase": "reference_account",
                        "raw_response": '{"reference_account": "fixture"}',
                    },
                    {
                        "phase": "synthesis",
                        "raw_response": '{"invalid": true}',
                    },
                ),
            )

    result_path = run_image_prompt_benchmark(
        ImagePromptBenchmarkSettings(
            output_dir=output_dir,
            benchmark_path=benchmark_path,
        ),
        preparer_factory=lambda settings: TwoStageFailingPreparer(
            settings,
            [],
        ),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["results"][0]["attempt_count"] == 2
    assert not result["results"][0]["repair_attempted"]


def test_runner_retains_each_repair_attempt_with_safe_model_paths(
    tmp_path: Path,
) -> None:
    benchmark_path = _write_benchmark(tmp_path)
    output_dir = tmp_path / "run"

    class RepairingPreparer(FakePreparer):
        def prepare(
            self,
            request: object,
            *,
            reference_image_path: Path | None = None,
        ) -> ImagePromptPreparationResult:
            return ImagePromptPreparationResult(
                image_prompt="A repaired result.",
                raw_response='{"image_prompt": "A repaired result."}',
                model_identifier=self.settings.model,
                prompt_version=request.prompt_version,  # type: ignore[attr-defined]
                duration_seconds=0.5,
                repair_applied=True,
                attempts=(
                    ImagePromptPreparationAttempt(
                        phase="initial",
                        raw_response='{"image_prompt": "invalid"}',
                        duration_seconds=0.2,
                    ),
                    ImagePromptPreparationAttempt(
                        phase="repair",
                        raw_response='{"image_prompt": "A repaired result."}',
                        duration_seconds=0.3,
                    ),
                ),
            )

    result_path = run_image_prompt_benchmark(
        ImagePromptBenchmarkSettings(
            output_dir=output_dir,
            benchmark_path=benchmark_path,
            ollama_models=("..",),
        ),
        preparer_factory=lambda settings: RepairingPreparer(settings, []),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    first = result["results"][0]
    assert first["attempt_count"] == 2
    assert first["repair_applied"]
    assert all(path.startswith("raw/model-") for path in first["raw_response_paths"])
    assert all((output_dir / path).is_file() for path in first["raw_response_paths"])
    assert not (output_dir / "referenced-repetition-1-attempt-1-initial.txt").exists()


def test_benchmark_settings_reject_duplicate_models(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        ImagePromptBenchmarkSettings(
            output_dir=tmp_path / "run",
            ollama_models=("same", "same"),
        )


def test_cli_exposes_image_prompt_benchmark_options() -> None:
    args = build_parser().parse_args(
        [
            "image-prompts",
            "--ollama-model",
            "small",
            "--ollama-model",
            "large",
            "--repetitions",
            "3",
        ]
    )

    assert args.command == "image-prompts"
    assert args.ollama_models == ["small", "large"]
    assert args.repetitions == 3


def test_cli_can_validate_benchmark_without_model_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_path = _write_benchmark(tmp_path)

    assert (
        run_cli(
            [
                "image-prompts",
                "--benchmark",
                str(benchmark_path),
                "--validate-only",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(benchmark_path)
