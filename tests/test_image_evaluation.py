"""Tests for controlled image evaluation behind production-adapter fakes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from hypergen.evaluation.images import (
    DEFAULT_MFLUX_MODELS,
    DEFAULT_OLLAMA_MODELS,
    ImageEvaluationCase,
    ImageEvaluationSettings,
    load_image_cases,
    run_image_evaluation,
)
from hypergen.generation.mflux_generator import MfluxGenerator
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeGeneratedImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(path, format="PNG")


class FakeMfluxModel:
    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


class FakeOllamaClient:
    def __init__(self, model: str, *, available: bool = True) -> None:
        self.model = model
        self.available = available

    def list(self) -> SimpleNamespace:
        models = (SimpleNamespace(model=self.model),) if self.available else ()
        return SimpleNamespace(models=models)

    def generate(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(done=True)

    def chat(self, **kwargs: object) -> SimpleNamespace:
        content = json.dumps(
            {
                "render_prompt": f"Rendered by {self.model} with distinct gate and chest",
                "interactive_subjects": ["gate", "chest"],
            }
        )
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            total_duration=2_000_000,
            load_duration=500_000,
        )


def _write_case(case_dir: Path) -> None:
    case_dir.mkdir()
    (case_dir / "one.json").write_text(
        json.dumps(
            {
                "case_version": "image-case-v1",
                "case_id": "one",
                "inputs": {
                    "scene_description": "Courtyard",
                    "interaction_description": "Gate leads to garden",
                    "stack_art_direction": "Watercolor",
                    "card_style": None,
                },
                "fixed_render_prompt": "Watercolor courtyard with a distinct gate",
                "required_visual_elements": ["gate"],
                "unwanted_artifacts": ["text"],
            }
        )
    )


def test_tracked_image_cases_load_in_stable_order() -> None:
    cases = load_image_cases(Path("evals/cases/images"))

    assert [case.case_id for case in cases] == ["courtyard", "workshop"]
    assert cases[0].required_visual_elements
    assert cases[0].unwanted_artifacts


def test_image_case_id_is_safe_for_artifact_paths() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        ImageEvaluationCase.model_validate(
            {
                "case_id": "../escape",
                "inputs": {
                    "scene_description": "Courtyard",
                    "interaction_description": "Gate",
                    "stack_art_direction": "Watercolor",
                },
                "fixed_render_prompt": "Watercolor courtyard",
                "required_visual_elements": ["gate"],
            }
        )


def test_image_suite_keeps_model_axes_separate_and_writes_raw_artifacts(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)
    runtime_settings: list[OllamaSettings] = []

    def runtime_factory(settings: OllamaSettings) -> OllamaRuntime:
        runtime_settings.append(settings)
        return OllamaRuntime(settings, client=FakeOllamaClient(settings.model))

    result_path = run_image_evaluation(
        ImageEvaluationSettings(output_dir=tmp_path / "run", case_dir=case_dir),
        runtime_factory=runtime_factory,
        mflux_factory=lambda: MfluxGenerator(model_factory=lambda *_: FakeMfluxModel()),
    )

    result = json.loads(result_path.read_text())
    assert [settings.model for settings in runtime_settings] == list(DEFAULT_OLLAMA_MODELS)
    assert len(result["prompt_axis"]) == 3
    assert len(result["prompt_downstream_axis"]) == 3
    assert len(result["mflux_axis"]) == 2
    assert {row["model"] for row in result["mflux_axis"]} == set(DEFAULT_MFLUX_MODELS)
    assert all(row["mflux_model"] == "flux2-klein-4b" for row in result["prompt_downstream_axis"])
    assert result["ollama_effective_settings"]["qwen3.5:4b"]["think"] is False
    assert result["prompt_axis"][0]["cold_metrics"]["inference_duration_ns"] == 1_500_000
    assert result["mflux_axis"][0]["cold"]["load_duration_seconds"] >= 0
    assert result["mflux_axis"][0]["warm"]["inference_duration_seconds"] >= 0
    assert result["mflux_axis"][0]["warm"]["serialization_duration_seconds"] >= 0
    assert "hotspot_suitability" in result["mflux_axis"][0]["rubric"]
    assert len(list((tmp_path / "run" / "raw" / "prompts").rglob("*.json"))) == 6
    assert len(list((tmp_path / "run" / "images" / "prompt-axis").rglob("*.png"))) == 3
    assert len(list((tmp_path / "run" / "images" / "mflux-axis").rglob("*.png"))) == 4
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    prompt_row = next(row for row in summary["rows"] if row["axis"] == "ollama_prompt")
    assert prompt_row["derived_prompt"].startswith("Rendered by qwen3.5:")
    assert prompt_row["interactive_subjects"] == ["gate", "chest"]


def test_image_suite_isolates_candidate_failure_and_runs_mflux_axis(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)

    def runtime_factory(settings: OllamaSettings) -> OllamaRuntime:
        return OllamaRuntime(
            settings,
            client=FakeOllamaClient(
                settings.model,
                available=settings.model != "qwen3.5:9b",
            ),
        )

    result_path = run_image_evaluation(
        ImageEvaluationSettings(output_dir=tmp_path / "run", case_dir=case_dir),
        runtime_factory=runtime_factory,
        mflux_factory=lambda: MfluxGenerator(model_factory=lambda *_: FakeMfluxModel()),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "completed_with_failures"
    assert {row["model"] for row in result["prompt_axis"]} == set(DEFAULT_OLLAMA_MODELS)
    failed = next(row for row in result["prompt_axis"] if row["model"] == "qwen3.5:9b")
    assert failed["cold"]["failure"]["classification"] == "transport_or_service"
    assert len(result["mflux_axis"]) == 2
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"
    assert not any("qwen3.5:9b" in stage for stage in manifest["completed_stages"])
    assert (tmp_path / "run" / "report.html").is_file()
    assert (tmp_path / "run" / "manifest.json").is_file()
    assert (tmp_path / "run" / "summary.csv").is_file()
    assert (tmp_path / "run" / "report.html").is_file()
    assert (tmp_path / "run" / "contact-sheet.png").is_file()
