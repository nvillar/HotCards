"""Tests for controlled image evaluation behind production-adapter fakes."""

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from hypergen.evaluation.images import (
    DEFAULT_MFLUX_MODELS,
    ImageEvaluationCase,
    ImageEvaluationSettings,
    load_image_cases,
    run_image_evaluation,
)
from hypergen.generation.mflux_generator import MfluxGenerator


class FakeGeneratedImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(path, format="PNG")


class FakeMfluxModel:
    def __init__(
        self,
        requests: list[dict[str, object]],
        *,
        fail: bool = False,
    ) -> None:
        self.requests = requests
        self.fail = fail

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError("candidate failed")
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


def _write_case(case_dir: Path) -> None:
    case_dir.mkdir()
    (case_dir / "one.json").write_text(
        json.dumps(
            {
                "case_version": "image-case-v2",
                "case_id": "one",
                "inputs": {
                    "description": "Courtyard",
                    "style_name": "Storybook",
                    "style_prompt": "Watercolor",
                },
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
                "case_version": "image-case-v2",
                "case_id": "../escape",
                "inputs": {
                    "description": "Courtyard",
                    "style_name": "Storybook",
                    "style_prompt": "Watercolor",
                },
                "required_visual_elements": ["gate"],
            }
        )


def test_image_suite_uses_deterministic_prompts_for_mflux_candidates(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)
    requests: list[dict[str, object]] = []

    result_path = run_image_evaluation(
        ImageEvaluationSettings(output_dir=tmp_path / "run", case_dir=case_dir),
        mflux_factory=lambda: MfluxGenerator(model_factory=lambda *_: FakeMfluxModel(requests)),
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "success"
    assert "prompt_axis" not in result
    assert "prompt_downstream_axis" not in result
    assert len(result["mflux_axis"]) == 2
    assert {row["model"] for row in result["mflux_axis"]} == set(DEFAULT_MFLUX_MODELS)
    assert {row["render_prompt"] for row in result["mflux_axis"]} == {"Courtyard\n\nWatercolor"}
    assert {request["prompt"] for request in requests} == {"Courtyard\n\nWatercolor"}
    assert result["mflux_axis"][0]["cold"]["metadata"]["render_prompt"] == (
        "Courtyard\n\nWatercolor"
    )
    assert result["mflux_axis"][0]["warm"]["inference_duration_seconds"] >= 0
    assert "hotspot_suitability" in result["mflux_axis"][0]["rubric"]
    assert len(list((tmp_path / "run" / "images" / "mflux-axis").rglob("*.png"))) == 4
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert {row["axis"] for row in summary["rows"]} == {"mflux"}
    assert {row["render_prompt"] for row in summary["rows"]} == {"Courtyard\n\nWatercolor"}


def test_image_suite_isolates_mflux_candidate_failure(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)

    def mflux_factory() -> MfluxGenerator:
        return MfluxGenerator(
            model_factory=lambda model, _quantization: FakeMfluxModel(
                [],
                fail=model == "flux2-klein-9b",
            )
        )

    result_path = run_image_evaluation(
        ImageEvaluationSettings(output_dir=tmp_path / "run", case_dir=case_dir),
        mflux_factory=mflux_factory,
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "completed_with_failures"
    assert len(result["mflux_axis"]) == 2
    failed = next(row for row in result["mflux_axis"] if row["model"] == "flux2-klein-9b")
    assert failed["cold"]["failure"]["classification"] == "image_generation"
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"
    assert not any("flux2-klein-9b" in stage for stage in manifest["completed_stages"])
    assert (tmp_path / "run" / "summary.csv").is_file()
    assert (tmp_path / "run" / "report.html").is_file()
    assert (tmp_path / "run" / "contact-sheet.png").is_file()
