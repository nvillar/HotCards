"""Tests for the proposed built-in Style screening suite."""

import json
from pathlib import Path

import pytest
from PIL import Image

from hotcards.domain.image_dimensions import AspectRatio, GenerateResolution
from hotcards.evaluation.cli import build_parser, run_cli
from hotcards.evaluation.style_presets import (
    DEFAULT_STYLE_PRESET_EXPERIMENT,
    StylePresetSettings,
    compose_style_preset_prompt,
    load_style_preset_experiment,
    run_style_preset_evaluation,
)
from hotcards.generation.errors import ImageGenerationError
from hotcards.generation.mflux_generator import MfluxGenerator


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
        fail_prompt_contains: str | None = None,
    ) -> None:
        self.requests = requests
        self.fail_prompt_contains = fail_prompt_contains

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.requests.append(kwargs)
        if self.fail_prompt_contains is not None and self.fail_prompt_contains in str(
            kwargs["prompt"]
        ):
            raise RuntimeError("candidate failed")
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


def test_tracked_style_experiment_has_control_ten_styles_and_two_scenes() -> None:
    experiment = load_style_preset_experiment()

    assert experiment.styles[0].style_id == "no-style"
    assert experiment.styles[0].prompt_text is None
    assert len(experiment.styles) == 11
    assert [scene.case_id for scene in experiment.scenes] == [
        "island-lighthouse",
        "clockmaker-workshop",
    ]
    assert all(style.expected_characteristics for style in experiment.styles[1:])


def test_style_prompt_composition_preserves_control_and_appends_treatment() -> None:
    assert compose_style_preset_prompt("A clear scene.", None) == "A clear scene."
    assert compose_style_preset_prompt("A clear scene", "Rendered in ink.") == (
        "A clear scene.\n\nRendered in ink."
    )


def test_style_settings_reject_obsolete_arbitrary_dimensions(
    tmp_path: Path,
) -> None:
    settings = StylePresetSettings(output_dir=tmp_path / "default")
    assert settings.resolution is GenerateResolution.RESOLUTION_1024
    assert settings.aspect_ratio is AspectRatio.LANDSCAPE
    with pytest.raises(ValueError, match="width"):
        StylePresetSettings(
            output_dir=tmp_path / "run",
            width=592,  # type: ignore[call-arg]
        )


def test_style_suite_renders_complete_matrix_and_review_sheets(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_style_preset_evaluation(
        StylePresetSettings(
            output_dir=output_dir,
            resolution=GenerateResolution.RESOLUTION_256,
            aspect_ratio=AspectRatio.LANDSCAPE,
        ),
        generator_factory=lambda: MfluxGenerator(model_factory=lambda *_: FakeMfluxModel(requests)),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert len(result["results"]) == 22
    assert len(requests) == 22
    assert {request["seed"] for request in requests} == {42}
    assert {
        (request["width"], request["height"]) for request in requests
    } == {(288, 224)}
    assert all(not request.get("image_paths") for request in requests)
    assert len(list((output_dir / "outputs").rglob("*.png"))) == 22
    assert len(list((output_dir / "review").glob("scene-*.png"))) == 2
    assert len(list((output_dir / "review").glob("style-*.png"))) == 11
    assert (output_dir / "review" / "index.html").is_file()
    review_html = (output_dir / "review" / "index.html").read_text(encoding="utf-8")
    assert "../outputs/island-lighthouse/seed-42/hypercard.png" in review_html

    control_records = [row for row in result["results"] if row["style_id"] == "no-style"]
    assert all(
        row["render_prompt"]
        == next(
            scene["description"]
            for scene in result["experiment"]["scenes"]
            if scene["case_id"] == row["case_id"]
        )
        for row in control_records
    )
    styled_record = next(row for row in result["results"] if row["style_id"] == "hypercard")
    assert "\n\nRendered in an early Macintosh HyperCard" in styled_record["render_prompt"]
    scorecard = json.loads((output_dir / "review" / "scorecard.json").read_text(encoding="utf-8"))
    assert len(scorecard["outputs"]) == 22
    assert all(len(output["criteria"]) == 4 for output in scorecard["outputs"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"


def test_style_suite_marks_empty_style_sheet_unavailable(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    result_path = run_style_preset_evaluation(
        StylePresetSettings(
            output_dir=output_dir,
            resolution=GenerateResolution.RESOLUTION_256,
            aspect_ratio=AspectRatio.LANDSCAPE,
        ),
        generator_factory=lambda: MfluxGenerator(
            model_factory=lambda *_: FakeMfluxModel(
                [],
                fail_prompt_contains="early Macintosh HyperCard",
            )
        ),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed_with_failures"
    assert not (output_dir / "review" / "style-hypercard.png").exists()
    review_html = (output_dir / "review" / "index.html").read_text(encoding="utf-8")
    assert 'href="style-hypercard.png"' not in review_html
    assert "No successful outputs were available for this group." in review_html


def test_style_suite_fails_when_no_image_can_be_generated(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    with pytest.raises(ImageGenerationError, match="produced no images"):
        run_style_preset_evaluation(
            StylePresetSettings(
                output_dir=output_dir,
                resolution=GenerateResolution.RESOLUTION_256,
                aspect_ratio=AspectRatio.LANDSCAPE,
            ),
            generator_factory=lambda: MfluxGenerator(
                model_factory=lambda *_: FakeMfluxModel(
                    [],
                    fail_prompt_contains="",
                )
            ),
            environment_provider=lambda: {"git_sha": "test"},
        )

    result = json.loads((output_dir / "style-preset-results.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_cli_exposes_style_preset_options() -> None:
    args = build_parser().parse_args(
        [
            "style-presets",
            "--mflux-model",
            "flux2-klein-4b",
            "--seed",
            "11",
            "--seed",
            "12",
            "--resolution",
            "768",
            "--aspect-ratio",
            "16:9",
        ]
    )

    assert args.command == "style-presets"
    assert args.mflux_model == "flux2-klein-4b"
    assert args.seeds == [11, 12]
    assert args.resolution is GenerateResolution.RESOLUTION_768
    assert args.aspect_ratio is AspectRatio.WIDESCREEN


def test_cli_validates_style_experiment_without_model_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_cli(["style-presets", "--validate-only"]) == 0
    assert capsys.readouterr().out.strip() == str(DEFAULT_STYLE_PRESET_EXPERIMENT)
