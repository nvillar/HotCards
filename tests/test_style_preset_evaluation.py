"""Tests for the proposed built-in Style screening suite."""

import json
from collections import Counter
from pathlib import Path

import pytest
from test_generation_smoke import FakeMfluxModel

import hotcards.evaluation.cli as cli_module
from hotcards.domain.image_dimensions import AspectRatio, ResolutionTier
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


@pytest.fixture
def small_experiment(tmp_path: Path) -> Path:
    experiment = load_style_preset_experiment()
    path = tmp_path / "small-experiment.json"
    path.write_text(
        experiment.model_copy(update={"styles": experiment.styles[:2]}).model_dump_json()
    )
    return path


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
    assert settings.tier is ResolutionTier.FULL
    assert settings.aspect_ratio is AspectRatio.LANDSCAPE
    with pytest.raises(ValueError, match="width"):
        StylePresetSettings(
            output_dir=tmp_path / "run",
            width=512,  # type: ignore[call-arg]
        )


def test_style_suite_renders_complete_matrix_and_review_sheets(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_style_preset_evaluation(
        StylePresetSettings(
            output_dir=output_dir,
            seeds=(42, 43),
            tier=ResolutionTier.SMALL,
            aspect_ratio=AspectRatio.LANDSCAPE,
        ),
        generator_factory=lambda: MfluxGenerator(model_factory=lambda *_: FakeMfluxModel(requests)),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    experiment = load_style_preset_experiment()
    expected_keys = {
        (scene.case_id, seed, style.style_id)
        for scene in experiment.scenes
        for seed in (42, 43)
        for style in experiment.styles
    }
    assert Counter(
        (row["case_id"], row["seed"], row["style_id"]) for row in result["results"]
    ) == Counter(dict.fromkeys(expected_keys, 1))
    assert len(requests) == len(expected_keys)
    assert {request["seed"] for request in requests} == {42, 43}
    assert {(request["width"], request["height"]) for request in requests} == {(256, 192)}
    assert all(not request.get("image_paths") for request in requests)
    assert len(list((output_dir / "outputs").rglob("*.png"))) == 44
    assert len(list((output_dir / "review").glob("scene-*.png"))) == 4
    assert len(list((output_dir / "review").glob("style-*.png"))) == 11
    assert (output_dir / "review" / "index.html").is_file()
    review_html = (output_dir / "review" / "index.html").read_text(encoding="utf-8")
    assert Counter((request["seed"], request["prompt"]) for request in requests) == Counter(
        (row["seed"], row["render_prompt"]) for row in result["results"]
    )
    for row in result["results"]:
        scene = next(scene for scene in experiment.scenes if scene.case_id == row["case_id"])
        style = next(style for style in experiment.styles if style.style_id == row["style_id"])
        expected_prompt = (
            scene.description
            if style.prompt_text is None
            else f"{scene.description}\n\n{style.prompt_text}"
        )
        assert row["render_prompt"] == expected_prompt
        assert row["generation"]["metadata"]["origin"]["render_prompt"] == expected_prompt
        assert (
            row["generation"]["metadata"]["authoring"]["inputs"]["description"] == scene.description
        )
        relative = f"outputs/{scene.case_id}/seed-{row['seed']}/{style.style_id}.png"
        assert row["generation"]["artifact_path"] == relative
        assert f'href="../{relative}"' in review_html
        assert f'src="../{relative}"' in review_html

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
    assert Counter(
        (row["case_id"], row["seed"], row["style_id"]) for row in scorecard["outputs"]
    ) == Counter(dict.fromkeys(expected_keys, 1))
    for output in scorecard["outputs"]:
        assert [criterion["criterion_id"] for criterion in output["criteria"]] == [
            "style-fidelity",
            "subject-preservation",
            "composition-preservation",
            "artifact-control",
        ]
        assert all(
            criterion["score"] is None and criterion["notes"] is None
            for criterion in output["criteria"]
        )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"


def test_style_suite_marks_empty_style_sheet_unavailable(
    tmp_path: Path, small_experiment: Path
) -> None:
    output_dir = tmp_path / "run"

    result_path = run_style_preset_evaluation(
        StylePresetSettings(
            output_dir=output_dir,
            experiment_path=small_experiment,
            tier=ResolutionTier.SMALL,
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


def test_style_suite_fails_when_no_image_can_be_generated(
    tmp_path: Path, small_experiment: Path
) -> None:
    output_dir = tmp_path / "run"

    with pytest.raises(ImageGenerationError, match="produced no images"):
        run_style_preset_evaluation(
            StylePresetSettings(
                output_dir=output_dir,
                experiment_path=small_experiment,
                tier=ResolutionTier.SMALL,
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
            "--tier",
            "Large",
            "--aspect-ratio",
            "16:9",
        ]
    )

    assert args.command == "style-presets"
    assert args.mflux_model == "flux2-klein-4b"
    assert args.seeds == [11, 12]
    assert args.tier is ResolutionTier.LARGE
    assert args.aspect_ratio is AspectRatio.WIDESCREEN


def test_cli_validates_style_experiment_without_model_calls(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_generation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("validate-only must not invoke an evaluation or model")

    monkeypatch.setattr(cli_module, "run_style_preset_evaluation", unexpected_generation)
    monkeypatch.setattr(MfluxGenerator, "generate", unexpected_generation)
    assert run_cli(["style-presets", "--validate-only"]) == 0
    assert capsys.readouterr().out.strip() == str(DEFAULT_STYLE_PRESET_EXPERIMENT)
