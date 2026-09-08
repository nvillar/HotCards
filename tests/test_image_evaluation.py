"""Tests for controlled image evaluation behind production-adapter fakes."""

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError
from test_generation_smoke import FakeMfluxModel

import hotcards.evaluation.cli as cli_module
from hotcards.domain.image_dimensions import AspectRatio, ResolutionTier
from hotcards.domain.models import GenerateInputs, PresetOutputSize
from hotcards.evaluation.images import (
    DEFAULT_MFLUX_MODELS,
    ImageEvaluationCase,
    ImageEvaluationSettings,
    load_image_cases,
    run_image_evaluation,
)
from hotcards.evaluation.results import generation_record
from hotcards.generation.mflux_generator import MfluxGenerateRequest, MfluxGenerator


def _write_case(case_dir: Path) -> None:
    case_dir.mkdir()
    (case_dir / "one.json").write_text(
        json.dumps(
            {
                "case_version": "image-case-v3",
                "case_id": "one",
                "inputs": {
                    "description": "A storybook watercolor courtyard",
                },
                "required_visual_elements": ["gate"],
                "unwanted_artifacts": ["text"],
            }
        )
    )


@pytest.mark.parametrize("corruption", ["receipt", "nested_dimensions"])
def test_record_rejects_and_disposes_invalid_complete_results(
    tmp_path: Path, corruption: str
) -> None:
    result = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel([])).generate(
        MfluxGenerateRequest(
            inputs=GenerateInputs(
                description="A courtyard",
                output_size=PresetOutputSize(tier=ResolutionTier.SMALL),
            ),
            render_prompt="A courtyard",
            output_path=tmp_path / "result.png",
            seed=42,
            width=256,
            height=192,
        )
    )
    provenance = result.provenance
    if corruption == "receipt":
        provenance = provenance.model_copy(update={"authoring": None})
    else:
        settings = provenance.settings.model_copy(update={"width": 0})
        origin = provenance.origin.model_copy(update={"settings": settings})
        provenance = provenance.model_copy(update={"origin": origin})
    invalid = result.model_copy(update={"provenance": provenance})
    assert result.output_path.is_file()

    with pytest.raises(ValidationError):
        generation_record(invalid, tmp_path)

    assert not result.output_path.exists()


def test_tracked_image_cases_load_in_stable_order() -> None:
    cases = load_image_cases(Path("evals/cases/images"))

    assert [case.case_id for case in cases] == ["courtyard", "workshop"]
    assert cases[0].required_visual_elements
    assert cases[0].unwanted_artifacts


def test_image_case_id_is_safe_for_artifact_paths() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        ImageEvaluationCase.model_validate(
            {
                "case_version": "image-case-v3",
                "case_id": "../escape",
                "inputs": {
                    "description": "A storybook watercolor courtyard",
                },
                "required_visual_elements": ["gate"],
            }
        )


def test_image_settings_reject_obsolete_arbitrary_dimensions(
    tmp_path: Path,
) -> None:
    settings = ImageEvaluationSettings(output_dir=tmp_path / "default")
    assert settings.tier is ResolutionTier.FULL
    assert settings.aspect_ratio is AspectRatio.LANDSCAPE
    with pytest.raises(ValidationError, match="height"):
        ImageEvaluationSettings(
            output_dir=tmp_path / "run",
            height=384,  # type: ignore[call-arg]
        )


def test_image_suite_uses_deterministic_prompts_for_mflux_candidates(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)
    requests: list[dict[str, object]] = []

    result_path = run_image_evaluation(
        ImageEvaluationSettings(
            output_dir=tmp_path / "run",
            case_dir=case_dir,
            tier=ResolutionTier.SMALL,
            aspect_ratio=AspectRatio.PORTRAIT,
        ),
        mflux_factory=lambda: MfluxGenerator(model_factory=lambda *_: FakeMfluxModel(requests)),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "success"
    assert "prompt_axis" not in result
    assert "prompt_downstream_axis" not in result
    assert len(result["mflux_axis"]) == 2
    assert {row["model"] for row in result["mflux_axis"]} == set(DEFAULT_MFLUX_MODELS)
    assert {row["render_prompt"] for row in result["mflux_axis"]} == {
        "A storybook watercolor courtyard"
    }
    assert {request["prompt"] for request in requests} == {"A storybook watercolor courtyard"}
    assert {(request["width"], request["height"]) for request in requests} == {(192, 256)}
    assert result["mflux_axis"][0]["cold"]["metadata"]["authoring"]["inputs"]["output_size"] == {
        "mode": "preset",
        "tier": 256,
    }
    assert result["mflux_axis"][0]["cold"]["metadata"]["origin"]["render_prompt"] == (
        "A storybook watercolor courtyard"
    )
    assert len(list((tmp_path / "run" / "images" / "mflux-axis").rglob("*.png"))) == 4
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert {row["axis"] for row in summary["rows"]} == {"mflux"}
    assert {row["render_prompt"] for row in summary["rows"]} == {"A storybook watercolor courtyard"}
    report = (tmp_path / "run" / "report.html").read_text()
    expected_keys = {(model, phase) for model in DEFAULT_MFLUX_MODELS for phase in ("cold", "warm")}
    assert {(row["mflux_model"], row["phase"]) for row in summary["rows"]} == expected_keys
    assert len(summary["rows"]) == len(expected_keys)
    for row in summary["rows"]:
        relative = f"images/mflux-axis/one/{row['mflux_model']}/{row['phase']}.png"
        assert row["artifact_path"] == relative
        assert f'href="{relative}"' in report
        assert f'src="{relative}"' in report
        generation = next(
            item for item in result["mflux_axis"] if item["model"] == row["mflux_model"]
        )[row["phase"]]
        assert row["elapsed_seconds"] == generation["total_duration_seconds"]
        assert row["cache_reused"] == generation["cache_reused"]
        assert generation["cache_reused"] is (row["phase"] == "warm")
        with Image.open(tmp_path / "run" / relative) as image:
            assert image.size == (192, 256)


def test_image_suite_isolates_mflux_candidate_failure(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)

    def mflux_factory() -> MfluxGenerator:
        return MfluxGenerator(
            model_factory=lambda model, _quantization: FakeMfluxModel(
                [],
                fail_prompt_contains="" if model == "flux2-klein-9b" else None,
            )
        )

    result_path = run_image_evaluation(
        ImageEvaluationSettings(
            output_dir=tmp_path / "run",
            case_dir=case_dir,
            tier=ResolutionTier.SMALL,
            aspect_ratio=AspectRatio.LANDSCAPE,
        ),
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


def test_image_suite_fails_with_retained_reports_when_all_candidates_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        cli_module,
        "run_image_evaluation",
        lambda settings: run_image_evaluation(
            settings,
            mflux_factory=lambda: MfluxGenerator(
                model_factory=lambda *_: FakeMfluxModel([], fail_prompt_contains="")
            ),
            environment_provider=lambda: {"git_sha": "test"},
        ),
    )
    assert (
        cli_module.run_cli(
            [
                "images",
                "--output-dir",
                str(output_dir),
                "--case-dir",
                str(case_dir),
                "--mflux-model",
                "flux2-klein-4b",
                "--tier",
                "Small",
            ]
        )
        == 1
    )
    assert "produced no images" in capsys.readouterr().err
    result = json.loads((output_dir / "image-results.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    assert result["status"] == manifest["status"] == summary["suite_status"] == "failed"
    assert [row["status"] for row in summary["rows"]] == ["failed", "skipped"]
    assert len(manifest["failure"]["stages"]) == 2
    assert 'href="contact-sheet.png"' not in (output_dir / "report.html").read_text()
