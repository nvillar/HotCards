"""Tests for the blinded inline Reference language evaluation."""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin
from pydantic import ValidationError

from hypergen.evaluation.cli import build_parser, run_cli
from hypergen.evaluation.inline_references import (
    DEFAULT_INLINE_REFERENCE_EXPERIMENT,
    InlineReferenceSettings,
    load_inline_reference_experiment,
    run_inline_reference_evaluation,
)
from hypergen.generation.mflux_generator import MfluxGenerator


class FakeGeneratedImage:
    def __init__(self, width: int, height: int, prompt: str) -> None:
        self.width = width
        self.height = height
        self.prompt = prompt

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("prompt", self.prompt)
        Image.new("RGB", (self.width, self.height), "navy").save(
            path,
            format="PNG",
            pnginfo=metadata,
        )


class FakeEditModel:
    def __init__(self, requests: list[dict[str, object]]) -> None:
        self.requests = requests

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.requests.append(kwargs)
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
            prompt=kwargs["prompt"],  # type: ignore[arg-type]
        )


def test_tracked_inline_reference_experiment_uses_diagnostic_cases() -> None:
    experiment, benchmark = load_inline_reference_experiment(
        DEFAULT_INLINE_REFERENCE_EXPERIMENT
    )

    assert [case.case_id for case in experiment.cases] == [
        "stairs-through-panel",
        "computer-front-viewpoint",
        "blue-cabinet-open",
        "cut-paper-city-style-only",
    ]
    assert all("input image" in case.inline_prompt for case in experiment.cases)
    benchmark_cases = {case.case_id: case for case in benchmark.cases}
    assert all(
        benchmark_cases[case.case_id].reference is not None
        for case in experiment.cases
    )


def test_inline_reference_suite_creates_balanced_blind_review(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_inline_reference_evaluation(
        InlineReferenceSettings(
            output_dir=output_dir,
            seeds=(1, 2, 3),
            width=48,
            height=32,
            blinding_seed=99,
        ),
        generator_factory=lambda: MfluxGenerator(
            edit_model_factory=lambda *_: FakeEditModel(requests)
        ),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert len(result["results"]) == 24
    public_payload = result_path.read_text(encoding="utf-8")
    assert "condition_id" not in public_payload
    assert "render_prompt" not in public_payload
    assert "blinding_seed" not in result["settings"]
    assert len(requests) == 24
    assert all(len(request["image_paths"]) == 1 for request in requests)
    assert all(request["use_kv_cache"] is True for request in requests)
    assert len(list((output_dir / "inputs").rglob("*.png"))) == 3
    assert all(
        path.relative_to(output_dir / "inputs").parts[0] == "references"
        for path in (output_dir / "inputs").rglob("*.png")
    )
    assert len(list((output_dir / "outputs").rglob("*.png"))) == 24
    assert len(list((output_dir / "audit" / "outputs").rglob("*.png"))) == 24
    for path in (output_dir / "outputs").rglob("*.png"):
        with Image.open(path) as image:
            assert "prompt" not in image.info
    for path in (output_dir / "audit" / "outputs").rglob("*.png"):
        with Image.open(path) as image:
            assert image.info["prompt"]

    scorecard_path = output_dir / "review" / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    assert len(scorecard["pairs"]) == 12
    assert all(
        [candidate["label"] for candidate in pair["candidates"]] == ["A", "B"]
        for pair in scorecard["pairs"]
    )
    assert all(
        criterion["score"] is None
        for pair in scorecard["pairs"]
        for candidate in pair["candidates"]
        for criterion in candidate["criteria"]
    )
    blinded_payload = scorecard_path.read_text(encoding="utf-8")
    assert "condition_id" not in blinded_payload
    assert "render_prompt" not in blinded_payload
    assert len(list((output_dir / "review").glob("*.png"))) == 12
    assert (output_dir / "review" / "index.html").is_file()

    key = json.loads(
        (output_dir / "audit" / "condition-key.json").read_text(encoding="utf-8")
    )
    assert key["blinding_seed"] == 99
    assert (output_dir / "audit" / "generation-records.json").is_file()
    inline_as_a = sum(
        pair["candidates"]["A"]["condition_id"] == "inline"
        for pair in key["pairs"]
    )
    assert inline_as_a == 6
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert "blinding_seed" not in manifest["settings"]
    assert manifest["status"] == "success"
    assert any(
        artifact["path"] == "review/scorecard.json"
        for artifact in manifest["artifacts"]
    )


def test_inline_reference_settings_reject_duplicate_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        InlineReferenceSettings(
            output_dir=tmp_path / "run",
            seeds=(1, 1),
        )


def test_inline_reference_loader_requires_visible_text_criterion(
    tmp_path: Path,
) -> None:
    source = Path("evals/cases/image_prompts")
    benchmark_dir = tmp_path / "image_prompts"
    shutil.copytree(source, benchmark_dir)
    benchmark_path = benchmark_dir / "benchmark.json"
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    payload["common_criteria"] = [
        criterion
        for criterion in payload["common_criteria"]
        if criterion["criterion_id"] != "no-unrequested-visible-text"
    ]
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires common criterion"):
        load_inline_reference_experiment(
            DEFAULT_INLINE_REFERENCE_EXPERIMENT,
            benchmark_path,
        )


def test_cli_exposes_inline_reference_options() -> None:
    args = build_parser().parse_args(
        [
            "inline-references",
            "--mflux-model",
            "flux2-klein-4b",
            "--seed",
            "11",
            "--seed",
            "12",
            "--blinding-seed",
            "99",
        ]
    )

    assert args.command == "inline-references"
    assert args.mflux_model == "flux2-klein-4b"
    assert args.seeds == [11, 12]
    assert args.blinding_seed == 99


def test_cli_validates_inline_reference_experiment_without_model_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_cli(["inline-references", "--validate-only"]) == 0
    assert capsys.readouterr().out.strip() == str(
        DEFAULT_INLINE_REFERENCE_EXPERIMENT
    )
