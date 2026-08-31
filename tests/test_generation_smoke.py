"""Tests for the retained production-path smoke orchestrator."""

import json
from pathlib import Path

import pytest
from PIL import Image

from hotcards.evaluation.cli import run_cli
from hotcards.evaluation.smoke import SmokeSettings, SmokeStageError, run_smoke
from hotcards.generation.mflux_generator import MfluxGenerator


class FakeGeneratedImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(
            path,
            format="PNG",
        )


class FakeMfluxModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        if self.fail:
            raise RuntimeError("candidate failed")
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


def test_smoke_runner_writes_cold_and_warm_stage_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    mflux = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel())

    result_path = run_smoke(
        SmokeSettings(output_dir=output_dir),
        mflux_generator=mflux,
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert set(result["stages"]) == {"image_generation"}
    assert result["render_prompt"] == (
        "A quiet stone castle courtyard at dusk with an arched wooden gate, "
        "a red travel chest, and a leafy tree. "
        "Restrained storybook ink and watercolor illustration with cool "
        "twilight shadows and warm lantern light."
    )
    cold = result["stages"]["image_generation"]["cold"]
    assert cold["provenance"]["operation"] == "generate"
    assert (
        cold["provenance"]["settings"]["width"],
        cold["provenance"]["settings"]["height"],
    ) == (592, 448)
    assert (output_dir / "generated-cold.png").is_file()
    assert (output_dir / "generated-warm.png").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "report.html").is_file()


def test_smoke_runner_records_actionable_generation_failure(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"

    with pytest.raises(SmokeStageError, match="image_generation_cold"):
        run_smoke(
            SmokeSettings(output_dir=output_dir),
            mflux_generator=MfluxGenerator(model_factory=lambda *_: FakeMfluxModel(fail=True)),
            environment_provider=lambda: {"git_sha": "test"},
        )

    result = json.loads((output_dir / "smoke-result.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert result["status"] == "failed"
    assert result["stage"] == "image_generation_cold"
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "image_generation_cold"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--quantization", "not-an-integer"),
    ],
)
def test_smoke_cli_rejects_invalid_numeric_values(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as caught:
        run_cli(["smoke", option, value])

    assert caught.value.code == 2
