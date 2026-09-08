"""Tests for the retained production-path smoke orchestrator."""

import csv
import json
from itertools import count
from pathlib import Path
from threading import get_ident

import pytest
from PIL import Image

import hotcards.generation.mflux_generator as mflux_module
from hotcards.domain.image_dimensions import AspectRatio, ResolutionTier
from hotcards.domain.models import GenerateInputs, PresetOutputSize
from hotcards.evaluation.cli import build_parser, run_cli
from hotcards.evaluation.smoke import SmokeSettings, SmokeStageError, run_smoke
from hotcards.generation.mflux_generator import MfluxGenerateRequest, MfluxGenerator


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


class FakeCallbacks:
    def register(self, callback: object) -> None:
        self.callback = callback


class FakeMfluxModel:
    """Dimension-respecting native fake shared only by evaluation tests."""

    def __init__(
        self,
        requests: list[dict[str, object]],
        *,
        fail_prompt_contains: str | None = None,
    ) -> None:
        self.requests = requests
        self.fail_prompt_contains = fail_prompt_contains
        self.callbacks = FakeCallbacks()

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


def test_smoke_runner_writes_cold_and_warm_stage_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = count()
    monkeypatch.setattr(mflux_module, "perf_counter", lambda: float(next(ticks)))
    output_dir = tmp_path / "run"
    requests: list[dict[str, object]] = []
    load_threads: list[int] = []

    def model_factory(*_args: object) -> FakeMfluxModel:
        load_threads.append(get_ident())
        return FakeMfluxModel(requests)

    MfluxGenerator(model_factory=model_factory).generate(
        MfluxGenerateRequest(
            inputs=GenerateInputs(
                description="Prewarm the process-global cache.",
                output_size=PresetOutputSize(tier=ResolutionTier.SMALL),
            ),
            render_prompt="Prewarm the process-global cache.",
            output_path=tmp_path / "prewarm.png",
            seed=42,
            width=256,
            height=192,
        )
    )
    requests.clear()
    mflux = MfluxGenerator(model_factory=model_factory)

    result_path = run_smoke(
        SmokeSettings(output_dir=output_dir, tier=ResolutionTier.SMALL),
        mflux_generator=mflux,
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(load_threads) == 2
    assert len(set(load_threads)) == 1
    assert load_threads[0] != get_ident()
    assert result["status"] == "success"
    assert set(result["stages"]) == {"image_generation"}
    assert result["render_prompt"] == (
        "A quiet stone castle courtyard at dusk with an arched wooden gate, "
        "a red travel chest, and a leafy tree. "
        "Restrained storybook ink and watercolor illustration with cool "
        "twilight shadows and warm lantern light."
    )
    cold = result["stages"]["image_generation"]["cold"]
    assert cold["metadata"]["authoring"]["operation"] == "generate"
    assert (
        cold["metadata"]["origin"]["settings"]["width"],
        cold["metadata"]["origin"]["settings"]["height"],
    ) == (256, 192)
    assert cold["metadata"]["authoring"]["inputs"]["output_size"] == {
        "mode": "preset",
        "tier": 256,
    }
    assert [request["prompt"] for request in requests] == [result["render_prompt"]] * 2
    summary = json.loads((output_dir / "summary.json").read_text())
    with (output_dir / "summary.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    report = (output_dir / "report.html").read_text()
    for phase, row, csv_row in zip(("cold", "warm"), summary["rows"], csv_rows, strict=True):
        generation = result["stages"]["image_generation"][phase]
        assert generation["metadata"]["origin"]["render_prompt"] == result["render_prompt"]
        assert row["phase"] == phase
        assert row["render_prompt"] == csv_row["render_prompt"] == result["render_prompt"]
        assert row["elapsed_seconds"] == generation["total_duration_seconds"]
        assert generation["queue_duration_seconds"] == 1
        assert generation["inference_duration_seconds"] == 1
        assert generation["serialization_duration_seconds"] == 1
        assert (
            json.loads(csv_row["timings"])
            == row["timings"]
            == {
                key: generation[key]
                for key in (
                    "queue_duration_seconds",
                    "load_duration_seconds",
                    "inference_duration_seconds",
                    "serialization_duration_seconds",
                    "total_duration_seconds",
                )
            }
        )
        assert row["artifact_path"] == csv_row["artifact_path"] == f"generated-{phase}.png"
        assert f'href="generated-{phase}.png"' in report
        assert f'src="generated-{phase}.png"' in report
        with Image.open(output_dir / str(row["artifact_path"])) as image:
            assert image.size == (256, 192)
    assert 'href="contact-sheet.png"' in report


def test_smoke_settings_reject_obsolete_arbitrary_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="width"):
        SmokeSettings(output_dir=tmp_path / "run", width=512)  # type: ignore[call-arg]


def test_smoke_runner_records_actionable_generation_failure(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"

    with pytest.raises(SmokeStageError, match="image_generation_cold"):
        run_smoke(
            SmokeSettings(output_dir=output_dir),
            mflux_generator=MfluxGenerator(
                model_factory=lambda *_: FakeMfluxModel([], fail_prompt_contains="")
            ),
            environment_provider=lambda: {"git_sha": "test"},
        )

    result = json.loads((output_dir / "smoke-result.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert result["status"] == "failed"
    assert result["stage"] == "image_generation_cold"
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "image_generation_cold"


def test_smoke_warm_failure_preserves_cold_report_and_image(tmp_path: Path) -> None:
    class FailWarmModel(FakeMfluxModel):
        def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
            if self.requests:
                raise RuntimeError("warm invocation failed")
            return super().generate_image(**kwargs)

    output_dir = tmp_path / "run"
    with pytest.raises(SmokeStageError, match="image_generation_warm"):
        run_smoke(
            SmokeSettings(output_dir=output_dir, tier=ResolutionTier.SMALL),
            mflux_generator=MfluxGenerator(model_factory=lambda *_: FailWarmModel([])),
            environment_provider=lambda: {"git_sha": "test"},
        )
    summary = json.loads((output_dir / "summary.json").read_text())
    assert [(row["phase"], row["status"]) for row in summary["rows"]] == [
        ("cold", "success"),
        ("failure", "failed"),
    ]
    assert summary["rows"][0]["artifact_path"] == "generated-cold.png"
    assert summary["rows"][1]["axis"] == "image_generation_warm"
    report = (output_dir / "report.html").read_text()
    assert 'href="generated-cold.png"' in report
    assert 'href="generated-warm.png"' not in report


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--quantization", "not-an-integer"),
        ("--tier", "300"),
        ("--aspect-ratio", "3:2"),
        ("--width", "512"),
    ],
)
def test_smoke_cli_rejects_invalid_numeric_values(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as caught:
        run_cli(["smoke", option, value])

    assert caught.value.code == 2


def test_smoke_cli_accepts_typed_generation_dimensions() -> None:
    args = build_parser().parse_args(["smoke", "--tier", "Large", "--aspect-ratio", "16:9"])

    assert args.tier is ResolutionTier.LARGE
    assert args.aspect_ratio is AspectRatio.WIDESCREEN
