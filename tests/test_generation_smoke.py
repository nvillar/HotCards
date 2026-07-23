"""Tests for the retained production-path smoke orchestrator."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from hypergen.evaluation.cli import run_cli
from hypergen.evaluation.smoke import SmokeSettings, SmokeStageError, run_smoke
from hypergen.generation.errors import ModelUnavailableError
from hypergen.generation.mflux_generator import MfluxGenerator
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeGeneratedImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        Image.new("RGB", (self.width, self.height), "navy").save(path, format="PNG")


class FakeMfluxModel:
    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


class FakeOllamaClient:
    def __init__(self, responses: list[str], *, model_available: bool = True) -> None:
        self.responses = responses
        self.model_available = model_available

    def list(self) -> SimpleNamespace:
        models = (SimpleNamespace(model="qwen3.5:9b"),) if self.model_available else ()
        return SimpleNamespace(models=models)

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=("vision", "thinking"))

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        stream: bool,
        keep_alive: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(done=True)

    def chat(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(content=self.responses.pop(0)),
            total_duration=1_000_000,
            load_duration=250_000,
        )


def hotspot_response(*, left_edge: int = 100) -> str:
    return json.dumps(
        {
            "interactions": [
                {
                    "source_interaction_index": 1,
                    "label": "Gate",
                    "destination_token": "C1",
                    "polygons": [
                        {
                            "points": [
                                {"x": left_edge, "y": 100},
                                {"x": 400, "y": 100},
                                {"x": 400, "y": 800},
                                {"x": left_edge, "y": 800},
                            ]
                        }
                    ],
                }
            ]
        }
    )


def test_smoke_runner_writes_cold_and_warm_stage_results(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.png"
    fixture.write_bytes(b"fixture")
    output_dir = tmp_path / "run"
    client = FakeOllamaClient([hotspot_response(), hotspot_response()])
    runtime = OllamaRuntime(OllamaSettings(), client=client)
    mflux = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel())

    result_path = run_smoke(
        SmokeSettings(output_dir=output_dir, fixture_image=fixture),
        ollama_runtime=runtime,
        mflux_generator=mflux,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert set(result["stages"]) == {
        "image_generation",
        "hotspot_generation",
    }
    assert result["render_prompt"] == (
        "A quiet stone castle courtyard at dusk with an arched wooden gate, "
        "a red travel chest, and a leafy tree.\n\n"
        "Cool twilight shadows with warm lantern light."
    )
    assert result["stages"]["image_generation"]["cold"]["metadata"]["width"] == 1024
    assert result["settings"]["ollama"]["think"] is False
    assert result["settings"]["ollama"]["temperature"] == 0.0
    assert result["settings"]["ollama"]["keep_alive"] == "10m"
    assert (output_dir / "generated-cold.png").is_file()
    assert (output_dir / "generated-warm.png").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "report.html").is_file()


def test_smoke_runner_records_actionable_expected_failure(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.png"
    fixture.write_bytes(b"fixture")
    output_dir = tmp_path / "run"
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([], model_available=False),
    )

    with pytest.raises(SmokeStageError, match="ollama_diagnostics") as caught:
        run_smoke(
            SmokeSettings(output_dir=output_dir, fixture_image=fixture),
            ollama_runtime=runtime,
            mflux_generator=MfluxGenerator(model_factory=lambda *_: FakeMfluxModel()),
        )

    assert isinstance(caught.value.cause, ModelUnavailableError)
    result = json.loads((output_dir / "smoke-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["stage"] == "ollama_diagnostics"
    assert result["error_type"] == "ModelUnavailableError"
    assert "ollama pull" in result["message"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "ollama_diagnostics"


def test_smoke_failure_promotes_retained_hotspot_warnings(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.png"
    fixture.write_bytes(b"fixture")
    output_dir = tmp_path / "run"
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient(
            [
                hotspot_response(left_edge=-10),
                '{"interactions":',
            ]
        ),
    )

    with pytest.raises(SmokeStageError, match="hotspot_generation_warm"):
        run_smoke(
            SmokeSettings(output_dir=output_dir, fixture_image=fixture),
            ollama_runtime=runtime,
            mflux_generator=MfluxGenerator(model_factory=lambda *_: FakeMfluxModel()),
        )

    result = json.loads((output_dir / "smoke-result.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result["warnings"] == ["model coordinates were clamped to the canvas"]
    assert manifest["warnings"] == result["warnings"]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--ollama-timeout", "0"),
        ("--ollama-timeout", "nan"),
        ("--ollama-timeout", "inf"),
        ("--ollama-num-predict", "-1"),
        ("--ollama-context-length", "0"),
    ],
)
def test_smoke_cli_rejects_nonpositive_limits(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as caught:
        run_cli(["smoke", option, value])

    assert caught.value.code == 2
