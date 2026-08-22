"""End-to-end suite tests using only production-adapter fakes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import hypergen.evaluation.e2e as e2e_module
from hypergen.evaluation.cli import run_cli
from hypergen.evaluation.e2e import (
    E2E_MFLUX_MODEL,
    E2E_OLLAMA_MODELS,
    E2EEvaluationSettings,
    run_e2e_evaluation,
)
from hypergen.evaluation.reports import ReportRenderingError
from hypergen.generation.mflux_generator import MfluxGenerator
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeGeneratedImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(path)


class FakeMfluxModel:
    def __init__(self, calls: list[dict[str, object]] | None = None) -> None:
        self.calls = calls

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        if self.calls is not None:
            self.calls.append(kwargs)
        return FakeGeneratedImage(
            kwargs["width"],  # type: ignore[arg-type]
            kwargs["height"],  # type: ignore[arg-type]
        )


class FakeOllamaClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.call_count = 0

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(models=(SimpleNamespace(model=self.model),))

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=("vision",))

    def generate(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(done=True)

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.call_count += 1
        if self.model == "qwen3.5:9b-mlx":
            content = '{"interactions":'
        else:
            far_edge = 1_500 if self.model == "qwen3.6:35b" else 500
            content = json.dumps(
                {
                    "interactions": [
                        {
                            "source_interaction_index": 1,
                            "label": "Gate",
                            "destination_token": "C1",
                            "polygons": [
                                {
                                    "points": [
                                        {"x": 100, "y": 100},
                                        {"x": far_edge, "y": 100},
                                        {"x": 500, "y": 600},
                                    ]
                                }
                            ],
                        }
                    ]
                }
            )
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            total_duration=2_000_000,
            load_duration=500_000,
            prompt_eval_count=20,
            eval_count=10,
            done_reason="stop",
        )


def _write_case(case_dir: Path) -> None:
    case_dir.mkdir()
    (case_dir / "one.json").write_text(
        json.dumps(
            {
                "case_version": "e2e-case-v2",
                "case_id": "one",
                "inputs": {
                    "scene_description": "Courtyard with gate",
                    "global_style": "Watercolor",
                    "card_style": None,
                },
                "interaction_description": "Gate leads to garden",
                "card_catalogue": [{"token": "C1", "name": "Garden"}],
                "expected_hotspots": [{"label": "Gate", "target_token": "C1"}],
                "provenance": "Project-authored synthetic input.",
                "reuse_terms": "Repository test fixture.",
            }
        )
    )


def test_e2e_uses_exact_models_fixed_mflux_and_preserves_partial_stages(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)
    requested_models: list[str] = []
    mflux_calls: list[dict[str, object]] = []

    def runtime_factory(settings: OllamaSettings) -> OllamaRuntime:
        requested_models.append(settings.model)
        return OllamaRuntime(settings, client=FakeOllamaClient(settings.model))

    result_path = run_e2e_evaluation(
        E2EEvaluationSettings(output_dir=tmp_path / "run", case_dir=case_dir),
        runtime_factory=runtime_factory,
        mflux_factory=lambda: MfluxGenerator(
            model_factory=lambda *_: FakeMfluxModel(mflux_calls)
        ),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert requested_models == list(E2E_OLLAMA_MODELS)
    assert len(mflux_calls) == 1
    assert {row["mflux_model"] for row in result["candidates"]} == {E2E_MFLUX_MODEL}
    assert len(result["candidates"]) == 3
    assert {row["render_prompt"] for row in result["candidates"]} == {
        "Courtyard with gate\n\nWatercolor"
    }
    assert all(
        stage["name"] != "prompt_derivation"
        for row in result["candidates"]
        for stage in row["stages"]
    )
    failed = next(
        row for row in result["candidates"] if row["ollama_model"] == "qwen3.5:9b-mlx"
    )
    assert failed["artifact_path"]
    assert failed["stages"][-1]["failure"]["classification"] == "structured_output_validation"
    assert (
        tmp_path / "run" / "raw" / "one" / "qwen3.5-9b-mlx" / "partial-hotspots.json"
    ).is_file()
    assert (tmp_path / "run" / "manifest.json").is_file()
    assert (tmp_path / "run" / "summary.csv").is_file()
    assert (tmp_path / "run" / "report.html").is_file()
    assert (tmp_path / "run" / "contact-sheet.png").is_file()
    assert (tmp_path / "run" / "report.html").read_text().count("<img src=") == 3
    assert len(list((tmp_path / "run" / "annotations").rglob("*.png"))) == 2
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["warnings"] == [
        "one:qwen3.6:35b:hotspot_generation: model coordinates were clamped to the canvas"
    ]
    assert not any(
        "qwen3.5:9b-mlx:hotspot_generation" in stage
        for stage in manifest["completed_stages"]
    )


def test_e2e_report_failure_finalizes_manifest_without_erasing_stage_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "cases"
    _write_case(case_dir)

    def runtime_factory(settings: OllamaSettings) -> OllamaRuntime:
        return OllamaRuntime(settings, client=FakeOllamaClient(settings.model))

    monkeypatch.setattr(
        e2e_module,
        "render_reports_checked",
        lambda path: (_ for _ in ()).throw(
            ReportRenderingError(path, RuntimeError("report failed"))
        ),
    )

    with pytest.raises(ReportRenderingError, match="report failed"):
        run_e2e_evaluation(
            E2EEvaluationSettings(output_dir=tmp_path / "run", case_dir=case_dir),
            runtime_factory=runtime_factory,
            mflux_factory=lambda: MfluxGenerator(model_factory=lambda *_: FakeMfluxModel()),
            environment_provider=lambda: {"git_sha": "test"},
        )

    result = json.loads((tmp_path / "run" / "e2e-results.json").read_text())
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert result["status"] == "completed_with_report_failure"
    assert all(
        stage["status"] == "success"
        for candidate in result["candidates"]
        for stage in candidate["stages"]
        if candidate["ollama_model"] != "qwen3.5:9b-mlx"
    )
    assert manifest["status"] == "completed_with_report_failure"
    assert manifest["failure"]["classification"] == "report_rendering"
    assert manifest["warnings"] == [
        "one:qwen3.6:35b:hotspot_generation: model coordinates were clamped to the canvas"
    ]


def test_cli_reports_preserved_result_path_on_report_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_path = tmp_path / "run" / "e2e-results.json"

    def fail_report(_settings: object) -> Path:
        raise ReportRenderingError(result_path, RuntimeError("report failed"))

    monkeypatch.setattr(e2e_module, "run_e2e_evaluation", fail_report)
    monkeypatch.setattr("hypergen.evaluation.cli.run_e2e_evaluation", fail_report)

    assert run_cli(["e2e"]) == 1
    captured = capsys.readouterr()
    assert str(result_path) in captured.out
    assert "report failed" in captured.err
