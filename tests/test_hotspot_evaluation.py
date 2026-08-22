"""Tests for retained hotspot-remap evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from hypergen.evaluation.hotspots import (
    HotspotEvaluationSettings,
    load_hotspot_cases,
    run_hotspot_evaluation,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeOllamaClient:
    def __init__(
        self,
        model: str,
        *,
        available: bool = True,
        fenced: bool = False,
    ) -> None:
        self.model = model
        self.available = available
        self.fenced = fenced

    def list(self) -> SimpleNamespace:
        models = (SimpleNamespace(model=self.model),) if self.available else ()
        return SimpleNamespace(models=models)

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=("vision", "thinking"))

    def generate(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(done=True)

    def chat(self, **kwargs: object) -> SimpleNamespace:
        content = json.dumps(
            {
                "mapped": [
                    {
                        "token": "H1",
                        "polygons": [
                            {
                                "points": [
                                    {"x": -10, "y": 100},
                                    {"x": 400, "y": 100},
                                    {"x": 400, "y": 500},
                                    {"x": -10, "y": 500},
                                ]
                            }
                        ],
                    }
                ],
                "unlocated": [{"token": "H2", "reason": "not visible"}],
            }
        )
        if self.fenced:
            content = f"```json\n{content}\n```"
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            total_duration=3_000_000,
            load_duration=750_000,
            prompt_eval_count=200,
            eval_count=100,
            done_reason="stop",
        )


def _write_case(case_dir: Path, fixture_root: Path) -> None:
    case_dir.mkdir()
    Image.new("RGB", (32, 24), "navy").save(fixture_root / "fixture.png")
    (case_dir / "one.json").write_text(
        json.dumps(
            {
                "case_version": "hotspot-remap-case-v1",
                "case_id": "one",
                "image_path": "fixture.png",
                "hotspots": [
                    {"token": "H1", "label": "Gate"},
                    {"token": "H2", "label": "Chest"},
                ],
                "regression_tags": ["coordinate-clamping"],
            }
        )
    )


def test_hotspot_suite_records_remap_metrics_and_artifacts(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    case_dir = tmp_path / "cases"
    _write_case(case_dir, fixture_root)

    def runtime_factory(settings: OllamaSettings) -> OllamaRuntime:
        return OllamaRuntime(
            settings,
            client=FakeOllamaClient(settings.model, fenced=True),
        )

    result_path = run_hotspot_evaluation(
        HotspotEvaluationSettings(
            output_dir=tmp_path / "run",
            case_dir=case_dir,
            fixture_root=fixture_root,
            include_ablations=False,
        ),
        runtime_factory=runtime_factory,
    )

    result = json.loads(result_path.read_text())
    assert result["result_version"] == "hotspot-result-v2"
    assert len(result["models"]) == 3
    assert len(result["summaries"]) == 3
    first = result["models"][0]["cold"]
    assert first["structured_valid"] is True
    assert first["token_usage"] == {
        "prompt_tokens": 200,
        "output_tokens": 100,
    }
    assert first["remap"] == {
        "expected_count": 2,
        "mapped_count": 1,
        "unlocated_count": 1,
        "mapping_rate": 0.5,
        "semantic_change_count": 0,
    }
    assert first["geometry"]["validity_rate_after_cleanup"] == 1.0
    assert len(list((tmp_path / "run" / "raw").rglob("*.json"))) == 6
    assert any(
        warning.startswith("one:qwen3.5:4b:cold:")
        for warning in result["warnings"]
    )
    assert (tmp_path / "run" / "manifest.json").is_file()
    assert (tmp_path / "run" / "summary.json").is_file()
    assert (tmp_path / "run" / "report.html").is_file()
    assert (tmp_path / "run" / "contact-sheet.png").is_file()
    assert len(list((tmp_path / "run" / "annotations").rglob("*.png"))) == 6


def test_hotspot_suite_isolates_unavailable_model(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    case_dir = tmp_path / "cases"
    _write_case(case_dir, fixture_root)

    def runtime_factory(settings: OllamaSettings) -> OllamaRuntime:
        return OllamaRuntime(
            settings,
            client=FakeOllamaClient(
                settings.model,
                available=settings.model != "qwen3.5:9b-mlx",
            ),
        )

    result_path = run_hotspot_evaluation(
        HotspotEvaluationSettings(
            output_dir=tmp_path / "run",
            case_dir=case_dir,
            fixture_root=fixture_root,
            include_ablations=False,
        ),
        runtime_factory=runtime_factory,
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "completed_with_failures"
    failed = next(
        row
        for row in result["models"]
        if row["model"] == "qwen3.5:9b-mlx"
    )
    assert failed["cold"]["failure"]["classification"] == (
        "transport_or_service"
    )
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"
    assert not any(
        "qwen3.5:9b-mlx" in stage
        for stage in manifest["completed_stages"]
    )
    assert len(list((tmp_path / "run" / "annotations").rglob("*.png"))) == 4


def test_hotspot_case_fixture_cannot_escape_root(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    (case_dir / "escape.json").write_text(
        json.dumps(
            {
                "case_version": "hotspot-remap-case-v1",
                "case_id": "escape",
                "image_path": "../outside.png",
                "hotspots": [{"token": "H1", "label": "Gate"}],
            }
        )
    )

    with pytest.raises(ValueError, match="escapes fixture root"):
        load_hotspot_cases(case_dir, fixture_root=fixture_root)
