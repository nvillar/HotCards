"""Tests for quantitative hotspot evaluation behind Ollama fakes."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from hypergen.domain.models import Point, ResolvedCardReference
from hypergen.evaluation.hotspots import (
    HotspotEvaluationSettings,
    load_hotspot_cases,
    run_hotspot_evaluation,
)
from hypergen.generation.hotspot_prompts import (
    CandidatePolygon,
    ExistingCandidateTarget,
    HotspotProposal,
    apply_hotspot_proposal,
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
                "interactions": [
                    {
                        "source_interaction_index": 1,
                        "label": "Gate",
                        "destination_token": "C1",
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
                ]
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
                "case_id": "one",
                "image_path": "fixture.png",
                "interaction_description": "The gate leads to the garden",
                "card_catalogue": [{"token": "C1", "name": "Garden"}],
                "expected_hotspots": [{"label": "Gate", "target_token": "C1"}],
                "regression_tags": ["degenerate-polygon"],
            }
        )
    )


def test_hotspot_suite_records_raw_metrics_and_separate_models(tmp_path: Path) -> None:
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
    assert len(result["models"]) == 3
    assert len(result["summaries"]) == 3
    first = result["models"][0]["cold"]
    assert first["structured_valid"] is True
    assert first["token_usage"] == {"prompt_tokens": 200, "output_tokens": 100}
    assert first["destinations"]["destination_accuracy"] == 1.0
    assert first["geometry"]["validity_rate_before_cleanup"] == 0.0
    assert first["geometry"]["validity_rate_after_cleanup"] == 1.0
    assert len(list((tmp_path / "run" / "raw").rglob("*.json"))) == 6
    assert any(warning.startswith("one:qwen3.5:4b:cold:") for warning in result["warnings"])
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["warnings"] == result["warnings"]
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    hotspot_row = next(row for row in summary["rows"] if row["axis"] == "ollama_hotspot")
    assert hotspot_row["structured_valid"] is True
    assert hotspot_row["repetition"]["rate"] == 0
    assert hotspot_row["geometry"]["validity_rate_after_cleanup"] == 1.0
    assert hotspot_row["destinations"]["destination_accuracy"] == 1.0


def test_hotspot_suite_isolates_unavailable_candidate(tmp_path: Path) -> None:
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
    assert {row["model"] for row in result["models"]} == {
        "qwen3.5:4b",
        "qwen3.5:9b-mlx",
        "qwen3.6:35b",
    }
    failed = next(row for row in result["models"] if row["model"] == "qwen3.5:9b-mlx")
    assert failed["cold"]["failure"]["classification"] == "transport_or_service"
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"
    assert not any("qwen3.5:9b-mlx" in stage for stage in manifest["completed_stages"])
    assert (tmp_path / "run" / "manifest.json").is_file()
    assert (tmp_path / "run" / "report.html").is_file()
    assert (tmp_path / "run" / "contact-sheet.png").is_file()
    assert len(list((tmp_path / "run" / "annotations").rglob("*.png"))) == 4


def test_recorded_regressions_exercise_required_failure_modes(tmp_path: Path) -> None:
    def runtime_factory(settings: OllamaSettings) -> OllamaRuntime:
        return OllamaRuntime(settings, client=FakeOllamaClient(settings.model))

    result_path = run_hotspot_evaluation(
        HotspotEvaluationSettings(
            output_dir=tmp_path / "run",
            include_ablations=False,
        ),
        runtime_factory=runtime_factory,
    )

    result = json.loads(result_path.read_text())
    regressions = {row["name"]: row for row in result["recorded_regressions"]}
    assert regressions["repeated-interactions"]["repetition"]["rate"] > 0
    assert regressions["repeated-interactions"]["destinations"]["destination_accuracy"] == 0
    assert regressions["repeated-interactions"]["destinations"]["ambiguous_duplicate_count"] == 1
    assert regressions["supplied-token-unresolved"]["destinations"]["destination_accuracy"] < 1.0
    assert regressions["invalid-geometry"]["geometry"]["validity_rate_before_cleanup"] < 1.0
    assert regressions["schema-saturation"]["schema_limits"]["interaction_limit_reached"]
    assert regressions["token-limit"]["schema_limits"]["token_limit_reached"]
    assert regressions["token-limit"]["response_metadata"]["eval_count"] == 1024
    assert any(
        warning.startswith("recorded_regression:invalid-geometry:")
        for warning in result["warnings"]
    )


def test_hotspot_case_fixture_cannot_escape_root(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    (case_dir / "escape.json").write_text(
        json.dumps(
            {
                "case_id": "escape",
                "image_path": "../outside.png",
                "interaction_description": "Gate",
                "card_catalogue": [{"token": "C1", "name": "Garden"}],
                "expected_hotspots": [{"label": "Gate", "target_token": "C1"}],
            }
        )
    )

    with pytest.raises(ValueError, match="escapes fixture root"):
        load_hotspot_cases(case_dir, fixture_root=fixture_root)


def test_apply_converts_candidate_token_to_persisted_reference() -> None:
    card_id = uuid4()
    proposal = HotspotProposal(
        source_interaction_index=1,
        label="Gate",
        target=ExistingCandidateTarget(
            card_token="C1",
            card_name="Garden",
        ),
        polygons=(
            CandidatePolygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.4, y=0.5),
                )
            ),
        ),
    )

    interaction = apply_hotspot_proposal(
        proposal,
        card_ids_by_token={"C1": card_id},
    )

    assert isinstance(interaction.action.target, ResolvedCardReference)
    assert interaction.action.target.target_card_id == card_id
    assert "C1" not in interaction.model_dump_json()
