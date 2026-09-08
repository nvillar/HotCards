"""Tests for offline, escaped evaluation report rendering."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from hotcards.domain.image_dimensions import ResolutionTier
from hotcards.domain.models import (
    GenerateInputs,
    GenerateOperation,
    ImageOperationSettings,
    ImageOriginFacts,
    ImageProvenance,
    PresetOutputSize,
)
from hotcards.evaluation.reports import render_reports

PROMPT = 'A courtyard with a sign reading "<script>alert(1)</script>".'


def _image_result(run_dir: Path, artifact_path: str) -> Path:
    provenance = ImageProvenance(
        authoring=GenerateOperation(
            inputs=GenerateInputs(
                description=PROMPT, output_size=PresetOutputSize(tier=ResolutionTier.SMALL)
            ),
        ),
        origin=ImageOriginFacts(
            render_prompt=PROMPT,
            settings=ImageOperationSettings(
                model_identifier="flux2-klein-4b",
                mflux_version="test",
                seed=42,
                width=256,
                height=192,
                step_count=4,
                generated_at=datetime(2026, 7, 19, tzinfo=UTC),
                duration_seconds=0,
            ),
        ),
    )
    result = {
        "result_version": "image-result-v2",
        "suite": "images",
        "status": "success",
        "mflux_axis": [
            {
                "case_id": "<script>alert(1)</script>",
                "model": "flux2-klein-4b",
                "render_prompt": PROMPT,
                "cold": {
                    "status": "success",
                    "artifact_path": artifact_path,
                    "queue_duration_seconds": 0,
                    "load_duration_seconds": 0,
                    "inference_duration_seconds": 0,
                    "serialization_duration_seconds": 0,
                    "total_duration_seconds": 0,
                    "metadata": provenance.model_dump(mode="json"),
                },
                "warm": {"status": "failed", "failure": {"classification": "test"}},
                "rubric": {"scene_fidelity": None},
            }
        ],
    }
    path = run_dir / "image-results.json"
    path.write_text(json.dumps(result))
    return path


def test_reports_are_offline_structured_escaped_and_include_contact_sheet(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    image_path = run_dir / "images" / "one.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (256, 192), "navy").save(image_path)

    paths = render_reports(_image_result(run_dir, "images/one.png"))

    summary = json.loads(paths["json"].read_text())
    assert summary["human_quality"]["primary_assessment"] is True
    assert summary["rows"][0]["mflux_model"] == "flux2-klein-4b"
    assert summary["rows"][0]["render_prompt"] == PROMPT
    assert summary["rows"][0]["elapsed_seconds"] == 0
    assert summary["rows"][0]["artifact_path"] == "images/one.png"
    assert summary["stage_metrics"] == [
        {"stage": "mflux", "count": 2, "failure_rate": 0.5, "timeout_rate": 0}
    ]
    with paths["csv"].open(newline="") as stream:
        row = next(csv.DictReader(stream))
        assert row["human_rubric"] == '{"scene_fidelity": null}'
        assert row["render_prompt"] == PROMPT
        assert row["elapsed_seconds"] == "0"
        assert row["artifact_path"] == "images/one.png"
    report = paths["html"].read_text()
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "UNSCORED / null" in report
    assert 'href="images/one.png"' in report
    assert 'src="images/one.png"' in report
    assert 'href="contact-sheet.png"' in report
    assert paths["contact_sheet"].is_file()


def test_reports_reject_artifact_path_traversal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="escapes run directory"):
        render_reports(_image_result(run_dir, "../outside.png"))


def test_reports_reject_obsolete_result_versions(tmp_path: Path) -> None:
    result_path = _image_result(tmp_path, "images/one.png")
    result = json.loads(result_path.read_text())
    result["result_version"] = "image-result-v1"
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="unsupported evaluation result version"):
        render_reports(result_path)
