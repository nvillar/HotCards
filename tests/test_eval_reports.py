"""Tests for offline, escaped evaluation report rendering."""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from hypergen.evaluation.reports import render_reports


def _image_result(run_dir: Path, artifact_path: str) -> Path:
    result = {
        "result_version": "image-result-v1",
        "suite": "images",
        "status": "success",
        "prompt_axis": [],
        "prompt_downstream_axis": [
            {
                "case_id": "<script>alert(1)</script>",
                "ollama_model": "qwen3.5:4b",
                "mflux_model": "flux2-klein-4b",
                "generation": {
                    "status": "success",
                    "artifact_path": artifact_path,
                    "inference_duration_seconds": 1.5,
                },
                "rubric": {"scene_fidelity": None},
            }
        ],
        "mflux_axis": [],
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
    Image.new("RGB", (32, 24), "navy").save(image_path)

    paths = render_reports(_image_result(run_dir, "images/one.png"))

    summary = json.loads(paths["json"].read_text())
    assert summary["human_quality"]["primary_assessment"] is True
    assert summary["rows"][0]["ollama_model"] == "qwen3.5:4b"
    assert summary["rows"][0]["mflux_model"] == "flux2-klein-4b"
    with paths["csv"].open(newline="") as stream:
        assert next(csv.DictReader(stream))["human_rubric"] == '{"scene_fidelity": null}'
    report = paths["html"].read_text()
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "UNSCORED / null" in report
    assert paths["contact_sheet"].is_file()


def test_reports_reject_artifact_path_traversal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="escapes run directory"):
        render_reports(_image_result(run_dir, "../outside.png"))
