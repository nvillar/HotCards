"""Tests for the coordinate-grid Remap experiment."""

from pathlib import Path

from PIL import Image

from hypergen.domain.models import Point, Polygon
from hypergen.evaluation.remap_grid import (
    _bbox_iou,
    _model_artifact_id,
    _polygon_iou,
    _summaries,
    render_coordinate_grid,
)


def _polygon(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[Polygon, ...]:
    return (
        Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        ),
    )


def test_grid_overlay_preserves_dimensions_and_changes_pixels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "grid.png"
    Image.new("RGB", (100, 80), "white").save(source)

    render_coordinate_grid(source, output, divisions=10)

    with Image.open(source) as original, Image.open(output) as gridded:
        assert gridded.size == original.size
        assert gridded.tobytes() != original.tobytes()


def test_overlap_metrics_reward_exact_geometry() -> None:
    reference = _polygon(0.1, 0.2, 0.5, 0.6)
    shifted = _polygon(0.3, 0.2, 0.7, 0.6)

    assert _polygon_iou(reference, reference, (100, 80)) == 1.0
    assert _bbox_iou(reference, reference) == 1.0
    assert 0.0 < _polygon_iou(reference, shifted, (100, 80)) < 1.0
    assert 0.0 < _bbox_iou(reference, shifted) < 1.0


def test_summary_scores_failed_trials_as_zero() -> None:
    summary = _summaries(
        [
            {
                "model": "model",
                "grid_divisions": 10,
                "status": "success",
                "mean_polygon_iou": 0.8,
                "mean_bbox_iou": 0.6,
                "mapping_rate": 1.0,
            },
            {
                "model": "model",
                "grid_divisions": 10,
                "status": "failed",
            },
        ]
    )[0]

    assert summary["success_rate"] == 0.5
    assert summary["mean_polygon_iou"] == 0.4
    assert summary["successful_mean_polygon_iou"] == 0.8


def test_model_artifact_ids_are_collision_resistant() -> None:
    first = _model_artifact_id("foo:bar-baz")
    second = _model_artifact_id("foo-bar:baz")

    assert first != second
    assert first.startswith("foo-bar-baz-")
    assert second.startswith("foo-bar-baz-")
