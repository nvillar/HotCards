"""Tests for normalized polygon geometry."""

import pytest

from hypergen.domain.geometry import (
    PolygonIssueCode,
    hit_test_interactions,
    model_coordinate_to_normalized,
    model_point_to_document,
    normalized_coordinate_to_model,
    point_in_polygon,
    polygon_signed_area,
    validate_polygon,
)
from hypergen.domain.models import (
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    UnresolvedCardReference,
)


def square(left: float, top: float, right: float, bottom: float) -> Polygon:
    return Polygon(
        points=(
            Point(x=left, y=top),
            Point(x=right, y=top),
            Point(x=right, y=bottom),
            Point(x=left, y=bottom),
        )
    )


def interaction(label: str, polygon: Polygon) -> Interaction:
    return Interaction(
        label=label,
        action=NavigateAction(target=UnresolvedCardReference(target_name=label)),
        polygons=(polygon,),
    )


def issue_codes(points: tuple[tuple[float, float], ...]) -> set[PolygonIssueCode]:
    return {issue.code for issue in validate_polygon(points)}


def test_model_coordinate_conversion_clamps_and_round_trips() -> None:
    assert model_coordinate_to_normalized(-10) == 0.0
    assert model_coordinate_to_normalized(500) == 0.5
    assert model_coordinate_to_normalized(1200) == 1.0
    assert normalized_coordinate_to_model(-0.5) == 0
    assert normalized_coordinate_to_model(0.5) == 500
    assert normalized_coordinate_to_model(1.5) == 1000
    assert model_point_to_document(250, 750) == Point(x=0.25, y=0.75)

    for model_coordinate in (0, 1, 499, 500, 999, 1000):
        normalized = model_coordinate_to_normalized(model_coordinate)
        assert normalized_coordinate_to_model(normalized) == model_coordinate


def test_coordinate_conversion_rejects_invalid_extent() -> None:
    with pytest.raises(ValueError, match="positive"):
        model_coordinate_to_normalized(1, extent=0)
    with pytest.raises(ValueError, match="positive"):
        normalized_coordinate_to_model(0.5, extent=-1)


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (
            ((0.1, 0.1), (0.1, 0.1), (0.2, 0.2)),
            PolygonIssueCode.TOO_FEW_DISTINCT_POINTS,
        ),
        (
            ((0.1, 0.1), (0.9, 0.1), (1.1, 0.9)),
            PolygonIssueCode.OUTSIDE_CANVAS,
        ),
        (
            ((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)),
            PolygonIssueCode.NEAR_ZERO_AREA,
        ),
        (
            ((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)),
            PolygonIssueCode.SELF_INTERSECTION,
        ),
    ],
)
def test_polygon_validation_reports_invalid_geometry(
    points: tuple[tuple[float, float], ...],
    expected: PolygonIssueCode,
) -> None:
    assert expected in issue_codes(points)


def test_polygon_model_rejects_invalid_document_geometry() -> None:
    with pytest.raises(ValueError, match="self-intersect"):
        Polygon(
            points=(
                Point(x=0.1, y=0.1),
                Point(x=0.9, y=0.9),
                Point(x=0.9, y=0.1),
                Point(x=0.1, y=0.9),
            )
        )


def test_area_and_point_in_polygon_include_boundary() -> None:
    polygon = square(0.1, 0.1, 0.8, 0.8)

    assert polygon_signed_area(polygon.points) == pytest.approx(0.49)
    assert point_in_polygon((0.5, 0.5), polygon)
    assert point_in_polygon((0.1, 0.4), polygon)
    assert not point_in_polygon((0.9, 0.5), polygon)


def test_later_interaction_wins_overlap_hit_testing() -> None:
    lower = interaction("Lower", square(0.1, 0.1, 0.8, 0.8))
    upper = interaction("Upper", square(0.4, 0.4, 0.9, 0.9))

    assert hit_test_interactions((0.2, 0.2), (lower, upper)) == lower
    assert hit_test_interactions((0.5, 0.5), (lower, upper)) == upper
    assert hit_test_interactions((0.95, 0.95), (lower, upper)) is None
