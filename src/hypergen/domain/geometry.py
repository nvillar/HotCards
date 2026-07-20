"""Geometry conversion, validation, and hit testing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from hypergen.domain.models import Interaction, Point, Polygon

DEFAULT_MODEL_COORDINATE_EXTENT = 1000
MIN_NORMALIZED_POLYGON_AREA = 1e-6
GEOMETRY_EPSILON = 1e-12

PointLike = Point | tuple[float, float]


class PolygonIssueCode(StrEnum):
    """Stable identifiers for polygon validation findings."""

    TOO_FEW_DISTINCT_POINTS = "too_few_distinct_points"
    REPEATED_VERTEX = "repeated_vertex"
    NEAR_ZERO_AREA = "near_zero_area"
    SELF_INTERSECTION = "self_intersection"
    OUTSIDE_CANVAS = "outside_canvas"


@dataclass(frozen=True, slots=True)
class PolygonIssue:
    """One actionable polygon validation finding."""

    code: PolygonIssueCode
    message: str


def _coordinates(point: PointLike) -> tuple[float, float]:
    if isinstance(point, Point):
        return point.x, point.y
    return point


def model_coordinate_to_normalized(
    coordinate: int,
    *,
    extent: int = DEFAULT_MODEL_COORDINATE_EXTENT,
) -> float:
    """Convert and clamp a model-space coordinate to normalized document space."""
    if extent <= 0:
        raise ValueError("coordinate extent must be positive")
    return min(max(coordinate / extent, 0.0), 1.0)


def normalized_coordinate_to_model(
    coordinate: float,
    *,
    extent: int = DEFAULT_MODEL_COORDINATE_EXTENT,
) -> int:
    """Convert and clamp a normalized document coordinate to model space."""
    if extent <= 0:
        raise ValueError("coordinate extent must be positive")
    return round(min(max(coordinate, 0.0), 1.0) * extent)


def model_point_to_document(
    x: int,
    y: int,
    *,
    extent: int = DEFAULT_MODEL_COORDINATE_EXTENT,
) -> Point:
    """Convert one model-space point into a validated document point."""
    return Point(
        x=model_coordinate_to_normalized(x, extent=extent),
        y=model_coordinate_to_normalized(y, extent=extent),
    )


def polygon_signed_area(points: Sequence[PointLike]) -> float:
    """Return the signed shoelace area of a polygon."""
    coordinates = tuple(_coordinates(point) for point in points)
    if len(coordinates) < 3:
        return 0.0
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(
            coordinates,
            coordinates[1:] + coordinates[:1],
            strict=True,
        )
    )


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _on_segment(
    first: tuple[float, float],
    point: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return (
        min(first[0], second[0]) - GEOMETRY_EPSILON
        <= point[0]
        <= max(first[0], second[0]) + GEOMETRY_EPSILON
        and min(first[1], second[1]) - GEOMETRY_EPSILON
        <= point[1]
        <= max(first[1], second[1]) + GEOMETRY_EPSILON
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    first_opposes = (orientations[0] > GEOMETRY_EPSILON) != (orientations[1] > GEOMETRY_EPSILON)
    second_opposes = (orientations[2] > GEOMETRY_EPSILON) != (orientations[3] > GEOMETRY_EPSILON)
    if (
        abs(orientations[0]) > GEOMETRY_EPSILON
        and abs(orientations[1]) > GEOMETRY_EPSILON
        and abs(orientations[2]) > GEOMETRY_EPSILON
        and abs(orientations[3]) > GEOMETRY_EPSILON
    ):
        return first_opposes and second_opposes
    return (
        (
            abs(orientations[0]) <= GEOMETRY_EPSILON
            and _on_segment(first_start, second_start, first_end)
        )
        or (
            abs(orientations[1]) <= GEOMETRY_EPSILON
            and _on_segment(first_start, second_end, first_end)
        )
        or (
            abs(orientations[2]) <= GEOMETRY_EPSILON
            and _on_segment(second_start, first_start, second_end)
        )
        or (
            abs(orientations[3]) <= GEOMETRY_EPSILON
            and _on_segment(second_start, first_end, second_end)
        )
    )


def _has_self_intersection(coordinates: tuple[tuple[float, float], ...]) -> bool:
    edge_count = len(coordinates)
    for first_index in range(edge_count):
        first_end_index = (first_index + 1) % edge_count
        for second_index in range(first_index + 1, edge_count):
            second_end_index = (second_index + 1) % edge_count
            if first_index == second_index or first_end_index == second_index:
                continue
            if second_end_index == first_index:
                continue
            if _segments_intersect(
                coordinates[first_index],
                coordinates[first_end_index],
                coordinates[second_index],
                coordinates[second_end_index],
            ):
                return True
    return False


def validate_polygon(
    points: Sequence[PointLike],
    *,
    minimum_area: float = MIN_NORMALIZED_POLYGON_AREA,
) -> tuple[PolygonIssue, ...]:
    """Return every deterministic validity issue for a simple polygon."""
    coordinates = tuple(_coordinates(point) for point in points)
    issues: list[PolygonIssue] = []
    distinct_coordinates = set(coordinates)
    if len(distinct_coordinates) < 3:
        issues.append(
            PolygonIssue(
                PolygonIssueCode.TOO_FEW_DISTINCT_POINTS,
                "polygon must contain at least three distinct points",
            )
        )
    if len(distinct_coordinates) != len(coordinates):
        issues.append(
            PolygonIssue(
                PolygonIssueCode.REPEATED_VERTEX,
                "polygon must not repeat a vertex",
            )
        )
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in coordinates):
        issues.append(
            PolygonIssue(
                PolygonIssueCode.OUTSIDE_CANVAS,
                "polygon coordinates must remain inside the normalized canvas",
            )
        )
    if abs(polygon_signed_area(coordinates)) <= minimum_area:
        issues.append(
            PolygonIssue(
                PolygonIssueCode.NEAR_ZERO_AREA,
                "polygon area is too small",
            )
        )
    if len(coordinates) >= 4 and _has_self_intersection(coordinates):
        issues.append(
            PolygonIssue(
                PolygonIssueCode.SELF_INTERSECTION,
                "polygon edges must not self-intersect",
            )
        )
    return tuple(issues)


def point_in_polygon(point: PointLike, polygon: Polygon) -> bool:
    """Return whether a point lies inside or on the boundary of a polygon."""
    x, y = _coordinates(point)
    coordinates = tuple(_coordinates(vertex) for vertex in polygon.points)
    inside = False
    previous = coordinates[-1]
    for current in coordinates:
        if abs(_orientation(previous, current, (x, y))) <= GEOMETRY_EPSILON and _on_segment(
            previous, (x, y), current
        ):
            return True
        if (current[1] > y) != (previous[1] > y):
            crossing_x = (previous[0] - current[0]) * (y - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def hit_test_interactions(
    point: PointLike,
    interactions: Sequence[Interaction],
) -> Interaction | None:
    """Return the topmost interaction containing a point."""
    for interaction in reversed(interactions):
        if any(point_in_polygon(point, polygon) for polygon in interaction.polygons):
            return interaction
    return None
