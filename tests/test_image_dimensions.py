"""Tests for fixed aspect ratios and square-equivalent resolutions."""

import pytest

from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
    output_dimensions,
)


@pytest.mark.parametrize(
    ("resolution", "aspect_ratio", "expected"),
    [
        (256, AspectRatio.SQUARE, (256, 256)),
        (256, AspectRatio.LANDSCAPE, (288, 224)),
        (256, AspectRatio.PORTRAIT, (224, 288)),
        (256, AspectRatio.WIDESCREEN, (336, 192)),
        (512, AspectRatio.SQUARE, (512, 512)),
        (512, AspectRatio.LANDSCAPE, (592, 448)),
        (512, AspectRatio.PORTRAIT, (448, 592)),
        (512, AspectRatio.WIDESCREEN, (688, 384)),
        (768, AspectRatio.SQUARE, (768, 768)),
        (768, AspectRatio.LANDSCAPE, (880, 672)),
        (768, AspectRatio.PORTRAIT, (672, 880)),
        (768, AspectRatio.WIDESCREEN, (1024, 576)),
        (1024, AspectRatio.SQUARE, (1024, 1024)),
        (1024, AspectRatio.LANDSCAPE, (1184, 880)),
        (1024, AspectRatio.PORTRAIT, (880, 1184)),
        (1024, AspectRatio.WIDESCREEN, (1360, 768)),
    ],
)
def test_output_dimensions_cover_every_fixed_format_and_resolution(
    resolution: int,
    aspect_ratio: AspectRatio,
    expected: tuple[int, int],
) -> None:
    assert output_dimensions(resolution, aspect_ratio) == expected
    assert output_dimensions(GenerateResolution(resolution), aspect_ratio) == expected


@pytest.mark.parametrize("resolution", (0, 255, 300, 2048, True, 512.0, "512"))
def test_output_dimensions_reject_unsupported_resolutions(resolution: object) -> None:
    with pytest.raises(ValueError, match="resolution must be one of"):
        output_dimensions(resolution, AspectRatio.LANDSCAPE)  # type: ignore[arg-type]


def test_output_dimensions_require_typed_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="AspectRatio"):
        output_dimensions(512, "4:3")  # type: ignore[arg-type]
