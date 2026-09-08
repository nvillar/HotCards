"""Tests for fixed aspect ratios and named long-edge tiers."""

import pytest

from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    higher_output_tiers,
    output_dimensions,
    validate_exact_output_dimensions,
)


@pytest.mark.parametrize(
    ("tier", "aspect_ratio", "expected"),
    [
        (256, AspectRatio.SQUARE, (256, 256)),
        (256, AspectRatio.LANDSCAPE, (256, 192)),
        (256, AspectRatio.PORTRAIT, (192, 256)),
        (256, AspectRatio.WIDESCREEN, (256, 144)),
        (512, AspectRatio.SQUARE, (512, 512)),
        (512, AspectRatio.LANDSCAPE, (512, 384)),
        (512, AspectRatio.PORTRAIT, (384, 512)),
        (512, AspectRatio.WIDESCREEN, (512, 288)),
        (768, AspectRatio.SQUARE, (768, 768)),
        (768, AspectRatio.LANDSCAPE, (768, 576)),
        (768, AspectRatio.PORTRAIT, (576, 768)),
        (768, AspectRatio.WIDESCREEN, (768, 432)),
        (1024, AspectRatio.SQUARE, (1024, 1024)),
        (1024, AspectRatio.LANDSCAPE, (1024, 768)),
        (1024, AspectRatio.PORTRAIT, (768, 1024)),
        (1024, AspectRatio.WIDESCREEN, (1024, 576)),
    ],
)
def test_output_dimensions_cover_every_fixed_format_and_resolution(
    tier: int,
    aspect_ratio: AspectRatio,
    expected: tuple[int, int],
) -> None:
    assert output_dimensions(tier, aspect_ratio) == expected
    assert output_dimensions(ResolutionTier(tier), aspect_ratio) == expected


@pytest.mark.parametrize("tier", (0, 255, 300, 2048, True, 512.0, "512"))
def test_output_dimensions_reject_unsupported_tiers(tier: object) -> None:
    with pytest.raises(ValueError, match="tier must be one of"):
        output_dimensions(tier, AspectRatio.LANDSCAPE)  # type: ignore[arg-type]


def test_output_dimensions_require_typed_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="AspectRatio"):
        output_dimensions(512, "4:3")  # type: ignore[arg-type]


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("source_tier", tuple(ResolutionTier))
def test_higher_output_tiers_filters_by_actual_pixel_area(
    aspect_ratio: AspectRatio,
    source_tier: ResolutionTier,
) -> None:
    width, height = output_dimensions(source_tier, aspect_ratio)

    assert higher_output_tiers(width, height, aspect_ratio) == tuple(
        tier for tier in ResolutionTier if tier > source_tier
    )


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (592, 448, (ResolutionTier.LARGE, ResolutionTier.FULL)),
        (1024, 192, (ResolutionTier.LARGE, ResolutionTier.FULL)),
        (128, 128, tuple(ResolutionTier)),
        (2048, 1536, ()),
    ],
)
def test_higher_output_tiers_uses_area_not_source_long_edge(
    width: int,
    height: int,
    expected: tuple[ResolutionTier, ...],
) -> None:
    assert higher_output_tiers(width, height, AspectRatio.LANDSCAPE) == expected


def test_resolution_tiers_expose_exact_user_facing_names() -> None:
    assert [(tier.label, tier.value) for tier in ResolutionTier] == [
        ("Small", 256),
        ("Medium", 512),
        ("Large", 768),
        ("Full", 1024),
    ]


@pytest.mark.parametrize(
    ("aspect_ratio", "dimensions"),
    [
        (AspectRatio.SQUARE, (592, 592)),
        (AspectRatio.LANDSCAPE, (592, 448)),
        (AspectRatio.PORTRAIT, (448, 592)),
        (AspectRatio.WIDESCREEN, (592, 336)),
    ],
)
def test_exact_output_dimensions_accept_aligned_ratio_rounded_sizes(
    aspect_ratio: AspectRatio,
    dimensions: tuple[int, int],
) -> None:
    assert (
        validate_exact_output_dimensions(
            dimensions[0],
            dimensions[1],
            aspect_ratio,
        )
        == dimensions
    )


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        ((641, 480), "aligned to 16"),
        ((1008, 784), "4:3 stack aspect ratio"),
        ((640, 496), "4:3 stack aspect ratio"),
    ],
)
def test_exact_output_dimensions_reject_unavailable_current_sizes(
    dimensions: tuple[int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_exact_output_dimensions(
            dimensions[0],
            dimensions[1],
            AspectRatio.LANDSCAPE,
        )


def test_higher_output_tiers_rejects_invalid_source_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        higher_output_tiers(0, 768, AspectRatio.LANDSCAPE)
