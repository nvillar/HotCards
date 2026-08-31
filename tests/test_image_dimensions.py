"""Tests for fixed aspect ratios and named long-edge tiers."""

import pytest

from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    higher_output_tiers,
    output_dimensions,
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


def test_full_landscape_pixels_are_the_maximum_tier() -> None:
    assert (
        higher_output_tiers(
            1024,
            768,
            AspectRatio.LANDSCAPE,
        )
        == ()
    )


def test_resolution_tiers_expose_exact_user_facing_names() -> None:
    assert [(tier.label, tier.value) for tier in ResolutionTier] == [
        ("Small", 256),
        ("Medium", 512),
        ("Large", 768),
        ("Full", 1024),
    ]


def test_higher_output_tiers_rejects_invalid_source_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        higher_output_tiers(0, 768, AspectRatio.LANDSCAPE)
