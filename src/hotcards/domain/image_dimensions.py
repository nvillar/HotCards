"""Fixed stack aspect ratios and named long-edge output tiers."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from math import floor


class AspectRatio(StrEnum):
    """One immutable stack-wide card aspect ratio."""

    SQUARE = "1:1"
    LANDSCAPE = "4:3"
    PORTRAIT = "3:4"
    WIDESCREEN = "16:9"

    @property
    def components(self) -> tuple[int, int]:
        """Return the ratio's integer width and height components."""
        width, height = self.value.split(":")
        return int(width), int(height)


class ResolutionTier(IntEnum):
    """Supported image-output tiers expressed as long-edge pixels."""

    SMALL = 256
    MEDIUM = 512
    LARGE = 768
    FULL = 1024

    @property
    def label(self) -> str:
        """Return the exact user-facing tier name."""
        return self.name.title()


def _nearest_16(value: float) -> int:
    return max(16, floor(value / 16 + 0.5) * 16)


def output_dimensions(
    tier: ResolutionTier | int,
    aspect_ratio: AspectRatio,
) -> tuple[int, int]:
    """Return dimensions whose long edge equals the selected tier."""
    if not isinstance(aspect_ratio, AspectRatio):
        raise ValueError("aspect_ratio must be a supported AspectRatio")
    if isinstance(tier, ResolutionTier):
        tier_value = tier.value
    elif type(tier) is int:
        try:
            tier_value = ResolutionTier(tier).value
        except ValueError as error:
            supported = ", ".join(str(item.value) for item in ResolutionTier)
            raise ValueError(f"tier must be one of: {supported}") from error
    else:
        supported = ", ".join(str(item.value) for item in ResolutionTier)
        raise ValueError(f"tier must be one of: {supported}")
    ratio_width, ratio_height = aspect_ratio.components
    if ratio_width == ratio_height:
        return tier_value, tier_value
    if ratio_width > ratio_height:
        return tier_value, _nearest_16(tier_value * ratio_height / ratio_width)
    return _nearest_16(tier_value * ratio_width / ratio_height), tier_value


def higher_output_tiers(
    current_width: int,
    current_height: int,
    aspect_ratio: AspectRatio,
) -> tuple[ResolutionTier, ...]:
    """Return tiers whose derived output area exceeds the current image area."""
    if current_width <= 0 or current_height <= 0:
        raise ValueError("current image dimensions must be positive")
    current_area = current_width * current_height
    return tuple(
        tier
        for tier in ResolutionTier
        if (output_dimensions(tier, aspect_ratio)[0] * output_dimensions(tier, aspect_ratio)[1])
        > current_area
    )


__all__ = [
    "AspectRatio",
    "ResolutionTier",
    "higher_output_tiers",
    "output_dimensions",
]
