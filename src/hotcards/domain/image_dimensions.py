"""Fixed stack aspect ratios and image resolution calculations."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from math import floor, sqrt


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


class GenerateResolution(IntEnum):
    """Supported square-equivalent image generation resolutions."""

    RESOLUTION_256 = 256
    RESOLUTION_512 = 512
    RESOLUTION_768 = 768
    RESOLUTION_1024 = 1024


def _nearest_16(value: float) -> int:
    return max(16, floor(value / 16 + 0.5) * 16)


def output_dimensions(
    resolution: GenerateResolution | int,
    aspect_ratio: AspectRatio,
) -> tuple[int, int]:
    """Return REM square-equivalent dimensions aligned to 16 pixels."""
    if not isinstance(aspect_ratio, AspectRatio):
        raise ValueError("aspect_ratio must be a supported AspectRatio")
    if isinstance(resolution, GenerateResolution):
        resolution_value = resolution.value
    elif type(resolution) is int:
        try:
            resolution_value = GenerateResolution(resolution).value
        except ValueError as error:
            supported = ", ".join(str(item.value) for item in GenerateResolution)
            raise ValueError(f"resolution must be one of: {supported}") from error
    else:
        supported = ", ".join(str(item.value) for item in GenerateResolution)
        raise ValueError(f"resolution must be one of: {supported}")
    ratio_width, ratio_height = aspect_ratio.components
    width = sqrt(resolution_value**2 * ratio_width / ratio_height)
    height = sqrt(resolution_value**2 * ratio_height / ratio_width)
    return _nearest_16(width), _nearest_16(height)


__all__ = ["AspectRatio", "GenerateResolution", "output_dimensions"]
