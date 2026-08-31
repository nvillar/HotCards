"""Validation helpers for image files used by application workflows."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError


class UnreadableImageError(ValueError):
    """An expected image file cannot be decoded completely."""


def require_readable_image(path: Path) -> None:
    """Raise when an image cannot be identified and fully decoded."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as error:
        raise UnreadableImageError(str(error)) from error


def readable_image_dimensions(path: Path) -> tuple[int, int]:
    """Return exact decoded image dimensions or raise for an unreadable file."""
    require_readable_image(path)
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as error:
        raise UnreadableImageError(str(error)) from error


__all__ = [
    "UnreadableImageError",
    "readable_image_dimensions",
    "require_readable_image",
]
