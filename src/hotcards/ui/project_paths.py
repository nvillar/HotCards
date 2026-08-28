"""Machine-local paths used by project lifecycle dialogs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths


def default_project_directory(
    documents_directory: Path | None = None,
) -> Path:
    """Return the default parent directory for HotCards stack bundles."""
    if documents_directory is None:
        location = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        documents_directory = Path(location) if location else Path.home() / "Documents"
    return documents_directory / "HotCards"


def bundle_path(path: str | Path) -> Path:
    """Ensure a selected project path uses the directory-bundle suffix."""
    selected = Path(path)
    if selected.suffix.casefold() == ".hotcards":
        return selected
    return selected.with_suffix(".hotcards")


__all__ = ["bundle_path", "default_project_directory"]
