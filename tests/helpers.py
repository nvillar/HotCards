"""Small typed fixtures shared by document and asset tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID, uuid4

import pytest

from hotcards.domain.models import (
    GeneratedBackground,
    GenerateInputs,
    GenerateOperation,
    ImageOperationSettings,
    ImageOriginFacts,
    ImageProvenance,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

    from hotcards.application.document_controller import DocumentController
    from hotcards.application.document_session import DocumentSession


class DocumentSessionFactory(Protocol):
    def __call__(
        self,
        controller: DocumentController,
        *,
        debounce_milliseconds: int = 500,
    ) -> DocumentSession: ...


@pytest.fixture
def document_session_factory(
    qt_application: QApplication,
) -> Iterator[DocumentSessionFactory]:
    from PySide6.QtCore import QCoreApplication, QEvent, QTimer

    from hotcards.application.document_session import DocumentSession

    sessions: list[DocumentSession] = []

    def create(
        controller: DocumentController, *, debounce_milliseconds: int = 500
    ) -> DocumentSession:
        session = DocumentSession(controller, debounce_milliseconds=debounce_milliseconds)
        sessions.append(session)
        return session

    try:
        yield create
    finally:
        for session in reversed(sessions):
            for timer in session.findChildren(QTimer):
                timer.stop()
            session.controller.set_autosave_hook(None)
            session.controller.set_owned_asset_release_hook(None)
            session.deleteLater()
            QCoreApplication.sendPostedEvents(session, QEvent.Type.DeferredDelete)


def generated_background(
    card_id: UUID,
    *,
    description: str = "Scene",
) -> GeneratedBackground:
    """Build current Medium landscape provenance without writing any image bytes."""
    asset_id = uuid4()
    generated_at = datetime(2026, 1, 1, tzinfo=UTC)
    return GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/{card_id}/image-{asset_id}.png",
        provenance=ImageProvenance(
            origin=ImageOriginFacts(
                render_prompt=description,
                settings=ImageOperationSettings(
                    model_identifier="test",
                    mflux_version="test",
                    dependency_versions={"mflux": "test"},
                    seed=7,
                    width=512,
                    height=384,
                    step_count=4,
                    generated_at=generated_at,
                    duration_seconds=1,
                ),
            ),
            authoring=GenerateOperation(inputs=GenerateInputs(description=description)),
        ),
        created_at=generated_at,
    )
