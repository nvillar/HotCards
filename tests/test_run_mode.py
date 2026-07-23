"""Deterministic tests for session-only stack playback."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession
from hypergen.application.run_session import RunSession
from hypergen.domain.models import (
    Card,
    HotspotSet,
    ImageOrigin,
    ImageRevision,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    RunOverlayMode,
    Stack,
    UnresolvedCardReference,
)
from hypergen.storage.stack_store import StackStore
from hypergen.ui.main_window import MainWindow


class FakeWorkers(QObject):
    availability_changed = Signal(object)


class FakeBackgroundWorkflow(QObject):
    candidate_changed = Signal(object)
    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.candidate = None
        self.busy = False

    def discard_candidate(self) -> None:
        self.candidate = None

    def close(self) -> None:
        pass


class FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(
        self,
        key: str,
        default: object = None,
        type: type[object] | None = None,
    ) -> object:
        value = self.values.get(key, default)
        return type(value) if type is not None else value

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def interaction(
    label: str,
    target: ResolvedCardReference | UnresolvedCardReference,
) -> Interaction:
    return Interaction(
        label=label,
        action=NavigateAction(target=target),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.5),
                )
            ),
        ),
    )


def run_stack() -> tuple[Stack, Interaction, Interaction]:
    second = Card(name="Second")
    third = Card(name="Third")
    to_second = interaction(
        "Second",
        ResolvedCardReference(target_card_id=second.id),
    )
    unresolved = interaction(
        "Missing",
        UnresolvedCardReference(target_name="Missing room"),
    )
    revision = ImageRevision(
        image_path="assets/cards/first/background.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=(to_second, unresolved)),
        created_at=datetime.now(UTC),
    )
    first = Card(
        name="First",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    return (
        Stack(
            name="Run",
            cards=(first, second, third),
            start_card_id=first.id,
        ),
        to_second,
        unresolved,
    )


def test_run_session_navigates_uuid_links_with_back_and_restart() -> None:
    document, to_second, _unresolved = run_stack()
    session = RunSession()

    state = session.start(document, document.cards[2].id)
    assert state.current_card_id == document.start_card_id
    assert state.warning is None

    state = session.navigate(document, to_second.id)
    assert state.current_card_id == document.cards[1].id
    assert state.history == (document.cards[0].id,)

    state = session.back()
    assert state.current_card_id == document.cards[0].id
    assert state.history == ()

    session.navigate(document, to_second.id)
    state = session.restart()
    assert state.current_card_id == document.start_card_id
    assert state.history == ()


def test_run_session_warns_without_moving_for_unresolved_or_stale_links() -> None:
    document, _to_second, unresolved = run_stack()
    session = RunSession()
    session.start(document)

    state = session.navigate(document, unresolved.id)
    assert state.current_card_id == document.cards[0].id
    assert state.warning == 'Link to "Missing room" is unresolved.'

    state = session.navigate(document, document.cards[1].id)
    assert state.current_card_id == document.cards[0].id
    assert state.warning == "That hotspot is no longer available on this card."


def test_run_session_uses_current_or_first_card_when_start_is_missing() -> None:
    first = Card(name="First")
    second = Card(name="Second")
    document = Stack(name="No start", cards=(first, second))
    session = RunSession()

    state = session.start(document, second.id)
    assert state.current_card_id == second.id
    assert state.warning == 'No start card is configured; previewing "Second".'
    assert session.restart().current_card_id == second.id

    state = session.start(Stack(name="Empty"))
    assert state.current_card_id is None
    assert state.warning == "This stack has no cards to run."


def polygon(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Polygon:
    return Polygon(
        points=(
            Point(x=left, y=top),
            Point(x=right, y=top),
            Point(x=(left + right) / 2, y=bottom),
        )
    )


def build_run_window(
    tmp_path: Path,
) -> tuple[
    MainWindow,
    DocumentSession,
    Interaction,
    Interaction,
    Interaction,
]:
    first = Card(name="First")
    incomplete = Card(name="Incomplete")
    third = Card(name="Third")
    to_incomplete = Interaction(
        label="Incomplete",
        action=NavigateAction(
            target=ResolvedCardReference(target_card_id=incomplete.id)
        ),
        polygons=(polygon(0.1, 0.1, 0.55, 0.55),),
    )
    to_third = Interaction(
        label="Third",
        action=NavigateAction(
            target=ResolvedCardReference(target_card_id=third.id)
        ),
        polygons=(polygon(0.2, 0.15, 0.5, 0.5),),
    )
    unresolved = Interaction(
        label="Missing",
        action=NavigateAction(
            target=UnresolvedCardReference(target_name="Missing room")
        ),
        polygons=(polygon(0.1, 0.1, 0.5, 0.5),),
    )
    bundle = tmp_path / "Run.hypergen"
    store = StackStore(bundle)

    def revision_for(
        card: Card,
        name: str,
        color: str,
        interactions: tuple[Interaction, ...],
    ) -> ImageRevision:
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (1024, 768), color).save(image)
        revision_id = uuid4()
        image_path = store.import_image(
            image,
            card_id=card.id,
            revision_id=revision_id,
        )
        return ImageRevision(
            id=revision_id,
            image_path=image_path,
            origin=ImageOrigin.IMPORTED,
            hotspot_set=HotspotSet(interactions=interactions),
            created_at=datetime.now(UTC),
        )

    first_revision = revision_for(
        first,
        "first",
        "navy",
        (to_incomplete, to_third),
    )
    third_revision = revision_for(
        third,
        "third",
        "green",
        (unresolved,),
    )
    first = first.model_copy(
        update={
            "image_revisions": (first_revision,),
            "active_revision_id": first_revision.id,
        }
    )
    third = third.model_copy(
        update={
            "image_revisions": (third_revision,),
            "active_revision_id": third_revision.id,
        }
    )
    store.save(
        Stack(
            name="Run",
            cards=(first, incomplete, third),
            start_card_id=first.id,
        )
    )
    controller = DocumentController(Stack(name="Bootstrap"))
    session = DocumentSession(controller)
    session.open(bundle)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(),  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    return window, session, to_incomplete, to_third, unresolved


def test_run_canvas_uses_topmost_hit_with_back_and_restart(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, _session, _to_incomplete, to_third, unresolved = (
        build_run_window(tmp_path)
    )
    window.resize(1000, 700)
    window.show()
    window.mode_selector.setCurrentText("Run")
    application.processEvents()

    assert window.canvas_card_name.text() == "First"
    assert window.card_sidebar.isHidden()
    assert window.inspector.isHidden()
    overlap = window.card_canvas.viewport_point_for(QPointF(0.3, 0.3))
    QTest.mouseMove(window.card_canvas.viewport(), overlap)
    assert window.card_canvas._hovered_interaction_id == to_third.id
    assert window.card_canvas.viewport().cursor().shape() == (
        Qt.CursorShape.PointingHandCursor
    )
    assert window.card_canvas._overlay_items == []
    assert window.card_canvas._run_interaction_at(QPointF(-0.1, 0.3)) is None

    QTest.mouseClick(
        window.card_canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=overlap,
    )
    assert window.canvas_card_name.text() == "Third"
    assert window.back_action.isEnabled()

    unresolved_point = window.card_canvas.viewport_point_for(
        QPointF(0.2, 0.2)
    )
    QTest.mouseClick(
        window.card_canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=unresolved_point,
    )
    assert window.canvas_card_name.text() == "Third"
    assert window.run_status_label.text() == (
        'Link to "Missing room" is unresolved.'
    )

    window.back_action.trigger()
    assert window.canvas_card_name.text() == "First"
    window.card_canvas.interaction_activated.emit(to_third.id)
    window.restart_action.trigger()
    assert window.canvas_card_name.text() == "First"
    assert not window.back_action.isEnabled()
    assert unresolved.id != to_third.id
    window.close()


def test_run_overlays_persist_and_incomplete_cards_warn(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, session, to_incomplete, _to_third, _unresolved = (
        build_run_window(tmp_path)
    )
    window.resize(1000, 700)
    window.show()
    window.mode_selector.setCurrentText("Run")
    application.processEvents()

    window.overlay_selector.setCurrentText("Visible")
    assert window.controller.document.run_overlay_mode is RunOverlayMode.VISIBLE
    assert len(window.card_canvas._overlay_items) == 2
    assert session.flush()
    assert StackStore(session.state.bundle_path).load().run_overlay_mode is (
        RunOverlayMode.VISIBLE
    )

    window.overlay_selector.setCurrentText("On hover")
    assert window.card_canvas._overlay_items == []
    point = window.card_canvas.viewport_point_for(QPointF(0.3, 0.3))
    QTest.mouseMove(window.card_canvas.viewport(), window.card_canvas.rect().topLeft())
    QTest.mouseMove(window.card_canvas.viewport(), point)
    assert len(window.card_canvas._overlay_items) == 1

    window.card_canvas.interaction_activated.emit(to_incomplete.id)
    assert window.canvas_card_name.text() == "Incomplete"
    assert window.card_canvas._message_item is not None
    assert window.card_canvas._message_item.toPlainText() == (
        "No background revision"
    )
    assert window.run_status_label.text() == (
        '"Incomplete" has no active background revision.'
    )
    window.close()
