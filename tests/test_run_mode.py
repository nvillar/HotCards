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

from hotcards.application.document_controller import DocumentController
from hotcards.application.document_session import DocumentSession
from hotcards.application.run_session import RunSession
from hotcards.domain.models import (
    Card,
    CardRevision,
    DirectGenerateProvenance,
    GeneratedBackground,
    GenerateInputs,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    RunOverlayMode,
    Stack,
    UnresolvedCardReference,
)
from hotcards.storage.stack_store import StackStore
from hotcards.ui.main_window import MainWindow


class FakeWorkers(QObject):
    availability_changed = Signal(object)


class FakeBackgroundWorkflow(QObject):
    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    generation_progress_changed = Signal(int, int)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    generation_applied = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.busy = False

    def close(self) -> None:
        pass

    def is_generating_for(self, _card_id: object) -> bool:
        return False

    def cancel(self) -> None:
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
    revision = CardRevision(
        hotspot_set=HotspotSet(interactions=(to_second, unresolved)),
    )
    first = Card(
        name="First",
        revisions=(revision,),
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
    assert state.current_card_id == document.cards[2].id
    assert state.warning is None
    assert session.restart().current_card_id == document.start_card_id

    session.start(document)
    state = session.activate(document, to_second.id)
    assert state.current_card_id == document.cards[1].id
    assert state.history == (document.cards[0].id,)

    state = session.back()
    assert state.current_card_id == document.cards[0].id
    assert state.history == ()

    session.activate(document, to_second.id)
    state = session.restart()
    assert state.current_card_id == document.start_card_id
    assert state.history == ()


def test_run_session_warns_without_moving_for_unresolved_or_stale_links() -> None:
    document, _to_second, unresolved = run_stack()
    session = RunSession()
    session.start(document)

    state = session.activate(document, unresolved.id)
    assert state.current_card_id == document.cards[0].id
    assert state.warning == 'Link to "Missing room" is unresolved.'

    state = session.activate(document, document.cards[1].id)
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


def test_run_session_filters_conditions_and_applies_keys_before_navigation() -> None:
    red_key = KeyDefinition(name="Red key")
    visited = KeyDefinition(name="Visited castle")
    destination = Card(name="Castle")
    take_key = Interaction(
        key_changes=HotspotKeyChanges(grant=(red_key.id,)),
    )
    enter = Interaction(
        conditions=HotspotConditions(
            requires=(red_key.id,),
            forbids=(visited.id,),
        ),
        key_changes=HotspotKeyChanges(
            remove=(red_key.id,),
            grant=(visited.id,),
        ),
        action=NavigateAction(
            target=ResolvedCardReference(target_card_id=destination.id)
        ),
    )
    placeholder = Interaction()
    revision = CardRevision(
        hotspot_set=HotspotSet(interactions=(take_key, enter, placeholder))
    )
    source = Card(
        name="Start",
        revisions=(revision,),
        active_revision_id=revision.id,
    )
    document = Stack(
        name="Run",
        keys=(red_key, visited),
        cards=(source, destination),
        start_card_id=source.id,
    )
    session = RunSession()

    state = session.start(document)
    active = session.active_hotspot_set(document)
    assert active is not None
    assert [item.id for item in active.interactions] == [take_key.id]
    assert state.keys == frozenset()

    state = session.activate(document, take_key.id)
    assert state.keys == frozenset({red_key.id})
    assert state.notice == "Granted Red key"
    active = session.active_hotspot_set(document)
    assert active is not None
    assert [item.id for item in active.interactions] == [take_key.id, enter.id]

    state = session.activate(document, enter.id)
    assert state.current_card_id == destination.id
    assert state.keys == frozenset({visited.id})
    assert state.notice is None
    assert session.back().keys == frozenset({visited.id})
    restarted = session.restart()
    assert restarted.current_card_id == source.id
    assert restarted.keys == frozenset()


def test_run_session_rechecks_conditions_and_changes_keys_before_link_warning() -> None:
    red_key = KeyDefinition(name="Red key")
    interaction = Interaction(
        conditions=HotspotConditions(forbids=(red_key.id,)),
        key_changes=HotspotKeyChanges(grant=(red_key.id,)),
        action=NavigateAction(
            target=UnresolvedCardReference(target_name="Missing room")
        ),
    )
    revision = CardRevision(
        hotspot_set=HotspotSet(interactions=(interaction,))
    )
    source = Card(
        name="Start",
        revisions=(revision,),
        active_revision_id=revision.id,
    )
    document = Stack(
        name="Run",
        keys=(red_key,),
        cards=(source,),
        start_card_id=source.id,
    )
    session = RunSession()
    session.start(document)

    state = session.activate(document, interaction.id)
    assert state.keys == frozenset({red_key.id})
    assert state.warning == 'Link to "Missing room" is unresolved.'
    state = session.activate(document, interaction.id)
    assert state.keys == frozenset({red_key.id})
    assert state.warning is None


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
    bundle = tmp_path / "Run.hotcards"
    store = StackStore(bundle)

    def revision_for(
        card: Card,
        name: str,
        color: str,
        interactions: tuple[Interaction, ...],
    ) -> CardRevision:
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (1024, 768), color).save(image)
        asset_id = uuid4()
        image_path = store.store_image_asset(
            image,
            card_id=card.id,
            asset_id=asset_id,
        )
        generated_at = datetime.now(UTC)
        return CardRevision(
            background=GeneratedBackground(
                id=asset_id,
                image_path=image_path,
                provenance=DirectGenerateProvenance(
                    inputs=GenerateInputs(
                        description=name,
                    ),
                    render_prompt=name,
                    settings=ImageOperationSettings(
                        model_identifier="test",
                        mflux_version="test",
                        seed=1,
                        width=592,
                        height=448,
                        step_count=4,
                        generated_at=generated_at,
                        duration_seconds=1,
                    ),
                ),
                created_at=generated_at,
            ),
            hotspot_set=HotspotSet(interactions=interactions),
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
            "revisions": (first_revision,),
            "active_revision_id": first_revision.id,
        }
    )
    third = third.model_copy(
        update={
            "revisions": (third_revision,),
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
    window.mode_button.click()
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
    assert window.notification_bar.message_label.text() == (
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


def test_run_canvas_skips_inactive_overlapping_hotspots_before_z_order(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, _session, to_incomplete, to_third, _unresolved = build_run_window(
        tmp_path
    )
    locked = KeyDefinition(name="Door unlocked")
    document = window.controller.document
    first = document.cards[0]
    revision = first.active_revision
    assert revision.hotspot_set is not None
    interactions = tuple(
        interaction.model_copy(
            update={
                "conditions": HotspotConditions(requires=(locked.id,))
            }
        )
        if interaction.id == to_third.id
        else interaction
        for interaction in revision.hotspot_set.interactions
    )
    revision = revision.model_copy(
        update={"hotspot_set": HotspotSet(interactions=interactions)}
    )
    first = first.model_copy(
        update={
            "revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    window.controller.replace_document(
        document.model_copy(
            update={
                "keys": (locked,),
                "cards": (first, *document.cards[1:]),
            }
        )
    )
    window.render_document()
    window.resize(1000, 700)
    window.show()
    window.mode_button.click()
    application.processEvents()

    overlap = window.card_canvas.viewport_point_for(QPointF(0.3, 0.3))
    QTest.mouseMove(
        window.card_canvas.viewport(),
        window.card_canvas.rect().topLeft(),
    )
    QTest.mouseMove(window.card_canvas.viewport(), overlap)
    assert window.card_canvas._hovered_interaction_id == to_incomplete.id

    QTest.mouseClick(
        window.card_canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=overlap,
    )
    assert window.canvas_card_name.text() == "Incomplete"
    window.close()


def test_run_starts_from_current_card_and_toolbar_can_restart(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, _session, _to_incomplete, _to_third, _unresolved = (
        build_run_window(tmp_path)
    )
    third = window.controller.document.cards[2]
    start = window.controller.document.cards[0]
    window.select_card(third.id)

    window.mode_button.click()
    application.processEvents()

    assert window.canvas_card_name.text() == "Third"
    assert window.notification_bar.current_key != "run-entry"

    window.restart_button.click()

    assert window.canvas_card_name.text() == "First"
    assert window._run_session.state.current_card_id == start.id
    window.close()


def test_run_card_without_hotspots_does_not_warn(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, _session, _to_incomplete, _to_third, _unresolved = (
        build_run_window(tmp_path)
    )
    document = window.controller.document
    third = document.cards[2]
    revision = third.active_revision.model_copy(
        update={"hotspot_set": HotspotSet()}
    )
    third = third.model_copy(
        update={
            "revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    window.controller.replace_document(
        document.model_copy(
            update={
                "cards": (
                    document.cards[0],
                    document.cards[1],
                    third,
                )
            }
        )
    )
    window.render_document()
    window.select_card(third.id)

    window.mode_button.click()
    application.processEvents()

    assert window.canvas_card_name.text() == "Third"
    assert window.notification_bar.current_key != "run-warning"
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
    window.mode_button.click()
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
        "No image"
    )
    assert window.notification_bar.message_label.text() == (
        '"Incomplete" has no image in this revision.'
    )
    window.close()
