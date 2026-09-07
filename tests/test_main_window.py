"""Offscreen integration tests for revision-centric application chrome."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QColor, QKeySequence, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel

import hotcards.generation.mflux_generator as mflux_module
import hotcards.ui.main_window as main_window_module
from hotcards.application.applied_image_change import (
    AppliedImageChange,
    EditImageOperation,
    ImageOperation,
)
from hotcards.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
)
from hotcards.application.commands import (
    ActivateRevisionCommand,
    CreateCardCommand,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    RenameCardCommand,
    ReplaceRevisionBackgroundCommand,
)
from hotcards.application.document_controller import DocumentController
from hotcards.application.document_session import (
    DocumentSession,
    DocumentSessionError,
    DocumentSessionState,
)
from hotcards.application.workers import (
    AdapterKind,
    AdapterWorkers,
    AvailabilityDiagnostic,
)
from hotcards.domain.image_dimensions import AspectRatio, ResolutionTier
from hotcards.domain.models import (
    HYPERCARD_STYLE_ID,
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    DirectGenerateProvenance,
    EditPreserveOptions,
    EditProvenance,
    GeneratedBackground,
    GeneratedSoundAsset,
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
    PresetOutputSize,
    RefineProvenance,
    RefineTransformation,
    ResolvedCardReference,
    SoundDefinition,
    SoundGenerationProvenance,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
    image_edit_lineage,
)
from hotcards.generation.errors import ImageGenerationCancelled
from hotcards.generation.mflux_generator import MfluxGenerator
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
)
from hotcards.ui.card_sidebar import CardSidebar
from hotcards.ui.main_window import MainWindow
from hotcards.ui.new_stack_dialog import NewStackDialog


class FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def value(self, key: str, default_value: Any = None) -> Any:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: Any) -> None:
        self.values[key] = value

    def sync(self) -> None:
        pass


def _hotspot_polygon() -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1, y=0.1),
            Point(x=0.4, y=0.1),
            Point(x=0.2, y=0.4),
        )
    )


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    @property
    def is_finished(self) -> bool:
        return self.cancelled

    def cancel(self) -> None:
        self.cancelled = True
        self.finished.emit()


class FakeWorkers(QObject):
    availability_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.mflux_operations: list[FakeOperation] = []
        self.shutdown_calls = 0

    def check_mflux(
        self,
        _check: object,
        *,
        emit_diagnostic: bool = True,
    ) -> FakeOperation:
        assert not emit_diagnostic
        operation = FakeOperation()
        self.mflux_operations.append(operation)
        return operation

    def shutdown(self, *, wait_milliseconds: int = 0) -> None:
        self.shutdown_calls += 1


class FakeBackgroundWorkflow(QObject):
    busy_changed = Signal(bool)
    invocation_active_changed = Signal(bool)
    progress_changed = Signal(str)
    generation_progress_changed = Signal(int, int)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    image_applied = Signal(object)

    def __init__(self, controller: DocumentController) -> None:
        super().__init__()
        self.controller = controller
        self.busy = False
        self.invocation_active = False
        self.active_operation: str | None = None
        self.generate_calls: list[object] = []
        self.refine_calls: list[tuple[object, object, object]] = []
        self.edit_calls: list[tuple[object, object, object]] = []
        self.clear_calls: list[object] = []
        self.cancel_calls = 0
        self.closed = False

    def generate(self, card_id: object) -> None:
        self.generate_calls.append(card_id)

    def refine(
        self,
        card_id: object,
        *,
        transformation: object,
        output_size: object,
    ) -> None:
        self.refine_calls.append((card_id, transformation, output_size))

    def edit(
        self,
        card_id: object,
        *,
        instruction: object,
        output_size: object,
    ) -> None:
        self.edit_calls.append((card_id, instruction, output_size))

    def clear_background(self, card_id: object) -> None:
        self.clear_calls.append(card_id)

    def activate_revision(self, card_id: object, revision_id: object) -> None:
        changed = self.controller.execute(
            ActivateRevisionCommand(card_id=card_id, revision_id=revision_id)  # type: ignore[arg-type]
        )
        self.document_changed.emit(changed)

    def duplicate_revision(self, card_id: object, revision_id: object) -> None:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DuplicateRevisionCommand(  # type: ignore[arg-type]
                card_id=card_id,
                source_revision_id=revision_id,
            )
        )
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit("Revision duplicated", token)

    def delete_revision(self, card_id: object, revision_id: object) -> None:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DeleteRevisionCommand(  # type: ignore[arg-type]
                card_id=card_id,
                revision_id=revision_id,
            )
        )
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit("Revision deleted", token)

    def is_generating_for(self, _card_id: object) -> bool:
        return self.busy and self.active_operation in {None, "generate"}

    def is_refining_for(self, _card_id: object) -> bool:
        return self.busy and self.active_operation == "refine"

    def is_editing_for(self, _card_id: object) -> bool:
        return self.busy and self.active_operation == "edit"

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.busy = False
        self.active_operation = None

    def close(self) -> None:
        self.closed = True


class FakeSoundWorkflow(QObject):
    generation_started = Signal(object)
    sampling_progress = Signal(int, int)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.generate_calls: list[UUID] = []
        self.cancel_calls = 0
        self.active = False

    @property
    def is_active(self) -> bool:
        return self.active

    def generate(self, sound_id: UUID) -> None:
        self.generate_calls.append(sound_id)
        self.active = True
        self.generation_started.emit(sound_id)

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.active:
            self.active = False
            self.cancelled.emit()
            self.finished.emit()


class FakeSoundPlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []
        self.stop_calls = 0

    def play(self, path: Path) -> None:
        self.played.append(path)

    def stop(self) -> None:
        self.stop_calls += 1


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _stack() -> Stack:
    first = CardRevision(description="First")
    second = CardRevision(description="Second")
    card = Card(
        name="Foyer",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    return Stack(name="Demo", cards=(card,), start_card_id=card.id)


def _generated_background(
    *,
    asset_id: UUID,
    image_path: str,
    description: str = "Test image",
) -> GeneratedBackground:
    generated_at = datetime.now(UTC)
    return GeneratedBackground(
        id=asset_id,
        image_path=image_path,
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(
                description=description,
            ),
            render_prompt=description,
            settings=ImageOperationSettings(
                model_identifier="test",
                mflux_version="test",
                seed=1,
                width=512,
                height=384,
                step_count=4,
                generated_at=generated_at,
                duration_seconds=1,
            ),
        ),
        created_at=generated_at,
    )


def _window(
    stack: Stack | None = None,
) -> tuple[MainWindow, DocumentController, FakeWorkers, FakeBackgroundWorkflow]:
    controller = DocumentController(stack or _stack())
    workers = FakeWorkers()
    background = FakeBackgroundWorkflow(controller)
    sound_workflow = FakeSoundWorkflow()
    sound_player = FakeSoundPlayer()
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={
            AdapterKind.MFLUX: lambda: None,
        },
        background_workflow=background,  # type: ignore[arg-type]
        sound_workflow=sound_workflow,  # type: ignore[arg-type]
        sound_player=sound_player,
        start_diagnostics=False,
    )
    return window, controller, workers, background


def _edited_background(card: Card, instruction: str) -> GeneratedBackground:
    revision = card.active_revision
    background = revision.background
    assert background is not None
    settings = background.provenance.settings
    asset_id = uuid4()
    return GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/{card.id}/image-{asset_id}.png",
        created_at=settings.generated_at,
        provenance=EditProvenance(
            source=DerivedImageSourceSnapshot(
                card_id=card.id,
                revision_id=revision.id,
                background_id=background.id,
                width=settings.width,
                height=settings.height,
                seed=settings.seed,
                edit_lineage=image_edit_lineage(background.provenance),
            ),
            instruction=instruction,
            preserve=EditPreserveOptions(),
            expanded_prompt=f"{instruction}\n\nHidden Style addendum",
            output_size=CurrentSourceSize(width=settings.width, height=settings.height),
            prompt_token_count=20,
            settings=settings.model_copy(update={"seed": settings.seed + 1}),
        ),
    )


def test_card_header_and_toolbar_match_revision_hierarchy(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()

    assert window.canvas_card_name.text() == "Foyer"
    assert [window.revision_combo.itemText(index) for index in range(2)] == [
        "1",
        "2",
    ]
    assert window.revision_combo.currentText() == "1"
    assert window.revision_label.text() == "Version:"
    assert window.revision_combo.width() < 100


def test_author_utility_windows_are_modeless_singletons_and_reopen(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window.show()
    application.processEvents()

    window.styles_button.click()
    application.processEvents()
    first_style_window = window.style_manager_window
    assert first_style_window is not None
    assert first_style_window.isVisible()
    assert first_style_window.windowModality() == Qt.WindowModality.NonModal
    assert window.save_action in first_style_window.actions()
    shortcut_triggers: list[None] = []
    window.save_action.triggered.connect(lambda: shortcut_triggers.append(None))
    window.save_action.setEnabled(True)
    first_style_window.name_edit.setFocus()
    application.processEvents()
    QTest.keySequence(
        first_style_window.name_edit,
        window.save_action.shortcut(),
    )
    application.processEvents()
    assert shortcut_triggers == [None]

    window.styles_button.click()
    application.processEvents()
    assert window.style_manager_window is first_style_window

    first_style_window.close()
    application.processEvents()
    assert window.style_manager_window is None
    assert MainWindow._STYLE_MANAGER_GEOMETRY_KEY in window.settings.values

    window.styles_button.click()
    window.sounds_button.click()
    window.keys_button.click()
    application.processEvents()
    assert window.style_manager_window is not None
    assert window.style_manager_window is not first_style_window
    assert window.sound_manager_window is not None
    assert window.sound_manager_window.windowModality() == Qt.WindowModality.NonModal
    assert window.key_manager_window is not None
    assert window.key_manager_window.windowModality() == Qt.WindowModality.NonModal
    managers = (
        window.style_manager_window,
        window.sound_manager_window,
        window.key_manager_window,
    )
    assert {manager.size() for manager in managers} == {QSize(420, 600)}
    assert all(manager.windowFlags() & Qt.WindowType.WindowStaysOnTopHint for manager in managers)
    assert all(manager.done_button.text() == "Done" for manager in managers)
    assert all(
        manager.done_button.geometry().right()
        <= manager.done_button.parentWidget().contentsRect().right()
        for manager in managers
    )
    assert not hasattr(window.sound_manager_window, "model_label")
    assert window.sound_manager_window.prompt_edit.minimumHeight() == (
        window.sound_manager_window.prompt_edit.maximumHeight()
    )
    assert window.sound_manager_window.usage_list.height() == 70
    assert window.key_manager_window.usage_list.height() == 70
    for manager in managers:
        manager.done_button.click()
    application.processEvents()
    assert window.style_manager_window is None
    assert window.sound_manager_window is None
    assert window.key_manager_window is None
    window.close()
    application.processEvents()


def test_pending_durability_disables_utility_manager_mutations(
    application: QApplication,
) -> None:
    key = KeyDefinition(name="Old")
    window, controller, _workers, _background = _window(
        Stack(name="Demo", keys=(key,), cards=(Card(name="Card"),))
    )
    window._show_style_manager()
    window._show_key_manager()
    style_manager = window.style_manager_window
    key_manager = window.key_manager_window
    assert style_manager is not None
    assert key_manager is not None
    key_manager.name_edit.setFocus()
    application.processEvents()
    key_manager.name_edit.setText("Draft")

    def fail_indeterminate(candidate: Stack) -> None:
        raise StackStoreTransactionError(
            RuntimeError("manifest fsync failed"),
            observed_stack=candidate,
            durability_indeterminate=True,
        )

    with pytest.raises(StackStoreTransactionError):
        controller.execute_persisted(
            CreateCardCommand(name="Observed"),
            fail_indeterminate,
        )

    window._session_state_changed(
        DocumentSessionState(
            bundle_path=Path("/tmp/Demo.hotcards"),
            dirty=True,
            error="save pending",
            mutation_blocked=True,
        )
    )

    assert not window.styles_button.isEnabled()
    assert not window.keys_button.isEnabled()
    assert not style_manager.add_button.isEnabled()
    assert not style_manager.name_edit.isEnabled()
    assert not key_manager.add_button.isEnabled()
    assert not key_manager.name_edit.isEnabled()
    assert key_manager.name_edit.text() == "Draft"
    assert key_manager.error_label.isVisible()
    window._close_utility_windows(commit_pending=False)
    window.close()
    application.processEvents()


def test_sound_manager_edits_catalog_shows_usage_and_starts_generation(
    application: QApplication,
) -> None:
    sound = SoundDefinition(name="Knock", prompt="A wooden knock")
    revision = CardRevision(
        hotspot_set=HotspotSet(
            interactions=(Interaction(sound_id=sound.id, polygons=(_hotspot_polygon(),)),)
        )
    )
    window, controller, _workers, _background = _window(
        Stack(
            name="Sounds",
            sounds=(sound,),
            cards=(Card(name="Door", revisions=(revision,)),),
        )
    )
    window._show_sound_manager()
    manager = window.sound_manager_window
    assert manager is not None

    assert manager.sound_list.item(0).text() == "Knock — 1 use"
    assert manager.usage_list.count() == 1
    assert manager.usage_list.item(0).text() == "Door V1: Play Knock"
    assert not manager.usage_list.styleSheet()
    assert not hasattr(manager, "show_usage_button")
    assert not manager.delete_button.isEnabled()
    requested: list[tuple[object, object, object]] = []
    manager.hotspot_usage_requested.connect(
        lambda card_id, revision_id, interaction_id: requested.append(
            (card_id, revision_id, interaction_id)
        )
    )
    manager.usage_list.setCurrentRow(0)
    manager.usage_list.itemDoubleClicked.emit(manager.usage_list.item(0))
    card = controller.document.cards[0]
    assert requested == [
        (
            card.id,
            card.active_revision.id,
            revision.hotspot_set.interactions[0].id,
        )
    ]
    manager.name_edit.setText("Door knock")
    manager.prompt_edit.setPlainText("A heavy wooden knock")
    manager.duration_spin.setValue(3)
    assert manager.commit_pending_edits(render_change=True)

    updated = controller.document.sound_by_id(sound.id)
    assert (updated.name, updated.prompt, updated.duration_seconds) == (
        "Door knock",
        "A heavy wooden knock",
        3,
    )
    manager.generate_button.click()
    assert window.sound_workflow is not None
    assert window.sound_workflow.generate_calls == [sound.id]  # type: ignore[attr-defined]
    window.close()
    application.processEvents()


def test_run_hotspot_stops_previous_audio_then_plays_after_navigation(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asset_id = uuid4()
    sound = SoundDefinition(
        name="Door",
        generated=GeneratedSoundAsset(
            id=asset_id,
            audio_path=f"assets/sounds/{uuid4()}/sound-{asset_id}.wav",
            provenance=SoundGenerationProvenance(
                prompt="A door opens",
                duration_seconds=2,
                seed=1,
                generation_duration_milliseconds=100,
            ),
            created_at=datetime.now(UTC),
        ),
    )
    destination = Card(name="Destination")
    hotspot = Interaction(
        action=NavigateAction(target=ResolvedCardReference(target_card_id=destination.id)),
        sound_id=sound.id,
        polygons=(_hotspot_polygon(),),
    )
    revision = CardRevision(hotspot_set=HotspotSet(interactions=(hotspot,)))
    source = Card(name="Source", revisions=(revision,))
    window, _controller, _workers, _background = _window(
        Stack(
            name="Run sounds",
            sounds=(sound,),
            cards=(source, destination),
            start_card_id=source.id,
        )
    )
    path = tmp_path / "door.wav"
    path.touch()
    monkeypatch.setattr(window, "_resolve_sound_asset_path", lambda _path: path)
    player = window.sound_player
    assert isinstance(player, FakeSoundPlayer)

    window._toggle_mode()
    stops_before = player.stop_calls
    window._run_interaction_activated(hotspot.id)

    assert window._selected_card_id == destination.id
    assert player.stop_calls == stops_before + 1
    assert player.played == [path]
    window._run_back()
    assert player.stop_calls == stops_before + 2
    window.close()
    application.processEvents()


def test_sound_picker_preview_uses_catalog_audio_and_stops_on_cancel(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sound = SoundDefinition(
        name="Door",
        generated=GeneratedSoundAsset(
            audio_path="assets/sounds/door.wav",
            provenance=SoundGenerationProvenance(
                prompt="A door opens",
                duration_seconds=2,
                seed=1,
                generation_duration_milliseconds=100,
            ),
            created_at=datetime.now(UTC),
        ),
    )
    hotspot = Interaction(sound_id=sound.id, polygons=(_hotspot_polygon(),))
    source = Card(
        name="Source",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(hotspot,))),),
    )
    window, _controller, _workers, _background = _window(
        Stack(name="Sound picker", sounds=(sound,), cards=(source,))
    )
    path = tmp_path / "door.wav"
    path.touch()
    monkeypatch.setattr(window, "_resolve_sound_asset_path", lambda _path: path)
    player = window.sound_player
    assert isinstance(player, FakeSoundPlayer)

    window.inspector.hotspot_sound_button.click()
    application.processEvents()
    tile = window.inspector.sound_picker.tile(sound.id)
    assert tile is not None
    tile.play_button.click()

    assert player.played == [path]
    stops_before = player.stop_calls
    window.inspector.sound_picker.cancel_button.click()
    application.processEvents()
    assert player.stop_calls == stops_before + 1
    window.close()
    application.processEvents()


def test_sound_picker_and_manager_keep_shared_preview_state_coherent(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sound = SoundDefinition(
        name="Door",
        generated=GeneratedSoundAsset(
            audio_path="assets/sounds/door.wav",
            provenance=SoundGenerationProvenance(
                prompt="A door opens",
                duration_seconds=2,
                seed=1,
                generation_duration_milliseconds=100,
            ),
            created_at=datetime.now(UTC),
        ),
    )
    hotspot = Interaction(sound_id=sound.id, polygons=(_hotspot_polygon(),))
    source = Card(
        name="Source",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(hotspot,))),),
    )
    window, _controller, _workers, _background = _window(
        Stack(name="Sound previews", sounds=(sound,), cards=(source,))
    )
    path = tmp_path / "door.wav"
    path.touch()
    monkeypatch.setattr(window, "_resolve_sound_asset_path", lambda _path: path)
    player = window.sound_player
    assert isinstance(player, FakeSoundPlayer)
    window._show_sound_manager()
    manager = window.sound_manager_window
    assert manager is not None
    manager.preview_button.click()
    assert manager.preview_active

    window.inspector.hotspot_sound_button.click()
    application.processEvents()
    tile = window.inspector.sound_picker.tile(sound.id)
    assert tile is not None
    stops_before_picker = player.stop_calls
    tile.play_button.click()
    assert not manager.preview_active
    assert player.stop_calls == stops_before_picker + 1

    manager.preview_button.click()
    assert manager.preview_active
    stops_before_cancel = player.stop_calls
    window.inspector.sound_picker.cancel_button.click()
    application.processEvents()
    assert player.stop_calls == stops_before_cancel
    assert manager.preview_active
    window.close()
    application.processEvents()


def test_sound_picker_preview_reports_unreadable_audio(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sound = SoundDefinition(
        name="Door",
        generated=GeneratedSoundAsset(
            audio_path="assets/sounds/door.wav",
            provenance=SoundGenerationProvenance(
                prompt="A door opens",
                duration_seconds=2,
                seed=1,
                generation_duration_milliseconds=100,
            ),
            created_at=datetime.now(UTC),
        ),
    )
    hotspot = Interaction(sound_id=sound.id, polygons=(_hotspot_polygon(),))
    source = Card(
        name="Source",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(hotspot,))),),
    )
    window, _controller, _workers, _background = _window(
        Stack(name="Sound picker", sounds=(sound,), cards=(source,))
    )

    def fail_to_resolve(_path: str) -> Path:
        raise StackStoreError("Sound asset is unavailable")

    monkeypatch.setattr(window, "_resolve_sound_asset_path", fail_to_resolve)
    window.inspector.hotspot_sound_button.click()
    application.processEvents()
    tile = window.inspector.sound_picker.tile(sound.id)
    assert tile is not None
    tile.play_button.click()

    assert window.notification_bar.current_key == "sound-preview"
    assert window.notification_bar.message_label.text() == "Could not play Sound"
    notification = window.notification_bar.current_notification
    assert notification is not None
    assert notification.detail == "Sound asset is unavailable"
    window.inspector.sound_picker.cancel_button.click()
    window.close()
    application.processEvents()


def test_utility_windows_close_on_project_replacement_without_stale_draft(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    window._show_style_manager()
    manager = window.style_manager_window
    assert manager is not None
    manager.show()
    manager.name_edit.setFocus()
    application.processEvents()
    manager.name_edit.setText("Old project draft")

    existing_style = controller.document.styles[0]
    replacement_style = StyleDefinition(
        id=existing_style.id,
        name="Replacement",
        prompt_text="Replacement treatment.",
    )
    replacement = Stack(
        name="Replacement",
        styles=(replacement_style,),
        new_card_style_id=None,
        cards=(Card(name="New card"),),
    )
    controller.replace_document(replacement)
    window._document_replaced(replacement)
    application.processEvents()

    assert window.style_manager_window is None
    assert controller.document.styles == (replacement_style,)
    window.close()
    application.processEvents()


def test_style_manager_updates_generate_selector_and_cancels_active_work(
    application: QApplication,
) -> None:
    selected_style = StyleDefinition(
        name="Ink",
        prompt_text="Rendered in ink.",
    )
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Scene", style_id=selected_style.id),),
    )
    window, controller, _workers, background = _window(
        Stack(
            name="Demo",
            styles=(selected_style,),
            new_card_style_id=selected_style.id,
            cards=(card,),
        )
    )
    background.busy = True
    background.active_operation = "generate"
    window._show_style_manager()
    manager = window.style_manager_window
    assert manager is not None

    manager.show()
    manager.name_edit.setFocus()
    application.processEvents()
    manager.name_edit.setText("Updated Style")
    assert background.cancel_calls == 1
    assert manager.commit_pending_edits(render_change=True)

    assert controller.document.style_by_id(selected_style.id).name == "Updated Style"
    assert window.inspector.style_combo.currentText() == "Updated Style"
    window.close()
    application.processEvents()


def test_hotspot_rule_editor_scrolls_without_growing_the_window(
    application: QApplication,
) -> None:
    keys = tuple(KeyDefinition(name=f"Key {number}") for number in range(6))
    interaction = Interaction(
        conditions=HotspotConditions(
            requires=tuple(key.id for key in keys[:3]),
        ),
        key_changes=HotspotKeyChanges(
            grant=tuple(key.id for key in keys[3:]),
        ),
        polygons=(_hotspot_polygon(),),
    )
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    window, _controller, _workers, _background = _window(
        Stack(name="Demo", keys=keys, cards=(card,))
    )
    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._hotspots_tab_index)
    window.resize(1180, 700)
    window.show()
    application.processEvents()
    try:
        assert window.minimumSizeHint().height() <= 700
        assert window.height() == 700
        assert window.inspector.hotspot_scroll.verticalScrollBar().maximum() > 0
        assert window.card_sidebar.card_list.geometry().right() == (
            window.card_sidebar.add_button.geometry().right()
        )
    finally:
        window.close()
        application.processEvents()
    assert window.revision_combo.width() >= (
        window.revision_combo.fontMetrics().horizontalAdvance("0000")
    )
    assert window.add_revision_button.text() == "+"
    assert window.delete_revision_button.text() == "−"
    assert window.card_header.indexOf(window.delete_revision_button) < (
        window.card_header.indexOf(window.add_revision_button)
    )
    assert window.overlay_label.text() == "Hotspots"
    assert window.toolbar_leading_spacer.width() == 8
    assert window.toolbar_trailing_spacer.width() == window.toolbar_leading_spacer.width()
    assert window.mode_button.text() == "Run"
    assert window.mode_button.toolTip() == "Switch to Run mode"
    toolbar_actions = window.authoring_toolbar.actions()
    assert toolbar_actions.index(window.back_button_action) < toolbar_actions.index(
        window.restart_button_action
    )
    assert toolbar_actions.index(window.restart_button_action) < toolbar_actions.index(
        window.run_overlay_separator
    )
    assert toolbar_actions.index(window.run_overlay_separator) < toolbar_actions.index(
        window.overlay_label_action
    )
    assert toolbar_actions.index(window.overlay_label_action) < (
        toolbar_actions.index(window.overlay_selector_action)
    )
    assert toolbar_actions.index(window.overlay_selector_action) < (
        toolbar_actions.index(window.styles_button_action)
    )
    assert toolbar_actions.index(window.mode_button_action) < (
        toolbar_actions.index(window.author_action_spacer_action)
    )
    assert toolbar_actions.index(window.author_action_spacer_action) < (
        toolbar_actions.index(window.styles_button_action)
    )
    assert toolbar_actions.index(window.styles_button_action) < (
        toolbar_actions.index(window.keys_button_action)
    )
    assert toolbar_actions.index(window.keys_button_action) < (
        toolbar_actions.index(window.toolbar_trailing_spacer_action)
    )
    assert window.back_button.font().pointSizeF() == (window.mode_button.font().pointSizeF())
    assert window.restart_button.font().pointSizeF() == (window.mode_button.font().pointSizeF())
    assert window.back_button.sizeHint().height() >= (window.mode_button.sizeHint().height())
    assert window.restart_button.sizeHint().height() >= (window.mode_button.sizeHint().height())
    assert not window.run_controls_separator.isVisible()
    assert not window.run_overlay_separator.isVisible()
    assert not window.overlay_label_action.isVisible()
    assert not window.overlay_selector_action.isVisible()
    assert window.styles_button.text() == "Styles"
    assert window.keys_button.text() == "Keys"
    assert window.styles_button_action.isVisible()
    assert window.keys_button_action.isVisible()
    assert not hasattr(window, "document_status_label")
    assert window.generation_progress_container.isHidden()
    margins = window.generation_progress_layout.contentsMargins()
    assert margins.left() == 8
    assert margins.right() == 8
    assert window.generation_progress_layout.indexOf(
        window.generation_progress_bar
    ) < window.generation_progress_layout.indexOf(window.cancel_generation_button)
    cancel_spacing = window.generation_progress_layout.itemAt(
        window.generation_progress_layout.indexOf(window.cancel_generation_button) - 1
    ).spacerItem()
    assert cancel_spacing is not None
    assert cancel_spacing.sizeHint().width() == 8
    assert window.cancel_generation_button.text() == "Cancel"
    central_layout = window.centralWidget().layout()
    assert central_layout is not None
    assert central_layout.indexOf(window.pane_splitter) < (
        central_layout.indexOf(window.notification_bar)
    )
    assert window.inspector.inspector_tabs.tabText(0) == "Generate"
    assert not hasattr(window, "fit_canvas_button")
    assert not hasattr(window, "clear_background_button")


def test_card_browser_uses_thumbnails_and_compact_action_row(
    application: QApplication,
    tmp_path: Path,
) -> None:
    thumbnail_path = tmp_path / "thumbnail.png"
    thumbnail = QPixmap(144, 96)
    thumbnail.fill(QColor("red"))
    assert thumbnail.save(str(thumbnail_path))
    asset_id = uuid4()
    revision = CardRevision(
        background=_generated_background(
            asset_id=asset_id,
            image_path=f"assets/cards/card/image-{asset_id}.png",
        )
    )
    card = Card(name="Preview", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,), start_card_id=card.id))
    sidebar = CardSidebar(
        controller,
        image_path_resolver=lambda _path: thumbnail_path,
    )

    assert not any(label.text() == "Cards" for label in sidebar.findChildren(QLabel))
    item = sidebar.card_list.item(0)
    icon = item.icon().pixmap(QSize(72, 48))
    assert icon.size() == QSize(72, 48)
    assert icon.toImage().pixelColor(10, 10) == QColor("red")
    assert not sidebar.start_button.icon().isNull()
    duplicate_icon = sidebar.duplicate_button.icon().pixmap(QSize(20, 20)).toImage()
    assert duplicate_icon.pixelColor(4, 4).alpha() > 0
    assert duplicate_icon.pixelColor(11, 11).alpha() == 0
    assert duplicate_icon.pixelColor(16, 16).alpha() > 0
    assert sidebar.start_button.toolTip() == ("Make the selected card the start card")
    assert sidebar.add_button.toolTip() == "Add a new card"
    assert sidebar.duplicate_button.toolTip() == "Duplicate the selected card"
    assert sidebar.delete_button.toolTip() == "Delete the selected card"
    assert sidebar.card_actions.indexOf(sidebar.move_up_button) == 0
    assert sidebar.card_actions.indexOf(sidebar.move_down_button) == 1
    assert sidebar.card_actions.indexOf(sidebar.start_button) == 3
    assert sidebar.card_actions.indexOf(sidebar.duplicate_button) == 4
    assert sidebar.card_actions.indexOf(sidebar.delete_button) == 5
    assert sidebar.card_actions.indexOf(sidebar.add_button) == 6
    control_sizes = {
        button.size()
        for button in (
            sidebar.move_up_button,
            sidebar.move_down_button,
            sidebar.start_button,
            sidebar.duplicate_button,
            sidebar.add_button,
            sidebar.delete_button,
        )
    }
    assert len(control_sizes) == 1
    assert sidebar.add_button.font().pointSizeF() > (sidebar.move_up_button.font().pointSizeF())


def test_new_card_is_inserted_after_selected_card(
    application: QApplication,
) -> None:
    first = Card(name="First")
    selected = Card(name="Selected")
    last = Card(name="Last")
    controller = DocumentController(Stack(name="Demo", cards=(first, selected, last)))
    sidebar = CardSidebar(controller)
    sidebar.select_card(selected.id)

    created_id = sidebar.add_card("Created")

    assert [card.name for card in controller.document.cards] == [
        "First",
        "Selected",
        "Created",
        "Last",
    ]
    assert sidebar.selected_card_id == created_id


def test_duplicate_card_sidebar_action_commits_cancels_and_restores_selection(
    application: QApplication,
    tmp_path: Path,
) -> None:
    leading = Card(name="Leading")
    source = Card(name="Source")
    stack = Stack(
        name="Demo",
        cards=(leading, source),
        start_card_id=leading.id,
    )
    controller = DocumentController(stack)
    session = DocumentSession(controller)
    session.create(stack, tmp_path / "Demo.hotcards")
    workers = FakeWorkers()
    background = FakeBackgroundWorkflow(controller)
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={AdapterKind.MFLUX: lambda: None},
        document_session=session,
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    window.select_card(source.id)
    window.canvas_card_name.setText("Renamed")
    window.inspector.description_edit.setPlainText("Committed before copy")
    background.busy = True

    window.card_sidebar.duplicate_button.click()

    changed = controller.document
    assert [card.name for card in changed.cards] == [
        "Leading",
        "Renamed",
        "Renamed Copy",
    ]
    duplicate = changed.cards[2]
    assert duplicate.active_revision.description == "Committed before copy"
    assert background.cancel_calls == 1
    assert window.card_sidebar.selected_card_id == duplicate.id
    assert window.notification_bar.message_label.text() == "Card duplicated"
    assert not window.notification_bar.primary_button.isHidden()
    duplicate_id = duplicate.id

    window.notification_bar.primary_button.click()
    assert [card.name for card in controller.document.cards] == [
        "Leading",
        "Renamed",
    ]
    assert window.card_sidebar.selected_card_id == source.id

    window.redo()
    assert controller.document.cards[2].id == duplicate_id
    assert window.card_sidebar.selected_card_id == duplicate_id
    window.close()


def test_duplicate_card_shortcut_is_author_only(
    application: QApplication,
    tmp_path: Path,
) -> None:
    source = Card(name="Source")
    stack = Stack(name="Demo", cards=(source,), start_card_id=source.id)
    controller = DocumentController(stack)
    session = DocumentSession(controller)
    session.create(stack, tmp_path / "Demo.hotcards")
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={AdapterKind.MFLUX: lambda: None},
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(controller),  # type: ignore[arg-type]
        start_diagnostics=False,
    )

    assert (
        window.duplicate_card_action.shortcut().toString(QKeySequence.SequenceFormat.PortableText)
        == "Ctrl+D"
    )
    assert window.duplicate_card_action.isEnabled()
    window.duplicate_card_action.trigger()
    assert [card.name for card in controller.document.cards] == [
        "Source",
        "Source Copy",
    ]

    window.mode_button.click()
    assert window._is_running
    assert not window.duplicate_card_action.isEnabled()
    assert window.card_sidebar.isHidden()
    window.duplicate_card_action.trigger()
    assert len(controller.document.cards) == 2
    window.close()


def test_pending_duplicate_durability_blocks_ui_until_save_retry(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Card(name="Source")
    stack = Stack(name="Demo", cards=(source,), start_card_id=source.id)
    controller = DocumentController(stack)
    session = DocumentSession(controller)
    session.create(stack, tmp_path / "Demo.hotcards")
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={AdapterKind.MFLUX: lambda: None},
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(controller),  # type: ignore[arg-type]
        start_diagnostics=False,
    )

    def fail_indeterminate(candidate: Stack) -> None:
        raise StackStoreTransactionError(
            RuntimeError("manifest directory fsync failed"),
            observed_stack=candidate,
            durability_indeterminate=True,
        )

    with pytest.raises(DocumentSessionError, match="durability remains indeterminate"):
        session.execute_persisted(
            CreateCardCommand(name="Source Copy"),
            persist=fail_indeterminate,
        )
    window.render_document()

    assert controller.mutation_blocked
    assert session.state.mutation_blocked
    assert session.state.dirty
    assert window.notification_bar.message_label.text() == "Save required before editing"
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()
    assert not window.duplicate_card_action.isEnabled()
    assert not window.card_sidebar.add_button.isEnabled()
    assert not window.card_sidebar.delete_button.isEnabled()
    assert not window.inspector.isEnabled()
    assert window.canvas_card_name.isReadOnly()
    assert not window.revision_combo.isEnabled()
    assert not window.add_revision_button.isEnabled()
    assert not window.inspector.generate_background_button.isEnabled()
    assert window.card_sidebar.card_list.isEnabled()
    window.card_sidebar.card_list.setCurrentRow(1)
    assert window.card_sidebar.selected_card_id == controller.document.cards[1].id

    pending_document = controller.document
    window.undo()
    assert controller.document == pending_document
    assert window.notification_bar.message_label.text() == "Save required before editing"

    replacement_store = StackStore(tmp_path / "Replacement.hotcards")
    replacement_store.create(Stack(name="Replacement"))
    assert session.store is not None
    real_save = session.store.save

    def fail_save(_stack: Stack) -> None:
        raise StackStoreError("retry storage unavailable")

    monkeypatch.setattr(session.store, "save", fail_save)
    with pytest.raises(DocumentSessionError, match="retry storage unavailable"):
        session.open(replacement_store.bundle_path)
    assert controller.document == pending_document
    assert controller.mutation_blocked

    monkeypatch.setattr(window, "_ask_retry_failed_close_save", lambda _message: False)
    close_event = QCloseEvent()
    window.closeEvent(close_event)
    assert not close_event.isAccepted()
    assert controller.mutation_blocked

    monkeypatch.setattr(session.store, "save", real_save)
    assert window.save_document()
    assert not controller.mutation_blocked
    assert not session.state.mutation_blocked
    assert not session.state.dirty
    assert window.undo_action.isEnabled()
    assert window.duplicate_card_action.isEnabled()
    assert window.inspector.isEnabled()
    assert not window.canvas_card_name.isReadOnly()
    window.close()


def test_save_resolves_pending_then_persists_utility_draft(
    application: QApplication,
    tmp_path: Path,
) -> None:
    stack = Stack(name="Demo", cards=(Card(name="Source"),))
    controller = DocumentController(stack)
    session = DocumentSession(controller)
    session.create(stack, tmp_path / "Demo.hotcards")
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={AdapterKind.MFLUX: lambda: None},
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(controller),  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    window._show_style_manager()
    manager = window.style_manager_window
    assert manager is not None
    manager.name_edit.setFocus()
    application.processEvents()
    manager.name_edit.setText("Draft Style")

    def fail_indeterminate(candidate: Stack) -> None:
        raise StackStoreTransactionError(
            RuntimeError("manifest fsync failed"),
            observed_stack=candidate,
            durability_indeterminate=True,
        )

    with pytest.raises(DocumentSessionError, match="durability remains indeterminate"):
        session.execute_persisted(
            CreateCardCommand(name="Observed"),
            persist=fail_indeterminate,
        )

    assert controller.mutation_blocked
    assert manager.name_edit.text() == "Draft Style"
    assert window.save_document()

    assert not controller.mutation_blocked
    assert controller.document.styles[0].name == "Draft Style"
    assert session.store is not None
    assert session.store.load().styles[0].name == "Draft Style"
    window.close()


def test_pending_refine_renders_authoritative_revision_and_promotes_undo(
    application: QApplication,
    tmp_path: Path,
) -> None:
    store = StackStore(tmp_path / "Refine.hotcards")
    card = Card(name="Source")
    source_asset_id = uuid4()
    source_png = tmp_path / "source.png"
    Image.new("RGB", (512, 384), "navy").save(source_png, format="PNG")
    source_image_path = store.store_image_asset(
        source_png,
        card_id=card.id,
        asset_id=source_asset_id,
    )
    source_revision = CardRevision(
        description="Source description",
        background=_generated_background(
            asset_id=source_asset_id,
            image_path=source_image_path,
        ),
    )
    card = card.model_copy(
        update={
            "revisions": (source_revision,),
            "active_revision_id": source_revision.id,
        }
    )
    stack = Stack(name="Demo", cards=(card,), start_card_id=card.id)
    store.create(stack)
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.open(store.bundle_path)
    background = FakeBackgroundWorkflow(controller)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={AdapterKind.MFLUX: lambda: None},
        document_session=session,
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    refined_asset_id = uuid4()
    refined_png = tmp_path / "refined.png"
    Image.new("RGB", (768, 576), "teal").save(refined_png, format="PNG")
    refined_image_path = store.store_image_asset(
        refined_png,
        card_id=card.id,
        asset_id=refined_asset_id,
    )
    direct = source_revision.background
    assert direct is not None
    refined_background = GeneratedBackground(
        id=refined_asset_id,
        image_path=refined_image_path,
        provenance=RefineProvenance(
            source=DerivedImageSourceSnapshot(
                card_id=card.id,
                revision_id=source_revision.id,
                background_id=source_asset_id,
                width=512,
                height=384,
                seed=direct.provenance.settings.seed,
                edit_lineage=(),
            ),
            description=source_revision.description,
            render_prompt=source_revision.description,
            output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
            transformation=RefineTransformation.BALANCED,
            strength=0.5,
            settings=direct.provenance.settings.model_copy(update={"width": 768, "height": 576}),
        ),
        created_at=direct.created_at,
    )

    def fail_indeterminate(candidate: Stack) -> None:
        raise StackStoreTransactionError(
            RuntimeError("manifest directory fsync failed"),
            observed_stack=candidate,
            durability_indeterminate=True,
        )

    with pytest.raises(DocumentSessionError, match="durability remains indeterminate"):
        session.execute_persisted(
            ReplaceRevisionBackgroundCommand(
                card_id=card.id,
                revision_id=source_revision.id,
                background=refined_background,
            ),
            persist=fail_indeterminate,
        )
    background.document_changed.emit(controller.document)

    assert controller.mutation_blocked
    assert window.revision_combo.currentText() == "1"
    assert window.inspector.description_edit.toPlainText() == source_revision.description
    assert window.card_canvas._current_image == store.asset_path(refined_image_path).resolve()
    assert not window.undo_action.isEnabled()

    assert window.save_document()
    assert not controller.mutation_blocked
    assert window.revision_combo.currentText() == "1"
    assert window.undo_action.isEnabled()
    window.undo()
    assert controller.document.cards[0].active_revision == source_revision
    assert window.revision_combo.currentText() == "1"
    window.close()


def test_open_candidate_validation_failure_preserves_active_ui_session(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = Card(name="First")
    second = Card(name="Second")
    stack = Stack(
        name="Active",
        cards=(first, second),
        start_card_id=first.id,
    )
    controller = DocumentController(stack)
    session = DocumentSession(controller)
    session.create(stack, tmp_path / "Active.hotcards")
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={AdapterKind.MFLUX: lambda: None},
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(controller),  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    window.select_card(second.id)
    controller.execute(RenameCardCommand(card_id=first.id, name="Pending edit"))
    window.render_document()
    before_document = controller.document
    before_store = session.store
    before_state = session.state
    before_token = controller.current_undo_token
    before_selection = window.card_sidebar.selected_card_id
    candidate_path = tmp_path / "Candidate.hotcards"

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(candidate_path),
    )

    def reject_candidate(_path: Path) -> Stack:
        raise DocumentSessionError(
            "could not securely open owned image: symbolic links are not allowed"
        )

    monkeypatch.setattr(session, "open", reject_candidate)

    window.open_stack()

    assert controller.document == before_document
    assert controller.current_undo_token == before_token
    assert session.store is before_store
    assert session.state == before_state
    assert window.card_sidebar.selected_card_id == before_selection
    assert window.notification_bar.message_label.text() == "Could Not Open Stack"
    notification = window.notification_bar.current_notification
    assert notification is not None
    assert "symbolic links are not allowed" in notification.detail

    controller.execute(RenameCardCommand(card_id=second.id, name="Still active"))
    assert session.flush()
    assert before_store is not None
    assert before_store.load() == controller.document
    window.close()


def test_close_reports_owned_asset_cleanup_failure_without_blocking(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Card(name="Source")
    stack = Stack(name="Demo", cards=(source,), start_card_id=source.id)
    controller = DocumentController(stack)
    session = DocumentSession(controller)
    session.create(stack, tmp_path / "Demo.hotcards")
    background = FakeBackgroundWorkflow(controller)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={AdapterKind.MFLUX: lambda: None},
        document_session=session,
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    monkeypatch.setattr(session, "close_history", lambda: False)
    event = QCloseEvent()

    window.closeEvent(event)

    assert event.isAccepted()
    assert background.closed
    assert window.notification_bar.message_label.text() == ("Could Not Clean Up Stack")


def test_selecting_scrolled_card_survives_focus_out_render(
    application: QApplication,
) -> None:
    cards = tuple(Card(name=f"Card {number}") for number in range(20))
    window, _controller, _workers, _background = _window(Stack(name="Demo", cards=cards))
    window.resize(900, 500)
    window.show()
    application.processEvents()
    try:
        window.canvas_card_name.setFocus()
        window.canvas_card_name.setText("Renamed Card")
        card_list = window.card_sidebar.card_list
        scroll_bar = card_list.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        application.processEvents()
        target = cards[-2]
        target_item = card_list.item(len(cards) - 2)
        target_position = card_list.visualItemRect(target_item).center()
        assert card_list.viewport().rect().contains(target_position)
        scroll_position = scroll_bar.value()
        assert scroll_position > scroll_bar.minimum()

        QTest.mouseClick(
            card_list.viewport(),
            Qt.MouseButton.LeftButton,
            pos=target_position,
        )
        application.processEvents()

        assert window.controller.document.cards[0].name == "Renamed Card"
        assert window.card_sidebar.selected_card_id == target.id
        assert window._selected_card_id == target.id
        assert scroll_bar.value() == scroll_position
    finally:
        window.close()
        application.processEvents()


def test_card_header_displays_serialized_active_revision(
    application: QApplication,
) -> None:
    stack = _stack()
    card = stack.cards[0]
    stack = stack.model_copy(
        update={"cards": (card.model_copy(update={"active_revision_id": card.revisions[1].id}),)}
    )
    serialized_stack = Stack.model_validate_json(stack.model_dump_json())

    window, _controller, _workers, _background = _window(serialized_stack)

    assert window.revision_combo.currentIndex() == 1
    assert window.revision_combo.currentText() == "2"


def test_card_name_edit_uses_controller_and_restores_invalid_value(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card_id = controller.document.cards[0].id

    window.canvas_card_name.setText("Renamed")
    window._commit_canvas_card_name()
    assert controller.document.cards[0].name == "Renamed"
    assert "Renamed" in window.card_sidebar.card_list.item(0).text()
    assert controller.undo()
    window.render_document()
    assert window.canvas_card_name.text() == "Foyer"

    window.canvas_card_name.setText("   ")
    window._commit_canvas_card_name()
    assert controller.document.cards[0].name == "Foyer"
    assert window.canvas_card_name.text() == "Foyer"
    assert not window.canvas_card_name_error.isHidden()
    assert not hasattr(window.inspector, "validation_error")
    assert card_id == controller.document.cards[0].id


def test_card_name_validation_error_does_not_follow_card_selection(
    application: QApplication,
) -> None:
    first = Card(name="First")
    second = Card(name="Second")
    window, _controller, _workers, _background = _window(Stack(name="Demo", cards=(first, second)))
    window.canvas_card_name.setText("   ")
    window._commit_canvas_card_name()
    assert not window.canvas_card_name_error.isHidden()

    window.select_card(second.id)

    assert window.canvas_card_name.text() == "Second"
    assert window.canvas_card_name_error.isHidden()


def test_focus_triggered_name_failure_aborts_run_transition(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window.show()
    window.canvas_card_name.setFocus()
    window.canvas_card_name.setText("   ")
    application.processEvents()

    window.mode_button.setFocus()
    application.processEvents()
    assert not window.canvas_card_name_error.isHidden()
    window.mode_button.click()

    assert window.mode_button.text() == "Run"
    assert not window._is_running
    window.close()


def test_render_preserves_focused_card_name_draft(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    window.show()
    window.canvas_card_name.setFocus()
    window.canvas_card_name.setText("Uncommitted name")
    application.processEvents()

    changed = controller.execute(
        DuplicateRevisionCommand(
            card_id=card.id,
            source_revision_id=card.active_revision.id,
        )
    )
    window.render_document(changed)

    assert window.canvas_card_name.text() == "Uncommitted name"
    window.close()


def test_revision_selection_duplicate_delete_and_undo(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]

    window.revision_combo.setCurrentIndex(1)
    assert controller.document.cards[0].active_revision.description == "Second"

    window.add_revision_button.click()
    changed = controller.document.cards[0]
    assert len(changed.revisions) == 3
    assert changed.active_revision.description == "Second"
    assert window.revision_combo.currentIndex() == 2

    background.busy = True
    window.delete_revision_button.click()
    assert len(controller.document.cards[0].revisions) == 2
    assert background.cancel_calls == 1
    assert window.notification_bar.message_label.text() == "Revision deleted"
    window.notification_bar.primary_button.click()
    assert len(controller.document.cards[0].revisions) == 3
    assert card.id == controller.document.cards[0].id


def test_revision_selection_synchronizes_generate_output_size(
    application: QApplication,
) -> None:
    first = CardRevision(generate_output_size=PresetOutputSize(tier=ResolutionTier.SMALL))
    second = CardRevision(generate_output_size=PresetOutputSize(tier=ResolutionTier.FULL))
    card = Card(
        name="Card",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))

    assert window.inspector.resolution_combo.currentData() == PresetOutputSize(
        tier=ResolutionTier.SMALL
    )
    window.revision_combo.setCurrentIndex(1)

    assert controller.document.cards[0].active_revision_id == second.id
    assert window.inspector.resolution_combo.currentData() == PresetOutputSize(
        tier=ResolutionTier.FULL
    )


def test_revision_copy_and_notification_mutations_cancel_generation(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()

    background.busy = True
    window.add_revision_button.click()
    assert background.cancel_calls == 1

    original_resolution = controller.document.cards[0].active_revision.generate_output_size
    window.inspector.resolution_combo.setCurrentIndex(
        window.inspector._combo_index_for_data(
            window.inspector.resolution_combo,
            PresetOutputSize(tier=ResolutionTier.FULL),
        )
    )
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Generate output size changed", token)
    background.busy = True
    window._undo_notification()

    assert background.cancel_calls == 2
    assert controller.document.cards[0].active_revision.generate_output_size == original_resolution


def test_final_revision_cannot_be_deleted(
    application: QApplication,
) -> None:
    card = Card(name="Only")
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    assert not window.delete_revision_button.isEnabled()


def test_card_delete_applies_immediately_and_offers_targeted_undo(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()

    window.card_sidebar.delete_button.click()
    assert controller.document.cards == ()
    assert window.notification_bar.message_label.text() == (
        "Card deleted; the start card was cleared"
    )

    window.notification_bar.primary_button.click()
    assert len(controller.document.cards) == 1
    assert controller.document.start_card_id == controller.document.cards[0].id


def test_card_delete_selects_preceding_card_and_restores_it_on_redo(
    application: QApplication,
) -> None:
    first = Card(name="First")
    preceding = Card(name="Preceding")
    selected = Card(name="Selected")
    last = Card(name="Last")
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(first, preceding, selected, last))
    )
    window.select_card(selected.id)

    window.card_sidebar.delete_button.click()

    assert [card.id for card in controller.document.cards] == [
        first.id,
        preceding.id,
        last.id,
    ]
    assert window.card_sidebar.selected_card_id == preceding.id
    assert window.canvas_card_name.text() == "Preceding"

    window.notification_bar.primary_button.click()
    assert window.card_sidebar.selected_card_id == selected.id

    window.redo()
    assert window.card_sidebar.selected_card_id == preceding.id


def test_deleting_first_card_selects_following_card(
    application: QApplication,
) -> None:
    first = Card(name="First")
    following = Card(name="Following")
    last = Card(name="Last")
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(first, following, last))
    )
    window.select_card(first.id)

    window.card_sidebar.delete_button.click()

    assert [card.id for card in controller.document.cards] == [following.id, last.id]
    assert window.card_sidebar.selected_card_id == following.id


def test_card_delete_ignores_destinationless_hotspots(
    application: QApplication,
) -> None:
    destination = Card(name="Destination")
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                hotspot_set=HotspotSet(interactions=(Interaction(polygons=(_hotspot_polygon(),)),))
            ),
        ),
    )
    window, controller, _workers, _background = _window(
        Stack(
            name="Demo",
            cards=(source, destination),
            start_card_id=source.id,
        )
    )

    window._delete_card(destination.id)

    assert [card.id for card in controller.document.cards] == [source.id]


def test_card_delete_keeps_independent_derived_backgrounds(
    application: QApplication,
) -> None:
    source_background = _generated_background(
        asset_id=uuid4(),
        image_path="assets/cards/source.png",
    )
    source_revision = CardRevision(background=source_background)
    source_card = Card(name="Source", revisions=(source_revision,))
    derived_background = GeneratedBackground(
        image_path="assets/cards/derived.png",
        provenance=RefineProvenance(
            source=DerivedImageSourceSnapshot(
                card_id=source_card.id,
                revision_id=source_revision.id,
                background_id=source_background.id,
                width=512,
                height=384,
                seed=source_background.provenance.settings.seed,
                edit_lineage=(),
            ),
            description="A refined source image",
            render_prompt="A refined source image",
            output_size=CurrentSourceSize(width=512, height=384),
            transformation=RefineTransformation.BALANCED,
            strength=0.50,
            settings=source_background.provenance.settings,
        ),
        created_at=datetime.now(UTC),
    )
    derived_card = Card(
        name="Derived",
        revisions=(CardRevision(background=derived_background),),
    )
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(source_card, derived_card))
    )

    window._delete_card(source_card.id)

    assert controller.document.cards == (derived_card,)
    assert window.notification_bar.current_key != "card-error"
    controller.undo()
    assert controller.document.cards == (source_card, derived_card)


def test_context_change_cancels_background_generation_without_prompt(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    background.busy = True

    window._cancel_background_generation()
    assert background.cancel_calls == 1
    assert not background.busy


def test_resolution_change_cancels_in_flight_generation(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    background.busy = True

    window.inspector.resolution_combo.setCurrentIndex(
        window.inspector._combo_index_for_data(
            window.inspector.resolution_combo,
            PresetOutputSize(tier=ResolutionTier.FULL),
        )
    )

    assert background.cancel_calls == 1
    assert not background.busy
    assert controller.document.cards[0].active_revision.generate_output_size == PresetOutputSize(
        tier=ResolutionTier.FULL
    )
    assert isinstance(window.settings, FakeSettings)
    assert all("resolution" not in key for key in window.settings.values)


@pytest.mark.parametrize("evolve", (False, True))
def test_description_style_and_reference_changes_cancel_in_flight_generation(
    application: QApplication,
    evolve: bool,
) -> None:
    style = StyleDefinition(name="Ink", prompt_text="Rendered in ink.")
    source = Card(name="Source")
    reference = Card(name="Reference")
    window, controller, _workers, background = _window(
        Stack(
            name="Demo",
            styles=(style,),
            new_card_style_id=None,
            cards=(source, reference),
        )
    )

    editor = (
        window.inspector.evolve_description_edit if evolve else window.inspector.description_edit
    )
    combo = window.inspector.evolve_style_combo if evolve else window.inspector.style_combo
    background.busy = True
    editor.setPlainText("Changed")
    background.busy = True
    combo.setCurrentIndex(window.inspector._combo_index_for_data(combo, style.id))
    background.busy = True
    window.inspector.reference_button.click()
    reference_item = next(
        window.inspector.card_picker.card_list.item(index)
        for index in range(window.inspector.card_picker.card_list.count())
        if window.inspector.card_picker.card_list.item(index).data(Qt.ItemDataRole.UserRole)
        == reference.id
    )
    window.inspector.card_picker.card_list.itemClicked.emit(reference_item)

    revision = controller.document.cards[0].active_revision
    assert background.cancel_calls == 3
    assert revision.style_id == style.id
    assert revision.references == (ResolvedCardReference(target_card_id=reference.id),)


def test_card_and_revision_changes_cancel_in_flight_generation(
    application: QApplication,
) -> None:
    first = Card(
        name="First",
        revisions=(CardRevision(), CardRevision()),
    )
    second = Card(name="Second")
    window, _controller, _workers, background = _window(Stack(name="Demo", cards=(first, second)))
    background.busy = True

    window.select_card(second.id)

    assert background.cancel_calls == 1
    background.busy = True
    window.select_card(first.id)
    revision_id = first.revisions[1].id
    background.busy = True
    window._activate_revision(revision_id)

    assert background.cancel_calls == 3


def test_successful_save_as_cancels_generation_and_expires_undo(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = DocumentController(_stack())
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Original.hotcards")
    workers = FakeWorkers()
    background = FakeBackgroundWorkflow(controller)
    background.busy = True
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (
            str(tmp_path / "Copy.hotcards"),
            "HotCards Stack (*.hotcards)",
        ),
    )
    window.save_as()

    assert background.cancel_calls == 1
    assert window.notification_bar.current_key != "undo"
    assert not controller.can_undo


def test_failed_authoring_commit_blocks_save_and_close(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _controller, _workers, background = _window()
    flush_calls: list[bool] = []
    window.document_session = SimpleNamespace(
        store=object(),
        flush=lambda: flush_calls.append(True) or True,
    )
    monkeypatch.setattr(window, "_commit_authoring_metadata", lambda: False)

    assert not window.save_document()
    assert flush_calls == []

    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert not background.closed


def test_failed_style_focus_commit_blocks_save(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    flush_calls: list[bool] = []
    window.document_session = SimpleNamespace(
        store=object(),
        flush=lambda: flush_calls.append(True) or True,
    )
    style = controller.document.styles[0]
    window._show_style_manager()
    manager = window.style_manager_window
    assert manager is not None
    manager.name_edit.setFocus()
    manager.name_edit.setText("")

    manager._editing_finished(
        window,
        Qt.FocusReason.ActiveWindowFocusReason,
    )

    assert manager.name_edit.text() == ""
    assert not window.save_document()
    assert flush_calls == []
    assert controller.document.style_by_id(style.id) == style
    manager.name_edit.setText(style.name)
    assert manager.commit_pending_edits(render_change=False)
    window.document_session = None
    window.close()
    application.processEvents()


def test_author_and_run_modes_apply_consistent_read_only_chrome(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window.show()
    application.processEvents()

    window.mode_button.click()
    application.processEvents()
    assert window.mode_button.text() == "Author"
    assert window.mode_button.toolTip() == "Switch to Author mode"
    assert window.canvas_card_name.isReadOnly()
    assert window.canvas_card_name.isHidden()
    assert window.canvas_card_name_error.isHidden()
    assert window.revision_label.isHidden()
    assert window.revision_combo.isHidden()
    assert not window.revision_combo.isEnabled()
    assert window.add_revision_button.isHidden()
    assert window.delete_revision_button.isHidden()
    assert window.card_sidebar.isHidden()
    assert window.inspector.isHidden()
    assert window.run_controls_separator.isVisible()
    assert window.run_overlay_separator.isVisible()
    assert window.back_action.isVisible()
    assert window.restart_action.isVisible()
    assert window.back_button_action.isVisible()
    assert window.restart_button_action.isVisible()
    assert window.back_button.isVisible()
    assert window.restart_button.isVisible()
    assert window.overlay_label_action.isVisible()
    assert window.overlay_selector_action.isVisible()
    assert not window.overlay_selector.isHidden()
    assert not hasattr(window, "llm_model_label")
    assert not hasattr(window, "llm_model_combo")
    assert not hasattr(window, "image_model_label")
    assert not hasattr(window, "image_model_combo")
    assert window.styles_button.isHidden()
    assert window.keys_button.isHidden()
    assert window.style_manager_window is None
    assert window.key_manager_window is None

    window.mode_button.click()
    application.processEvents()
    assert window.mode_button.text() == "Run"
    assert window.mode_button.toolTip() == "Switch to Run mode"
    assert not window.canvas_card_name.isReadOnly()
    assert not window.canvas_card_name.isHidden()
    assert not window.revision_label.isHidden()
    assert not window.revision_combo.isHidden()
    assert window.revision_combo.isEnabled()
    assert not window.add_revision_button.isHidden()
    assert not window.styles_button.isHidden()
    assert not window.keys_button.isHidden()
    assert not window.run_controls_separator.isVisible()
    assert not window.run_overlay_separator.isVisible()
    assert not window.back_action.isVisible()
    assert not window.restart_action.isVisible()
    assert not window.back_button_action.isVisible()
    assert not window.restart_button_action.isVisible()
    assert window.back_button.isHidden()
    assert window.restart_button.isHidden()
    assert not window.overlay_label_action.isVisible()
    assert not window.overlay_selector_action.isVisible()
    window.hide()
    application.processEvents()


def test_status_bar_is_passive_and_ai_recovery_uses_notification_bar(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.MFLUX,
            available=False,
            message="MFLUX is unavailable",
        )
    )

    assert not hasattr(window, "check_services_button")
    assert not hasattr(window, "review_settings_button")
    assert not hasattr(window, "service_status_label")
    assert not hasattr(window, "image_model_combo")
    assert "MFLUX is unavailable" in window._service_status_detail
    assert window.notification_bar.current_key == "ai-services"
    assert window.notification_bar.primary_button.text() == "Settings"
    assert window.notification_bar.secondary_button.text() == "Check Again"

    window.notification_bar.dismiss_current()
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.MFLUX,
            available=True,
            message="MFLUX is available",
        )
    )
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.MFLUX,
            available=False,
            message="MFLUX is unavailable again",
        )
    )
    assert window.notification_bar.current_key == "ai-services"

    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    assert window.notification_bar.current_key == "undo"


def test_settings_models_menu_persists_image_model_and_cancels_active_work(
    application: QApplication,
) -> None:
    settings = FakeSettings()
    controller = DocumentController(_stack())
    workers = FakeWorkers()
    background = FakeBackgroundWorkflow(controller)
    sound_workflow = FakeSoundWorkflow()

    class AcceptedModelDialog:
        def __init__(self, settings_store: FakeSettings, _parent: object) -> None:
            self.settings_store = settings_store

        def exec(self) -> QDialog.DialogCode:
            self.settings_store.setValue(
                "generation/mflux_model",
                "flux2-klein-9b-kv",
            )
            return QDialog.DialogCode.Accepted

    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        settings,
        availability_checks={AdapterKind.MFLUX: lambda: None},
        settings_dialog_factory=AcceptedModelDialog,  # type: ignore[arg-type]
        background_workflow=background,  # type: ignore[arg-type]
        sound_workflow=sound_workflow,  # type: ignore[arg-type]
        start_diagnostics=False,
    )

    assert window.settings_menu.title() == "Settings"
    assert [action.text() for action in window.settings_menu.actions()] == ["Models…"]
    assert not hasattr(window, "image_model_label")
    assert not hasattr(window, "image_model_combo")

    window.models_action.trigger()

    assert settings.values["generation/mflux_model"] == "flux2-klein-9b-kv"
    assert background.cancel_calls == 1
    assert sound_workflow.cancel_calls == 0
    assert len(workers.mflux_operations) == 1

    window.mode_button.click()
    assert not window.models_action.isEnabled()


def test_timed_out_mflux_model_change_and_window_close_never_block_qt_thread(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = Event()
    release = Event()
    unwound = Event()
    cache_released = Event()

    class CallbackRegistry:
        def __init__(self) -> None:
            self.registered: list[object] = []

        def register(self, callback: object) -> None:
            self.registered.append(callback)

    class BlockingModel:
        def __init__(self) -> None:
            self.callbacks = CallbackRegistry()

        def generate_image(self, **kwargs: object) -> object:
            config = SimpleNamespace(num_inference_steps=kwargs["num_inference_steps"])
            for callback in self.callbacks.registered:
                callback.call_before_loop(config=config)
            entered.set()
            try:
                assert release.wait(2)
                for callback in self.callbacks.registered:
                    callback.call_in_loop()
            except ImageGenerationCancelled:
                raise
            finally:
                unwound.set()
            raise AssertionError("cancelled MFLUX inference returned")

    def release_cache() -> None:
        assert unwound.is_set()
        cache_released.set()

    monkeypatch.setattr(mflux_module, "_CACHED_MODEL", None)
    mflux_module._DEFERRED_RELEASES.clear()
    monkeypatch.setattr(mflux_module, "_release_model_cache", release_cache)
    revision = CardRevision(description="A blocked image")
    card = Card(name="Blocked", revisions=(revision,))
    controller = DocumentController(Stack(name="Blocked", cards=(card,)))
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Blocked.hotcards")
    workers = AdapterWorkers(mflux_timeout_seconds=0.05)
    workflow = BackgroundWorkflow(
        controller,
        session,
        workers,
        lambda: BackgroundGenerationSettings(
            mflux_model="flux2-klein-4b",
            step_count=4,
            quantization=None,
            random_seed=False,
            fixed_seed=42,
        ),
        mflux_generator=MfluxGenerator(model_factory=lambda *_: BlockingModel()),
    )
    temporary_directory = workflow._temporary_directory
    settings = FakeSettings()

    class AcceptedModelDialog:
        def __init__(self, settings_store: FakeSettings, _parent: object) -> None:
            self.settings_store = settings_store

        def exec(self) -> QDialog.DialogCode:
            self.settings_store.setValue(
                "generation/mflux_model",
                "flux2-klein-9b-kv",
            )
            return QDialog.DialogCode.Accepted

    window = MainWindow(
        controller,
        workers,
        settings,
        settings_dialog_factory=AcceptedModelDialog,  # type: ignore[arg-type]
        document_session=session,
        background_workflow=workflow,
        start_diagnostics=False,
        owns_workers=True,
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    workflow.generate(card.id)
    assert entered.wait(0.5)
    deadline = monotonic() + 1
    while workflow.busy and monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert not workflow.busy
    assert len(failures) == 1

    changed_started = monotonic()
    window.open_model_settings()
    assert monotonic() - changed_started < 0.25

    close_event = QCloseEvent()
    close_started = monotonic()
    window.closeEvent(close_event)
    assert monotonic() - close_started < 0.5
    assert close_event.isAccepted()
    assert temporary_directory.is_dir()
    assert not list(temporary_directory.glob("generated-*.png"))

    release.set()
    assert cache_released.wait(0.5)
    deadline = monotonic() + 1
    while temporary_directory.exists() and monotonic() < deadline:
        QTest.qWait(10)

    assert not temporary_directory.exists()
    assert mflux_module._CACHED_MODEL is None
    assert mflux_module._DEFERRED_RELEASES == []


def test_normal_window_close_releases_idle_mflux_and_temporary_directory(
    application: QApplication,
    tmp_path: Path,
) -> None:
    assert application is not None
    revision = CardRevision(description="An idle image")
    card = Card(name="Idle", revisions=(revision,))
    controller = DocumentController(Stack(name="Idle", cards=(card,)))
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Idle.hotcards")
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    workflow = BackgroundWorkflow(
        controller,
        session,
        workers,
        lambda: BackgroundGenerationSettings(
            mflux_model="flux2-klein-4b",
            step_count=4,
            quantization=None,
            random_seed=False,
            fixed_seed=42,
        ),
    )
    temporary_directory = workflow._temporary_directory
    window = MainWindow(
        controller,
        workers,
        FakeSettings(),
        document_session=session,
        background_workflow=workflow,
        start_diagnostics=False,
        owns_workers=True,
    )

    close_event = QCloseEvent()
    close_started = monotonic()
    window.closeEvent(close_event)

    assert monotonic() - close_started < 0.5
    assert close_event.isAccepted()
    assert not temporary_directory.exists()


def test_generation_failure_keeps_description_reusable(
    application: QApplication,
) -> None:
    revision = CardRevision(description="A courtyard")
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, background = _window(Stack(name="Demo", cards=(card,)))
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()
    assert window.inspector.generate_background_button.isEnabled()

    background.busy = True
    window._update_generation_actions()
    background.busy = False
    background.failed.emit(RuntimeError("transient MFLUX failure"))

    assert window.inspector.description_edit.toPlainText() == "A courtyard"
    assert window.inspector.generate_background_button.isEnabled()


def test_generate_replacement_starts_without_a_second_confirmation(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    window._availability[AdapterKind.MFLUX] = True
    card_id = controller.document.cards[0].id

    window._generate_background()
    assert background.generate_calls == [card_id]

    revision = controller.document.cards[0].active_revision
    asset_id = uuid4()
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card_id,
            revision_id=revision.id,
            background=_generated_background(
                asset_id=asset_id,
                image_path=f"assets/cards/{card_id}/image-{asset_id}.png",
            ),
        )
    )
    window.render_document()
    assert window.inspector.generate_background_button.text() == "Generate Image"

    window._generate_background()
    assert background.generate_calls == [card_id, card_id]


def test_generate_is_disabled_without_description(
    application: QApplication,
) -> None:
    revision = CardRevision()
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, background = _window(Stack(name="Demo", cards=(card,)))
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()

    assert not window.inspector.generate_background_button.isEnabled()


def test_generate_tab_wires_evolve_current_image_options_and_cancels_live_changes(
    application: QApplication,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "Refine.hotcards"
    store = StackStore(bundle)
    card = Card(name="Card")
    asset_id = uuid4()
    source = tmp_path / "source.png"
    Image.new("RGB", (512, 384), "navy").save(source)
    image_path = store.store_image_asset(
        source,
        card_id=card.id,
        asset_id=asset_id,
    )
    revision = CardRevision(
        description="A courtyard",
        background=_generated_background(
            asset_id=asset_id,
            image_path=image_path,
            description="A courtyard",
        ),
    )
    card = card.model_copy(
        update={
            "revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    stack = Stack(name="Demo", cards=(card,), start_card_id=card.id)
    store.save(stack)
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.open(bundle)
    background = FakeBackgroundWorkflow(controller)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()

    assert window.inspector.inspector_tabs.tabText(1) == "Evolve"
    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._evolve_tab_index)
    assert window.inspector.refine_background_button.isEnabled()
    assert window.inspector.refine_resolution_combo.currentData() == CurrentSourceSize(
        width=512, height=384
    )
    assert window.inspector.refine_background_button.toolTip() == (
        "Evolve the current image using Description."
    )
    window.inspector.evolve_description_edit.setPlainText("A courtyard with an open gate.")
    assert window.inspector.description_edit.toPlainText() == "A courtyard with an open gate."
    window.inspector.refine_background_button.click()
    assert controller.document.cards[0].active_revision.description == (
        "A courtyard with an open gate."
    )
    description_token = controller.current_undo_token
    window.render_document()
    assert controller.current_undo_token == description_token
    assert background.refine_calls == [
        (
            card.id,
            RefineTransformation.BALANCED,
            CurrentSourceSize(width=512, height=384),
        )
    ]

    background.busy = True
    background.active_operation = "refine"
    window._update_generation_actions()
    assert window.inspector.refine_background_button.text() == "Evolving…"
    assert not window.inspector.generate_background_button.isEnabled()
    window.inspector.refine_transformation_combo.setCurrentIndex(
        window.inspector._combo_index_for_data(
            window.inspector.refine_transformation_combo,
            RefineTransformation.PRESERVE,
        )
    )
    assert background.cancel_calls == 1

    background.busy = True
    background.active_operation = "refine"
    window.mode_button.click()
    assert background.cancel_calls == 2
    assert not window.inspector.isVisible()


@pytest.mark.parametrize("description", ("A courtyard", ""))
def test_edit_tab_wires_current_image_defaults_errors_and_cancellation(
    application: QApplication,
    tmp_path: Path,
    description: str,
) -> None:
    bundle = tmp_path / "Edit.hotcards"
    store = StackStore(bundle)
    card = Card(name="Card")
    asset_id = uuid4()
    source = tmp_path / "source.png"
    Image.new("RGB", (1024, 768), "navy").save(source)
    image_path = store.store_image_asset(
        source,
        card_id=card.id,
        asset_id=asset_id,
    )
    revision = CardRevision(
        description=description,
        background=_generated_background(
            asset_id=asset_id,
            image_path=image_path,
            description="A courtyard",
        ),
    )
    card = card.model_copy(
        update={
            "revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    stack = Stack(name="Demo", cards=(card,), start_card_id=card.id)
    store.save(stack)
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.open(bundle)
    background = FakeBackgroundWorkflow(controller)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()

    assert window.inspector.inspector_tabs.tabText(2) == "Edit"
    assert window.inspector.generate_background_button.isEnabled() == bool(description)
    assert window.inspector.refine_background_button.isEnabled() == bool(description)
    assert not window.inspector.edit_background_button.isEnabled()
    assert window.inspector.edit_resolution_combo.count() == 1
    assert window.inspector.edit_resolution_combo.itemText(0) == "Full"
    window.inspector.edit_instruction_edit.setPlainText("Open the gate.")
    assert window.inspector.edit_background_button.isEnabled()
    window.inspector.edit_background_button.click()

    assert len(background.edit_calls) == 1
    card_id, instruction, output_size = background.edit_calls[0]
    assert card_id == card.id
    assert instruction == "Open the gate."
    assert output_size == CurrentSourceSize(width=1024, height=768)

    background.busy = True
    background.active_operation = "edit"
    window._update_generation_actions()
    assert window.inspector.edit_background_button.text() == "Editing…"
    assert not window.inspector.generate_background_button.isEnabled()
    window.inspector.edit_instruction_edit.setPlainText("Close the gate.")
    assert background.cancel_calls == 1

    window.inspector.edit_instruction_edit.setPlainText("Keep this draft.")
    assert window.inspector.edit_instruction_edit.toPlainText() == "Keep this draft."

    window._background_progress_changed("Image editing failed")
    window._background_failed(RuntimeError("513 model tokens; the limit is 512"))
    assert "513 model tokens" in window.inspector.edit_instruction_error.text()
    assert "513 model tokens" in window.notification_bar.message_label.text()


@pytest.mark.parametrize("cleanup_error", (False, True))
def test_notification_undo_restores_completed_edit_instruction(
    application: QApplication,
    cleanup_error: bool,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    original_name = card.name
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Edited"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None

    background.image_applied.emit(
        AppliedImageChange(
            operation=EditImageOperation(instruction="Open the garden gate."),
            token=token,
            card_id=card.id,
            revision_id=card.active_revision.id,
            previous_revision=card.active_revision,
        )
    )
    if cleanup_error:
        background.failed.emit(RuntimeError("committed image cleanup failed"))
        assert window._applied_image_change.token == token
        assert window._edit_undo_changes[token].operation.instruction == "Open the garden gate."
        window.undo()
    else:
        window.notification_bar.secondary_button.click()

    assert controller.document.cards[0].name == original_name
    assert window.inspector.edit_instruction_edit.toPlainText() == "Open the garden gate."


def test_edit_instruction_waits_for_its_exact_undo_token(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Edited"))
    window.render_document(changed)
    edit_token = controller.current_undo_token
    assert edit_token is not None
    background.image_applied.emit(
        AppliedImageChange(
            token=edit_token,
            card_id=card.id,
            revision_id=changed.cards[0].active_revision.id,
            previous_revision=card.active_revision,
            operation=EditImageOperation(instruction="Open the garden gate."),
        )
    )
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Unrelated change"))
    window.render_document(changed)

    window.undo()
    assert window.inspector.edit_instruction_edit.toPlainText() == ""

    window.undo()
    assert window.inspector.edit_instruction_edit.toPlainText() == "Open the garden gate."


def test_edit_undo_does_not_overwrite_a_new_instruction(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Edited"))
    window.render_document(changed)
    edit_token = controller.current_undo_token
    assert edit_token is not None
    background.image_applied.emit(
        AppliedImageChange(
            token=edit_token,
            card_id=card.id,
            revision_id=changed.cards[0].active_revision.id,
            previous_revision=card.active_revision,
            operation=EditImageOperation(instruction="Open the garden gate."),
        )
    )
    window.inspector.edit_instruction_edit.setPlainText("Try a blue gate instead.")

    window.undo()

    assert window.inspector.edit_instruction_edit.toPlainText() == "Try a blue gate instead."


@pytest.mark.parametrize("newer_draft", ("unchanged", "new-text", "recalled", "changed-back"))
def test_completed_edit_clears_only_its_matching_instruction(
    application: QApplication, newer_draft: str
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    instruction = "Open the garden gate."
    window.inspector.set_edit_instruction(f"  {instruction}  ")
    window._edit_background(instruction, CurrentSourceSize(width=512, height=384))
    if newer_draft == "new-text":
        window.inspector.edit_instruction_edit.setPlainText("Keep this newer draft.")
    elif newer_draft == "recalled":
        window.inspector.set_edit_instruction(f"  {instruction}  ")
    elif newer_draft == "changed-back":
        window.inspector.edit_instruction_edit.setPlainText("Something else")
        window.inspector.edit_instruction_edit.setPlainText(f"  {instruction}  ")
    draft = window.inspector.edit_instruction_edit.toPlainText()
    controller.execute(RenameCardCommand(card_id=card.id, name="Edited"))
    token = controller.current_undo_token
    background.image_applied.emit(
        AppliedImageChange(
            token=token,
            card_id=card.id,
            revision_id=card.active_revision.id,
            previous_revision=card.active_revision,
            operation=EditImageOperation(instruction=instruction),
        )
    )
    assert window.inspector.edit_instruction_edit.toPlainText() == (
        "" if newer_draft == "unchanged" else draft
    )


def test_notification_undo_expires_after_another_command(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    controller.execute(RenameCardCommand(card_id=card.id, name="First"))
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    assert not window.notification_bar.isHidden()
    assert window.notification_bar.message_label.text() == "Renamed"

    controller.execute(RenameCardCommand(card_id=card.id, name="Second"))
    window.render_document()
    assert window.notification_bar.isHidden()
    window._undo_notification()
    assert controller.document.cards[0].name == "Second"


@pytest.mark.parametrize("source_tab", (0, 1))
def test_generate_evolve_tab_switch_commits_shared_description_once(
    application: QApplication,
    source_tab: int,
) -> None:
    card = Card(name="Card", revisions=(CardRevision(description="Saved description"),))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    window.resize(1100, 800)
    window.show()
    tabs = window.inspector.inspector_tabs
    tabs.setCurrentIndex(source_tab)
    editor = (
        window.inspector.evolve_description_edit
        if source_tab == 1
        else window.inspector.description_edit
    )
    editor.setFocus()
    editor.setPlainText("A description authored in either tab")
    application.processEvents()
    assert editor.hasFocus()
    assert controller.current_undo_token is None
    target_tab = 1 - source_tab
    bar = tabs.tabBar()
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(target_tab).center())
    application.processEvents()

    assert tabs.currentIndex() == target_tab
    assert controller.document.cards[0].active_revision.description == (
        "A description authored in either tab"
    )
    assert window.inspector.description_edit.toPlainText() == (
        window.inspector.evolve_description_edit.toPlainText()
    )
    token = controller.current_undo_token
    assert token is not None
    assert controller.retained_history_tokens == frozenset((token,))
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(source_tab).center())
    application.processEvents()
    assert controller.current_undo_token == token
    window.close()


@pytest.mark.parametrize("target", ("mouse", "evolve"))
def test_mouse_focus_commit_does_not_render_before_button_release(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    revision = CardRevision(description="A courtyard")
    card = Card(name="Card", revisions=(revision,))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    window.show()
    draft = "A changed courtyard"
    editor = (
        window.inspector.evolve_description_edit
        if target == "evolve"
        else window.inspector.description_edit
    )
    if target == "evolve":
        window.inspector.inspector_tabs.setCurrentIndex(window.inspector._evolve_tab_index)
    editor.setPlainText(draft)
    editor.setFocus()
    application.processEvents()
    renders: list[object] = []
    monkeypatch.setattr(
        window.inspector,
        "render",
        lambda *_args: renders.append(object()),
    )

    editor.editing_finished.emit(
        window.inspector.refine_background_button if target == "evolve" else None,
        Qt.FocusReason.OtherFocusReason if target == "evolve" else Qt.FocusReason.MouseFocusReason,
    )

    changed_revision = controller.document.cards[0].active_revision
    assert changed_revision.description == draft
    assert renders == []
    window.close()


@pytest.mark.parametrize(
    "operation",
    (ImageOperation("generate"), ImageOperation("refine"), EditImageOperation("Open it.")),
)
@pytest.mark.parametrize("hotspots", ("none", "empty", "populated"))
def test_applied_image_can_move_to_a_new_complete_version(
    application: QApplication,
    operation: ImageOperation | EditImageOperation,
    hotspots: str,
) -> None:
    hotspot_set = (
        None
        if hotspots == "none"
        else HotspotSet()
        if hotspots == "empty"
        else HotspotSet(
            interactions=(
                Interaction(
                    label="Door",
                    action=NavigateAction(target=UnresolvedCardReference()),
                    polygons=(_hotspot_polygon(),),
                ),
            )
        )
    )
    style = StyleDefinition(name="Ink", prompt_text="Ink on paper")
    reference = Card(name="Reference")
    original = CardRevision(
        description="A courtyard",
        hotspot_set=hotspot_set,
        style_id=style.id,
        references=(
            ResolvedCardReference(target_card_id=reference.id),
            UnresolvedCardReference(target_name="Former Reference"),
        ),
        generate_output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
    )
    card = Card(name="Card", revisions=(original,))
    window, controller, _workers, background_workflow = _window(
        Stack(name="Demo", cards=(card, reference), styles=(style,), new_card_style_id=style.id)
    )
    background = _generated_background(
        asset_id=uuid4(),
        image_path=f"assets/cards/{card.id}/generated.png",
        description=original.description,
    )
    changed = controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=original.id,
            background=background,
        )
    )
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    change = AppliedImageChange(
        operation=operation,
        token=token,
        card_id=card.id,
        revision_id=original.id,
        previous_revision=original,
    )
    background_workflow.image_applied.emit(change)

    assert (
        window.notification_bar.message_label.text() == f"{change.message} on the current version"
    )
    assert window.notification_bar.primary_button.text() == "Create New Version"
    assert window.notification_bar.secondary_button.text() == "Undo"
    assert window.notification_bar.dismiss_button.text() == "Keep"
    background_workflow.busy = True
    window.notification_bar.primary_button.click()

    assert background_workflow.cancel_calls == 1
    changed_card = controller.document.cards[0]
    assert len(changed_card.revisions) == 2
    assert changed_card.revisions[0].id == original.id
    assert changed_card.revisions[0].background is None
    assert changed_card.revisions[0].hotspot_set == original.hotspot_set
    assert changed_card.active_revision.id != original.id
    assert changed_card.active_revision.description == original.description
    assert changed_card.active_revision.hotspot_set == original.hotspot_set
    assert changed_card.active_revision.background == background
    assert changed_card.revisions[0] == original
    assert changed_card.active_revision.model_copy(update={"id": original.id}) == (
        changed.cards[0].active_revision
    )
    assert not background_workflow.generate_calls
    assert not background_workflow.refine_calls
    assert not background_workflow.edit_calls
    assert window.notification_bar.message_label.text() == "New version created"
    assert window.notification_bar.primary_button.text() == "Undo"
    assert window.notification_bar.secondary_button.isHidden()

    window.notification_bar.primary_button.click()
    restored_card = controller.document.cards[0]
    assert len(restored_card.revisions) == 1
    assert restored_card.active_revision.id == original.id
    assert restored_card.active_revision.background == background
    assert window.inspector.edit_instruction_edit.toPlainText() == ""
    assert controller.current_undo_token == token

    window.undo()
    assert controller.document.cards[0].active_revision == original
    assert window.inspector.edit_instruction_edit.toPlainText() == (
        operation.instruction if isinstance(operation, EditImageOperation) else ""
    )


@pytest.mark.parametrize("undo_action", ("notification", "global"))
def test_edit_instruction_survives_repeated_undo_redo(
    application: QApplication, undo_action: str
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    instruction = "Open the gate.\nKeep the tree's shadow exactly as it is."
    window.inspector.set_edit_instruction(instruction)
    window._edit_background(instruction, CurrentSourceSize(width=512, height=384))
    controller.execute(RenameCardCommand(card_id=card.id, name="Edited"))
    token = controller.current_undo_token
    change = AppliedImageChange(
        token=token,
        card_id=card.id,
        revision_id=card.active_revision.id,
        previous_revision=card.active_revision,
        operation=EditImageOperation(instruction),
    )
    background.image_applied.emit(change)
    assert window.inspector.edit_instruction_edit.toPlainText() == ""
    if undo_action == "notification":
        window.notification_bar.secondary_button.click()
    else:
        window.undo()
    for _ in range(3):
        assert window.inspector.edit_instruction_edit.toPlainText() == instruction
        assert window._edit_undo_changes[token] is change
        window.redo()
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
        window.undo()
    assert window.inspector.edit_instruction_edit.toPlainText() == instruction


@pytest.mark.parametrize("newer_draft", ("typed", "recalled", "changed-back"))
def test_redo_preserves_newer_edit_drafts_including_identical_recalls(
    application: QApplication, newer_draft: str
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    instruction = "Open the gate."
    controller.execute(RenameCardCommand(card_id=card.id, name="Edited"))
    background.image_applied.emit(
        AppliedImageChange(
            token=controller.current_undo_token,
            card_id=card.id,
            revision_id=card.active_revision.id,
            previous_revision=card.active_revision,
            operation=EditImageOperation(instruction),
        )
    )
    window.undo()
    assert window.inspector.edit_instruction_edit.toPlainText() == instruction
    if newer_draft == "typed":
        window.inspector.edit_instruction_edit.setPlainText("Repaint the gate blue.")
    elif newer_draft == "recalled":
        window.inspector.set_edit_instruction(instruction)
    else:
        window.inspector.edit_instruction_edit.setPlainText("Another draft")
        window.inspector.edit_instruction_edit.setPlainText(instruction)
    draft = window.inspector.edit_instruction_draft
    window.redo()
    assert window.inspector.edit_instruction_draft == draft
    window.undo()
    assert window.inspector.edit_instruction_draft == draft


@pytest.mark.parametrize("recall_at", ("completion", "redo"))
@pytest.mark.parametrize("activation", ("click", "keyboard"))
def test_history_recall_survives_edit_completion_or_redo_on_same_version(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    recall_at: str,
    activation: str,
) -> None:
    instruction = "Open the gate.\nKeep its blue paint."
    card = Card(
        name="Card",
        revisions=(
            CardRevision(
                description="A courtyard",
                background=_generated_background(
                    asset_id=uuid4(),
                    image_path="assets/cards/card/generated.png",
                ),
            ),
        ),
    )
    previous = card.active_revision.model_copy(
        update={"background": _edited_background(card, instruction)}
    )
    card = card.model_copy(update={"revisions": (previous,)})
    window, controller, _workers, workflow = _window(Stack(name="Demo", cards=(card,)))
    monkeypatch.setattr(window, "_refine_source_size", lambda _card: (512, 384))
    window.render_document()
    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._edit_tab_index)
    window.inspector.set_edit_instruction(instruction)
    window._edit_background(instruction, CurrentSourceSize(width=512, height=384))
    result = _edited_background(card, instruction)
    changed = controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id, revision_id=previous.id, background=result
        )
    )
    window.render_document(changed)
    change = AppliedImageChange(
        token=controller.current_undo_token,
        card_id=card.id,
        revision_id=previous.id,
        previous_revision=previous,
        operation=EditImageOperation(instruction),
    )
    assert window.inspector.edit_history_list.count() == 2
    if recall_at == "redo":
        workflow.image_applied.emit(change)
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
        window.undo()
        assert window.inspector.edit_history_list.count() == 1
    before = window.inspector.edit_instruction_draft
    document = controller.document
    token = controller.current_undo_token
    history = window.inspector.edit_history_list
    if activation == "click":
        history.itemClicked.emit(history.item(0))
        history.itemActivated.emit(history.item(0))
    else:
        history.setCurrentRow(0)
        QTest.keyClick(history, Qt.Key.Key_Return)
    recalled = window.inspector.edit_instruction_draft
    assert recalled.text == instruction
    assert recalled.sequence == before.sequence + 1
    assert controller.document == document
    assert controller.current_undo_token == token
    if recall_at == "completion":
        workflow.image_applied.emit(change)
    else:
        window.redo()
    assert window.inspector.edit_instruction_draft == recalled
    assert window.inspector.edit_history_list.count() == 2
    window.undo()
    assert window.inspector.edit_instruction_draft == recalled
    assert window.inspector.edit_history_list.count() == 1
    assert len(workflow.edit_calls) == 1
    assert workflow.generate_calls == workflow.refine_calls == []


@pytest.mark.parametrize("context", ("card", "revision", "away-and-back"))
def test_edit_completion_and_undo_do_not_touch_unrelated_context(
    application: QApplication, context: str
) -> None:
    other = Card(name="Other")
    first = _stack().cards[0]
    window, controller, _workers, background = _window(Stack(name="Stack", cards=(first, other)))
    instruction = "Open the gate."
    window.inspector.set_edit_instruction(instruction)
    window._edit_background(instruction, CurrentSourceSize(width=512, height=384))
    controller.execute(RenameCardCommand(card_id=first.id, name="Edited"))
    token = controller.current_undo_token
    change = AppliedImageChange(
        token=token,
        card_id=first.id,
        revision_id=first.active_revision.id,
        previous_revision=first.active_revision,
        operation=EditImageOperation(instruction),
    )
    if context == "revision":
        controller.execute(
            ActivateRevisionCommand(card_id=first.id, revision_id=first.revisions[1].id)
        )
    else:
        window._selected_card_id = other.id
    window.render_document()
    if context == "away-and-back":
        window._selected_card_id = first.id
        window.render_document()
    draft = window.inspector.edit_instruction_draft
    background.image_applied.emit(change)
    assert window.inspector.edit_instruction_draft == draft
    window.inspector.clear_edit_instruction()
    if context == "revision":
        # Undo activation outside this window to keep its unrelated rendered context.
        assert controller.undo()
        assert controller.undo_if_current(token)
        window._restore_edit_instruction_after_undo(token)
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
    elif context == "card":
        window.undo()
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
    else:
        window.undo()
        assert window.inspector.edit_instruction_edit.toPlainText() == instruction


@pytest.mark.parametrize("action_id", ("undo", "create-image-revision"))
@pytest.mark.parametrize("stale_reason", ("command", "replacement", "run"))
@pytest.mark.parametrize(
    "operation",
    (ImageOperation("generate"), ImageOperation("refine"), EditImageOperation("Open it.")),
)
def test_image_actions_reject_stale_history_and_run_mode(
    application: QApplication,
    action_id: str,
    stale_reason: str,
    operation: ImageOperation | EditImageOperation,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    controller.execute(RenameCardCommand(card_id=card.id, name="Applied"))
    change = AppliedImageChange(
        token=controller.current_undo_token,
        card_id=card.id,
        revision_id=card.active_revision.id,
        previous_revision=card.active_revision,
        operation=operation,
    )
    background.image_applied.emit(change)
    if stale_reason == "command":
        controller.execute(RenameCardCommand(card_id=card.id, name="Newer"))
    elif stale_reason == "replacement":
        controller.replace_document(Stack(name="Replacement", cards=(card,)))
        window._document_replaced(controller.document)
    else:
        window.mode_button.click()
    before = controller.document
    before_token = controller.current_undo_token
    window._notification_action_requested(action_id)
    assert controller.document == before
    assert controller.current_undo_token == before_token
    background.image_applied.emit(change)
    assert window._applied_image_change is None
    assert window.inspector.edit_instruction_edit.toPlainText() == ""


def test_version_checkpoint_preserves_card_selection_history(
    application: QApplication,
) -> None:
    card, other = Card(name="Result"), Card(name="Selected")
    window, controller, _workers, background = _window(Stack(name="Stack", cards=(card, other)))
    controller.execute(RenameCardCommand(card_id=card.id, name="Applied"))
    background.image_applied.emit(
        AppliedImageChange(
            token=controller.current_undo_token,
            card_id=card.id,
            revision_id=card.active_revision.id,
            previous_revision=card.active_revision,
            operation=ImageOperation("generate"),
        )
    )
    window._selected_card_id = other.id
    window.render_document()
    window._create_image_revision()
    created = controller.document.cards[0].active_revision
    assert window._selected_card_id == card.id
    window.undo()
    assert window._selected_card_id == other.id
    window.redo()
    assert window._selected_card_id == card.id
    assert controller.document.cards[0].active_revision.id == created.id


@pytest.mark.parametrize("discard", ("branch", "clear", "replace"))
def test_edit_undo_metadata_lives_only_as_long_as_history(
    application: QApplication, discard: str
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    controller.execute(RenameCardCommand(card_id=card.id, name="Edited"))
    token = controller.current_undo_token
    change = AppliedImageChange(
        token=token,
        card_id=card.id,
        revision_id=card.active_revision.id,
        previous_revision=card.active_revision,
        operation=EditImageOperation("Open the gate."),
    )
    background.image_applied.emit(change)
    window.undo()
    assert token in window._edit_undo_changes
    assert token in window._restored_edit_drafts
    if discard == "branch":
        controller.execute(RenameCardCommand(card_id=card.id, name="Unrelated branch"))
    elif discard == "clear":
        controller.clear_history()
    else:
        controller.replace_document(controller.document)
        window._document_replaced(controller.document)
    window.render_document()
    assert window._edit_undo_changes == {}
    assert window._restored_edit_drafts == {}
    background.image_applied.emit(change)
    assert window._edit_undo_changes == {}


@pytest.mark.parametrize(
    "operation",
    (ImageOperation("generate"), ImageOperation("refine"), EditImageOperation("Open it.")),
)
def test_dismissing_image_result_keeps_it_on_current_version(
    application: QApplication,
    operation: ImageOperation | EditImageOperation,
) -> None:
    original = CardRevision(description="A courtyard")
    card = Card(name="Card", revisions=(original,))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    background = _generated_background(
        asset_id=uuid4(),
        image_path=f"assets/cards/{card.id}/generated.png",
        description=original.description,
    )
    changed = controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=original.id,
            background=background,
        )
    )
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._image_applied(
        AppliedImageChange(
            operation=operation,
            token=token,
            card_id=card.id,
            revision_id=original.id,
            previous_revision=original,
        )
    )

    window.notification_bar.dismiss_button.click()

    revision = controller.document.cards[0].active_revision
    assert revision.id == original.id
    assert revision.background == background
    assert controller.current_undo_token == token
    assert controller.document == changed
    assert window._applied_image_change is None
    window.undo()
    assert controller.document.cards[0].active_revision == original
    assert window.inspector.edit_instruction_edit.toPlainText() == (
        operation.instruction if isinstance(operation, EditImageOperation) else ""
    )


def test_notification_undo_cannot_mutate_document_in_run_mode(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)

    window.mode_button.click()
    assert window.notification_bar.current_key != "undo"
    window._notification_action_requested("undo")
    assert controller.document.cards[0].name == "Renamed"


def test_new_workflow_progress_clears_stale_failure_notification(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    window._show_error(
        "background-error",
        "Image generation failed",
        detail="Old failure",
    )
    background.busy = True
    background.progress_changed.emit("Generating image…")

    card = controller.document.cards[0]
    controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    assert window.notification_bar.current_key == "undo"


def test_generation_progress_bar_uses_image_steps(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    progress_container = window.generation_progress_container
    progress = window.generation_progress_bar

    background.busy = True
    background.progress_changed.emit("Generating image...")
    assert not progress_container.isHidden()
    assert window.generation_step_label.text() == "Generating image..."
    assert progress.minimum() == 0
    assert progress.maximum() == 0

    background.generation_progress_changed.emit(1, 4)
    assert progress.minimum() == 0
    assert progress.maximum() == 4
    assert progress.value() == 1

    background.generation_progress_changed.emit(3, 4)
    assert progress.value() == 3

    background.generation_progress_changed.emit(4, 4)
    assert progress.value() == 4

    background.generation_progress_changed.emit(0, 0)
    assert progress.minimum() == 0
    assert progress.maximum() == 0

    background.busy = False
    background.progress_changed.emit("Image generated")
    assert progress_container.isHidden()
    assert window.generation_step_label.text() == ""


def test_generation_cancel_button_cancels_the_active_process(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    background.busy = True
    background.progress_changed.emit("Generating image...")
    window.cancel_generation_button.click()

    assert background.cancel_calls == 1
    assert not background.busy


def test_background_success_clears_previous_cancellation_notice(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window._background_progress_changed("Generation cancelled")
    assert window.notification_bar.current_key == "background-cancelled"

    window._background_progress_changed("Image generated")
    assert window.notification_bar.isHidden()


def test_document_replacement_clears_document_specific_notifications(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    background.busy = True
    window._show_error(
        "background-error",
        "Image generation failed",
    )
    window._set_canvas_card_name_error("Invalid card name")
    window.inspector.set_hotspot_error("Invalid hotspot")

    window._document_replaced(object())
    assert window.notification_bar.isHidden()
    assert background.cancel_calls == 1
    assert window.canvas_card_name_error.isHidden()
    assert window.inspector.hotspot_error.isHidden()


def test_clean_session_state_clears_previous_document_error(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, _controller, _workers, _background = _window()
    window._show_document_error("Could Not Save Stack", "Disk is full")

    window._session_state_changed(
        DocumentSessionState(
            bundle_path=tmp_path / "Demo.hotcards",
            dirty=False,
            error=None,
        )
    )
    assert window.notification_bar.isHidden()


def test_hotspot_deletion_shows_targeted_undo(
    application: QApplication,
) -> None:
    interaction = Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    revision = CardRevision(hotspot_set=HotspotSet(interactions=(interaction,)))
    card = Card(name="Card", revisions=(revision,))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))

    window._delete_hotspot_interaction(interaction.id)

    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions == ()
    assert window.notification_bar.message_label.text() == "Hotspot deleted"
    assert not window.notification_bar.isHidden()

    window._undo_notification()

    restored = controller.document.cards[0].active_revision.hotspot_set
    assert restored is not None
    assert restored.interactions[0].id == interaction.id
    assert restored.interactions[0].polygons == interaction.polygons
    assert restored.interactions[0].label == "Go to Unresolved destination"


def test_completed_polygon_creates_and_selects_hotspot(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))

    polygon = Polygon(
        points=(
            Point(x=0.2, y=0.2),
            Point(x=0.4, y=0.2),
            Point(x=0.3, y=0.4),
        )
    )
    window.card_canvas.polygon_created.emit(polygon)

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    assert hotspot_set.interactions[0].polygons == (polygon,)
    assert window.inspector.selected_interaction_id == hotspot_set.interactions[0].id
    assert window.notification_bar.message_label.text() == "Hotspot created"


@pytest.mark.parametrize("image_tab", (0, 1))
def test_hotspot_editing_is_scoped_to_hotspots_tab(
    application: QApplication,
    tmp_path: Path,
    image_tab: int,
) -> None:
    image_path = tmp_path / "background.png"
    image = QPixmap(1024, 768)
    image.fill(QColor("navy"))
    assert image.save(str(image_path))
    interaction = Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    asset_id = uuid4()
    revision = CardRevision(
        description="A doorway",
        background=_generated_background(
            asset_id=asset_id,
            image_path=f"assets/cards/card/image-{asset_id}.png",
        ),
        hotspot_set=HotspotSet(interactions=(interaction,)),
    )
    card = Card(name="Card", revisions=(revision,))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    window.document_session = SimpleNamespace(
        store=SimpleNamespace(
            asset_path=lambda _path: image_path,
            image_asset_dimensions=lambda *_args, **_kwargs: (1024, 768),
        ),
        state=DocumentSessionState(bundle_path=None, dirty=False, error=None),
    )
    window.render_document()

    assert not window.inspector.hotspots_active
    assert not window.card_canvas._editable
    assert window.card_canvas._overlay_items == []

    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._hotspots_tab_index)
    assert window.inspector.hotspots_active
    assert window.card_canvas._editable
    assert window.card_canvas._overlay_items
    original_hotspots = controller.document.cards[0].active_revision.hotspot_set
    window.inspector.add_hotspot_button.click()
    assert window.card_canvas.drawing
    assert controller.document.cards[0].active_revision.hotspot_set == original_hotspots
    assert window.inspector.selected_interaction_id is None

    window.inspector.inspector_tabs.setCurrentIndex(image_tab)
    assert not window.card_canvas.drawing
    assert not window.card_canvas._editable
    assert window.card_canvas._overlay_items == []


def test_image_operation_explicitly_cancels_and_blocks_hotspot_draft(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "background.png"
    image = QPixmap(1024, 768)
    image.fill(QColor("navy"))
    assert image.save(str(image_path))
    interaction = Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    asset_id = uuid4()
    revision = CardRevision(
        description="A doorway",
        background=_generated_background(
            asset_id=asset_id,
            image_path=f"assets/cards/card/image-{asset_id}.png",
        ),
        hotspot_set=HotspotSet(interactions=(interaction,)),
    )
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, background = _window(Stack(name="Demo", cards=(card,)))
    window.document_session = SimpleNamespace(
        store=SimpleNamespace(
            asset_path=lambda _path: image_path,
            image_asset_dimensions=lambda *_args, **_kwargs: (1024, 768),
        ),
        state=DocumentSessionState(bundle_path=None, dirty=False, error=None),
    )
    window.render_document()
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.MFLUX,
            available=True,
            message="MFLUX is available",
        )
    )
    assert window.inspector.refine_background_button.isEnabled()
    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._hotspots_tab_index)
    window.card_canvas.begin_polygon(initial_point=Point(x=0.3, y=0.3))
    assert window.card_canvas.drawing

    background.busy = True
    background.active_operation = "refine"
    background.busy_changed.emit(True)

    assert not window.card_canvas.drawing
    assert not window.card_canvas._editable
    assert (
        window.notification_bar.message_label.text()
        == "Unfinished hotspot drawing cancelled before image processing"
    )
    window.card_canvas.begin_polygon(initial_point=Point(x=0.4, y=0.4))
    assert not window.card_canvas.drawing

    background.busy = False
    background.invocation_active = True
    background.busy_changed.emit(False)
    background.invocation_active_changed.emit(True)
    assert not window.card_canvas._editable
    assert not window.inspector.refine_background_button.isEnabled()

    background.invocation_active = False
    background.invocation_active_changed.emit(False)
    assert window.card_canvas._editable


def test_new_stack_dialog_creates_one_blank_revision(
    application: QApplication,
) -> None:
    dialog = NewStackDialog()
    assert not hasattr(dialog, "global_style_edit")
    assert not hasattr(dialog, "width_spin")
    assert not hasattr(dialog, "height_spin")
    assert [
        dialog.format_combo.itemText(index) for index in range(dialog.format_combo.count())
    ] == [
        "Square 1:1",
        "Landscape 4:3",
        "Portrait 3:4",
        "Widescreen 16:9",
    ]
    stack = dialog.stack()
    assert stack.aspect_ratio is AspectRatio.LANDSCAPE
    assert len(stack.cards) == 1
    assert len(stack.cards[0].revisions) == 1
    assert stack.cards[0].active_revision.style_id == HYPERCARD_STYLE_ID
    assert stack.new_card_style_id == HYPERCARD_STYLE_ID

    for aspect_ratio in AspectRatio:
        dialog.format_combo.setCurrentIndex(dialog.format_combo.findData(aspect_ratio))
        assert dialog.stack().aspect_ratio is aspect_ratio


def test_empty_stack_has_clear_first_card_path(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window(Stack(name="Empty"))
    assert window.canvas_pages.currentIndex() == 0
    assert window.empty_canvas_title.text() == "Create your first card"
