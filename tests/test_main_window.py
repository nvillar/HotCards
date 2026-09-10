"""Offscreen integration tests for revision-centric application chrome."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QKeySequence, QPixmap
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget
from shiboken6 import isValid

import hotcards.generation.mflux_generator as mflux_module
import hotcards.storage.stack_store as stack_store_module
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
    ApplyEditResultCommand,
    CreateCardCommand,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    RenameCardCommand,
    ReplaceGeneratedSoundCommand,
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
    AcceptedEdit,
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    EditDraft,
    EditOperation,
    GeneratedBackground,
    GeneratedSoundAsset,
    GenerateInputs,
    GenerateOperation,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    ImageOriginFacts,
    ImageProvenance,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
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
        self.edit_calls: list[tuple[object, EditDraft, object]] = []
        self.cancel_calls = 0
        self.closed = False

    def generate(self, card_id: object) -> None:
        self.generate_calls.append(card_id)

    def edit(
        self,
        card_id: object,
        *,
        draft: EditDraft,
        output_size: object,
    ) -> None:
        self.edit_calls.append((card_id, draft, output_size))

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


class FakeSoundPlayer(QObject):
    playback_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.played: list[Path] = []
        self.stop_calls = 0

    def play(self, path: Path) -> None:
        self.played.append(path)

    def stop(self) -> None:
        self.stop_calls += 1


@pytest.fixture
def application(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[QApplication]:
    resources: list[QObject] = []

    def retain_instances(resource_type: type[QObject]) -> None:
        initialize = resource_type.__init__

        def initialize_owned(resource: QObject, *args: object, **kwargs: object) -> None:
            initialize(resource, *args, **kwargs)
            resources.append(resource)

        monkeypatch.setattr(resource_type, "__init__", initialize_owned)

    for resource_type in (MainWindow, CardSidebar, NewStackDialog, DocumentSession, AdapterWorkers):
        retain_instances(resource_type)
    try:
        yield qt_application
    finally:
        for resource in reversed(resources):
            if not isValid(resource):
                continue
            if isinstance(resource, QWidget):
                if resource.parentWidget() is not None:
                    continue
                resource.close()
            elif isinstance(resource, AdapterWorkers):
                resource.shutdown(wait_milliseconds=1_000)
            elif isinstance(resource, DocumentSession):
                for timer in resource.findChildren(QTimer):
                    timer.stop()
                resource.controller.set_autosave_hook(None)
                resource.controller.set_owned_asset_release_hook(None)
            resource.deleteLater()
            QCoreApplication.sendPostedEvents(resource, QEvent.Type.DeferredDelete)


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
        provenance=ImageProvenance(
            authoring=GenerateOperation(inputs=GenerateInputs(description=description)),
            origin=ImageOriginFacts(
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
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.STABLE_AUDIO,
            available=True,
            message="Stable Audio is available.",
        )
    )
    return window, controller, workers, background


def _edited_background(card: Card, instruction: str) -> GeneratedBackground:
    revision = card.active_revision
    background = revision.background
    assert background is not None
    settings = background.provenance.settings
    asset_id = uuid4()
    expanded_prompt = f"{instruction}\n\nHidden Style addendum"
    return GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/{card.id}/image-{asset_id}.png",
        created_at=settings.generated_at,
        provenance=ImageProvenance(
            origin=ImageOriginFacts(
                render_prompt=expanded_prompt,
                settings=settings.model_copy(update={"seed": settings.seed + 1}),
                edit_lineage=(
                    *image_edit_lineage(background.provenance),
                    AcceptedEdit(instruction=instruction, expanded_prompt=expanded_prompt),
                ),
            ),
            authoring=EditOperation(
                source=DerivedImageSourceSnapshot(
                    card_id=card.id,
                    revision_id=revision.id,
                    background_id=background.id,
                    width=settings.width,
                    height=settings.height,
                    seed=settings.seed,
                ),
                output_size=CurrentSourceSize(width=settings.width, height=settings.height),
                prompt_token_count=20,
            ),
        ),
    )


def _apply_edit_for_test(
    controller: DocumentController,
    card_id: UUID,
    instruction: str,
) -> AppliedImageChange:
    card = next(card for card in controller.document.cards if card.id == card_id)
    if card.active_revision.background is None:
        asset_id = uuid4()
        controller.execute(
            ReplaceRevisionBackgroundCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                background=_generated_background(
                    asset_id=asset_id,
                    image_path=f"assets/cards/{card.id}/image-{asset_id}.png",
                ),
            )
        )
        card = next(card for card in controller.document.cards if card.id == card_id)
    submitted = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        instruction,
    )
    current = next(card for card in controller.document.cards if card.id == card_id)
    previous_revision = current.active_revision
    background = _edited_background(current, instruction.strip())
    controller.execute(
        ApplyEditResultCommand(
            card_id=card.id,
            revision_id=previous_revision.id,
            background=background,
            submitted_draft_generation_id=submitted.generation_id,
        )
    )
    token = controller.current_undo_token
    assert token is not None
    return AppliedImageChange(
        operation=EditImageOperation(instruction=instruction.strip()),
        token=token,
        card_id=card.id,
        revision_id=previous_revision.id,
        previous_revision=previous_revision,
    )


def _apply_image_for_test(
    controller: DocumentController,
    card_id: UUID,
    operation: ImageOperation | EditImageOperation,
) -> AppliedImageChange:
    if isinstance(operation, EditImageOperation):
        return _apply_edit_for_test(controller, card_id, operation.instruction)
    card = next(card for card in controller.document.cards if card.id == card_id)
    asset_id = uuid4()
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            background=_generated_background(
                asset_id=asset_id,
                image_path=f"assets/cards/{card.id}/image-{asset_id}.png",
                description=card.active_revision.description or "A courtyard",
            ),
        )
    )
    token = controller.current_undo_token
    assert token is not None
    return AppliedImageChange(
        operation=operation,
        token=token,
        card_id=card.id,
        revision_id=card.active_revision.id,
        previous_revision=card.active_revision,
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
    sound = SoundDefinition(name="Knock", prompt="A knock")
    window, controller, _workers, _background = _window(
        Stack(name="Demo", keys=(key,), sounds=(sound,), cards=(Card(name="Card"),))
    )
    window._show_style_manager()
    window._show_key_manager()
    window._show_sound_manager()
    style_manager = window.style_manager_window
    key_manager = window.key_manager_window
    sound_manager = window.sound_manager_window
    assert style_manager is not None
    assert key_manager is not None
    assert sound_manager is not None
    sound_manager.prompt_edit.setPlainText("An unfinished sound prompt")
    assert sound_manager.generate_button.isEnabled()
    key_manager.activateWindow()
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
    assert not window.sounds_button.isEnabled()
    assert not sound_manager.add_button.isEnabled()
    assert not sound_manager.name_edit.isEnabled()
    assert not sound_manager.prompt_edit.isEnabled()
    assert not sound_manager.duration_spin.isEnabled()
    assert not sound_manager.generate_button.isEnabled()
    assert sound_manager.prompt_edit.toPlainText() == "An unfinished sound prompt"
    assert not style_manager.add_button.isEnabled()
    assert not style_manager.name_edit.isEnabled()
    assert not key_manager.add_button.isEnabled()
    assert not key_manager.name_edit.isEnabled()
    assert key_manager.name_edit.text() == "Draft"
    assert key_manager.error_label.isVisible()
    window._close_utility_windows(commit_pending=False)
    window.close()
    application.processEvents()


@pytest.mark.parametrize("manager_state", ("never-opened", "closed", "reopened"))
def test_sound_completion_and_failure_are_owned_by_main_window(
    application: QApplication, manager_state: str
) -> None:
    sound = SoundDefinition(name="Knock", prompt="A knock")
    window, controller, _workers, _background = _window(
        Stack(name="Sounds", cards=(Card(name="Card"),), sounds=(sound,))
    )
    if manager_state != "never-opened":
        window._show_sound_manager()
        manager = window.sound_manager_window
        window._close_utility_windows()
        QCoreApplication.sendPostedEvents(manager, QEvent.Type.DeferredDelete)
    if manager_state == "reopened":
        window._show_sound_manager()
    asset_id = uuid4()
    generated = GeneratedSoundAsset(
        id=asset_id,
        audio_path=f"assets/sounds/{sound.id}/sound-{asset_id}.wav",
        provenance=SoundGenerationProvenance(
            prompt=sound.prompt,
            duration_seconds=sound.duration_seconds,
            seed=17,
            generation_duration_milliseconds=100,
        ),
        created_at=datetime.now(UTC),
    )
    changed = controller.execute(
        ReplaceGeneratedSoundCommand(sound_id=sound.id, generated=generated)
    )
    token = controller.current_undo_token
    workflow = window.sound_workflow
    assert workflow is not None
    workflow.document_changed.emit(changed)
    workflow.change_applied.emit("Sound generated", token)
    assert window._undo_notification_token == token
    assert window.notification_bar.message_label.text() == "Sound generated"
    if manager_state == "reopened":
        assert window.sound_manager_window.preview_button.isEnabled()
    assert controller.document.sound_by_id(sound.id).generated == generated
    workflow.failed.emit("Decoder rejected the generated audio")
    assert window.notification_bar.current_key == "sound-error"
    assert "Decoder rejected" in window.notification_bar.current_notification.detail
    assert controller.current_undo_token == token


@pytest.mark.parametrize("image_available,sound_available", ((True, False), (False, True)))
def test_image_and_sound_availability_gate_only_their_own_operations(
    application: QApplication, image_available: bool, sound_available: bool
) -> None:
    window, _controller, _workers, _background = _window(
        Stack(
            name="Services",
            cards=(Card(name="Card", revisions=(CardRevision(description="A scene"),)),),
            sounds=(SoundDefinition(name="Knock", prompt="A knock"),),
        )
    )
    window._show_sound_manager()
    for adapter, available in (
        (AdapterKind.MFLUX, image_available),
        (AdapterKind.STABLE_AUDIO, sound_available),
    ):
        window.apply_availability_diagnostic(
            AvailabilityDiagnostic(adapter, available, f"{adapter.value}: {available}")
        )
    assert window.inspector.generate_background_button.isEnabled() is image_available
    assert window.sound_manager_window.generate_button.isEnabled() is sound_available
    message = window.notification_bar.message_label.text()
    assert message == ("Stable Audio unavailable" if image_available else "MFLUX unavailable")


@pytest.mark.parametrize("player_kind", ("fake", "qt"))
def test_async_sound_playback_errors_reach_global_notifications(
    application: QApplication, player_kind: str
) -> None:
    if player_kind == "fake":
        window, _controller, _workers, _background = _window()
        player = window.sound_player
        assert isinstance(player, FakeSoundPlayer)
        player.playback_failed.emit("Audio device disconnected")
    else:
        window = MainWindow(
            DocumentController(_stack()), FakeWorkers(), FakeSettings(), start_diagnostics=False
        )
        window.sound_player._player.errorOccurred.emit(
            QMediaPlayer.Error.ResourceError, "Audio device disconnected"
        )
    assert window.notification_bar.current_key == "sound-playback"
    assert window.notification_bar.current_notification.detail == "Audio device disconnected"


@pytest.mark.parametrize("utilities_open", (False, True))
def test_render_uses_one_authoritative_snapshot_and_inspector_change_renders_once(
    application: QApplication, monkeypatch: pytest.MonkeyPatch, utilities_open: bool
) -> None:
    window, controller, _workers, _background = _window(
        _stack().model_copy(
            update={
                "keys": (KeyDefinition(name="Gate key"),),
                "sounds": (SoundDefinition(name="Knock", prompt="A knock"),),
            }
        )
    )
    if utilities_open:
        window._show_style_manager()
        window._show_key_manager()
        window._show_sound_manager()
    stale = controller.document
    controller.execute(RenameCardCommand(card_id=stale.cards[0].id, name="Current"))
    document_getter = DocumentController.document.fget
    snapshots: list[None] = []
    renders: list[Stack] = []
    original_render = window.inspector.render

    def counted_document(owner: DocumentController) -> Stack:
        snapshots.append(None)
        return document_getter(owner)

    def counted_render(document: Stack, card_id: UUID | None, **kwargs: object) -> None:
        renders.append(document)
        original_render(document, card_id, **kwargs)

    monkeypatch.setattr(DocumentController, "document", property(counted_document))
    monkeypatch.setattr(window.inspector, "render", counted_render)
    window.render_document(stale)
    assert len(snapshots) == 1
    assert len(renders) == 1
    assert window.canvas_card_name.text() == "Current"
    renders.clear()
    window.inspector.description_edit.setPlainText("Updated Description")
    assert window.inspector.commit_revision_metadata()
    assert len(renders) == 1
    assert renders[0].cards[0].active_revision.description == "Updated Description"


def test_repeated_render_reuses_decoded_dimensions_but_rechecks_asset_identity(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StackStore(tmp_path / "Dimensions.hotcards")
    card = Card(name="Card")
    asset_id = uuid4()
    source = tmp_path / "source.png"
    Image.new("RGB", (512, 384), "navy").save(source)
    image_path = store.store_image_asset(source, card_id=card.id, asset_id=asset_id)
    revision = CardRevision(
        description="A courtyard",
        background=_generated_background(asset_id=asset_id, image_path=image_path),
    )
    card = card.model_copy(update={"revisions": (revision,), "active_revision_id": revision.id})
    store.save(Stack(name="Dimensions", cards=(card,)))
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.open(store.bundle_path)
    decode = stack_store_module._validate_png_fd
    decodes: list[None] = []

    def counted_decode(fd: int, relative_path: str) -> tuple[int, int]:
        decodes.append(None)
        return decode(fd, relative_path)

    monkeypatch.setattr(stack_store_module, "_validate_png_fd", counted_decode)
    window = MainWindow(
        controller,
        FakeWorkers(),
        FakeSettings(),
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(controller),
        start_diagnostics=False,
    )
    assert len(decodes) == 1
    for _ in range(3):
        window.render_document()
    assert len(decodes) == 1
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (768, 576), "green").save(replacement)
    replacement.replace(store.asset_path(image_path))
    window.render_document()
    assert len(decodes) == 2
    assert window.inspector._image_source_size == (768, 576)
    store.asset_path(image_path).unlink()
    (store.bundle_path / image_path).symlink_to(source)
    window.render_document()
    assert len(decodes) == 2
    assert window.inspector._image_source_size is None
    assert "unavailable" in window.inspector.edit_output_error.text().lower()
    (store.bundle_path / image_path).unlink()
    source.replace(store.bundle_path / image_path)


def test_key_manager_done_discards_invalid_name_and_notifies(
    application: QApplication,
) -> None:
    first = KeyDefinition(name="First")
    second = KeyDefinition(name="Second")
    window, controller, _workers, _background = _window(
        Stack(name="Demo", keys=(first, second), cards=(Card(name="Card"),))
    )
    window._show_key_manager()
    manager = window.key_manager_window
    assert manager is not None

    manager.name_edit.setText("Second")
    manager.done_button.click()
    application.processEvents()

    assert window.key_manager_window is None
    assert controller.document.key_by_id(first.id).name == "First"
    assert window.notification_bar.current_key == "key-name-discarded"
    assert window.notification_bar.current_notification is not None
    assert window.notification_bar.current_notification.message == 'Kept Key name "First"'
    window.close()
    application.processEvents()


def test_sound_manager_invalid_name_keeps_other_edits_and_does_not_trap(
    application: QApplication,
) -> None:
    first = SoundDefinition(name="First", prompt="Original", duration_seconds=2)
    second = SoundDefinition(name="Second", prompt="Other", duration_seconds=3)
    window, controller, _workers, _background = _window(
        Stack(name="Demo", sounds=(first, second), cards=(Card(name="Card"),))
    )
    window._show_sound_manager()
    manager = window.sound_manager_window
    assert manager is not None

    manager.name_edit.setText("Second")
    manager.prompt_edit.setPlainText("Updated prompt")
    manager.duration_spin.setValue(4)
    assert "unique" in manager.error_label.text()
    QTest.keyClick(manager.name_edit, Qt.Key.Key_Return)
    assert manager.name_edit.text() == "Second"
    QTest.keyClick(manager.name_edit, Qt.Key.Key_Escape)
    assert manager.name_edit.text() == "First"
    assert manager.prompt_edit.toPlainText() == "Updated prompt"

    manager.name_edit.setText("Second")
    manager.done_button.click()
    application.processEvents()

    assert window.sound_manager_window is None
    changed = controller.document.sound_by_id(first.id)
    assert (changed.name, changed.prompt, changed.duration_seconds) == (
        "First",
        "Updated prompt",
        4,
    )
    assert window.notification_bar.current_key == "sound-name-discarded"
    assert window.notification_bar.current_notification is not None
    assert window.notification_bar.current_notification.message == 'Kept Sound name "First"'

    window._show_sound_manager()
    manager = window.sound_manager_window
    assert manager is not None
    manager.name_edit.setText("Second")
    manager.delete_button.click()
    assert [sound.name for sound in controller.document.sounds] == ["Second"]
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
    derived_background = _edited_background(source_card, "Open the door.")
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


def test_description_style_and_reference_changes_cancel_in_flight_generation(
    application: QApplication,
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

    editor = window.inspector.description_edit
    combo = window.inspector.style_combo
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


def test_edit_draft_change_cancels_in_flight_work_without_losing_text(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    background.busy = True

    window.inspector.edit_instruction_edit.setPlainText("Keep the blue gate.")

    assert background.cancel_calls == 1
    assert (
        controller.edit_draft(
            card.id,
            card.active_revision.id,
        ).instruction
        == "Keep the blue gate."
    )


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
    draft = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "Unfinished Edit",
    )
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
    assert controller.edit_draft(card.id, card.active_revision.id) == draft
    assert (
        StackStore(tmp_path / "Original.hotcards").load().cards[0].active_revision.edit_draft
        == draft
    )
    assert (
        StackStore(tmp_path / "Copy.hotcards").load().cards[0].active_revision.edit_draft == draft
    )


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


def test_entering_run_flushes_the_persisted_edit_draft(
    application: QApplication,
    tmp_path: Path,
) -> None:
    stack = _stack()
    controller = DocumentController(stack)
    session = DocumentSession(controller, debounce_milliseconds=60_000)
    bundle = tmp_path / "RunDraft.hotcards"
    session.create(stack, bundle)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(controller),  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    card = controller.document.cards[0]
    window.inspector.edit_instruction_edit.setPlainText("Finish the painted arch.")
    assert session.state.dirty

    window.mode_button.click()

    assert window._is_running
    assert not session.state.dirty
    assert StackStore(bundle).load().cards[0].active_revision.edit_draft == (
        controller.edit_draft(card.id, card.active_revision.id)
    )
    window.close()


def test_failed_edit_draft_flush_blocks_run_without_losing_the_draft(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack()
    controller = DocumentController(stack)
    session = DocumentSession(controller, debounce_milliseconds=60_000)
    bundle = tmp_path / "BlockedRunDraft.hotcards"
    session.create(stack, bundle)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(controller),  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    card = controller.document.cards[0]
    window.inspector.edit_instruction_edit.setPlainText("Keep this instruction.")
    assert session.store is not None
    real_save = session.store.save

    def fail_save(_stack: Stack) -> None:
        raise StackStoreError("draft storage unavailable")

    monkeypatch.setattr(session.store, "save", fail_save)
    window.mode_button.click()

    assert not window._is_running
    assert (
        controller.edit_draft(
            card.id,
            card.active_revision.id,
        ).instruction
        == "Keep this instruction."
    )
    assert session.state.dirty
    assert window.notification_bar.message_label.text() == "Could Not Save Stack"

    monkeypatch.setattr(session.store, "save", real_save)
    assert session.flush()
    window.close()


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
    assert not window.image_model_combo.isEnabled()
    assert not window.sound_model_combo.isEnabled()
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
    assert window.image_model_combo.isEnabled()
    assert window.sound_model_combo.isEnabled()
    window.hide()
    application.processEvents()


def test_ai_recovery_uses_notification_bar_and_bottom_model_picker(
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
    assert window.statusBar().isAncestorOf(window.image_model_combo)
    assert "MFLUX is unavailable" in window._service_status_detail
    assert window.notification_bar.current_key == "ai-services"
    assert window.notification_bar.primary_button.text() == "Select Model"
    assert window.notification_bar.secondary_button.text() == "Check Again"
    window.show()
    window.notification_bar.primary_button.click()
    assert window.image_model_combo.view().isVisible()
    window.image_model_combo.hidePopup()

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


def test_bottom_model_pickers_persist_image_model_and_cancel_active_work(
    application: QApplication,
) -> None:
    settings = FakeSettings()
    controller = DocumentController(_stack())
    workers = FakeWorkers()
    background = FakeBackgroundWorkflow(controller)
    sound_workflow = FakeSoundWorkflow()

    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        settings,
        availability_checks={AdapterKind.MFLUX: lambda: None},
        background_workflow=background,  # type: ignore[arg-type]
        sound_workflow=sound_workflow,  # type: ignore[arg-type]
        start_diagnostics=False,
    )

    assert "Settings" not in [action.text() for action in window.menuBar().actions()]
    assert window.image_model_combo.currentText() == "FLUX.2 Klein 4B"
    assert window.sound_model_combo.currentText() == "Stable Audio 3 Small-SFX"
    assert settings.values == {}
    assert background.cancel_calls == 0
    assert not workers.mflux_operations

    window.image_model_combo.setCurrentIndex(window.image_model_combo.findData("flux2-klein-9b-kv"))

    assert settings.values["generation/mflux_model"] == "flux2-klein-9b-kv"
    assert settings.values["generation/stable_audio_model"] == (
        "stabilityai/stable-audio-3-small-sfx"
    )
    assert background.cancel_calls == 1
    assert sound_workflow.cancel_calls == 0
    assert len(workers.mflux_operations) == 1

    reopened = MainWindow(
        DocumentController(_stack()),
        FakeWorkers(),  # type: ignore[arg-type]
        settings,
        start_diagnostics=False,
    )
    assert reopened.image_model_combo.currentText() == "FLUX.2 Klein 9B KV"
    assert reopened.sound_model_combo.currentText() == "Stable Audio 3 Small-SFX"

    window.mode_button.click()
    assert not window.image_model_combo.isEnabled()
    assert not window.sound_model_combo.isEnabled()


def test_bottom_model_pickers_stay_right_aligned_beside_progress(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window.show()
    application.processEvents()

    layout = window.model_pickers.layout()
    assert layout is not None
    assert [label.text() for label in window.model_pickers.findChildren(QLabel)] == [
        "Image",
        "Sound",
    ]
    assert layout.contentsMargins() == window.generation_progress_layout.contentsMargins()
    assert layout.contentsMargins().left() == layout.contentsMargins().right() == 8
    assert layout.spacing() == 8
    image_label, sound_label = window.model_pickers.findChildren(QLabel)
    group_gap = sound_label.x() - (window.image_model_combo.x() + window.image_model_combo.width())
    image_label_gap = window.image_model_combo.x() - (image_label.x() + image_label.width())
    sound_label_gap = window.sound_model_combo.x() - (sound_label.x() + sound_label.width())
    assert group_gap >= max(image_label_gap, sound_label_gap) + 8
    assert window.image_model_combo.geometry().right() < window.sound_model_combo.geometry().left()
    assert window.statusBar().isAncestorOf(window.model_pickers)
    assert window.generation_progress_container.isHidden()
    assert window.model_pickers.isVisible()
    right_margin = window.statusBar().width() - window.model_pickers.geometry().right()
    assert 0 <= right_margin <= 40

    window.generation_progress_container.show()
    window.resize(window.width() + 200, window.height())
    application.processEvents()
    assert window.model_pickers.isVisible()
    assert window.generation_progress_container.geometry().right() < (
        window.model_pickers.geometry().left()
    )
    assert window.statusBar().width() - window.model_pickers.geometry().right() == right_margin


def test_edit_menu_has_only_history_actions_and_duplicate_shortcut_is_retained(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    edit_action = next(action for action in window.menuBar().actions() if action.text() == "Edit")
    edit_menu = edit_action.menu()
    assert edit_menu is not None
    assert edit_menu.actions() == [window.undo_action, window.redo_action]
    assert window.duplicate_card_action in window.actions()
    assert window.duplicate_card_action.shortcut() == QKeySequence("Ctrl+D")


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

    window = MainWindow(
        controller,
        workers,
        settings,
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
    window.image_model_combo.setCurrentIndex(window.image_model_combo.findData("flux2-klein-9b-kv"))
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

    assert window.inspector.inspector_tabs.tabText(1) == "Edit"
    assert window.inspector.generate_background_button.isEnabled() == bool(description)
    assert not window.inspector.edit_background_button.isEnabled()
    assert window.inspector.edit_resolution_combo.count() == 1
    assert window.inspector.edit_resolution_combo.itemText(0) == "Full"
    window.inspector.edit_instruction_edit.setPlainText("Open the gate.")
    assert window.inspector.edit_background_button.isEnabled()
    window.inspector.edit_background_button.click()

    assert len(background.edit_calls) == 1
    card_id, draft, output_size = background.edit_calls[0]
    assert card_id == card.id
    assert draft.instruction == "Open the gate."
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
    change = _apply_edit_for_test(controller, card.id, "Open the garden gate.")
    window.render_document()
    background.image_applied.emit(change)
    if cleanup_error:
        background.failed.emit(RuntimeError("committed image cleanup failed"))
        assert window._applied_image_change == change
        window.undo()
    else:
        window.notification_bar.secondary_button.click()

    assert window.inspector.edit_instruction_edit.toPlainText() == "Open the garden gate."
    assert (
        controller.document.cards[0].active_revision.edit_draft
        == change.previous_revision.edit_draft
    )


def test_edit_instruction_waits_for_its_exact_undo_token(
    application: QApplication,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    change = _apply_edit_for_test(controller, card.id, "Open the garden gate.")
    window.render_document()
    background.image_applied.emit(change)
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
    change = _apply_edit_for_test(controller, card.id, "Open the garden gate.")
    window.render_document()
    background.image_applied.emit(change)
    window.inspector.edit_instruction_edit.setPlainText("Try a blue gate instead.")

    window.undo()

    assert window.inspector.edit_instruction_edit.toPlainText() == "Try a blue gate instead."


@pytest.mark.parametrize("newer_draft", ("unchanged", "new-text", "recalled", "changed-back"))
def test_completed_edit_clears_only_its_matching_instruction(
    application: QApplication, newer_draft: str
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    source_id = uuid4()
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            background=_generated_background(
                asset_id=source_id,
                image_path=f"assets/cards/{card.id}/image-{source_id}.png",
            ),
        )
    )
    card = controller.document.cards[0]
    window.render_document()
    instruction = "Open the garden gate."
    window.inspector.set_edit_instruction(f"  {instruction}  ")
    window._edit_background(instruction, CurrentSourceSize(width=512, height=384))
    submitted = background.edit_calls[-1][1]
    if newer_draft == "new-text":
        window.inspector.edit_instruction_edit.setPlainText("Keep this newer draft.")
    elif newer_draft == "recalled":
        window.inspector.set_edit_instruction(f"  {instruction}  ")
    elif newer_draft == "changed-back":
        window.inspector.edit_instruction_edit.setPlainText("Something else")
        window.inspector.edit_instruction_edit.setPlainText(f"  {instruction}  ")
    draft = window.inspector.edit_instruction_edit.toPlainText()
    current = controller.document.cards[0].active_revision
    changed = controller.execute(
        ApplyEditResultCommand(
            card_id=card.id,
            revision_id=current.id,
            background=_edited_background(controller.document.cards[0], instruction),
            submitted_draft_generation_id=submitted.generation_id,
        )
    )
    window.render_document(changed)
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


def test_mouse_focus_commit_does_not_render_before_button_release(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = CardRevision(description="A courtyard")
    card = Card(name="Card", revisions=(revision,))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    window.show()
    draft = "A changed courtyard"
    editor = window.inspector.description_edit
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
        None,
        Qt.FocusReason.MouseFocusReason,
    )

    changed_revision = controller.document.cards[0].active_revision
    assert changed_revision.description == draft
    assert renders == []
    window.close()


@pytest.mark.parametrize(
    "operation",
    (ImageOperation("generate"), EditImageOperation("Open it.")),
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
        background=(
            _generated_background(asset_id=uuid4(), image_path="assets/cards/source.png")
            if isinstance(operation, EditImageOperation)
            else None
        ),
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
    if isinstance(operation, EditImageOperation):
        background = _edited_background(card, operation.instruction)
        submitted = controller.replace_edit_draft(
            card.id,
            original.id,
            operation.instruction,
        )
        previous_revision = controller.document.cards[0].active_revision
        image_command = ApplyEditResultCommand(
            card_id=card.id,
            revision_id=original.id,
            background=background,
            submitted_draft_generation_id=submitted.generation_id,
        )
    else:
        previous_revision = original
        image_command = ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=original.id,
            background=background,
        )
    changed = controller.execute(image_command)
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    change = AppliedImageChange(
        operation=operation,
        token=token,
        card_id=card.id,
        revision_id=original.id,
        previous_revision=previous_revision,
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
    assert changed_card.revisions[0].background == original.background
    assert changed_card.revisions[0].hotspot_set == original.hotspot_set
    assert changed_card.active_revision.id != original.id
    assert changed_card.active_revision.description == original.description
    assert changed_card.active_revision.hotspot_set == original.hotspot_set
    assert changed_card.active_revision.background == background
    assert (
        changed_card.revisions[0].model_copy(update={"edit_draft": original.edit_draft}) == original
    )
    assert (
        changed_card.active_revision.model_copy(
            update={
                "id": original.id,
                "edit_draft": changed.cards[0].active_revision.edit_draft,
            }
        )
        == changed.cards[0].active_revision
    )
    assert changed_card.active_revision.edit_draft.instruction == ""
    assert (
        changed_card.active_revision.edit_draft.generation_id
        != changed.cards[0].active_revision.edit_draft.generation_id
    )
    assert not background_workflow.generate_calls
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
    assert controller.document.cards[0].active_revision == previous_revision
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
    change = _apply_edit_for_test(controller, card.id, instruction)
    window.render_document()
    background.image_applied.emit(change)
    assert window.inspector.edit_instruction_edit.toPlainText() == ""
    if undo_action == "notification":
        window.notification_bar.secondary_button.click()
    else:
        window.undo()
    for _ in range(3):
        assert window.inspector.edit_instruction_edit.toPlainText() == instruction
        window.redo()
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
        window.undo()
    assert window.inspector.edit_instruction_edit.toPlainText() == instruction


def test_focused_edit_instruction_routes_global_undo_and_redo_to_draft_history(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    window.show()
    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._edit_tab_index)
    window.inspector.edit_instruction_edit.setFocus()
    window.inspector.edit_instruction_edit.setPlainText("Open the gate.")
    application.processEvents()
    token = controller.current_undo_token

    assert window.undo_action.isEnabled()
    QTest.keySequence(
        window.inspector.edit_instruction_edit,
        QKeySequence(QKeySequence.StandardKey.Undo),
    )

    assert controller.current_undo_token == token
    assert controller.edit_draft(card.id, card.active_revision.id).instruction == ""
    assert window.redo_action.isEnabled()
    QTest.keySequence(
        window.inspector.edit_instruction_edit,
        QKeySequence(QKeySequence.StandardKey.Redo),
    )
    assert controller.current_undo_token == token
    assert (
        controller.edit_draft(
            card.id,
            card.active_revision.id,
        ).instruction
        == "Open the gate."
    )
    window.close()
    application.processEvents()


@pytest.mark.parametrize("newer_draft", ("typed", "recalled", "changed-back"))
def test_redo_preserves_newer_edit_drafts_including_identical_recalls(
    application: QApplication, newer_draft: str
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    instruction = "Open the gate."
    change = _apply_edit_for_test(controller, card.id, instruction)
    window.render_document()
    background.image_applied.emit(change)
    window.undo()
    assert window.inspector.edit_instruction_edit.toPlainText() == instruction
    if newer_draft == "typed":
        window.inspector.edit_instruction_edit.setPlainText("Repaint the gate blue.")
    elif newer_draft == "recalled":
        window.inspector.set_edit_instruction(instruction)
    else:
        window.inspector.edit_instruction_edit.setPlainText("Another draft")
        window.inspector.edit_instruction_edit.setPlainText(instruction)
    draft = controller.edit_draft(card.id, card.active_revision.id)
    window.redo()
    assert controller.edit_draft(card.id, card.active_revision.id) == draft
    window.undo()
    assert controller.edit_draft(card.id, card.active_revision.id) == draft


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
    monkeypatch.setattr(window, "_image_source_size", lambda _card: (512, 384))
    window.render_document()
    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._edit_tab_index)
    window.inspector.set_edit_instruction(instruction)
    window._edit_background(instruction, CurrentSourceSize(width=512, height=384))
    submitted = workflow.edit_calls[-1][1]
    before_edit = controller.document.cards[0].active_revision
    result = _edited_background(card, instruction)
    changed = controller.execute(
        ApplyEditResultCommand(
            card_id=card.id,
            revision_id=previous.id,
            background=result,
            submitted_draft_generation_id=submitted.generation_id,
        )
    )
    window.render_document(changed)
    change = AppliedImageChange(
        token=controller.current_undo_token,
        card_id=card.id,
        revision_id=previous.id,
        previous_revision=before_edit,
        operation=EditImageOperation(instruction),
    )
    assert window.inspector.edit_history_list.count() == 2
    if recall_at == "redo":
        workflow.image_applied.emit(change)
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
        window.undo()
        assert window.inspector.edit_history_list.count() == 1
    before = controller.edit_draft(card.id, previous.id)
    document = controller.document
    token = controller.current_undo_token
    history = window.inspector.edit_history_list
    if activation == "click":
        history.itemClicked.emit(history.item(0))
        history.itemActivated.emit(history.item(0))
    else:
        history.setCurrentRow(0)
        QTest.keyClick(history, Qt.Key.Key_Return)
    recalled = controller.edit_draft(card.id, previous.id)
    assert recalled.instruction == instruction
    assert recalled.generation_id != before.generation_id
    assert controller.document != document
    assert controller.current_undo_token == token
    if recall_at == "completion":
        workflow.image_applied.emit(change)
    else:
        window.redo()
    assert controller.edit_draft(card.id, previous.id) == recalled
    assert window.inspector.edit_history_list.count() == 2
    window.undo()
    assert controller.edit_draft(card.id, previous.id) == recalled
    assert window.inspector.edit_history_list.count() == 1
    assert len(workflow.edit_calls) == 1
    assert workflow.generate_calls == []


@pytest.mark.parametrize("context", ("card", "revision", "away-and-back"))
def test_edit_completion_and_undo_do_not_touch_unrelated_context(
    application: QApplication, context: str
) -> None:
    other = Card(name="Other")
    first = _stack().cards[0]
    window, controller, _workers, background = _window(Stack(name="Stack", cards=(first, other)))
    instruction = "Open the gate."
    change = _apply_edit_for_test(controller, first.id, instruction)
    window.render_document()
    background.image_applied.emit(change)
    if context == "revision":
        controller.execute(
            ActivateRevisionCommand(card_id=first.id, revision_id=first.revisions[1].id)
        )
        unrelated_revision_id = first.revisions[1].id
        controller.replace_edit_draft(first.id, unrelated_revision_id, "Other version draft")
    else:
        window._selected_card_id = other.id
        controller.replace_edit_draft(
            other.id,
            other.active_revision.id,
            "Other card draft",
        )
    window.render_document()
    if context == "away-and-back":
        window._selected_card_id = first.id
        window.render_document()
        window.inspector.set_edit_instruction("Newer owner draft")
    if context == "revision":
        assert controller.undo()
        assert controller.undo_if_current(change.token)
        assert (
            controller.edit_draft(first.id, unrelated_revision_id).instruction
            == "Other version draft"
        )
        assert controller.edit_draft(first.id, first.active_revision.id).instruction == instruction
    elif context == "card":
        window.undo()
        assert controller.edit_draft(other.id, other.active_revision.id).instruction == (
            "Other card draft"
        )
    else:
        window.undo()
        assert (
            controller.edit_draft(first.id, first.active_revision.id).instruction
            == "Newer owner draft"
        )


@pytest.mark.parametrize("action_id", ("undo", "create-image-revision"))
@pytest.mark.parametrize("stale_reason", ("command", "replacement", "run"))
@pytest.mark.parametrize(
    "operation",
    (ImageOperation("generate"), EditImageOperation("Open it.")),
)
def test_image_actions_reject_stale_history_and_run_mode(
    application: QApplication,
    action_id: str,
    stale_reason: str,
    operation: ImageOperation | EditImageOperation,
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]
    change = _apply_image_for_test(controller, card.id, operation)
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
    background.image_applied.emit(
        _apply_image_for_test(controller, card.id, ImageOperation("generate"))
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


@pytest.mark.parametrize(
    "operation",
    (ImageOperation("generate"), EditImageOperation("Open it.")),
)
def test_dismissing_image_result_keeps_it_on_current_version(
    application: QApplication,
    operation: ImageOperation | EditImageOperation,
) -> None:
    original = CardRevision(description="A courtyard")
    card = Card(name="Card", revisions=(original,))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    change = _apply_image_for_test(controller, card.id, operation)
    changed = controller.document
    background = changed.cards[0].active_revision.background
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._image_applied(change)

    window.notification_bar.dismiss_button.click()

    revision = controller.document.cards[0].active_revision
    assert revision.id == original.id
    assert revision.background == background
    assert controller.current_undo_token == token
    assert controller.document == changed
    assert window._applied_image_change is None
    window.undo()
    assert controller.document.cards[0].active_revision == change.previous_revision
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
        flush=lambda: True,
        close_history=lambda: True,
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
        flush=lambda: True,
        close_history=lambda: True,
    )
    window.render_document()
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.MFLUX,
            available=True,
            message="MFLUX is available",
        )
    )
    assert window.inspector.generate_background_button.isEnabled()
    window.inspector.inspector_tabs.setCurrentIndex(window.inspector._hotspots_tab_index)
    window.card_canvas.begin_polygon(initial_point=Point(x=0.3, y=0.3))
    assert window.card_canvas.drawing

    background.busy = True
    background.active_operation = "edit"
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
    assert not window.inspector.generate_background_button.isEnabled()

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
