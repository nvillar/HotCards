"""Cancellable generated-sound workflow with stale-result suppression."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from hotcards.application.commands import (
    CommandError,
    ReplaceGeneratedSoundCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
    OwnedSoundAsset,
)
from hotcards.application.document_session import (
    DocumentSession,
    DocumentSessionError,
)
from hotcards.application.workers import AdapterWorkers, WorkerFailure, WorkerOperation
from hotcards.domain.models import (
    GeneratedSoundAsset,
    SoundGenerationProvenance,
    Stack,
)
from hotcards.generation.stable_audio import (
    STABLE_AUDIO_CFG,
    STABLE_AUDIO_CHANNELS,
    STABLE_AUDIO_MODEL,
    STABLE_AUDIO_RUNTIME,
    STABLE_AUDIO_SAMPLE_RATE,
    STABLE_AUDIO_SAMPLER,
    STABLE_AUDIO_STEPS,
    StableAudioCancellation,
    StableAudioGenerator,
    StableAudioRequest,
    StableAudioResult,
)
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
    StoredSoundAsset,
)


class SoundWorkflowError(ValueError):
    """A sound generation cannot start or be accepted safely."""


@dataclass(frozen=True, slots=True)
class _SoundTarget:
    stack_id: UUID
    bundle_path: Path
    sound_id: UUID
    prompt: str
    duration_seconds: int
    generated_asset_id: UUID | None


class SoundWorkflow(QObject):
    """Coordinate one Stable Audio generation and persisted catalog replacement."""

    generation_started = Signal(object)
    sampling_progress = Signal(int, int)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        controller: DocumentController,
        session: DocumentSession,
        workers: AdapterWorkers,
        generator: StableAudioGenerator,
        *,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.session = session
        self.workers = workers
        self.generator = generator
        self._request_id: UUID | None = None
        self._operation: WorkerOperation | None = None
        self._cancellation: StableAudioCancellation | None = None
        self._temporary_path: Path | None = None

    @property
    def is_active(self) -> bool:
        return self._request_id is not None

    def generate(self, sound_id: UUID) -> None:
        """Start one generation from the current persisted Sound authoring fields."""
        if self.is_active:
            raise SoundWorkflowError("another sound generation is already running")
        store = self.session.store
        if store is None:
            raise SoundWorkflowError("the stack must be saved before generating a Sound")
        document = self.controller.document
        try:
            sound = document.sound_by_id(sound_id)
        except StopIteration as error:
            raise SoundWorkflowError("the selected Sound no longer exists") from error
        if not sound.prompt:
            raise SoundWorkflowError("enter a Sound prompt before generating")
        target = _SoundTarget(
            stack_id=document.id,
            bundle_path=store.bundle_path,
            sound_id=sound.id,
            prompt=sound.prompt,
            duration_seconds=sound.duration_seconds,
            generated_asset_id=sound.generated.id if sound.generated is not None else None,
        )
        request_id = uuid4()
        cancellation = StableAudioCancellation()
        with tempfile.NamedTemporaryFile(
            prefix="hotcards-sound-",
            suffix=".wav",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        self._request_id = request_id
        self._cancellation = cancellation
        self._temporary_path = temporary_path
        request = StableAudioRequest(
            prompt=target.prompt,
            duration_seconds=target.duration_seconds,
        )
        operation = self.workers.run_stable_audio(
            lambda: self.generator.generate(
                request,
                temporary_path,
                cancellation=cancellation,
                on_sampling_step=self.sampling_progress.emit,
            ),
            stage="generating Sound",
            request_cancel=cancellation.request,
            dispose_result=_dispose_result,
        )
        self._operation = operation
        operation.succeeded.connect(
            lambda result: self._generation_succeeded(request_id, target, result)
        )
        operation.failed.connect(lambda failure: self._generation_failed(request_id, failure))
        operation.cancelled.connect(lambda: self._generation_cancelled(request_id))
        self.generation_started.emit(sound_id)

    def cancel(self) -> None:
        """Cancel and discard the active generation."""
        request_id = self._request_id
        if request_id is None:
            return
        operation = self._operation
        if operation is not None:
            operation.cancel()
        if self._request_id == request_id:
            self._finish(cancelled=True)

    def _generation_succeeded(
        self,
        request_id: UUID,
        target: _SoundTarget,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            if isinstance(result, StableAudioResult):
                _dispose_result(result)
            return
        if not isinstance(result, StableAudioResult):
            self._fail("Stable Audio returned an unexpected result")
            return
        if not self._target_is_current(target):
            _dispose_result(result)
            self._fail("the Sound or stack changed before generation completed")
            return
        store = self.session.store
        assert store is not None
        asset_id = uuid4()
        generated = GeneratedSoundAsset(
            id=asset_id,
            audio_path=store.sound_asset_path(target.sound_id, asset_id),
            provenance=SoundGenerationProvenance(
                model=STABLE_AUDIO_MODEL,
                runtime=STABLE_AUDIO_RUNTIME,
                prompt=target.prompt,
                duration_seconds=target.duration_seconds,
                seed=result.seed,
                steps=STABLE_AUDIO_STEPS,
                cfg=STABLE_AUDIO_CFG,
                sampler=STABLE_AUDIO_SAMPLER,
                sample_rate=STABLE_AUDIO_SAMPLE_RATE,
                channels=STABLE_AUDIO_CHANNELS,
                generation_duration_milliseconds=result.generation_duration_milliseconds,
            ),
            created_at=datetime.now(UTC),
        )
        command = ReplaceGeneratedSoundCommand(
            sound_id=target.sound_id,
            generated=generated,
        )
        before = self.controller.document
        owned_assets: list[OwnedSoundAsset] = []

        def persist(candidate: Stack) -> None:
            try:
                stored = store.store_sound_asset_and_save(
                    result.output_path,
                    sound_id=target.sound_id,
                    asset_id=asset_id,
                    duration_seconds=target.duration_seconds,
                    previous_stack=before,
                    changed_stack=candidate,
                )
            except StackStoreTransactionError as error:
                if isinstance(error.owned_asset, StoredSoundAsset):
                    owned_assets.append(
                        _owned_sound(store, target.sound_id, asset_id, error.owned_asset)
                    )
                raise
            owned_assets.append(_owned_sound(store, target.sound_id, asset_id, stored))

        previous_token = self.controller.current_undo_token
        try:
            changed = self.session.execute_persisted(
                command,
                persist=persist,
                owned_assets=owned_assets,
            )
        except (
            CommandError,
            DocumentMutationBlockedError,
            DocumentSessionError,
            StackStoreError,
            ValidationError,
        ) as error:
            if self.controller.document != before:
                self.document_changed.emit(self.controller.document)
            self._fail(str(error))
            return
        current_token = self.controller.current_undo_token
        self.document_changed.emit(changed)
        if current_token is not None and current_token != previous_token:
            self.change_applied.emit("Sound generated", current_token)
        self._temporary_path = None
        _dispose_result(result)
        self._finish()

    def _generation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id != self._request_id:
            return
        message = (
            failure.message
            if isinstance(failure, WorkerFailure)
            else "Stable Audio generation failed"
        )
        self._fail(message)

    def _generation_cancelled(self, request_id: UUID) -> None:
        if request_id == self._request_id:
            self._finish(cancelled=True)

    def _target_is_current(self, target: _SoundTarget) -> bool:
        store = self.session.store
        if store is None or store.bundle_path != target.bundle_path:
            return False
        document = self.controller.document
        if document.id != target.stack_id:
            return False
        try:
            sound = document.sound_by_id(target.sound_id)
        except StopIteration:
            return False
        return (
            sound.prompt == target.prompt
            and sound.duration_seconds == target.duration_seconds
            and (sound.generated.id if sound.generated is not None else None)
            == target.generated_asset_id
        )

    def _fail(self, message: str) -> None:
        self.failed.emit(message)
        self._finish()

    def _finish(self, *, cancelled: bool = False) -> None:
        path = self._temporary_path
        self._request_id = None
        self._operation = None
        self._cancellation = None
        self._temporary_path = None
        if path is not None:
            path.unlink(missing_ok=True)
        if cancelled:
            self.cancelled.emit()
        self.finished.emit()


def _dispose_result(result: object) -> None:
    if isinstance(result, StableAudioResult):
        result.output_path.unlink(missing_ok=True)


def _owned_sound(
    store: StackStore,
    sound_id: UUID,
    asset_id: UUID,
    stored: StoredSoundAsset,
) -> OwnedSoundAsset:
    return OwnedSoundAsset(
        bundle_path=store.bundle_path,
        relative_path=stored.relative_path,
        sound_id=sound_id,
        asset_id=asset_id,
        device=stored.device,
        inode=stored.inode,
        directory_device=stored.directory_device,
        directory_inode=stored.directory_inode,
    )


__all__ = ["SoundWorkflow", "SoundWorkflowError"]
