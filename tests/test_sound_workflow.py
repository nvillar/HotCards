"""Generated-sound workflow tests using a fake adapter."""

from __future__ import annotations

import os
import wave
from collections.abc import Callable
from pathlib import Path
from threading import Event

import pytest
from PySide6.QtCore import QEventLoop, QTimer

import hotcards.storage.stack_store as storage_module
from hotcards.application.commands import UpdateSoundCommand
from hotcards.application.document_controller import DocumentController
from hotcards.application.document_session import DocumentSession
from hotcards.application.sound_workflow import SoundWorkflow, SoundWorkflowError
from hotcards.application.workers import AdapterWorkers
from hotcards.domain.models import SoundDefinition, Stack
from hotcards.generation.stable_audio import (
    StableAudioCancellation,
    StableAudioRequest,
    StableAudioResult,
)
from hotcards.storage.stack_store import StackStoreTransactionError


class FakeStableAudioGenerator:
    def __init__(self, *, gate: Event | None = None) -> None:
        self.gate = gate
        self.output_path: Path | None = None

    def generate(
        self,
        request: StableAudioRequest,
        output_path: Path,
        *,
        cancellation: StableAudioCancellation,
        on_sampling_step: Callable[[int, int], None],
    ) -> StableAudioResult:
        if self.gate is not None:
            assert self.gate.wait(timeout=2)
        cancellation.raise_if_requested()
        on_sampling_step(8, 8)
        self.output_path = output_path
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(44_100)
            output.writeframes(b"\0" * request.duration_seconds * 44_100 * 4)
        return StableAudioResult(
            output_path=output_path,
            seed=123,
            generation_duration_milliseconds=250,
        )


def wait_for_workflow(workflow: SoundWorkflow) -> None:
    if not workflow.is_active:
        return
    loop = QEventLoop()
    workflow.finished.connect(loop.quit)
    QTimer.singleShot(2_000, loop.quit)
    loop.exec()
    assert not workflow.is_active


def sound_session(
    tmp_path: Path,
    generator: FakeStableAudioGenerator,
    request: pytest.FixtureRequest,
) -> tuple[SoundDefinition, DocumentController, DocumentSession, AdapterWorkers, SoundWorkflow]:
    sound = SoundDefinition(name="Knock", prompt="A dry wooden knock")
    controller = DocumentController(Stack(name="Sounds", sounds=(sound,)))
    session = DocumentSession(controller, debounce_milliseconds=0)
    session.create(controller.document, tmp_path / "Sounds.hotcards")
    workers = AdapterWorkers(stable_audio_timeout_seconds=2)
    workflow = SoundWorkflow(  # type: ignore[arg-type]
        controller,
        session,
        workers,
        generator,
    )

    def cleanup() -> None:
        if generator.gate is not None:
            generator.gate.set()
        workflow.cancel()
        workers.shutdown(wait_milliseconds=2_000)
        session.deleteLater()

    request.addfinalizer(cleanup)
    return sound, controller, session, workers, workflow


def test_generation_persists_exact_provenance_and_is_undoable(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    generator = FakeStableAudioGenerator()
    sound, controller, session, workers, workflow = sound_session(
        tmp_path,
        generator,
        request,
    )
    changes: list[str] = []
    workflow.change_applied.connect(lambda message, _token: changes.append(message))

    workflow.generate(sound.id)
    wait_for_workflow(workflow)

    generated = controller.document.sound_by_id(sound.id).generated
    assert generated is not None
    assert generated.provenance.prompt == sound.prompt
    assert generated.provenance.seed == 123
    assert generated.provenance.steps == 8
    assert session.store is not None
    asset_path = session.store.asset_path(generated.audio_path)
    assert asset_path.is_file()
    assert changes == ["Sound generated"]
    assert generator.output_path is not None
    assert not generator.output_path.exists()
    assert controller.undo()
    assert session.flush()
    assert asset_path.is_file()
    controller.clear_history()
    assert session.flush()
    assert not asset_path.exists()
    workers.shutdown(wait_milliseconds=500)


def test_changed_sound_suppresses_stale_generation_result(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    gate = Event()
    sound, controller, _session, workers, workflow = sound_session(
        tmp_path,
        FakeStableAudioGenerator(gate=gate),
        request,
    )
    failures: list[str] = []
    workflow.failed.connect(failures.append)

    workflow.generate(sound.id)
    controller.execute(
        UpdateSoundCommand(
            sound_id=sound.id,
            name=sound.name,
            prompt="A metal knock",
            duration_seconds=2,
        )
    )
    gate.set()
    wait_for_workflow(workflow)

    assert controller.document.sound_by_id(sound.id).generated is None
    assert failures == ["the Sound or stack changed before generation completed"]
    workers.shutdown(wait_milliseconds=500)


@pytest.mark.parametrize("outcome", ["committed_warning", "pending", "rollback"])
def test_sound_completion_keeps_exact_token_and_resolves_pending_once(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    generator = FakeStableAudioGenerator()
    sound, controller, session, _workers, workflow = sound_session(tmp_path, generator, request)
    assert session.store is not None
    store = session.store
    real_commit = store.store_sound_asset_and_save
    before = controller.document
    changes: list[object] = []
    failures: list[str] = []
    workflow.change_applied.connect(lambda _message, token: changes.append(token))
    workflow.failed.connect(failures.append)

    def commit_with_error(*args, **kwargs):
        asset = real_commit(*args, **kwargs)
        changed = kwargs["changed_stack"]
        if outcome == "rollback":
            store.save(before)
        raise StackStoreTransactionError(
            OSError("injected transaction outcome"),
            persisted_stack=(
                None if outcome == "pending" else (before if outcome == "rollback" else changed)
            ),
            observed_stack=before if outcome == "rollback" else changed,
            durability_indeterminate=outcome == "pending",
            owned_asset=asset,
        )

    monkeypatch.setattr(store, "store_sound_asset_and_save", commit_with_error)
    captured: list[object] = []

    def rename_after_history(_state: object) -> None:
        if controller.can_undo and not captured:
            captured.append(controller.current_undo_token)
            controller.execute(
                UpdateSoundCommand(
                    sound_id=sound.id,
                    name="New name",
                    prompt=sound.prompt,
                    duration_seconds=sound.duration_seconds,
                )
            )

    session.state_changed.connect(rename_after_history)
    workflow.generate(sound.id)
    wait_for_workflow(workflow)
    assert len(failures) == 1
    assert generator.output_path is not None and not generator.output_path.exists()
    if outcome == "pending":
        assert controller.mutation_blocked
        assert session.state.dirty
        assert changes == []
        with pytest.raises(SoundWorkflowError, match="pending asset"):
            workflow.generate(sound.id)
        assert session.flush() is False  # Reentrant rename schedules a newer save.
        assert session.flush()
    if outcome == "rollback":
        assert controller.document == before
        assert changes == []
        assert not list(store.bundle_path.rglob("*.wav"))
    else:
        assert changes == captured
        assert len(changes) == 1
        assert changes[0] != controller.current_undo_token
        assert controller.document.sound_by_id(sound.id).name == "New name"
        assert session.flush()
        assert changes == captured


def test_sound_copy_error_finishes_and_cleans_temporary_output(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = FakeStableAudioGenerator()
    sound, controller, session, _workers, workflow = sound_session(tmp_path, generator, request)
    failures: list[str] = []
    workflow.failed.connect(failures.append)
    before = controller.document

    def fail_copy(*_args: object) -> None:
        raise OSError("copy interrupted")

    monkeypatch.setattr(storage_module.shutil, "copyfileobj", fail_copy)
    workflow.generate(sound.id)
    wait_for_workflow(workflow)
    assert len(failures) == 1 and "copy interrupted" in failures[0]
    assert controller.document == before
    assert not controller.can_undo
    assert generator.output_path is not None and not generator.output_path.exists()
    assert session.store is not None
    assert not list(session.store.bundle_path.rglob("*.wav"))


def test_failed_sound_manifest_and_rollback_complete_only_after_durable_retry(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = FakeStableAudioGenerator()
    sound, controller, session, _workers, workflow = sound_session(tmp_path, generator, request)
    assert session.store is not None
    store = session.store
    before = controller.document
    directory = store.bundle_path.stat()
    replaced = False
    changes: list[object] = []
    failures: list[str] = []
    workflow.change_applied.connect(lambda _message, token: changes.append(token))
    workflow.failed.connect(failures.append)
    real_fsync, real_write = storage_module.os.fsync, storage_module._write_manifest_at

    def checkpoint(name: str) -> None:
        nonlocal replaced
        if name == "manifest-replaced":
            replaced = True

    def fail_manifest_fsync(descriptor: int) -> None:
        current = os.fstat(descriptor)
        if replaced and (current.st_dev, current.st_ino) == (directory.st_dev, directory.st_ino):
            raise OSError("manifest directory unavailable")
        real_fsync(descriptor)

    def fail_rollback(descriptor: int, payload: bytes, **kwargs) -> str:
        if not kwargs["checkpoints"]:
            raise OSError("rollback unavailable")
        return real_write(descriptor, payload, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(storage_module, "_io_checkpoint", checkpoint)
        fault.setattr(storage_module, "_write_manifest_at", fail_rollback)
        fault.setattr(storage_module.os, "fsync", fail_manifest_fsync)
        workflow.generate(sound.id)
        wait_for_workflow(workflow)

    assert controller.document != before
    assert store.load() == controller.document
    assert controller.mutation_blocked and session.state.dirty
    assert controller.current_undo_token is None
    assert len(failures) == 1 and "durability remains indeterminate" in failures[0]
    assert changes == []
    assert len(list(store.bundle_path.rglob("*.wav"))) == 1
    assert generator.output_path is not None and not generator.output_path.exists()
    assert session.flush()
    assert not controller.mutation_blocked
    assert changes == [controller.current_undo_token]
    assert controller.undo()
    assert controller.document == before
    assert session.flush()
    controller.clear_history()
    assert session.flush()
    assert not list(store.bundle_path.rglob("*.wav"))
