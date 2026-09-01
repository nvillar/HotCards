"""Generated-sound workflow tests using a fake adapter."""

from __future__ import annotations

import os
import wave
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from hotcards.application.commands import UpdateSoundCommand
from hotcards.application.document_controller import DocumentController
from hotcards.application.document_session import DocumentSession
from hotcards.application.sound_workflow import SoundWorkflow
from hotcards.application.workers import AdapterWorkers
from hotcards.domain.models import SoundDefinition, Stack
from hotcards.generation.stable_audio import StableAudioRequest, StableAudioResult


@pytest.fixture(scope="session", autouse=True)
def qt_application() -> Iterator[QApplication]:
    application = QApplication.instance() or QApplication([])
    yield application


class FakeStableAudioGenerator:
    def __init__(self, *, gate: Event | None = None) -> None:
        self.gate = gate

    def generate(
        self,
        request: StableAudioRequest,
        output_path: Path,
        *,
        cancellation: object,
        on_sampling_step: object,
    ) -> StableAudioResult:
        if self.gate is not None:
            assert self.gate.wait(timeout=2)
        cancellation.raise_if_requested()  # type: ignore[attr-defined]
        on_sampling_step(8, 8)  # type: ignore[operator]
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
    return sound, controller, session, workers, workflow


def test_generation_persists_exact_provenance_and_is_undoable(tmp_path: Path) -> None:
    sound, controller, session, workers, workflow = sound_session(
        tmp_path,
        FakeStableAudioGenerator(),
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
    assert controller.undo()
    assert session.flush()
    assert asset_path.is_file()
    controller.clear_history()
    assert session.flush()
    assert not asset_path.exists()
    workers.shutdown(wait_milliseconds=500)


def test_changed_sound_suppresses_stale_generation_result(tmp_path: Path) -> None:
    gate = Event()
    sound, controller, _session, workers, workflow = sound_session(
        tmp_path,
        FakeStableAudioGenerator(gate=gate),
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
