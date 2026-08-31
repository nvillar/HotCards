"""Qt event-loop-safe tests for serialized MFLUX workers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from threading import Event, Lock, get_ident
from time import sleep

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from hotcards.application.workers import (
    AdapterKind,
    AdapterWorkers,
    OperationStatus,
    WorkerFailureKind,
)
from hotcards.generation.errors import ModelUnavailableError


@pytest.fixture(scope="session", autouse=True)
def qt_application() -> Iterator[QApplication]:
    application = QApplication.instance() or QApplication([])
    yield application


def wait_for(operation: object, *, timeout_seconds: float = 1.0) -> None:
    if operation.is_finished:  # type: ignore[attr-defined]
        return
    loop = QEventLoop()
    operation.finished.connect(loop.quit)  # type: ignore[attr-defined]
    QTimer.singleShot(round(timeout_seconds * 1000), loop.quit)
    loop.exec()
    assert operation.is_finished  # type: ignore[attr-defined]


def spin_event_loop(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def test_mflux_success_runs_off_the_main_thread() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    main_thread = get_ident()
    operation = workers.run_mflux(
        lambda: (get_ident(), "result"),
        stage="generating image",
    )

    wait_for(operation)

    assert operation.status is OperationStatus.SUCCEEDED
    assert operation.result[0] != main_thread
    workers.shutdown(wait_milliseconds=500)


def test_cancellation_discards_a_late_success() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    entered = Event()
    release = Event()

    def delayed() -> str:
        entered.set()
        release.wait()
        return "must be discarded"

    operation = workers.run_mflux(delayed, stage="generating image")
    results: list[object] = []
    failures: list[object] = []
    operation.succeeded.connect(results.append)
    operation.failed.connect(failures.append)
    assert entered.wait(0.5)

    operation.cancel()
    release.set()
    spin_event_loop(50)

    assert operation.status is OperationStatus.CANCELLED
    assert results == []
    assert failures == []
    workers.shutdown(wait_milliseconds=500)


def test_mflux_operations_are_serialized() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    lock = Lock()
    active = 0
    maximum_active = 0
    order: list[str] = []

    def generate(name: str) -> str:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            order.append(f"{name}:start")
        sleep(0.04)
        with lock:
            order.append(f"{name}:end")
            active -= 1
        return name

    first = workers.run_mflux(lambda: generate("first"), stage="generating first image")
    second = workers.run_mflux(lambda: generate("second"), stage="generating second image")
    wait_for(first)
    wait_for(second)

    assert maximum_active == 1
    assert order == ["first:start", "first:end", "second:start", "second:end"]
    workers.shutdown(wait_milliseconds=500)


def test_mflux_stays_serialized_after_ui_timeout() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.05)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def stalled_first() -> str:
        first_entered.set()
        release_first.wait()
        return "discarded"

    first = workers.run_mflux(stalled_first, stage="stalled image")
    assert first_entered.wait(0.5)
    wait_for(first)
    assert first.failure is not None
    assert first.failure.kind is WorkerFailureKind.TIMEOUT

    second = workers.run_mflux(
        lambda: second_entered.set() or "second",
        stage="next image",
        timeout_seconds=0.5,
    )
    spin_event_loop(75)
    assert not second_entered.is_set()

    release_first.set()
    wait_for(second)
    assert second.result == "second"
    workers.shutdown(wait_milliseconds=500)


def test_shutdown_rejects_reentrant_and_later_submissions() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    release = Event()
    operation = workers.run_mflux(lambda: release.wait(), stage="active image")
    rejected: list[str] = []

    def submit_during_shutdown() -> None:
        try:
            workers.run_mflux(lambda: None, stage="reentrant image")
        except RuntimeError as error:
            rejected.append(str(error))

    operation.cancelled.connect(submit_during_shutdown)
    workers.shutdown(wait_milliseconds=100)
    release.set()

    assert rejected == ["adapter workers are shut down"]
    with pytest.raises(RuntimeError, match="shut down"):
        workers.run_mflux(lambda: None, stage="late image")


def test_availability_diagnostics_are_deduplicated_and_report_recovery() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    diagnostics: list[object] = []
    workers.availability_changed.connect(diagnostics.append)

    def unavailable() -> None:
        raise ModelUnavailableError("MFLUX model is not cached.")

    first = workers.check_mflux(unavailable)
    wait_for(first)
    repeated = workers.check_mflux(unavailable)
    wait_for(repeated)
    recovered = workers.check_mflux(lambda: None)
    wait_for(recovered)
    unavailable_again = workers.check_mflux(unavailable)
    wait_for(unavailable_again)

    assert [diagnostic.available for diagnostic in diagnostics] == [False, True, False]
    assert all(diagnostic.adapter is AdapterKind.MFLUX for diagnostic in diagnostics)
    assert "not cached" in diagnostics[0].message
    assert "available again" in diagnostics[1].message
    workers.shutdown(wait_milliseconds=500)


def test_availability_check_can_skip_global_diagnostic() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    diagnostics: list[object] = []
    workers.availability_changed.connect(diagnostics.append)

    operation = workers.check_mflux(lambda: None, emit_diagnostic=False)
    wait_for(operation)

    assert operation.status is OperationStatus.SUCCEEDED
    assert diagnostics == []
    workers.shutdown(wait_milliseconds=500)


def test_mflux_unexpected_error_is_typed_and_does_not_disable_follow_up() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)

    def fail() -> None:
        raise LookupError("fake adapter bug")

    failed = workers.run_mflux(fail, stage="loading image generator")
    wait_for(failed)
    follow_up = workers.run_mflux(lambda: "image", stage="generating image")
    wait_for(follow_up)

    assert failed.failure is not None
    assert failed.failure.kind is WorkerFailureKind.ADAPTER_ERROR
    assert failed.failure.exception_type == "LookupError"
    assert follow_up.result == "image"
    workers.shutdown(wait_milliseconds=500)
