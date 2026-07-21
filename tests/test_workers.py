"""Qt event-loop-safe tests for bounded generation workers."""

from __future__ import annotations

from collections.abc import Iterator
from threading import Event, Lock, get_ident
from time import monotonic, sleep

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from hypergen.application.workers import (
    AdapterKind,
    AdapterWorkers,
    OperationStatus,
    WorkerFailure,
    WorkerFailureKind,
)
from hypergen.generation.errors import ModelResponseError, ServiceUnavailableError


@pytest.fixture(scope="session", autouse=True)
def qt_application() -> Iterator[QCoreApplication]:
    application = QCoreApplication.instance() or QCoreApplication([])
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


def test_ollama_success_runs_off_the_main_thread() -> None:
    workers = AdapterWorkers(ollama_timeout_seconds=0.5)
    main_thread = get_ident()
    operation = workers.run_ollama(
        lambda: (get_ident(), "result"),
        stage="generating hotspots",
    )
    delivered: list[tuple[int, str]] = []
    operation.succeeded.connect(delivered.append)

    wait_for(operation)

    assert operation.status is OperationStatus.SUCCEEDED
    assert operation.result[0] != main_thread
    assert delivered == [operation.result]
    workers.shutdown(wait_milliseconds=500)


def test_typed_adapter_failure_preserves_stage_context() -> None:
    workers = AdapterWorkers(ollama_timeout_seconds=0.5)

    def fail() -> None:
        raise ModelResponseError("invalid structured response")

    operation = workers.run_ollama(fail, stage="validating hotspot response")
    delivered: list[WorkerFailure] = []
    operation.failed.connect(delivered.append)

    wait_for(operation)

    assert operation.status is OperationStatus.FAILED
    assert operation.failure == delivered[0]
    assert delivered[0].kind is WorkerFailureKind.MODEL_RESPONSE
    assert delivered[0].stage == "validating hotspot response"
    assert "validating hotspot response" in delivered[0].message
    workers.shutdown(wait_milliseconds=500)


def test_cancellation_discards_a_late_success() -> None:
    workers = AdapterWorkers(ollama_timeout_seconds=0.5)
    entered = Event()
    release = Event()

    def delayed() -> str:
        entered.set()
        release.wait()
        return "must be discarded"

    operation = workers.run_ollama(delayed, stage="generating hotspots")
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


def test_timeout_releases_capacity_while_stalled_daemon_is_ignored() -> None:
    workers = AdapterWorkers(
        ollama_timeout_seconds=0.05,
        ollama_max_concurrency=1,
    )
    stalled = Event()
    operation = workers.run_ollama(
        lambda: stalled.wait(),
        stage="contacting Ollama",
    )
    started = monotonic()

    wait_for(operation, timeout_seconds=0.5)

    assert monotonic() - started < 0.3
    assert operation.failure is not None
    assert operation.failure.kind is WorkerFailureKind.TIMEOUT
    assert operation.failure.timeout_seconds == 0.05
    assert "contacting Ollama" in operation.failure.message

    retry_entered = Event()
    bounded_retry = workers.run_ollama(
        lambda: retry_entered.set(),
        stage="retrying stalled Ollama",
        timeout_seconds=0.05,
    )
    wait_for(bounded_retry)
    assert bounded_retry.failure is not None
    assert bounded_retry.failure.kind is WorkerFailureKind.TIMEOUT
    assert not retry_entered.is_set()

    stalled.set()
    follow_up = workers.run_ollama(
        lambda: "worker remains usable",
        stage="retrying recovered Ollama",
        timeout_seconds=0.5,
    )
    wait_for(follow_up)
    assert follow_up.result == "worker remains usable"
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


def test_shutdown_prevents_queued_adapter_call_from_starting() -> None:
    workers = AdapterWorkers(
        ollama_timeout_seconds=0.05,
        ollama_max_concurrency=1,
    )
    release_first = Event()
    first = workers.run_ollama(
        lambda: release_first.wait(),
        stage="stalled call",
    )
    wait_for(first)

    queued_entered = Event()
    workers.run_ollama(
        lambda: queued_entered.set(),
        stage="queued call",
        timeout_seconds=0.5,
    )
    spin_event_loop(50)
    workers.shutdown(wait_milliseconds=100)
    release_first.set()
    spin_event_loop(100)

    assert not queued_entered.is_set()


def test_shutdown_rejects_reentrant_and_later_submissions() -> None:
    workers = AdapterWorkers(ollama_timeout_seconds=0.5)
    release = Event()
    operation = workers.run_ollama(
        lambda: release.wait(),
        stage="active call",
    )
    rejected: list[str] = []

    def submit_during_shutdown() -> None:
        try:
            workers.run_ollama(lambda: None, stage="reentrant call")
        except RuntimeError as error:
            rejected.append(str(error))

    operation.cancelled.connect(submit_during_shutdown)
    workers.shutdown(wait_milliseconds=100)
    release.set()

    assert rejected == ["adapter workers are shut down"]
    with pytest.raises(RuntimeError, match="shut down"):
        workers.run_ollama(lambda: None, stage="late call")


def test_availability_diagnostics_are_deduplicated_and_report_recovery() -> None:
    workers = AdapterWorkers(ollama_timeout_seconds=0.5)
    diagnostics: list[object] = []
    workers.availability_changed.connect(diagnostics.append)

    def unavailable() -> None:
        raise ServiceUnavailableError("Start Ollama and verify its endpoint.")

    first = workers.check_ollama(unavailable)
    wait_for(first)
    repeated = workers.check_ollama(unavailable)
    wait_for(repeated)
    recovered = workers.check_ollama(lambda: None)
    wait_for(recovered)
    unavailable_again = workers.check_ollama(unavailable)
    wait_for(unavailable_again)

    assert [diagnostic.available for diagnostic in diagnostics] == [False, True, False]
    assert all(diagnostic.adapter is AdapterKind.OLLAMA for diagnostic in diagnostics)
    assert "Start Ollama" in diagnostics[0].message
    assert "available again" in diagnostics[1].message
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
