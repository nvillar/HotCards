"""Qt event-loop-safe tests for serialized MFLUX workers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Lock, current_thread, get_ident
from time import monotonic, sleep
from types import SimpleNamespace
from weakref import ref

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

import hotcards.generation.mflux_generator as mflux_module
from hotcards.application.workers import (
    AdapterKind,
    AdapterWorkers,
    OperationStatus,
    WorkerFailureKind,
)
from hotcards.domain.models import GenerateInputs
from hotcards.generation.errors import (
    ImageGenerationCancelled,
    ModelUnavailableError,
)
from hotcards.generation.mflux_generator import (
    MfluxCancellationToken,
    MfluxGenerateRequest,
    MfluxGenerator,
    dispose_mflux_result,
)


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


def test_mflux_operations_share_one_stable_invocation_thread() -> None:
    first_workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    first = first_workers.run_mflux(
        current_thread,
        stage="generating first image",
    )
    wait_for(first)
    first_workers.shutdown(wait_milliseconds=500)

    second_workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    second = second_workers.run_mflux(
        current_thread,
        stage="generating second image",
    )
    wait_for(second)

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert second.result is first.result
    second_workers.shutdown(wait_milliseconds=500)


def test_cached_mflux_model_is_reused_on_its_creation_thread(
    tmp_path: Path,
) -> None:
    class GeneratedImage:
        def save(self, path: Path, *, overwrite: bool) -> None:
            assert not overwrite
            Image.new("RGB", (512, 384), "navy").save(path, format="PNG")

    class ThreadAffineModel:
        def __init__(self) -> None:
            self.owner = current_thread()

        def generate_image(self, **_kwargs: object) -> GeneratedImage:
            if current_thread() is not self.owner:
                raise RuntimeError("cached model used from a different thread")
            return GeneratedImage()

    models: list[ThreadAffineModel] = []

    def model_factory(*_args: object) -> ThreadAffineModel:
        model = ThreadAffineModel()
        models.append(model)
        return model

    generator = MfluxGenerator(model_factory=model_factory)  # type: ignore[arg-type]
    workers = [
        AdapterWorkers(mflux_timeout_seconds=0.5),
        AdapterWorkers(mflux_timeout_seconds=0.5),
    ]
    operations = []
    for name, worker in zip(("first", "second"), workers, strict=True):
        request = MfluxGenerateRequest(
            output_path=tmp_path / f"{name}.png",
            inputs=GenerateInputs(description=f"{name} image"),
            render_prompt=f"{name} image",
            seed=42,
        )
        operation = worker.run_mflux(
            lambda request=request: generator.generate(request),
            stage=f"generating {name} image",
            dispose_result=dispose_mflux_result,
        )
        wait_for(operation)
        operations.append(operation)

    assert [operation.status for operation in operations] == [
        OperationStatus.SUCCEEDED,
        OperationStatus.SUCCEEDED,
    ]
    assert len(models) == 1
    generator.release()
    for worker in workers:
        worker.shutdown(wait_milliseconds=500)


def test_invocation_callback_failure_does_not_break_later_work(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)

    def fail_completion() -> None:
        raise RuntimeError("injected completion failure")

    first = workers.run_mflux(
        lambda: "first",
        stage="generating first image",
        invocation_finished=fail_completion,
    )
    wait_for(first)
    second = workers.run_mflux(
        lambda: "second",
        stage="generating second image",
    )
    wait_for(second)

    assert first.result == "first"
    assert second.result == "second"
    assert "MFLUX invocation lifecycle cleanup failed" in caplog.text
    workers.shutdown(wait_milliseconds=500)


def test_startup_cleanup_failure_releases_slot_and_reports_original_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)

    def fail_start() -> None:
        raise ValueError("injected startup failure")

    def fail_completion() -> None:
        raise RuntimeError("injected completion failure")

    failed = workers.run_mflux(
        lambda: "unreachable",
        stage="starting first image",
        invocation_started=fail_start,
        invocation_finished=fail_completion,
    )
    wait_for(failed)
    follow_up = workers.run_mflux(
        lambda: "second",
        stage="generating second image",
    )
    wait_for(follow_up)

    assert failed.status is OperationStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.exception_type == "ValueError"
    assert "injected startup failure" in failed.failure.message
    assert follow_up.result == "second"
    assert "MFLUX invocation startup cleanup failed" in caplog.text
    workers.shutdown(wait_milliseconds=500)


def test_disposer_failure_does_not_abort_cancellation_or_shutdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    returned = Event()
    release_delivery = Event()

    def return_then_pause() -> object:
        returned.set()
        release_delivery.wait()
        return object()

    disposal_attempts: list[None] = []

    def fail_disposal(_result: object) -> None:
        disposal_attempts.append(None)
        raise OSError("injected disposal failure")

    operation = workers.run_mflux(
        return_then_pause,
        stage="generating disposable image",
        dispose_result=fail_disposal,
    )
    assert returned.wait(0.5)
    operation.cancel()
    release_delivery.set()
    spin_event_loop(50)
    workers.shutdown(wait_milliseconds=500)

    assert operation.status is OperationStatus.CANCELLED
    assert disposal_attempts
    assert "Could not dispose a discarded adapter result" in caplog.text


def test_cancellation_discards_a_late_success() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    entered = Event()
    release = Event()
    cancel_requests: list[None] = []
    disposed: list[object] = []
    result = object()

    def delayed() -> object:
        entered.set()
        release.wait()
        return result

    operation = workers.run_mflux(
        delayed,
        stage="generating image",
        request_cancel=lambda: cancel_requests.append(None),
        dispose_result=disposed.append,
    )
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
    assert cancel_requests == [None]
    assert disposed == [result]
    workers.shutdown(wait_milliseconds=500)


def test_cancellation_before_invocation_requests_cancel_without_disposal() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    first_entered = Event()
    release_first = Event()
    second_called = Event()
    cancel_requests: list[None] = []
    disposed: list[object] = []

    first = workers.run_mflux(
        lambda: first_entered.set() or release_first.wait(),
        stage="blocking image",
    )
    assert first_entered.wait(0.5)
    second = workers.run_mflux(
        lambda: second_called.set() or object(),
        stage="queued image",
        request_cancel=lambda: cancel_requests.append(None),
        dispose_result=disposed.append,
    )

    second.cancel()
    release_first.set()
    wait_for(first)
    spin_event_loop(50)

    assert second.status is OperationStatus.CANCELLED
    assert not second_called.is_set()
    assert cancel_requests == [None]
    assert disposed == []
    workers.shutdown(wait_milliseconds=500)


def test_cancel_after_return_before_delivery_disposes_result_once() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    returned = Event()
    result = object()
    disposed: list[object] = []
    operation = workers.run_mflux(
        lambda: returned.set() or result,
        stage="returning image",
        dispose_result=disposed.append,
    )
    assert returned.wait(0.5)

    operation.cancel()
    spin_event_loop(50)

    assert operation.status is OperationStatus.CANCELLED
    assert disposed == [result]
    workers.shutdown(wait_milliseconds=500)


def test_normal_success_transfers_result_without_disposal() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    result = object()
    disposed: list[object] = []
    operation = workers.run_mflux(
        lambda: result,
        stage="successful image",
        dispose_result=disposed.append,
    )

    wait_for(operation)

    assert operation.result is result
    assert disposed == []
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


def test_cross_worker_timeouts_do_not_queue_abandoned_invocations() -> None:
    first_workers = AdapterWorkers(mflux_timeout_seconds=0.05)
    second_workers = AdapterWorkers(mflux_timeout_seconds=0.05)
    first_entered = Event()
    release_first = Event()
    first_finished = Event()
    second_called = Event()

    def stalled_first() -> None:
        first_entered.set()
        release_first.wait()
        first_finished.set()

    first = first_workers.run_mflux(
        stalled_first,
        stage="stalled first image",
    )
    assert first_entered.wait(0.5)
    wait_for(first)
    second = second_workers.run_mflux(
        second_called.set,
        stage="queued second image",
    )
    wait_for(second)

    release_first.set()
    assert first_finished.wait(0.5)
    follow_up = second_workers.run_mflux(
        lambda: "follow-up",
        stage="generating follow-up image",
        timeout_seconds=0.5,
    )
    wait_for(follow_up)

    assert first.status is OperationStatus.FAILED
    assert second.status is OperationStatus.FAILED
    assert not second_called.is_set()
    assert follow_up.result == "follow-up"
    first_workers.shutdown(wait_milliseconds=500)
    second_workers.shutdown(wait_milliseconds=500)


def test_timeout_requests_cancel_and_disposes_late_success_once() -> None:
    workers = AdapterWorkers(mflux_timeout_seconds=0.05)
    entered = Event()
    release = Event()
    result = object()
    cancel_requests: list[None] = []
    disposed: list[object] = []
    failures: list[object] = []

    def delayed_result() -> object:
        entered.set()
        release.wait()
        return result

    operation = workers.run_mflux(
        delayed_result,
        stage="timed image",
        request_cancel=lambda: cancel_requests.append(None),
        dispose_result=disposed.append,
    )
    operation.failed.connect(failures.append)
    assert entered.wait(0.5)

    wait_for(operation)
    release.set()
    spin_event_loop(75)

    assert operation.status is OperationStatus.FAILED
    assert operation.failure is not None
    assert operation.failure.kind is WorkerFailureKind.TIMEOUT
    assert failures == [operation.failure]
    assert cancel_requests == [None]
    assert disposed == [result]
    workers.shutdown(wait_milliseconds=500)


def test_mflux_deadline_cancels_adapter_before_single_failure_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = Event()
    release = Event()
    cancellation_seen = Event()
    unwound = Event()
    cache_released = Event()
    model_ref: ref[object] | None = None

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
                    try:
                        callback.call_in_loop()
                    except ImageGenerationCancelled:
                        cancellation_seen.set()
                        raise
            finally:
                unwound.set()
            raise AssertionError("cancelled MFLUX inference returned")

    def model_factory(*_args: object) -> object:
        nonlocal model_ref
        model = BlockingModel()
        model_ref = ref(model)
        return model

    def release_cache() -> None:
        assert unwound.is_set()
        assert model_ref is not None
        assert model_ref() is None
        cache_released.set()

    monkeypatch.setattr(mflux_module, "_CACHED_MODEL", None)
    mflux_module._DEFERRED_RELEASES.clear()
    monkeypatch.setattr(mflux_module, "_release_model_cache", release_cache)
    token = MfluxCancellationToken()
    generator = MfluxGenerator(model_factory=model_factory)  # type: ignore[arg-type]
    output_path = tmp_path / "deadline.png"
    request = MfluxGenerateRequest(
        output_path=output_path,
        inputs=GenerateInputs(description="A timed image"),
        render_prompt="A timed image",
        seed=42,
    )
    workers = AdapterWorkers(mflux_timeout_seconds=0.05)
    failures: list[object] = []
    operation = workers.run_mflux(
        lambda: generator.generate(request, cancellation=token),
        stage="timed MFLUX image",
        request_cancel=token.cancel,
        dispose_result=dispose_mflux_result,
    )
    operation.failed.connect(failures.append)
    assert entered.wait(0.5)

    wait_for(operation)

    assert token.is_cancelled
    assert operation.failure is not None
    assert operation.failure.kind is WorkerFailureKind.TIMEOUT
    assert failures == [operation.failure]
    release_started = monotonic()
    generator.release()
    assert monotonic() - release_started < 0.25
    release.set()
    assert cancellation_seen.wait(0.5)
    assert cache_released.wait(0.5)
    spin_event_loop(50)

    assert failures == [operation.failure]
    assert not output_path.exists()
    assert list(tmp_path.glob(".hotcards-mflux-*")) == []
    assert mflux_module._CACHED_MODEL is None
    assert mflux_module._DEFERRED_RELEASES == []
    workers.shutdown(wait_milliseconds=500)


def test_cancel_after_mflux_return_disposes_owned_output_before_delivery(
    tmp_path: Path,
) -> None:
    adapter_returned = Event()
    allow_delivery = Event()

    class CallbackRegistry:
        def __init__(self) -> None:
            self.registered: list[object] = []

        def register(self, callback: object) -> None:
            self.registered.append(callback)

    class GeneratedImage:
        def save(self, path: Path, *, overwrite: bool) -> None:
            assert not overwrite
            Image.new("RGB", (512, 384), "navy").save(path, format="PNG")

    class Model:
        def __init__(self) -> None:
            self.callbacks = CallbackRegistry()

        def generate_image(self, **kwargs: object) -> GeneratedImage:
            config = SimpleNamespace(num_inference_steps=kwargs["num_inference_steps"])
            for callback in self.callbacks.registered:
                callback.call_before_loop(config=config)
                callback.call_after_loop()
            return GeneratedImage()

    token = MfluxCancellationToken()
    generator = MfluxGenerator(model_factory=lambda *_: Model())
    output_path = tmp_path / "discarded.png"
    request = MfluxGenerateRequest(
        output_path=output_path,
        inputs=GenerateInputs(description="A discarded image"),
        render_prompt="A discarded image",
        seed=42,
    )

    def generate_then_pause() -> object:
        result = generator.generate(request, cancellation=token)
        adapter_returned.set()
        assert allow_delivery.wait(2)
        return result

    workers = AdapterWorkers(mflux_timeout_seconds=0.5)
    operation = workers.run_mflux(
        generate_then_pause,
        stage="discarded MFLUX image",
        request_cancel=token.cancel,
        dispose_result=dispose_mflux_result,
    )
    assert adapter_returned.wait(0.5)
    assert output_path.is_file()

    operation.cancel()
    allow_delivery.set()
    spin_event_loop(75)

    assert operation.status is OperationStatus.CANCELLED
    assert not output_path.exists()
    assert list(tmp_path.glob(".hotcards-mflux-*")) == []
    generator.release()
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
