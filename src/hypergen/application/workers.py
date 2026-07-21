"""Bounded Qt worker infrastructure for synchronous generation adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from math import ceil, isfinite
from queue import Empty, Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from hypergen.generation.errors import (
    ImageGenerationError,
    ModelLoadError,
    ModelResponseError,
    ModelUnavailableError,
    ServiceUnavailableError,
)


class AdapterKind(StrEnum):
    """Production adapter families supported by the worker boundary."""

    OLLAMA = "ollama"
    MFLUX = "mflux"


class WorkerFailureKind(StrEnum):
    """Stable failure categories suitable for UI decisions."""

    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_RESPONSE = "model_response"
    MODEL_LOAD = "model_load"
    IMAGE_GENERATION = "image_generation"
    ADAPTER_ERROR = "adapter_error"


class OperationStatus(StrEnum):
    """Lifecycle state for a submitted adapter operation."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Typed, actionable failure delivered instead of an adapter exception."""

    adapter: AdapterKind
    stage: str
    kind: WorkerFailureKind
    message: str
    exception_type: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AvailabilityDiagnostic:
    """An adapter availability transition; repeated identical states are suppressed."""

    adapter: AdapterKind
    available: bool
    message: str
    failure: WorkerFailure | None = None


class WorkerOperation(QObject):
    """Qt-facing handle for one bounded operation."""

    succeeded = Signal(object)
    failed = Signal(object)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        operation_id: UUID,
        cancel_event: Event,
        start_lock: Lock,
    ) -> None:
        super().__init__()
        self._operation_id = operation_id
        self._cancel_event = cancel_event
        self._start_lock = start_lock
        self._lock = Lock()
        self._status = OperationStatus.PENDING
        self._result: Any = None
        self._failure: WorkerFailure | None = None

    @property
    def operation_id(self) -> UUID:
        return self._operation_id

    @property
    def status(self) -> OperationStatus:
        with self._lock:
            return self._status

    @property
    def is_finished(self) -> bool:
        return self.status is not OperationStatus.PENDING

    @property
    def result(self) -> Any:
        with self._lock:
            return self._result

    @property
    def failure(self) -> WorkerFailure | None:
        with self._lock:
            return self._failure

    @Slot()
    def cancel(self) -> None:
        """Discard this operation and suppress any eventual adapter outcome."""
        with self._start_lock:
            with self._lock:
                if self._status is not OperationStatus.PENDING:
                    return
                self._status = OperationStatus.CANCELLED
                self._cancel_event.set()
        self.cancelled.emit()
        self.finished.emit()

    def _succeed(self, result: Any) -> None:
        with self._lock:
            if self._status is not OperationStatus.PENDING:
                return
            self._status = OperationStatus.SUCCEEDED
            self._result = result
        self.succeeded.emit(result)
        self.finished.emit()

    def _fail(self, failure: WorkerFailure) -> None:
        with self._lock:
            if self._status is not OperationStatus.PENDING:
                return
            self._status = OperationStatus.FAILED
            self._failure = failure
        self.failed.emit(failure)
        self.finished.emit()


@dataclass(frozen=True, slots=True)
class _Success:
    value: Any


@dataclass(frozen=True, slots=True)
class _Error:
    error: Exception


@dataclass(frozen=True, slots=True)
class _Timeout:
    pass


@dataclass(slots=True)
class _OperationRecord:
    handle: WorkerOperation
    adapter: AdapterKind
    stage: str
    timeout_seconds: float
    availability_check: bool
    cancel_event: Event
    start_lock: Lock
    deadline: float


class _CompletionDispatcher(QObject):
    completed = Signal(object, object)


class _BoundedRunnable(QRunnable):
    def __init__(
        self,
        *,
        operation_id: UUID,
        operation: Callable[[], Any],
        deadline: float,
        cancel_event: Event,
        start_lock: Lock,
        invocation_slots: BoundedSemaphore,
        dispatcher: _CompletionDispatcher,
    ) -> None:
        super().__init__()
        self._operation_id = operation_id
        self._operation = operation
        self._deadline = deadline
        self._cancel_event = cancel_event
        self._start_lock = start_lock
        self._invocation_slots = invocation_slots
        self._dispatcher = dispatcher

    def run(self) -> None:
        if self._cancel_event.is_set():
            self._dispatcher.completed.emit(self._operation_id, _Timeout())
            return
        remaining = self._deadline - monotonic()
        if remaining <= 0:
            self._dispatcher.completed.emit(self._operation_id, _Timeout())
            return

        while not self._cancel_event.is_set():
            remaining = self._deadline - monotonic()
            if remaining <= 0:
                self._dispatcher.completed.emit(self._operation_id, _Timeout())
                return
            if self._invocation_slots.acquire(timeout=min(remaining, 0.05)):
                break
        else:
            self._dispatcher.completed.emit(self._operation_id, _Timeout())
            return

        outcomes: Queue[_Success | _Error] = Queue(maxsize=1)

        def invoke() -> None:
            try:
                try:
                    outcome: _Success | _Error = _Success(self._operation())
                except Exception as error:
                    outcome = _Error(error)
                outcomes.put(outcome)
            finally:
                self._invocation_slots.release()

        with self._start_lock:
            if self._cancel_event.is_set() or self._deadline <= monotonic():
                self._invocation_slots.release()
                self._dispatcher.completed.emit(self._operation_id, _Timeout())
                return
            try:
                Thread(
                    target=invoke,
                    name=f"hypergen-adapter-{self._operation_id}",
                    daemon=True,
                ).start()
            except Exception as error:
                self._invocation_slots.release()
                self._dispatcher.completed.emit(self._operation_id, _Error(error))
                return
        while True:
            remaining = self._deadline - monotonic()
            if remaining <= 0 or self._cancel_event.is_set():
                outcome: _Success | _Error | _Timeout = _Timeout()
                break
            try:
                outcome = outcomes.get(timeout=min(remaining, 0.05))
                break
            except Empty:
                continue
        self._dispatcher.completed.emit(self._operation_id, outcome)


class AdapterWorkers(QObject):
    """Run bounded Ollama and serialized MFLUX operations away from the UI thread."""

    availability_changed = Signal(object)

    def __init__(
        self,
        *,
        ollama_timeout_seconds: float = 300.0,
        mflux_timeout_seconds: float = 600.0,
        ollama_max_concurrency: int = 4,
    ) -> None:
        super().__init__()
        self._timeouts = {
            AdapterKind.OLLAMA: _validate_timeout(ollama_timeout_seconds),
            AdapterKind.MFLUX: _validate_timeout(mflux_timeout_seconds),
        }
        if (
            not isinstance(ollama_max_concurrency, int)
            or isinstance(ollama_max_concurrency, bool)
            or ollama_max_concurrency <= 0
        ):
            raise ValueError("Ollama worker concurrency must be a positive integer")
        self._ollama_pool = QThreadPool(self)
        self._ollama_pool.setMaxThreadCount(ollama_max_concurrency)
        self._mflux_pool = QThreadPool(self)
        self._mflux_pool.setMaxThreadCount(1)
        self._invocation_slots = {
            AdapterKind.OLLAMA: BoundedSemaphore(ollama_max_concurrency),
            AdapterKind.MFLUX: BoundedSemaphore(1),
        }
        self._dispatcher = _CompletionDispatcher(self)
        self._dispatcher.completed.connect(self._complete)
        self._deadline_timer = QTimer(self)
        self._deadline_timer.setSingleShot(True)
        self._deadline_timer.timeout.connect(self._expire_operations)
        self._state_lock = Lock()
        self._closed = False
        self._records: dict[UUID, _OperationRecord] = {}
        self._availability: dict[AdapterKind, bool | None] = {
            AdapterKind.OLLAMA: None,
            AdapterKind.MFLUX: None,
        }

    def run_ollama(
        self,
        operation: Callable[[], Any],
        *,
        stage: str,
        timeout_seconds: float | None = None,
    ) -> WorkerOperation:
        """Submit any synchronous Ollama adapter operation."""
        return self._submit(
            AdapterKind.OLLAMA,
            operation,
            stage=stage,
            timeout_seconds=timeout_seconds,
            availability_check=False,
        )

    def run_mflux(
        self,
        operation: Callable[[], Any],
        *,
        stage: str,
        timeout_seconds: float | None = None,
    ) -> WorkerOperation:
        """Submit a synchronous MFLUX operation to the serialized queue."""
        return self._submit(
            AdapterKind.MFLUX,
            operation,
            stage=stage,
            timeout_seconds=timeout_seconds,
            availability_check=False,
        )

    def check_ollama(
        self,
        check: Callable[[], Any],
        *,
        stage: str = "checking Ollama model availability",
        timeout_seconds: float | None = None,
    ) -> WorkerOperation:
        """Run a bounded list/show-style Ollama availability check."""
        return self._submit(
            AdapterKind.OLLAMA,
            check,
            stage=stage,
            timeout_seconds=timeout_seconds,
            availability_check=True,
        )

    def check_mflux(
        self,
        check: Callable[[], Any],
        *,
        stage: str = "checking MFLUX model availability",
        timeout_seconds: float | None = None,
    ) -> WorkerOperation:
        """Run a bounded MFLUX model availability check."""
        return self._submit(
            AdapterKind.MFLUX,
            check,
            stage=stage,
            timeout_seconds=timeout_seconds,
            availability_check=True,
        )

    def shutdown(self, *, wait_milliseconds: int = 0) -> None:
        """Discard queued work and optionally wait for bounded supervisors."""
        with self._state_lock:
            self._closed = True
            records = tuple(self._records.values())
        for record in records:
            record.handle.cancel()
        self._ollama_pool.clear()
        self._mflux_pool.clear()
        self._records.clear()
        self._deadline_timer.stop()
        if wait_milliseconds > 0:
            self._ollama_pool.waitForDone(wait_milliseconds)
            self._mflux_pool.waitForDone(wait_milliseconds)

    def _submit(
        self,
        adapter: AdapterKind,
        operation: Callable[[], Any],
        *,
        stage: str,
        timeout_seconds: float | None,
        availability_check: bool,
    ) -> WorkerOperation:
        if not callable(operation):
            raise TypeError("worker operation must be callable")
        stage = stage.strip()
        if not stage:
            raise ValueError("worker stage must not be empty")
        timeout = (
            self._timeouts[adapter]
            if timeout_seconds is None
            else _validate_timeout(timeout_seconds)
        )
        operation_id = uuid4()
        cancel_event = Event()
        start_lock = Lock()
        handle = WorkerOperation(operation_id, cancel_event, start_lock)
        deadline = monotonic() + timeout
        with self._state_lock:
            if self._closed:
                raise RuntimeError("adapter workers are shut down")
            self._records[operation_id] = _OperationRecord(
                handle=handle,
                adapter=adapter,
                stage=stage,
                timeout_seconds=timeout,
                availability_check=availability_check,
                cancel_event=cancel_event,
                start_lock=start_lock,
                deadline=deadline,
            )
            handle.cancelled.connect(partial(self._discard, operation_id))
            runnable = _BoundedRunnable(
                operation_id=operation_id,
                operation=operation,
                deadline=deadline,
                cancel_event=cancel_event,
                start_lock=start_lock,
                invocation_slots=self._invocation_slots[adapter],
                dispatcher=self._dispatcher,
            )
            pool = self._mflux_pool if adapter is AdapterKind.MFLUX else self._ollama_pool
            pool.start(runnable)
        self._schedule_deadline()
        return handle

    @Slot(object, object)
    def _complete(self, operation_id: UUID, outcome: object) -> None:
        record = self._records.pop(operation_id, None)
        if record is None:
            return
        self._schedule_deadline()
        if record.handle.status is OperationStatus.CANCELLED:
            return
        if isinstance(outcome, _Success):
            record.handle._succeed(outcome.value)
            self._mark_available(record)
            return
        failure = _failure_for(record, outcome)
        record.handle._fail(failure)
        if record.availability_check or failure.kind in {
            WorkerFailureKind.SERVICE_UNAVAILABLE,
            WorkerFailureKind.MODEL_UNAVAILABLE,
            WorkerFailureKind.MODEL_LOAD,
        }:
            self._mark_unavailable(record.adapter, failure)

    def _discard(self, operation_id: UUID) -> None:
        if self._records.pop(operation_id, None) is not None:
            self._schedule_deadline()

    def _expire_operations(self) -> None:
        now = monotonic()
        expired = [
            operation_id for operation_id, record in self._records.items() if record.deadline <= now
        ]
        for operation_id in expired:
            record = self._records.pop(operation_id, None)
            if record is None:
                continue
            with record.start_lock:
                record.cancel_event.set()
            if record.handle.status is OperationStatus.CANCELLED:
                continue
            failure = _failure_for(record, _Timeout())
            record.handle._fail(failure)
            if record.availability_check:
                self._mark_unavailable(record.adapter, failure)
        self._schedule_deadline()

    def _schedule_deadline(self) -> None:
        if not self._records:
            self._deadline_timer.stop()
            return
        delay = min(record.deadline for record in self._records.values()) - monotonic()
        self._deadline_timer.start(max(1, ceil(delay * 1000)))

    def _mark_available(self, record: _OperationRecord) -> None:
        previous = self._availability[record.adapter]
        self._availability[record.adapter] = True
        if previous is False:
            self.availability_changed.emit(
                AvailabilityDiagnostic(
                    adapter=record.adapter,
                    available=True,
                    message=f"{_adapter_name(record.adapter)} is available again.",
                )
            )
        elif previous is None and record.availability_check:
            self.availability_changed.emit(
                AvailabilityDiagnostic(
                    adapter=record.adapter,
                    available=True,
                    message=f"{_adapter_name(record.adapter)} is available.",
                )
            )

    def _mark_unavailable(self, adapter: AdapterKind, failure: WorkerFailure) -> None:
        if self._availability[adapter] is False:
            return
        self._availability[adapter] = False
        self.availability_changed.emit(
            AvailabilityDiagnostic(
                adapter=adapter,
                available=False,
                message=failure.message,
                failure=failure,
            )
        )


def _validate_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("worker timeout must be a built-in number")
    timeout = float(timeout_seconds)
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("worker timeout must be positive and finite")
    return timeout


def _adapter_name(adapter: AdapterKind) -> str:
    return "Ollama" if adapter is AdapterKind.OLLAMA else "MFLUX"


def _failure_for(record: _OperationRecord, outcome: object) -> WorkerFailure:
    adapter_name = _adapter_name(record.adapter)
    if isinstance(outcome, _Timeout):
        message = (
            f"{adapter_name} timed out during {record.stage} after "
            f"{record.timeout_seconds:g} seconds. The late result was discarded; "
            "check model/service responsiveness and retry."
        )
        return WorkerFailure(
            adapter=record.adapter,
            stage=record.stage,
            kind=WorkerFailureKind.TIMEOUT,
            message=message,
            timeout_seconds=record.timeout_seconds,
        )

    assert isinstance(outcome, _Error)
    error = outcome.error
    kinds = (
        (ServiceUnavailableError, WorkerFailureKind.SERVICE_UNAVAILABLE),
        (ModelUnavailableError, WorkerFailureKind.MODEL_UNAVAILABLE),
        (ModelResponseError, WorkerFailureKind.MODEL_RESPONSE),
        (ModelLoadError, WorkerFailureKind.MODEL_LOAD),
        (ImageGenerationError, WorkerFailureKind.IMAGE_GENERATION),
    )
    kind = WorkerFailureKind.ADAPTER_ERROR
    for error_type, candidate in kinds:
        if isinstance(error, error_type):
            kind = candidate
            break
    detail = str(error).strip() or "the adapter returned an unspecified error"
    return WorkerFailure(
        adapter=record.adapter,
        stage=record.stage,
        kind=kind,
        message=f"{adapter_name} failed during {record.stage}: {detail}",
        exception_type=type(error).__name__,
    )


__all__ = [
    "AdapterKind",
    "AdapterWorkers",
    "AvailabilityDiagnostic",
    "OperationStatus",
    "WorkerFailure",
    "WorkerFailureKind",
    "WorkerOperation",
]
