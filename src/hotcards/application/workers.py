"""Bounded Qt worker infrastructure for synchronous generation adapters."""

from __future__ import annotations

import logging
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

from hotcards.generation.errors import (
    ImageGenerationError,
    ModelLoadError,
    ModelUnavailableError,
)

logger = logging.getLogger(__name__)


class AdapterKind(StrEnum):
    """Production adapter families supported by the worker boundary."""

    MFLUX = "mflux"


class WorkerFailureKind(StrEnum):
    """Stable failure categories suitable for UI decisions."""

    TIMEOUT = "timeout"
    MODEL_UNAVAILABLE = "model_unavailable"
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
        cancellation: _CancellationControl,
        start_lock: Lock,
    ) -> None:
        super().__init__()
        self._operation_id = operation_id
        self._cancellation = cancellation
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
                self._cancellation.request()
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


class _Success:
    def __init__(
        self,
        value: Any,
        disposer: Callable[[Any], None] | None,
    ) -> None:
        self.value = value
        self._disposer = disposer
        self._lock = Lock()
        self._disposed = False

    def dispose(self) -> None:
        with self._lock:
            if self._disposed:
                return
            try:
                if self._disposer is not None:
                    self._disposer(self.value)
            except Exception:
                logger.exception("Could not dispose a discarded adapter result")
                return
            self._disposed = True


@dataclass(frozen=True, slots=True)
class _Error:
    error: Exception


@dataclass(frozen=True, slots=True)
class _Timeout:
    pass


@dataclass(frozen=True, slots=True)
class _InvocationTask:
    operation: Callable[[], Any]
    dispose_result: Callable[[Any], None] | None
    invocation_finished: Callable[[], None] | None
    cancellation: _CancellationControl
    invocation_slots: BoundedSemaphore
    outcomes: Queue[_Success | _Error]


class _InvocationThread:
    """Keep thread-affine adapter state on one daemon thread."""

    def __init__(self) -> None:
        self._tasks: Queue[_InvocationTask] = Queue()
        self._thread = Thread(
            target=self._run,
            name="hotcards-mflux-invocations",
            daemon=True,
        )
        self._thread.start()

    def submit(self, task: _InvocationTask) -> None:
        self._tasks.put(task)

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                self._invoke(task)
            except Exception:
                logger.exception("MFLUX invocation lifecycle cleanup failed")

    @staticmethod
    def _invoke(task: _InvocationTask) -> None:
        try:
            if task.cancellation.event.is_set():
                return
            try:
                outcome: _Success | _Error = _Success(
                    task.operation(),
                    task.dispose_result,
                )
            except Exception as error:
                outcome = _Error(error)
            if not isinstance(outcome, _Success) or task.cancellation.offer_success(outcome):
                task.outcomes.put(outcome)
        finally:
            try:
                if task.invocation_finished is not None:
                    task.invocation_finished()
            finally:
                task.invocation_slots.release()


_PROCESS_MFLUX_INVOCATIONS = _InvocationThread()
_PROCESS_MFLUX_INVOCATION_SLOT = BoundedSemaphore(1)


class _CancellationControl:
    def __init__(
        self,
        request_adapter_cancel: Callable[[], None] | None,
    ) -> None:
        self.event = Event()
        self._request_adapter_cancel = request_adapter_cancel
        self._lock = Lock()
        self._requested = False
        self._pending_success: _Success | None = None

    def request(self) -> None:
        pending: _Success | None
        request_adapter_cancel: Callable[[], None] | None
        with self._lock:
            if self._requested:
                return
            self._requested = True
            self.event.set()
            pending = self._pending_success
            self._pending_success = None
            request_adapter_cancel = self._request_adapter_cancel
        try:
            if request_adapter_cancel is not None:
                request_adapter_cancel()
        finally:
            if pending is not None:
                pending.dispose()

    def offer_success(self, success: _Success) -> bool:
        with self._lock:
            if self._requested:
                accepted = False
            else:
                self._pending_success = success
                accepted = True
        if not accepted:
            success.dispose()
        return accepted

    def claim_success(self, success: _Success) -> bool:
        with self._lock:
            if self._requested or self._pending_success is not success:
                return False
            self._pending_success = None
            return True


@dataclass(slots=True)
class _OperationRecord:
    handle: WorkerOperation
    adapter: AdapterKind
    stage: str
    timeout_seconds: float
    availability_check: bool
    emit_availability: bool
    cancellation: _CancellationControl
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
        dispose_result: Callable[[Any], None] | None,
        invocation_started: Callable[[], None] | None,
        invocation_finished: Callable[[], None] | None,
        deadline: float,
        cancellation: _CancellationControl,
        start_lock: Lock,
        invocation_slots: BoundedSemaphore,
        invocation_thread: _InvocationThread,
        dispatcher: _CompletionDispatcher,
    ) -> None:
        super().__init__()
        self._operation_id = operation_id
        self._operation = operation
        self._dispose_result = dispose_result
        self._invocation_started = invocation_started
        self._invocation_finished = invocation_finished
        self._deadline = deadline
        self._cancellation = cancellation
        self._start_lock = start_lock
        self._invocation_slots = invocation_slots
        self._invocation_thread = invocation_thread
        self._dispatcher = dispatcher

    def run(self) -> None:
        if self._cancellation.event.is_set():
            self._cancellation.request()
            self._dispatcher.completed.emit(self._operation_id, _Timeout())
            return
        remaining = self._deadline - monotonic()
        if remaining <= 0:
            self._cancellation.request()
            self._dispatcher.completed.emit(self._operation_id, _Timeout())
            return

        while not self._cancellation.event.is_set():
            remaining = self._deadline - monotonic()
            if remaining <= 0:
                self._cancellation.request()
                self._dispatcher.completed.emit(self._operation_id, _Timeout())
                return
            if self._invocation_slots.acquire(timeout=min(remaining, 0.05)):
                break
        else:
            self._dispatcher.completed.emit(self._operation_id, _Timeout())
            return

        outcomes: Queue[_Success | _Error] = Queue(maxsize=1)

        with self._start_lock:
            if self._cancellation.event.is_set() or self._deadline <= monotonic():
                self._cancellation.request()
                self._invocation_slots.release()
                self._dispatcher.completed.emit(self._operation_id, _Timeout())
                return
            try:
                if self._invocation_started is not None:
                    self._invocation_started()
                self._invocation_thread.submit(
                    _InvocationTask(
                        operation=self._operation,
                        dispose_result=self._dispose_result,
                        invocation_finished=self._invocation_finished,
                        cancellation=self._cancellation,
                        invocation_slots=self._invocation_slots,
                        outcomes=outcomes,
                    )
                )
            except Exception as error:
                try:
                    if self._invocation_finished is not None:
                        self._invocation_finished()
                except Exception:
                    logger.exception("MFLUX invocation startup cleanup failed")
                finally:
                    self._invocation_slots.release()
                self._dispatcher.completed.emit(self._operation_id, _Error(error))
                return
        while True:
            remaining = self._deadline - monotonic()
            if remaining <= 0 or self._cancellation.event.is_set():
                self._cancellation.request()
                outcome: _Success | _Error | _Timeout = _Timeout()
                break
            try:
                outcome = outcomes.get(timeout=min(remaining, 0.05))
                if isinstance(outcome, _Success) and not self._cancellation.claim_success(outcome):
                    outcome.dispose()
                    outcome = _Timeout()
                break
            except Empty:
                continue
        self._dispatcher.completed.emit(self._operation_id, outcome)


class AdapterWorkers(QObject):
    """Run serialized MFLUX operations away from the UI thread."""

    availability_changed = Signal(object)

    def __init__(
        self,
        *,
        mflux_timeout_seconds: float = 600.0,
    ) -> None:
        super().__init__()
        self._timeouts = {
            AdapterKind.MFLUX: _validate_timeout(mflux_timeout_seconds),
        }
        self._mflux_pool = QThreadPool(self)
        self._mflux_pool.setMaxThreadCount(1)
        self._invocation_slots = {
            AdapterKind.MFLUX: _PROCESS_MFLUX_INVOCATION_SLOT,
        }
        self._invocation_thread = _PROCESS_MFLUX_INVOCATIONS
        self._dispatcher = _CompletionDispatcher(self)
        self._dispatcher.completed.connect(self._complete)
        self._deadline_timer = QTimer(self)
        self._deadline_timer.setSingleShot(True)
        self._deadline_timer.timeout.connect(self._expire_operations)
        self._state_lock = Lock()
        self._closed = False
        self._records: dict[UUID, _OperationRecord] = {}
        self._availability: dict[AdapterKind, bool | None] = {
            AdapterKind.MFLUX: None,
        }

    def run_mflux(
        self,
        operation: Callable[[], Any],
        *,
        stage: str,
        timeout_seconds: float | None = None,
        request_cancel: Callable[[], None] | None = None,
        dispose_result: Callable[[Any], None] | None = None,
        invocation_started: Callable[[], None] | None = None,
        invocation_finished: Callable[[], None] | None = None,
    ) -> WorkerOperation:
        """Submit a synchronous MFLUX operation to the serialized queue."""
        return self._submit(
            AdapterKind.MFLUX,
            operation,
            stage=stage,
            timeout_seconds=timeout_seconds,
            availability_check=False,
            emit_availability=True,
            request_cancel=request_cancel,
            dispose_result=dispose_result,
            invocation_started=invocation_started,
            invocation_finished=invocation_finished,
        )

    def check_mflux(
        self,
        check: Callable[[], Any],
        *,
        stage: str = "checking MFLUX model availability",
        timeout_seconds: float | None = None,
        emit_diagnostic: bool = True,
    ) -> WorkerOperation:
        """Run a bounded MFLUX model availability check."""
        return self._submit(
            AdapterKind.MFLUX,
            check,
            stage=stage,
            timeout_seconds=timeout_seconds,
            availability_check=True,
            emit_availability=emit_diagnostic,
            request_cancel=None,
            dispose_result=None,
            invocation_started=None,
            invocation_finished=None,
        )

    def shutdown(self, *, wait_milliseconds: int = 0) -> None:
        """Discard queued work and optionally wait for bounded supervisors."""
        with self._state_lock:
            self._closed = True
            records = tuple(self._records.values())
        for record in records:
            record.handle.cancel()
        self._mflux_pool.clear()
        self._records.clear()
        self._deadline_timer.stop()
        if wait_milliseconds > 0:
            self._mflux_pool.waitForDone(wait_milliseconds)

    def _submit(
        self,
        adapter: AdapterKind,
        operation: Callable[[], Any],
        *,
        stage: str,
        timeout_seconds: float | None,
        availability_check: bool,
        emit_availability: bool,
        request_cancel: Callable[[], None] | None,
        dispose_result: Callable[[Any], None] | None,
        invocation_started: Callable[[], None] | None,
        invocation_finished: Callable[[], None] | None,
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
        cancellation = _CancellationControl(request_cancel)
        start_lock = Lock()
        handle = WorkerOperation(operation_id, cancellation, start_lock)
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
                emit_availability=emit_availability,
                cancellation=cancellation,
                start_lock=start_lock,
                deadline=deadline,
            )
            handle.cancelled.connect(partial(self._discard, operation_id))
            runnable = _BoundedRunnable(
                operation_id=operation_id,
                operation=operation,
                dispose_result=dispose_result,
                invocation_started=invocation_started,
                invocation_finished=invocation_finished,
                deadline=deadline,
                cancellation=cancellation,
                start_lock=start_lock,
                invocation_slots=self._invocation_slots[adapter],
                invocation_thread=self._invocation_thread,
                dispatcher=self._dispatcher,
            )
            self._mflux_pool.start(runnable)
        self._schedule_deadline()
        return handle

    @Slot(object, object)
    def _complete(self, operation_id: UUID, outcome: object) -> None:
        record = self._records.pop(operation_id, None)
        if record is None:
            if isinstance(outcome, _Success):
                outcome.dispose()
            return
        self._schedule_deadline()
        if record.handle.status is OperationStatus.CANCELLED:
            if isinstance(outcome, _Success):
                outcome.dispose()
            return
        if isinstance(outcome, _Success):
            record.handle._succeed(outcome.value)
            if record.emit_availability:
                self._mark_available(record)
            return
        if isinstance(outcome, _Timeout):
            record.cancellation.request()
        failure = _failure_for(record, outcome)
        record.handle._fail(failure)
        if record.emit_availability and (
            record.availability_check
            or failure.kind
            in {
                WorkerFailureKind.MODEL_UNAVAILABLE,
                WorkerFailureKind.MODEL_LOAD,
            }
        ):
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
                record.cancellation.request()
            if record.handle.status is OperationStatus.CANCELLED:
                continue
            failure = _failure_for(record, _Timeout())
            record.handle._fail(failure)
            if record.availability_check and record.emit_availability:
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
    return "MFLUX"


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
        (ModelUnavailableError, WorkerFailureKind.MODEL_UNAVAILABLE),
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
