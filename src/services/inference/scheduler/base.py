import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Self
from uuid import UUID

from src.services.inference.scheduler.delivery import (
    ClientDelivery,
    DeliveryClosedError,
    FutureClientDelivery,
)
from src.services.inference.scheduler.retention import UntilDeliveredResultRetention
from src.services.inference.scheduler.storage import (
    InMemoryRequestStorage,
    RequestState,
    RequestStorage,
    StoredRequest,
)


logger = logging.getLogger(__name__)


class SchedulerUnavailableError(RuntimeError):
    pass


class BaseScheduler[Payload, Result](ABC):
    """Common request lifecycle, owned by one event loop.

    Subclasses select/claim work in run() and report stopped execution through
    complete(), fail(), or cancelled(). They must keep blocking runner work off
    the event loop and stop all owned execution in stop_running().
    """

    def __init__(
        self, storage: RequestStorage[Payload], delivery: ClientDelivery[Result]
    ) -> None:
        self._storage = storage
        self._delivery = delivery
        self._live: set[UUID] = set()
        self._attached: set[UUID] = set()
        self._waiting: set[UUID] = set()
        self._cancel_requested: set[UUID] = set()
        self._work = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False
        self._closing = False
        self._failure: Exception | None = None
        self._shutdown_complete = False
        self._shutdown_lock = asyncio.Lock()

    @classmethod
    def with_defaults(cls, max_requests: int) -> Self:
        return cls(
            InMemoryRequestStorage[Payload](max_requests),
            FutureClientDelivery[Result](UntilDeliveredResultRetention()),
        )

    @abstractmethod
    async def run(self) -> None:
        """Run scheduling policy until shutdown; returning unexpectedly is fatal."""

    @abstractmethod
    async def stop_running(self) -> None:
        """Return only after all runner work has stopped, including worker work."""

    def _check_ready(self) -> None:
        if self._failure is not None:
            raise SchedulerUnavailableError("Scheduler failed.") from self._failure
        if self._closed or self._task is None:
            raise SchedulerUnavailableError("Scheduler is not running.")

    async def start(self) -> None:
        if self._closed:
            raise SchedulerUnavailableError("Scheduler is closed.")
        if self._task is not None:
            raise ValueError("Scheduler has already been started.")
        self._task = asyncio.create_task(self._serve(), name=type(self).__name__)
        self._task.add_done_callback(self._observe_failure)
        await asyncio.sleep(0)
        self._check_ready()

    @staticmethod
    def _observe_failure(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _serve(self) -> None:
        try:
            await self.run()
        except asyncio.CancelledError:
            if self._closing:
                return
            error = SchedulerUnavailableError("Scheduling loop was cancelled.")
            logger.error("%s", error)
        except Exception as exc:
            error = exc
            logger.exception("Scheduling loop failed.")
        else:
            if self._closing:
                return
            error = SchedulerUnavailableError("Scheduling loop exited unexpectedly.")
            logger.error("%s", error)
        self._failure = error
        try:
            await self._shutdown(SchedulerUnavailableError("Scheduler failed."))
        except Exception:
            logger.exception("Failed to stop runner work after scheduler failure.")
            raise

    def submit(self, request_id: UUID, payload: Payload) -> StoredRequest[Payload]:
        self._check_ready()
        self._delivery.register(request_id)
        try:
            record = self._storage.submit(request_id, payload)
        except Exception:
            self._delivery.discard(request_id)
            raise
        self._live.add(request_id)
        self._attached.add(request_id)
        self._work.set()
        return record

    async def wait(self, request_id: UUID) -> Result:
        if request_id not in self._attached:
            raise KeyError(request_id)
        if request_id in self._waiting:
            raise ValueError(f"Request {request_id} already has a waiter.")
        self._waiting.add(request_id)
        try:
            return await self._delivery.wait(request_id)
        finally:
            self._waiting.remove(request_id)
            self._attached.discard(request_id)
            if request_id in self._live:
                self._request_cancellation(request_id)

    async def submit_and_wait(self, request_id: UUID, payload: Payload) -> Result:
        self.submit(request_id, payload)
        return await self.wait(request_id)

    def pending(self) -> tuple[StoredRequest[Payload], ...]:
        self._check_ready()
        return self._storage.pending()

    async def wait_for_pending(self) -> tuple[StoredRequest[Payload], ...]:
        while True:
            candidates = self.pending()
            if candidates:
                return candidates
            self._work.clear()
            await self._work.wait()

    def claim(self, request_ids: tuple[UUID, ...]) -> tuple[StoredRequest[Payload], ...]:
        self._check_ready()
        return self._storage.claim(request_ids)

    def cancellation_requested(self, request_id: UUID) -> bool:
        self._storage.get(request_id)
        return request_id in self._cancel_requested

    def _finish(self, request_id: UUID, state: RequestState) -> None:
        self._storage.finish(request_id, state)
        self._live.remove(request_id)
        self._cancel_requested.discard(request_id)
        self._work.set()

    def _request_cancellation(self, request_id: UUID) -> None:
        record = self._storage.get(request_id)
        self._cancel_requested.add(request_id)
        if record.state is RequestState.QUEUED:
            self._finish(request_id, RequestState.CANCELLED)

    def cancel(self, request_id: UUID) -> None:
        self._storage.get(request_id)
        if request_id in self._attached:
            self._delivery.discard(request_id)
            self._attached.remove(request_id)
        self._request_cancellation(request_id)

    def _validate_completion(self, request_id: UUID) -> bool:
        if self._storage.get(request_id).state is not RequestState.RUNNING:
            raise ValueError("Only claimed requests can complete.")
        if request_id in self._cancel_requested:
            logger.info("Discarding output for cancelled request %s.", request_id)
            self._finish(request_id, RequestState.CANCELLED)
            return False
        return True

    def complete(self, request_id: UUID, result: Result) -> None:
        if self._validate_completion(request_id):
            self._delivery.complete(request_id, result)
            self._finish(request_id, RequestState.SUCCEEDED)

    def fail(self, request_id: UUID, error: Exception) -> None:
        if self._validate_completion(request_id):
            self._delivery.fail(request_id, error)
            self._finish(request_id, RequestState.FAILED)

    def cancelled(self, request_id: UUID) -> None:
        if not self.cancellation_requested(request_id):
            raise ValueError("Cancellation has not been requested.")
        self._finish(request_id, RequestState.CANCELLED)

    async def _shutdown(self, error: Exception) -> None:
        async with self._shutdown_lock:
            if not self._shutdown_complete:
                await self._shutdown_impl(error)

    async def _shutdown_impl(self, error: Exception) -> None:
        self._closed = True
        self._work.set()
        for request_id in tuple(self._live):
            if request_id in self._attached:
                self._delivery.fail(request_id, error)
            self._request_cancellation(request_id)
        try:
            await self.stop_running()
            for request_id in tuple(self._live):
                self._finish(request_id, RequestState.CANCELLED)
        finally:
            self._delivery.close()
            self._attached.intersection_update(self._waiting)
        self._shutdown_complete = True

    async def close(self) -> None:
        if asyncio.current_task() in (self._task, self._close_task):
            raise ValueError("The scheduler lifecycle cannot close itself.")
        if self._close_task is None:
            self._closing = True
            self._closed = True
            self._close_task = asyncio.create_task(
                self._close(), name=f"{type(self).__name__}.close"
            )
            self._close_task.add_done_callback(self._observe_failure)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        try:
            if self._task is not None:
                if self._failure is None and not self._task.done():
                    self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    if not self._task.cancelled():
                        raise
            await self._shutdown(DeliveryClosedError("Scheduler is shutting down."))
        except Exception:
            logger.exception("Scheduler shutdown failed.")
            raise
