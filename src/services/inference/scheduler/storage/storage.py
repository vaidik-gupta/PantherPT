from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from itertools import islice
from time import monotonic
from uuid import UUID


class RequestState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CapacityExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredRequest[Payload]:
    request_id: UUID
    payload: Payload
    admitted_at: float
    state: RequestState = RequestState.QUEUED


class RequestStorage[Payload](ABC):
    """Event-loop-owned storage; implementations define pending organization."""

    @abstractmethod
    def submit(self, request_id: UUID, payload: Payload) -> StoredRequest[Payload]:
        pass

    @abstractmethod
    def get(self, request_id: UUID) -> StoredRequest[Payload]:
        pass

    @abstractmethod
    def pending(
        self, limit: int | None = None
    ) -> tuple[StoredRequest[Payload], ...]:
        """Return up to `limit` candidates without reserving them.

        `limit=None` returns every pending request. A positive `limit` returns
        only the first that many in selection order; implementations should read
        just that prefix rather than materializing the full set.
        """

    @abstractmethod
    def claim(self, request_ids: tuple[UUID, ...]) -> tuple[StoredRequest[Payload], ...]:
        """Atomically claim all selected requests or raise without changing storage."""

    @abstractmethod
    def finish(self, request_id: UUID, state: RequestState) -> StoredRequest[Payload]:
        """Remove a terminal request after execution has stopped."""


class InMemoryStorageBase[Payload](RequestStorage[Payload]):
    """Shared single-process registry, capacity bound, and atomic transitions.

    All in-memory storage implementations inherit from this. It owns the request
    registry, admission limit, and the QUEUED -> RUNNING -> terminal transitions,
    which are identical regardless of how pending requests are organized.

    Subclasses own only the pending organization through the abstract
    `_enqueue_pending`, `_dequeue_pending`, and `pending` hooks.

    All calls must run on one event loop. Methods do not suspend, so claiming is
    atomic relative to other tasks on that loop. Payload ownership stays with the
    caller; payloads should not be mutated after submission.
    """

    def __init__(self, max_requests: int) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive.")
        self._max_requests = max_requests
        self._requests: dict[UUID, StoredRequest[Payload]] = {}

    @abstractmethod
    def _enqueue_pending(self, record: StoredRequest[Payload]) -> None:
        """Add a newly submitted QUEUED request to the pending organization."""

    @abstractmethod
    def _dequeue_pending(self, request_id: UUID) -> None:
        """Remove a request from the pending organization once it leaves QUEUED."""

    def submit(self, request_id: UUID, payload: Payload) -> StoredRequest[Payload]:
        if request_id in self._requests:
            raise ValueError(f"Request {request_id} is already stored.")
        if len(self._requests) >= self._max_requests:
            raise CapacityExceededError("Request storage is at capacity.")
        record = StoredRequest(request_id, payload, monotonic())
        self._requests[request_id] = record
        self._enqueue_pending(record)
        return record

    def get(self, request_id: UUID) -> StoredRequest[Payload]:
        return self._requests[request_id]

    def claim(self, request_ids: tuple[UUID, ...]) -> tuple[StoredRequest[Payload], ...]:
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("A claim cannot contain duplicate request IDs.")
        records = tuple(self.get(request_id) for request_id in request_ids)
        if any(record.state is not RequestState.QUEUED for record in records):
            raise ValueError("Only queued requests can be claimed.")

        claimed = tuple(replace(record, state=RequestState.RUNNING) for record in records)
        for record in claimed:
            self._requests[record.request_id] = record
            self._dequeue_pending(record.request_id)
        return claimed

    def finish(self, request_id: UUID, state: RequestState) -> StoredRequest[Payload]:
        if state not in (
            RequestState.SUCCEEDED, RequestState.FAILED, RequestState.CANCELLED
        ):
            raise ValueError("finish requires a terminal state.")
        record = self.get(request_id)
        if record.state is RequestState.QUEUED and state is not RequestState.CANCELLED:
            raise ValueError("Queued requests can only finish through cancellation.")
        finished = replace(record, state=state)
        del self._requests[request_id]
        if record.state is RequestState.QUEUED:
            self._dequeue_pending(request_id)
        return finished


class InMemoryFifo[Payload](InMemoryStorageBase[Payload]):
    """Insertion-ordered, non-destructive FIFO pending index.

    Pending requests are offered oldest-first; selection does not reserve them.
    """

    def __init__(self, max_requests: int) -> None:
        super().__init__(max_requests)
        self._pending: dict[UUID, StoredRequest[Payload]] = {}

    def _enqueue_pending(self, record: StoredRequest[Payload]) -> None:
        self._pending[record.request_id] = record

    def _dequeue_pending(self, request_id: UUID) -> None:
        del self._pending[request_id]

    def pending(self, limit: int | None = None) -> tuple[StoredRequest[Payload], ...]:
        if limit is None:
            return tuple(self._pending.values())
        if limit < 0:
            raise ValueError("limit must be non-negative.")
        return tuple(islice(self._pending.values(), limit))
