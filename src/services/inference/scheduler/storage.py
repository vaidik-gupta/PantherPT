from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
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
    def pending(self) -> tuple[StoredRequest[Payload], ...]:
        """Return candidates without reserving them."""

    @abstractmethod
    def claim(self, request_ids: tuple[UUID, ...]) -> tuple[StoredRequest[Payload], ...]:
        """Atomically claim all selected requests or raise without changing storage."""

    @abstractmethod
    def finish(self, request_id: UUID, state: RequestState) -> StoredRequest[Payload]:
        """Remove a terminal request after execution has stopped."""


class InMemoryRequestStorage[Payload](RequestStorage[Payload]):
    """Single-process registry and insertion-ordered, non-destructive pending index.

    All calls must run on one event loop. Methods do not suspend, so claiming is
    atomic relative to other tasks on that loop. Payload ownership stays with the
    caller; payloads should not be mutated after submission.
    """

    def __init__(self, max_requests: int) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive.")
        self._max_requests = max_requests
        self._requests: dict[UUID, StoredRequest[Payload]] = {}
        self._pending: dict[UUID, StoredRequest[Payload]] = {}

    def submit(self, request_id: UUID, payload: Payload) -> StoredRequest[Payload]:
        if request_id in self._requests:
            raise ValueError(f"Request {request_id} is already stored.")
        if len(self._requests) >= self._max_requests:
            raise CapacityExceededError("Request storage is at capacity.")
        record = StoredRequest(request_id, payload, monotonic())
        self._requests[request_id] = record
        self._pending[request_id] = record
        return record

    def get(self, request_id: UUID) -> StoredRequest[Payload]:
        return self._requests[request_id]

    def pending(self) -> tuple[StoredRequest[Payload], ...]:
        return tuple(self._pending.values())

    def claim(self, request_ids: tuple[UUID, ...]) -> tuple[StoredRequest[Payload], ...]:
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("A claim cannot contain duplicate request IDs.")
        records = tuple(self.get(request_id) for request_id in request_ids)
        if any(record.state is not RequestState.QUEUED for record in records):
            raise ValueError("Only queued requests can be claimed.")

        claimed = tuple(replace(record, state=RequestState.RUNNING) for record in records)
        for record in claimed:
            self._requests[record.request_id] = record
            del self._pending[record.request_id]
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
            del self._pending[request_id]
        return finished
