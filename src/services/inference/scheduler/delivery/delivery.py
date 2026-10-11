import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from src.services.inference.scheduler.retention import ResultRetention


@dataclass(frozen=True)
class DeliverySuccess[Result]:
    value: Result


@dataclass(frozen=True)
class DeliveryFailure:
    error: Exception


type DeliveryResult[Result] = DeliverySuccess[Result] | DeliveryFailure


class DeliveryClosedError(RuntimeError):
    pass


class ClientDelivery[Result](ABC):
    @abstractmethod
    def register(self, request_id: UUID) -> None:
        pass

    @abstractmethod
    async def wait(self, request_id: UUID) -> Result:
        pass

    @abstractmethod
    def complete(self, request_id: UUID, result: Result) -> None:
        pass

    @abstractmethod
    def fail(self, request_id: UUID, error: Exception) -> None:
        pass

    @abstractmethod
    def discard(self, request_id: UUID) -> None:
        """Detach delivery; execution cancellation belongs to the scheduler."""

    @abstractmethod
    def close(self) -> None:
        pass


class FutureClientDelivery[Result](ClientDelivery[Result]):
    """One asynchronous waiter per request, suitable for a held-open HTTP call.

    All methods run on the same event loop. Completion notifies the waiter; the
    injected retention policy owns the result until consumption or abandonment.
    Cancellation of the waiter detaches delivery but does not stop runner work.
    """

    def __init__(self, retention: ResultRetention[DeliveryResult[Result]]) -> None:
        self._retention = retention
        self._notifications: dict[UUID, asyncio.Future[None]] = {}
        self._waiting: set[UUID] = set()
        self._closed = False

    def register(self, request_id: UUID) -> None:
        if self._closed:
            raise DeliveryClosedError("Client delivery is closed.")
        if request_id in self._notifications or request_id in self._waiting:
            raise ValueError(f"Request {request_id} is already registered.")
        self._notifications[request_id] = asyncio.get_running_loop().create_future()

    async def wait(self, request_id: UUID) -> Result:
        notification = self._notifications[request_id]
        if request_id in self._waiting:
            raise ValueError(f"Request {request_id} already has a delivery waiter.")
        self._waiting.add(request_id)
        try:
            await asyncio.shield(notification)
            result = self._retention.get(request_id)
            if isinstance(result, DeliveryFailure):
                raise result.error
            return result.value
        finally:
            self._waiting.remove(request_id)
            if request_id in self._notifications:
                self.discard(request_id)

    def _publish(self, request_id: UUID, result: DeliveryResult[Result]) -> None:
        notification = self._notifications[request_id]
        if notification.done():
            raise ValueError(f"Request {request_id} already has a completion.")
        self._retention.retain(request_id, result)
        notification.set_result(None)

    def complete(self, request_id: UUID, result: Result) -> None:
        self._publish(request_id, DeliverySuccess(result))

    def fail(self, request_id: UUID, error: Exception) -> None:
        self._publish(request_id, DeliveryFailure(error))

    def discard(self, request_id: UUID) -> None:
        notification = self._notifications[request_id]
        if notification.done() and not notification.cancelled():
            self._retention.release(request_id)
        else:
            notification.cancel()
        del self._notifications[request_id]

    def close(self) -> None:
        self._closed = True
        for request_id, notification in tuple(self._notifications.items()):
            if request_id not in self._waiting:
                self.discard(request_id)
            elif not notification.done():
                self.fail(request_id, DeliveryClosedError("Client delivery is closed."))
