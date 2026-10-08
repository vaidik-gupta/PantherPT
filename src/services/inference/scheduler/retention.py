from abc import ABC, abstractmethod
from uuid import UUID


class ResultRetention[Result](ABC):
    @abstractmethod
    def retain(self, request_id: UUID, result: Result) -> None:
        pass

    @abstractmethod
    def get(self, request_id: UUID) -> Result:
        pass

    @abstractmethod
    def release(self, request_id: UUID) -> None:
        pass


class UntilDeliveredResultRetention[Result](ResultRetention[Result]):
    """Keep results in memory until delivery releases them; no history or TTL."""

    def __init__(self) -> None:
        self._results: dict[UUID, Result] = {}

    def retain(self, request_id: UUID, result: Result) -> None:
        if request_id in self._results:
            raise ValueError(f"Request {request_id} already has a retained result.")
        self._results[request_id] = result

    def get(self, request_id: UUID) -> Result:
        return self._results[request_id]

    def release(self, request_id: UUID) -> None:
        del self._results[request_id]
