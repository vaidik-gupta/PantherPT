import asyncio
from uuid import uuid4

import pytest

from src.services.inference.scheduler.delivery import (
    ClientDelivery,
    DeliveryClosedError,
    DeliveryFailure,
    DeliveryResult,
    FutureClientDelivery,
)
from src.services.inference.scheduler.retention import (
    ResultRetention,
    UntilDeliveredResultRetention,
)
from src.services.inference.scheduler.storage import (
    CapacityExceededError,
    InMemoryRequestStorage,
    RequestState,
    RequestStorage,
)
from src.services.inference.schemas import InferenceRequest


@pytest.mark.parametrize("abstraction", [RequestStorage, ClientDelivery, ResultRetention])
def test_interfaces_require_implementations(abstraction):
    with pytest.raises(TypeError):
        abstraction()


@pytest.mark.parametrize("capacity", [0, -1])
def test_storage_requires_positive_capacity(capacity):
    with pytest.raises(ValueError, match="positive"):
        InMemoryRequestStorage(capacity)


def test_pending_selection_is_non_destructive_and_not_limited_to_fifo():
    storage = InMemoryRequestStorage[InferenceRequest](3)
    ids = tuple(uuid4() for _ in range(3))
    records = tuple(
        storage.submit(request_id, InferenceRequest(prompt="Hello", model=model))
        for request_id, model in zip(ids, ("gpt2", "other", "gpt2"))
    )

    assert storage.pending() == records
    assert storage.pending() == records
    selected = tuple(
        record.request_id for record in reversed(storage.pending())
        if record.payload.model == "gpt2"
    )
    claimed = storage.claim(selected)

    assert tuple(record.request_id for record in claimed) == (ids[2], ids[0])
    assert all(record.state is RequestState.RUNNING for record in claimed)
    assert storage.pending() == (records[1],)
    assert records[0].state is RequestState.QUEUED
    assert storage.get(ids[0]).state is RequestState.RUNNING


@pytest.mark.parametrize("invalid_claim", ["unknown", "running", "duplicate"])
def test_batch_claim_is_all_or_nothing(invalid_claim):
    storage = InMemoryRequestStorage[str](2)
    first, second = uuid4(), uuid4()
    storage.submit(first, "first")
    storage.submit(second, "second")
    if invalid_claim == "unknown":
        selected = (first, uuid4())
        error = KeyError
    elif invalid_claim == "running":
        storage.claim((second,))
        selected = (first, second)
        error = ValueError
    else:
        selected = (first, first)
        error = ValueError
    before = storage.pending()

    with pytest.raises(error):
        storage.claim(selected)

    assert storage.pending() == before
    assert storage.get(first).state is RequestState.QUEUED


def test_capacity_includes_running_requests_and_releases_on_finish():
    storage = InMemoryRequestStorage[str](1)
    request_id = uuid4()
    storage.submit(request_id, "first")
    storage.claim((request_id,))

    with pytest.raises(CapacityExceededError):
        storage.submit(uuid4(), "second")

    finished = storage.finish(request_id, RequestState.SUCCEEDED)
    assert finished.state is RequestState.SUCCEEDED
    with pytest.raises(KeyError):
        storage.get(request_id)
    storage.submit(uuid4(), "second")


def test_duplicate_submission_does_not_replace_payload():
    storage = InMemoryRequestStorage[str](2)
    request_id = uuid4()
    original = storage.submit(request_id, "original")

    with pytest.raises(ValueError, match="already stored"):
        storage.submit(request_id, "replacement")

    assert storage.get(request_id) == original


@pytest.mark.parametrize("state", list(RequestState))
def test_queued_requests_can_only_be_cancelled(state):
    storage = InMemoryRequestStorage[str](1)
    request_id = uuid4()
    storage.submit(request_id, "queued")
    if state is RequestState.CANCELLED:
        assert storage.finish(request_id, state).state is state
        assert storage.pending() == ()
        storage.submit(uuid4(), "replacement")
    else:
        with pytest.raises(ValueError):
            storage.finish(request_id, state)
        assert storage.get(request_id).state is RequestState.QUEUED


@pytest.mark.parametrize(
    "state", [RequestState.SUCCEEDED, RequestState.FAILED, RequestState.CANCELLED]
)
def test_running_requests_finish_once(state):
    storage = InMemoryRequestStorage[str](1)
    request_id = uuid4()
    storage.submit(request_id, "running")
    storage.claim((request_id,))

    assert storage.finish(request_id, state).state is state
    with pytest.raises(KeyError):
        storage.finish(request_id, state)


def test_retention_releases_results_and_rejects_overwrites():
    retention = UntilDeliveredResultRetention[str]()
    request_id = uuid4()
    retention.retain(request_id, "result")
    assert retention.get(request_id) == "result"
    assert retention.get(request_id) == "result"
    with pytest.raises(ValueError, match="retained result"):
        retention.retain(request_id, "replacement")
    retention.release(request_id)
    with pytest.raises(KeyError):
        retention.get(request_id)
    with pytest.raises(KeyError):
        retention.release(request_id)


@pytest.mark.parametrize("result", ["generated text", None])
def test_completion_before_wait_is_delivered_and_released(result):
    async def scenario():
        retention = UntilDeliveredResultRetention[DeliveryResult[str | None]]()
        delivery = FutureClientDelivery[str | None](retention)
        request_id = uuid4()
        delivery.register(request_id)
        delivery.complete(request_id, result)
        assert retention.get(request_id).value == result
        with pytest.raises(ValueError, match="completion"):
            delivery.complete(request_id, result)
        assert await delivery.wait(request_id) == result
        with pytest.raises(KeyError):
            retention.get(request_id)
        with pytest.raises(KeyError):
            await delivery.wait(request_id)

    asyncio.run(scenario())


def test_runner_to_waiting_client_lifecycle():
    async def scenario():
        storage = InMemoryRequestStorage[InferenceRequest](1)
        retention = UntilDeliveredResultRetention[DeliveryResult[str]]()
        delivery = FutureClientDelivery[str](retention)
        request_id = uuid4()
        payload = InferenceRequest(prompt="Hello", id="untrusted-client-id")
        storage.submit(request_id, payload)
        delivery.register(request_id)
        waiter = asyncio.create_task(delivery.wait(request_id))
        await asyncio.sleep(0)
        assert not waiter.done()

        claimed, = storage.claim((request_id,))
        assert claimed.request_id == request_id
        assert claimed.payload.prompt == "Hello"
        storage.finish(request_id, RequestState.SUCCEEDED)
        delivery.complete(request_id, "Hello world")

        assert await waiter == "Hello world"
        assert storage.pending() == ()
        with pytest.raises(KeyError):
            storage.get(request_id)
        with pytest.raises(KeyError):
            retention.get(request_id)

    asyncio.run(scenario())


def test_runner_failure_is_raised_and_released():
    async def scenario():
        retention = UntilDeliveredResultRetention[DeliveryResult[str]]()
        delivery = FutureClientDelivery[str](retention)
        request_id = uuid4()
        delivery.register(request_id)
        error = RuntimeError("Runner failed.")
        delivery.fail(request_id, error)
        assert isinstance(retention.get(request_id), DeliveryFailure)

        with pytest.raises(RuntimeError, match="Runner failed") as raised:
            await delivery.wait(request_id)
        assert raised.value is error
        with pytest.raises(KeyError):
            retention.get(request_id)

    asyncio.run(scenario())


def test_only_one_waiter_and_registration_per_request():
    async def scenario():
        delivery = FutureClientDelivery[str](UntilDeliveredResultRetention())
        request_id = uuid4()
        delivery.register(request_id)
        with pytest.raises(ValueError, match="registered"):
            delivery.register(request_id)
        waiter = asyncio.create_task(delivery.wait(request_id))
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="waiter"):
            await delivery.wait(request_id)
        delivery.complete(request_id, "result")
        assert await waiter == "result"

    asyncio.run(scenario())


@pytest.mark.parametrize("complete_first", [False, True])
def test_waiter_cancellation_cleans_delivery_without_freeing_runner_capacity(complete_first):
    async def scenario():
        storage = InMemoryRequestStorage[str](1)
        retention = UntilDeliveredResultRetention[DeliveryResult[str]]()
        delivery = FutureClientDelivery[str](retention)
        request_id = uuid4()
        storage.submit(request_id, "input")
        storage.claim((request_id,))
        delivery.register(request_id)
        waiter = asyncio.create_task(delivery.wait(request_id))
        await asyncio.sleep(0)
        if complete_first:
            delivery.complete(request_id, "result")
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        with pytest.raises(KeyError):
            retention.get(request_id)
        with pytest.raises(KeyError):
            delivery.complete(request_id, "late output")
        with pytest.raises(CapacityExceededError):
            storage.submit(uuid4(), "new input")
        storage.finish(request_id, RequestState.CANCELLED)
        storage.submit(uuid4(), "new input")

    asyncio.run(scenario())


def test_discard_cancels_waiter_and_prevents_reuse_until_it_exits():
    async def scenario():
        delivery = FutureClientDelivery[str](UntilDeliveredResultRetention())
        request_id = uuid4()
        delivery.register(request_id)
        waiter = asyncio.create_task(delivery.wait(request_id))
        await asyncio.sleep(0)
        delivery.discard(request_id)
        with pytest.raises(ValueError, match="registered"):
            delivery.register(request_id)
        with pytest.raises(asyncio.CancelledError):
            await waiter
        delivery.register(request_id)
        delivery.complete(request_id, "new result")
        assert await delivery.wait(request_id) == "new result"

    asyncio.run(scenario())


def test_shutdown_wakes_waiters_and_discards_unclaimed_results():
    async def scenario():
        retention = UntilDeliveredResultRetention[DeliveryResult[str]]()
        delivery = FutureClientDelivery[str](retention)
        waiting_id, unclaimed_id, queued_id = uuid4(), uuid4(), uuid4()
        for request_id in (waiting_id, unclaimed_id, queued_id):
            delivery.register(request_id)
        delivery.complete(unclaimed_id, "unclaimed")
        waiter = asyncio.create_task(delivery.wait(waiting_id))
        await asyncio.sleep(0)
        delivery.close()
        delivery.close()

        with pytest.raises(DeliveryClosedError):
            await waiter
        with pytest.raises(DeliveryClosedError):
            delivery.register(uuid4())
        for request_id in (waiting_id, unclaimed_id, queued_id):
            with pytest.raises(KeyError):
                retention.get(request_id)
            with pytest.raises(KeyError):
                await delivery.wait(request_id)

    asyncio.run(scenario())


def test_unknown_delivery_requests_raise_explicitly():
    async def scenario():
        delivery = FutureClientDelivery[str](UntilDeliveredResultRetention())
        request_id = uuid4()
        for operation in (
            lambda: delivery.complete(request_id, "result"),
            lambda: delivery.fail(request_id, RuntimeError("failure")),
            lambda: delivery.discard(request_id),
        ):
            with pytest.raises(KeyError):
                operation()
        with pytest.raises(KeyError):
            await delivery.wait(request_id)

    asyncio.run(scenario())
