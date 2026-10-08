import asyncio
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from src.services.inference.scheduler.base import BaseScheduler, SchedulerUnavailableError
from src.services.inference.scheduler.delivery import (
    DeliveryClosedError,
    DeliveryResult,
    FutureClientDelivery,
)
from src.services.inference.scheduler.retention import UntilDeliveredResultRetention
from src.services.inference.scheduler.storage import (
    CapacityExceededError,
    InMemoryRequestStorage,
    RequestState,
)
from src.services.inference.schemas import InferenceRequest
from src.services.inference.server import _deliver_inference, create_app, lifespan


class IdleScheduler(BaseScheduler[InferenceRequest, str]):
    async def run(self):
        await asyncio.Future()

    async def stop_running(self):
        pass


class EchoScheduler(BaseScheduler[InferenceRequest, str]):
    async def run(self):
        while True:
            candidates = await self.wait_for_pending()
            for record in self.claim(tuple(item.request_id for item in candidates)):
                self.complete(record.request_id, f"{record.payload.id}: {record.payload.prompt}")

    async def stop_running(self):
        pass


class FailingRunnerScheduler(EchoScheduler):
    async def run(self):
        while True:
            candidates = await self.wait_for_pending()
            for record in self.claim(tuple(item.request_id for item in candidates)):
                self.fail(record.request_id, RuntimeError("Private runner details."))


class FailingLoopScheduler(EchoScheduler):
    async def run(self):
        await self.wait_for_pending()
        raise RuntimeError("Scheduling policy failed.")


class HoldingScheduler(BaseScheduler[InferenceRequest, str]):
    def __init__(self, storage, delivery):
        super().__init__(storage, delivery)
        self.claimed = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self):
        while True:
            candidates = await self.wait_for_pending()
            records = self.claim(tuple(item.request_id for item in candidates))
            self.claimed.set()
            await self.release.wait()
            self.release.clear()
            for record in records:
                self.complete(record.request_id, "finished")

    async def stop_running(self):
        pass


def test_scheduler_requires_policy_and_runner_shutdown_implementations():
    with pytest.raises(TypeError):
        BaseScheduler.with_defaults(max_requests=1)


def test_coordinator_admission_claim_completion_and_retention():
    async def scenario():
        storage = InMemoryRequestStorage[InferenceRequest](1)
        retention = UntilDeliveredResultRetention[DeliveryResult[str]]()
        scheduler = IdleScheduler(storage, FutureClientDelivery(retention))
        await scheduler.start()
        request_id = uuid4()
        try:
            scheduler.submit(request_id, InferenceRequest(prompt="Hello"))
            record, = scheduler.claim((request_id,))
            assert record.state is RequestState.RUNNING
            with pytest.raises(CapacityExceededError):
                scheduler.submit(uuid4(), InferenceRequest(prompt="Overload"))
            scheduler.complete(request_id, "Hello world")
            assert retention.get(request_id).value == "Hello world"
            assert await scheduler.wait(request_id) == "Hello world"
            with pytest.raises(KeyError):
                retention.get(request_id)
            with pytest.raises(KeyError):
                storage.get(request_id)
            with pytest.raises(KeyError):
                scheduler.complete(request_id, "duplicate")
        finally:
            await scheduler.close()

    asyncio.run(scenario())


def test_admission_failure_rolls_back_delivery_registration():
    async def scenario():
        scheduler = IdleScheduler.with_defaults(max_requests=1)
        await scheduler.start()
        first, rejected = uuid4(), uuid4()
        try:
            scheduler.submit(first, InferenceRequest(prompt="first"))
            with pytest.raises(CapacityExceededError):
                scheduler.submit(rejected, InferenceRequest(prompt="second"))
            scheduler.cancel(first)
            scheduler.submit(rejected, InferenceRequest(prompt="retry"))
            scheduler.claim((rejected,))
            scheduler.complete(rejected, "result")
            assert await scheduler.wait(rejected) == "result"
        finally:
            await scheduler.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("running", [False, True])
def test_client_cancellation_cleans_queue_or_keeps_running_capacity(running):
    async def scenario():
        scheduler = IdleScheduler.with_defaults(max_requests=1)
        await scheduler.start()
        request_id = uuid4()
        try:
            waiter = asyncio.create_task(
                scheduler.submit_and_wait(request_id, InferenceRequest(prompt="Hello"))
            )
            await asyncio.sleep(0)
            if running:
                scheduler.claim((request_id,))
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert scheduler.pending() == ()
            if running:
                assert scheduler.cancellation_requested(request_id)
                with pytest.raises(CapacityExceededError):
                    scheduler.submit(uuid4(), InferenceRequest(prompt="new"))
                scheduler.complete(request_id, "late output")
                with pytest.raises(KeyError):
                    await scheduler.wait(request_id)
            scheduler.submit(uuid4(), InferenceRequest(prompt="replacement"))
        finally:
            await scheduler.close()

    asyncio.run(scenario())


def test_runner_can_acknowledge_cancellation_and_invalid_completion_is_rejected():
    async def scenario():
        scheduler = IdleScheduler.with_defaults(max_requests=1)
        await scheduler.start()
        request_id = uuid4()
        try:
            scheduler.submit(request_id, InferenceRequest(prompt="Hello"))
            with pytest.raises(ValueError, match="claimed"):
                scheduler.complete(request_id, "premature")
            scheduler.claim((request_id,))
            with pytest.raises(ValueError, match="not been requested"):
                scheduler.cancelled(request_id)
            scheduler.cancel(request_id)
            scheduler.cancelled(request_id)
            scheduler.submit(uuid4(), InferenceRequest(prompt="replacement"))
        finally:
            await scheduler.close()

    asyncio.run(scenario())


def test_failure_wakes_clients_and_stops_admission(caplog):
    async def scenario():
        scheduler = FailingLoopScheduler.with_defaults(max_requests=1)
        await scheduler.start()
        with pytest.raises(SchedulerUnavailableError, match="failed"):
            await asyncio.wait_for(
                scheduler.submit_and_wait(uuid4(), InferenceRequest(prompt="Hello")), 1
            )
        with pytest.raises(SchedulerUnavailableError):
            scheduler.submit(uuid4(), InferenceRequest(prompt="new"))
        await scheduler.close()
        await scheduler.close()

    asyncio.run(scenario())
    assert "Scheduling loop failed" in caplog.text


def test_shutdown_retains_capacity_until_runner_stops():
    async def scenario():
        stopped = asyncio.Event()
        stop_requested = asyncio.Event()

        class SlowStoppingScheduler(IdleScheduler):
            async def stop_running(self):
                stop_requested.set()
                await stopped.wait()

        storage = InMemoryRequestStorage[InferenceRequest](1)
        scheduler = SlowStoppingScheduler(storage, FutureClientDelivery(
            UntilDeliveredResultRetention()
        ))
        await scheduler.start()
        request_id = uuid4()
        waiter = asyncio.create_task(
            scheduler.submit_and_wait(request_id, InferenceRequest(prompt="Hello"))
        )
        await asyncio.sleep(0)
        scheduler.claim((request_id,))
        closing = asyncio.create_task(scheduler.close())
        await stop_requested.wait()
        with pytest.raises(DeliveryClosedError):
            await waiter
        assert storage.get(request_id).state is RequestState.RUNNING
        assert not closing.done()
        stopped.set()
        await closing
        with pytest.raises(KeyError):
            storage.get(request_id)
        with pytest.raises(SchedulerUnavailableError):
            scheduler.submit(uuid4(), InferenceRequest(prompt="new"))

    asyncio.run(scenario())


def test_unstarted_closed_and_duplicate_start_are_rejected():
    async def scenario():
        scheduler = IdleScheduler.with_defaults(max_requests=1)
        with pytest.raises(SchedulerUnavailableError):
            scheduler.submit(uuid4(), InferenceRequest(prompt="Hello"))
        await scheduler.start()
        with pytest.raises(ValueError, match="already"):
            await scheduler.start()
        await asyncio.gather(scheduler.close(), scheduler.close())
        with pytest.raises(SchedulerUnavailableError):
            await scheduler.start()

    asyncio.run(scenario())


def test_cancelled_shutdown_caller_does_not_interrupt_cleanup():
    async def scenario():
        stop_requested = asyncio.Event()
        allow_stop = asyncio.Event()

        class SlowStoppingScheduler(IdleScheduler):
            async def stop_running(self):
                stop_requested.set()
                await allow_stop.wait()

        scheduler = SlowStoppingScheduler.with_defaults(max_requests=1)
        await scheduler.start()
        closing = asyncio.create_task(scheduler.close())
        await stop_requested.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        allow_stop.set()
        await scheduler.close()

    asyncio.run(scenario())


def test_shutdown_failure_is_logged_and_does_not_release_running_capacity(caplog):
    async def scenario():
        class CannotStopScheduler(IdleScheduler):
            async def stop_running(self):
                raise RuntimeError("Runner could not be stopped.")

        storage = InMemoryRequestStorage[InferenceRequest](1)
        scheduler = CannotStopScheduler(storage, FutureClientDelivery(
            UntilDeliveredResultRetention()
        ))
        await scheduler.start()
        request_id = uuid4()
        scheduler.submit(request_id, InferenceRequest(prompt="Hello"))
        scheduler.claim((request_id,))
        with pytest.raises(RuntimeError, match="could not be stopped"):
            await scheduler.close()
        assert storage.get(request_id).state is RequestState.RUNNING

    asyncio.run(scenario())
    assert "Scheduler shutdown failed" in caplog.text


def test_shutdown_hook_cannot_await_its_own_close_task():
    async def scenario():
        class ReentrantShutdownScheduler(IdleScheduler):
            async def stop_running(self):
                await self.close()

        scheduler = ReentrantShutdownScheduler.with_defaults(max_requests=1)
        await scheduler.start()
        with pytest.raises(ValueError, match="cannot close itself"):
            await asyncio.wait_for(scheduler.close(), 0.1)

    asyncio.run(scenario())


def test_http_uses_authenticated_id_and_returns_runner_result(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    scheduler = EchoScheduler.with_defaults(max_requests=1)
    application = create_app(scheduler)
    with TestClient(application) as client:
        response = client.post(
            "/infer",
            headers={"X-API-Key": "test-key", "X-Request-ID": str(uuid4())},
            json={"prompt": "Hello", "id": "client-supplied-id"},
        )
        unauthorized = client.post("/infer", json={"prompt": "Hello"})

    assert response.status_code == 200
    result = response.json()
    assert result["request_id"] == response.headers["X-Request-ID"]
    assert UUID(result["request_id"]).version == 4
    assert result["output"] == f"{result['request_id']}: Hello"
    assert unauthorized.status_code == 401
    assert not hasattr(application.state, "inference_api_key")
    assert application.openapi()["paths"]["/infer"]["post"]["responses"]["200"]


@pytest.mark.parametrize(
    ("scheduler_class", "expected_status"),
    [(FailingRunnerScheduler, 500), (FailingLoopScheduler, 503)],
)
def test_http_runner_and_scheduler_failures_are_explicit(
    monkeypatch, scheduler_class, expected_status, caplog
):
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    with TestClient(create_app(scheduler_class.with_defaults(max_requests=1))) as client:
        response = client.post(
            "/infer", headers={"X-API-Key": "test-key"}, json={"prompt": "Hello"}
        )

    assert response.status_code == expected_status
    assert UUID(response.headers["X-Request-ID"]).version == 4
    assert "Private runner details" not in response.text
    assert caplog.records


def test_http_overload_and_cancelled_client_do_not_free_running_capacity(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")

    async def scenario():
        storage = InMemoryRequestStorage[InferenceRequest](1)
        scheduler = HoldingScheduler(storage, FutureClientDelivery(
            UntilDeliveredResultRetention()
        ))
        application = create_app(scheduler)
        async with lifespan(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://test",
                headers={"X-API-Key": "test-key"},
            ) as client:
                first = asyncio.create_task(client.post("/infer", json={"prompt": "first"}))
                await asyncio.wait_for(scheduler.claimed.wait(), 1)
                response = await client.post("/infer", json={"prompt": "second"})
                assert response.status_code == 503
                assert "capacity" in response.json()["detail"]
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                response = await client.post("/infer", json={"prompt": "third"})
                assert response.status_code == 503

    asyncio.run(scenario())


def test_disconnect_detaches_delivery_and_requests_cooperative_cancellation():
    async def scenario():
        scheduler = HoldingScheduler.with_defaults(max_requests=1)
        await scheduler.start()
        messages = asyncio.Queue()
        request = Request({"type": "http", "headers": []}, receive=messages.get)
        request_id = uuid4()
        request.state.request_id = request_id
        try:
            delivery = asyncio.create_task(_deliver_inference(
                request, InferenceRequest(prompt="Hello"), scheduler
            ))
            await asyncio.wait_for(scheduler.claimed.wait(), 1)
            messages.put_nowait({"type": "http.disconnect"})
            with pytest.raises(asyncio.CancelledError):
                await delivery
            assert scheduler.cancellation_requested(request_id)
            with pytest.raises(CapacityExceededError):
                scheduler.submit(uuid4(), InferenceRequest(prompt="new"))
            scheduler.cancelled(request_id)
        finally:
            await scheduler.close()

    asyncio.run(scenario())
