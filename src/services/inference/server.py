import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from starlette.middleware.base import RequestResponseEndpoint

from src.services.inference.auth import require_api_key
from src.services.inference.scheduler.base import BaseScheduler, SchedulerUnavailableError
from src.services.inference.scheduler.delivery import DeliveryClosedError
from src.services.inference.scheduler.storage import CapacityExceededError
from src.services.inference.schemas import InferenceRequest, InferenceResponse


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    api_key = os.environ.get("INFERENCE_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("INFERENCE_API_KEY must be set to a nonempty API key.")

    app.state.inference_api_key = api_key.encode("utf-8")
    scheduler: BaseScheduler[InferenceRequest, str] | None = getattr(
        app.state, "scheduler", None
    )
    try:
        if scheduler is not None:
            await scheduler.start()
        yield
    finally:
        try:
            if scheduler is not None:
                await scheduler.close()
        finally:
            del app.state.inference_api_key


async def add_request_id_header(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    response = await call_next(request)
    request_id = getattr(request.state, "request_id", None)
    if request_id is not None:
        response.headers["X-Request-ID"] = str(request_id)
    return response


async def health() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Liveness check is not implemented.",
    )


async def ready() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Readiness check is not implemented.",
    )


async def models() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model discovery is not implemented.",
    )


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _deliver_inference(
    request: Request,
    payload: InferenceRequest,
    scheduler: BaseScheduler[InferenceRequest, str],
) -> str:
    inference = asyncio.create_task(
        scheduler.submit_and_wait(request.state.request_id, payload)
    )
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            (inference, disconnect), return_when=asyncio.FIRST_COMPLETED
        )
        if inference in done:
            return await inference
        await disconnect
        raise asyncio.CancelledError("Inference client disconnected.")
    finally:
        for task in (inference, disconnect):
            if not task.done():
                task.cancel()
        for task in (inference, disconnect):
            with suppress(asyncio.CancelledError):
                await task


async def infer(request: InferenceRequest, http_request: Request) -> InferenceResponse:
    scheduler: BaseScheduler[InferenceRequest, str] | None = http_request.app.state.scheduler
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Inference is not implemented.",
        )
    request_id = http_request.state.request_id
    payload = request.model_copy(deep=True, update={"id": str(request_id)})
    try:
        output = await _deliver_inference(http_request, payload, scheduler)
    except (CapacityExceededError, SchedulerUnavailableError, DeliveryClosedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed for request %s.", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed.",
        ) from exc
    return InferenceResponse(request_id=request_id, output=output)


def create_app(
    scheduler: BaseScheduler[InferenceRequest, str] | None = None,
) -> FastAPI:
    application = FastAPI(
        title="PantherPT Inference Service",
        description="API-key authenticated inference service with an injectable scheduler.",
        lifespan=lifespan,
        dependencies=[Depends(require_api_key)],
        responses={401: {"description": "Missing or invalid API key."}},
    )
    application.state.scheduler = scheduler
    application.middleware("http")(add_request_id_header)
    application.add_api_route(
        "/health", health, methods=["GET"], summary="Check service liveness", status_code=501
    )
    application.add_api_route(
        "/ready", ready, methods=["GET"], summary="Check model readiness", status_code=501
    )
    application.add_api_route(
        "/models", models, methods=["GET"], summary="List available models", status_code=501
    )
    application.add_api_route(
        "/infer",
        infer,
        methods=["POST"],
        summary="Run inference",
        response_model=InferenceResponse,
        responses={
            500: {"description": "Runner execution failed."},
            501: {"description": "No scheduler configured."},
            503: {"description": "Scheduler unavailable or at capacity."},
        },
    )
    return application


app = create_app()
