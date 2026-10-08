from secrets import compare_digest
from typing import Annotated
from uuid import uuid4

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="InferenceAPIKey",
    description="API key configured through INFERENCE_API_KEY.",
    auto_error=False,
)


async def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
) -> None:
    if (
        api_key is None
        or len(request.headers.getlist("X-API-Key")) != 1
        or not compare_digest(
            api_key.encode("utf-8"), request.app.state.inference_api_key
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    request.state.request_id = uuid4()
