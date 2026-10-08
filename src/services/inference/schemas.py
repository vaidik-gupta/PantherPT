import uuid
from uuid import UUID

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Basic text inference request; subclass to add service-specific fields."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the inference request.",
    )
    prompt: str = Field(min_length=1, description="Text to run inference on.")
    model: str | None = Field(
        default=None,
        min_length=1,
        description="Model identifier; omit to use the service's default model.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature; higher values mean more random outputs.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens to generate in the output.",
    )
    stop: list[str] | None = Field(
        default=None,
        description="List of tokens at which to stop generation.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability; higher values mean more diverse outputs.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling; higher values mean more diverse outputs.",
    )


class InferenceResponse(BaseModel):
    request_id: UUID
    output: str
