from pydantic import BaseModel, Field, field_validator

MODEL_PATTERN = r"^[a-zA-Z0-9._-]{1,50}$"


class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: str = Field(default="mock-llm-1", pattern=MODEL_PATTERN)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value.strip()


class InferenceResponse(BaseModel):
    model: str
    output: str
    prompt_tokens: int
    completion_tokens: int


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    model: str = Field(default="mock-embed-1", pattern=MODEL_PATTERN)

    @field_validator("texts")
    @classmethod
    def texts_valid(cls, values: list[str]) -> list[str]:
        for text in values:
            if not text.strip():
                raise ValueError("texts must not contain blank strings")
            if len(text) > 8000:
                raise ValueError("each text must be at most 8000 characters")
        return values


class EmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    embeddings: list[list[float]]


class StatusResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    database: str
