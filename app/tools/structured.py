import json
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.rag.llm import LLM
from app.tools.base import format_validation_error

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class StructuredOutputError(Exception):
    pass


class Ticket(BaseModel):
    """Example schema: extract a support ticket from free text."""

    model_config = ConfigDict(extra="forbid")

    error_code: str = Field(pattern=r"^ERR-\d{4}$")
    severity: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1, max_length=200)


def _strictify(node: Any) -> None:
    """OpenAI strict mode wants every object closed, with every property required."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def response_format_for(model: type[BaseModel], name: str | None = None) -> dict[str, Any]:
    """The response_format argument for OpenAI JSON Schema mode (constrained decoding).
    Pydantic stays the source of truth: the reply is still validated after decoding."""
    schema = model.model_json_schema()
    _strictify(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": name or model.__name__, "strict": True, "schema": schema},
    }


async def extract_structured(llm: LLM, model: type[T], text: str, *, max_retries: int = 2) -> T:
    """Ask for JSON, validate with Pydantic, and feed validation errors back on failure."""
    system = (
        "TASK: EXTRACT\n"
        "Return only one JSON object that matches this JSON Schema. No prose, no code fences.\n"
        f"Schema: {json.dumps(model.model_json_schema())}"
    )
    last_error = ""
    for _ in range(max_retries + 1):
        prompt = text
        if last_error:
            prompt = (
                f"{text}\n\nYour previous answer was invalid: {last_error}\n"
                "Return corrected JSON only."
            )
        raw = await llm.complete(system, prompt)
        try:
            return model.model_validate_json(_FENCE.sub("", raw.strip()).strip())
        except ValidationError as exc:
            last_error = format_validation_error(exc)
    raise StructuredOutputError(
        f"No valid {model.__name__} after {max_retries + 1} attempts: {last_error}"
    )
