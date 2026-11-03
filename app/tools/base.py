import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """A failure whose message is safe and useful to show the model."""


class StrictArgs(BaseModel):
    """Base class for tool arguments. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


def format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors()
    )


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[StrictArgs]
    run: Callable[[Any], Awaitable[Any]]

    def spec(self) -> dict[str, Any]:
        """OpenAI-style tool declaration generated from the Pydantic schema."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]


def _error(message: str) -> str:
    return json.dumps({"error": message})


async def execute_call(
    registry: ToolRegistry, name: str, raw_arguments: str, timeout: float = 10.0
) -> str:
    """Run one tool call. Always returns a JSON string and never raises, so the model
    can read the error and correct itself (that is the retry loop)."""
    tool = registry.get(name)
    if tool is None:
        return _error(f"Unknown tool '{name}'. Available tools: {', '.join(registry.names)}")
    try:
        payload = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return _error(f"Arguments are not valid JSON: {exc.msg}")
    try:
        args = tool.args_model.model_validate(payload)
    except ValidationError as exc:
        return _error(f"Invalid arguments: {format_validation_error(exc)}")
    try:
        result = await asyncio.wait_for(tool.run(args), timeout)
    except TimeoutError:
        return _error(f"Tool '{name}' timed out after {timeout}s")
    except ToolError as exc:
        return _error(str(exc))
    except Exception as exc:
        logger.exception("Tool %s crashed", name)
        return _error(f"Tool '{name}' failed with {type(exc).__name__}")
    return json.dumps({"result": result}, default=str)
