import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.tools.base import ToolRegistry, execute_call

SYSTEM = (
    "You can call tools. Use tool results as the only source for numbers and facts. "
    "If a tool returns an error, fix the arguments and call it again, or explain the "
    "problem to the user."
)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass(frozen=True)
class AssistantTurn:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ToolModel(Protocol):
    async def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn: ...


@dataclass
class ToolRecord:
    name: str
    arguments: str
    result: str
    ok: bool


@dataclass
class AgentResult:
    answer: str
    records: list[ToolRecord] = field(default_factory=list)
    steps: int = 0
    finished: bool = True


class ToolAgent:
    """prompt -> model tool_calls -> run tools concurrently -> feed results back -> answer.
    max_steps stops runaway loops (for example a tool that keeps failing)."""

    def __init__(
        self,
        model: ToolModel,
        registry: ToolRegistry,
        *,
        max_steps: int = 5,
        tool_timeout: float = 10.0,
    ) -> None:
        self._model = model
        self._registry = registry
        self._max_steps = max_steps
        self._timeout = tool_timeout

    async def run(self, prompt: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        records: list[ToolRecord] = []
        for step in range(1, self._max_steps + 1):
            turn = await self._model.respond(messages, self._registry.specs())
            if not turn.tool_calls:
                return AgentResult(turn.content or "", records, step, True)
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in turn.tool_calls
                    ],
                }
            )
            results = await asyncio.gather(
                *(
                    execute_call(self._registry, call.name, call.arguments, self._timeout)
                    for call in turn.tool_calls
                )
            )
            for call, result in zip(turn.tool_calls, results, strict=True):
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                ok = "error" not in json.loads(result)
                records.append(ToolRecord(call.name, call.arguments, result, ok))
        answer = f"Stopped after {self._max_steps} steps without a final answer."
        return AgentResult(answer, records, self._max_steps, False)
