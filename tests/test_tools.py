import asyncio
import json
import time
from typing import Any

import pytest

from app.tools.agent import AssistantTurn, ToolAgent, ToolCall
from app.tools.base import StrictArgs, Tool, ToolError, ToolRegistry, execute_call
from app.tools.builtin import offline_registry, safe_eval
from app.tools.models import MockToolModel
from app.tools.structured import (
    StructuredOutputError,
    Ticket,
    extract_structured,
    response_format_for,
)


def _call(registry: ToolRegistry, name: str, args: dict[str, Any] | str) -> dict[str, Any]:
    raw = args if isinstance(args, str) else json.dumps(args)
    out: dict[str, Any] = json.loads(asyncio.run(execute_call(registry, name, raw)))
    return out


# ------------------------------------------------------------ calculator / safety


def test_safe_eval_computes_arithmetic() -> None:
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("-(2 ** 3)") == -8


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "open('f')", "2 ** 999", "1 / 0"])
def test_safe_eval_rejects_unsafe_or_invalid(bad: str) -> None:
    with pytest.raises(ToolError):
        safe_eval(bad)


# ---------------------------------------------------------- defensive execution


def test_unknown_tool_lists_available_tools() -> None:
    out = _call(offline_registry(), "send_email", {})
    assert "calculator" in out["error"]


def test_invalid_json_arguments_return_error() -> None:
    assert "not valid JSON" in _call(offline_registry(), "calculator", "{oops")["error"]


def test_wrong_arguments_name_the_bad_field() -> None:
    error = _call(offline_registry(), "calculator", {"expr": "1+1"})["error"]
    assert "expr" in error
    assert "expression" in error


def test_tool_failure_becomes_error_not_exception() -> None:
    out = _call(offline_registry(), "get_weather", {"city": "Atlantis"})
    assert "Known cities" in out["error"]


def test_weather_units_conversion() -> None:
    out = _call(offline_registry(), "get_weather", {"city": "London", "units": "fahrenheit"})
    assert out["result"]["temp_f"] == 53.6


# ----------------------------------------------------------------- sql safety


def test_sql_is_scoped_to_the_tenant() -> None:
    acme = _call(
        offline_registry("acme"), "query_orders", {"query": "SELECT COUNT(*) AS n FROM orders"}
    )
    globex = _call(
        offline_registry("globex"), "query_orders", {"query": "SELECT COUNT(*) AS n FROM orders"}
    )
    assert acme["result"] == [{"n": 3}]
    assert globex["result"] == [{"n": 1}]


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "UPDATE orders SET amount = 0",
        "SELECT * FROM orders WHERE id = 1; --",
    ],
)
def test_sql_rejects_writes_and_multiple_statements(sql: str) -> None:
    registry = offline_registry()
    assert "error" in _call(registry, "query_orders", {"query": sql})
    after = _call(registry, "query_orders", {"query": "SELECT COUNT(*) AS n FROM orders"})
    assert after["result"] == [{"n": 3}]


def test_sql_error_message_helps_the_model_correct() -> None:
    error = _call(offline_registry(), "query_orders", {"query": "SELECT * FROM users"})["error"]
    assert "orders(id, product, amount, status)" in error


# ----------------------------------------------------------------- agent loop


class ScriptedModel:
    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = list(turns)
        self.seen: list[list[dict[str, Any]]] = []

    async def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        self.seen.append(list(messages))
        return self._turns.pop(0)


def _tool_turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(tool_calls=calls)


def test_full_tool_protocol_feeds_result_back_to_model() -> None:
    model = ScriptedModel(
        [
            _tool_turn(ToolCall("c1", "calculator", '{"expression": "12*7"}')),
            AssistantTurn(content="12 times 7 is 84."),
        ]
    )
    result = asyncio.run(ToolAgent(model, offline_registry()).run("What is 12*7?"))
    assert result.answer == "12 times 7 is 84."
    assert result.finished
    second_call_messages = model.seen[1]
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "c1"
    assert "84" in second_call_messages[-1]["content"]


def test_model_self_corrects_after_invalid_arguments() -> None:
    model = ScriptedModel(
        [
            _tool_turn(ToolCall("c1", "calculator", '{"expr": "1+1"}')),
            _tool_turn(ToolCall("c2", "calculator", '{"expression": "1+1"}')),
            AssistantTurn(content="2"),
        ]
    )
    result = asyncio.run(ToolAgent(model, offline_registry()).run("1+1?"))
    assert [r.ok for r in result.records] == [False, True]
    assert result.answer == "2"


def test_agent_stops_at_max_steps() -> None:
    bad = ToolCall("c", "calculator", "{}")
    model = ScriptedModel([_tool_turn(bad), _tool_turn(bad), _tool_turn(bad)])
    result = asyncio.run(ToolAgent(model, offline_registry(), max_steps=3).run("loop"))
    assert not result.finished
    assert result.steps == 3


class SleepArgs(StrictArgs):
    ms: int


async def _sleep(args: SleepArgs) -> str:
    await asyncio.sleep(args.ms / 1000)
    return "done"


async def _boom(args: SleepArgs) -> str:
    raise RuntimeError("secret internal detail")


def test_parallel_tool_calls_run_concurrently() -> None:
    registry = ToolRegistry([Tool("sleep", "sleep", SleepArgs, _sleep)])
    calls = tuple(ToolCall(f"c{i}", "sleep", '{"ms": 200}') for i in range(3))
    model = ScriptedModel([_tool_turn(*calls), AssistantTurn(content="ok")])
    start = time.perf_counter()
    result = asyncio.run(ToolAgent(model, registry).run("go"))
    elapsed = time.perf_counter() - start
    assert len(result.records) == 3
    assert elapsed < 0.5  # sequential execution would take about 0.6 s


def test_tool_timeout_is_reported_not_raised() -> None:
    registry = ToolRegistry([Tool("sleep", "sleep", SleepArgs, _sleep)])
    out = json.loads(asyncio.run(execute_call(registry, "sleep", '{"ms": 500}', timeout=0.05)))
    assert "timed out" in out["error"]


def test_crashing_tool_does_not_leak_internal_details() -> None:
    registry = ToolRegistry([Tool("boom", "boom", SleepArgs, _boom)])
    out = json.loads(asyncio.run(execute_call(registry, "boom", '{"ms": 1}')))
    assert "RuntimeError" in out["error"]
    assert "secret" not in out["error"]


def test_mock_model_runs_parallel_weather_calls_end_to_end() -> None:
    agent = ToolAgent(MockToolModel(), offline_registry())
    result = asyncio.run(agent.run("What is the weather in Hyderabad and Mumbai?"))
    assert len(result.records) == 2
    assert all(r.ok for r in result.records)
    assert result.finished


def test_mock_model_combines_calculator_and_search() -> None:
    agent = ToolAgent(MockToolModel(), offline_registry())
    result = asyncio.run(agent.run("Calculate 12 * 7 and search for refund policy"))
    assert {r.name for r in result.records} == {"calculator", "web_search"}
    assert "84" in result.answer


# ----------------------------------------------------------------- structured


class FakeLLM:
    name = "fake"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self._replies.pop(0)


GOOD = '{"error_code": "ERR-5034", "severity": "high", "summary": "Login fails"}'
BAD = '{"error_code": "5034", "severity": "high", "summary": "Login fails"}'


def test_structured_extraction_retries_with_error_feedback() -> None:
    llm = FakeLLM([BAD, f"```json\n{GOOD}\n```"])
    ticket = asyncio.run(extract_structured(llm, Ticket, "Login fails with ERR-5034"))
    assert ticket.error_code == "ERR-5034"
    assert len(llm.prompts) == 2
    assert "invalid" in llm.prompts[1]
    assert "error_code" in llm.prompts[1]


def test_structured_extraction_gives_up_after_retries() -> None:
    llm = FakeLLM([BAD, BAD, BAD])
    with pytest.raises(StructuredOutputError):
        asyncio.run(extract_structured(llm, Ticket, "x", max_retries=2))
    assert len(llm.prompts) == 3


def test_response_format_is_strict_json_schema() -> None:
    fmt = response_format_for(Ticket)
    assert fmt["type"] == "json_schema"
    schema = fmt["json_schema"]["schema"]
    assert fmt["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"error_code", "severity", "summary"}
