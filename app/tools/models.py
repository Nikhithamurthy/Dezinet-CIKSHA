import json
import re
from typing import Any

from app.tools.agent import AssistantTurn, ToolCall

_WEATHER_RE = re.compile(r"weather in ([A-Z][a-z]+(?:(?:,| and) *[A-Z][a-z]+)*)")
_SEARCH_RE = re.compile(r"search (?:the web )?for ([^?.]+)", re.IGNORECASE)


class MockToolModel:
    """Offline stand-in. It does NOT understand language: it spots a few patterns, emits
    tool calls (several at once when asked), then summarises the tool results."""

    name = "mock-tools"

    async def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        tool_messages = [m for m in messages[last_user + 1 :] if m["role"] == "tool"]
        if tool_messages:
            return AssistantTurn(content=self._summarise(tool_messages))
        calls = self._plan(str(messages[last_user]["content"]))
        if not calls:
            return AssistantTurn(content="I can't answer that with the available tools.")
        return AssistantTurn(tool_calls=tuple(calls))

    def _plan(self, text: str) -> list[ToolCall]:
        planned: list[tuple[str, dict[str, Any]]] = []
        weather = _WEATHER_RE.search(text)
        if weather:
            for city in re.split(r",| and ", weather.group(1)):
                if city.strip():
                    planned.append(("get_weather", {"city": city.strip()}))
        expression = self._expression(text)
        if expression:
            planned.append(("calculator", {"expression": expression}))
        lower = text.lower()
        if "how many orders" in lower:
            planned.append(("query_orders", {"query": "SELECT COUNT(*) AS n FROM orders"}))
        elif "total" in lower and "orders" in lower:
            planned.append(("query_orders", {"query": "SELECT SUM(amount) AS total FROM orders"}))
        found = _SEARCH_RE.search(text)
        if found:
            planned.append(("web_search", {"query": found.group(1).strip()}))
        return [
            ToolCall(f"call_{i}", name, json.dumps(args)) for i, (name, args) in enumerate(planned)
        ]

    def _expression(self, text: str) -> str | None:
        candidates: list[str] = re.findall(r"[\d(][\d\s.+\-*/()%]*", text)
        for candidate in candidates:
            if re.search(r"\d\s*[+\-*/%]\s*[\d(]", candidate):
                return candidate.strip()
        return None

    def _summarise(self, tool_messages: list[dict[str, Any]]) -> str:
        parts = []
        for message in tool_messages:
            data = json.loads(message["content"])
            parts.append(
                f"Error: {data['error']}" if "error" in data else json.dumps(data["result"])
            )
        return "Tool results: " + " | ".join(parts)


class OpenAIToolModel:
    """Real model via the OpenAI tools API. Needs OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self.name = model
        self._model = model
        self._client: Any = AsyncOpenAI(api_key=api_key)

    async def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=messages,
            tools=tools,
            parallel_tool_calls=True,
        )
        message = response.choices[0].message
        calls = tuple(
            ToolCall(call.id, call.function.name, call.function.arguments)
            for call in (message.tool_calls or [])
        )
        return AssistantTurn(content=message.content, tool_calls=calls)
