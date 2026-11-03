import ast
import asyncio
import operator
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import Field

from app.rag.grounding import support_score
from app.tools.base import StrictArgs, Tool, ToolError, ToolRegistry

# ---------------------------------------------------------------- calculator

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate plain arithmetic by walking the AST. Never uses eval()."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, int | float) and not isinstance(value, bool):
                return value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("Exponent too large (limit is 100)")
            return float(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](walk(node.operand)))
        raise ToolError("Only numbers and + - * / // % ** are allowed")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ToolError("Not a valid arithmetic expression") from None
    try:
        return walk(tree)
    except ZeroDivisionError:
        raise ToolError("Division by zero") from None
    except OverflowError:
        raise ToolError("Number too large") from None


class CalculatorArgs(StrictArgs):
    expression: str = Field(min_length=1, max_length=200, description="e.g. '(12 + 3) * 4'")


def calculator_tool() -> Tool:
    async def run(args: CalculatorArgs) -> float | int:
        value = safe_eval(args.expression)
        return int(value) if value.is_integer() else round(value, 10)

    return Tool("calculator", "Evaluate an arithmetic expression.", CalculatorArgs, run)


# ------------------------------------------------------------------- weather


class WeatherClient(Protocol):
    async def current(self, city: str) -> dict[str, Any]: ...


_WEATHER: dict[str, tuple[float, str]] = {
    "hyderabad": (31.0, "Partly cloudy"),
    "mumbai": (29.0, "Humid"),
    "london": (12.0, "Rain"),
}


class OfflineWeather:
    """Canned data with simulated latency. Swap in an httpx client for a real API."""

    async def current(self, city: str) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        key = city.strip().lower()
        if key not in _WEATHER:
            known = ", ".join(sorted(name.title() for name in _WEATHER))
            raise ToolError(f"No weather data for '{city}'. Known cities: {known}")
        temp_c, condition = _WEATHER[key]
        return {"city": city.strip().title(), "temp_c": temp_c, "condition": condition}


class WeatherArgs(StrictArgs):
    city: str = Field(min_length=1, max_length=80)
    units: Literal["celsius", "fahrenheit"] = "celsius"


def weather_tool(client: WeatherClient) -> Tool:
    async def run(args: WeatherArgs) -> dict[str, Any]:
        data = await client.current(args.city)
        if args.units == "fahrenheit":
            return {
                "city": data["city"],
                "temp_f": round(data["temp_c"] * 9 / 5 + 32, 1),
                "condition": data["condition"],
            }
        return data

    return Tool("get_weather", "Current weather for one city.", WeatherArgs, run)


# ----------------------------------------------------------------------- sql

MAX_ROWS = 50
ORDERS_SCHEMA = (
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, product TEXT NOT NULL, "
    "amount REAL NOT NULL, status TEXT NOT NULL);"
)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b",
    re.IGNORECASE,
)
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def validate_select(sql: str) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ToolError("Empty query")
    if ";" in cleaned:
        raise ToolError("Only a single statement is allowed")
    if not re.match(r"select\b", cleaned, re.IGNORECASE):
        raise ToolError("Only SELECT statements are allowed")
    if _FORBIDDEN.search(cleaned):
        raise ToolError("Query contains a forbidden keyword; this tool is read-only")
    return cleaned


def _authorizer(
    action: int, arg1: str | None, arg2: str | None, db: str | None, source: str | None
) -> int:
    """Engine-level guard: even if validation missed something, writes are denied."""
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class TenantDb:
    """One in-memory SQLite database per tenant, read-only after seeding. Isolation is by
    construction: a tenant's connection physically contains only that tenant's rows. In
    Postgres the equivalent is row-level security plus a read-only role."""

    def __init__(self, orders: Mapping[str, Sequence[tuple[int, str, float, str]]]) -> None:
        self._lock = threading.Lock()
        self._conns: dict[str, sqlite3.Connection] = {}
        for tenant, rows in orders.items():
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn.executescript(ORDERS_SCHEMA)
            conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", rows)
            conn.commit()
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(_authorizer)
            self._conns[tenant] = conn

    def query(self, tenant: str, sql: str) -> list[dict[str, Any]]:
        conn = self._conns.get(tenant)
        if conn is None:
            raise ToolError("Unknown tenant")
        cleaned = validate_select(sql)
        try:
            with self._lock:
                cursor = conn.execute(cleaned)
                columns = [col[0] for col in cursor.description or []]
                rows = cursor.fetchmany(MAX_ROWS)
        except sqlite3.Error as exc:
            raise ToolError(
                f"SQL error: {exc}. Table: orders(id, product, amount, status)"
            ) from exc
        return [dict(zip(columns, row, strict=True)) for row in rows]


class SqlArgs(StrictArgs):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="One SELECT on orders(id, product, amount, status). Read-only.",
    )


def sql_tool(db: TenantDb, tenant_id: str) -> Tool:
    """The tenant comes from the server, never from the model's arguments."""

    async def run(args: SqlArgs) -> list[dict[str, Any]]:
        return await asyncio.to_thread(db.query, tenant_id, args.query)

    return Tool("query_orders", "Run a read-only SELECT on the orders table.", SqlArgs, run)


# ---------------------------------------------------------------- web search


class SearchBackend(Protocol):
    async def search(self, query: str, k: int) -> list[dict[str, Any]]: ...


class CorpusSearch:
    """Offline stand-in for a web search API, ranked with the lexical support score."""

    def __init__(self, docs: Sequence[tuple[str, str]]) -> None:
        self._docs = list(docs)

    async def search(self, query: str, k: int) -> list[dict[str, Any]]:
        scored = sorted(
            ((support_score(query, text), title, text) for title, text in self._docs),
            reverse=True,
        )
        return [
            {"title": title, "snippet": text[:200], "score": round(score, 2)}
            for score, title, text in scored
            if score > 0
        ][:k]


class SearchArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=200)
    max_results: int = Field(default=3, ge=1, le=5)


def search_tool(backend: SearchBackend) -> Tool:
    async def run(args: SearchArgs) -> list[dict[str, Any]]:
        return await backend.search(args.query, args.max_results)

    return Tool("web_search", "Search the web and return titles and snippets.", SearchArgs, run)


# ------------------------------------------------------------------- wiring

_ORDERS: dict[str, list[tuple[int, str, float, str]]] = {
    "acme": [(1, "Teams plan", 120.0, "paid"), (2, "Teams plan", 120.0, "refunded"),
             (3, "Support addon", 30.0, "paid")],
    "globex": [(1, "Teams plan", 120.0, "paid")],
}  # fmt: skip
_DOCS = [
    ("Refund policy", "Refunds for annual plans are issued within 30 days of purchase."),
    ("Support hours", "Support is available by email on weekdays from 9am to 6pm."),
    ("Error ERR-5034", "ERR-5034 means the login token expired. Sign in again."),
]


def offline_registry(tenant_id: str = "acme") -> ToolRegistry:
    return ToolRegistry(
        [
            calculator_tool(),
            weather_tool(OfflineWeather()),
            sql_tool(TenantDb(_ORDERS), tenant_id),
            search_tool(CorpusSearch(_DOCS)),
        ]
    )
