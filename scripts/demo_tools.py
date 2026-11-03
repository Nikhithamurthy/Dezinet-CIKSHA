import argparse
import asyncio
import time

from app.rag.llm import LLM, get_llm
from app.tools.agent import ToolAgent, ToolModel
from app.tools.builtin import offline_registry
from app.tools.models import MockToolModel, OpenAIToolModel
from app.tools.structured import Ticket, extract_structured

PROMPTS = [
    "What is the weather in Hyderabad, Mumbai and London?",
    "Calculate (12 + 3) * 4 and search for refund policy",
    "How many orders do we have?",
    "What is the weather in Atlantis?",
]


class FlakyLLM:
    """Offline stand-in: the first reply breaks the schema, the second is correct."""

    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return '{"error_code": "5034", "severity": "urgent", "summary": "Login fails"}'
        return '{"error_code": "ERR-5034", "severity": "high", "summary": "Login fails"}'


async def main(kind: str) -> None:
    llm: LLM = FlakyLLM() if kind == "mock" else get_llm(kind)
    ticket = await extract_structured(llm, Ticket, "Customers cannot log in, ERR-5034, urgent.")
    print(f"structured: {ticket.model_dump()}")

    model: ToolModel = MockToolModel() if kind == "mock" else OpenAIToolModel()
    agent = ToolAgent(model, offline_registry())
    for prompt in PROMPTS:
        start = time.perf_counter()
        result = await agent.run(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\nQ: {prompt}")
        for record in result.records:
            print(f"  tool {record.name} ok={record.ok} args={record.arguments}")
        print(f"A: {result.answer}  [{result.steps} steps, {elapsed:.0f} ms]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="mock", choices=["mock", "openai"])
    asyncio.run(main(parser.parse_args().llm))
