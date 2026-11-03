from dataclasses import dataclass

from app.agents.nodes import Review


@dataclass
class MockAgentModel:
    """Deterministic stand-in for an LLM. The flags script specific failure modes."""

    broken_code: bool = False  # first program has a bug the debugger must fix
    unfixable: bool = False  # the "fix" changes nothing, so the loop guard must fire
    sloppy_first_draft: bool = False  # first draft lacks a citation, reviewer sends it back
    reviewer_always_fails: bool = False  # revisions run out, so the fallback must fire

    async def plan(self, goal: str) -> list[str]:
        return ["retrieve facts", "compute the figure", "draft the report", "review", "send"]

    async def write_code(self, goal: str, context: list[str]) -> str:
        if self.broken_code or self.unfixable:
            return "print(round(price * 0.8, 2))"  # NameError: price
        return "print(round(120 * 0.8, 2))"

    async def fix_code(self, code: str, error: str) -> str:
        return code if self.unfixable else code.replace("price", "120")

    async def draft(self, goal: str, context: list[str], result: str, feedback: str) -> str:
        text = f"Goal: {goal}. Computed result: {result.strip()}."
        if context:
            text += f" Fact: {context[0]}"
        cited = bool(feedback) or not self.sloppy_first_draft
        return f"{text} [1]" if context and cited else text

    async def review(self, goal: str, draft: str, context: list[str]) -> Review:
        if self.reviewer_always_fails:
            return Review(False, "Needs more detail.")
        if context and "[1]" not in draft:
            return Review(False, "Add a [1] citation for the retrieved fact.")
        return Review(True, "")
