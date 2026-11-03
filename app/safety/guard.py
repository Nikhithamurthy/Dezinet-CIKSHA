"""Defence in depth: input guard -> context guard -> output guard."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.safety.injection import scan
from app.safety.pii import Recognizer, Redaction, placeholder_kind, redact

BLOCKED_MESSAGE = "I can't help with that request."
LEAK_MESSAGE = "I can't share internal configuration."


@dataclass(frozen=True)
class InputDecision:
    allowed: bool
    text: str  # what may be sent to the LLM (PII already replaced by placeholders)
    redaction: Redaction
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OutputDecision:
    text: str
    reasons: tuple[str, ...]


class Guard:
    def __init__(
        self,
        *,
        input_threshold: float = 0.7,
        doc_threshold: float = 0.5,
        canary: str = "CANARY-7F3A9C21",
        extra_recognizers: Sequence[Recognizer] = (),
    ) -> None:
        self.input_threshold = input_threshold
        self.doc_threshold = doc_threshold
        self.canary = canary  # put this token in the system prompt; it must never be echoed
        self._extra = tuple(extra_recognizers)

    def check_input(self, text: str) -> InputDecision:
        """Direct injection and jailbreaks are blocked; PII is replaced before the LLM call."""
        verdict = scan(text, threshold=self.input_threshold)
        if verdict.blocked:
            reasons = tuple(f"injection:{c}" for c in verdict.categories)
            return InputDecision(False, "", Redaction("", {}), reasons)
        redaction = redact(text, extra=self._extra)
        kinds = sorted({placeholder_kind(p) for p in redaction.mapping})
        return InputDecision(
            True, redaction.text, redaction, tuple(f"pii_in_input:{k}" for k in kinds)
        )

    def filter_contexts(self, contexts: Sequence[str]) -> tuple[list[str], list[str]]:
        """Indirect injection: quarantine retrieved chunks that carry instructions."""
        clean: list[str] = []
        flags: list[str] = []
        for chunk in contexts:
            if scan(chunk, threshold=self.doc_threshold).blocked:
                flags.append("injection_in_context")
            else:
                clean.append(chunk)
        return clean, flags

    def check_output(self, answer: str, redaction: Redaction | None) -> OutputDecision:
        """Block system-prompt leaks, redact third-party PII, restore only the user's own."""
        if self.canary in answer:
            return OutputDecision(LEAK_MESSAGE, ("canary_leak",))
        seed = redaction.mapping if redaction is not None else {}
        result = redact(answer, extra=self._extra, seed=seed)
        leaked = sorted({placeholder_kind(p) for p in result.mapping if p not in seed})
        text = result.restore(result.text, keys=seed)
        return OutputDecision(text, tuple(f"pii_in_output:{k}" for k in leaked))
