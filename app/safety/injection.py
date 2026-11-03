"""Prompt-injection and jailbreak detection for user input and retrieved documents.

Rules are weighted; the risk is a noisy-or over the rules that fired. The same scanner runs
on user input (direct injection) and on retrieved chunks (indirect injection) with different
thresholds. Pattern matching is one layer, not a guarantee: combine it with the output guard."""

import re
import unicodedata
from dataclasses import dataclass

_HIDDEN = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_I = re.IGNORECASE

_RULES: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        0.9,
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|all|any|your)\b[^.\n]{0,30}"
            r"\b(?:instructions?|rules?|prompts?|guidelines?|directions?)\b",
            _I,
        ),
    ),
    (
        "instruction_override_weak",
        0.5,
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(?:instructions?|rules?|prompts?|guidelines?)\b",
            _I,
        ),
    ),
    (
        "prompt_exfiltration",
        0.9,
        re.compile(
            r"\b(?:reveal|show|print|output|repeat|display|leak|tell me)\b[^.\n]{0,40}"
            r"\b(?:system|hidden|initial|original)\s+(?:prompt|instructions?|message)\b",
            _I,
        ),
    ),
    (
        "prompt_probe",
        0.6,
        re.compile(
            r"\bwhat (?:is|are) your (?:system |initial |hidden )?(?:prompt|instructions)\b",
            _I,
        ),
    ),
    ("jailbreak_persona", 0.8, re.compile(r"\bDAN\b")),
    ("jailbreak_persona_phrase", 0.8, re.compile(r"\bdo anything now\b", _I)),
    (
        "jailbreak_mode",
        0.8,
        re.compile(r"\b(?:developer|god|jailbreak|unrestricted|sudo|no[- ]?filter)\s+mode\b", _I),
    ),
    (
        "restriction_removal",
        0.8,
        re.compile(
            r"\b(?:pretend|act|behave)\b[^.\n]{0,60}"
            r"\b(?:no|without|free of)\s+(?:any\s+)?"
            r"(?:rules|restrictions|limits|limitations|filters|guidelines)\b",
            _I,
        ),
    ),
    ("role_hijack", 0.4, re.compile(r"\byou are (?:now|no longer)\b", _I)),
    (
        "fake_role_tag",
        0.8,
        re.compile(r"<\|?(?:im_start|im_end|system|endoftext)\|?>|\[/?INST\]|<<SYS>>", _I),
    ),
    ("fake_turn", 0.4, re.compile(r"(?:^|\n)\s*(?:system|assistant)\s*:", _I)),
    (
        "exfiltration",
        0.6,
        re.compile(
            r"\b(?:send|forward|email|mail|post|upload|exfiltrate|leak)\b[^.\n]{0,60}"
            r"\b(?:to|at)\b[^.\n]{0,40}"
            r"(?:https?://|@|\b[\w-]+\.(?:com|net|org|io|ru|xyz|info)\b)",
            _I,
        ),
    ),
)
_HIDDEN_TEXT_RISK = 0.3


@dataclass(frozen=True)
class InjectionVerdict:
    blocked: bool
    risk: float
    categories: tuple[str, ...]


def _normalize(text: str) -> str:
    """Undo cheap obfuscation: width tricks, zero-width characters, runs of spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = _HIDDEN.sub("", text)
    return re.sub(r"[ \t]+", " ", text)


def scan(text: str, *, threshold: float = 0.7) -> InjectionVerdict:
    cleaned = _normalize(text)
    weights: dict[str, float] = {}
    for category, weight, pattern in _RULES:
        if pattern.search(cleaned):
            weights[category] = weight
    if _HIDDEN.search(text):
        weights["hidden_text"] = _HIDDEN_TEXT_RISK
    survive = 1.0
    for weight in weights.values():
        survive *= 1.0 - weight
    risk = round(1.0 - survive, 3)
    return InjectionVerdict(risk >= threshold, risk, tuple(sorted(weights)))
