"""PII detection and reversible redaction: regex plus checksums, with an optional NER hook."""

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(number: str) -> bool:
    """Aadhaar numbers carry a Verhoeff check digit, which weeds out random 12-digit values."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(body: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(body)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


def luhn_valid(number: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


@dataclass(frozen=True)
class PiiMatch:
    kind: str
    start: int
    end: int
    text: str


Recognizer = Callable[[str], list[PiiMatch]]

# Order is priority: when two detectors overlap, the earlier one wins.
_DETECTORS: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    (
        "SSN",
        re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"),
        None,
    ),
    (
        "AADHAAR",
        re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
        lambda value: verhoeff_valid(_digits(value)),
    ),
    (
        "CARD",
        re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"),
        lambda value: luhn_valid(_digits(value)),
    ),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), None),
    (
        "PHONE",
        re.compile(r"(?<!\d)(?:\+1[ -]?)?\(?[2-9]\d{2}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
        None,
    ),
)


def spacy_recognizer(model: str = "en_core_web_sm") -> Recognizer:
    """Optional NER for names. Needs `pip install spacy` plus the model download."""
    spacy = importlib.import_module("spacy")
    nlp = spacy.load(model)

    def recognize(text: str) -> list[PiiMatch]:
        return [
            PiiMatch("PERSON", ent.start_char, ent.end_char, ent.text)
            for ent in nlp(text).ents
            if ent.label_ == "PERSON"
        ]

    return recognize


def find_pii(text: str, extra: Sequence[Recognizer] = ()) -> list[PiiMatch]:
    found: list[tuple[int, PiiMatch]] = []
    for priority, (kind, pattern, check) in enumerate(_DETECTORS):
        for m in pattern.finditer(text):
            if check is None or check(m.group()):
                found.append((priority, PiiMatch(kind, m.start(), m.end(), m.group())))
    for recognizer in extra:
        found.extend((len(_DETECTORS), match) for match in recognizer(text))
    accepted: list[PiiMatch] = []
    for _, match in sorted(found, key=lambda item: (item[0], item[1].start)):
        if all(match.end <= o.start or match.start >= o.end for o in accepted):
            accepted.append(match)
    return sorted(accepted, key=lambda match: match.start)


def placeholder_kind(placeholder: str) -> str:
    return placeholder[1:-1].rsplit("_", 1)[0]


@dataclass(frozen=True)
class Redaction:
    text: str
    mapping: dict[str, str]  # placeholder -> original value

    def restore(self, text: str, keys: Mapping[str, str] | None = None) -> str:
        """Re-hydrate placeholders locally. `keys` limits which placeholders are restored."""
        for placeholder, original in self.mapping.items():
            if keys is None or placeholder in keys:
                text = text.replace(placeholder, original)
        return text


def redact(
    text: str,
    *,
    extra: Sequence[Recognizer] = (),
    seed: Mapping[str, str] | None = None,
) -> Redaction:
    """Replace PII with <KIND_n> placeholders. The same value always gets the same one."""
    mapping: dict[str, str] = dict(seed or {})
    reverse = {value: placeholder for placeholder, value in mapping.items()}
    counters: dict[str, int] = {}
    for seeded in mapping:
        kind, number = seeded[1:-1].rsplit("_", 1)
        counters[kind] = max(counters.get(kind, 0), int(number))
    parts: list[str] = []
    cursor = 0
    for match in find_pii(text, extra):
        existing = reverse.get(match.text)
        if existing is None:
            counters[match.kind] = counters.get(match.kind, 0) + 1
            existing = f"<{match.kind}_{counters[match.kind]}>"
            mapping[existing] = match.text
            reverse[match.text] = existing
        parts.append(text[cursor : match.start])
        parts.append(existing)
        cursor = match.end
    parts.append(text[cursor:])
    return Redaction("".join(parts), mapping)
