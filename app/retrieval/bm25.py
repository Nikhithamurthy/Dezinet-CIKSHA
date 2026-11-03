import math
import re
from collections import Counter
from collections.abc import Sequence

# Keeps identifiers whole: "ERR-5034", "DZ-TEAM-25" and "9.4" are single tokens.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over (doc_id, text) pairs. Exact token matches only."""

    def __init__(self, docs: Sequence[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._ids = [doc_id for doc_id, _ in docs]
        self._tf = [Counter(tokenize(text)) for _, text in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len) if self._len else 0.0
        self._k1 = k1
        self._b = b
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._ids)
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        scored: list[tuple[str, float]] = []
        for index, doc_id in enumerate(self._ids):
            tf = self._tf[index]
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                length_norm = 1 - self._b + self._b * self._len[index] / self._avg
                norm = freq + self._k1 * length_norm
                score += self._idf[term] * freq * (self._k1 + 1) / norm
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: -item[1])
        return scored[:k]
