from collections.abc import Sequence


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / len(relevant)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
