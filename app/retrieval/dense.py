from collections.abc import Sequence

from app.embeddings.metrics import dot_product, l2_normalize


class DenseIndex:
    """Cosine search. Vectors are normalized once, so dot product equals cosine."""

    def __init__(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        self._ids = list(ids)
        self._vectors = [l2_normalize(v) for v in vectors]

    def search(self, query: Sequence[float], k: int = 10) -> list[tuple[str, float]]:
        unit = l2_normalize(query)
        scored = [
            (doc_id, dot_product(unit, vec))
            for doc_id, vec in zip(self._ids, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: -item[1])
        return scored[:k]
