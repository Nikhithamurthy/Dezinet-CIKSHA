from collections.abc import Sequence

import numpy as np


def l2_norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def l2_normalize(v: Sequence[float]) -> list[float]:
    """Scale a vector to unit length (L2 norm = 1)."""
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return [float(x) for x in arr]
    return [float(x) for x in arr / norm]


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """sum(a_i * b_i). Sensitive to vector magnitude."""
    return float(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """dot(a, b) / (|a| * |b|). Measures only the angle between vectors."""
    na, nb = l2_norm(a), l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot_product(a, b) / (na * nb)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """L2 distance. Smaller means closer."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.linalg.norm(diff))


def truncate_and_normalize(v: Sequence[float], dims: int) -> list[float]:
    """Matryoshka-style truncation: keep the first `dims` values, then re-normalize."""
    if dims < 1 or dims > len(v):
        raise ValueError(f"dims must be between 1 and {len(v)}")
    return l2_normalize(list(v)[:dims])
