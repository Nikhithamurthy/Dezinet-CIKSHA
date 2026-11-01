import math
import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.tokens import count_tokens

CHUNK_TOKENS = 200
OVERLAP_TOKENS = 30  # 15% of chunk size
SECTION_MAX_TOKENS = 400


def fixed_chars(text: str, size: int = 800) -> list[str]:
    """Naive baseline: cuts through words and sentences. Kept only for comparison."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def recursive_split(
    text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS
) -> list[str]:
    """Paragraph -> line -> sentence -> word, measured in real tokens."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [str(chunk) for chunk in splitter.split_text(text)]


def sliding_window(
    text: str, window_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS
) -> list[str]:
    """Fixed token windows over words, each repeating the tail of the previous one."""
    words = text.split()
    counts = [max(1, count_tokens(" " + word)) for word in words]
    chunks: list[str] = []
    start, n = 0, len(words)
    while start < n:
        total, end = 0, start
        while end < n and (total + counts[end] <= window_tokens or end == start):
            total += counts[end]
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= n:
            break
        new_start, back = end, 0
        while new_start > start + 1 and back + counts[new_start - 1] <= overlap_tokens:
            new_start -= 1
            back += counts[new_start]
        start = new_start
    return chunks


def header_split(text: str) -> list[str]:
    """One chunk per header section (the parsers already split on headers).
    Oversized sections fall back to recursive splitting."""
    if count_tokens(text) <= SECTION_MAX_TOKENS:
        return [text]
    return recursive_split(text)


def _bag_of_words(sentence: str) -> Counter[str]:
    return Counter(w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if len(w) > 2)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(word, 0) for word, count in a.items())
    norm_a = math.sqrt(sum(c * c for c in a.values()))
    norm_b = math.sqrt(sum(c * c for c in b.values()))
    return dot / (norm_a * norm_b)


def semantic_split(
    text: str,
    max_tokens: int = CHUNK_TOKENS,
    min_tokens: int = 40,
    threshold: float = 0.12,
) -> list[str]:
    """Start a new chunk when the next sentence stops resembling the current chunk.
    Uses word-count vectors for now; Task 04 swaps in real embeddings."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_vec: Counter[str] = Counter()
    current_tokens = 0
    for sentence in sentences:
        vec = _bag_of_words(sentence)
        tokens = count_tokens(sentence)
        too_big = current_tokens + tokens > max_tokens
        topic_shift = current_tokens >= min_tokens and _cosine(current_vec, vec) < threshold
        if current and (too_big or topic_shift):
            chunks.append(" ".join(current))
            current, current_vec, current_tokens = [], Counter(), 0
        current.append(sentence)
        current_vec.update(vec)
        current_tokens += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks
