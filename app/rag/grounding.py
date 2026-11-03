from app.retrieval.bm25 import tokenize

STOPWORDS = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "did", "do", "does",
    "for", "from", "get", "have", "how", "i", "in", "is", "it", "its", "many", "me",
    "much", "my", "of", "on", "or", "so", "that", "the", "there", "to", "was", "we",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
})  # fmt: skip


def _stem(token: str) -> str:
    """Crude plural stripping so 'refunds' matches 'refund'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_terms(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text) if t not in STOPWORDS]


def support_score(question: str, text: str) -> float:
    """Share of the question's content words that appear in the text (0 to 1).
    This is a lexical proxy for relevance. With a real embedder you would use
    cosine similarity or the reranker score here instead."""
    terms = content_terms(question)
    if not terms:
        return 0.0
    present = set(content_terms(text))
    return sum(1 for term in terms if term in present) / len(terms)
