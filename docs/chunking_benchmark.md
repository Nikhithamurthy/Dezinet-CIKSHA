# Chunking Benchmark

Files: handbook.md, help_center.html, long_article.md (the last is the handbook text repeated 8 times, so results are indicative only).

| Strategy | Chunks | Avg tokens | Max tokens | Mid-sentence cuts | Time (ms) |
|---|---|---|---|---|---|
| fixed | 23 | 109 | 171 | 61% | 293 |
| recursive | 22 | 118 | 187 | 9% | 9 |
| sliding | 22 | 128 | 201 | 55% | 12 |
| markdown_header | 22 | 118 | 187 | 9% | 7 |
| semantic | 42 | 59 | 170 | 5% | 14 |

## Takeaways
- Fixed-character chunking cut 61% of chunks mid-sentence versus 9% for recursive splitting, so recursive is the default for this project.
- Semantic splitting had the cleanest boundaries (5%) but produced about twice as many chunks, trading storage and embedding cost for coherence.
- Sliding window keeps overlap but splits on words, so 55% of chunks still end mid-sentence.
- The 293 ms for fixed is a one-time tiktoken load, not a real speed difference.
- Limits: the data is small and repetitive; scanned PDFs and PDF tables are not covered yet.