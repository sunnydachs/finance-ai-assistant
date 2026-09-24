# Retrieval Quality Metrics Report (T2)

- Evaluated queries with ground truth source: `25`
- Retrieval engine: Character 2/3-gram BM25 (zero dependency)
- Mean Reciprocal Rank (MRR): `0.903`

## Metrics by k

| Metric | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| Recall@k (%) | 84.0% | 96.0% | 100.0% | 100.0% |
| Precision@k (%) | 84.0% | 32.0% | 20.0% | 10.0% |

## Analysis & Findings

- **Recall Performance**: Recall reaches 100.0% at $k=5$ and plateaus at 100.0% through $k=10$; note refs are credited when their parent note ranks (whole-file indexing).
- **Precision Tradeoff**: As $k$ increases from 1 to 10, precision decreases from 84.0% to 10.0% because more distractor chunks enter the result set. The current `RETRIEVAL_TOP_K = 4` is a measured compromise, not a claim of perfect retrieval.

