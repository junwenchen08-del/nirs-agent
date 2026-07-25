# BGE-M3 retrieval baseline v1

- Evaluation date: 2026-07-23
- Case file: `retrieval_eval_cases.bge-m3.v1.json`
- Embedding model: `D:\Models\bge-m3`
- Index version: `nir-papers-bge-m3-v1`
- Corpus: 4 published documents, 377 chunks
- Retrieval depth: 5 chunks per query

## Dataset composition

| Slice | Cases |
|---|---:|
| Single-document retrieval | 20 |
| Cross-document retrieval | 4 |
| No-answer negatives | 4 |
| Chinese | 20 |
| English | 8 |
| Total | 28 |

## Baseline metrics

| Metric | Result |
|---|---:|
| Recall@5 (answerable cases) | 0.9583 |
| MRR (answerable cases) | 0.9583 |
| nDCG@5 (answerable cases) | 0.9692 |
| Top-1 document accuracy | 0.9167 |
| No-hit accuracy | 0.0000 |
| Mean query latency | 2171.7 ms |
| Maximum query latency | 2401.3 ms |

English answerable queries achieved Recall@5, MRR, and nDCG@5 of 1.0.
Chinese answerable queries achieved Recall@5 and MRR of 0.9444, with nDCG@5
of 0.9590.

## Observed failure modes

1. Two Chinese queries about the English meat-scatter paper placed the relevant
   document at rank 2 rather than rank 1.
2. Two cross-document queries retrieved only one of their two labeled relevant
   documents within the first five chunks. Repeated chunks from one document
   reduce document diversity.
3. Every negative query returned at least one semantic neighbor. The current
   retriever always returns top-K chunks and has no calibrated abstention or
   no-answer decision. The four negative cases must remain in the set so a
   later score threshold, reranker, or answerability classifier can be measured.

The fabricated NIR/scatter negative reached a top similarity score of 0.6182,
while unrelated Raman, NMR, and terahertz negatives ranged from 0.4816 to
0.5195. A single uncalibrated similarity threshold is therefore unlikely to be
sufficient on its own.
