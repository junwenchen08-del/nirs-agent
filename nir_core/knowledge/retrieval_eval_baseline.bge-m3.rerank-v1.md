# BGE-M3 retrieval with BGE reranker v1

Measured against `retrieval_eval_cases.bge-m3.v1.json` on 2026-07-29.
The deployed corpus contained 24 published documents and 805 chunks.
The existing BGE-M3 index was reused without re-embedding.

Configuration:

- dense embedding model: local BGE-M3, 1024 dimensions
- cross-encoder model: local `bge-reranker-v2-m3`
- vector candidate multiplier: 4
- maximum cross-encoder candidates: 12
- pre-rerank document cap: 2 chunks per document
- cross-encoder maximum sequence length: 512 tokens
- cross-encoder batch size: 8
- dense strong score threshold: 0.63
- dense weak score threshold: 0.60
- minimum top-document margin in the weak/strong interval: 0.02

Dense scores remain authoritative for answerability. Cross-encoder scores only
control the order of accepted candidates. Model failures fall back to dense
ordering.

## Versioned evaluation results

| Metric | Reranker v1 |
|---|---:|
| Cases | 28 |
| Answerable cases | 24 |
| Negative cases | 4 |
| Recall@5 | **1.0000** |
| MRR | **0.9514** |
| nDCG@5 | **0.9634** |
| No-hit accuracy | **1.0000** |
| Mean latency | 6919.89 ms |
| Maximum latency | 10891.90 ms |

All 24 answerable cases returned every labeled relevant document within the
first five distinct documents. All four negative cases abstained. The hardest
fabricated NIR/quantum negative scored 0.6218 with a 0.0036 cross-document
margin, so it is below the 0.63 strong threshold and fails the 0.02 weak-band
margin rule. The lowest accepted labeled positive scored 0.6364.

## Expanded-corpus topic checks

Three additional Chinese queries covered nonlinear modeling, calibration
transfer, and wavelength selection across the newly imported papers:

| Metric | Result |
|---|---:|
| Recall@5 | 0.7222 |
| MRR | **1.0000** |
| nDCG@5 | 0.7547 |
| Mean latency | 9332.77 ms |

Every supplemental query placed a labeled relevant document first. Partial
Recall@5 reflects multi-document labels: only five returned documents are
available and the expanded corpus contains additional relevant reviews not
listed in these supplemental labels.

## Latency comparison

On the same eight-query sample (four positives and all four negatives), the
previous 20-candidate/768-token configuration averaged 16352.25 ms. The final
12-candidate/512-token document-diversified configuration averaged 6469.13 ms,
a 60.4% reduction. The production service remained stable during the complete
31-query evaluation.
