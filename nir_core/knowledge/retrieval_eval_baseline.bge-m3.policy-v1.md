# BGE-M3 retrieval baseline with policy v1

Measured against `retrieval_eval_cases.bge-m3.v1.json` on 2026-07-23.
The deployed corpus contained four published documents and 377 chunks.

Policy:

- vector candidate multiplier: 4
- maximum vector candidates: 80
- maximum returned chunks per document: 2
- strong score threshold: 0.62
- weak score threshold: 0.58
- minimum top-document margin in the weak/strong interval: 0.02
- answerability rejection: enabled
- document diversity: enabled

## Results

| Metric | Raw BGE-M3 baseline | Policy v1 |
|---|---:|---:|
| Recall@5 | 0.9583 | **1.0000** |
| MRR | 0.9583 | **0.9583** |
| nDCG@5 | **0.9692** | 0.9659 |
| No-hit accuracy | 0.0000 | **1.0000** |
| Top-1 document accuracy | 0.9167 | **0.9167** |
| Mean latency | 2171.70 ms | **2168.92 ms** |
| Maximum latency | 2401.25 ms | **2310.88 ms** |

Language slices:

| Language | Answerable cases | Negative cases | Recall@5 | MRR | nDCG@5 | No-hit accuracy |
|---|---:|---:|---:|---:|---:|---:|
| English | 6 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Chinese | 18 | 2 | 1.0000 | 0.9444 | 0.9545 | 1.0000 |

## Remaining errors and interpretation

Two Chinese meat-paper queries still rank the relevant document second, so the
policy does not change Top-1 accuracy or MRR. Candidate over-fetching and the
per-document cap recover both documents for every cross-document query,
raising Recall@5 to 1.0. The small nDCG decrease is the expected tradeoff from
placing a diverse document ahead of additional chunks from the top-scoring
document.

All four negative cases now abstain. Three fall below the weak score threshold;
the fabricated near-infrared/quantum query sits between the weak and strong
thresholds but is rejected because the two best documents are separated by
only 0.0025, below the 0.02 document-margin requirement.

These thresholds are calibrated on a small corpus and evaluation set. Add more
real user queries and hard negatives before treating them as portable to a
larger or materially different knowledge base.
