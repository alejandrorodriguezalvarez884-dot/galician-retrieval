# Analysis plan for the reviewed DOG 2025 test set

Fixed on 2026-11-16, after the reviewed questions came back and before any system was run on them. It sets what is compared, how, and what each outcome lets the paper say.

## The set

108 questions over 94 DOG 2025 documents that are in our index but not in DOG-RAG, sampled with `sample_candidates.py`. The questions were drafted by a language model other than the ones that wrote DOG-RAG's questions (DeepSeek-V3) and our training questions (Salamandra-7b-instruct); the paper names it. A Galician linguist who took no part in building the models then reviewed them one by one with the form built by `build_kit.py`. The drafts were frozen before the review.

Before writing this plan only the format and the revision counts were read, which involve no retrieval: 97 questions unchanged, 11 edited, none added or removed, no type changed. 50 general, 40 data, 14 scattered, 4 table. The set is described as "drafted by a language model, lightly revised by a linguist", not as human-written.

## Systems

The five default systems of `eval_human.py`, on the DOG-RAG index (2,661 documents, 36,318 chunks), top 10 chunks, top 3 kept: `bm25`, `bge-m3`, `nos` (bge-m3 + bge-reranker-v2-m3), `mix` (the released embedder) and `run5`.

## Metric and test

- Primary: MRR@3 at document level, strict (only the annotator's document counts), as DOG-RAG is scored.
- Paired bootstrap over questions, 10,000 resamples, seed 0, two-sided p. The interval that resamples documents is reported as a check.
- Lenient MRR, P@3 and R@3 are reported; no claim rests on them.

## Family E

Six tests, Holm at family-wise 0.05 (`analysis/holm_e.py`):

1. mix − BM25
2. mix − bge-m3
3. mix − bge-m3 + bge-reranker-v2-m3
4. run 5 − BM25
5. run 5 − bge-m3
6. run 5 − bge-m3 + bge-reranker-v2-m3

Sensitivity check: family A of `analysis/holm.py` (the DOG-RAG headline comparisons) and E together. Reported with raw intervals only: mix − run 5 and bge-m3 + bge-reranker-v2-m3 − BM25.

## Wording fixed in advance

- Survives Holm in E: "also better on the reviewed set".
- p < 0.05 but does not survive: "higher, but not significant after correction".
- Otherwise: "no significant difference on the reviewed set", a tie and not a failure to replicate.
- If the ordering contradicts DOG-RAG on a comparison that survives, the abstract says so.

## Descriptive only

Edited vs unchanged questions (11 edited is too few to test), the scores of the frozen drafts, and the breakdowns by category, code in the question, overlap with training and twin documents.

## Added after the plan

The reranker rows (`bm25-bge`, `bm25-ours`, `mix-ours` in `eval_human.py`) were scored after this plan was fixed and are exploratory, outside family E.

## Exclusions

None: all 108 questions are scored.
