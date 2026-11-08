# galician-retrieval

Code and per-query results for a Galician sentence embedder and reranker tuned on administrative text, evaluated on [DOG-RAG](https://huggingface.co/datasets/proxectonos/dog-rag) with the [nos-rag-eval](https://github.com/proxectonos/nos-rag-eval) protocol.

Models:

- Embedder: [alejandro-rodriguez/galician-embedding-me5-base](https://huggingface.co/alejandro-rodriguez/galician-embedding-me5-base) (revision `v0.2-mix`): multilingual-e5-base fine-tuned on synthetic Galician questions and translated legal triplets, then interpolated 0.9 / 0.1 with its base.
- Reranker: [alejandro-rodriguez/galician-reranker-me5-base](https://huggingface.co/alejandro-rodriguez/galician-reranker-me5-base) (revision `v0.1-rerank1`): a multilingual-e5-base cross-encoder trained on the same data, with listwise distillation from bge-reranker-v2-m3.

## Main results

DOG-RAG, 917 queries (501 questions with their paraphrases), document level, top 10 then top 3. Our index is the public DOG 2025 text plus the 122 DOG-RAG source documents (2,661 documents, 36,318 chunks), not the 12,000+ documents of the DOG-RAG paper, so absolute numbers are not comparable with that paper's tables.

| Pipeline | Params | MRR |
|---|---|---|
| BM25 | – | 0.780 |
| bge-m3 | 568M | 0.740 |
| bge-m3 + bge-reranker-v2-m3 | 568M + 568M | 0.807 |
| BM25 + our reranker | 278M | 0.804 |
| Our embedder (mix 0.9) | 278M | 0.833 |
| Our embedder + our reranker | 278M + 278M | 0.816 |

Over three training seeds the embedder scores 0.828–0.832 and is ahead of bge-m3 + bge-reranker-v2-m3 by 2.1–2.5 points in every seed. After a Holm correction across the comparisons we report, that difference is not significant (adjusted p 0.17–0.19). Against bge-m3 on its own the gain is large and holds. On nos-rag-news paragraph retrieval and WebFAQ glg the embedder trails bge-m3. Every number, interval and the correction is reproducible from `results/` with the scripts in `analysis/`.

## Layout

| Path | What |
|---|---|
| `src/` | Data building, training, evaluation and bootstrap. The scripts import each other, so they live in one folder. |
| `analysis/` | Paired bootstraps, seed spread, Holm correction, error analysis, efficiency. Reads `results/`. |
| `human-test/` | Sampling, annotation form and scorer for a second, linguist-reviewed DOG 2025 test set. |
| `kaggle/` | Pushes and follows the seed and ablation kernels with the Kaggle CLI. |
| `results/` | `results.jsonl`, per-query run files and training summaries, organised by where each job ran (`kaggle/`, `gcp/`, `mac/`). |

Per-query files exist for the runs from run 4 on. Runs 1–3 (MrBERT-nos-gl and the first mE5-base run) and the baseline sweep were scored before per-query files were kept, so only their aggregate numbers exist.

## Pipeline

Training data (all on Kaggle T4 or a GCP L4):

1. `gen_queries.py`: synthetic Galician questions with Salamandra-7b-instruct over DOG (up to 2023), the Coruña and Pontevedra provincial bulletins, and ParlaMint-ES-GA (`prep_parlamint.py` prepares the speeches).
2. `translate_sinai.py`: SINAI's Spanish legal-administrative triplets translated to Galician with Nós MT (CTranslate2).
3. `mine.py`: hard negatives and a bge-m3 round-trip filter; `rescue.py` brings back pairs the filter dropped when BM25 still finds their passage. DOG-RAG documents and questions are excluded.
4. `prep5.py`: run 5 only, adds short passages and WebFAQ glg training rows from sites outside the WebFAQ test split.

Models:

- `train.py`: CachedMNRL, batch 256, lr 5e-5, max length 512, dev (454 DOG 2024 questions) every 50 steps, best checkpoint on dev.
- `mix.py`: weight interpolation with the base; the mixing weight is chosen on dev.
- `prep_rerank.py` and `train_reranker.py`: groups of 8 from BM25, teacher scores from bge-reranker-v2-m3, cross-entropy against `alpha · one-hot + (1 − alpha) · softmax(teacher)`.

Evaluation:

- `eval_dog.py`: DOG-RAG with the nos-rag-eval protocol (exact cosine in numpy, bm25s for BM25, optional reciprocal rank fusion and reranking). The docstring lists every difference from nos-rag-eval.
- `eval_extra.py`: nos-rag-news (document and paragraph), PAWS-gl (average precision) and WebFAQ glg (nDCG@10, held-out sites).
- `bootstrap.py`: paired bootstrap that resamples questions with all their paraphrases.

The Kaggle notebooks are generated, not stored: `src/build_notebooks.py` writes them (flags such as `--train3-cells DIR`, `--variance-kernels DIR` and `--rerank-kernel DIR`; see the end of the file), `src/extra_cells.py` writes the secondary evaluation, and `src/gcp_run5.sh` / `src/gcp_rerank1.sh` run the L4 jobs.

## Reproducing the analysis

```bash
pip install -r requirements.txt
python analysis/boot_variance.py > variance.jsonl
python analysis/holm.py --out holm-results.md
python analysis/issuer_confusion.py
```

These need no GPU. `issuer_confusion.py` and `error_analysis.py` download the DOG 2025 corpus and DOG-RAG from the Hub to rebuild the index. Scoring a model again:

```bash
python src/eval_dog.py --retrievers bm25,galician=alejandro-rodriguez/galician-embedding-me5-base \
    --modes plain --out work/eval
```

Local working files (downloads, checkpoints, caches) go under `work/`, which is ignored by git.

## Data and licences

- DOG 2025 corpus and DOG-RAG: Proxecto Nós (CC BY 4.0 and Apache-2.0). DOG-RAG is used only for testing.
- ParlaMint-ES-GA: CC BY 4.0.
- SINAI legal-administrative triplets: CC BY-SA 4.0; the Galician translation keeps that licence.
- Synthetic questions were generated with Salamandra-7b-instruct (Apache-2.0).

The code is MIT-licensed.
