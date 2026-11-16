# Reviewed DOG 2025 test set

A second test set for Galician retrieval over the Diario Oficial de Galicia, on the same index as DOG-RAG (2,661 documents of DOG 2025, 36,318 chunks).

| File | What |
|---|---|
| `questions_reviewed.csv` | The test set: 108 questions over 94 documents, after review by a Galician linguist. One row per candidate document (120), up to two questions each, with the question type (`xeral`, `dato`, `taboa`, `disperso`) and a note giving the answer and where it is in the document. Semicolon-separated, UTF-8. |
| `drafts_llm.csv` | The questions as drafted by a language model, frozen before the review. Same slots as the reviewed file, so each final question can be compared with its draft. |
| `candidates.jsonl` | The 120 sampled documents: id, DOG link, date, issuer, type, title, length, table lines, overlap with the training passages, twin documents in the index and the most similar siblings. |
| `sample_candidates.py` | How the candidates were sampled. |
| `build_kit.py` | The annotation form the reviewer used. |
| `eval_human.py` | Scores the systems on the set. |
| `analysis-plan.md` | Comparisons and correction fixed before scoring. |

The review kept 97 questions as drafted and edited 11; none were added or removed. Results are in `../results/human-test/`.

To score a model on it:

```bash
HF_HOME=work/hf-cache python human-test/eval_human.py --sheet human-test/questions_reviewed.csv \
    --systems bm25,mix --out work/human-test/eval
```

## Licence

The questions and notes in `questions_reviewed.csv` and `drafts_llm.csv` are released under CC BY 4.0. The titles, excerpts and links to DOG documents are public-sector information from the Xunta de Galicia; the document text comes from Proxecto Nós's legal-administrative corpus (CC BY 4.0). Some documents name people (appointments, grants), as published in the DOG.
