"""Evaluate the retrievers on the second DOG 2025 test set (LLM-drafted, linguist-revised questions).

Reads the reviewed sheet (the CSV saved by the HTML form, or annotation_sheet.csv),
turns every question into a query in src/eval_dog.py's format and scores it on the same
index (2,661 documents, 36,318 chunks, fingerprint fb26bfd556) with the same protocol: top 10
chunks, optionally reranked, top 3 kept, document-level P@3, R@3 and MRR.

Run from the repo root (the paper's numbers: Apple M5, MPS, fp32, as results/mac/pipeline/):

    HF_HOME=work/hf-cache python human-test/eval_human.py \
        --sheet path/to/preguntas_dog_DATE.csv

Systems (--systems, default all):
  bm25      bm25s, k1 1.2, b 0.75, as in eval_dog.py
  bge-m3    BAAI/bge-m3, plain encode (no prompts), as nos-rag-eval runs it
  nos       bge-m3 top 10 reranked by BAAI/bge-reranker-v2-m3: Nós's pipeline
  mix       mix 0.9 (release candidate), work/kaggle/mix/mix/final
  run5      run 5, work/gcp/run5/run5/final

Chunk embeddings are read from --cache-dir (eval_dog.py's file names, e.g.
galician-mix_plain_fb26bfd556.npy) and computed and saved there when missing; bge-m3 has no
cache yet and takes a while on the first run. --model NAME=PATH replaces a model, e.g. to
smoke-test the bge-m3 code paths with a local checkpoint.

Outputs in --out: queries.jsonl (what was scored), runs/<system>.jsonl (per-query top 3 and
MRR), results.json (metrics, breakdowns, bootstraps) and a printed summary.

Two scores per system:
  strict   the annotator's document only (as DOG-RAG scores)
  lenient  the document or its twin in the index (an EXTRACTO and the full order or
           resolution it summarises answer the same questions)

Bootstraps are paired and resample questions (each question is its own unit; DOG-RAG's
paraphrases don't exist here). A second interval resamples documents, keeping both
questions about a document together, as a check that the two questions per document don't
narrow the interval.

With --drafts (default drafts_llm.csv next to this file, when it exists and isn't the sheet itself)
every question is matched to the frozen LLM draft in the same slot (candidate, question 1 or 2) and
labelled unchanged, edited or new; results.json gets the revision counts, a word-level similarity
between draft and final question, type changes, and MRR by label. To score the drafts themselves,
pass --sheet drafts_llm.csv with another --out.

--dograg scores DOG-RAG's own 917 queries through this same code instead of the sheet, to check
that the harness reproduces the logged numbers (BM25 0.7803, mix 0.8328, run 5 0.836).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import gc
import hashlib
import io
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import eval_dog  # noqa: E402
from bootstrap import paired_bootstrap  # noqa: E402

FINGERPRINT = "fb26bfd556"
DOG2025 = ("work/hf-cache/hub/datasets--proxectonos--corpus_dominio_legal_administrativo_tmp/snapshots/"
           "829ca66f2718b58a34b62e62ee600707a8380cc3/XuntaGalicia_DOG/gl/2025_gl.jsonl")
DOGRAG_DIR = "work/hf-cache/hub/datasets--proxectonos--dog-rag/snapshots/dca1fa77e7b74d498d7deadc08905461456773fa"

MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "galician-mix": "work/kaggle/mix/mix/final",
    "galician-run5": "work/gcp/run5/run5/final",
}
# system -> (first-stage embedder or "bm25", reranker or None)
SYSTEMS = {
    "bm25": ("bm25", None),
    "bge-m3": ("bge-m3", None),
    "nos": ("bge-m3", "bge-reranker-v2-m3"),
    "mix": ("galician-mix", None),
    "run5": ("galician-run5", None),
}
PAIRS = [("mix", "bm25"), ("mix", "bge-m3"), ("mix", "nos"), ("mix", "run5"),
         ("run5", "bm25"), ("run5", "bge-m3"), ("run5", "nos"), ("nos", "bm25")]
# DOG-RAG MRR of the same systems on the same index (results/).
DOGRAG_MRR = {"bm25": 0.780, "bge-m3": 0.740, "nos": 0.807, "mix": 0.833, "run5": 0.836}

# Galician labels in the sheet -> DOG-RAG categories.
CATEGORY = {
    "xeral": "general", "general": "general",
    "dato": "data_extracted", "datos": "data_extracted", "data_extracted": "data_extracted",
    "taboa": "table", "table": "table",
    "disperso": "sparsed_context", "dispersa": "sparsed_context", "sparsed_context": "sparsed_context",
}
CODE = re.compile(r"\b[A-Z]{1,4}\d{2,}[A-Z]?\b|\b\d+/\d{2,4}\b|\b\d{4}-\d+\b")  # as error_analysis.py
WORD_RE = re.compile(r"\w+")


# ---------------------------------------------------------------- sheet


def plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def read_sheet(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # a spreadsheet that saved in Windows-1252
        text = raw.decode("cp1252")
    header = text.split("\n", 1)[0]
    delimiter = max(";,\t", key=header.count)  # the sheet uses ";"; a spreadsheet may resave with "," or tabs
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    return [{plain(k or ""): (v or "").strip() for k, v in r.items()} for r in rows]


def sheet_queries(rows: list[dict]) -> tuple[list[dict], list[str]]:
    queries, problems = [], []
    for r in rows:
        cand, did = r.get("cand", ""), r.get("doc_id", "")
        for k in (1, 2):
            text = r.get(f"pregunta_{k}", "")
            if not text:
                continue
            label = plain(r.get(f"tipo_{k}", ""))
            category = CATEGORY.get(label, "unlabelled")
            if label and category == "unlabelled":
                problems.append(f"{cand} question {k}: unknown type '{label}', scored as unlabelled")
            queries.append({"id": f"{cand}q{k}", "cand": cand, "text": " ".join(text.split()),
                            "doc_id": did, "category": category})
    return queries, problems


def revision(queries: list[dict], draft_rows: list[dict]) -> dict:
    """Label each question against the draft in the same slot and summarise the review."""
    drafts = {}
    for r in draft_rows:
        for k in (1, 2):
            if r.get(f"pregunta_{k}"):
                drafts[f"{r.get('cand', '')}q{k}"] = (" ".join(r[f"pregunta_{k}"].split()), plain(r.get(f"tipo_{k}", "")))
    sims, type_changed = [], 0
    for q in queries:
        d = drafts.get(q["id"])
        if d is None:
            q["origin"] = "new"
            continue
        sim = difflib.SequenceMatcher(None, d[0].lower().split(), q["text"].lower().split()).ratio()
        q["origin"] = "unchanged" if d[0] == q["text"] else "edited"
        q["draft_similarity"] = round(sim, 3)
        sims.append(sim)
        type_changed += CATEGORY.get(d[1], "unlabelled") != q["category"]
    kept = {q["id"] for q in queries}
    return {
        "drafted": len(drafts),
        "unchanged": sum(q["origin"] == "unchanged" for q in queries),
        "edited": sum(q["origin"] == "edited" for q in queries),
        "new": sum(q["origin"] == "new" for q in queries),
        "removed": sum(i not in kept for i in drafts),
        "type_changed": type_changed,
        "word_similarity_median": round(float(np.median(sims)), 3) if sims else None,
        "word_similarity_mean": round(float(np.mean(sims)), 3) if sims else None,
    }


def longest_shared_run(question: str, doc: str) -> int:
    """Longest run of consecutive words the question shares with the document (verbatim copying)."""
    q = WORD_RE.findall(question.lower())
    d = " " + " ".join(WORD_RE.findall(doc.lower())) + " "
    best = 0
    for i in range(len(q)):
        for j in range(i + best + 1, len(q) + 1):
            if " " + " ".join(q[i:j]) + " " in d:
                best = j - i
            else:
                break
    return best


# ---------------------------------------------------------------- retrieval


def dense_first_stage(name: str, path: str, chunks: list[dict], queries: list[dict], args) -> np.ndarray:
    spec = {"model": path, "prompts": {}}
    model = eval_dog.load_embedder(spec, args.device, fp16=False)
    cache = Path(args.cache_dir) / f"{name}_plain_{FINGERPRINT}.npy"
    if cache.exists():
        c_emb = np.load(cache)
        if len(c_emb) != len(chunks):
            raise SystemExit(f"{cache} has {len(c_emb)} rows, the index has {len(chunks)} chunks")
    else:
        print(f"  encoding {len(chunks)} chunks with {name} (one-off; saved to {cache})", flush=True)
        c_emb = eval_dog.encode(model, [c["text"] for c in chunks], "document", spec, False, args.batch_size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, c_emb)
    q_emb = eval_dog.encode(model, [q["text"] for q in queries], "query", spec, False, args.batch_size)
    del model
    free(args.device)
    return eval_dog.dense_search(q_emb.astype(np.float32), c_emb.astype(np.float32), eval_dog.FIRST_STAGE_K)


def free(device: str):
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def lenient_mrr(final_docs: list[str], gold: set[str]) -> float:
    return next((1.0 / (i + 1) for i, d in enumerate(final_docs) if d in gold), 0.0)


# ---------------------------------------------------------------- reporting


def mean(xs) -> float:
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else float("nan")


def breakdown(queries, rows, key) -> dict:
    groups = defaultdict(list)
    for q, r in zip(queries, rows):
        groups[key(q)].append(r["mrr"])
    return {g: {"n": len(v), "MRR": mean(v)} for g, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", type=Path, help="filled annotation sheet (CSV)")
    ap.add_argument("--candidates", type=Path, default=HERE / "candidates.jsonl")
    ap.add_argument("--drafts", type=Path, default=HERE / "drafts_llm.csv",
                    help="frozen LLM drafts to compare the sheet with (skipped if missing or same file)")
    ap.add_argument("--dograg", action="store_true", help="score DOG-RAG's queries instead (harness check)")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--model", action="append", default=[], help="NAME=PATH, replaces a model in MODELS")
    ap.add_argument("--cache-dir", default="work/mac/pipeline/cache")
    ap.add_argument("--dog2025", default=DOG2025)
    ap.add_argument("--dograg-dir", default=DOGRAG_DIR)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("work/human-test/eval"))
    args = ap.parse_args()
    if not args.sheet and not args.dograg:
        ap.error("--sheet or --dograg is required")
    models = dict(MODELS)
    for item in args.model:
        name, path = item.split("=", 1)
        if name not in models:
            ap.error(f"--model: unknown name {name}; known: {', '.join(models)}")
        models[name] = path
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in SYSTEMS:
            ap.error(f"unknown system {s}")

    chunks, dograg_queries = eval_dog.load_inputs(SimpleNamespace(
        dog2025=args.dog2025, dograg_dir=args.dograg_dir, corpus_revision=None, dograg_revision=None,
        max_distractors=None, max_queries=None, seed=0))
    fingerprint = hashlib.sha1("".join(c["text"] for c in chunks).encode()).hexdigest()[:10]
    if fingerprint != FINGERPRINT:
        raise SystemExit(f"index fingerprint {fingerprint} != {FINGERPRINT}")

    candidates = {}
    if args.candidates.exists():
        for line in open(args.candidates, encoding="utf-8"):
            row = json.loads(line)
            candidates[row["doc_id"]] = row

    problems: list[str] = []
    if args.dograg:
        queries = dograg_queries
        unit = lambda q: q["id"].split("_")[0]  # noqa: E731  paraphrases of one question stay together
    else:
        queries, problems = sheet_queries(read_sheet(args.sheet))
        unit = lambda q: q["id"]  # noqa: E731
        index_docs = {c["doc_id"] for c in chunks}
        for q in queries:
            if q["doc_id"] not in index_docs:
                raise SystemExit(f"{q['id']}: document {q['doc_id']!r} is not in the index")
            if q["doc_id"] not in candidates:
                problems.append(f"{q['id']}: {q['doc_id']} is not one of the sampled candidates")
        full_text = {}
        for line in open(args.dog2025, encoding="utf-8"):
            d = json.loads(line)
            full_text[eval_dog.doc_id_from_url(d["url"])] = d["text"]
        for q in queries:
            q["has_code"] = bool(CODE.search(q["text"]))
            q["verbatim_run"] = longest_shared_run(q["text"], full_text.get(q["doc_id"], ""))
            q["twins"] = candidates.get(q["doc_id"], {}).get("twins", [])
            q["train_overlap"] = candidates.get(q["doc_id"], {}).get("train_overlap")
        review = None
        if args.drafts.exists() and args.drafts.resolve() != args.sheet.resolve():
            review = revision(queries, read_sheet(args.drafts))
            print("review of the drafts:", json.dumps(review))
    if not queries:
        raise SystemExit("no questions found in the sheet")
    for p in problems:
        print("warning:", p)

    out = args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    with open(out / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    info = {"index_fingerprint": fingerprint, "chunks": len(chunks), "queries": len(queries),
            "units": len({unit(q) for q in queries}), "documents": len({q["doc_id"] for q in queries}),
            "device": args.device, "models": {k: v for k, v in models.items()}}
    print(json.dumps(info))

    first: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    per_query: dict[str, list[dict]] = {}
    for system in systems:
        stage, rr = SYSTEMS[system]
        t0 = time.time()
        if stage not in first:
            if stage == "bm25":
                first[stage] = eval_dog.bm25_search(chunks, queries, eval_dog.FIRST_STAGE_K)
            else:
                first[stage] = dense_first_stage(stage, models[stage], chunks, queries, args)
        reranker = eval_dog.load_reranker("ce:" + models[rr], args.device, False) if rr else None
        metrics, rows = eval_dog.evaluate(queries, chunks, first[stage], reranker)
        del reranker
        free(args.device)
        for q, r in zip(queries, rows):
            r["doc_id"] = q["doc_id"]
            gold = {q["doc_id"], *q.get("twins", [])}
            r["mrr_lenient"] = lenient_mrr([d for d, _ in r["final"]], gold)
        metrics["MRR_lenient"] = mean(r["mrr_lenient"] for r in rows)
        metrics["seconds"] = round(time.time() - t0, 1)
        results[system] = metrics
        per_query[system] = rows
        with open(out / "runs" / f"{system}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"system": system, **{k: metrics[k] for k in ("P@3", "R@3", "MRR", "MRR_lenient")},
                          "seconds": metrics["seconds"]}), flush=True)

    report = {"info": info, "problems": problems, "revision": None if args.dograg else review, "metrics": results, "breakdowns": {}, "bootstrap": []}
    for system, rows in per_query.items():
        b = {"category": breakdown(queries, rows, lambda q: q["category"])}
        if not args.dograg:
            b["has_code"] = breakdown(queries, rows, lambda q: "code" if q["has_code"] else "no code")
            b["train_overlap"] = breakdown(queries, rows, lambda q: "none" if not q["train_overlap"] else
                                           ("< 10 %" if q["train_overlap"] < 0.1 else ">= 10 %"))
            b["twin_in_index"] = breakdown(queries, rows, lambda q: "has twin" if q["twins"] else "no twin")
            if review:
                b["origin"] = breakdown(queries, rows, lambda q: q["origin"])
        report["breakdowns"][system] = b

    unit_ids = [unit(q) for q in queries]
    doc_ids = [q["doc_id"].replace("_", "-") for q in queries]
    for a, b in PAIRS:
        if a not in per_query or b not in per_query:
            continue
        for metric in ("mrr", "mrr_lenient"):
            xa = np.array([r[metric] for r in per_query[a]])
            xb = np.array([r[metric] for r in per_query[b]])
            # paired_bootstrap groups ids on the text before the first "_": make every unit one group.
            res = paired_bootstrap([u.replace("_", "-") for u in unit_ids], xa, xb, args.samples, args.seed)
            line = {"a": a, "b": b, "metric": metric, **res}
            if not args.dograg:
                by_doc = paired_bootstrap(doc_ids, xa, xb, args.samples, args.seed)
                line["by_document"] = {"ci95": by_doc["ci95"], "p_two_sided": by_doc["p_two_sided"]}
            report["bootstrap"].append(line)

    if not args.dograg:
        runs = [q["verbatim_run"] for q in queries]
        report["questions"] = {
            "by_category": Counter(q["category"] for q in queries),
            "with_code": sum(q["has_code"] for q in queries),
            "words_median": float(np.median([len(q["text"].split()) for q in queries])),
            "verbatim_run_median": float(np.median(runs)),
            "verbatim_run_ge_8": sum(r >= 8 for r in runs),
        }
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print("\nsystem    MRR    lenient  P@3    R@3    (DOG-RAG MRR, same index)")
    for s, m in results.items():
        print(f"{s:8s}  {m['MRR']:.3f}  {m['MRR_lenient']:.3f}    {m['P@3']:.3f}  {m['R@3']:.3f}  ({DOGRAG_MRR[s]:.3f})")
    print("\nA - B (MRR, strict)  diff    95 % interval        p      better/worse")
    for line in report["bootstrap"]:
        if line["metric"] == "mrr":
            ci = line["ci95"]
            print(f"{line['a']:5s} - {line['b']:7s}     {line['diff']:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
                  f"{line['p_two_sided']:.3f}  {line['better']}/{line['worse']}")
    if not args.dograg:
        print("\nquestions:", json.dumps(report["questions"], ensure_ascii=False))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
