"""Secondary evaluations: nos-rag-news, PAWS-gl and the WebFAQ glg held-out split.

These check that a model tuned on DOG text is still a usable general Galician embedder.
Models, prompts and modes come from eval_dog.py (imported, not copied), so every row here
lines up with a DOG-RAG row.

news    proxectonos/nos-rag-news with the nos-rag-eval press protocol: the 30 test articles plus
        nos-rag-eval's praza_gal_sample.jsonl (2,202 Praza.gal articles from 2022) as distractors,
        split on line breaks ("paragraph" chunking, es_indexing_press.py), top 10 retrieved,
        optionally reranked, top 3 kept. 114 questions, one each. P@3, R@3 and MRR at document
        level and at paragraph level (source_id-paragraph against context_paragraph_indices), as in
        ir-metrics/evaluate_ir_metrics.py. The Praza articles have no line breaks, so each is a
        single passage of ~700 words that embedders truncate at 1,024 tokens (eval_dog's limit).
        Praza articles get nos-rag-eval's fallback id "Praza-<published_on>", including its
        collisions (130 articles share a timestamp), because the document-level metrics dedupe on it.
paws    PAWS-gl test (2,000 pairs, the only split there is) as pair classification: cosine
        similarity of the two sentence embeddings; average precision (main score, as in MTEB),
        and the best accuracy and F1 over all thresholds. The prompted mode puts the query prompt
        on both sides. Embedders only; rerankers are not paraphrase classifiers.
webfaq  WebFAQ v2 glg held-out split (see webfaq_heldout): questions retrieve their answer among
        all held-out answers. nDCG@10, MRR@10, R@1, R@10; rerankers reorder the top 10.

The WebFAQ split is by site, so templated FAQs from one site never sit on both sides. A site is
held out when sha1("webfaq-glg-heldout:<origin>") % 3 == 0, except the two sites that hold
59 % of the rows (seo.computer, xiaomiui.net), which always stay on the training side. Any
script that trains on WebFAQ glg (v1 or v2) must drop every row where webfaq_heldout(origin) is
True:

    from eval_extra import webfaq_heldout
    rows = [r for r in rows if not webfaq_heldout(r["origin"])]

Test pool filters: semantic_similarity_score >= 0.5, the gen_queries.py Galician word check
on question + answer, at least 3 words in the question, questions that appear more than once
anywhere in the file dropped, answers that appear more than once in the pool dropped. At most
100 queries per site; the corpus keeps every answer of the pool.

Output: OUT/results.jsonl, one JSON line per run as in eval_dog.py, with an extra "task" key;
per-query rows in OUT/runs/. Typical runs:

    python eval_extra.py --retrievers bm25,bge-m3 --rerankers bge-reranker-v2-m3 --out extra
    python eval_extra.py --retrievers galician=run1/final --rerankers ce:reranker1/final --out extra
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_dog as dog  # noqa: E402
from gen_queries import ES_ONLY, GL_ONLY  # noqa: E402

NEWS_REPO = "proxectonos/nos-rag-news"
NEWS_REVISION = "b721e0e502dde94a71ad36625a76dac0f1af5c88"
NOS_RAG_EVAL_COMMIT = "a060d9e766a634a455176c61074f01f53019ae8d"
PRAZA_URL = (
    f"https://raw.githubusercontent.com/proxectonos/nos-rag-eval/{NOS_RAG_EVAL_COMMIT}"
    "/datasets/News/Documents/praza_gal_sample.jsonl"
)
PRAZA_SHA1 = "601eb6a21f09a36e3cef93629fbbfcfce1927827"  # git blob sha of that file

PAWS_REPO = "proxectonos/PAWS-gl"
PAWS_REVISION = "9e9e8cc223458f88c2d65da6af39414779ebd8a7"

WEBFAQ_REPO = "PaDaS-Lab/webfaq-v2"
WEBFAQ_REVISION = "e8e1f78975a2475b43cadb9367aa2d1d7f9ccd92"
WEBFAQ_SALT = "webfaq-glg-heldout"
WEBFAQ_MOD = 3
WEBFAQ_TRAIN_ONLY = {"https://seo.computer", "https://xiaomiui.net"}
WEBFAQ_MIN_SCORE = 0.5
WEBFAQ_PER_SITE = 100

TASKS = ("news", "paws", "webfaq")


def webfaq_heldout(origin: str) -> bool:
    """True if every WebFAQ glg row from this site belongs to the test split."""
    origin = origin.strip().rstrip("/").lower()
    if origin in WEBFAQ_TRAIN_ONLY:
        return False
    return int(hashlib.sha1(f"{WEBFAQ_SALT}:{origin}".encode()).hexdigest(), 16) % WEBFAQ_MOD == 0


# ---------------------------------------------------------------- data


def fetch(repo: str, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def paragraphs(text: str) -> list[str]:
    # Same as ElasticSearchProxy.index_article_with_paragraphs in es_indexing_press.py.
    return [p.strip() for p in text.split("\n") if p.strip()]


def praza_path(args) -> Path:
    if args.praza:
        return Path(args.praza)
    path = Path(args.out) / "data" / "praza_gal_sample.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(PRAZA_URL, timeout=120).read()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != PRAZA_SHA1:
            raise SystemExit(f"praza_gal_sample.jsonl changed: {blob}")
        path.write_bytes(data)
    return path


def load_news(args) -> tuple[list[dict], list[dict]]:
    news = json.loads(fetch(NEWS_REPO, "nos-rag-dataset_news.json", NEWS_REVISION).read_text(encoding="utf-8"))
    items = json.loads(fetch(NEWS_REPO, "nos-rag-dataset_questions.json", NEWS_REVISION).read_text(encoding="utf-8"))

    chunks = []
    for n in news:
        chunks += [{"doc_id": n["source_id"], "pos": i, "text": p} for i, p in enumerate(paragraphs(n["text"]))]
    with open(praza_path(args), encoding="utf-8") as f:
        praza = [json.loads(line) for line in f if line.strip()]
    if args.max_distractors is not None:
        random.Random(args.seed).shuffle(praza)
        praza = praza[: args.max_distractors]
    for a in praza:
        did = a.get("source_id") or f"Praza-{a.get('published_on')}"  # PressAdapter.get_source_id
        chunks += [{"doc_id": did, "pos": i, "text": p} for i, p in enumerate(paragraphs(a["text"]))]

    queries = [
        {
            "id": str(q["id"]),
            "text": q["question"],
            "doc_id": q["source_id"],
            "paragraphs": q["context_paragraph_indices"],
            "category": q.get("category") or "none",
        }
        for q in items
    ]
    return chunks, cap_queries(queries, args)


def load_paws(args) -> list[dict]:
    path = fetch(PAWS_REPO, "paws_gl_test.tsv", PAWS_REVISION)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))  # some fields are quoted and span two lines
    pairs = [{"id": r["id"], "s1": r["sentence1"].strip(), "s2": r["sentence2"].strip(), "label": int(r["label"])} for r in rows]
    if args.max_queries is not None:
        pairs = random.Random(args.seed).sample(pairs, min(args.max_queries, len(pairs)))
    return pairs


def is_galician_pair(question: str, answer: str) -> bool:
    words = re.findall(r"\w+", f"{question} {answer}".lower())
    gl = sum(w in GL_ONLY for w in words)
    return gl >= 1 and sum(w in ES_ONLY for w in words) <= gl


def load_webfaq(args) -> tuple[list[dict], list[dict], dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(fetch(WEBFAQ_REPO, "data/glg.parquet", WEBFAQ_REVISION)).to_pylist()
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    question_count = Counter(norm(r["question"]) for r in rows)
    pool = [
        r
        for r in rows
        if webfaq_heldout(r["origin"])
        and r["semantic_similarity_score"] >= WEBFAQ_MIN_SCORE
        and len(r["question"].split()) >= 3
        and question_count[norm(r["question"])] == 1
        and is_galician_pair(r["question"], r["answer"])
    ]
    answer_count = Counter(norm(r["answer"]) for r in pool)
    pool = [r for r in pool if answer_count[norm(r["answer"])] == 1]

    by_site = defaultdict(list)
    for r in sorted(pool, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest()):
        by_site[r["origin"]].append(r)
    picked = [r for site in sorted(by_site) for r in by_site[site][:WEBFAQ_PER_SITE]]

    chunks = [{"doc_id": r["id"], "pos": 0, "text": r["answer"].strip()} for r in pool]
    queries = [{"id": r["id"], "text": r["question"].strip(), "doc_id": r["id"], "category": r["origin"]} for r in picked]
    split = {
        "heldout_sites": len(by_site),
        "heldout_rows_all": sum(webfaq_heldout(r["origin"]) for r in rows),
        "pool": len(pool),
    }
    return chunks, cap_queries(queries, args), split


def cap_queries(queries: list[dict], args) -> list[dict]:
    if args.max_queries is None:
        return queries
    queries = list(queries)
    random.Random(args.seed).shuffle(queries)
    return sorted(queries[: args.max_queries], key=lambda q: q["id"])


# ---------------------------------------------------------------- metrics


def score_news(queries, chunks, first_stage, reranker=None) -> tuple[dict, list[dict]]:
    metrics, rows = dog.evaluate(queries, chunks, first_stage, reranker)
    p_sum = r_sum = mrr_sum = 0.0
    for q, row in zip(queries, rows):
        # Paragraph level as in evaluate_ir_metrics.py: no dedup, recall over the reference paragraphs.
        ref = {f"{q['doc_id']}-{i}" for i in q["paragraphs"]}
        got = [f"{d}-{p}" for d, p in row["final"]]
        hits = [g in ref for g in got]
        p_sum += sum(hits) / len(got) if got else 0.0
        r_sum += sum(hits) / len(ref) if ref else 0.0
        mrr_sum += next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    n = len(queries)
    metrics.update({"P@3_paragraph": round(p_sum / n, 4), "R@3_paragraph": round(r_sum / n, 4), "MRR_paragraph": round(mrr_sum / n, 4)})
    return metrics, rows


def score_webfaq(queries, chunks, first_stage, reranker=None) -> tuple[dict, list[dict]]:
    sums = defaultdict(float)
    rows = []
    for q, cand in zip(queries, first_stage):
        cand = [int(c) for c in cand]
        if reranker is not None:
            s = reranker.scores(q["text"], [chunks[c]["text"] for c in cand])
            cand = [c for _, c in sorted(zip(s, cand), key=lambda x: -x[0])]
        ids = [chunks[c]["doc_id"] for c in cand]
        rank = ids.index(q["doc_id"]) + 1 if q["doc_id"] in ids else None
        sums["nDCG@10"] += 1.0 / np.log2(rank + 1) if rank else 0.0
        sums["MRR@10"] += 1.0 / rank if rank else 0.0
        sums["R@1"] += rank == 1
        sums["R@10"] += rank is not None
        rows.append({"id": q["id"], "rank": rank, "top3": ids[:3]})
    return {k: round(v / len(queries), 4) for k, v in sums.items()}, rows


def pair_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    lab = labels[order]
    tp = np.cumsum(lab)
    k = np.arange(1, len(lab) + 1)
    pos, n = int(lab.sum()), len(lab)
    ap = float((tp / k * lab).sum() / max(pos, 1))
    # Predicting the top k as paraphrases, for every k.
    fp = k - tp
    acc = (tp + (n - pos - fp)) / n
    f1 = 2 * tp / (2 * tp + fp + (pos - tp))
    return {"AP_cosine": round(ap, 4), "accuracy_best": round(float(acc.max()), 4), "F1_best": round(float(f1.max()), 4), "positives": pos, "pairs": n}


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma list of news, paws, webfaq")
    ap.add_argument("--retrievers", default="bm25,bge-m3", help="as in eval_dog.py: bm25, a name from EMBEDDERS, or name=path")
    ap.add_argument("--rerankers", default="none", help="as in eval_dog.py: none, a name from RERANKERS, or ce:path")
    ap.add_argument("--rerank-on", default="bm25,bge-m3", help="first-stage runs (name/mode) the rerankers are applied to")
    ap.add_argument("--modes", default="plain,prompted")
    ap.add_argument("--praza", help="local praza_gal_sample.jsonl (default: download from nos-rag-eval at a fixed commit)")
    ap.add_argument("--max-distractors", type=int, default=None, help="smoke tests only: Praza articles kept")
    ap.add_argument("--max-queries", type=int, default=None, help="smoke tests only: queries (or PAWS pairs) per task")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--devices", default=None, help="comma list to split corpus encoding (default: all GPUs)")
    ap.add_argument("--first-stage-from", default=None, help="folder with first_*.npy from an earlier run; reranks only")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_extra")
    args = ap.parse_args()

    fp16 = not args.no_fp16
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",")]
    elif args.device.startswith("cuda"):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = None
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    results_path = out / "results.jsonl"

    tasks = [t.strip() for t in args.tasks.split(",")]
    retrieval = {}  # task -> dict(chunks, queries, scorer, info)
    for task in tasks:
        if task == "news":
            chunks, queries = load_news(args)
            retrieval[task] = {"chunks": chunks, "queries": queries, "scorer": score_news, "info": {}}
        elif task == "webfaq":
            chunks, queries, split = load_webfaq(args)
            retrieval[task] = {"chunks": chunks, "queries": queries, "scorer": score_webfaq, "info": split}
        elif task != "paws":
            raise SystemExit(f"unknown task {task}")
    for task, t in retrieval.items():
        fingerprint = hashlib.sha1("\n".join(c["text"] for c in t["chunks"]).encode()).hexdigest()[:10]
        t["info"] = {
            "passages": len(t["chunks"]),
            "documents": len({c["doc_id"] for c in t["chunks"]}),
            "queries": len(t["queries"]),
            **t["info"],
            "index_fingerprint": fingerprint,
        }
        t["first"] = {}
        print(json.dumps({"task": task, **t["info"]}))
    pairs = load_paws(args) if "paws" in tasks else []
    paws_info = {"pairs": len(pairs), "index_fingerprint": hashlib.sha1(json.dumps(pairs).encode()).hexdigest()[:10]}
    if pairs:
        print(json.dumps({"task": "paws", **paws_info}))

    def safe(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._@-]+", "_", name)

    def record(task: str, run: str, reranker: str, metrics: dict, rows: list[dict], seconds: float, info: dict):
        line = {"task": task, "retriever": run, "reranker": reranker, **metrics, "seconds": round(seconds, 1), **info}
        with open(out / "runs" / f"{task}__{safe(run)}__{safe(reranker)}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        main_keys = [k for k in ("MRR", "MRR_paragraph", "nDCG@10", "AP_cosine") if k in line]
        print(json.dumps({k: line[k] for k in ["task", "retriever", "reranker", *main_keys]}))

    def failed(task: str, run: str, reranker: str, exc: Exception):
        line = {"task": task, "retriever": run, "reranker": reranker, "error": f"{type(exc).__name__}: {exc}"[:500]}
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(json.dumps(line, ensure_ascii=False))

    def free():
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def first_stage_file(task: str, run: str) -> Path:
        return out / "cache" / f"first_{task}__{run.replace('/', '@')}_{retrieval[task]['info']['index_fingerprint']}.npy"

    def run_first_stage(task: str, run: str, first: np.ndarray, t0: float):
        t = retrieval[task]
        t["first"][run] = first
        np.save(first_stage_file(task, run), first)
        metrics, rows = t["scorer"](t["queries"], t["chunks"], first)
        record(task, run, "none", metrics, rows, time.time() - t0, t["info"])

    if args.first_stage_from:
        for task, t in retrieval.items():
            fp = t["info"]["index_fingerprint"]
            for path in sorted(Path(args.first_stage_from).glob(f"first_{task}__*_{fp}.npy")):
                run = path.name[len(f"first_{task}__") : -len(f"_{fp}.npy")].replace("@", "/")
                t["first"][run] = np.load(path)
            print(json.dumps({"task": task, "first_stages_loaded": sorted(t["first"])}))
        retrievers = []
    else:
        retrievers = dog.parse_retrievers(args.retrievers)

    modes = [m.strip() for m in args.modes.split(",")]
    for name, spec in retrievers:
        if name == "bm25":
            for task, t in retrieval.items():
                t0 = time.time()
                run_first_stage(task, "bm25", dog.bm25_search(t["chunks"], t["queries"], dog.FIRST_STAGE_K), t0)
            continue
        try:
            model = dog.load_embedder(spec, args.device, fp16)
        except Exception as exc:
            failed(",".join(tasks), name, "none", exc)
            continue
        run_modes = [m for m in modes if m == "plain" or spec.get("prompts") != {}] or ["plain"]
        for mode in run_modes:
            run = f"{name}/{mode}"
            prompted = mode == "prompted"
            for task, t in retrieval.items():
                try:
                    t0 = time.time()
                    cache = out / "cache" / f"{task}_{safe(name)}_{mode}_{t['info']['index_fingerprint']}.npy"
                    if cache.exists():
                        c_emb = np.load(cache)
                    else:
                        texts = [c["text"] for c in t["chunks"]]
                        c_emb = dog.encode(model, texts, "document", spec, prompted, args.batch_size, devices)
                        np.save(cache, c_emb)
                    q_emb = dog.encode(model, [q["text"] for q in t["queries"]], "query", spec, prompted, args.batch_size)
                    first = dog.dense_search(q_emb.astype(np.float32), c_emb.astype(np.float32), dog.FIRST_STAGE_K)
                    run_first_stage(task, run, first, t0)
                except Exception as exc:
                    failed(task, run, "none", exc)
            if pairs:
                try:
                    t0 = time.time()
                    # Symmetric task: both sentences go through the query side.
                    e1 = dog.encode(model, [p["s1"] for p in pairs], "query", spec, prompted, args.batch_size)
                    e2 = dog.encode(model, [p["s2"] for p in pairs], "query", spec, prompted, args.batch_size)
                    scores = (e1.astype(np.float32) * e2.astype(np.float32)).sum(axis=1)
                    labels = np.array([p["label"] for p in pairs])
                    rows = [{"id": p["id"], "score": round(float(s), 5), "label": p["label"]} for p, s in zip(pairs, scores)]
                    record("paws", run, "none", pair_metrics(scores, labels), rows, time.time() - t0, paws_info)
                except Exception as exc:
                    failed("paws", run, "none", exc)
        del model
        free()

    rerank_on = [r.strip() for r in args.rerank_on.split(",")]
    for rr in [r.strip() for r in args.rerankers.split(",") if r.strip() != "none"]:
        try:
            reranker = dog.load_reranker(rr, args.device, fp16)
        except Exception as exc:
            failed(",".join(retrieval), "-", rr, exc)
            continue
        for task, t in retrieval.items():
            runs = [k for k in t["first"] for r in rerank_on if k == r or k.startswith(r + "/")]
            for run in dict.fromkeys(runs):
                try:
                    t0 = time.time()
                    metrics, rows = t["scorer"](t["queries"], t["chunks"], t["first"][run], reranker)
                    record(task, run, rr, metrics, rows, time.time() - t0, t["info"])
                except Exception as exc:
                    failed(task, run, rr, exc)
        del reranker
        free()


if __name__ == "__main__":
    main()
