"""Training groups for the Galician reranker (phase 3): one question, its positive, up to 7 negatives.

Two stages:

1. candidates (CPU, deterministic, a few minutes). Starts from mine.py's train.jsonl with the
   same SINAI cap as runs 2-5 (--cap SINAI=30000, seed 0), so the questions are the embedder's.
   Negatives per question:
   - the bge-m3 negatives already in the row (ranks 10-50, other documents, already filtered
     with bge-reranker-v2-m3 by mine.py);
   - BM25 negatives: the top --bm25-candidates hits over the pool of every positive and negative
     text, skipping the row's own positive and its bge-m3 negatives. The reranker is judged on
     BM25's top 10, so it has to learn to push down what BM25 likes.
   Writes OUT/candidates.jsonl and OUT/summary.json with a short sha1 of the file.

2. teacher (GPU). bge-reranker-v2-m3 scores the positive and every candidate. A BM25 candidate
   whose probability is not below the positive's by --margin is dropped (it probably answers the
   question too; the same rule mine.py used for the bge-m3 negatives). The first --group - 1
   survivors are kept, bge-m3 ones first, then BM25 ones in rank order. Rows left with no
   negative are dropped. Writes OUT/groups.jsonl ({id, source, query, docs, kinds, teacher});
   docs[0] is the positive and teacher holds the raw logits, used for distillation.

    python prep_rerank.py candidates --mined mined2 --out rr1
    python prep_rerank.py teacher --out rr1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import cap_sources, read_jsonl  # noqa: E402

TEACHER = "BAAI/bge-reranker-v2-m3"
TOKEN_RE = r"(?u)\b\w+\b"  # eval_dog.py's BM25 tokenizer


def log(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def short_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def candidates(args):
    import bm25s
    import numpy as np

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.mined) / "train.jsonl", args.limit)
    rows = cap_sources(rows, args.cap, args.seed)
    rows.sort(key=lambda r: r["id"])
    log(step="rows", rows=len(rows), by_source=dict(Counter(r["source"] for r in rows)))

    pool = sorted({r["positive"] for r in rows} | {n for r in rows for n in r["negatives"]})
    log(step="pool", passages=len(pool))
    tokens = bm25s.tokenize(pool, lower=True, token_pattern=TOKEN_RE, stopwords=None, show_progress=False)
    bm25 = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    bm25.index(tokens, show_progress=False)
    vocab = tokens.vocab
    query_ids = []
    for r in rows:
        ids = [vocab[t] for t in re.findall(TOKEN_RE, r["query"].lower()) if t in vocab]
        query_ids.append(ids or [0])
    depth = args.bm25_candidates + 4  # room for the positive and the bge-m3 negatives
    hits, _ = bm25.retrieve(query_ids, k=depth, show_progress=False, n_threads=args.threads)
    hits = np.asarray(hits)
    log(step="bm25", depth=depth, minutes=round((time.time() - t0) / 60, 1))

    pos_rank = Counter()
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r, row_hits in zip(rows, hits):
            skip = {r["positive"], *r["negatives"]}
            negs = [{"text": n, "kind": "m3"} for n in r["negatives"]]
            rank = None
            for k, j in enumerate(row_hits.tolist()):
                text = pool[j]
                if text == r["positive"]:
                    rank = k + 1
                if text in skip:
                    continue
                if sum(n["kind"] == "bm25" for n in negs) < args.bm25_candidates:
                    negs.append({"text": text, "kind": "bm25", "rank": k + 1})
            pos_rank[rank if rank is not None else "out"] += 1
            f.write(json.dumps({"id": r["id"], "source": r["source"], "query": r["query"], "positive": r["positive"], "negatives": negs}, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "pool": len(pool),
        "bm25_candidates": args.bm25_candidates,
        "positive_bm25_rank": {str(k): v for k, v in sorted(pos_rank.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))},
        "candidates_sha1": short_sha1(out / "candidates.jsonl"),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(step="done", **{k: v for k, v in summary.items() if k != "positive_bm25_rank"})
    if args.expect_sha1 and summary["candidates_sha1"] != args.expect_sha1:
        raise SystemExit(f"candidates.jsonl hash {summary['candidates_sha1']} != expected {args.expect_sha1}")


def teacher(args):
    import torch
    from sentence_transformers import CrossEncoder

    t0 = time.time()
    out = Path(args.out)
    rows = read_jsonl(out / "candidates.jsonl", args.limit)
    cuda = torch.cuda.is_available()
    model = CrossEncoder(
        args.teacher,
        device="cuda" if cuda else "cpu",
        max_length=args.max_length,
        model_kwargs={"dtype": torch.float16} if cuda else None,
    )
    # Every (query, passage) pair once, scored longest first so batches pad little.
    pairs = []
    for i, r in enumerate(rows):
        pairs.append((i, -1, r["query"], r["positive"]))
        for k, n in enumerate(r["negatives"]):
            pairs.append((i, k, r["query"], n["text"]))
    order = sorted(range(len(pairs)), key=lambda p: -len(pairs[p][3]))
    logits = [0.0] * len(pairs)
    step = 20000
    for s in range(0, len(order), step):
        part = order[s : s + step]
        scores = model.predict(
            [(pairs[p][2], pairs[p][3]) for p in part],
            batch_size=args.batch_size,
            activation_fn=torch.nn.Identity(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for p, v in zip(part, scores.tolist()):
            logits[p] = float(v)
        log(step="teacher_progress", scored=min(s + step, len(order)), total=len(order), minutes=round((time.time() - t0) / 60, 1))

    pos = {}
    neg = {}
    for (i, k, _, _), v in zip(pairs, logits):
        if k < 0:
            pos[i] = v
        else:
            neg[(i, k)] = v

    def prob(x):
        return 1 / (1 + math.exp(-x))

    stats = Counter()
    sizes = Counter()
    with open(out / "groups.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            docs, kinds, scores = [r["positive"]], ["pos"], [round(pos[i], 4)]
            for k, n in enumerate(r["negatives"]):
                v = neg[(i, k)]
                if n["kind"] == "bm25" and prob(v) >= prob(pos[i]) - args.margin:
                    stats["bm25_dropped"] += 1
                    continue
                if len(docs) < args.group:
                    docs.append(n["text"])
                    kinds.append(n["kind"])
                    scores.append(round(v, 4))
                    stats[n["kind"] + "_kept"] += 1
            if len(docs) == 1:
                stats["rows_without_negatives"] += 1
                continue
            sizes[len(docs)] += 1
            f.write(json.dumps({"id": r["id"], "source": r["source"], "query": r["query"], "docs": docs, "kinds": kinds, "teacher": scores}, ensure_ascii=False) + "\n")

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["teacher"] = {
        "model": args.teacher,
        "pairs": len(pairs),
        "margin": args.margin,
        "group": args.group,
        "group_sizes": {str(k): v for k, v in sorted(sizes.items())},
        **stats,
        "groups_sha1": short_sha1(out / "groups.jsonl"),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(step="teacher_done", **{k: v for k, v in summary["teacher"].items() if k != "group_sizes"}, group_sizes=summary["teacher"]["group_sizes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--mined", required=True, help="mine.py output folder (train.jsonl)")
    c.add_argument("--out", required=True)
    c.add_argument("--cap", action="append", default=None, metavar="SOURCE=N")
    c.add_argument("--bm25-candidates", type=int, default=8)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--limit", type=int, default=None, help="smoke tests only: first N rows of train.jsonl")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--expect-sha1", default=None)
    t = sub.add_parser("teacher")
    t.add_argument("--out", required=True)
    t.add_argument("--teacher", default=TEACHER)
    t.add_argument("--group", type=int, default=8, help="positive + negatives kept per question")
    t.add_argument("--margin", type=float, default=0.1, help="teacher probability margin below the positive")
    t.add_argument("--max-length", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "candidates":
        if args.cap is None:
            args.cap = ["SINAI=30000"]
        candidates(args)
    else:
        teacher(args)


if __name__ == "__main__":
    main()
