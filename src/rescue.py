"""Add back the training pairs that bge-m3's round trip dropped but BM25 finds (run 2).

mine.py --filter union does the same from scratch, but that means re-encoding ~114k passages
with bge-m3 and rescoring every negative with bge-reranker-v2-m3 (~3.5 h on a T4). Steps 1-2
of mine.py are deterministic, so the rows a finished mining run dropped are exactly the
candidates that are missing from its train.jsonl. This script only works on those:

1. Rebuild the candidates with mine.py's text filters and DOG-RAG guard, and the same passage pool.
2. Rows missing from MINED/train.jsonl whose own passage BM25 ranks in the top --keep-rank of
   the pool are the rescue candidates.
3. bge-reranker-v2-m3 scores each (question, own passage); a row stays only if the score is at
   least --min-pos-score. BM25 alone would bring back pairs like "Que é o Decreto 11/2001?" over
   any passage that cites that decree.
4. Hard negatives for the kept rows come from BM25 ranks --neg-from to --neg-to (never the same
   document), filtered by the reranker margin as in mine.py.

Output: OUT/train.jsonl (MINED/train.jsonl unchanged, then the rescued rows with "rescued": true),
OUT/dev.jsonl (copied from MINED, so runs are compared on the same questions), and
OUT/rescue_sample.jsonl (kept and rejected rows with their scores).

    python rescue.py --mined data2/mined --synth data1/synth data2/synth_parl --sinai data1/sinai_gl --out mined2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import torch

from mine import RERANKER, bm25_search, candidates, log, passage_pool


def score_pairs(reranker, pairs: list[tuple[str, str]], batch_size: int, t0: float, what: str) -> list[float]:
    scores = []
    step = 50_000
    for s in range(0, len(pairs), step):
        scores += [float(x) for x in reranker.predict(pairs[s : s + step], batch_size=batch_size, show_progress_bar=False)]
        log(step="rerank_progress", what=what, scored=min(s + step, len(pairs)), total=len(pairs), minutes=round((time.time() - t0) / 60, 1))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mined", required=True, help="output folder of the mining run to extend (train.jsonl, dev.jsonl)")
    ap.add_argument("--synth", nargs="*", default=[], help="the same gen_queries.py folders that run used")
    ap.add_argument("--sinai", default=None)
    ap.add_argument("--dograg-triplets", default=None)
    ap.add_argument("--dograg-revision", default=None)
    ap.add_argument("--keep-rank", type=int, default=10)
    ap.add_argument("--min-pos-score", type=float, default=0.5, help="reranker probability the own passage must reach")
    ap.add_argument("--neg-from", type=int, default=10)
    ap.add_argument("--neg-to", type=int, default=50)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--reranker", default=RERANKER)
    ap.add_argument("--rerank-batch-size", type=int, default=32)
    ap.add_argument("--rerank-max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: first N pairs per source")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mined2")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mined, out = Path(args.mined), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dev, train = candidates(
        [Path(d) for d in args.synth],
        Path(args.sinai) if args.sinai else None,
        Path(args.dograg_triplets) if args.dograg_triplets else None,
        args.dograg_revision,
        args.limit,
    )
    pool_text, pool_doc = passage_pool(train)

    have = set()
    with open(mined / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            have.add(json.loads(line)["id"])
    ids = {r["id"] for r in train}
    missing = [r for r in train if r["id"] not in have]
    # Every kept id must come from the rebuilt candidates, or the inputs differ from the mining run.
    log(step="candidates", pool=len(pool_text), train=len(train), already_kept=len(have),
        kept_not_rebuilt=len(have - ids), missing=len(missing), dev=len(dev))
    if len(have - ids) > 0.01 * len(have):
        raise SystemExit("more than 1% of MINED/train.jsonl is not in the rebuilt candidates: different inputs?")

    k = min(args.neg_to, len(pool_text))
    top = bm25_search(pool_text, [r["query"] for r in missing], k)
    hits = []
    for r, idx in zip(missing, top):
        idx = [int(c) for c in idx]
        if r["pos_idx"] in idx[: args.keep_rank]:
            r["bm25_rank"] = idx.index(r["pos_idx"])
            r["cands"] = [c for c in idx[args.neg_from :] if c != r["pos_idx"] and pool_doc[c] != r["doc_id"]]
            hits.append(r)
    log(step="bm25_hits", hits=len(hits), by_source=dict(Counter(r["source"] for r in hits)),
        minutes=round((time.time() - t0) / 60, 1))

    from sentence_transformers import CrossEncoder

    fp16 = args.device.startswith("cuda")
    # Sigmoid on purpose: --min-pos-score and --margin are probabilities. bge-reranker-v2-m3 already
    # defaults to it; some cross-encoders (the ms-marco ones) return raw logits otherwise.
    reranker = CrossEncoder(args.reranker, device=args.device, activation_fn=torch.nn.Sigmoid(),
                            model_kwargs={"dtype": torch.float16} if fp16 else None)
    reranker.max_length = args.rerank_max_length

    pos = score_pairs(reranker, [(r["query"], r["positive"]) for r in hits], args.rerank_batch_size, t0, "positives")
    hist = Counter(min(int(s * 10), 9) for s in pos)
    kept, rejected = [], []
    for r, s in zip(hits, pos):
        r["pos_score"] = round(s, 4)
        (kept if s >= args.min_pos_score else rejected).append(r)
    log(step="positive_check", kept=len(kept), rejected=len(rejected),
        score_deciles={f"{b / 10:.1f}": hist[b] for b in range(10)},
        kept_by_source=dict(Counter(r["source"] for r in kept)))

    for r in kept:
        r["sampled"] = rng.sample(r["cands"], min(args.candidates, len(r["cands"])))
    pairs, owners = [], []
    for i, r in enumerate(kept):
        for c in r["sampled"]:
            pairs.append((r["query"], pool_text[c]))
            owners.append((i, c))
    neg = score_pairs(reranker, pairs, args.rerank_batch_size, t0, "negatives") if pairs else []
    bad = {(i, c) for (i, c), s in zip(owners, neg) if s >= kept[i]["pos_score"] - args.margin}
    for i, r in enumerate(kept):
        r["sampled"] = [c for c in r["sampled"] if (i, c) not in bad]
    log(step="rerank_negatives", pairs_scored=len(pairs), negatives_dropped=len(bad),
        minutes=round((time.time() - t0) / 60, 1))

    shutil.copy(mined / "dev.jsonl", out / "dev.jsonl")
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        with open(mined / "train.jsonl", encoding="utf-8") as old:
            shutil.copyfileobj(old, f)
        for r in kept:
            row = {
                "id": r["id"],
                "source": r["source"],
                "query": r["query"],
                "positive": r["positive"],
                "negatives": [pool_text[c] for c in r["sampled"]],
                "pos_rank": args.keep_rank,  # outside bge-m3's top keep-rank, as mine.py records BM25 rescues
                "pos_score": r["pos_score"],
                "bm25_rank": r["bm25_rank"],
                "rescued": True,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "rescue_sample.jsonl", "w", encoding="utf-8") as f:
        for status, group in (("kept", kept), ("rejected", rejected)):
            for r in rng.sample(group, min(300, len(group))):
                f.write(json.dumps({"status": status, "source": r["source"], "pos_score": r["pos_score"],
                                    "bm25_rank": r["bm25_rank"], "query": r["query"], "positive": r["positive"][:300]},
                                   ensure_ascii=False) + "\n")
    log(step="done", train=len(have) + len(kept), rescued=len(kept),
        with_negatives=sum(1 for r in kept if r["sampled"]), minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
