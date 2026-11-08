"""Clean the Galician training pairs and mine hard negatives with bge-m3.

Inputs (any subset):
- --synth DIR ...: gen_queries.py output (DOG / BOP and ParlaMint), shards with a passage and its questions;
- --sinai DIR: translate_sinai.py output (query_gl, positive_gl).

Steps:
1. Text filters: questions that point at "the text" (including the Galician contractions "no texto",
   "deste anuncio"...), demonstratives ("esta convocatoria"), fewer than 4 words, no "?", duplicates,
   and the same question asked about different passages (ambiguous by construction).
2. Guard against the test set: drop pairs whose question shares an 8-word n-gram with a DOG-RAG question,
   or whose passage shares a 20-word n-gram with a DOG-RAG context.
3. Round-trip filter: encode every positive and question with bge-m3 and keep a question only if its own
   passage ranks in the top --keep-rank among all passages. Vague questions ("Cal é o prazo?") and
   questions the passage doesn't answer fall out here. With --filter union, a question also stays
   if BM25 ranks its passage in the top --keep-rank: bge-m3 alone keeps only what bge-m3 already
   solves, and drops the lexical questions (procedure codes, file numbers) where BM25 beats it on DOG.
4. Hard negatives: up to --candidates passages sampled from ranks 10-50, never from the same document,
   scored with bge-reranker-v2-m3 together with the positive; a candidate is kept only if it scores
   below positive - --margin (otherwise it is probably another correct answer).

Output: OUT/train.jsonl ({id, source, query, positive, negatives, pos_rank}) and OUT/dev.jsonl
(the DOG 2024 dev questions, text filters only), plus a stats line per step on stdout.

    python mine.py --synth synth synth_parl --sinai sinai_gl --out mined
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

DOGRAG_REPO = "proxectonos/dog-rag"
EMBEDDER = "BAAI/bge-m3"
RERANKER = "BAAI/bge-reranker-v2-m3"

TEXT_REF = re.compile(
    r"\b(o|no|do|ao|polo|este|neste|deste|nese|dese|a|na|da|á|pola|esta|nesta|desta)\s+"
    r"(texto|documento|fragmento|anuncio|edicto|extracto|pasaxe)\b"
    r"(?!\s+(refundido|consolidado|nacional|único|de identidade))",  # "texto refundido da Lei", "DNI"
    re.IGNORECASE,
)
DEMONSTRATIVE = re.compile(
    r"\b(este|esta|estes|estas|neste|nesta|nestes|nestas|deste|desta|destes|destas|"
    r"ese|esa|eses|esas|nese|nesa|dese|desa|isto|iso)\b",
    re.IGNORECASE,
)
WORD = re.compile(r"\w+", re.UNICODE)


def log(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def norm_words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def text_ok(q: str) -> str | None:
    """None if the question passes, else the reason."""
    q = q.strip()
    if not q.endswith("?"):
        return "no_question_mark"
    if len(norm_words(q)) < 4:
        return "too_short"
    if TEXT_REF.search(q):
        return "text_reference"
    if DEMONSTRATIVE.search(q):
        return "demonstrative"
    return None


# ---------------------------------------------------------------- loading


def load_synth(dirs: list[Path]):
    for d in dirs:
        for shard in sorted(d.glob("shard_*.jsonl")):
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    for i, q in enumerate(row.get("questions") or []):
                        yield {
                            "id": f"{row['id']}-{i}",
                            "source": row["source"],
                            "split": row.get("split", "train"),
                            "doc_id": row["doc_id"],
                            "query": q.strip(),
                            "positive": row["passage"],
                        }


def load_sinai(d: Path):
    for shard in sorted(d.glob("shard_*.jsonl")):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("query_gl") and row.get("positive_gl"):
                    yield {
                        "id": row["id"],
                        "source": "SINAI",
                        "split": "train",
                        "doc_id": row["id"],
                        "query": row["query_gl"].strip(),
                        "positive": row["positive_gl"],
                    }


def dograg_guard(path: Path | None, revision: str | None):
    if path is None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(DOGRAG_REPO, "triplets.json", repo_type="dataset", revision=revision))
    triplets = json.loads(Path(path).read_text(encoding="utf-8"))
    q_grams, c_grams = set(), set()
    for t in triplets:
        for q in t["question"]:
            q_grams |= ngrams(norm_words(q), 8)
        for c in t["context"]:
            c_grams |= ngrams(norm_words(c), 20)
    return q_grams, c_grams


# ---------------------------------------------------------------- encoding and search


def encode(model, texts: list[str], batch_size: int, cache: Path | None) -> torch.Tensor:
    if cache is not None and cache.exists():
        return torch.from_numpy(np.load(cache))
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype(np.float16)
    if cache is not None:
        np.save(cache, emb)
    return torch.from_numpy(emb)


def search(q_emb: torch.Tensor, p_emb: torch.Tensor, k: int, device: str, block: int = 2048):
    """Top-k passage indices and scores for every query (exact inner product)."""
    p = p_emb.to(device)
    idx_out = np.empty((len(q_emb), k), dtype=np.int64)
    score_out = np.empty((len(q_emb), k), dtype=np.float32)
    for s in range(0, len(q_emb), block):
        q = q_emb[s : s + block].to(device)
        scores = q @ p.T
        top = torch.topk(scores.float(), k, dim=1)
        idx_out[s : s + block] = top.indices.cpu().numpy()
        score_out[s : s + block] = top.values.cpu().numpy()
    return idx_out, score_out


def bm25_search(pool_text: list[str], queries: list[str], k: int) -> np.ndarray:
    """Top-k pool indices per query by BM25 (lowercase word tokens, k1=1.2, b=0.75)."""
    import bm25s

    pattern = r"(?u)\b\w+\b"
    corpus = bm25s.tokenize(pool_text, lower=True, token_pattern=pattern, stopwords=None, show_progress=False)
    retriever = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    retriever.index(corpus, show_progress=False)
    vocab = corpus.vocab
    k = min(k, len(pool_text))
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), 4096):
        ids = [[vocab[t] for t in WORD.findall(q.lower()) if t in vocab] or [0] for q in queries[s : s + 4096]]
        idx, _ = retriever.retrieve(ids, k=k, show_progress=False)
        out[s : s + len(ids)] = idx
    return out


def bm25_round_trip(pool_text: list[str], rows: list[dict], keep_rank: int) -> list[bool]:
    """True where BM25 ranks the row's own passage in its top keep_rank."""
    idx = bm25_search(pool_text, [r["query"] for r in rows], keep_rank)
    return [r["pos_idx"] in set(map(int, row)) for r, row in zip(rows, idx)]


def candidates(synth: list[Path], sinai: Path | None, triplets: Path | None, revision: str | None, limit: int | None):
    """Steps 1 and 2: load, text filters and the DOG-RAG guard. Returns (dev, train) rows."""
    rows = list(load_synth(synth))
    if sinai:
        rows += list(load_sinai(sinai))
    if limit:
        per = defaultdict(list)
        for r in rows:
            per[r["source"], r["split"]].append(r)
        rows = [r for group in per.values() for r in group[:limit]]
    log(step="loaded", pairs=len(rows), by_source=dict(Counter(r["source"] for r in rows)), by_split=dict(Counter(r["split"] for r in rows)))

    reasons = Counter()
    seen = set()
    kept = []
    for r in rows:
        why = text_ok(r["query"])
        key = (r["query"].lower(), hashlib.sha1(r["positive"].encode()).hexdigest())
        if why is None and key in seen:
            why = "duplicate"
        if why:
            reasons[why] += 1
            continue
        seen.add(key)
        kept.append(r)
    per_query = defaultdict(set)
    for r in kept:
        per_query[r["query"].lower()].add(r["positive"])
    ambiguous = {q for q, ps in per_query.items() if len(ps) > 1}
    reasons["ambiguous"] = sum(1 for r in kept if r["query"].lower() in ambiguous)
    kept = [r for r in kept if r["query"].lower() not in ambiguous]
    log(step="text_filter", kept=len(kept), dropped=dict(reasons))

    q_grams, c_grams = dograg_guard(triplets, revision)
    guarded, hits = [], Counter()
    for r in kept:
        if ngrams(norm_words(r["query"]), 8) & q_grams:
            hits["question"] += 1
        elif ngrams(norm_words(r["positive"]), 20) & c_grams:
            hits["passage"] += 1
        else:
            guarded.append(r)
    log(step="dograg_guard", kept=len(guarded), dropped=dict(hits))

    return [r for r in guarded if r["split"] == "dev"], [r for r in guarded if r["split"] != "dev"]


def passage_pool(train: list[dict]) -> tuple[list[str], list[str]]:
    """Unique training passages (texts, doc ids); sets r["pos_idx"] on every row."""
    pool_index: dict[str, int] = {}
    pool_text, pool_doc = [], []
    for r in train:
        h = hashlib.sha1(r["positive"].encode()).hexdigest()
        if h not in pool_index:
            pool_index[h] = len(pool_text)
            pool_text.append(r["positive"])
            pool_doc.append(r["doc_id"])
        r["pos_idx"] = pool_index[h]
    return pool_text, pool_doc


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", nargs="*", default=[], help="gen_queries.py output folders")
    ap.add_argument("--sinai", default=None, help="translate_sinai.py output folder")
    ap.add_argument("--dograg-triplets", default=None, help="local triplets.json (default: download)")
    ap.add_argument("--dograg-revision", default=None)
    ap.add_argument("--keep-rank", type=int, default=10, help="round-trip: own passage must rank below this")
    ap.add_argument("--filter", choices=["bge", "union"], default="bge", help="round-trip by bge-m3, or bge-m3 or BM25")
    ap.add_argument("--neg-from", type=int, default=10)
    ap.add_argument("--neg-to", type=int, default=50)
    ap.add_argument("--candidates", type=int, default=3, help="negatives scored with the reranker per query")
    ap.add_argument("--margin", type=float, default=0.1, help="reranker probability margin below the positive")
    ap.add_argument("--no-rerank", action="store_true", help="keep the sampled negatives unscored (smoke tests)")
    ap.add_argument("--max-per-source", type=int, default=None, help="cap kept pairs per source after filtering")
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--reranker", default=RERANKER)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rerank-batch-size", type=int, default=32)
    ap.add_argument("--rerank-max-length", type=int, default=512, help="384 roughly halves the reranker step")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: first N pairs per source")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mined")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1-2. load, text filters, test-set guard
    dev, train = candidates(
        [Path(d) for d in args.synth],
        Path(args.sinai) if args.sinai else None,
        Path(args.dograg_triplets) if args.dograg_triplets else None,
        args.dograg_revision,
        args.limit,
    )
    with open(out / "dev.jsonl", "w", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps({k: r[k] for k in ("id", "source", "doc_id", "query", "positive")}, ensure_ascii=False) + "\n")

    # 3. round-trip filter over the pool of all training passages
    from sentence_transformers import CrossEncoder, SentenceTransformer

    pool_text, pool_doc = passage_pool(train)
    log(step="pool", passages=len(pool_text), queries=len(train))

    fp16 = args.device.startswith("cuda")
    embedder = SentenceTransformer(args.embedder, device=args.device, model_kwargs={"dtype": torch.float16} if fp16 else None)
    embedder.max_seq_length = args.max_seq_length
    tag = hashlib.sha1(("".join(pool_text[:50]) + str(len(pool_text))).encode()).hexdigest()[:8]
    p_emb = encode(embedder, pool_text, args.batch_size, out / f"pool_{tag}.npy")
    log(step="encoded_pool", minutes=round((time.time() - t0) / 60, 1))
    q_emb = encode(embedder, [r["query"] for r in train], args.batch_size * 4, out / f"queries_{tag}.npy")
    log(step="encoded_queries", minutes=round((time.time() - t0) / 60, 1))
    del embedder
    if fp16:
        torch.cuda.empty_cache()

    k = min(args.neg_to, len(pool_text))
    top_idx, _ = search(q_emb, p_emb, k, args.device)
    bm25_hit = bm25_round_trip(pool_text, train, args.keep_rank) if args.filter == "union" else None
    ranked, dropped_rt = [], []
    rank_hist = Counter()
    rescued = Counter()
    for i, (r, idx) in enumerate(zip(train, top_idx)):
        hit = np.nonzero(idx == r["pos_idx"])[0]
        rank = int(hit[0]) if len(hit) else k
        rank_hist[min(rank, args.keep_rank)] += 1
        if rank >= args.keep_rank and bm25_hit is not None and bm25_hit[i]:
            rescued[r["source"]] += 1
            rank = args.keep_rank  # kept by BM25; recorded as "outside bge-m3's top keep-rank"
            r["pos_rank"] = rank
            r["cands"] = [int(c) for c in idx[args.neg_from : k] if c != r["pos_idx"] and pool_doc[c] != r["doc_id"]]
            ranked.append(r)
            continue
        if rank >= args.keep_rank:
            dropped_rt.append({"source": r["source"], "query": r["query"], "rank": rank, "positive": r["positive"][:300]})
        else:
            r["pos_rank"] = rank
            r["cands"] = [
                int(c) for c in idx[args.neg_from : k] if c != r["pos_idx"] and pool_doc[c] != r["doc_id"]
            ]
            ranked.append(r)
    with open(out / "dropped_round_trip_sample.jsonl", "w", encoding="utf-8") as f:
        for r in rng.sample(dropped_rt, min(300, len(dropped_rt))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_src_before = Counter(r["source"] for r in train)
    by_src_after = Counter(r["source"] for r in ranked)
    log(
        step="round_trip",
        kept=len(ranked),
        keep_rate={s: round(by_src_after[s] / by_src_before[s], 3) for s in by_src_before},
        rank_hist={str(k_): v for k_, v in sorted(rank_hist.items())},
        rescued_by_bm25=dict(rescued),
    )

    if args.max_per_source:
        per = defaultdict(list)
        for r in ranked:
            per[r["source"]].append(r)
        ranked = []
        for group in per.values():
            rng.shuffle(group)
            ranked += group[: args.max_per_source]

    # 4. hard negatives
    for r in ranked:
        r["sampled"] = rng.sample(r["cands"], min(args.candidates, len(r["cands"])))
    if not args.no_rerank:
        reranker = CrossEncoder(args.reranker, device=args.device, model_kwargs={"dtype": torch.float16} if fp16 else None)
        reranker.max_length = args.rerank_max_length
        pairs, owners = [], []
        for i, r in enumerate(ranked):
            pairs.append((r["query"], r["positive"]))
            owners.append((i, -1))
            for c in r["sampled"]:
                pairs.append((r["query"], pool_text[c]))
                owners.append((i, c))
        scores = []
        step = 50_000
        for s0 in range(0, len(pairs), step):
            scores += list(reranker.predict(pairs[s0 : s0 + step], batch_size=args.rerank_batch_size, show_progress_bar=False))
            log(step="rerank_progress", scored=min(s0 + step, len(pairs)), total=len(pairs), minutes=round((time.time() - t0) / 60, 1))
        pos_score = {}
        cand_score = defaultdict(dict)
        for (i, c), s in zip(owners, scores):
            if c < 0:
                pos_score[i] = float(s)
            else:
                cand_score[i][c] = float(s)
        dropped = 0
        for i, r in enumerate(ranked):
            keep = [c for c in r["sampled"] if cand_score[i][c] < pos_score[i] - args.margin]
            dropped += len(r["sampled"]) - len(keep)
            r["sampled"] = keep
            r["pos_score"] = round(pos_score[i], 4)
        log(step="rerank_negatives", pairs_scored=len(pairs), negatives_dropped=dropped, minutes=round((time.time() - t0) / 60, 1))

    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in ranked:
            row = {
                "id": r["id"],
                "source": r["source"],
                "query": r["query"],
                "positive": r["positive"],
                "negatives": [pool_text[c] for c in r["sampled"]],
                "pos_rank": r["pos_rank"],
            }
            if "pos_score" in r:
                row["pos_score"] = r["pos_score"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(
        step="done",
        train=len(ranked),
        dev=len(dev),
        by_source=dict(Counter(r["source"] for r in ranked)),
        with_negatives=sum(1 for r in ranked if r["sampled"]),
        minutes=round((time.time() - t0) / 60, 1),
    )


if __name__ == "__main__":
    main()
