"""Training rows for run 5: run 4's rows plus short passages and WebFAQ glg (training side).

Run 4 trained only on ~250-word positives and lost ground on short-passage retrieval (news
paragraphs, WebFAQ answers). This adds two kinds of short rows to run 4's exact rows:

1. Short passages. Every run 4 positive is split into pieces of 40-100 words at sentence
   boundaries. BM25 (IDF over all pieces) scores the pieces of a row's own positive against its
   question; the best piece becomes a short positive only if it matches at least --min-terms
   distinct question words and beats the second piece by --margin (top >= (1 + margin) * second).
   The short negative is the best piece of the row's mined hard negative, which bge-reranker-v2-m3
   already scored below the positive. --short rows are sampled from the eligible ones.
2. WebFAQ v2 glg, training side only: sites where eval_extra.webfaq_heldout is False, with the
   test pool's filters (score, Galician word check, >= 3 question words, question unique in the
   whole file, answer unique), at most --per-site rows per site. Negative: a random answer from
   another site; there is no reranker here to filter BM25 false negatives.

Guards on the new rows: drop a row whose question shares an 8-word n-gram with a DOG-RAG,
nos-rag-news or held-out WebFAQ question, or whose positive shares a 20-word n-gram with a
DOG-RAG context, a nos-rag-news article or a held-out WebFAQ answer. Run 4's rows are kept
as they were (they passed the DOG-RAG guard in mine.py); news hits among them are only counted.

Output: OUT/train.jsonl (train.py reads it with no --cap), OUT/dev.jsonl copied unchanged,
OUT/summary.json and OUT/short_sample.jsonl. CPU only, a few minutes.

    python prep5.py --mined mined2 --out mined5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine import dograg_guard, ngrams, norm_words  # noqa: E402
from train import cap_sources, read_jsonl  # noqa: E402
from translate_sinai import split_sentences  # noqa: E402

WORD = re.compile(r"\w+", re.UNICODE)


def log(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def pieces(text: str, low: int = 40, high: int = 100) -> list[str]:
    """Consecutive sentences packed into pieces of low-high words; a short tail joins the last piece."""
    out, cur, n = [], [], 0
    for s in split_sentences(text, max_words=high):
        w = len(s.split())
        if cur and n + w > high and n >= low:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        if out and (n < low and len(out[-1].split()) + n <= high + 30):
            out[-1] += " " + " ".join(cur)
        elif n >= 20:
            out.append(" ".join(cur))
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg = sum(map(len, docs)) / max(1, n)
        self.k1, self.b = k1, b

    def score(self, query: set[str], doc: list[str]) -> tuple[float, int]:
        tf = Counter(doc)
        norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avg)
        s, hits = 0.0, 0
        for t in sorted(query):  # fixed order: float sums must not depend on PYTHONHASHSEED
            if tf[t]:
                s += self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                hits += 1
        return s, hits


def short_rows(rows: list[dict], args, rng: random.Random) -> tuple[list[dict], Counter]:
    split = [(r, pieces(r["positive"])) for r in rows]
    tok = {}
    for _, ps in split:
        for p in ps:
            tok.setdefault(p, norm_words(p))
    bm25 = BM25(list(tok.values()))
    why = Counter()
    eligible = []
    for r, ps in split:
        if len(ps) < 2:
            why["one_piece"] += 1
            continue
        q = set(norm_words(r["query"]))
        scored = sorted(((*bm25.score(q, tok[p]), p) for p in ps), key=lambda x: -x[0])
        (top, hits, best), second = scored[0], scored[1][0]
        if hits < args.min_terms:
            why["few_terms"] += 1
            continue
        if top < (1 + args.margin) * second:
            why["no_margin"] += 1
            continue
        neg = []
        if r.get("negatives"):
            neg_pieces = pieces(r["negatives"][0]) or [r["negatives"][0]]
            neg = [max(neg_pieces, key=lambda p: bm25.score(q, norm_words(p))[0])]
        eligible.append({
            "id": f"short-{r['id']}",
            "source": f"short-{r['source']}",
            "query": r["query"],
            "positive": best,
            "negatives": neg,
            "pieces": len(ps),
            "margin": round(top / second, 2) if second else None,
        })
    why["eligible"] = len(eligible)
    picked = rng.sample(eligible, min(args.short, len(eligible)))
    return picked, why


def webfaq_rows(args, rng: random.Random) -> tuple[list[dict], list[dict], dict]:
    """Training-side rows, and the held-out test pool (for the guard)."""
    import pyarrow.parquet as pq
    from eval_extra import WEBFAQ_MIN_SCORE, WEBFAQ_REPO, WEBFAQ_REVISION, fetch, is_galician_pair, webfaq_heldout

    path = Path(args.webfaq) if args.webfaq else fetch(WEBFAQ_REPO, "data/glg.parquet", WEBFAQ_REVISION)
    rows = pq.read_table(path).to_pylist()
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    question_count = Counter(norm(r["question"]) for r in rows)

    def clean(r):
        return (
            r["semantic_similarity_score"] >= WEBFAQ_MIN_SCORE
            and len(r["question"].split()) >= 3
            and question_count[norm(r["question"])] == 1
            and is_galician_pair(r["question"], r["answer"])
        )

    heldout = [r for r in rows if webfaq_heldout(r["origin"])]
    pool = [r for r in rows if not webfaq_heldout(r["origin"]) and clean(r)]
    answer_count = Counter(norm(r["answer"]) for r in pool)
    pool = [r for r in pool if answer_count[norm(r["answer"])] == 1]
    by_site = defaultdict(list)
    for r in sorted(pool, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest()):
        by_site[r["origin"]].append(r)
    picked = [r for site in sorted(by_site) for r in by_site[site][: args.per_site]]
    out = []
    for r in picked:
        other = rng.choice(picked)
        while other["origin"] == r["origin"]:
            other = rng.choice(picked)
        out.append({
            "id": f"webfaq-{r['id']}",
            "source": "WebFAQ",
            "query": r["question"].strip(),
            "positive": r["answer"].strip(),
            "negatives": [other["answer"].strip()],
        })
    info = {"rows_all": len(rows), "heldout_rows": len(heldout), "train_side_clean": len(pool), "sites": len(by_site), "picked": len(out)}
    return out, heldout, info


def news_guard(args) -> tuple[set, set]:
    from eval_extra import NEWS_REPO, NEWS_REVISION, fetch

    news = Path(args.news_dir) if args.news_dir else None
    articles = json.loads((news / "nos-rag-dataset_news.json" if news else fetch(NEWS_REPO, "nos-rag-dataset_news.json", NEWS_REVISION)).read_text(encoding="utf-8"))
    items = json.loads((news / "nos-rag-dataset_questions.json" if news else fetch(NEWS_REPO, "nos-rag-dataset_questions.json", NEWS_REVISION)).read_text(encoding="utf-8"))
    q = set().union(*(ngrams(norm_words(i["question"]), 8) for i in items))
    c = set().union(*(ngrams(norm_words(a["text"]), 20) for a in articles))
    return q, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mined", required=True, help="mined2 folder (train.jsonl, dev.jsonl)")
    ap.add_argument("--cap", action="append", default=["SINAI=30000"], metavar="SOURCE=N", help="run 4's caps, applied with train.py's cap_sources")
    ap.add_argument("--short", type=int, default=25000, help="short-passage rows to add")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--min-terms", type=int, default=2)
    ap.add_argument("--per-site", type=int, default=1000, help="WebFAQ rows per site; 0 leaves WebFAQ out")
    ap.add_argument("--webfaq", default=None, help="local glg.parquet (default: download the pinned revision)")
    ap.add_argument("--news-dir", default=None, help="local nos-rag-news files (default: download)")
    ap.add_argument("--dograg-triplets", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: first N rows of train.jsonl")
    ap.add_argument("--expect-sha1", default=None, help="fail unless train.jsonl hashes to this (the Mac run's value)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mined5")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    base = cap_sources(read_jsonl(Path(args.mined) / "train.jsonl", args.limit), args.cap, args.seed)
    log(step="base", rows=len(base), by_source=dict(Counter(r["source"] for r in base)))

    short, why = short_rows(base, args, rng)
    log(step="short", added=len(short), **why)

    web, heldout, web_info = webfaq_rows(args, rng) if args.per_site else ([], [], {})
    log(step="webfaq", **web_info)

    q_grams, c_grams = dograg_guard(Path(args.dograg_triplets) if args.dograg_triplets else None, None)
    nq, nc = news_guard(args)
    q_grams |= nq
    c_grams |= nc
    for r in heldout:
        q_grams |= ngrams(norm_words(r["question"]), 8)
        c_grams |= ngrams(norm_words(r["answer"]), 20)

    def hit(r):
        if ngrams(norm_words(r["query"]), 8) & q_grams:
            return "question"
        if ngrams(norm_words(r["positive"]), 20) & c_grams:
            return "passage"
        return None

    hits = Counter()
    new = []
    for r in short + web:
        h = hit(r)
        if h:
            hits[f"{r['source'].split('-')[0]}_{h}"] += 1
        else:
            new.append(r)
    base_news = sum(1 for r in base if ngrams(norm_words(r["query"]), 8) & nq or ngrams(norm_words(r["positive"]), 20) & nc)
    log(step="guard", dropped=dict(hits), base_rows_hitting_news=base_news)

    rows = base + new
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(Path(args.mined) / "dev.jsonl", out / "dev.jsonl")
    with open(out / "short_sample.jsonl", "w", encoding="utf-8") as f:
        for r in random.Random(args.seed).sample(new, min(100, len(new))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    words = lambda rs: round(sum(len(r["positive"].split()) for r in rs) / max(1, len(rs)), 1)  # noqa: E731
    summary = {
        "rows": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "positive_words_mean": {"base": words(base), "short": words([r for r in new if r["source"].startswith("short-")]), "webfaq": words([r for r in new if r["source"] == "WebFAQ"])},
        "short": dict(why),
        "webfaq": web_info,
        "guard": dict(hits),
        "base_rows_hitting_news": base_news,
        "train_sha1": hashlib.sha1((out / "train.jsonl").read_bytes()).hexdigest()[:12],
        "args": vars(args),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(step="done", rows=len(rows), train_sha1=summary["train_sha1"], minutes=summary["minutes"])
    if args.expect_sha1 and summary["train_sha1"] != args.expect_sha1:
        raise SystemExit(f"train.jsonl hash {summary['train_sha1']} != expected {args.expect_sha1}")


if __name__ == "__main__":
    main()
