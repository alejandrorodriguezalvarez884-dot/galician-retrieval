"""Sample the candidate DOG 2025 documents for the second test set.

Run from the repo root:

    HF_HOME=work/hf-cache python human-test/sample_candidates.py

Writes candidates.jsonl next to this file (one line per candidate, in the order the
annotator sees them). build_kit.py turns it into the annotation sheet and the form.

What it does:
- Rebuilds the DOG-RAG index exactly as src/eval_dog.py does and checks its fingerprint
  (fb26bfd556), so every candidate is a document of the index the models are scored on.
- Excludes the 122 DOG-RAG source documents and their twins (an EXTRACTO and the full
  order or resolution it summarises share the title), corrections of errors, and texts
  shorter than 100 or longer than 10,000 words.
- Measures, for every eligible document, the share of its 20-word n-grams that occur
  anywhere in the training data of the embedders and the reranker (questions, positives
  and negatives of runs 1-5, the reranker groups and the DOG 2024 dev set) and excludes
  documents above --max-overlap. Training text is DOG <= 2023, BOP, BOPPO, ParlaMint,
  SINAI and WebFAQ, so overlap here means boilerplate or a call that repeats an older one.
- Stratifies by document kind (call for applications, other resolution, extract,
  regulation, staff) and, inside each kind, round-robins over issuers, keeping at most one
  document per title template (same issuer, title word Jaccard >= 0.6) so near-duplicates
  of each other aren't both picked. About 15 % of each kind's slots go to documents with
  a table, so the annotator can write table questions.
- Lists, for each candidate, the closest same-issuer documents in the index ("siblings"),
  which the questions have to tell apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import eval_dog  # noqa: E402

FINGERPRINT = "fb26bfd556"
TRAINING_FILES = [
    "work/kaggle/train2/mined2/train.jsonl",  # runs 1-4 (run 1 = these rows minus the BM25-rescued ones)
    "work/kaggle/mined5/train.jsonl",  # run 5 (adds short passages and WebFAQ)
    "work/gcp/rerank1/rr1/groups.jsonl",  # reranker groups (query + 8 passages)
    "work/kaggle/train2/mined2/dev.jsonl",  # DOG 2024 dev set, used for model selection
]
QUOTAS = {"call": 40, "other": 34, "extract": 18, "regulation": 14, "staff": 14}
TABLE_SHARE = 0.15
NGRAM = 20
WORD_RE = re.compile(r"\w+")
SHORT_WORDS = {"de", "da", "do", "das", "dos", "e", "a", "o", "as", "os", "en", "no", "na", "nos", "nas", "pola",
               "polo", "que", "se", "para", "ao", "aos", "á", "ás", "por", "con", "un", "unha", "cal"}


# ---------------------------------------------------------------- document features


def doc_id(row: dict) -> str:
    return eval_dog.doc_id_from_url(row["url"])


def issuer(did: str) -> str:
    m = re.match(r"Anuncio([A-Z]+\d+)", did)
    return m.group(1) if m else did


def kind(row: dict) -> str:
    title = row["title"]
    if title.upper().startswith("CORRECCI") or row["type"].startswith("Corrección"):
        return "correction"
    if title.upper().startswith("EXTRACTO"):
        return "extract"
    if row["section"].startswith("II."):
        return "staff"
    if re.search(r"bases reguladoras|convoca", title.lower()):
        return "call"
    if row["section"].startswith("I.") or row["type"] in ("Decreto", "Lei", "Real decreto"):
        return "regulation"
    return "other"


def twin_key(title: str) -> str:
    """Title with the EXTRACTO prefix removed: an extract and its full text share it."""
    t = re.sub(r"^\s*extracto\s+(da|do|de)\s+", "", title.lower())
    return re.sub(r"\W+", " ", t).strip()


def template_words(title: str) -> set[str]:
    t = re.sub(r"^\s*extracto\s+(da|do|de)\s+", "", title.lower())
    return {w for w in WORD_RE.findall(t) if not any(ch.isdigit() for ch in w) and w not in SHORT_WORDS and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def table_lines(text: str) -> int:
    """Lines inside runs of >= 8 consecutive short lines (<= 6 words) where a third carry digits.

    DOG tables reach the corpus flattened to one cell per line, so this finds them."""
    total, run = 0, []
    for line in text.split("\n") + [""]:
        words = line.split()
        if words and len(words) <= 6:
            run.append(line)
            continue
        if len(run) >= 8 and sum(any(c.isdigit() for c in x) for x in run) >= len(run) / 3:
            total += len(run)
        run = []
    return total


def excerpt(text: str, words: int = 60) -> str:
    lines = [ln for ln in text.split("\n") if ln.strip() and not ln.startswith("BDNS")]
    body = " ".join(" ".join(lines).split())
    # Every extract opens with the same sentence about the national grants database; skip it.
    start = body.find("Primeiro.")
    if body.startswith("De conformidade co previsto nos artigos 17.3.b)") and 0 < start < 1500:
        body = body[start:]
    out = " ".join(body.split()[:words])
    return out + " …"


# ---------------------------------------------------------------- overlap with training text


def word_hashes(text: str) -> np.ndarray:
    return np.fromiter((hash(w) for w in WORD_RE.findall(text.lower())), dtype=np.int64)


def ngram_hashes(text: str) -> np.ndarray:
    wh = word_hashes(text).view(np.uint64)
    n = len(wh) - NGRAM + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    p = np.uint64(1_000_003)
    with np.errstate(over="ignore"):
        for j in range(NGRAM):
            h = h * p + wh[j : j + n]
    return h


def training_texts(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row.get("query", "")
            for key in ("positive",):
                if row.get(key):
                    yield row[key]
            for neg in row.get("negatives", []) or []:
                yield neg if isinstance(neg, str) else neg.get("text", "")
            for doc in row.get("docs", []) or []:
                yield doc


def training_overlap(texts: dict[str, str], root: Path) -> dict[str, float]:
    doc_hashes = {did: np.unique(ngram_hashes(t)) for did, t in texts.items()}
    pool = np.unique(np.concatenate([h for h in doc_hashes.values() if len(h)]))
    hits: set[int] = set()
    seen: set[str] = set()
    for rel in TRAINING_FILES:
        t0, n = time.time(), 0
        buf: list[np.ndarray] = []
        for text in training_texts(root / rel):
            key = hashlib.md5(text.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            n += 1
            buf.append(ngram_hashes(text))
            if len(buf) >= 2000:
                hits |= _hits(buf, pool)
                buf = []
        hits |= _hits(buf, pool)
        print(f"  {rel}: {n} new texts, {time.time() - t0:.0f} s, {len(hits)} pool n-grams hit so far", flush=True)
    hit_arr = np.fromiter(hits, dtype=np.uint64) if hits else np.empty(0, dtype=np.uint64)
    return {
        did: (float(np.isin(h, hit_arr).mean()) if len(h) else 0.0) for did, h in doc_hashes.items()
    }


def _hits(buf: list[np.ndarray], pool: np.ndarray) -> set[int]:
    if not buf:
        return set()
    h = np.concatenate(buf)
    idx = np.searchsorted(pool, h)
    idx[idx == len(pool)] = 0
    return set(h[pool[idx] == h].tolist())


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dog2025", default="work/hf-cache/hub/datasets--proxectonos--corpus_dominio_legal_administrativo_tmp/"
                    "snapshots/829ca66f2718b58a34b62e62ee600707a8380cc3/XuntaGalicia_DOG/gl/2025_gl.jsonl")
    ap.add_argument("--dograg-dir", default="work/hf-cache/hub/datasets--proxectonos--dog-rag/snapshots/"
                    "dca1fa77e7b74d498d7deadc08905461456773fa")
    ap.add_argument("--overlap-cache", default="work/human-test/train_overlap.json")
    ap.add_argument("--max-overlap", type=float, default=0.5)
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--max-words", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "candidates.jsonl"))
    args = ap.parse_args()
    root = Path.cwd()

    chunks, queries = eval_dog.load_inputs(SimpleNamespace(
        dog2025=args.dog2025, dograg_dir=args.dograg_dir, corpus_revision=None, dograg_revision=None,
        max_distractors=None, max_queries=None, seed=0))
    fingerprint = hashlib.sha1("".join(c["text"] for c in chunks).encode()).hexdigest()[:10]
    index_docs = {c["doc_id"] for c in chunks}
    print(json.dumps({"chunks": len(chunks), "documents": len(index_docs), "index_fingerprint": fingerprint}))
    if fingerprint != FINGERPRINT:
        raise SystemExit(f"index fingerprint {fingerprint} != {FINGERPRINT}: not the paper's index")
    # All 122 DOG-RAG source documents, not only the 116 that questions point to.
    dograg_docs = {q["doc_id"] for q in queries}
    dograg_docs |= {d["file_id"] for d in json.loads((Path(args.dograg_dir) / "dog.json").read_text(encoding="utf-8"))}

    rows = {}
    with open(args.dog2025, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            did = doc_id(r)
            if did in index_docs and r.get("text"):
                rows[did] = r
    by_twin = defaultdict(list)
    for did, r in rows.items():
        by_twin[twin_key(r["title"])].append(did)
    twins = {did: [d for d in by_twin[twin_key(r["title"])] if d != did] for did, r in rows.items()}

    excluded = Counter()
    eligible = {}
    for did, r in rows.items():
        k = kind(r)
        n = len(r["text"].split())
        if did in dograg_docs:
            excluded["DOG-RAG source document"] += 1
        elif any(t in dograg_docs for t in twins[did]):
            excluded["twin of a DOG-RAG document"] += 1
        elif k == "correction":
            excluded["correction of errors"] += 1
        elif not args.min_words <= n <= args.max_words:
            excluded[f"outside {args.min_words}-{args.max_words} words"] += 1
        else:
            eligible[did] = r
    print(json.dumps({"eligible": len(eligible), "excluded": excluded}, ensure_ascii=False))

    cache = Path(args.overlap_cache)
    if cache.exists():
        overlap = json.loads(cache.read_text())
    else:
        print("measuring 20-gram overlap with the training data (a few minutes)…", flush=True)
        overlap = training_overlap({d: r["text"] for d, r in eligible.items()}, root)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(overlap))
    ov = np.array([overlap[d] for d in eligible])
    print(json.dumps({"overlap_quantiles": {q: round(float(np.quantile(ov, q)), 3) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
                      f"above_{args.max_overlap}": int((ov > args.max_overlap).sum())}))

    rng = random.Random(args.seed)
    tmpl = {d: template_words(r["title"]) for d, r in rows.items()}
    tables = {d: table_lines(r["text"]) for d, r in eligible.items()}
    picked: list[str] = []
    blocked: set[str] = set()

    def same_template(did: str) -> bool:
        return any(issuer(p) == issuer(did) and jaccard(tmpl[p], tmpl[did]) >= 0.6 for p in picked)

    def take(pool: list[str], n: int) -> int:
        by_issuer = defaultdict(list)
        for d in pool:
            by_issuer[issuer(d)].append(d)
        for lst in by_issuer.values():
            rng.shuffle(lst)
        order = sorted(by_issuer)
        rng.shuffle(order)
        got = 0
        while got < n and any(by_issuer.values()):
            for iss in order:
                while by_issuer[iss]:
                    d = by_issuer[iss].pop()
                    if d in blocked or d in picked or same_template(d):
                        continue
                    picked.append(d)
                    blocked.update(twins[d])
                    got += 1
                    break
                if got >= n:
                    break
        return got

    strata = defaultdict(list)
    for d, r in eligible.items():
        if overlap[d] <= args.max_overlap:
            strata[kind(r)].append(d)
    for k, quota in QUOTAS.items():
        n_tab = round(quota * TABLE_SHARE)
        got = take([d for d in strata[k] if tables[d] >= 10], n_tab)
        take([d for d in strata[k] if d not in picked], quota - got)

    rng.shuffle(picked)
    by_issuer_all = defaultdict(list)
    for d in rows:
        by_issuer_all[issuer(d)].append(d)
    out = []
    for i, d in enumerate(picked, 1):
        r = rows[d]
        sibs = sorted(
            (x for x in by_issuer_all[issuer(d)] if x != d and x not in twins[d] and kind(rows[x]) != "correction"),
            key=lambda x: (-jaccard(tmpl[x], tmpl[d]), x),
        )[:3]
        out.append({
            "cand": f"C{i:03d}",
            "doc_id": d,
            "url": r["url"],
            "date": r["date"],
            "dog_number": r["DOG_number"],
            "issuer": r["public_body"],
            "issuer_code": issuer(d),
            "type": r["type"],
            "kind": kind(r),
            "title": r["title"],
            "words": len(r["text"].split()),
            "table_lines": tables[d],
            "train_overlap": round(overlap[d], 4),
            "twins": twins[d],
            "siblings": [{"doc_id": x, "date": rows[x]["date"], "title": rows[x]["title"],
                          "title_jaccard": round(jaccard(tmpl[x], tmpl[d]), 2)} for x in sibs],
            "excerpt": excerpt(r["text"]),
        })
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "candidates": len(out),
        "by_kind": Counter(o["kind"] for o in out),
        "issuers": len({o["issuer_code"] for o in out}),
        "max_per_issuer": Counter(o["issuer_code"] for o in out).most_common(3),
        "with_table": sum(o["table_lines"] >= 10 for o in out),
        "with_twin_in_index": sum(bool(o["twins"]) for o in out),
        "words_median": int(np.median([o["words"] for o in out])),
        "train_overlap_max": max(o["train_overlap"] for o in out),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
