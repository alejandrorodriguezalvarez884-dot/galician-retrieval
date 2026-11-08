"""Retrieval evaluation on DOG-RAG, following the nos-rag-eval protocol.

Protocol (https://github.com/proxectonos/nos-rag-eval and the DOG-RAG paper,
SEPLN 2026): DOG 2025 documents split into 250-word chunks with 50 words of
overlap, top 10 chunks retrieved, optionally reranked, top 3 kept. Every
question paraphrase is a separate query. Precision@3, Recall@3 and MRR are
computed at document level exactly as in ir-metrics/evaluate_ir_metrics.py.

Differences from nos-rag-eval:
- the index is the public DOG 2025 text (proxectonos/corpus_dominio_legal_administrativo_tmp)
  plus the 122 DOG-RAG source documents, not their private 12k-document dump;
- exact cosine search in numpy instead of Elasticsearch HNSW, and bm25s with a
  lowercase word tokenizer and k1=1.2, b=0.75 instead of the ES standard analyzer;
- dense models run both as in nos-rag-eval (no prompts) and with their
  recommended query/document prompts;
- extra metrics on the first stage: document Recall@10 and nDCG@10;
- --hybrid adds BM25 + dense runs fused with reciprocal rank fusion (top --fusion-depth of
  each list, score 1 / (--rrf-k + rank)), named hybrid/<retriever>-<mode>. Elasticsearch
  offers the same fusion, so the setup is reproducible in nos-rag-eval.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

CORPUS_REPO = "proxectonos/corpus_dominio_legal_administrativo_tmp"
CORPUS_FILE = "XuntaGalicia_DOG/gl/2025_gl.jsonl"
DOGRAG_REPO = "proxectonos/dog-rag"

CHUNK_WORDS = 250
CHUNK_OVERLAP = 50
FIRST_STAGE_K = 10
FINAL_K = 3

# name -> how to load and prompt it. "prompts" are used in the prompted mode only;
# None means the model's own sentence-transformers prompts ("query"/"document").
EMBEDDERS = {
    "bge-m3": {"model": "BAAI/bge-m3", "prompts": {}},
    "qwen3-embedding-0.6b": {"model": "Qwen/Qwen3-Embedding-0.6B", "prompts": None},
    "embeddinggemma-300m": {"model": "google/embeddinggemma-300m", "prompts": None, "fp32": True},
    "embeddinggemma-2": {"model": "google/embeddinggemma-2", "prompts": None, "fp32": True},
    "multilingual-e5-base": {
        "model": "intfloat/multilingual-e5-base",
        "prompts": {"query": "query: ", "document": "passage: "},
    },
    "gte-multilingual-base": {
        "model": "Alibaba-NLP/gte-multilingual-base",
        "prompts": {},
        "trust_remote_code": True,
    },
    "mrbert-nos-gl": {"model": "proxectonos/MrBERT-nos-gl", "prompts": {}},
    # Added for the stricter publish bar (2026-11-04): Apache-2.0 multilingual models a reader would ask about.
    "arctic-embed-l-v2.0": {"model": "Snowflake/snowflake-arctic-embed-l-v2.0", "prompts": None},
    "granite-embedding-311m-r2": {"model": "ibm-granite/granite-embedding-311m-multilingual-r2", "prompts": {}},
    "granite-embedding-97m-r2": {"model": "ibm-granite/granite-embedding-97m-multilingual-r2", "prompts": {}},
}

RERANKERS = {
    "bge-reranker-v2-m3": {"model": "BAAI/bge-reranker-v2-m3", "kind": "cross-encoder"},
    "qwen3-reranker-0.6b": {"model": "Qwen/Qwen3-Reranker-0.6B", "kind": "qwen3"},
    "jina-reranker-v3": {"model": "jinaai/jina-reranker-v3", "kind": "jina"},
    "mxbai-rerank-base-v2": {"model": "mixedbread-ai/mxbai-rerank-base-v2", "kind": "cross-encoder"},
    # Remote code that needs transformers 4.x: run it from a separate environment.
    "gte-multilingual-reranker-base": {
        "model": "Alibaba-NLP/gte-multilingual-reranker-base",
        "kind": "cross-encoder",
        "trust_remote_code": True,
    },
}


# ---------------------------------------------------------------- data


def doc_id_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")


def fetch(repo: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def load_inputs(args) -> tuple[list[dict], list[dict]]:
    corpus_path = Path(args.dog2025) if args.dog2025 else fetch(CORPUS_REPO, CORPUS_FILE, args.corpus_revision)
    dograg_dir = Path(args.dograg_dir) if args.dograg_dir else None
    docs_path = dograg_dir / "dog.json" if dograg_dir else fetch(DOGRAG_REPO, "dog.json", args.dograg_revision)
    triplets_path = (
        dograg_dir / "triplets.json" if dograg_dir else fetch(DOGRAG_REPO, "triplets.json", args.dograg_revision)
    )

    eval_docs = json.loads(docs_path.read_text(encoding="utf-8"))
    triplets = json.loads(triplets_path.read_text(encoding="utf-8"))

    documents: dict[str, str] = {}
    for d in eval_docs:
        documents[d["file_id"]] = d["content"]
    eval_ids = set(documents)

    distractors = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            did = doc_id_from_url(d["url"])
            if did not in eval_ids and d.get("text"):
                distractors.append((did, d["text"]))
    if args.max_distractors is not None:
        random.Random(args.seed).shuffle(distractors)
        distractors = distractors[: args.max_distractors]
    for did, text in distractors:
        documents.setdefault(did, text)

    queries = []
    for item in triplets:
        uid = item.get("uid", item.get("id"))  # 19 of the 502 entries use "id"
        for i, q in enumerate(item["question"]):
            queries.append({"id": f"{uid}_{i}", "text": q, "doc_id": item["file_name"], "category": item["category"]})
    if args.max_queries is not None:
        random.Random(args.seed).shuffle(queries)
        queries = sorted(queries[: args.max_queries], key=lambda q: q["id"])

    missing = {q["doc_id"] for q in queries} - set(documents)
    if missing:
        raise SystemExit(f"{len(missing)} reference documents are not in the index, e.g. {sorted(missing)[:3]}")

    chunks = []
    for did, text in documents.items():
        for j, chunk in enumerate(split_in_chunks(text)):
            chunks.append({"doc_id": did, "pos": j, "text": chunk})
    return chunks, queries


def split_in_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Same as ElasticSearchProxy.split_text_in_chunks in nos-rag-eval.
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap) if words[i : i + size]]


# ---------------------------------------------------------------- first stage


TOKEN_RE = r"(?u)\b\w+\b"


def bm25_search(chunks: list[dict], queries: list[dict], k: int) -> np.ndarray:
    import bm25s

    corpus_tokens = bm25s.tokenize(
        [c["text"] for c in chunks], lower=True, token_pattern=TOKEN_RE, stopwords=None, show_progress=False
    )
    retriever = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = [re.findall(TOKEN_RE, q["text"].lower()) for q in queries]
    vocab = corpus_tokens.vocab
    query_ids = [[vocab[t] for t in toks if t in vocab] or [0] for toks in query_tokens]
    idx, _ = retriever.retrieve(query_ids, k=k, show_progress=False)
    return np.asarray(idx)


def load_embedder(spec: dict, device: str, fp16: bool):
    from sentence_transformers import SentenceTransformer

    use_fp16 = fp16 and device.startswith("cuda") and not spec.get("fp32")
    model = SentenceTransformer(
        spec["model"],
        device=device,
        trust_remote_code=spec.get("trust_remote_code", False),
        model_kwargs={"dtype": torch.float16} if use_fp16 else None,
    )
    if model.max_seq_length is None or model.max_seq_length > 1024:
        model.max_seq_length = 1024
    return model


def encode(model, texts: list[str], side: str, spec: dict, prompted: bool, batch_size: int, devices=None) -> np.ndarray:
    kwargs = {"batch_size": batch_size, "normalize_embeddings": True, "convert_to_numpy": True}
    if devices and len(devices) > 1 and len(texts) > 1000:
        kwargs["device"] = devices  # one worker process per device
    if not prompted:
        return model.encode(texts, **kwargs)
    prompts = spec["prompts"]
    if prompts is None:
        fn = model.encode_query if side == "query" else model.encode_document
        return fn(texts, **kwargs)
    return model.encode(texts, prompt=prompts.get(side, ""), **kwargs)


def dense_search(q_emb: np.ndarray, c_emb: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(q_emb), k), dtype=np.int64)
    for start in range(0, len(q_emb), 256):
        scores = q_emb[start : start + 256] @ c_emb.T
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        out[start : start + 256] = np.take_along_axis(top, order, axis=1)
    return out


def rrf(rankings: list[np.ndarray], k: int, top: int) -> np.ndarray:
    """Reciprocal rank fusion of several ranked candidate lists per query."""
    out = np.empty((len(rankings[0]), top), dtype=np.int64)
    for i in range(len(rankings[0])):
        score: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, c in enumerate(ranking[i]):
                score[int(c)] += 1.0 / (k + rank + 1)
        out[i] = sorted(score, key=lambda c: (-score[c], c))[:top]
    return out


# ---------------------------------------------------------------- rerankers


class CrossEncoderReranker:
    def __init__(self, model_id: str, device: str, fp16: bool, trust_remote_code: bool = False):
        from sentence_transformers import CrossEncoder

        use_fp16 = fp16 and device.startswith("cuda")
        self.model = CrossEncoder(
            model_id,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"dtype": torch.float16} if use_fp16 else None,
        )
        limit = getattr(self.model.config, "max_position_embeddings", 514) - 2
        self.model.max_length = min(1024, self.model.max_length or limit, limit)

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return self.model.predict([(query, p) for p in passages], batch_size=16).tolist()


class Qwen3Reranker:
    # Same prompt and yes/no scoring as Reranker.Qwen3Reranker in nos-rag-eval.
    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
        'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    TASK = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, model_id: str, device: str, fp16: bool, max_length: int = 2048):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        self.max_length = max_length

    @torch.no_grad()
    def scores(self, query: str, passages: list[str], batch_size: int = 2) -> list[float]:
        out = []
        for i in range(0, len(passages), batch_size):
            out += self._scores(query, passages[i : i + batch_size])
        return out

    def _scores(self, query: str, passages: list[str]) -> list[float]:
        texts = [f"<Instruct>: {self.TASK}\n<Query>: {query}\n<Document>: {p}" for p in passages]
        enc = self.tok(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix) - len(self.suffix),
        )
        enc["input_ids"] = [self.prefix + ids + self.suffix for ids in enc["input_ids"]]
        batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.model.device)
        # Only the last position is needed; full logits over 151k tokens for 10 x ~1k tokens run out of memory on a T4.
        logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
        pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
        return torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()


class JinaReranker:
    def __init__(self, model_id: str, device: str, fp16: bool):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id, dtype="auto", trust_remote_code=True).to(device).eval()

    def scores(self, query: str, passages: list[str]) -> list[float]:
        out = [0.0] * len(passages)
        for r in self.model.rerank(query, passages):
            out[r["index"]] = float(r["relevance_score"])
        return out


def load_reranker(name: str, device: str, fp16: bool):
    if name.startswith("ce:"):
        return CrossEncoderReranker(name[3:], device, fp16)
    spec = RERANKERS[name]
    if spec["kind"] == "cross-encoder":
        return CrossEncoderReranker(spec["model"], device, fp16, spec.get("trust_remote_code", False))
    cls = {"qwen3": Qwen3Reranker, "jina": JinaReranker}[spec["kind"]]
    return cls(spec["model"], device, fp16)


# ---------------------------------------------------------------- metrics


def nos_rag_metrics(reference: str, retrieved: list[str]) -> tuple[float, float, float]:
    # Document-level P, R and MRR as in ir-metrics/evaluate_ir_metrics.py (method='document').
    mrr = next((1.0 / (i + 1) for i, d in enumerate(retrieved) if d == reference), 0.0)
    unique = list(dict.fromkeys(retrieved))
    if not unique:
        return 0.0, 0.0, mrr
    hit = sum(1 for d in unique if d == reference)
    return hit / len(unique), float(hit), mrr


def first_stage_metrics(reference: str, retrieved: list[str]) -> tuple[float, float]:
    unique = list(dict.fromkeys(retrieved))
    if reference not in unique:
        return 0.0, 0.0
    return 1.0, 1.0 / math.log2(unique.index(reference) + 2)


def evaluate(queries, chunks, first_stage: np.ndarray, reranker=None) -> tuple[dict, list[dict]]:
    sums = defaultdict(float)
    per_cat = defaultdict(lambda: [0.0, 0])
    rows = []
    for q, cand in zip(queries, first_stage):
        cand = [int(c) for c in cand]
        cand_docs = [chunks[c]["doc_id"] for c in cand]
        r10, ndcg10 = first_stage_metrics(q["doc_id"], cand_docs)
        if reranker is not None:
            s = reranker.scores(q["text"], [chunks[c]["text"] for c in cand])
            cand = [c for _, c in sorted(zip(s, cand), key=lambda x: -x[0])]
        final = cand[:FINAL_K]
        final_docs = [chunks[c]["doc_id"] for c in final]
        p, r, mrr = nos_rag_metrics(q["doc_id"], final_docs)
        for key, val in (("P@3", p), ("R@3", r), ("MRR", mrr), ("R@10_first_stage", r10), ("nDCG@10_first_stage", ndcg10)):
            sums[key] += val
        per_cat[q["category"]][0] += mrr
        per_cat[q["category"]][1] += 1
        rows.append({"id": q["id"], "final": [[chunks[c]["doc_id"], chunks[c]["pos"]] for c in final], "mrr": mrr})
    n = len(queries)
    metrics = {k: round(v / n, 4) for k, v in sums.items()}
    metrics["MRR_by_category"] = {c: round(s / m, 4) for c, (s, m) in sorted(per_cat.items())}
    return metrics, rows


# ---------------------------------------------------------------- main


def parse_retrievers(value: str) -> list[tuple[str, dict]]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item == "bm25":
            out.append(("bm25", {}))
        elif "=" in item:  # name=path for a local or Hub checkpoint with no prompts
            name, path = item.split("=", 1)
            out.append((name, {"model": path, "prompts": {}}))
        else:
            out.append((item, EMBEDDERS[item]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrievers", default="bm25,bge-m3", help="comma list: bm25, a name from EMBEDDERS, or name=path")
    ap.add_argument("--rerankers", default="none", help="comma list: none, a name from RERANKERS, or ce:path")
    ap.add_argument("--rerank-on", default="bm25,bge-m3", help="first-stage runs (name/mode) the rerankers are applied to")
    ap.add_argument("--modes", default="plain,prompted", help="plain = as nos-rag-eval, prompted = model prompts")
    ap.add_argument("--dog2025", help="local 2025_gl.jsonl (default: download from the Hub)")
    ap.add_argument("--dograg-dir", help="local folder with dog.json and triplets.json (default: download)")
    ap.add_argument("--corpus-revision", default=None)
    ap.add_argument("--dograg-revision", default=None)
    ap.add_argument("--max-distractors", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-queries", type=int, default=None, help="smoke tests only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--devices", default=None, help="comma list to split corpus encoding, e.g. cuda:0,cuda:1 (default: all GPUs)"
    )
    ap.add_argument(
        "--first-stage-from",
        default=None,
        help="folder with first_*.npy saved by an earlier run; reranks them without loading any embedder",
    )
    ap.add_argument("--hybrid", action="store_true", help="also fuse BM25 with every dense run (reciprocal rank fusion)")
    ap.add_argument("--fusion-depth", type=int, default=100, help="candidates taken from each list before fusing")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
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

    chunks, queries = load_inputs(args)
    fingerprint = hashlib.sha1("".join(c["text"] for c in chunks).encode()).hexdigest()[:10]
    info = {
        "chunks": len(chunks),
        "documents": len({c["doc_id"] for c in chunks}),
        "queries": len(queries),
        "index_fingerprint": fingerprint,
    }
    print(json.dumps(info))

    first_stages: dict[str, np.ndarray] = {}
    results_path = out / "results.jsonl"

    def record(run: str, reranker: str, metrics: dict, rows: list[dict], seconds: float):
        line = {"retriever": run, "reranker": reranker, **metrics, "seconds": round(seconds, 1), **info}
        fname = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{run}__{reranker}") + ".jsonl"
        with open(out / "runs" / fname, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(json.dumps({k: line[k] for k in ("retriever", "reranker", "P@3", "R@3", "MRR", "nDCG@10_first_stage")}))

    def failed(run: str, reranker: str, exc: Exception):
        line = {"retriever": run, "reranker": reranker, "error": f"{type(exc).__name__}: {exc}"[:500], **info}
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(json.dumps(line, ensure_ascii=False))

    def free():
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def save_first_stage(run: str):
        fname = re.sub(r"[^A-Za-z0-9._-]+", "_", run)
        np.save(out / "cache" / f"first_{fname}_{fingerprint}.npy", first_stages[run])

    if args.first_stage_from:
        # Reranking in a fresh process: a model that just encoded the corpus can leave the GPU
        # too fragmented for a 0.6B reranker on a 16 GB T4.
        for path in sorted(Path(args.first_stage_from).glob(f"first_*_{fingerprint}.npy")):
            run = path.name[len("first_") : -len(f"_{fingerprint}.npy")].replace("_", "/", 1)
            first_stages[run] = np.load(path)
        print(json.dumps({"first_stages_loaded": sorted(first_stages)}))
        retrievers = []
    else:
        retrievers = parse_retrievers(args.retrievers)

    depth = max(FIRST_STAGE_K, args.fusion_depth) if args.hybrid else FIRST_STAGE_K
    bm25_deep = None
    if args.hybrid and any(name != "bm25" for name, _ in retrievers):
        bm25_deep = bm25_search(chunks, queries, depth)

    modes = [m.strip() for m in args.modes.split(",")]
    for name, spec in retrievers:
        t0 = time.time()
        if name == "bm25":
            first_stages["bm25"] = (bm25_deep if bm25_deep is not None else bm25_search(chunks, queries, depth))[:, :FIRST_STAGE_K]
            save_first_stage("bm25")
            metrics, rows = evaluate(queries, chunks, first_stages["bm25"])
            record("bm25", "none", metrics, rows, time.time() - t0)
            continue
        try:
            model = load_embedder(spec, args.device, fp16)
        except Exception as exc:
            failed(name, "none", exc)
            continue
        run_modes = [m for m in modes if m == "plain" or spec.get("prompts") != {}] or ["plain"]
        for mode in run_modes:
            run = f"{name}/{mode}"
            try:
                t0 = time.time()
                prompted = mode == "prompted"
                cache = out / "cache" / f"{name}_{mode}_{fingerprint}.npy"
                if cache.exists():
                    c_emb = np.load(cache)
                else:
                    c_emb = encode(model, [c["text"] for c in chunks], "document", spec, prompted, args.batch_size, devices)
                    np.save(cache, c_emb)
                q_emb = encode(model, [q["text"] for q in queries], "query", spec, prompted, args.batch_size)
                deep = dense_search(q_emb.astype(np.float32), c_emb.astype(np.float32), depth)
                first_stages[run] = deep[:, :FIRST_STAGE_K]
                save_first_stage(run)
                metrics, rows = evaluate(queries, chunks, first_stages[run])
                record(run, "none", metrics, rows, time.time() - t0)
                if bm25_deep is not None:
                    t0 = time.time()
                    hrun = f"hybrid/{name}-{mode}"
                    first_stages[hrun] = rrf([bm25_deep, deep], args.rrf_k, FIRST_STAGE_K)
                    save_first_stage(hrun)
                    metrics, rows = evaluate(queries, chunks, first_stages[hrun])
                    record(hrun, "none", metrics, rows, time.time() - t0)
            except Exception as exc:
                failed(run, "none", exc)
        del model
        free()

    rerank_on = []
    for r in args.rerank_on.split(","):
        r = r.strip()
        rerank_on += [k for k in first_stages if k == r or k.startswith(r + "/")]
    for rr in [r.strip() for r in args.rerankers.split(",") if r.strip() != "none"]:
        try:
            reranker = load_reranker(rr, args.device, fp16)
        except Exception as exc:
            failed("-", rr, exc)
            continue
        for run in dict.fromkeys(rerank_on):
            try:
                t0 = time.time()
                metrics, rows = evaluate(queries, chunks, first_stages[run], reranker)
                record(run, rr, metrics, rows, time.time() - t0)
            except Exception as exc:
                failed(run, rr, exc)
        del reranker
        free()


if __name__ == "__main__":
    main()
