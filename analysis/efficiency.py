"""Cost of our models on a laptop: parameters, query latency, passage throughput, reranking time.

Uses real DOG-RAG questions and index chunks. Run once per device:

    python efficiency.py --embedder PATH --reranker PATH --device cpu --out eff_cpu.json
    python efficiency.py --embedder PATH --reranker PATH --device mps --out eff_mps.json
"""

import argparse
import json
import platform
import random
import resource
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import eval_dog  # noqa: E402


def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device: str) -> float:
    sync(device)
    t = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's)")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--passages", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    inputs = SimpleNamespace(dog2025=None, dograg_dir=None, corpus_revision=None, dograg_revision=None,
                             max_distractors=None, max_queries=None, seed=0)
    chunks, queries = eval_dog.load_inputs(inputs)
    rng = random.Random(0)
    qs = [q["text"] for q in rng.sample(queries, args.queries)]
    ps = [c["text"] for c in rng.sample(chunks, args.passages)]

    res = {"device": args.device, "torch_threads": torch.get_num_threads(), "machine": platform.machine(),
           "processor": platform.processor(), "torch": torch.__version__}

    emb = SentenceTransformer(args.embedder, device=args.device)
    res["embedder_params"] = sum(p.numel() for p in emb.parameters())
    emb.encode(qs[:4] + ps[:4])  # warm-up
    lat = [timed(lambda q=q: emb.encode([q]), args.device) for q in qs]
    res["query_latency_ms"] = {"median": 1000 * statistics.median(lat), "p90": 1000 * sorted(lat)[int(0.9 * len(lat))]}
    t = timed(lambda: emb.encode(ps, batch_size=32), args.device)
    res["passages_per_s_batch32"] = len(ps) / t
    res["index_36k_minutes_est"] = len(chunks) / res["passages_per_s_batch32"] / 60
    del emb

    rr = CrossEncoder(args.reranker, device=args.device)
    res["reranker_params"] = sum(p.numel() for p in rr.model.parameters())
    rr.predict([(qs[0], ps[0])])  # warm-up
    per_query = []
    for q in qs[:50]:
        cands = rng.sample(ps, 10)  # top 10 as in the DOG-RAG protocol
        per_query.append(timed(lambda q=q, c=cands: rr.predict([(q, p) for p in c], batch_size=10), args.device))
    res["rerank_top10_ms"] = {"median": 1000 * statistics.median(per_query),
                              "p90": 1000 * sorted(per_query)[int(0.9 * len(per_query))]}
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res["peak_rss_mb"] = rss / 2**20 if sys.platform == "darwin" else rss / 1024
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
