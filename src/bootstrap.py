"""Paired bootstrap over eval_dog.py's per-query run files.

Each DOG-RAG question has one or more paraphrases, and every paraphrase is a query, so
queries are not independent. The bootstrap resamples questions (query id `12_1` belongs to
question `12`) and keeps all their paraphrases together; resampling queries one by one would
give intervals that are too narrow.

    python bootstrap.py runs/hybrid_galician-run2-plain__none.jsonl runs/bm25__none.jsonl

Prints the mean MRR difference (A - B), a 95 % percentile interval, a two-sided p-value
and how many queries got better or worse. Rows are paired by position, so both files must
come from the same eval_dog.py index and query order (checked through the ids).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[list[str], np.ndarray]:
    ids, mrr = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(str(row["id"]))
            mrr.append(row["mrr"])
    return ids, np.array(mrr, dtype=float)


def paired_bootstrap(ids: list[str], a: np.ndarray, b: np.ndarray, samples: int, seed: int) -> dict:
    diff = a - b
    groups: dict[str, list[int]] = defaultdict(list)
    for i, qid in enumerate(ids):
        groups[qid.split("_")[0]].append(i)
    sums = np.array([diff[idx].sum() for idx in groups.values()])
    sizes = np.array([len(idx) for idx in groups.values()])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(samples, len(sums)))
    boot = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "queries": len(ids),
        "questions": len(groups),
        "mrr_a": round(float(a.mean()), 4),
        "mrr_b": round(float(b.mean()), 4),
        "diff": round(float(diff.mean()), 4),
        "ci95": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)],
        "p_two_sided": round(float(min(1.0, p)), 4),
        "better": int((diff > 0).sum()),
        "worse": int((diff < 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ids_a, a = load(args.a)
    ids_b, b = load(args.b)
    if ids_a != ids_b:
        raise SystemExit("the two files don't have the same queries in the same order")
    result = paired_bootstrap(ids_a, a, b, args.samples, args.seed)
    print(json.dumps({"a": args.a.name, "b": args.b.name, **result}))


if __name__ == "__main__":
    main()
