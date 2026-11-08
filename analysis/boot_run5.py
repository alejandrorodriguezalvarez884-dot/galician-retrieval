"""Paired bootstraps of run 5 against bge-m3, run 4 and the 0.9 mix.

Inputs: per-query files in results/ (run 5 from results/gcp/run5/; bge-m3 and run 4 from
results/kaggle/boot5/extra/ (e_gpu0 = bge-m3, e_gpu1 = run 4); run 4 on DOG-RAG from
results/kaggle/boot5/train4/; the mix from results/kaggle/boot5/mix/). nos-rag-news questions
are downloaded at revision b721e0e.

DOG-RAG and news use bootstrap.py (questions resampled, 10,000 samples); news paragraph MRR is
recomputed from the `final` lists against context_paragraph_indices; WebFAQ uses per-query
nDCG@10 from the rank; PAWS-gl recomputes average precision on each resample of pairs (2,000).
"""
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from bootstrap import paired_bootstrap  # noqa: E402

UP = REPO / "results"
K = UP / "kaggle/boot5"
R5 = UP / "gcp/run5"
QUESTIONS = Path(hf_hub_download("proxectonos/nos-rag-news", "nos-rag-dataset_questions.json", repo_type="dataset",
                             revision="b721e0e502dde94a71ad36625a76dac0f1af5c88"))

FILES = {
    "dog": {
        "run5": R5 / "test_run5/runs/galician-run5_plain__none.jsonl",
        "run4": K / "train4/test_run4/runs/galician-run4_plain__none.jsonl",
        "mix": K / "mix/t_mix/runs/galician-mix_plain__none.jsonl",
    },
}
for task in ("news", "paws", "webfaq"):
    FILES[task] = {
        "run5": R5 / f"e_run5/runs/{task}__galician-run5_plain__none.jsonl",
        "run4": K / f"extra/e_gpu1/runs/{task}__galician-run4_plain__none.jsonl",
        "mix": K / f"mix/e_mix/runs/{task}__mix-a09_plain__none.jsonl",
        "bge-m3": K / f"extra/e_gpu0/runs/{task}__bge-m3_plain__none.jsonl",
    }


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


questions = {str(q["id"]): q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}


def per_query(task, path):
    rs = rows(path)
    ids = [str(r["id"]) for r in rs]
    if task in ("dog", "news_doc"):
        return ids, np.array([r["mrr"] for r in rs], float)
    if task == "news_par":
        vals = []
        for r in rs:
            q = questions[str(r["id"])]
            ref = {f"{q['source_id']}-{i}" for i in q["context_paragraph_indices"]}
            hits = [f"{d}-{p}" in ref for d, p in r["final"]]
            vals.append(next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0))
        return ids, np.array(vals)
    if task == "webfaq":
        return ids, np.array([1.0 / np.log2(r["rank"] + 1) if r["rank"] else 0.0 for r in rs])
    raise ValueError(task)


def ap(scores, labels):
    order = np.argsort(-scores, kind="stable")
    lab = labels[order]
    tp = np.cumsum(lab)
    k = np.arange(1, len(lab) + 1)
    return float((tp / k * lab).sum() / max(lab.sum(), 1))


def paws_boot(pa, pb, samples=2000, seed=0):
    ra, rb = rows(pa), rows(pb)
    assert [r["id"] for r in ra] == [r["id"] for r in rb]
    la = np.array([r["label"] for r in ra])
    assert (la == np.array([r["label"] for r in rb])).all()
    sa = np.array([r["score"] for r in ra])
    sb = np.array([r["score"] for r in rb])
    base_a, base_b = ap(sa, la), ap(sb, la)
    rng = np.random.default_rng(seed)
    n = len(la)
    boot = np.empty(samples)
    for s in range(samples):
        idx = rng.integers(0, n, n)
        boot[s] = ap(sa[idx], la[idx]) - ap(sb[idx], la[idx])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "queries": n, "mrr_a": round(base_a, 4), "mrr_b": round(base_b, 4), "diff": round(base_a - base_b, 4),
        "ci95": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)],
        "p_two_sided": round(float(min(1.0, p)), 4), "below_m2": round(float((boot < -0.02).mean()), 3),
    }


def boot(task, a, b):
    key = {"news_doc": "news", "news_par": "news"}.get(task, task)
    ia, va = per_query(task, FILES[key][a])
    ib, vb = per_query(task, FILES[key][b])
    assert ia == ib, (task, a, b)
    res = paired_bootstrap(ia, va, vb, 10000, 0)
    # Share of samples below -2 points, from the same resamples as paired_bootstrap (same seed).
    diff = va - vb
    groups = {}
    for i, q in enumerate(ia):
        groups.setdefault(q.split("_")[0], []).append(i)
    sums = np.array([diff[idx].sum() for idx in groups.values()])
    sizes = np.array([len(idx) for idx in groups.values()])
    picks = np.random.default_rng(0).integers(0, len(sums), size=(10000, len(sums)))
    bs = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    res["below_m2"] = round(float((bs < -0.02).mean()), 3)
    return res


out = []
pairs = [("dog", "run5", "run4"), ("dog", "run5", "mix")]
for task in ("news_doc", "news_par", "webfaq"):
    pairs += [(task, "run5", "bge-m3"), (task, "run5", "run4"), (task, "run5", "mix")]
for task, a, b in pairs:
    r = boot(task, a, b)
    out.append({"task": task, "a": a, "b": b, **r})
for b in ("bge-m3", "run4", "mix"):
    out.append({"task": "paws", "a": "run5", "b": b, **paws_boot(FILES["paws"]["run5"], FILES["paws"][b])})
for r in out:
    print(json.dumps(r))
