"""Seed spread and the no-SINAI ablation.

Inputs: the Kaggle outputs of galician-seed1, galician-seed2 and galician-nosinai in
results/kaggle/<slug>/, the seed-0 files used by boot_run5.py (results/kaggle/boot5/), and
Nós's pipeline on our index (bge-m3 + bge-reranker-v2-m3, results/gcp/rerank1/test_rerank1/).
The output is kept as results/boot_variance.jsonl.

Prints the MRR of each model on every test with the seed mean and spread, then paired bootstraps
with the same methods as boot_run5.py (DOG-RAG and news by question, WebFAQ by query, PAWS-gl AP
by pair).
"""
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from bootstrap import paired_bootstrap  # noqa: E402

UP = HERE.parent / "results"
K = UP / "kaggle"
QUESTIONS = Path(hf_hub_download("proxectonos/nos-rag-news", "nos-rag-dataset_questions.json", repo_type="dataset",
                             revision="b721e0e502dde94a71ad36625a76dac0f1af5c88"))

DOG = {
    "run4": K / "boot5/train4/test_run4/runs/galician-run4_plain__none.jsonl",
    "mix": K / "boot5/mix/t_mix/runs/galician-mix_plain__none.jsonl",
    "nos": UP / "gcp/rerank1/test_rerank1/runs/bge-m3_plain__bge-reranker-v2-m3.jsonl",
    "bm25": K / "galician-seed1/test_seed1/runs/bm25__none.jsonl",
    "nosinai": K / "galician-nosinai/test_nosinai/runs/galician-nosinai_plain__none.jsonl",
}
EXTRA = {
    "run4": lambda t: K / f"boot5/extra/e_gpu1/runs/{t}__galician-run4_plain__none.jsonl",
    "mix": lambda t: K / f"boot5/mix/e_mix/runs/{t}__mix-a09_plain__none.jsonl",
    "bge-m3": lambda t: K / f"boot5/extra/e_gpu0/runs/{t}__bge-m3_plain__none.jsonl",
    "nosinai": lambda t: K / f"galician-nosinai/e_nosinai/runs/{t}__galician-nosinai_plain__none.jsonl",
}
for s in (1, 2):
    DOG[f"run4-s{s}"] = K / f"galician-seed{s}/test_seed{s}/runs/galician-seed{s}_plain__none.jsonl"
    DOG[f"mix-s{s}"] = K / f"galician-seed{s}/test_seed{s}/runs/galician-mix-seed{s}_plain__none.jsonl"
    EXTRA[f"run4-s{s}"] = lambda t, s=s: K / f"galician-seed{s}/e_seed{s}/runs/{t}__galician-seed{s}_plain__none.jsonl"
    EXTRA[f"mix-s{s}"] = lambda t, s=s: K / f"galician-seed{s}/e_seed{s}/runs/{t}__galician-mix-seed{s}_plain__none.jsonl"

questions = {str(q["id"]): q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def per_query(task, model):
    rs = rows(DOG[model] if task == "dog" else EXTRA[model]({"news_doc": "news", "news_par": "news"}.get(task, task)))
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
    return float((tp / np.arange(1, len(lab) + 1) * lab).sum() / max(lab.sum(), 1))


def paws(model):
    rs = rows(EXTRA[model]("paws"))
    return [r["id"] for r in rs], np.array([r["score"] for r in rs]), np.array([r["label"] for r in rs])


def paws_boot(a, b, samples=2000, seed=0):
    ia, sa, la = paws(a)
    ib, sb, lb = paws(b)
    assert ia == ib and (la == lb).all()
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for s in range(samples):
        idx = rng.integers(0, len(la), len(la))
        boot[s] = ap(sa[idx], la[idx]) - ap(sb[idx], la[idx])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mrr_a": round(ap(sa, la), 4), "mrr_b": round(ap(sb, la), 4), "diff": round(ap(sa, la) - ap(sb, la), 4),
            "ci95": [round(float(np.quantile(boot, q)), 4) for q in (0.025, 0.975)], "p_two_sided": round(float(min(1.0, p)), 4)}


def boot(task, a, b):
    if task == "paws":
        return paws_boot(a, b)
    ia, va = per_query(task, a)
    ib, vb = per_query(task, b)
    assert ia == ib, (task, a, b)
    return paired_bootstrap(ia, va, vb, 10000, 0)


def score(task, model):
    if task == "paws":
        _, s, l = paws(model)
        return ap(s, l)
    return float(per_query(task, model)[1].mean())


TASKS = ("dog", "news_doc", "news_par", "paws", "webfaq")
for fam in (("run4", "run4-s1", "run4-s2"), ("mix", "mix-s1", "mix-s2"), ("nosinai",)):
    for task in TASKS:
        v = [score(task, m) for m in fam]
        print(json.dumps({"summary": task, "models": fam, "values": [round(x, 4) for x in v],
                          "mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None,
                          "range": round(max(v) - min(v), 4)}))

pairs = []
for s in ("s1", "s2"):
    pairs += [("dog", f"mix-{s}", "nos"), ("dog", f"mix-{s}", "bm25"), ("dog", f"mix-{s}", "mix"), ("dog", f"run4-{s}", "run4")]
pairs += [("dog", "nosinai", "run4"), ("dog", "nosinai", "nos"), ("dog", "nosinai", "bm25")]
for task in ("news_doc", "news_par", "paws", "webfaq"):
    pairs += [(task, "nosinai", "run4"), (task, "nosinai", "bge-m3")]
    pairs += [(task, f"mix-{s}", "bge-m3") for s in ("s1", "s2")]
for task, a, b in pairs:
    print(json.dumps({"task": task, "a": a, "b": b, **boot(task, a, b)}))
