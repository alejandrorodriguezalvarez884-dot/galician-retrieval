"""Holm correction over the bootstrap p-values the paper reports.

The four families and the rule for entering values (holm.md) were fixed before this ran.
Every p-value is listed in ROWS with the table it is printed in and its source. Where both
per-query files are in results/ the script also recomputes:

- "<0.001" rows: replaced by the numpy bootstrap value (a value of 0 enters as 2/(B+1));
- "--" rows (interval reported, no p): computed;
- a few exact rows: recomputed as a check against the printed value.

Rows whose per-query files were not kept (runs 1-3, baselines; only their logged results
exist) keep the printed value; "<0.001" without files enters
as 0.001. Both are upper bounds, so the adjusted values stay conservative.

    python analysis/holm.py --out holm-results.md
"""
import argparse
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
G = UP / "gcp"
MAC = UP / "mac/pipeline/runs"
VARIANCE = UP / "boot_variance.jsonl"
QUESTIONS = Path(hf_hub_download("proxectonos/nos-rag-news", "nos-rag-dataset_questions.json", repo_type="dataset",
                             revision="b721e0e502dde94a71ad36625a76dac0f1af5c88"))
SAMPLES = 10000
PAWS_SAMPLES = 2000

DOG = {
    "bm25-kaggle": K / "galician-seed1/test_seed1/runs/bm25__none.jsonl",  # 0.7823, as in the M rows
    "bm25-gcp": G / "run5/test_run5/runs/bm25__none.jsonl",  # 0.7799, as in the N rows
    "bge-m3": G / "rerank1/test_rerank1/runs/bge-m3_plain__none.jsonl",
    "bge-m3+ours": G / "rerank1/test_rerank1/runs/bge-m3_plain__ce_rerank1_final.jsonl",
    "nos": G / "rerank1/test_rerank1/runs/bge-m3_plain__bge-reranker-v2-m3.jsonl",
    "run4": K / "boot5/train4/test_run4/runs/galician-run4_plain__none.jsonl",
    "mix": K / "boot5/mix/t_mix/runs/galician-mix_plain__none.jsonl",
    "run5": G / "run5/test_run5/runs/galician-run5_plain__none.jsonl",
    "hybrid-run5": G / "run5/test_run5/runs/hybrid_galician-run5-plain__none.jsonl",
    "run5+bge-rr": G / "run5/test_run5/runs/galician-run5_plain__bge-reranker-v2-m3.jsonl",
    "bm25-rr1": G / "rerank1/test_rerank1/runs/bm25__none.jsonl",
    "bm25+ours": G / "rerank1/test_rerank1/runs/bm25__ce_rerank1_final.jsonl",
    "bm25+bge-rr": G / "rerank1/test_rerank1/runs/bm25__bge-reranker-v2-m3.jsonl",
    "bm25+gte-rr": K / "train4_gte/b_gte/runs/bm25__gte-multilingual-reranker-base.jsonl",
    # Apple MPS pipeline run (2026-11-06)
    "mix-mac": MAC / "galician-mix_plain__none.jsonl",
    "mix-mac+ours": MAC / "galician-mix_plain__ce_upstream_gcp_rerank1_rerank1_final.jsonl",
    "hybrid-mix-mac": MAC / "hybrid_galician-mix-plain__none.jsonl",
    "run5-mac": MAC / "galician-run5_plain__none.jsonl",
    "run5-mac+ours": MAC / "galician-run5_plain__ce_upstream_gcp_rerank1_rerank1_final.jsonl",
}
EXTRA = {
    "run4": lambda t: K / f"boot5/extra/e_gpu1/runs/{t}__galician-run4_plain__none.jsonl",
    "mix": lambda t: K / f"boot5/mix/e_mix/runs/{t}__mix-a09_plain__none.jsonl",
    "bge-m3": lambda t: K / f"boot5/extra/e_gpu0/runs/{t}__bge-m3_plain__none.jsonl",
    "run5": lambda t: G / f"run5/e_run5/runs/{t}__galician-run5_plain__none.jsonl",
}

T_DOG, T_SEC, T_VAR, FACTS = "bootstrap-dog", "bootstrap-secondary", "variance", "text"
N11 = "log:"

# (family, table, label, printed p, source of the printed value, recompute spec or None)
# spec: (task, a, b) with task in dog / news_doc / news_par / webfaq / paws,
#   or ("jsonl", task, a, b) for a row of boot_variance.jsonl.
ROWS = [
    # A. Headline, DOG-RAG
    ("A", T_DOG, "run 4 - BM25", "<0.001", N11 + " run 4 entry (M)", ("dog", "run4", "bm25-kaggle")),
    ("A", T_DOG, "run 4 - bge-m3", "<0.001", N11 + " run 4 entry (M)", ("dog", "run4", "bge-m3")),
    ("A", T_DOG, "run 4 - gte-multilingual-base", "<0.001", N11 + " run 4 entry (M)", None),
    ("A", T_DOG, "hybrid run 4 - hybrid bge-m3", "<0.001", N11 + " run 4 entry (M)", None),
    ("A", T_DOG, "mix 0.9 - BM25", "<0.001", N11 + " WiSE-FT entry (M)", ("dog", "mix", "bm25-kaggle")),
    ("A", T_DOG, "mix 0.9 - bge-m3", "<0.001", N11 + " WiSE-FT entry (M)", ("dog", "mix", "bge-m3")),
    ("A", T_DOG, "run 5 - BM25", "<0.001", N11 + " run 5 results (N)", ("dog", "run5", "bm25-gcp")),
    ("A", T_DOG, "run 5 - bge-m3", "<0.001", "recomputed for tables.tex (N)", ("dog", "run5", "bge-m3")),
    ("A", T_DOG, "run 5 - BM25 + ours", "0.005", "recomputed for tables.tex (N)", ("dog", "run5", "bm25+ours")),
    ("A", T_DOG, "mix 0.9 - BM25 + ours", "0.018", "recomputed for tables.tex (N)", ("dog", "mix", "bm25+ours")),
    ("A", T_DOG, "mix 0.9 - bge-m3 + bge-rr (Nos pipeline, MPS)", "0.024", "results/mac/pipeline/runs (N)", ("dog", "mix-mac", "nos")),
    ("A", T_DOG, "mix 0.9 - BM25 + bge-rr", "0.007", "results/mac/pipeline/runs (N)", ("dog", "mix-mac", "bm25+bge-rr")),
    ("A", T_DOG, "mix 0.9 + ours - bge-m3 + bge-rr", "0.32", "results/mac/pipeline/runs (N)", ("dog", "mix-mac+ours", "nos")),
    ("A", T_VAR, "run 4 seed 0 - Nos pipeline", "0.048", N11 + " 2026-11-07 seed entry (N)", ("dog", "run4", "nos")),
    ("A", T_VAR, "mix 0.9 seed 0 - Nos pipeline (Kaggle)", "0.029", N11 + " 2026-11-07 seed entry (N)", ("dog", "mix", "nos")),
    ("A", T_VAR, "mix 0.9 seed 1 - Nos pipeline", "0.064", "boot_variance.jsonl (N)", ("jsonl", "dog", "mix-s1", "nos")),
    ("A", T_VAR, "mix 0.9 seed 2 - Nos pipeline", "0.033", "boot_variance.jsonl (N)", ("jsonl", "dog", "mix-s2", "nos")),
    ("A", T_VAR, "no SINAI - Nos pipeline", "0.088", "boot_variance.jsonl (N)", ("jsonl", "dog", "nosinai", "nos")),
    # B. Development and ablations, DOG-RAG
    ("B", T_DOG, "hybrid run 2 - BM25", "0.003", N11 + " hybrid results (M)", None),
    ("B", T_DOG, "hybrid run 2 - hybrid bge-m3", "0.10", N11 + " hybrid results (M)", None),
    ("B", T_DOG, "hybrid run 1 - hybrid bge-m3", "0.92", N11 + " hybrid results (M)", None),
    ("B", T_DOG, "hybrid run 2 - hybrid run 1", "0.007", N11 + " hybrid results (M)", None),
    ("B", T_DOG, "hybrid bge-m3 - BM25", "0.18", N11 + " hybrid results (M)", None),
    ("B", T_DOG, "hybrid run 1 + bge-rr - hybrid bge-m3 + bge-rr", "0.43", N11 + " hybrid results (M)", None),
    ("B", T_DOG, "run 3 - BM25", "0.054", N11 + " run 3 entry (M)", None),
    ("B", T_DOG, "run 3 - bge-m3", "<0.001", N11 + " run 3 entry (M)", None),
    ("B", T_DOG, "run 3 - run 2", "<0.001", N11 + " run 3 entry (M)", None),
    ("B", T_DOG, "hybrid run 3 - hybrid bge-m3", "<0.001", N11 + " run 3 entry (M)", None),
    ("B", T_DOG, "hybrid run 3 - hybrid run 2", "0.053", N11 + " run 3 entry (M)", None),
    ("B", T_DOG, "hybrid run 3 - BM25", "<0.001", N11 + " run 3 entry (M)", None),
    ("B", T_DOG, "hybrid run 3 + bge-rr - hybrid bge-m3 + bge-rr", "0.55", N11 + " run 3 entry (M)", None),
    ("B", T_DOG, "run 4 - run 3", "0.001", N11 + " run 4 entry (M)", None),
    ("B", T_DOG, "hybrid run 4 - run 4", "0.74", N11 + " run 4 entry (M)", None),
    ("B", T_DOG, "mix 0.9 - run 4", "0.64", N11 + " WiSE-FT entry (M)", ("dog", "mix", "run4")),
    ("B", T_DOG, "run 5 - run 4", "0.37", N11 + " run 5 bootstraps (N)", ("dog", "run5", "run4")),
    ("B", T_DOG, "run 5 - mix 0.9", "0.50", N11 + " run 5 bootstraps (N)", ("dog", "run5", "mix")),
    ("B", T_DOG, "hybrid BM25 + mix 0.9 - mix 0.9", "0.49", "results/mac/pipeline/runs (N)", ("dog", "hybrid-mix-mac", "mix-mac")),
    ("B", FACTS, "hybrid run 5 - run 5", "0.71", "recomputed for tables.tex (N)", ("dog", "hybrid-run5", "run5")),
    ("B", T_VAR, "no SINAI - run 4 (DOG-RAG)", "0.68", "boot_variance.jsonl (N)", ("jsonl", "dog", "nosinai", "run4")),
    # C. Secondary tests
    ("C", T_SEC, "news doc: run 4 - bge-m3", "--", "interval only (M); p computed here", ("news_doc", "run4", "bge-m3")),
    ("C", T_SEC, "news par.: run 4 - bge-m3", "--", "interval only (M); p computed here", ("news_par", "run4", "bge-m3")),
    ("C", T_SEC, "PAWS-gl: run 4 - bge-m3", "--", "interval only (M); p computed here", ("paws", "run4", "bge-m3")),
    ("C", T_SEC, "news doc: mix 0.9 - bge-m3", "0.89", N11 + " WiSE-FT entry (M)", None),
    ("C", T_SEC, "news par.: mix 0.9 - bge-m3", "0.06", N11 + " WiSE-FT entry (M)", None),
    ("C", T_SEC, "PAWS-gl: mix 0.9 - bge-m3", "--", "interval only (M); p computed here", ("paws", "mix", "bge-m3")),
    ("C", T_SEC, "WebFAQ: mix 0.9 - bge-m3", "<0.001", N11 + " WiSE-FT entry (M)", ("webfaq", "mix", "bge-m3")),
    ("C", T_SEC, "news doc: mix 0.5 - bge-m3", "0.45", N11 + " WiSE-FT entry (M)", None),
    ("C", T_SEC, "news par.: mix 0.5 - bge-m3", "0.19", N11 + " WiSE-FT entry (M)", None),
    ("C", T_SEC, "news doc: run 5 - bge-m3", "0.50", "analysis/boot_run5.py (N)", ("news_doc", "run5", "bge-m3")),
    ("C", T_SEC, "news doc: run 5 - run 4", "0.83", "analysis/boot_run5.py (N)", ("news_doc", "run5", "run4")),
    ("C", T_SEC, "news doc: run 5 - mix 0.9", "0.53", "analysis/boot_run5.py (N)", ("news_doc", "run5", "mix")),
    ("C", T_SEC, "news par.: run 5 - bge-m3", "0.006", "analysis/boot_run5.py (N)", ("news_par", "run5", "bge-m3")),
    ("C", T_SEC, "news par.: run 5 - run 4", "0.63", "analysis/boot_run5.py (N)", ("news_par", "run5", "run4")),
    ("C", T_SEC, "news par.: run 5 - mix 0.9", "0.17", "analysis/boot_run5.py (N)", ("news_par", "run5", "mix")),
    ("C", T_SEC, "PAWS-gl: run 5 - bge-m3", "0.98", "analysis/boot_run5.py (N)", ("paws", "run5", "bge-m3")),
    ("C", T_SEC, "PAWS-gl: run 5 - run 4", "0.32", "analysis/boot_run5.py (N)", ("paws", "run5", "run4")),
    ("C", T_SEC, "PAWS-gl: run 5 - mix 0.9", "0.06", "analysis/boot_run5.py (N)", ("paws", "run5", "mix")),
    ("C", T_SEC, "WebFAQ: run 5 - bge-m3", "<0.001", "analysis/boot_run5.py (N)", ("webfaq", "run5", "bge-m3")),
    ("C", T_SEC, "WebFAQ: run 5 - run 4", "0.02", "analysis/boot_run5.py (N)", ("webfaq", "run5", "run4")),
    ("C", T_SEC, "WebFAQ: run 5 - mix 0.9", "0.20", "analysis/boot_run5.py (N)", ("webfaq", "run5", "mix")),
    ("C", T_VAR, "news par.: mix 0.9 seed 1 - bge-m3", "0.008", "boot_variance.jsonl (N)", ("jsonl", "news_par", "mix-s1", "bge-m3")),
    ("C", T_VAR, "news par.: mix 0.9 seed 2 - bge-m3", "0.027", "boot_variance.jsonl (N)", ("jsonl", "news_par", "mix-s2", "bge-m3")),
    ("C", T_VAR, "WebFAQ: no SINAI - run 4", "0.13", "boot_variance.jsonl (N)", ("jsonl", "webfaq", "nosinai", "run4")),
    ("C", T_VAR, "news doc: no SINAI - run 4", ">0.4", "boot_variance.jsonl (N)", ("jsonl", "news_doc", "nosinai", "run4")),
    ("C", T_VAR, "news par.: no SINAI - run 4", ">0.4", "boot_variance.jsonl (N)", ("jsonl", "news_par", "nosinai", "run4")),
    ("C", T_VAR, "PAWS-gl: no SINAI - run 4", ">0.4", "boot_variance.jsonl (N)", ("jsonl", "paws", "nosinai", "run4")),
    # D. Rerankers, DOG-RAG
    ("D", T_DOG, "BM25 + ours - BM25 + bge-rr", "0.65", N11 + " reranker rerun (N)", ("dog", "bm25+ours", "bm25+bge-rr")),
    ("D", T_DOG, "bge-m3 + ours - bge-m3 + bge-rr", "0.80", N11 + " reranker rerun (N)", ("dog", "bge-m3+ours", "nos")),
    ("D", T_DOG, "BM25 + ours - BM25", "0.028", N11 + " reranker rerun (N)", ("dog", "bm25+ours", "bm25-rr1")),
    ("D", T_DOG, "bge-m3 + ours - bge-m3", "<0.001", N11 + " reranker rerun (N)", ("dog", "bge-m3+ours", "bge-m3")),
    ("D", T_DOG, "bge-m3 + ours - BM25 + bge-rr", "0.71", N11 + " reranker rerun (N)", ("dog", "bge-m3+ours", "bm25+bge-rr")),
    ("D", T_DOG, "BM25 + ours - BM25 + gte-rr", "0.34", "recomputed for tables.tex (N)", ("dog", "bm25+ours", "bm25+gte-rr")),
    ("D", T_DOG, "BM25 + bge-rr - BM25 + gte-rr", "0.56", "recomputed for tables.tex (N)", ("dog", "bm25+bge-rr", "bm25+gte-rr")),
    ("D", T_DOG, "BM25 + gte-rr - BM25", "0.19", "recomputed for tables.tex (N)", ("dog", "bm25+gte-rr", "bm25-rr1")),
    ("D", T_DOG, "mix 0.9 + ours - mix 0.9", "0.13", "results/mac/pipeline/runs (N)", ("dog", "mix-mac+ours", "mix-mac")),
    ("D", T_DOG, "run 5 + ours - run 5", "0.13", "results/mac/pipeline/runs (N)", ("dog", "run5-mac+ours", "run5-mac")),
    ("D", FACTS, "run 5 + bge-rr - run 5", "0.26", "recomputed for tables.tex (N)", ("dog", "run5+bge-rr", "run5")),
]

FAMILIES = {
    "A": "Headline, DOG-RAG: final embedders vs baselines and pipelines",
    "B": "Development and ablations, DOG-RAG",
    "C": "Secondary tests: news, PAWS-gl, WebFAQ",
    "D": "Rerankers, DOG-RAG",
}


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


_questions = None


def per_query(task, model):
    global _questions
    if task == "dog":
        rs = rows(DOG[model])
    else:
        rs = rows(EXTRA[model]({"news_doc": "news", "news_par": "news"}.get(task, task)))
    ids = [str(r["id"]) for r in rs]
    if task in ("dog", "news_doc"):
        return ids, np.array([r["mrr"] for r in rs], float)
    if task == "news_par":
        if _questions is None:
            _questions = {str(q["id"]): q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
        vals = []
        for r in rs:
            q = _questions[str(r["id"])]
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


def paws_p(a, b):
    ra, rb = rows(EXTRA[a]("paws")), rows(EXTRA[b]("paws"))
    assert [r["id"] for r in ra] == [r["id"] for r in rb]
    la = np.array([r["label"] for r in ra])
    sa = np.array([r["score"] for r in ra])
    sb = np.array([r["score"] for r in rb])
    rng = np.random.default_rng(0)
    boot = np.empty(PAWS_SAMPLES)
    for s in range(PAWS_SAMPLES):
        idx = rng.integers(0, len(la), len(la))
        boot[s] = ap(sa[idx], la[idx]) - ap(sb[idx], la[idx])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(min(1.0, p)), PAWS_SAMPLES


def recompute(spec):
    """Two-sided bootstrap p and the number of samples behind it."""
    if spec[0] == "jsonl":
        _, task, a, b = spec
        for r in rows(VARIANCE):
            if r.get("task") == task and r.get("a") == a and r.get("b") == b:
                return r["p_two_sided"], PAWS_SAMPLES if task == "paws" else SAMPLES
        raise KeyError(spec)
    task, a, b = spec
    if task == "paws":
        return paws_p(a, b)
    ia, va = per_query(task, a)
    ib, vb = per_query(task, b)
    assert ia == ib, spec
    return paired_bootstrap(ia, va, vb, SAMPLES, 0)["p_two_sided"], SAMPLES


def entered_value(printed, source, spec):
    """(p entered, is an upper bound, note).

    N rows with files use the recomputed value (same code, more decimals); M rows keep the
    printed value, which the numpy recomputation only checks, unless it was "<0.001" or "--".
    """
    if spec is not None:
        p, n = recompute(spec)
        if p == 0:
            return 2 / (n + 1), True, f"recomputed 0 of {n:,}"
        if printed.startswith("<"):
            assert p < float(printed[1:]), (printed, p)
            return p, False, "recomputed"
        if printed.startswith(">"):
            assert p > float(printed[1:]), (printed, p)
            return p, False, "exact value from file"
        if printed == "--":
            return p, False, "computed (N)"
        decimals = len(printed.split(".")[1])
        assert abs(round(p, decimals) - float(printed)) <= 10 ** -decimals + 1e-9, (printed, p)
        if source.endswith("(N)"):
            return p, False, "recomputed (N)"
        return float(printed), False, f"printed (M); N check {p:.4f}"
    if printed.startswith("<"):
        return float(printed[1:]), True, "bound, no files"
    return float(printed), False, "printed"


def holm(ps, bounds):
    """Holm step-down adjusted p-values; flags values driven by a bound."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, is_bound = [0.0] * m, [False] * m
    best, best_bound = 0.0, False
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * ps[i])
        if val >= best:
            best, best_bound = val, bounds[i]
        adj[i], is_bound[i] = best, best_bound
    return adj, is_bound


def fmt(p, bound=False):
    if p >= 0.9995:
        s = "1"
    elif p < 0.001:
        s = f"{p:.4f}" if bound else "<0.001"
    elif p >= 0.1:
        s = f"{p:.2f}"
    else:
        s = f"{p:.3f}"
    return ("≤" + s) if bound else s


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--out", type=Path, help="write the markdown report here as well")
    args = ap_.parse_args()

    entered = [entered_value(r[3], r[4], r[5]) for r in ROWS]
    ps = [e[0] for e in entered]
    bounds = [e[1] for e in entered]
    fam_adj, fam_bound = [0.0] * len(ROWS), [False] * len(ROWS)
    for fam in FAMILIES:
        idx = [i for i, r in enumerate(ROWS) if r[0] == fam]
        adj, isb = holm([ps[i] for i in idx], [bounds[i] for i in idx])
        for k, i in enumerate(idx):
            fam_adj[i], fam_bound[i] = adj[k], isb[k]
    all_adj, all_bound = holm(ps, bounds)

    out = ["# Holm-adjusted p-values (generated by holm.py)", "",
           f"Rule and families: `holm.md`. {len(ROWS)} p-values; alpha 0.05 family-wise. "
           "'≤' marks an adjusted value set by an upper bound (see holm.md section 3). "
           "'Raw used' is the value entered; 'Printed' is what the table shows.", ""]
    for fam, title in FAMILIES.items():
        idx = [i for i, r in enumerate(ROWS) if r[0] == fam]
        idx.sort(key=lambda i: ps[i])
        n_raw = sum(ps[i] < 0.05 for i in idx)
        n_fam = sum(fam_adj[i] < 0.05 for i in idx)
        n_all = sum(all_adj[i] < 0.05 for i in idx)
        out += [f"## {fam}. {title} (m = {len(idx)})", "",
                f"p < 0.05: raw {n_raw}, Holm in family {n_fam}, Holm over all {len(ROWS)}: {n_all}.", "",
                "| Comparison (A - B) | Table | Printed | Raw used | Holm (family) | Holm (all) | Entered as |",
                "|---|---|---|---|---|---|---|"]
        for i in idx:
            r = ROWS[i]
            out.append(f"| {r[2]} | {r[1]} | {r[3]} | {fmt(ps[i], bounds[i])} | {fmt(fam_adj[i], fam_bound[i])} "
                       f"| {fmt(all_adj[i], all_bound[i])} | {entered[i][2]} |")
        out.append("")
    changed = [i for i in range(len(ROWS)) if ps[i] < 0.05 and fam_adj[i] >= 0.05]
    changed_all = [i for i in range(len(ROWS)) if ps[i] < 0.05 and all_adj[i] >= 0.05 and fam_adj[i] < 0.05]
    out += ["## Raw p < 0.05 that lose significance", "", "Within their family:", ""]
    out += [f"- {ROWS[i][0]}: {ROWS[i][2]} (raw {fmt(ps[i])}, Holm {fmt(fam_adj[i], fam_bound[i])})" for i in changed]
    out += ["", "Only when all p-values form one family:", ""]
    out += [f"- {ROWS[i][0]}: {ROWS[i][2]} (raw {fmt(ps[i])}, family {fmt(fam_adj[i], fam_bound[i])}, "
            f"all {fmt(all_adj[i], all_bound[i])})" for i in changed_all] or ["- none"]
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
