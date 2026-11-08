"""Error analysis on DOG-RAG from per-query run files (eval_dog.py's runs/*.jsonl).

For each run: MRR by category and by question type (has a number, has a code such as a
procedure or decree number, neither). Then, for pairs of runs, the questions one gets right
(MRR 1) and the other misses (MRR 0), with the question, the expected document and the top
chunk each run returned.

    python error_analysis.py --run NAME=path/to/run.jsonl ... --pair A,B ... --out error-analysis.md
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import eval_dog  # noqa: E402

# Codes: procedure codes (PR471A, BS403A), file numbers, decree/law numbers (11/2001), dates.
CODE = re.compile(r"\b[A-Z]{1,4}\d{2,}[A-Z]?\b|\b\d+/\d{2,4}\b|\b\d{4}-\d+\b")
NUMBER = re.compile(r"\d")


def qtype(text: str) -> str:
    if CODE.search(text):
        return "code"
    if NUMBER.search(text):
        return "number"
    return "words"


def read_run(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=path to a runs/*.jsonl file")
    ap.add_argument("--pair", action="append", default=[], help="A,B: list questions A gets and B misses, and back")
    ap.add_argument("--per-cell", type=int, default=4, help="examples per category and direction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inputs = SimpleNamespace(dog2025=None, dograg_dir=None, corpus_revision=None, dograg_revision=None,
                             max_distractors=None, max_queries=None, seed=0)
    chunks, queries = eval_dog.load_inputs(inputs)
    text = {(c["doc_id"], c["pos"]): c["text"] for c in chunks}
    # Rows are paired by position: DOG-RAG has two entries with the same uid (478), so ids repeat.
    qs = dict(enumerate(queries))
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = read_run(path)
        if [r["id"] for r in runs[name]] != [q["id"] for q in queries]:
            raise SystemExit(f"{name}: rows don't match the query order")

    cats = sorted({q["category"] for q in queries})
    types = ["code", "number", "words"]
    count = defaultdict(int)
    for q in queries:
        count[q["category"]] += 1
        count[qtype(q["text"])] += 1

    out = ["# DOG-RAG error analysis", "",
           f"917 questions. Question types: code {count['code']}, number (no code) {count['number']}, "
           f"words only {count['words']}. Categories: " + ", ".join(f"{c} {count[c]}" for c in cats) + ".", "",
           "## MRR by category and question type", "",
           "| Run | " + " | ".join(cats) + " | " + " | ".join(types) + " | all |",
           "|---" * (len(cats) + len(types) + 2) + "|"]
    summary = {}
    for name, run in runs.items():
        by = defaultdict(list)
        for qid, q in qs.items():
            m = run[qid]["mrr"]
            by[q["category"]].append(m)
            by[qtype(q["text"])].append(m)
            by["all"].append(m)
        row = {k: sum(v) / len(v) for k, v in by.items()}
        summary[name] = row
        out.append(f"| {name} | " + " | ".join(f"{row[c]:.3f}" for c in cats + types + ["all"]) + " |")

    rng = random.Random(args.seed)
    for pair in args.pair:
        a, b = pair.split(",")
        out += ["", f"## {a} vs {b}", ""]
        wins = defaultdict(list)
        for qid, q in qs.items():
            ma, mb = runs[a][qid]["mrr"], runs[b][qid]["mrr"]
            if ma == 1.0 and mb == 0.0:
                wins[(a, q["category"])].append(qid)
            elif mb == 1.0 and ma == 0.0:
                wins[(b, q["category"])].append(qid)
        out.append(f"| Category | {a} right, {b} misses | {b} right, {a} misses |")
        out.append("|---|---|---|")
        for c in cats:
            out.append(f"| {c} | {len(wins[(a, c)])} | {len(wins[(b, c)])} |")
        for winner, loser in ((a, b), (b, a)):
            out += ["", f"### {winner} right, {loser} misses", ""]
            for c in cats:
                ids = sorted(wins[(winner, c)])
                for qid in rng.sample(ids, min(args.per_cell, len(ids))):
                    q = qs[qid]
                    got = runs[loser][qid]["final"][0]
                    out.append(f"- **{c}, {qtype(q['text'])}** `{q['id']}`: {q['text']}")
                    out.append(f"  - expected `{q['doc_id']}`; {loser} returned `{got[0]}` #{got[1]}: "
                               f"\"{text[tuple(got)][:220].strip()}…\"")

    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "runs": list(runs), "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
