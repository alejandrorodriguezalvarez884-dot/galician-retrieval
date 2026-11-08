"""Kaggle cells for eval_extra.py (nos-rag-news, PAWS-gl, WebFAQ glg held out).

Same layout as the DOG baselines: one process per model, one queue per T4 through
CUDA_VISIBLE_DEVICES, and the LLM rerankers last in fresh processes on the saved first
stages. Cells are one line each so they can be typed into Kaggle's editor.

    python extra_cells.py DIR                                   # every baseline (~1 h on T4 x2)
    python extra_cells.py DIR --jina                            # plus jina-reranker-v3 (~+40 min)
    python extra_cells.py DIR --model PATH [--reranker PATH]    # only our checkpoints (~10 min)

With --model or --reranker, append the cells to the training notebook: they reuse its pip
install, so only the payload and run cells are written.
"""

import argparse
from pathlib import Path

from build_notebooks import OUT, PINS, TF4_PINS, scripts_cell

SCRIPTS = ["eval_extra.py", "eval_dog.py", "gen_queries.py"]
ALLOC = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
FIRST = f"{OUT}/e_gpu1/cache"

GPU_QUEUES = {
    0: [
        "--retrievers embeddinggemma-2 --modes prompted --batch-size 8",
        "--retrievers qwen3-embedding-0.6b --batch-size 8",
    ],
    1: [
        "--retrievers bm25,bge-m3,multilingual-e5-base --rerankers bge-reranker-v2-m3 --rerank-on bm25,bge-m3",
        "--retrievers arctic-embed-l-v2.0,granite-embedding-311m-r2,granite-embedding-97m-r2 --batch-size 16",
        "--retrievers mrbert-nos-gl --batch-size 16",
    ],
}


def queue(gpu: int, calls: list[str], log: str) -> str:
    joined = "; ".join(calls)
    return f'{ALLOC} CUDA_VISIBLE_DEVICES={gpu} sh -c "{joined}" > {log} 2>&1'


def table_cell() -> str:
    key = "r.get('MRR', r.get('nDCG@10', r.get('AP_cosine', 0)))"
    return (
        '!python -c "import json, glob; '
        f"R = [json.loads(l) for f in sorted(glob.glob('{OUT}/e_*/results.jsonl')) for l in open(f)]; "
        f"[print(r['task'], r['retriever'], r['reranker'], {key}, r.get('MRR_paragraph', ''), r.get('error', '')[:100]) "
        f"for r in sorted(R, key=lambda r: (r['task'], -({key} or 0)))]\""
    )


def baseline_cells(jina: bool) -> list[str]:
    ex = "python eval_extra.py"
    tf4 = "/tmp/tf4/bin/python eval_extra.py"
    first = [queue(g, [f"{ex} {a} --out {OUT}/e_gpu{g}" for a in runs], f"extra_gpu{g}.log") for g, runs in GPU_QUEUES.items()]
    gpu0 = [f"{ex} --first-stage-from {FIRST} --rerankers qwen3-reranker-0.6b --rerank-on bm25,bge-m3 --out {OUT}/e_qwen3rr"]
    if jina:
        gpu0.append(f"{ex} --first-stage-from {FIRST} --rerankers jina-reranker-v3 --rerank-on bm25 --out {OUT}/e_jina")
    gpu1 = [
        f"{ex} --first-stage-from {FIRST} --rerankers mxbai-rerank-base-v2 --rerank-on bm25,bge-m3 --out {OUT}/e_mxbai",
        f"{tf4} --retrievers gte-multilingual-base --batch-size 16 --out {OUT}/e_tf4",
        f"{tf4} --first-stage-from {FIRST} --rerankers gte-multilingual-reranker-base --rerank-on bm25,bge-m3 --out {OUT}/e_tf4",
    ]
    second = [queue(0, gpu0, "extra_rr0.log"), queue(1, gpu1, "extra_rr1.log")]
    return [
        f"!pip install -q {PINS} uv && uv venv -q --system-site-packages /tmp/tf4 --python $(which python) && "
        f"uv pip install -q --python /tmp/tf4/bin/python {TF4_PINS}",
        scripts_cell(SCRIPTS),
        f"!{first[0]} & {first[1]} & wait; {second[0]} & {second[1]} & wait; grep -h -E 'task|Error' extra_*.log | grep -v first_stages",
        table_cell(),
    ]


def model_cells(model: str | None, reranker: str | None, name: str) -> list[str]:
    retrievers = "bm25" + (f",{name}={model}" if model else "")
    rerank_on = "bm25" + (f",{name}" if model else "")
    rerankers = "bge-reranker-v2-m3" + (f",ce:{reranker}" if reranker else "")
    run = (
        f"!{ALLOC} python eval_extra.py --retrievers {retrievers} --rerankers {rerankers} "
        f"--rerank-on {rerank_on} --out {OUT}/e_{name} 2>&1 | grep --line-buffered -E '^\\{{'"
    )
    return [scripts_cell(SCRIPTS), run, table_cell()]


def bar_cells(run: str = "run4") -> list[str]:
    """galician-eval-extra: the second half of the publish bar for one checkpoint.

    "No worse than bge-m3 on nos-rag-news and PAWS-gl by more than 2 points", judged with bootstrap
    intervals. Embedders only. GPU 0 runs bge-m3; GPU 1 runs BM25, our checkpoint and its base
    (mE5-base), so the table also shows what fine-tuning on DOG did to general Galician.
    Needs the training notebook's output attached (its final/ folder).
    """
    from build_notebooks import final_dir

    ex = "python eval_extra.py"
    gpu0 = queue(0, [f"{ex} --retrievers bge-m3 --out {OUT}/e_gpu0"], "extra_gpu0.log")
    gpu1 = queue(
        1,
        [f"{ex} --retrievers bm25,galician-{run}={final_dir(run)},multilingual-e5-base --out {OUT}/e_gpu1"],
        "extra_gpu1.log",
    )
    return [
        f"!pip install -q {PINS} && ls -d {final_dir(run)}",
        scripts_cell(SCRIPTS),
        f"!{gpu0} & {gpu1} & wait; grep -h -E '\"task\"|rror' extra_gpu*.log | grep -v first_stages; rm -rf {OUT}/e_*/cache",
        table_cell(),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--jina", action="store_true", help="add jina-reranker-v3 on BM25 (CC BY-NC, reference row only)")
    ap.add_argument("--model", help="our bi-encoder checkpoint on Kaggle, e.g. /kaggle/working/run1/final")
    ap.add_argument("--reranker", help="our cross-encoder checkpoint on Kaggle")
    ap.add_argument("--name", default="galician-run1")
    ap.add_argument("--bar", metavar="RUN", help="news + PAWS-gl + WebFAQ for RUN against bge-m3 (its own notebook)")
    args = ap.parse_args()

    if args.bar:
        cells = bar_cells(args.bar)
    elif args.model or args.reranker:
        cells = model_cells(args.model, args.reranker, args.name)
    else:
        cells = baseline_cells(args.jina)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cells):
        (out / f"extra_cell_{i}.txt").write_text(c, encoding="utf-8")
        print(i, len(c), c[:220])
