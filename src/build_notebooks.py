"""Build the Kaggle notebooks from the scripts in this folder.

The scripts are embedded with %%writefile so the notebook runs without access
to this repository. Run after editing any script:

    python build_notebooks.py
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).parent

PINS = '"transformers==5.22.1" "sentence-transformers==6.2.0" "bm25s==0.3.16"'

OUT = "/kaggle/working"

# One process per model, so a model that breaks the CUDA context (remote code) can't take the
# others down and each gets its own batch size. The two T4s run separate queues in parallel
# (CUDA_VISIBLE_DEVICES); sentence-transformers' multi-GPU encode hung on Kaggle.
# gte-multilingual-base is left out: its remote code calls get_extended_attention_mask,
# removed in transformers 5. embeddinggemma-2 runs in fp32 (no fp16 support), the slowest row.
# The LLM rerankers run last, in their own processes, on the first stages saved by the
# bm25 + bge-m3 run (--first-stage-from): in the same process as bge-m3 they ran out of memory.
FIRST = f"{OUT}/b_gpu1/cache"
GPU_QUEUES = {
    0: [
        "--retrievers embeddinggemma-2 --modes prompted --batch-size 8",
        "--retrievers qwen3-embedding-0.6b --batch-size 8",
        f"--first-stage-from {FIRST} --rerankers qwen3-reranker-0.6b --rerank-on bm25,bge-m3",
    ],
    1: [
        "--retrievers bm25,bge-m3,multilingual-e5-base --rerankers bge-reranker-v2-m3 --rerank-on bm25,bge-m3",
        "--retrievers mrbert-nos-gl --batch-size 16",
        f"--first-stage-from {FIRST} --rerankers jina-reranker-v3 --rerank-on bm25,bge-m3",
    ],
}



def rerank_only_cell() -> str:
    """Version 5 of the Kaggle notebook: only the two LLM rerankers that ran out of memory before.

    bm25 + bge-m3 first stages on GPU 0 (~10 min), then each reranker in a fresh process on its own GPU.
    """
    first = f"python eval_dog.py --retrievers bm25,bge-m3 --modes plain --out {OUT}/b_first > first.log 2>&1"
    rr = [
        f"CUDA_VISIBLE_DEVICES={g} python eval_dog.py --first-stage-from {OUT}/b_first/cache "
        f"--rerankers {name} --rerank-on bm25,bge-m3 --out {OUT}/b_{name} > {name}.log 2>&1"
        for g, name in ((0, "qwen3-reranker-0.6b"), (1, "jina-reranker-v3"))
    ]
    return f"!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; CUDA_VISIBLE_DEVICES=0 {first}; {rr[0]} & {rr[1]} & wait; grep -h retriever *.log"


def queue_command(gpu: int, runs: list[str]) -> str:
    calls = "; ".join(f"python eval_dog.py {r} --out {OUT}/b_gpu{gpu}" for r in runs)
    return f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES={gpu} sh -c "{calls}" > gpu{gpu}.log 2>&1'


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def baselines() -> dict:
    script = (HERE / "eval_dog.py").read_text(encoding="utf-8")
    cells = [
        md(
            """
# DOG-RAG retrieval baselines

Measures the multilingual baselines on DOG-RAG with the nos-rag-eval protocol
(250-word chunks, top 10, rerank, top 3; P@3, R@3, MRR at document level) over the
public DOG 2025 text plus the 122 DOG-RAG documents.

**Settings before running:** Accelerator = GPU T4 x2, Internet = on.
Optional: a Kaggle secret `HF_TOKEN` (read-only Hugging Face token) to include the gated
`google/embeddinggemma-300m`; accept its terms on the Hub first. Without it that row is skipped.

Expected time: about 1.5 hours, both GPUs in parallel. Outputs go to `/kaggle/working/b_gpu0/` and `b_gpu1/`:
`results.jsonl` (one line per configuration) and `runs/` (top-3 per query).
Nothing is uploaded anywhere.
"""
        ),
        code(
            f"""
!pip install -q {PINS}
import torch, transformers, sentence_transformers
print(torch.__version__, transformers.__version__, sentence_transformers.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
"""
        ),
        code(
            """
import os
extra = ""
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    extra = "1"
    print("HF_TOKEN found: embeddinggemma-300m included")
except Exception:
    print("No HF_TOKEN secret: embeddinggemma-300m skipped")
"""
        ),
        code("%%writefile eval_dog.py\n" + script),
        code("!" + " & ".join(queue_command(g, runs) for g, runs in GPU_QUEUES.items()) + "; wait; grep -h retriever gpu*.log"),
        code(f'!if [ -n "{{extra}}" ]; then python eval_dog.py --retrievers embeddinggemma-300m --batch-size 8 --out {OUT}/b_gpu0; fi'),
        code(
            f"""
import glob, shutil
import pandas as pd
for d in glob.glob("{OUT}/b_*/cache"):
    shutil.rmtree(d, ignore_errors=True)
df = pd.concat([pd.read_json(p, lines=True) for p in glob.glob("{OUT}/b_*/results.jsonl")])
df = df.drop_duplicates(["retriever", "reranker"])
cols = ["retriever", "reranker", "P@3", "R@3", "MRR", "R@10_first_stage", "nDCG@10_first_stage", "seconds"]
df = df[[c for c in cols + ["error"] if c in df.columns]]
df.sort_values("MRR", ascending=False)
"""
        ),
    ]
    return notebook(cells)


# ---------------------------------------------------------------- data notebook (phase 1, first run)

DATA_PINS = '"ctranslate2==4.8.0" "subword-nmt==0.3.8" nvidia-cublas-cu12'
VLLM_PINS = '"vllm==0.32.0" "accelerate==1.10.1"'
DATA_SCRIPTS = ["gen_queries.py", "translate_sinai.py"]


def scripts_cell(names: list[str]) -> str:
    """One line that writes the scripts, packed as zlib+base64 JSON with a hash check.

    Kaggle's editor is JupyterLab in a cross-origin iframe: the browser pane can only type into it,
    so the whole payload has to fit in one typed line.
    """
    import base64, hashlib, zlib

    files = {n: (HERE / n).read_text(encoding="utf-8") for n in names}
    raw = json.dumps(files, ensure_ascii=False, sort_keys=True).encode()
    h = hashlib.sha1(raw).hexdigest()[:12]
    b = base64.b64encode(zlib.compress(raw, 9)).decode()
    return (
        f'import base64, zlib, hashlib, json; s = zlib.decompress(base64.b64decode("{b}")); '
        f'assert hashlib.sha1(s).hexdigest()[:12] == "{h}", "payload corrupted"; '
        f'[open(n, "w").write(t) for n, t in json.loads(s).items()]; print("scripts ok", sorted(json.loads(s)))'
    )


def data_cells() -> list[str]:
    cublas = "$(python -c 'import nvidia.cublas, os; print(os.path.join(list(nvidia.cublas.__path__)[0], \"lib\"))')"
    return [
        # vLLM pins its own torch and transformers, so it lives in a separate venv.
        f"!pip install -q {DATA_PINS} uv && uv venv -q /tmp/vllm-env --python $(which python) && "
        f"uv pip install -q --python /tmp/vllm-env/bin/python {VLLM_PINS} && "
        '/tmp/vllm-env/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.device_count())"',
        scripts_cell(DATA_SCRIPTS),
        # Salamandra first: if vLLM fails on the T4s it shows in the log within minutes (then the HF fallback runs).
        f"!/tmp/vllm-env/bin/python -u gen_queries.py --passages 40000 --dev-passages 400 --out {OUT}/synth --max-hours 5.5 "
        "2>&1 | grep --line-buffered -v -E 'Warning|it/s]'",
        f"!LD_LIBRARY_PATH={cublas}:$LD_LIBRARY_PATH python -u translate_sinai.py --pairs 100000 --out {OUT}/sinai_gl "
        "--max-hours 3 2>&1 | grep --line-buffered -v Warning",
        f"!cat {OUT}/synth/shard_*.jsonl | wc -l; cat {OUT}/sinai_gl/shard_*.jsonl | wc -l; du -sh {OUT}/*",
    ]


def parlamint_cells() -> list[str]:
    """Questions over Parlamento de Galicia debates, for the start of the next GPU notebook.

    About 12k passages at the ~166 passages/min the first run showed, so ~75 min plus vLLM setup.
    """
    return [
        f"!pip install -q uv && uv venv -q /tmp/vllm-env --python $(which python) && "
        f"uv pip install -q --python /tmp/vllm-env/bin/python {VLLM_PINS}",
        scripts_cell(["gen_queries.py", "prep_parlamint.py"]),
        f"!python -u prep_parlamint.py --passages 12000 --tgz {OUT}/parlamint/ParlaMint-ES-GA.tgz --out {OUT}/synth_parl && "
        f"/tmp/vllm-env/bin/python -u gen_queries.py --out {OUT}/synth_parl --max-hours 1.5 "
        "2>&1 | grep --line-buffered -v -E 'Warning|it/s]'",
    ]


TF4_PINS = '"transformers==4.57.3" "sentence-transformers==5.1.2" "bm25s==0.3.16"'
DATA1 = "$(dirname $(find /kaggle/input -maxdepth 6 -type d -name sinai_gl | head -1))"


def data2_cells() -> list[str]:
    """Second data notebook (2026-11-04). Needs galician-data-1's output attached as input.

    1. ParlaMint questions on both GPUs (~1.5 h with vLLM setup).
    2. In parallel: mining on GPU 0 (~1 h: bge-m3 over ~140k passages, round-trip filter,
       bge-reranker-v2-m3 on ~4 pairs per kept question); the extra baselines for the stricter
       publish bar on GPU 1 (~45 min), with the gte models in a transformers 4.x environment.
    """
    first_stage = f"--retrievers bm25 --out {OUT}/b_extra"
    gpu1 = "; ".join(
        [
            f"python eval_dog.py --retrievers bm25,arctic-embed-l-v2.0,granite-embedding-311m-r2,granite-embedding-97m-r2 --batch-size 16 --out {OUT}/b_extra",
            f"python eval_dog.py {first_stage} --rerankers mxbai-rerank-base-v2 --rerank-on bm25",
            f"/tmp/tf4/bin/python eval_dog.py --retrievers bm25,gte-multilingual-base --rerankers gte-multilingual-reranker-base --rerank-on bm25 --batch-size 16 --out {OUT}/b_tf4",
        ]
    )
    mine = (
        f"python -u mine.py --synth {DATA1}/synth {OUT}/synth_parl --sinai {DATA1}/sinai_gl "
        f"--out {OUT}/mined"
    )
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    return [
        f"!pip install -q {PINS} uv && uv venv -q /tmp/vllm-env --python $(which python) && "
        f"uv pip install -q --python /tmp/vllm-env/bin/python {VLLM_PINS} && "
        f"uv venv -q --system-site-packages /tmp/tf4 --python $(which python) && "
        f"uv pip install -q --python /tmp/tf4/bin/python {TF4_PINS} && "
        f"ls {DATA1}",
        scripts_cell(["gen_queries.py", "prep_parlamint.py", "mine.py", "eval_dog.py"]),
        parlamint_cells()[2],
        f'!{alloc} CUDA_VISIBLE_DEVICES=0 sh -c "{mine}" 2>&1 | tee mine.log | grep --line-buffered "^{{" & '
        f'{alloc} CUDA_VISIBLE_DEVICES=1 sh -c "{gpu1}" > extra.log 2>&1; wait; '
        f"grep -h '^{{' mine.log; grep -h retriever extra.log",
        f"!rm -f {OUT}/mined/*.npy; wc -l {OUT}/mined/*.jsonl; du -sh {OUT}/*; "
        f"python -c \"import json, random; L = open('{OUT}/mined/train.jsonl').readlines(); "
        f"[print(json.loads(l)['source'], '|', json.loads(l)['query']) for l in random.Random(0).sample(L, 25)]\"",
    ]


DATA2 = "$(dirname $(find /kaggle/input -maxdepth 6 -type d -name mined | head -1))"


def train_cells(run: str = "run1", extra: str = "--epochs 2 --eval-steps 100") -> list[str]:
    """Bi-encoder training on both T4s (DDP), then the DOG-RAG test with the selected checkpoint.

    Needs galician-data-2's output attached. Model selection happens on the DOG 2024 dev set
    inside train.py; the test cell only runs once, on the final model. The last cell only prints
    why transformers 4.57.3 failed to import in galician-data-2, so the gte baselines can be fixed.
    """
    tf4_check = (
        "!pip install -q uv && uv venv -q --system-site-packages /tmp/tf4 --python $(which python) && "
        f"uv pip install -q --python /tmp/tf4/bin/python {TF4_PINS} && "
        "/tmp/tf4/bin/python -c \"import torch, transformers, huggingface_hub; "
        "print(torch.__version__, transformers.__version__, huggingface_hub.__version__); "
        "import transformers.modeling_utils\" 2>&1 | tail -15"
    )
    return [
        f"!pip install -q {PINS} \"datasets==5.1.0\" \"accelerate==1.15.0\" && ls {DATA2}/mined",
        scripts_cell(["train.py", "eval_dog.py"]),
        f"!torchrun --nproc_per_node 2 train.py --train {DATA2}/mined/train.jsonl --dev {DATA2}/mined/dev.jsonl "
        f"--out {OUT}/{run} {extra} 2>&1 | grep --line-buffered -v -E 'Warning|it/s]'",
        f"!python eval_dog.py --retrievers bm25,galician-{run}={OUT}/{run}/final --rerankers bge-reranker-v2-m3 "
        f"--rerank-on bm25,galician-{run} --out {OUT}/test_{run} && rm -rf {OUT}/{run}/checkpoints {OUT}/test_{run}/cache",
        tf4_check,
    ]


DATA1_ROOT = "$(dirname $(find /kaggle/input -maxdepth 6 -type d -name sinai_gl | head -1))"


def train2_cells(run: str = "run2", extra: str = "--epochs 1 --eval-steps 50 --cap SINAI=30000") -> list[str]:
    """Run 2 (galician-train-2). Needs galician-data-1's and galician-data-2's outputs attached.

    1. rescue.py on GPU 0 adds back the pairs bge-m3's round trip dropped but BM25 finds and the
       reranker accepts (~30 min, against ~3.5 h for a full mine.py --filter union). In parallel,
       GPU 1 measures gte-multilingual-base and its reranker in the transformers 4 venv.
    2. One epoch on both T4s (run 1 peaked at the end of epoch 1), SINAI capped at 30k rows
       (69 % of run 1's data), dev set copied from galician-data-2 so dev numbers compare with run 1.
    3. DOG-RAG test: our model alone and under bge-reranker-v2-m3, BM25 alone as the control.

    The transformers 4 venv is built with pip, not uv: uv put its own torch into the venv, which
    then clashed with Kaggle's torchvision ("operator torchvision::nms does not exist").
    pip inside a --system-site-packages venv reuses Kaggle's torch and torchvision.
    """
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    rescue = (
        f"python -u rescue.py --mined {DATA2}/mined --synth {DATA1_ROOT}/synth {DATA2}/synth_parl "
        f"--sinai {DATA1_ROOT}/sinai_gl --out {OUT}/mined2"
    )
    gte = (
        "/tmp/tf4/bin/python eval_dog.py --retrievers bm25,gte-multilingual-base "
        f"--rerankers gte-multilingual-reranker-base --rerank-on bm25 --batch-size 16 --out {OUT}/b_tf4"
    )
    sample = (
        "python -c \"import json; L = [json.loads(l) for l in open('" + OUT + "/mined2/rescue_sample.jsonl')]; "
        "[print(r['status'], r['source'], r['pos_score'], '|', r['query']) for r in L[:15] + [x for x in L if x['status'] == 'rejected'][:10]]\""
    )
    return [
        f"!pip install -q {PINS} \"datasets==5.1.0\" \"accelerate==1.15.0\" uv && "
        "uv venv -q --seed --system-site-packages /tmp/tf4 --python $(which python) && "
        f"/tmp/tf4/bin/python -m pip install -q {TF4_PINS}; "
        "/tmp/tf4/bin/python -c \"import torch, transformers; print(torch.__file__, torch.__version__, transformers.__version__); "
        "import transformers.modeling_utils\" 2>&1 | tail -3; "
        f"ls {DATA1_ROOT} {DATA2}",
        scripts_cell(["mine.py", "rescue.py", "train.py", "eval_dog.py"]),
        f'!{alloc} CUDA_VISIBLE_DEVICES=0 sh -c "{rescue}" 2>&1 | grep --line-buffered "^{{" & '
        f'{alloc} CUDA_VISIBLE_DEVICES=1 sh -c "{gte}" > gte.log 2>&1; wait; '
        f"grep -h retriever gte.log; grep -h -A3 Error gte.log | tail -8; {sample}",
        f"!torchrun --nproc_per_node 2 train.py --train {OUT}/mined2/train.jsonl --dev {OUT}/mined2/dev.jsonl "
        f"--out {OUT}/{run} {extra} 2>&1 | grep --line-buffered -v -E 'Warning|it/s]'",
        f"!python eval_dog.py --retrievers bm25,galician-{run}={OUT}/{run}/final --rerankers bge-reranker-v2-m3 "
        f"--rerank-on galician-{run} --out {OUT}/test_{run} && rm -rf {OUT}/{run}/checkpoints {OUT}/test_{run}/cache",
    ]


MINED2 = "$(find /kaggle/input -maxdepth 6 -type d -name mined2 | head -1)"


def train3_cells(run: str = "run3", base: str = "intfloat/multilingual-e5-base", epochs: int = 1) -> list[str]:
    """Run 3 (galician-train-3): the mE5-base ablation. Needs galician-train-2's output attached.

    One change against run 2: the base model. Same training rows (run 2's mined2/train.jsonl,
    SINAI capped at 30k), same dev set, 1 epoch, no prompts. The test cell adds the BM25 hybrid
    so run 3 lines up with every row of galician-eval-hybrid.
    """
    extra = f"--base {base} --epochs {epochs} --eval-steps 50 --cap SINAI=30000"
    ours = f"galician-{run}={OUT}/{run}/final"
    return [
        f"!pip install -q {PINS} \"datasets==5.1.0\" \"accelerate==1.15.0\" && ls {MINED2}",
        scripts_cell(["train.py", "eval_dog.py"]),
        f"!torchrun --nproc_per_node 2 train.py --train {MINED2}/train.jsonl --dev {MINED2}/dev.jsonl "
        f"--out {OUT}/{run} {extra} 2>&1 | grep --line-buffered -v -E 'Warning|it/s]'",
        f"!python eval_dog.py --retrievers bm25,{ours} --hybrid --rerankers bge-reranker-v2-m3 "
        f"--rerank-on galician-{run},hybrid --out {OUT}/test_{run} && rm -rf {OUT}/{run}/checkpoints {OUT}/test_{run}/cache",
    ]


GTE_PINS = '"torch==2.8.0" "transformers==4.57.3" "sentence-transformers==5.1.2" "bm25s==0.3.16"'


def gte_cell() -> str:
    """gte-multilingual-base and its reranker, which need transformers 4 (remote code).

    A venv that sees Kaggle's system packages failed twice (torchvision, then a huggingface_hub
    mismatch), so this one is fully isolated with its own torch.
    """
    return (
        "!pip install -q uv && uv venv -q /tmp/gte --python $(which python) && "
        f"uv pip install -q --python /tmp/gte/bin/python {GTE_PINS} && "
        "/tmp/gte/bin/python eval_dog.py --retrievers bm25,gte-multilingual-base "
        "--rerankers gte-multilingual-reranker-base --rerank-on bm25 --batch-size 16 "
        f"--out {OUT}/b_gte 2>&1 | grep -E '^{{|rror' | tail -8; rm -rf {OUT}/b_gte/cache"
    )


def train4_cells() -> list[str]:
    """Run 4 (galician-train-4): run 3 with 2 epochs, best checkpoint picked on dev; then gte.

    Needs galician-train-2's output attached (mined2/). The only change against run 3 is the
    number of epochs; dev is evaluated every 50 steps and the best checkpoint is the one tested.
    """
    return train3_cells(run="run4", epochs=2) + [gte_cell()]


def train5_cells(run: str = "run5", expect_sha1: str = "SET_ME") -> list[str]:
    """Run 5 (galician-train-5): run 4's recipe plus short passages and WebFAQ glg (training side).

    Needs galician-train-2's output attached (mined2/). prep5.py rebuilds the training file from
    mined2 on CPU and stops the cell unless its hash matches the one computed on the Mac, so
    Kaggle trains on exactly the rows that were inspected there. Then 2 epochs with dev every
    50 steps (the best dev checkpoint is the one saved as final), and in parallel: GPU 0 reads
    the DOG-RAG test once for that checkpoint (alone, hybrid, under bge-reranker-v2-m3, as for
    run 4); GPU 1 runs news, PAWS-gl and WebFAQ.
    """
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    ours = f"galician-{run}={OUT}/{run}/final"
    data = f"{OUT}/mined5"
    gpu0 = (
        f"python eval_dog.py --retrievers bm25,{ours} --hybrid --rerankers bge-reranker-v2-m3 "
        f"--rerank-on galician-{run},hybrid --out {OUT}/test_{run}"
    )
    gpu1 = f"python eval_extra.py --retrievers bm25,{ours} --out {OUT}/e_{run}"
    return [
        f"!pip install -q {PINS} \"datasets==5.1.0\" \"accelerate==1.15.0\" && ls {MINED2}",
        scripts_cell(["prep5.py", "mine.py", "train.py", "translate_sinai.py", "eval_dog.py", "eval_extra.py", "gen_queries.py"]),
        f"!python -u prep5.py --mined {MINED2} --out {data} --expect-sha1 {expect_sha1} && "
        f"torchrun --nproc_per_node 2 train.py --base intfloat/multilingual-e5-base --train {data}/train.jsonl "
        f"--dev {data}/dev.jsonl --out {OUT}/{run} --epochs 2 --eval-steps 50 "
        "2>&1 | grep --line-buffered -v -E 'Warning|it/s]'",
        f'!ls {OUT}/{run}/final && ({alloc} CUDA_VISIBLE_DEVICES=0 sh -c "{gpu0}" > t0.log 2>&1 & '
        f'{alloc} CUDA_VISIBLE_DEVICES=1 sh -c "{gpu1}" > e1.log 2>&1; wait); '
        f"grep -h -E '\"retriever\"|\"task\"|rror' t0.log e1.log | grep -v first_stages; "
        f"rm -rf {OUT}/{run}/checkpoints {OUT}/test_{run}/cache {OUT}/e_{run}/cache {data}/train.jsonl",
    ]


def final_dir(run: str) -> str:
    return f"$(find /kaggle/input -maxdepth 7 -type d -name final -path '*{run}*' | head -1)"


def hybrid_cells(runs: tuple[str, ...] = ("run1", "run2")) -> list[str]:
    """galician-eval-hybrid: BM25 + dense fusion on DOG-RAG. Needs the training notebooks' outputs attached.

    GPU 0: BM25 + bge-m3 fused, the baseline any hybrid claim has to beat (Lynx uses a hybrid).
    GPU 1: BM25 fused with each of our checkpoints. Both then rerank the fused lists with
    bge-reranker-v2-m3. About 40 min on two T4s.
    """
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    rr = "--hybrid --rerankers bge-reranker-v2-m3 --rerank-on hybrid"
    ours = ",".join(f"galician-{r}={final_dir(r)}" for r in runs)
    gpu0 = f"python eval_dog.py --retrievers bm25,bge-m3 --modes plain {rr} --out {OUT}/h_gpu0"
    gpu1 = f"python eval_dog.py --retrievers {ours} {rr} --out {OUT}/h_gpu1"
    table = (
        "import glob, pandas as pd; "
        f"df = pd.concat([pd.read_json(p, lines=True) for p in glob.glob('{OUT}/h_*/results.jsonl')]); "
        "cols = [c for c in ['retriever', 'reranker', 'P@3', 'R@3', 'MRR', 'nDCG@10_first_stage', 'error'] if c in df.columns]; "
        "print(df[cols].sort_values('MRR', ascending=False).to_string(index=False))"
    )
    return [
        f"!pip install -q {PINS} && ls -d " + " ".join(final_dir(r) for r in runs),
        scripts_cell(["eval_dog.py"]),
        f'!{alloc} CUDA_VISIBLE_DEVICES=0 sh -c "{gpu0}" > h0.log 2>&1 & '
        f'{alloc} CUDA_VISIBLE_DEVICES=1 sh -c "{gpu1}" > h1.log 2>&1; wait; '
        f"grep -h -E 'retriever|rror' h0.log h1.log; rm -rf {OUT}/h_*/cache",
        table,
    ]


MIX_ALPHAS = ("1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.0")


def mix_cells(run: str = "run4") -> list[str]:
    """galician-mix: run 4's weights interpolated with mE5-base (mix.py), no training.

    Needs galician-train-2 (mined2/dev.jsonl) and galician-train-4 (run4/final) attached.
    mix.py scores every mixing weight on DOG 2024 dev and copies the chosen one to mix/final
    (smallest weight within 0.01 dev MRR@10 of run 4). Then, in parallel: GPU 0 reads the
    DOG-RAG test once for the chosen mix; GPU 1 runs news, PAWS-gl and WebFAQ for every mix,
    so the card can show the whole curve (a = 1 is run 4, a = 0 is plain mE5-base).
    """
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    curve = ",".join(f"mix-a{a.replace('.', '')}=/tmp/mix/a{float(a):.2f}" for a in MIX_ALPHAS)
    gpu0 = f"python eval_dog.py --retrievers bm25,galician-mix={OUT}/mix/final --out {OUT}/t_mix"
    gpu1 = f"python eval_extra.py --retrievers {curve} --out {OUT}/e_mix"
    return [
        f"!pip install -q {PINS} && ls {MINED2} && ls -d {final_dir(run)}",
        scripts_cell(["mix.py", "train.py", "eval_dog.py", "eval_extra.py", "gen_queries.py"]),
        f"!python mix.py --tuned {final_dir(run)} --dev {MINED2}/dev.jsonl --out {OUT}/mix --scratch /tmp/mix "
        "--alphas " + ",".join(MIX_ALPHAS) + " 2>&1 | grep --line-buffered -E '^\\{|rror'",
        f'!{alloc} CUDA_VISIBLE_DEVICES=0 sh -c "{gpu0}" > t0.log 2>&1 & '
        f'{alloc} CUDA_VISIBLE_DEVICES=1 sh -c "{gpu1}" > e1.log 2>&1; wait; '
        f"grep -h -E '\"retriever\"|\"task\"|rror' t0.log e1.log | grep -v first_stages; rm -rf {OUT}/t_mix/cache {OUT}/e_mix/cache",
    ]


def variance_cells(run: str, seed: int, cap_sinai: int, mix: bool) -> list[str]:
    """Checks for the paper: training variance (galician-seed1/2) and the SINAI ablation (galician-nosinai).

    Run 4's recipe (mE5-base, 2 epochs, dev every 50 steps, best checkpoint on dev) with one change:
    the seed (which also redraws the 30k SINAI rows) or no SINAI rows at all. Needs galician-train-2's
    output attached (mined2/). With mix set, the run is also mixed 0.9 with mE5-base, the release
    recipe (a fixed, not chosen on dev here). Then, in parallel: DOG-RAG test on GPU 0, news,
    PAWS-gl and WebFAQ on GPU 1, for the run and its mix, with BM25 as the control in both.
    """
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    extra = f"--base intfloat/multilingual-e5-base --epochs 2 --eval-steps 50 --cap SINAI={cap_sinai} --seed {seed}"
    models = f"galician-{run}={OUT}/{run}/final"
    cells = [
        f"!pip install -q {PINS} \"datasets==5.1.0\" \"accelerate==1.15.0\" && ls {MINED2}",
        scripts_cell(["train.py", "eval_dog.py", "eval_extra.py", "mix.py", "gen_queries.py"]),
        f"!torchrun --nproc_per_node 2 train.py --train {MINED2}/train.jsonl --dev {MINED2}/dev.jsonl "
        f"--out {OUT}/{run} {extra} 2>&1 | grep --line-buffered -v -E 'Warning|it/s]' "
        f"&& rm -rf {OUT}/{run}/checkpoints",
    ]
    if mix:
        cells.append(
            f"!python mix.py --tuned {OUT}/{run}/final --dev {MINED2}/dev.jsonl --out {OUT}/mix_{run} "
            "--scratch /tmp/mix --alphas 1.0,0.9 --tolerance 1.0 2>&1 | grep --line-buffered -E '^\\{|rror'"
        )
        models += f",galician-mix-{run}={OUT}/mix_{run}/final"
    gpu0 = f"python eval_dog.py --retrievers bm25,{models} --modes plain --out {OUT}/test_{run}"
    gpu1 = f"python eval_extra.py --retrievers bm25,{models} --modes plain --out {OUT}/e_{run}"
    cells.append(
        f'!{alloc} CUDA_VISIBLE_DEVICES=0 sh -c "{gpu0}" > t0.log 2>&1 & '
        f'{alloc} CUDA_VISIBLE_DEVICES=1 sh -c "{gpu1}" > e1.log 2>&1; wait; '
        f"grep -h -E '\"retriever\"|\"task\"|rror' t0.log e1.log | grep -v first_stages; "
        f"rm -rf {OUT}/test_{run}/cache {OUT}/e_{run}/cache"
    )
    return cells


VARIANCE_RUNS = {
    # kernel slug: (run name, seed, SINAI cap, mix with mE5-base)
    "galician-nosinai": ("nosinai", 0, 0, False),
    "galician-seed1": ("seed1", 1, 30000, True),
    "galician-seed2": ("seed2", 2, 30000, True),
}


def write_kaggle_kernels(root: Path, owner: str = os.environ.get("KAGGLE_USERNAME", "KAGGLE_USER")) -> None:
    """One folder per run with notebook.ipynb and kernel-metadata.json, for `kaggle kernels push -p`."""
    for slug, (run, seed, cap, mix) in VARIANCE_RUNS.items():
        d = root / slug
        d.mkdir(parents=True, exist_ok=True)
        nb = notebook([code(c) for c in variance_cells(run, seed, cap, mix)])
        (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        meta = {
            "id": f"{owner}/{slug}",
            "title": slug,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [f"{owner}/galician-train-2"],
            "model_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {d}")


GROUPS_SHA1 = "925e63ef5eb7"  # rr1/groups.jsonl from the reranker rerun on the L4


def rerank_ablation_cells(run: str, alpha: float) -> list[str]:
    """Reranker ablation for the paper: run 1's recipe with another --alpha (1.0 = no teacher).

    Reuses the rerun's teacher groups (private dataset galician-rerank-groups, groups.jsonl.gz),
    so the 2 h teacher stage is skipped. Everything else as rerank1_vm.sh on the L4: mE5-base,
    16 questions per optimizer step on one GPU (CUDA_VISIBLE_DEVICES pins it, otherwise the HF
    Trainer falls back to DataParallel over both T4s and doubles the batch), seed 0, dev every
    500 steps. The L4 (24 GB) ran 4 x grad-accum 4; on the 15 GB T4 it is 2 x grad-accum 8 (same
    samples per optimizer step, same order, same number of steps).
    Versions 1-4 ran out of memory at the first training step with only 3.4 GB held by PyTorch:
    smi.log showed GPU 0 at 11.3 GB (75 % of the T4) before the model was even loaded. That is JAX's
    default preallocation: Kaggle's image has JAX, and bm25s uses it when it is installed (the L4 VM
    had none). JAX_PLATFORMS=cpu keeps it off the GPU. nvidia-smi logs both GPUs every 30 s to
    smi.log. Then the test is read once on one GPU as on the L4: BM25 and bge-m3 top 10 reranked by the new model and by bge-reranker-v2-m3,
    first-stage lists kept in the cache so the alpha 0.5 model can be rescored on exactly them.
    """
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    groups = "/tmp/groups.jsonl"
    smi = "nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 30"
    return [
        f"!pip install -q {PINS} \"datasets==5.1.0\" \"accelerate==1.15.0\" && ls {MINED2} && "
        f"gz=$(find /kaggle/input -name 'groups.jsonl*' | head -1) && echo $gz && "
        f"(case $gz in *.gz) gzip -dc $gz;; *) cat $gz;; esac) > {groups} && "
        f'test "$(sha1sum {groups} | cut -c1-12)" = {GROUPS_SHA1} && wc -l {groups}',
        scripts_cell(["train_reranker.py", "train.py", "eval_dog.py", "gen_queries.py"]),
        f"!{smi} > {OUT}/smi.log 2>&1 & "
        f"JAX_PLATFORMS=cpu {alloc} CUDA_VISIBLE_DEVICES=0 python -u train_reranker.py --groups {groups} --dev {MINED2}/dev.jsonl "
        f"--out {OUT}/{run} --alpha {alpha} --batch-size 2 --grad-accum 8 --eval-steps 500 2>&1 "
        f"| tee {OUT}/train.log | grep --line-buffered -E '^\\{{|rror'; "
        f"pkill -f 'nvidia-smi --query'; rm -rf {OUT}/{run}/checkpoints; sort -k3 -t, -n {OUT}/smi.log | tail -2",
        f"!JAX_PLATFORMS=cpu {alloc} CUDA_VISIBLE_DEVICES=0 python eval_dog.py --retrievers bm25,bge-m3 --modes plain "
        f"--rerankers ce:{OUT}/{run}/final,bge-reranker-v2-m3 --rerank-on bm25,bge-m3 --out {OUT}/test_{run} "
        f"> {OUT}/t_{run}.log 2>&1; grep -h -E '\"retriever\"|rror' {OUT}/t_{run}.log; ls {OUT}/test_{run}/cache",
    ]


def write_rerank_kernel(root: Path, slug: str = "galician-rerank-alpha1", run: str = "alpha1", alpha: float = 1.0, owner: str = os.environ.get("KAGGLE_USERNAME", "KAGGLE_USER")) -> None:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    nb = notebook([code(c) for c in rerank_ablation_cells(run, alpha)])
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/galician-rerank-groups"],
        "competition_sources": [],
        "kernel_sources": [f"{owner}/galician-train-2"],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {d}")


def data() -> dict:
    cells = [
        md(
            """
# Galician embedder: training data, first run

1. Synthetic Galician questions over DOG / BOP passages with Salamandra-7b-instruct (vLLM, both T4s), up to 5.5 h.
2. Spanish SINAI legal triplets (query + positive) translated to Galician with Nós MT, up to 3 h.

**Settings:** Accelerator = GPU T4 x2, Internet = on. Outputs in `/kaggle/working/synth` and
`/kaggle/working/sinai_gl`, written in shards so a stopped run keeps what it finished. Nothing is uploaded.
"""
        )
    ] + [code(c) for c in data_cells()]
    return notebook(cells)


if __name__ == "__main__":
    import sys

    if "--data-cells" in sys.argv:
        for i, c in enumerate(data_cells()):
            (Path(sys.argv[-1]) / f"data_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:160])
        raise SystemExit
    if "--parl-cells" in sys.argv:
        for i, c in enumerate(parlamint_cells()):
            (Path(sys.argv[-1]) / f"parl_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:160])
        raise SystemExit
    if "--data2-cells" in sys.argv:
        for i, c in enumerate(data2_cells()):
            (Path(sys.argv[-1]) / f"data2_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:200])
        raise SystemExit
    if "--train-cells" in sys.argv:
        for i, c in enumerate(train_cells()):
            (Path(sys.argv[-1]) / f"train_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:200])
        raise SystemExit
    if "--train2-cells" in sys.argv:
        for i, c in enumerate(train2_cells()):
            (Path(sys.argv[-1]) / f"train2_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:200])
        raise SystemExit
    if "--train4-cells" in sys.argv:
        for i, c in enumerate(train4_cells()):
            (Path(sys.argv[-1]) / f"train4_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:300])
        raise SystemExit
    if "--train5-cells" in sys.argv:
        # python build_notebooks.py --train5-cells SHA1 DIR, SHA1 from prep5.py's "done" line on the Mac
        for i, c in enumerate(train5_cells(expect_sha1=sys.argv[-2])):
            (Path(sys.argv[-1]) / f"train5_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:300])
        raise SystemExit
    if "--train3-cells" in sys.argv:
        for i, c in enumerate(train3_cells()):
            (Path(sys.argv[-1]) / f"train3_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:220])
        raise SystemExit
    if "--mix-cells" in sys.argv:
        for i, c in enumerate(mix_cells()):
            (Path(sys.argv[-1]) / f"mix_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:300])
        raise SystemExit
    if "--hybrid-cells" in sys.argv:
        for i, c in enumerate(hybrid_cells()):
            (Path(sys.argv[-1]) / f"hybrid_cell_{i}.txt").write_text(c, encoding="utf-8")
            print(i, len(c), c[:200])
        raise SystemExit
    if "--variance-kernels" in sys.argv:
        # python build_notebooks.py --variance-kernels DIR, then kaggle kernels push -p DIR/<slug>
        write_kaggle_kernels(Path(sys.argv[-1]))
        raise SystemExit
    if "--rerank-kernel" in sys.argv:
        # python build_notebooks.py --rerank-kernel DIR, then kaggle kernels push -p DIR/galician-rerank-alpha1
        write_rerank_kernel(Path(sys.argv[-1]))
        raise SystemExit
    if "--rerank-cell" in sys.argv:
        print(rerank_only_cell())
        raise SystemExit
    for name, nb in (("kaggle_baselines.ipynb", baselines()), ("kaggle_data.ipynb", data())):
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out}")
