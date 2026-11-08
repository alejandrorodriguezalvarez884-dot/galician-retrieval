#!/usr/bin/env bash
# Run 5 on one L4 (GCP g2-standard-8), started by gcp_run5.sh. Same rows, recipe and
# evaluations as the Kaggle cells (build_notebooks.train5_cells), with the global batch of
# 256 on one GPU instead of 128 on each of two T4s. Writes DONE or FAILED at the end.
set -euo pipefail
cd "$(dirname "$0")"
trap 'touch FAILED' ERR
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPECT=c6dcde2a18de
got=$(sha1sum mined5/train.jsonl | cut -c1-12)
[ "$got" = "$EXPECT" ] || { echo "train.jsonl hash $got != $EXPECT"; false; }

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -q .venv --python 3.12
uv pip install -q --python .venv/bin/python "torch==2.15.0" "transformers==5.22.1" "sentence-transformers==6.2.0" \
    "bm25s==0.3.16" "datasets==5.1.0" "accelerate==1.15.0" pyarrow pandas
PY=.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"

$PY -u train.py --base intfloat/multilingual-e5-base --train mined5/train.jsonl --dev mined5/dev.jsonl \
    --out run5 --epochs 2 --eval-steps 50 --batch-size 256 > train.log 2>&1
$PY eval_dog.py --retrievers bm25,galician-run5=run5/final --hybrid --rerankers bge-reranker-v2-m3 \
    --rerank-on galician-run5,hybrid --out test_run5 > t0.log 2>&1
$PY eval_extra.py --retrievers bm25,galician-run5=run5/final --out e_run5 > e1.log 2>&1
rm -rf run5/checkpoints test_run5/cache e_run5/cache
touch DONE
