#!/usr/bin/env bash
# Reranker run 1 on one L4 (GCP g2-standard-8), started by gcp_rerank1.sh: candidates and
# bge-reranker-v2-m3 teacher scores (prep_rerank.py), training from multilingual-e5-base
# (train_reranker.py, chosen on dev), then the DOG-RAG test read once: BM25 and bge-m3 top 10
# reranked by the new model and by bge-reranker-v2-m3 as the control. Writes DONE or FAILED.
set -euo pipefail
cd "$(dirname "$0")"
trap 'touch FAILED' ERR
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -q .venv --python 3.12
uv pip install -q --python .venv/bin/python "torch==2.15.0" "transformers==5.22.1" "sentence-transformers==6.2.0" \
    "bm25s==0.3.16" "datasets==5.1.0" "accelerate==1.15.0" pyarrow pandas
PY=.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"

# The laptop's candidates.jsonl hashed to 89ca96cef859; a different hash here is logged, not fatal
# (bm25s tie order could differ between platforms), and summary.json keeps this run's value.
$PY -u prep_rerank.py candidates --mined mined2 --out rr1 > prep.log 2>&1
$PY -u prep_rerank.py teacher --out rr1 --batch-size 64 >> prep.log 2>&1
$PY -u train_reranker.py --groups rr1/groups.jsonl --dev mined2/dev.jsonl --out rerank1 \
    --batch-size 4 --grad-accum 4 --eval-steps 500 > train.log 2>&1
$PY eval_dog.py --retrievers bm25,bge-m3 --modes plain --rerankers ce:rerank1/final,bge-reranker-v2-m3 \
    --rerank-on bm25,bge-m3 --out test_rerank1 > t0.log 2>&1
rm -rf rerank1/checkpoints test_rerank1/cache rr1/candidates.jsonl
touch DONE
