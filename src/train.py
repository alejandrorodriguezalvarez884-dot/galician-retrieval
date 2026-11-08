"""Fine-tune a Galician bi-encoder on the mined pairs (phase 2, contrastive stage).

Input is mine.py's output: train.jsonl (query, positive, hard negatives) and dev.jsonl
(DOG 2024 questions). Model selection uses only the dev set: the dev corpus is the dev
positives plus distractor chunks from other DOG 2024 documents, chunked like the test
(250 words, 50 overlap). DOG-RAG is never touched here.

Loss: CachedMultipleNegativesRankingLoss (in-batch negatives plus one mined hard negative)
with a large effective batch; the no-duplicates sampler keeps a passage from showing up
twice in a batch. The model has no query/document prompts, so it works as a drop-in
replacement in pipelines that call plain encode() (nos-rag-eval, Lynx).

Two T4s:
    torchrun --nproc_per_node 2 train.py --train mined/train.jsonl --dev mined/dev.jsonl --out run1
One device (or CPU smoke test):
    python train.py --train mined/train.jsonl --dev mined/dev.jsonl --out run1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch

BASE = "proxectonos/MrBERT-nos-gl"
CORPUS_REPO = "proxectonos/corpus_dominio_legal_administrativo_tmp"
DEV_FILE = "XuntaGalicia_DOG/gl/2024_gl.jsonl"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def chunks_of(text: str, size: int = 250, overlap: int = 50) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap) if words[i : i + size]]


def cap_sources(rows: list[dict], caps: list[str], seed: int) -> list[dict]:
    """Keep a random --cap SOURCE=N rows of each capped source; other sources stay whole."""
    limits = {}
    for c in caps:
        source, n = c.rsplit("=", 1)
        limits[source] = int(n)
    rng = random.Random(seed)
    out = []
    for source in sorted({r["source"] for r in rows}):
        group = [r for r in rows if r["source"] == source]
        if source in limits and len(group) > limits[source]:
            group = rng.sample(group, limits[source])
        out += group
    return out


def train_dataset(rows: list[dict], seed: int):
    from datasets import Dataset

    rng = random.Random(seed)
    positives = [r["positive"] for r in rows]
    data = {"anchor": [], "positive": [], "negative": []}
    for r in rows:
        neg = r["negatives"][0] if r.get("negatives") else rng.choice(positives)
        data["anchor"].append(r["query"])
        data["positive"].append(r["positive"])
        data["negative"].append(neg)
    return Dataset.from_dict(data).shuffle(seed=seed)


def dev_evaluator(dev_rows: list[dict], dev_file: Path | None, distractors: int, seed: int, revision: str | None):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    corpus, queries, relevant = {}, {}, {}
    pos_id: dict[str, str] = {}
    for i, r in enumerate(dev_rows):
        pid = pos_id.setdefault(r["positive"], f"p{len(pos_id)}")
        corpus[pid] = r["positive"]
        queries[f"q{i}"] = r["query"]
        relevant[f"q{i}"] = {pid}
    if distractors:
        if dev_file is None:
            from huggingface_hub import hf_hub_download

            dev_file = Path(hf_hub_download(CORPUS_REPO, DEV_FILE, repo_type="dataset", revision=revision))
        dev_docs = {r["doc_id"] for r in dev_rows}
        pool = []
        with open(dev_file, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("url") not in dev_docs and d.get("text"):
                    pool += chunks_of(d["text"])
        random.Random(seed).shuffle(pool)
        for j, c in enumerate(pool[:distractors]):
            corpus[f"d{j}"] = c
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name="dog2024-dev", mrr_at_k=[3, 10], ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 10], precision_recall_at_k=[3, 10], map_at_k=[10], batch_size=64,
    ), len(corpus)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--dev-file", default=None, help="local DOG 2024 jsonl for dev distractors (default: download)")
    ap.add_argument("--corpus-revision", default=None)
    ap.add_argument("--dev-distractors", type=int, default=5000)
    ap.add_argument("--out", default="run")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128, help="per device; CachedMNRL makes it cheap")
    ap.add_argument("--mini-batch", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--cap", action="append", default=[], metavar="SOURCE=N", help="at most N rows of SOURCE (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: first N training rows")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    cuda = torch.cuda.is_available()
    fp16 = cuda and not args.no_fp16

    def log(**kw):
        if rank == 0:
            print(json.dumps(kw, ensure_ascii=False), flush=True)

    rows = read_jsonl(Path(args.train), args.limit)
    if args.cap:
        rows = cap_sources(rows, args.cap, args.seed)
    dev_rows = read_jsonl(Path(args.dev))
    train = train_dataset(rows, args.seed)
    evaluator, corpus_size = dev_evaluator(
        dev_rows, Path(args.dev_file) if args.dev_file else None, args.dev_distractors, args.seed, args.corpus_revision
    )
    sources = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    log(step="data", train=len(train), by_source=sources, rescued=sum(1 for r in rows if r.get("rescued")), dev_queries=len(dev_rows), dev_corpus=corpus_size, world=world)

    # MrBERT-nos-gl is stored in fp16 and was trained in bf16: load fp32 weights, train with fp16 autocast.
    model = SentenceTransformer(args.base, model_kwargs={"dtype": torch.float32})
    model.max_seq_length = args.max_seq_length
    # BERT-style bases (mE5 is XLM-R) carry a pooler that mean pooling never uses; under DDP its
    # parameters get no gradient and the second step fails. ModernBERT has no pooler.
    auto_model = model[0].auto_model
    if getattr(auto_model, "pooler", None) is not None:
        auto_model.pooler = None
        log(step="dropped_pooler", base=args.base)

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch, gather_across_devices=world > 1)

    class NanGuard(TrainerCallback):
        """Stop early if fp16 overflows (the base was trained in bf16)."""

        def on_log(self, a, state, control, logs=None, **kw):
            value = (logs or {}).get("loss")
            if value is not None and not math.isfinite(value):
                log(step="nan_loss", global_step=state.global_step)
                control.should_training_stop = True

    steps_per_epoch = max(1, len(train) // (args.batch_size * world))
    eval_steps = min(args.eval_steps, steps_per_epoch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        fp16=fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dog2024-dev_cosine_mrr@10",
        greater_is_better=True,
        logging_steps=max(1, eval_steps // 10),
        dataloader_drop_last=True,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    base_scores = evaluator(model)
    log(step="dev_before", **{k: round(v, 4) for k, v in base_scores.items() if "mrr" in k or "ndcg" in k})

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train, loss=loss, evaluator=evaluator, callbacks=[NanGuard()]
    )
    trainer.train()

    if rank == 0:
        final = out / "final"
        model.save(str(final))
        scores = evaluator(model)
        summary = {
            "base": args.base,
            "train_rows": len(train),
            "by_source": sources,
            "dev_before": base_scores,
            "dev_after": scores,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "steps": trainer.state.global_step,
            "minutes": round((time.time() - t0) / 60, 1),
            "args": vars(args),
        }
        (out / "train_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        log(step="done", steps=trainer.state.global_step, minutes=summary["minutes"],
            **{k: round(v, 4) for k, v in scores.items() if "mrr" in k or "ndcg" in k})


if __name__ == "__main__":
    main()
