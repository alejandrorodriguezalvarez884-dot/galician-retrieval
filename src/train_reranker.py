"""Fine-tune a Galician cross-encoder reranker on prep_rerank.py's groups (phase 3).

Each training example is a question with its positive and up to 7 negatives (bge-m3 and BM25
hard negatives). The loss is listwise: a softmax over the group's scores against a target that
puts --alpha on the positive and spreads 1 - alpha as bge-reranker-v2-m3's softmax over the same
group (its logits divided by --temperature). alpha 1 is plain contrastive training, alpha 0 is
pure distillation (ListNet on the teacher's scores).

Model selection uses only the dev set, the way the reranker is used in DOG-RAG: BM25 top 10 over
the dev corpus (the DOG 2024 dev positives plus --dev-distractors chunks from other DOG 2024
documents, as in train.py), reranked, MRR@10. A question whose positive is not in BM25's top 10
scores 0, so the ceiling is BM25's recall@10. DOG-RAG is never touched here.

    python train_reranker.py --groups rr1/groups.jsonl --dev mined2/dev.jsonl --out rerank1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CORPUS_REPO, DEV_FILE, chunks_of, read_jsonl  # noqa: E402

BASE = "intfloat/multilingual-e5-base"
TEACHER = "BAAI/bge-reranker-v2-m3"
TOKEN_RE = r"(?u)\b\w+\b"


class TeacherListLoss(nn.Module):
    """Listwise cross-entropy against alpha * one-hot(positive) + (1 - alpha) * softmax(teacher / T).

    Inputs are (query, [positive, negative, ...]) with the teacher's logits as labels; the positive
    is always the first document. Groups may have different sizes (padding is masked).
    """

    def __init__(self, model, alpha: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, inputs: list, labels: list[Tensor], prompt: str | None = None, task: str | None = None) -> Tensor:
        queries, docs_list = inputs
        sizes = [len(d) for d in docs_list]
        pairs = [(q, d) for q, docs in zip(queries, docs_list) for d in docs]
        tokens = self.model.preprocess(pairs, prompt=prompt, task=task)
        tokens = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in tokens.items()}
        logits = self.model(tokens)["scores"].view(-1).float()

        device = logits.device
        batch, width = len(queries), max(sizes)
        rows = torch.repeat_interleave(torch.arange(batch), torch.tensor(sizes)).to(device)
        cols = torch.cat([torch.arange(n) for n in sizes]).to(device)
        student = torch.full((batch, width), -1e4, device=device)
        student[rows, cols] = logits
        teacher = torch.full((batch, width), float("-inf"), device=device)
        teacher[rows, cols] = torch.cat([torch.as_tensor(lab, dtype=torch.float32) for lab in labels]).to(device) / self.temperature

        target = (1 - self.alpha) * teacher.softmax(dim=1)
        target[:, 0] += self.alpha
        return -(target * student.log_softmax(dim=1)).sum(dim=1).mean()

    def get_config_dict(self) -> dict:
        return {"alpha": self.alpha, "temperature": self.temperature}


def dev_samples(dev_rows: list[dict], dev_file: Path | None, distractors: int, seed: int, revision: str | None, k: int) -> tuple[list[dict], int, float]:
    """BM25 top-k over train.py's dev corpus for every dev question, in CrossEncoderRerankingEvaluator's format."""
    import bm25s
    import numpy as np

    corpus: list[str] = []
    seen: dict[str, int] = {}
    for r in dev_rows:
        if r["positive"] not in seen:
            seen[r["positive"]] = len(corpus)
            corpus.append(r["positive"])
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
        corpus += pool[:distractors]

    tokens = bm25s.tokenize(corpus, lower=True, token_pattern=TOKEN_RE, stopwords=None, show_progress=False)
    bm25 = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    bm25.index(tokens, show_progress=False)
    vocab = tokens.vocab
    qids = [[vocab[t] for t in re.findall(TOKEN_RE, r["query"].lower()) if t in vocab] or [0] for r in dev_rows]
    hits, _ = bm25.retrieve(qids, k=min(k, len(corpus)), show_progress=False)
    samples, found = [], 0
    for r, row_hits in zip(dev_rows, np.asarray(hits)):
        docs = [corpus[j] for j in row_hits.tolist()]
        found += r["positive"] in docs
        samples.append({"query": r["query"], "positive": [r["positive"]], "documents": docs})
    return samples, len(corpus), found / max(1, len(dev_rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--groups", required=True, help="prep_rerank.py teacher output (groups.jsonl)")
    ap.add_argument("--dev", required=True, help="mine.py dev.jsonl (DOG 2024 questions)")
    ap.add_argument("--dev-file", default=None, help="local DOG 2024 jsonl for dev distractors (default: download)")
    ap.add_argument("--corpus-revision", default=None)
    ap.add_argument("--dev-distractors", type=int, default=5000)
    ap.add_argument("--dev-reference", default=TEACHER, help="reranker scored on dev once for comparison ('none' to skip)")
    ap.add_argument("--out", default="rerank")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8, help="questions per device step (each brings up to 8 passages)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the positive's one-hot in the target")
    ap.add_argument("--temperature", type=float, default=1.0, help="teacher logits are divided by this")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: first N groups")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
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

    groups = read_jsonl(Path(args.groups), args.limit)
    train = Dataset.from_dict(
        {"query": [g["query"] for g in groups], "docs": [g["docs"] for g in groups], "labels": [g["teacher"] for g in groups]}
    ).shuffle(seed=args.seed)
    sources, kinds = {}, {}
    for g in groups:
        sources[g["source"]] = sources.get(g["source"], 0) + 1
        for k in g["kinds"][1:]:
            kinds[k] = kinds.get(k, 0) + 1
    dev_rows = read_jsonl(Path(args.dev))
    samples, corpus_size, bm25_recall = dev_samples(
        dev_rows, Path(args.dev_file) if args.dev_file else None, args.dev_distractors, args.seed, args.corpus_revision, 10
    )
    evaluator = CrossEncoderRerankingEvaluator(samples, at_k=10, always_rerank_positives=False, name="dog2024-dev", batch_size=32)
    log(step="data", groups=len(train), negatives=kinds, by_source=sources, dev_queries=len(samples), dev_corpus=corpus_size, dev_bm25_recall10=round(bm25_recall, 4), world=world)

    def dev_scores(model) -> dict:
        return {k: round(float(v), 4) for k, v in evaluator(model).items() if "mrr" in k or "ndcg" in k}

    reference = None
    if args.dev_reference != "none" and rank == 0:
        ref = CrossEncoder(args.dev_reference, max_length=args.max_length, model_kwargs={"dtype": torch.float16} if cuda else None)
        reference = dev_scores(ref)
        log(step="dev_reference", model=args.dev_reference, **reference)
        del ref
        if cuda:
            torch.cuda.empty_cache()

    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length, model_kwargs={"dtype": torch.float32})
    loss = TeacherListLoss(model, alpha=args.alpha, temperature=args.temperature)

    class NanGuard(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            value = (logs or {}).get("loss")
            if value is not None and not math.isfinite(value):
                log(step="nan_loss", global_step=state.global_step)
                control.should_training_stop = True

    steps_per_epoch = max(1, len(train) // (args.batch_size * args.grad_accum * world))
    eval_steps = min(args.eval_steps, steps_per_epoch)
    targs = CrossEncoderTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dog2024-dev_mrr@10",
        greater_is_better=True,
        logging_steps=max(1, eval_steps // 10),
        dataloader_drop_last=True,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    before = dev_scores(model)
    log(step="dev_before", **before)
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train, loss=loss, evaluator=evaluator, callbacks=[NanGuard()])
    trainer.train()

    if rank == 0:
        final = out / "final"
        model.save_pretrained(str(final))
        after = dev_scores(model)
        summary = {
            "base": args.base,
            "groups": len(train),
            "negatives": kinds,
            "by_source": sources,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "effective_batch": args.batch_size * args.grad_accum * world,
            "dev_bm25_recall10": round(bm25_recall, 4),
            "dev_reference": reference,
            "dev_before": before,
            "dev_after": after,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "history": [h for h in trainer.state.log_history if any("mrr" in k for k in h)],
            "minutes": round((time.time() - t0) / 60, 1),
        }
        (out / "train_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        log(step="done", dev_after=after, best=summary["best_checkpoint"], minutes=summary["minutes"])


if __name__ == "__main__":
    main()
