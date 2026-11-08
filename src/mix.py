"""Interpolate a fine-tuned checkpoint with its base model (weight-space ensembling, WiSE-FT).

    w(a) = a * tuned + (1 - a) * base

Run 4 gained on DOG but lost ground on short passages (news paragraphs, WebFAQ answers), where
the untouched mE5-base does better. Moving the weights part of the way back can recover some
of that while keeping most of the DOG gain, with no extra training.

The mixing weight is chosen on DOG 2024 dev only, with a rule fixed before any result: the
smallest a whose dev MRR@10 stays within --tolerance of a = 1 (run 4 itself). Every mix is
saved under --scratch so the secondary evals can show the whole curve; only the chosen one is
copied to OUT/final for the DOG-RAG test.

    python mix.py --tuned run4/final --dev mined2/dev.jsonl --out mix --scratch /tmp/mix
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from train import dev_evaluator, read_jsonl


def mrr10(scores: dict) -> float:
    return next(v for k, v in scores.items() if k.endswith("mrr@10"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuned", required=True, help="fine-tuned SentenceTransformer folder")
    ap.add_argument("--base", default="intfloat/multilingual-e5-base")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--dev-file", default=None)
    ap.add_argument("--dev-distractors", type=int, default=5000, help="same as train.py, so dev numbers line up")
    ap.add_argument("--alphas", default="1.0,0.9,0.8,0.7,0.6,0.5,0.0")
    ap.add_argument("--tolerance", type=float, default=0.01, help="largest dev MRR@10 drop allowed against a = 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True, help="where every mix is saved (not kept as notebook output)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel

    out, scratch = Path(args.out), Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "dev.jsonl", "w", encoding="utf-8")

    def log(**kw):
        line = json.dumps(kw, ensure_ascii=False)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.tuned, device=device)
    tuned = {k: v.detach().clone() for k, v in model[0].auto_model.state_dict().items()}
    base = AutoModel.from_pretrained(args.base).state_dict()
    # Run 4 dropped the unused pooler for DDP, so the base has keys the tuned model lacks.
    missing = [k for k in tuned if k not in base or base[k].shape != tuned[k].shape]
    assert not missing, f"tuned weights not in the base: {missing[:5]}"
    log(step="loaded", tuned=args.tuned, base=args.base, tensors=len(tuned), base_only=sorted(set(base) - set(tuned))[:4])

    evaluator, corpus_size = dev_evaluator(
        read_jsonl(Path(args.dev)), Path(args.dev_file) if args.dev_file else None, args.dev_distractors, args.seed, None
    )
    log(step="dev", corpus=corpus_size)

    alphas = [float(a) for a in args.alphas.split(",")]
    results = {}
    for a in alphas:
        t0 = time.time()
        with torch.no_grad():
            for k, p in model[0].auto_model.state_dict().items():
                if p.is_floating_point():
                    p.copy_(a * tuned[k] + (1 - a) * base[k].to(p.device, p.dtype))
        scores = evaluator(model)
        results[a] = mrr10(scores)
        path = scratch / f"a{a:.2f}"
        model.save(str(path))
        log(step="alpha", alpha=a, path=str(path), seconds=round(time.time() - t0),
            **{k.split("_", 1)[-1]: round(v, 4) for k, v in scores.items() if "mrr" in k or "ndcg" in k or "accuracy@1" in k})

    ref = results[max(alphas)]
    ok = [a for a in alphas if results[a] >= ref - args.tolerance]
    chosen = min(ok)
    shutil.copytree(scratch / f"a{chosen:.2f}", out / "final", dirs_exist_ok=True)
    log(step="chosen", alpha=chosen, dev_mrr10=round(results[chosen], 4), reference=round(ref, 4), tolerance=args.tolerance)


if __name__ == "__main__":
    main()
