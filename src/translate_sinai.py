"""Translate a sample of the SINAI Spanish legal triplets into Galician with Nós MT.

Only the query and the positive passage are kept; hard negatives are mined again
later on the Galician side. The 32 GB training file is never downloaded whole:
the script reads byte ranges at evenly spread random offsets, so the sample covers
the whole file instead of the first few documents.

Pipeline per text (as in the Nos_MT-CT2-es-gl card): sentence split, the model's
own tokenizer.perl, BPE with es.code, CTranslate2, de-BPE, detokenizer.perl.

Positives are cut to at most --max-words words at a sentence boundary, the same
size as the 250-word chunks of the DOG-RAG evaluation. Output is written in
shards as it goes, so a run stopped by --max-hours keeps everything finished.

    python translate_sinai.py --pairs 100000 --out sinai_gl --max-hours 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

SINAI_REPO = "SINAI/ALIA-es-legal-administrative-triplets"
SINAI_FILE = "ALIA-es-legal-administrative-triplets-train.jsonl"
SINAI_REVISION = "0a3753db69cd99923f953b8a3fde3d42fb6e8ca8"
SINAI_SIZE = 32373691524
MT_REPO = "proxectonos/Nos_MT-CT2-es-gl"
MT_REVISION = "435742ee156b8d5d2531141ac8aa368785228b5e"

SENT_END = re.compile(r"(?<=[.!?;:])\s+(?=[\"«(¿¡A-ZÁÉÍÓÚÑ0-9])")
WORD = re.compile(r"\w+", re.UNICODE)
VOWEL = re.compile(r"[aeiouáéíóúü]", re.IGNORECASE)


# ---------------------------------------------------------------- sampling


def read_range(url: str, start: int, length: int, token: str | None) -> list[str]:
    import requests

    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            text = r.content.decode("utf-8", "ignore")
            return text.split("\n")[1:-1]  # drop the partial first and last lines
        except Exception as exc:  # network hiccups on Kaggle
            print(f"range {start}: {exc}, retry {attempt + 1}")
            time.sleep(5 * (attempt + 1))
    return []


def clean(text: str) -> str:
    text = re.sub(r"[‐‑‒–−]", "-", text)  # the MT tokenizer turns these into "_"
    text = re.sub(r"-\n(?=\w)", "", text)  # words hyphenated across PDF lines
    return re.sub(r"\s+", " ", text).strip()


def looks_clean(text: str) -> bool:
    """Drops PDF extractions that are mostly broken words or tables."""
    words = WORD.findall(text)
    if len(words) < 40:
        return False
    alpha = [w for w in words if w.isalpha()]
    if len(alpha) < 0.6 * len(words):
        return False
    good = sum(1 for w in alpha if len(w) <= 20 and VOWEL.search(w))
    return good >= 0.92 * len(alpha)


def split_sentences(text: str, max_words: int = 80) -> list[str]:
    out = []
    for sent in SENT_END.split(text):
        words = sent.split()
        while len(words) > max_words:  # long legal sentences: cut at a comma near the limit, else hard
            cut = next((i + 1 for i in range(max_words - 1, max_words // 2, -1) if words[i].endswith((",", ";"))), max_words)
            out.append(" ".join(words[:cut]))
            words = words[cut:]
        if words:
            out.append(" ".join(words))
    return out


def truncate_words(sentences: list[str], max_words: int) -> list[str]:
    kept, n = [], 0
    for s in sentences:
        w = len(s.split())
        if n + w > max_words and kept:
            break
        kept.append(s)
        n += w
    return kept


def sample(args) -> list[dict]:
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(SINAI_REPO, SINAI_FILE, repo_type="dataset", revision=SINAI_REVISION)
    rng = random.Random(args.seed)
    per_range = max(1, args.chunk_bytes // 18_000)  # ~18 KB per line with 5 negatives
    n_ranges = args.pairs // per_range * 13 // 10 + 1  # ~30% are dropped by the filters
    offsets = sorted(rng.randrange(0, SINAI_SIZE - args.chunk_bytes) for _ in range(n_ranges))
    seen, rows = set(), []
    for i, off in enumerate(offsets):
        for line in read_range(url, off, args.chunk_bytes, args.hf_token):
            try:
                d = json.loads(line)
                query = clean(d["messages"][0]["content"])
                positive = clean(d["positive_messages"][0][0]["content"])
            except Exception:
                continue
            key = positive[:300]
            if key in seen or not looks_clean(positive) or len(query.split()) < 4:
                continue
            seen.add(key)
            rows.append(
                {
                    "id": f"sinai-{off}-{len(rows)}",
                    "difficulty": d.get("difficulty"),
                    "query_es": query,
                    "positive_es": " ".join(truncate_words(split_sentences(positive), args.max_words)),
                }
            )
        if (i + 1) % 50 == 0:
            print(f"sampled {i + 1}/{len(offsets)} ranges, {len(rows)} pairs")
        if len(rows) >= args.pairs:
            break
    rng.shuffle(rows)
    return rows[: args.pairs]


# ---------------------------------------------------------------- translation


class NosMT:
    def __init__(self, model_dir: Path, device: str, compute_type: str, beam: int, batch_tokens: int):
        import ctranslate2
        from subword_nmt.apply_bpe import BPE

        self.dir = model_dir
        with open(model_dir / "es.code", encoding="utf-8") as f:
            self.bpe = BPE(f)
        if device == "cuda":
            import torch

            index = list(range(max(1, torch.cuda.device_count())))
        else:
            index = [0]
        self.beam = beam
        self.batch_tokens = batch_tokens
        try:
            self.tr = ctranslate2.Translator(str(model_dir), device=device, device_index=index, compute_type=compute_type)
            print(json.dumps({"mt_device": device, "devices": index, "test": self.translate(["La solicitud se presentará en el plazo de un mes."])}, ensure_ascii=False))
        except Exception as exc:  # e.g. no CUDA 12 cuBLAS in the image: keep going on CPU, slowly
            print(json.dumps({"mt_cuda_failed": f"{type(exc).__name__}: {exc}"[:500]}))
            self.tr = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8")

    def _perl(self, script: str, lines: list[str]) -> list[str]:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines) + "\n")
        res = subprocess.run(["perl", str(self.dir / script)], stdin=open(f.name, encoding="utf-8"), capture_output=True)
        Path(f.name).unlink()
        out = res.stdout.decode("utf-8").split("\n")
        if len(out) < len(lines) or res.returncode != 0:
            raise RuntimeError(f"{script} returned {len(out)} lines for {len(lines)}: {res.stderr[:300]!r}")
        return [o.strip() for o in out[: len(lines)]]

    def translate(self, sentences: list[str]) -> list[str]:
        toks = self._perl("tokenizer.perl", sentences)
        bpe = [self.bpe.process_line(t).split() for t in toks]
        res = self.tr.translate_batch(
            bpe,
            beam_size=self.beam,
            length_penalty=0,
            disable_unk=True,
            replace_unknowns=True,
            max_decoding_length=250,
            max_batch_size=self.batch_tokens,
            batch_type="tokens",
        )
        hyp = [" ".join(r.hypotheses[0]).replace("@@ ", "") for r in res]
        return self._perl("detokenizer.perl", hyp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=100_000)
    ap.add_argument("--max-words", type=int, default=256, help="positive passages are cut to this many words")
    ap.add_argument("--chunk-bytes", type=int, default=2_000_000, help="bytes read per random offset")
    ap.add_argument("--shard-size", type=int, default=2_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sinai_gl")
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample_path = out / "sample_es.jsonl"
    if sample_path.exists():
        rows = [json.loads(line) for line in open(sample_path, encoding="utf-8")]
    else:
        rows = sample(args)
        with open(sample_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"pairs_sampled": len(rows), "minutes": round((time.time() - t_start) / 60, 1)}))

    from huggingface_hub import snapshot_download

    mt = NosMT(
        Path(snapshot_download(MT_REPO, revision=MT_REVISION)),
        args.device,
        args.compute_type,
        args.beam,
        args.batch_tokens,
    )

    done = 0
    for s, start in enumerate(range(0, len(rows), args.shard_size)):
        path = out / f"shard_{s:04d}.jsonl"
        shard = rows[start : start + args.shard_size]
        if path.exists():
            done += len(shard)
            continue
        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            print(json.dumps({"stopped": "max-hours", "translated": done}))
            break
        t0 = time.time()
        sentences, spans = [], []
        for r in shard:
            parts = []
            for field in ("query_es", "positive_es"):
                sents = split_sentences(r[field])
                parts.append((len(sentences), len(sents)))
                sentences += sents
            spans.append(parts)
        gl = mt.translate(sentences)
        with open(path, "w", encoding="utf-8") as f:
            for r, ((qa, qn), (pa, pn)) in zip(shard, spans):
                r = dict(r, query_gl=" ".join(gl[qa : qa + qn]), positive_gl=" ".join(gl[pa : pa + pn]))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done += len(shard)
        words = sum(len(x.split()) for x in sentences)
        print(json.dumps({"shard": s, "pairs": done, "words_per_s": round(words / (time.time() - t0)), "minutes": round((time.time() - t_start) / 60, 1)}))
    print(json.dumps({"translated": done}))


if __name__ == "__main__":
    main()
