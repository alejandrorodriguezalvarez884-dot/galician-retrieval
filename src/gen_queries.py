"""Synthetic Galician questions over DOG and provincial bulletin passages, with Salamandra-7b-instruct.

Passages are 250-word chunks (as in the DOG-RAG evaluation) of Galician documents from:
- DOG 1995-2023 (train) and DOG 2024 (dev, --dev-passages), never 2025 or later (the test set);
- BOP A Coruña 2009-2022 and BOPPO 2014-2024, whose "gl" folders also hold Spanish
  documents, so every chunk goes through a Galician/Spanish check.

The model writes three self-contained questions per passage: a general one, one
that extracts a concrete datum (amount, date, deadline, body) and one that needs
two parts of the passage. No DOG-RAG item is ever shown to it.

Backends: vLLM with tensor parallelism over all GPUs (default), or plain
transformers with device_map="auto" if vLLM is missing or fails. Output is written
in shards, so a run stopped by --max-hours keeps what it finished.

    python gen_queries.py --passages 40000 --dev-passages 400 --out synth --max-hours 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

CORPUS_REPO = "proxectonos/corpus_dominio_legal_administrativo_tmp"
CORPUS_REVISION = "829ca66f2718b58a34b62e62ee600707a8380cc3"
MODEL = "BSC-LT/salamandra-7b-instruct"
MODEL_REVISION = "a3ed5452fafb3698a0423b1a18bfb5888f4e611b"

TRAIN_FILES = (
    [f"XuntaGalicia_DOG/gl/{y}_gl.jsonl" for y in range(1995, 2024)]
    + [f"DeputacionCoruna_BOP/gl/{y}.jsonl" for y in range(2009, 2023)]
    + [f"DeputacionPontevedra_BOPPO/gl/{y}.jsonl" for y in (2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024)]
)
DEV_FILES = ["XuntaGalicia_DOG/gl/2024_gl.jsonl"]

CHUNK_WORDS, CHUNK_OVERLAP = 250, 50
SHARES = {"XuntaGalicia_DOG": 0.75, "DeputacionCoruna_BOP": 0.15, "DeputacionPontevedra_BOPPO": 0.10}
GL_WORDS = {"do", "da", "dos", "das", "ao", "aos", "á", "ás", "polo", "pola", "unha", "non", "tamén", "deste", "desta", "cando", "como", "xunta", "concello", "segundo", "nas", "nos", "coa", "co", "mediante", "este", "esta"}
ES_WORDS = {"del", "el", "los", "las", "al", "una", "y", "también", "este", "esta", "ayuntamiento", "según", "como", "lo", "sus", "pero", "hasta"}
GL_ONLY = GL_WORDS - ES_WORDS
ES_ONLY = ES_WORDS - GL_WORDS

SYSTEM = "Es un experto en dereito administrativo galego. Respondes sempre en galego normativo."
PROMPT = """Le este fragmento dun boletín oficial de Galicia.

Título: {title}
Fragmento:
\"\"\"
{passage}
\"\"\"

Escribe tres preguntas en galego que unha persoa cidadá ou un profesional faría a un buscador e que se respondan co contido deste fragmento:
1. unha pregunta xeral sobre o asunto do fragmento;
2. unha pregunta sobre un dato concreto (unha cantidade, unha data, un prazo, un requisito ou un organismo);
3. unha pregunta que precise combinar dúas partes distintas do fragmento.

Cada pregunta debe entenderse soa, sen ver o fragmento: nomea o procedemento, o organismo, o concello ou a norma do que trata. Non uses expresións como "o fragmento", "o texto", "o documento", "este anuncio" ou "esta resolución". Non copies frases enteiras do fragmento.

Responde só cunha lista JSON de tres cadeas, sen nada máis."""

# ParlaMint-ES-GA passages (prep_parlamint.py): speeches, so the questions must be
# about the policies, laws and figures discussed, not about who said what.
PARL_SOURCE = "ParlaMint-ES-GA"
PARL_SYSTEM = "Es un experto en política e administración pública galega. Respondes sempre en galego normativo."
PARL_PROMPT = """Le este fragmento dunha intervención no Parlamento de Galicia.

{title}
Fragmento:
\"\"\"
{passage}
\"\"\"

Escribe ata tres preguntas en galego que unha persoa cidadá faría a un buscador e que se respondan co contido deste fragmento. Pregunta polas políticas, leis, plans, servizos públicos, orzamentos ou datos dos que se fala (cantidades, datas, prazos, organismos, concellos):
1. unha pregunta xeral sobre o asunto;
2. unha pregunta sobre un dato concreto;
3. unha pregunta que precise combinar dúas partes distintas do fragmento.

Cada pregunta debe entenderse soa: nomea a lei, o plan, o servizo, o organismo ou o lugar do que trata. Non preguntes pola opinión de quen fala nin uses expresións como "o fragmento", "o orador", "esta intervención" ou "o debate". Non copies frases enteiras. Se o fragmento é só opinión ou retórica, sen ningún feito concreto, responde [].

Responde só cunha lista JSON de cadeas, sen nada máis."""

BAD_REFERENCES = re.compile(
    r"\b(fragmento|o texto|este texto|o documento|este documento|este anuncio|esta resoluci[oó]n|esta orde|este decreto|o anuncio)\b",
    re.IGNORECASE,
)
PARL_BAD_REFERENCES = re.compile(r"\b(o orador|a oradora|esta intervenci[oó]n|este debate|o debate|vostede|vostedes)\b", re.IGNORECASE)


# ---------------------------------------------------------------- passages


def is_galician(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    gl = sum(w in GL_ONLY for w in words)
    es = sum(w in ES_ONLY for w in words)
    return gl >= 8 and gl > 3 * es


def chunks_of(text: str) -> list[str]:
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, len(words), step) if len(words[i : i + CHUNK_WORDS]) >= 80]


def load_passages(files: list[str], n: int, seed: int, split: str, per_doc: int = 2) -> list[dict]:
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    pool = []
    for name in files:
        path = hf_hub_download(CORPUS_REPO, name, repo_type="dataset", revision=CORPUS_REVISION)
        source = name.split("/")[0]
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                text = d.get("text") or ""
                if len(text.split()) < 80:
                    continue
                title = d.get("title") or d.get("summary") or ""
                doc_id = d.get("url") or d.get("id")
                cs = chunks_of(text)
                for j in rng.sample(range(len(cs)), min(per_doc, len(cs))):
                    pool.append({"doc_id": doc_id, "source": source, "file": name, "pos": j, "title": title[:300], "passage": cs[j]})
    rng.shuffle(pool)
    # The test set is DOG, so DOG gets most of the passages; leftover quota goes to whatever has more.
    quota = {s: round(n * SHARES.get(s, 0)) for s in {p["source"] for p in pool}}
    if sum(quota.values()) < n:
        quota[max(quota, key=quota.get)] += n - sum(quota.values())
    out, taken = [], {s: 0 for s in quota}
    for p in pool:
        if taken[p["source"]] < quota[p["source"]] and is_galician(p["passage"]):
            taken[p["source"]] += 1
            p["id"] = f"{split}-{len(out)}"
            p["split"] = split
            out.append(p)
    if len(out) < n:  # a source ran short: top up from the others
        used = {id(p) for p in out}
        for p in pool:
            if len(out) >= n:
                break
            if id(p) not in used and is_galician(p["passage"]):
                p["id"] = f"{split}-{len(out)}"
                p["split"] = split
                out.append(p)
    rng.shuffle(out)
    print(json.dumps({"split": split, "pool": len(pool), "kept": len(out), "by_source": dict(Counter(p["source"] for p in out))}))
    return out


# ---------------------------------------------------------------- generation


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _strings(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _strings(v)]
    return []


def parse_questions(text: str, extra_bad: re.Pattern | None = None) -> list[str]:
    """Questions from the model output: a JSON list (or any JSON shape), else numbered lines."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    qs = None
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if m:
        try:
            qs = _strings(json.loads(m.group(0)))
        except Exception:
            qs = None
    if qs is None:
        qs = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip(' ",') for line in text.splitlines()]
    out = []
    for q in qs:
        q = re.sub(r"\s+", " ", q).strip().strip('"')
        n = len(q.split())
        if 5 <= n <= 60 and q.endswith("?") and not BAD_REFERENCES.search(q) and not (extra_bad and extra_bad.search(q)) and is_galician_question(q):
            out.append(q)
    return list(dict.fromkeys(out))[:3]


def is_galician_question(q: str) -> bool:
    words = re.findall(r"\w+", q.lower())
    return sum(w in ES_ONLY for w in words) <= sum(w in GL_ONLY for w in words) + 1


class VLLMBackend:
    def __init__(self, model: str, revision: str, tp: int, max_model_len: int):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model,
            revision=revision,
            tensor_parallel_size=tp,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.92,
            disable_custom_all_reduce=True,  # two T4s over PCIe, no NVLink
            enforce_eager=False,
            seed=0,
        )
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=300, seed=0)

    def generate(self, prompts: list[str]) -> list[str]:
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class HFBackend:
    def __init__(self, model: str, revision: str, batch_size: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.float16, device_map="auto").eval()
        self.batch_size = batch_size

    def generate(self, prompts: list[str]) -> list[str]:
        import torch

        out = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[i : i + self.batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return out


def build_prompt(tok, p: dict) -> str:
    system, prompt = (PARL_SYSTEM, PARL_PROMPT) if p.get("source") == PARL_SOURCE else (SYSTEM, PROMPT)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt.format(title=p["title"] or "(sen título)", passage=p["passage"])},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passages", type=int, default=40_000)
    ap.add_argument("--dev-passages", type=int, default=400)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--tp", type=int, default=None, help="tensor parallel size (default: all GPUs)")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--hf-batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=2_000)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--model", default=MODEL, help="smoke tests only: a small chat model")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--train-files", default=None, help="smoke tests only: comma list of corpus files")
    ap.add_argument("--only-passages", action="store_true", help="build the passage files and stop (CPU smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="synth")
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    passages_path = out / "passages.jsonl"
    if passages_path.exists():
        passages = [json.loads(line) for line in open(passages_path, encoding="utf-8")]
    else:
        passages = load_passages(DEV_FILES, args.dev_passages, args.seed, "dev") if args.dev_passages else []
        train_files = args.train_files.split(",") if args.train_files else TRAIN_FILES
        passages += load_passages(train_files, args.passages, args.seed, "train")
        with open(passages_path, "w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"passages": len(passages), "minutes": round((time.time() - t_start) / 60, 1)}))
    if args.only_passages:
        return
    revision = args.model_revision if args.model != MODEL else MODEL_REVISION

    backend = None
    if args.backend == "vllm":
        try:
            import torch

            backend = VLLMBackend(args.model, revision, args.tp or torch.cuda.device_count(), args.max_model_len)
        except Exception as exc:
            print(json.dumps({"vllm_failed": f"{type(exc).__name__}: {exc}"[:1000]}))
    if backend is None:
        backend = HFBackend(args.model, revision, args.hf_batch_size)
    print(json.dumps({"backend": type(backend).__name__, "minutes": round((time.time() - t_start) / 60, 1)}))

    total_q = 0
    for s, start in enumerate(range(0, len(passages), args.shard_size)):
        path = out / f"shard_{s:04d}.jsonl"
        shard = passages[start : start + args.shard_size]
        if path.exists():
            continue
        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            print(json.dumps({"stopped": "max-hours", "shards_done": s}))
            break
        t0 = time.time()
        texts = backend.generate([build_prompt(backend.tok, p) for p in shard])
        n_q = 0
        with open(path, "w", encoding="utf-8") as f:
            for p, raw in zip(shard, texts):
                qs = parse_questions(raw, PARL_BAD_REFERENCES if p.get("source") == PARL_SOURCE else None)
                n_q += len(qs)
                f.write(json.dumps(dict(p, questions=qs, raw=raw), ensure_ascii=False) + "\n")
        total_q += n_q
        print(
            json.dumps(
                {
                    "shard": s,
                    "passages": start + len(shard),
                    "questions": total_q,
                    "q_per_passage": round(n_q / len(shard), 2),
                    "passages_per_min": round(len(shard) / (time.time() - t0) * 60),
                    "minutes": round((time.time() - t_start) / 60, 1),
                    "example": next((json.loads(line)["questions"] for line in open(path, encoding="utf-8")), None),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
