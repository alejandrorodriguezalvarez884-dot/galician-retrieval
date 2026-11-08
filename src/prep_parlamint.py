"""Galician passages from Parlamento de Galicia debates (ParlaMint-ES-GA 5.0) for gen_queries.py.

ParlaMint-ES-GA covers plenary sittings from 2015-01 to 2022-05, 17.8M words,
CC BY 4.0 (ILG-USC, CLARIN.SI handle 11356/2004). Only speeches by deputies and
government members are used; the chair's turns and the "other" and "mixed"
topics are dropped, because they are mostly procedure. Speeches are cut into the
same 250-word chunks as the DOG passages, at most --per-speech chunks each, and
sampled across topics with a cap so health and the economy don't take over.

The output is a passages.jsonl in the schema gen_queries.py reads, so the
questions come from running it on the same folder:

    python prep_parlamint.py --passages 12000 --out synth_parl
    python gen_queries.py --out synth_parl --max-hours 1.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from gen_queries import chunks_of, is_galician

URL = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2004/ParlaMint-ES-GA.tgz"
MD5 = "7c8c60ce3cd6f7a52ccc0bd7c8fd5724"
SOURCE = "ParlaMint-ES-GA"
TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
SKIP_TOPICS = {"other", "mixed"}
SKIP_TAGS = {f"{TEI}{t}" for t in ("note", "kinesic", "vocal", "incident", "gap", "desc")}
TOPIC_CAP = 0.15
MIN_SPEECH_WORDS = 120


def fetch(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != MD5:
        raise SystemExit(f"{path}: md5 {md5}, expected {MD5}")
    return path


def seg_text(seg: ET.Element) -> str:
    parts = [seg.text or ""]
    for child in seg:
        if child.tag not in SKIP_TAGS:
            parts.append(seg_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def topic_labels(root: ET.Element) -> dict[str, str]:
    labels = {}
    for cat in root.iter(f"{TEI}category"):
        for desc in cat.findall(f"{TEI}catDesc"):
            term = desc.find(f"{TEI}term")
            if desc.get(XML_LANG) == "gl" and term is not None and term.text:
                labels[cat.get(XML_ID)] = term.text.strip()
    return labels


def speeches(tgz: Path):
    """Yields (session_id, date, term, topic, speech_text) for every regular speech."""
    with tarfile.open(tgz, "r:gz") as tar:
        labels = {}
        for m in tar:
            if m.isfile() and m.name.endswith("ParlaMint-taxonomy-topic.xml"):
                labels = topic_labels(ET.parse(tar.extractfile(m)).getroot())
                break
        yield labels
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if not (m.isfile() and re.fullmatch(r"ParlaMint-ES-GA_\d{4}-\d\d-\d\d.*\.xml", name)):
                continue
            root = ET.parse(tar.extractfile(m)).getroot()
            sitting = term = ""
            for meeting in root.iter(f"{TEI}meeting"):
                ana = meeting.get("ana", "")
                if "#parla.sitting" in ana:
                    sitting = (meeting.text or "").strip()
                elif "#parla.term" in ana:
                    term = (meeting.text or "").strip()
            for u in root.iter(f"{TEI}u"):
                ana = u.get("ana", "").split()
                topic = next((a.split(":", 1)[1] for a in ana if a.startswith("topic:")), "other")
                if "#regular" not in ana or topic in SKIP_TOPICS:
                    continue
                text = " ".join(" ".join(seg_text(s).split()) for s in u.iter(f"{TEI}seg"))
                if len(text.split()) >= MIN_SPEECH_WORDS:
                    yield name[:-4], sitting, term, topic, text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passages", type=int, default=12_000)
    ap.add_argument("--per-speech", type=int, default=2)
    ap.add_argument("--tgz", default="parlamint/ParlaMint-ES-GA.tgz", help="downloaded here if missing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="synth_parl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = speeches(fetch(Path(args.tgz)))
    labels = next(gen)
    pool, n_speeches = [], 0
    for session, sitting, term, topic, text in gen:
        n_speeches += 1
        # A digit is a cheap sign of something concrete (an amount, a date, a law number);
        # chunks without one are mostly rhetoric and give poor questions.
        cs = [c for c in chunks_of(text) if is_galician(c) and re.search(r"\d", c)]
        for j in rng.sample(range(len(cs)), min(args.per_speech, len(cs))):
            # The topic label is per speech and often doesn't fit a 250-word chunk, so it stays out of the title.
            title = f"Diario de Sesións do Parlamento de Galicia, {term}, sesión do {sitting}"
            pool.append({"doc_id": f"{session}#{n_speeches}", "source": SOURCE, "file": session, "pos": j, "topic": topic, "title": title, "passage": cs[j]})
    rng.shuffle(pool)

    by_topic = Counter(p["topic"] for p in pool)
    cap = int(args.passages * TOPIC_CAP)
    quota = {t: min(c, cap) for t, c in by_topic.items()}
    while sum(quota.values()) < min(args.passages, len(pool)):  # spread what the cap freed over topics with room
        room = [t for t in quota if quota[t] < by_topic[t]]
        if not room:
            break
        for t in room:
            quota[t] += 1
    scale = min(1.0, args.passages / sum(quota.values()))
    quota = {t: int(q * scale) for t, q in quota.items()}

    kept, taken = [], Counter()
    for p in pool:
        if taken[p["topic"]] < quota[p["topic"]]:
            taken[p["topic"]] += 1
            p["id"] = f"parl-{len(kept)}"
            p["split"] = "train"
            kept.append(p)
    with open(out / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"speeches": n_speeches, "pool": len(pool), "kept": len(kept), "by_topic": {labels.get(t, t): n for t, n in taken.most_common()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
