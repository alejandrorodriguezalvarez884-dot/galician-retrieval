"""Build the annotation sheet and the offline form from candidates.jsonl.

Run from the repo root after sample_candidates.py:

    python human-test/build_kit.py

Writes, next to this file:
- annotation_sheet.csv: one row per candidate, semicolon-separated, UTF-8 with BOM, so it
  opens in a Spanish-locale Excel or LibreOffice without an import dialog. Columns
  pregunta_1, tipo_1, pregunta_2, tipo_2 and notas hold the drafts (empty without drafts);
  revisado is for the reviewer.
- anotacion.html: a single self-contained page (no network) with the instructions, every
  candidate with its full text, and the same fields, pre-filled with the drafts. It keeps
  the answers in the browser while typing and saves them as a CSV with the sheet's columns,
  which eval_human.py reads. Each card also shows the frozen draft for reference.

Drafts: drafts_llm.csv (same columns) when it exists, or --drafts PATH. It is never written here:
it is the frozen record of the LLM drafts that the review is compared with.

The headers are in Galician because the annotator reads them; eval_human.py documents them.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOG2025 = Path("work/hf-cache/hub/datasets--proxectonos--corpus_dominio_legal_administrativo_tmp/snapshots/"
               "829ca66f2718b58a34b62e62ee600707a8380cc3/XuntaGalicia_DOG/gl/2025_gl.jsonl")
COLUMNS = ["cand", "doc_id", "ligazon", "data", "organismo", "tipo_documento", "titulo", "extracto", "irmans",
           "pregunta_1", "tipo_1", "pregunta_2", "tipo_2", "notas", "revisado"]
FIELDS = ["pregunta_1", "tipo_1", "pregunta_2", "tipo_2", "notas"]


def sibling_text(c: dict) -> str:
    return " | ".join(f"{s['date']}: {s['title'][:140]}" for s in c["siblings"])


def read_drafts(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return {r["cand"]: {k: (r.get(k) or "").strip() for k in FIELDS if (r.get(k) or "").strip()} for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", type=Path, default=HERE / "drafts_llm.csv")
    args = ap.parse_args()
    drafts = read_drafts(args.drafts) if args.drafts.exists() else {}
    cands = [json.loads(line) for line in open(HERE / "candidates.jsonl", encoding="utf-8")]
    wanted = {c["doc_id"] for c in cands}
    texts = {}
    for line in open(DOG2025, encoding="utf-8"):
        d = json.loads(line)
        did = d["url"].rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
        if did in wanted:
            texts[did] = d["text"]

    with open(HERE / "annotation_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(COLUMNS)
        for c in cands:
            d = drafts.get(c["cand"], {})
            w.writerow([c["cand"], c["doc_id"], c["url"], c["date"], c["issuer"], c["type"], c["title"],
                        c["excerpt"], sibling_text(c)] + [d.get(k, "") for k in FIELDS] + [""])

    instructions = (HERE / "INSTRUCIONS.md").read_text(encoding="utf-8")
    data = [{
        "cand": c["cand"], "doc_id": c["doc_id"], "url": c["url"], "date": c["date"], "issuer": c["issuer"],
        "type": c["type"], "title": c["title"], "excerpt": c["excerpt"], "words": c["words"],
        "has_table": c["table_lines"] >= 10, "siblings": [{"date": s["date"], "title": s["title"]} for s in c["siblings"]],
        "text": texts[c["doc_id"]],
    } for c in cands]
    page = TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("/*COLUMNS*/null", json.dumps(COLUMNS))
    page = page.replace("/*DRAFTS*/null", json.dumps(drafts, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("<!--INSTRUCTIONS-->", md_to_html(instructions))
    (HERE / "anotacion.html").write_text(page, encoding="utf-8")
    n_drafts = sum(bool(d.get("pregunta_1")) + bool(d.get("pregunta_2")) for d in drafts.values())
    print(f"{len(cands)} candidates, {n_drafts} drafted questions; annotation_sheet.csv and anotacion.html "
          f"({(HERE / 'anotacion.html').stat().st_size / 1e6:.1f} MB) written")


def md_to_html(md: str) -> str:
    """Just enough Markdown for INSTRUCIONS.md: headings, lists, bold, italics, code, paragraphs."""
    out, para, in_list = [], [], None

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    def flush():
        nonlocal para, in_list
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    for line in md.splitlines():
        s = line.strip()
        m_ul, m_ol = re.match(r"^[-*] (.*)", s), re.match(r"^\d+\. (.*)", s)
        if not s:
            flush()
        elif s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            out.append(f"<h{level + 1}>{inline(s[level:].strip())}</h{level + 1}>")
        elif m_ul or m_ol:
            tag = "ul" if m_ul else "ol"
            if para:
                out.append("<p>" + inline(" ".join(para)) + "</p>")
                para = []
            if in_list != tag:
                if in_list:
                    out.append(f"</{in_list}>")
                out.append(f"<{tag}>")
                in_list = tag
            out.append("<li>" + inline((m_ul or m_ol).group(1)) + "</li>")
        elif in_list and line.startswith("  "):
            out[-1] = out[-1][:-5] + " " + inline(s) + "</li>"
        else:
            if in_list:
                out.append(f"</{in_list}>")
                in_list = None
            para.append(s)
    flush()
    return "\n".join(out)


TEMPLATE = r"""<!doctype html>
<html lang="gl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Preguntas sobre o DOG</title>
<style>
:root { --bg:#fbfaf7; --fg:#1f1f1f; --muted:#666; --line:#ddd8cc; --card:#fff; --accent:#1d5d8f; --warn:#a24b00; --ok:#2e7d32; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:17px/1.5 Georgia, "Times New Roman", serif; }
header { position:sticky; top:0; z-index:5; background:#f1eee6; border-bottom:1px solid var(--line); padding:8px 16px;
         display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; font-family:system-ui, sans-serif; font-size:15px; }
header strong { font-size:16px; }
button, select, input[type=file] { font:inherit; }
button { padding:6px 12px; border:1px solid var(--accent); background:#fff; color:var(--accent); border-radius:6px; cursor:pointer; }
button.primary { background:var(--accent); color:#fff; }
main { max-width:900px; margin:0 auto; padding:16px; }
details.help { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 16px; margin-bottom:24px; }
details.help summary { cursor:pointer; font-weight:bold; font-family:system-ui, sans-serif; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:0 0 24px; }
.card.done { border-left:6px solid var(--ok); }
.card h2 { font-size:18px; margin:0 0 4px; line-height:1.35; }
.meta { color:var(--muted); font-family:system-ui, sans-serif; font-size:14px; margin-bottom:8px; }
.meta a { color:var(--accent); }
.sibs { font-family:system-ui, sans-serif; font-size:14px; background:#f6f3ea; border-radius:6px; padding:8px 12px; margin:8px 0; }
.sibs ul { margin:4px 0 0; padding-left:20px; }
.excerpt { font-style:italic; margin:8px 0; }
details.full summary { cursor:pointer; color:var(--accent); font-family:system-ui, sans-serif; font-size:15px; }
.fulltext { white-space:pre-wrap; max-height:480px; overflow:auto; border:1px solid var(--line); padding:12px; margin-top:8px;
            font-size:15px; background:#fffef9; }
.q { margin-top:12px; display:grid; grid-template-columns:1fr 150px; gap:6px 10px; }
.q label { grid-column:1 / -1; font-family:system-ui, sans-serif; font-size:14px; color:var(--muted); margin-top:6px; }
textarea { width:100%; font:inherit; padding:6px 8px; border:1px solid #bbb; border-radius:6px; min-height:2.6em; resize:vertical; }
select { padding:6px; border:1px solid #bbb; border-radius:6px; background:#fff; }
.warn { grid-column:1 / -1; color:var(--warn); font-family:system-ui, sans-serif; font-size:14px; }
.notes { grid-column:1 / -1; }
.rev { grid-column:1 / -1; font-family:system-ui, sans-serif; font-size:15px; color:var(--fg); }
details.draft { margin-top:10px; font-family:system-ui, sans-serif; font-size:14px; color:var(--muted); }
details.draft summary { cursor:pointer; }
@media (max-width:600px) { .q { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <strong>Preguntas sobre o DOG</strong>
  <span id="progress"></span>
  <button class="primary" id="save">Gardar ficheiro</button>
  <label>Recuperar un ficheiro gardado: <input type="file" id="load" accept=".csv,text/csv"></label>
  <select id="jump" aria-label="Ir ao documento"></select>
</header>
<main>
<details class="help" open>
<summary>Instrucións (premer para pregar ou despregar)</summary>
<!--INSTRUCTIONS-->
</details>
<div id="cards"></div>
</main>
<script>
const DATA = /*DATA*/null;
const COLUMNS = /*COLUMNS*/null;
const DRAFTS = /*DRAFTS*/null;
const KEY = "dog-preguntas-v2";
const TYPES = [["", "tipo…"], ["xeral", "xeral"], ["dato", "dato"], ["taboa", "táboa"], ["disperso", "disperso"]];
// Start from what this browser saved; the first time, from the drafts.
let answers = null;
try { answers = JSON.parse(localStorage.getItem(KEY)); } catch (e) { answers = null; }
if (!answers) answers = JSON.parse(JSON.stringify(DRAFTS || {}));

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function words(s) { return (s.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []); }

// Longest run of consecutive words the question shares with the document.
function sharedRun(q, docWords) {
  const w = words(q); let best = 0;
  for (let i = 0; i < w.length; i++) {
    for (let j = i + best + 1; j <= w.length; j++) {
      if (docWords.includes(" " + w.slice(i, j).join(" ") + " ")) best = j - i; else break;
    }
  }
  return best;
}

const docWords = {};
function render() {
  const box = document.getElementById("cards");
  const jump = document.getElementById("jump");
  jump.innerHTML = '<option value="">Ir ao documento…</option>';
  DATA.forEach(d => {
    const a = answers[d.cand] || {};
    const card = document.createElement("section");
    card.className = "card"; card.id = d.cand;
    const sibs = d.siblings.length ? `<div class="sibs">Documentos parecidos do mesmo organismo no índice (a pregunta debería distinguilo deles):
      <ul>${d.siblings.map(s => `<li>${esc(s.date)}: ${esc(s.title)}</li>`).join("")}</ul></div>` : "";
    const qfields = [1, 2].map(k => `
      <label for="${d.cand}-p${k}">Pregunta ${k}${k === 2 ? " (opcional)" : ""}</label>
      <textarea id="${d.cand}-p${k}" data-f="pregunta_${k}" rows="2">${esc(a["pregunta_" + k] || "")}</textarea>
      <select data-f="tipo_${k}" aria-label="Tipo da pregunta ${k}">${TYPES.map(([v, t]) =>
        `<option value="${v}"${(a["tipo_" + k] || "") === v ? " selected" : ""}>${t}</option>`).join("")}</select>
      <div class="warn" data-w="${k}"></div>`).join("");
    card.innerHTML = `
      <h2>${esc(d.cand)} · ${esc(d.title)}</h2>
      <div class="meta">${esc(d.issuer)} · ${esc(d.type)} · DOG do ${esc(d.date)} · ${d.words} palabras${d.has_table ? " · <b>ten táboa</b>" : ""}
        · <a href="${esc(d.url)}" target="_blank" rel="noopener">ver no DOG</a></div>
      ${sibs}
      <div class="excerpt">${esc(d.excerpt)}</div>
      <details class="full"><summary>Ler o texto completo</summary><div class="fulltext">${esc(d.text)}</div></details>
      <div class="q">${qfields}
        <label for="${d.cand}-n">Notas (onde está a resposta, dúbidas, outro documento que tamén responde…)</label>
        <textarea class="notes" id="${d.cand}-n" data-f="notas" rows="2">${esc(a.notas || "")}</textarea>
        <label class="rev"><input type="checkbox" data-f="revisado" value="si"${a.revisado ? " checked" : ""}> Revisado</label>
      </div>
      ${draftBox(d.cand)}`;
    box.appendChild(card);
    jump.insertAdjacentHTML("beforeend", `<option value="${d.cand}">${d.cand} · ${esc(d.title.slice(0, 70))}…</option>`);
    card.querySelectorAll("[data-f]").forEach(el => ["input", "change"].forEach(ev => el.addEventListener(ev, () => update(d, card))));
    check(d, card);
  });
  progress();
}

function draftBox(cand) {
  const o = (DRAFTS || {})[cand];
  if (!o || !(o.pregunta_1 || o.pregunta_2)) return "";
  const qs = [1, 2].filter(k => o["pregunta_" + k]).map(k => `<li>${esc(o["pregunta_" + k])} <i>(${esc(o["tipo_" + k] || "")})</i></li>`).join("");
  return `<details class="draft"><summary>Ver o borrador orixinal</summary><ul>${qs}</ul></details>`;
}

function update(d, card) {
  // Kept even when empty: an emptied field means the question was removed, not "use the draft".
  const a = {};
  card.querySelectorAll("[data-f]").forEach(el => {
    const v = el.type === "checkbox" ? (el.checked ? "si" : "") : el.value.trim();
    if (v) a[el.dataset.f] = v;
  });
  answers[d.cand] = a;
  try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
  check(d, card); progress();
}

function check(d, card) {
  if (!docWords[d.cand]) docWords[d.cand] = " " + words(d.text).join(" ") + " ";
  const a = answers[d.cand] || {};
  [1, 2].forEach(k => {
    const q = a["pregunta_" + k] || "", w = card.querySelector(`[data-w="${k}"]`);
    const msgs = [];
    if (q) {
      const run = sharedRun(q, docWords[d.cand]);
      if (run >= 8) msgs.push(`Comparte ${run} palabras seguidas co documento: mellor con palabras propias, se é posible.`);
      if (!a["tipo_" + k]) msgs.push("Falta o tipo.");
    }
    w.textContent = msgs.join(" ");
  });
  card.classList.toggle("done", !!a.revisado);
}

function progress() {
  let n = 0, docs = 0, rev = 0;
  Object.values(answers).forEach(a => { const m = (a.pregunta_1 ? 1 : 0) + (a.pregunta_2 ? 1 : 0); n += m; if (m) docs++; if (a.revisado) rev++; });
  document.getElementById("progress").textContent = `${n} preguntas en ${docs} documentos · ${rev} de ${DATA.length} revisados`;
}

function csvCell(v) { v = String(v ?? ""); return /[";\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
document.getElementById("save").addEventListener("click", () => {
  const rows = [COLUMNS.join(";")];
  DATA.forEach(d => {
    const a = answers[d.cand] || {};
    const base = {cand: d.cand, doc_id: d.doc_id, ligazon: d.url, data: d.date, organismo: d.issuer,
      tipo_documento: d.type, titulo: d.title, extracto: d.excerpt,
      irmans: d.siblings.map(s => s.date + ": " + s.title.slice(0, 140)).join(" | ")};
    rows.push(COLUMNS.map(c => csvCell(c in base ? base[c] : (a[c] || ""))).join(";"));
  });
  const blob = new Blob(["﻿" + rows.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  const day = new Date().toISOString().slice(0, 10);
  link.href = URL.createObjectURL(blob); link.download = `preguntas_dog_${day}.csv`;
  document.body.appendChild(link); link.click(); link.remove();
});

function parseCsv(text) {
  const rows = []; let row = [], cell = "", q = false;
  const sep = text.split("\n")[0].includes(";") ? ";" : ",";
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true;
    else if (c === sep) { row.push(cell); cell = ""; }
    else if (c === "\n") { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
document.getElementById("load").addEventListener("change", ev => {
  const file = ev.target.files[0]; if (!file) return;
  file.text().then(text => {
    const rows = parseCsv(text.replace(/^﻿/, "")); const head = rows.shift();
    const loaded = {};
    rows.forEach(r => {
      const o = {}; head.forEach((h, i) => o[h.trim()] = (r[i] || "").trim());
      const a = {}; ["pregunta_1", "tipo_1", "pregunta_2", "tipo_2", "notas", "revisado"].forEach(f => { if (o[f]) a[f] = o[f]; });
      if (o.cand) loaded[o.cand] = a;
    });
    if (!confirm("Isto substitúe o que hai agora no formulario polo do ficheiro. Seguir?")) return;
    answers = loaded;
    try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
    document.getElementById("cards").innerHTML = ""; render();
  });
});
document.getElementById("jump").addEventListener("change", ev => {
  if (ev.target.value) document.getElementById(ev.target.value).scrollIntoView({behavior: "smooth"});
});
window.addEventListener("beforeunload", () => { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} });
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
