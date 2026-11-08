"""Share of top-1 errors that return a document from the same issuer as the expected one.

DOG ids look like AnuncioG0759-220525-0002_gl; the issuer is the code after "Anuncio".
Run from the repo root: python analysis/issuer_confusion.py
"""
import json,re,sys
sys.path.insert(0,'src')
from types import SimpleNamespace
import eval_dog
M='results/mac/pipeline/runs'; G='results/gcp/rerank1/test_rerank1/runs'; CE='__ce_upstream_gcp_rerank1_rerank1_final.jsonl'
runs={'bm25':f'{M}/bm25__none.jsonl','bge-m3':f'{G}/bge-m3_plain__none.jsonl','bge-m3+bge-rr':f'{G}/bge-m3_plain__bge-reranker-v2-m3.jsonl','bm25+bge-rr':f'{G}/bm25__bge-reranker-v2-m3.jsonl','bm25+ours-rr':f'{M}/bm25{CE}','mix':f'{M}/galician-mix_plain__none.jsonl','mix+ours-rr':f'{M}/galician-mix_plain{CE}'}
_,qs=eval_dog.load_inputs(SimpleNamespace(dog2025=None,dograg_dir=None,corpus_revision=None,dograg_revision=None,max_distractors=None,max_queries=None,seed=0))
def issuer(d):
    m=re.match(r'Anuncio([A-Z]+\d+)',d); return m.group(1) if m else d
print(f"{'run':16s} misses  top1-wrong  same-issuer-as-gold")
for n,p in runs.items():
    rows=[json.loads(l) for l in open(p)]
    miss=sum(r['mrr']==0 for r in rows)
    wrong=[(q,r) for q,r in zip(qs,rows) if r['final'][0][0]!=q['doc_id']]
    same=sum(issuer(r['final'][0][0])==issuer(q['doc_id']) for q,r in wrong)
    print(f"{n:16s} {miss:6d}  {len(wrong):10d}  {same:5d} ({same/len(wrong):.0%})")
from collections import Counter
chunks,_=eval_dog.load_inputs(SimpleNamespace(dog2025=None,dograg_dir=None,corpus_revision=None,dograg_revision=None,max_distractors=None,max_queries=None,seed=0))
docs={c['doc_id'] for c in chunks}; cnt=Counter(issuer(d) for d in docs)
base=sum((cnt[issuer(q['doc_id'])]-1)/(len(docs)-1) for q in qs)/len(qs)
print(f"chance that a random other document shares the gold's issuer: {base:.0%}; issuers: {len(cnt)}; top: {cnt.most_common(4)}")
