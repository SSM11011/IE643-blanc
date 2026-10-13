import json, collections, numpy as np
from transformers import WhisperTokenizer
t = WhisperTokenizer.from_pretrained("/home/shibam/IE643/models/whisper-large-v3-turbo")
M = "/home/shibam/IE643/data/processed"
rows = [json.loads(l) for l in open(f"{M}/splits/train.jsonl", encoding="utf8")]
by = collections.defaultdict(list)
for r in rows: by[(r["lang"], r["corpus"] + ("-cs" if r["cs"] else ""))].append(r)
print("lang corpus  n   mean_tok  p95   max   frac>440   (labels = normalized text; decoder budget ~440)")
for k in sorted(by):
    rs = by[k][:3000]; n = np.array([len(t(r["text_norm"], add_special_tokens=False).input_ids) for r in rs]); d = np.array([r["dur"] for r in rs])
    print(f"{k[0]} {k[1]:14s} {len(by[k]):6d} {n.mean():6.0f} {np.percentile(n,95):5.0f} {n.max():5d}   {np.mean(n>440):.1%}   tok/s {n.sum()/d.sum():.1f}")
