"""Label/audio alignment check: transcribe a sample of TRAIN utterances per (corpus,lang) with IndicConformer(GPU, batched) and compare to the manifest text.
High CER = misaligned/noisy labels or wrong audio<->text pairing."""
import sys, json, random, collections, soundfile as sf
sys.path.insert(0, ".."); from metrics import score; from ic_gpu import ICGpu
M = "/home/shibam/IE643/data/processed"
rows = [json.loads(l) for l in open(f"{M}/splits/train.jsonl", encoding="utf8")]
g = collections.defaultdict(list)
for r in rows:
    if r["lang"] != "en": g[(r["corpus"] + ("-cs" if r["cs"] else ""), r["lang"])].append(r)
ic = ICGpu(); rng = random.Random(3)
for k in sorted(g):
    rs = rng.sample(g[k], min(96, len(g[k]))); w = [sf.read(r["path"], dtype="float32")[0] for r in rs]
    order = sorted(range(len(rs)), key=lambda i: len(w[i])); out = {}
    for s in range(0, len(order), 32):
        ids = order[s:s + 32]
        for i, h in zip(ids, ic([w[i] for i in ids], k[1])): out[i] = h
    ref = [r["text"] for r in rs]; hyp = [out[i] for i in range(len(rs))]
    per = [score([a], [b], k[1])["cer"] for a, b in zip(ref, hyp)]
    s = score(ref, hyp, k[1]); print(f"{k[0]:16s} {k[1]}  CER {s['cer']:.3f}  WER {s['wer']:.3f}  utts with CER>0.5: {sum(p>0.5 for p in per)}/{len(per)}", flush=True)
