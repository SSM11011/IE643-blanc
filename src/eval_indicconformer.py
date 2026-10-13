"""IndicConformer-600M (CTC) baseline on the same sampled utterances as eval_whisper.py (same seed). No English, no LID signal."""
import json, random, collections, time, argparse
import soundfile as sf
from common import DATA as M
from metrics import score
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", required=True); ap.add_argument("--hyps", default="")
a = ap.parse_args()
from ic_gpu import ICGpu
model = ICGpu()
rows = [json.loads(l) for l in open(f"{M}/splits/{a.split}.jsonl", encoding="utf8")]
rng = random.Random(0); groups = collections.defaultdict(list)
for r in rows: groups[(r["corpus"], r["lang"])].append(r)
sel = {k: rng.sample(v, min(a.n, len(v))) for k, v in groups.items()}   # identical to eval_whisper.py
res = {}; t0 = time.time(); hyp_rows = []
for (corpus, lang), rs in sorted(sel.items()):
    if lang == "en": continue
    ref, hyp = [], []
    wavs = [sf.read(r["path"], dtype="float32")[0] for r in rs]
    order = sorted(range(len(rs)), key=lambda i: len(wavs[i])); out = {}
    for k in range(0, len(order), 32):
        ids = order[k:k + 32]
        for i, h in zip(ids, model([wavs[i] for i in ids], lang)): out[i] = h
    hyp = [out[i] for i in range(len(rs))]; ref = [r["text"] for r in rs]
    hyp_rows += [{"set": f"{corpus}/{lang}", "id": r["id"], "lang": lang, "ref": r["text"], "oracle": h} for r, h in zip(rs, hyp)]
    res[f"{corpus}/{lang}"] = {"n": len(rs), "norm": score(ref, hyp, lang), "raw": score(ref, hyp, lang, normalized=False)}
    print(f"{corpus}/{lang} n={len(rs)} WER/CER {res[f'{corpus}/{lang}']['norm']['wer']:.3f}/{res[f'{corpus}/{lang}']['norm']['cer']:.3f} [{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
if a.hyps:
    with open(a.hyps, "w", encoding="utf8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in hyp_rows)
