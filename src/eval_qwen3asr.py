"""Qwen3-ASR baseline on the same sampled utterances as eval_whisper.py (same seed).
Qwen3-ASR supports only Hindi and English of our 7 languages. Every set is decoded with automatic language detection
(-> LID accuracy/confusion and "predicted-LID" CER); hi/en sets are also decoded with the language forced (-> "oracle" CER).
Run with the .venv-qwen environment (qwen-asr pins transformers 4.57)."""
import argparse, collections, json, random, time
import numpy as np, soundfile as sf, torch
from qwen_asr import Qwen3ASRModel
from common import SPLITS, MODELS, read_jsonl, write_jsonl
from metrics import score

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{MODELS}/Qwen3-ASR-0.6B")
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", required=True)
ap.add_argument("--hyps", default="", help="optional JSONL: one row per utterance")
a = ap.parse_args()

NAME = {"hi": "Hindi", "en": "English"}
CODE = {v: k for k, v in NAME.items()}
model = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0",
                                      max_inference_batch_size=a.bs, max_new_tokens=440)   # same budget as Whisper evals

groups = collections.defaultdict(list)
for r in read_jsonl(f"{SPLITS}/{a.split}.jsonl"):
    groups[(r["corpus"], r["lang"])].append(r)
rng = random.Random(0)
sel = {k: rng.sample(v, min(a.n, len(v))) for k, v in groups.items()}     # same selection as eval_whisper.py


def run(waves, language):
    out = []
    for i in range(0, len(waves), a.bs):
        lang_b = language[i:i + a.bs] if isinstance(language, list) else language
        out += model.transcribe(audio=[(w, 16000) for w in waves[i:i + a.bs]], language=lang_b)
    return out


res, t0, hyp_rows = {}, time.time(), []
for (corpus, lang), rs in sorted(sel.items()):
    waves = [sf.read(r["path"], dtype="float32")[0] for r in rs]
    ref = [r["text"] for r in rs]
    auto = run(waves, None)
    plid = [CODE.get(o.language, o.language or "none") for o in auto]
    r_ = {"n": len(rs), "pred_norm": score(ref, [o.text for o in auto], lang),
          "lid_acc": float(np.mean([p == lang for p in plid])), "lid_conf": dict(collections.Counter(plid))}
    if lang in NAME:
        ora = run(waves, [NAME[lang]] * len(waves))
        r_["oracle_norm"] = score(ref, [o.text for o in ora], lang)
        r_["oracle_raw"] = score(ref, [o.text for o in ora], lang, normalized=False)
    res[f"{corpus}/{lang}"] = r_
    for i, r in enumerate(rs):
        hyp_rows.append({"set": f"{corpus}/{lang}", "id": r["id"], "lang": lang, "ref": ref[i], "pred": auto[i].text, "lid": plid[i],
                         "oracle": ora[i].text if lang in NAME else None})
    print(f"{corpus}/{lang} n={len(rs)} oracle CER {r_.get('oracle_norm', {}).get('cer', '-')} pred CER {r_['pred_norm']['cer']:.3f} "
          f"LID {r_['lid_acc']:.2f} {collections.Counter(plid).most_common(3)} [{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
if a.hyps:
    write_jsonl(a.hyps, hyp_rows)
