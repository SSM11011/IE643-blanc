"""Robustness eval: CER per language under clean / SNR sweep (held-out MUSAN noise+music) / reverb (held-out RIRs).
--adapter '' = vanilla model. Non-speech hallucination is measured by eval_halluc.py, not here: Whisper's default
begin_suppress_tokens forbids EOT as the first token, so an empty output is impossible with this decoding setup."""
import argparse, collections, json, random
import soundfile as sf, torch
from transformers import GenerationMixin
import augment as A
from common import SPLITS, MODELS, MAX_TEXT_TOKENS, read_jsonl, special_ids
from load_ft import load
from metrics import score

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{MODELS}/whisper-small")
ap.add_argument("--adapter", default="")
ap.add_argument("--langs", default="hi,te,as")
ap.add_argument("--n", type=int, default=60, help="utts per (corpus, lang)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
langs = a.langs.split(",")

proc, tok, model, LANGID = load(a.model, a.adapter)
SOT, TR, NO, _ = special_ids(tok)
LID = {l: LANGID[l] for l in langs}     # KeyError for 'or' with a model that has no Odia token (intended)

test = read_jsonl(f"{SPLITS}/test.jsonl")
rng = random.Random(0)
sel = {}
for lang in langs:
    for corpus in ["fleurs", "indicvoices"]:
        v = [r for r in test if r["lang"] == lang and r["corpus"] == corpus and r["ntok"] <= MAX_TEXT_TOKENS]
        sel[(corpus, lang)] = rng.sample(v, min(a.n, len(v)))


@torch.no_grad()
def transcribe(waves, lang):
    out = []
    for i in range(0, len(waves), 40):
        f = proc.feature_extractor(waves[i:i + 40], sampling_rate=16000, return_tensors="pt", device="cuda").input_features.cuda().to(torch.bfloat16)
        prompt = torch.tensor([[SOT, LID[lang], TR, NO]] * f.shape[0], device="cuda")
        g = GenerationMixin.generate(model, input_features=f, decoder_input_ids=prompt, max_new_tokens=440, do_sample=False)[:, 4:]
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out


conds = [("clean", None)] + [(f"snr{s}", s) for s in [20, 15, 10, 5, 0]] + [("reverb", "rir")]
res = collections.defaultdict(dict)
for (corpus, lang), rs in sel.items():
    clean = [sf.read(r["path"], dtype="float32")[0] for r in rs]
    ref = [r["text"] for r in rs]
    for name, c in conds:
        r_ = random.Random(1)               # same noise files/offsets for every SNR -> paired comparison
        if c is None:
            w = clean
        elif c == "rir":
            w = [A.reverb(x, r_, A.RIR_EVAL) for x in clean]
        else:
            w = [A.add_noise(x, c, r_, A.NOISE_EVAL) for x in clean]
        s = score(ref, transcribe(w, lang), lang)
        res[f"{corpus}/{lang}"][name] = {"cer": s["cer"], "wer": s["wer"]}
    print(corpus, lang, {k: round(v["cer"], 3) for k, v in res[f"{corpus}/{lang}"].items()}, flush=True)
json.dump({"cer": res, "args": vars(a)}, open(a.out, "w"), indent=1)
