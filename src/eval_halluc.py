"""Non-speech hallucination + false-silence evaluation, with begin_suppress_tokens DISABLED (Whisper's default forbids EOT as the
first token, which makes an empty transcript impossible and invalidates any hallucination measurement).
 hallucination rate = fraction of noise/music-only clips (held-out MUSAN files) with >5 output tokens
 false-silence rate = fraction of real-speech utterances (FLEURS + IndicVoices test) whose output is EMPTY"""
import argparse, json, random
import numpy as np, soundfile as sf, torch
from transformers import GenerationMixin
import augment as A
from common import SPLITS, MODELS, read_jsonl, special_ids, noise_clip
from load_ft import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{MODELS}/whisper-small")
ap.add_argument("--adapter", default="")
ap.add_argument("--langs", default="hi,te,as")
ap.add_argument("--n", type=int, default=100, help="clips / utterances per language")
ap.add_argument("--out", required=True)
a = ap.parse_args()
langs = a.langs.split(",")

proc, tok, m, LANGID = load(a.model, a.adapter)
m.generation_config.begin_suppress_tokens = None
SOT, TR, NO, _ = special_ids(tok)


@torch.no_grad()
def n_tokens(waves, lang):
    out = []
    for i in range(0, len(waves), 50):
        f = proc.feature_extractor(waves[i:i + 50], sampling_rate=16000, return_tensors="pt", device="cuda").input_features.cuda().to(torch.bfloat16)
        p = torch.tensor([[SOT, LANGID[lang], TR, NO]] * f.shape[0], device="cuda")
        g = GenerationMixin.generate(m, input_features=f, decoder_input_ids=p, max_new_tokens=200, do_sample=False)[:, 4:]
        out += [len(tok(o, add_special_tokens=False).input_ids) for o in tok.batch_decode(g, skip_special_tokens=True)]
    return np.array(out)


test = read_jsonl(f"{SPLITS}/test.jsonl")
res = {"hallucination": {}, "false_silence": {}, "mean_tokens_on_noise": {}}
for lang in langs:
    r = random.Random(11)
    n = n_tokens([noise_clip(r, A.NOISE_EVAL, A.read_mono) for _ in range(a.n)], lang)
    res["hallucination"][lang] = float((n > 5).mean())
    res["mean_tokens_on_noise"][lang] = float(n.mean())
    speech = [x for x in test if x["lang"] == lang and x["corpus"] in ("fleurs", "indicvoices") and x["ntok"] <= 200]
    random.Random(5).shuffle(speech)
    waves = [sf.read(x["path"], dtype="float32")[0] for x in speech[:a.n]]
    res["false_silence"][lang] = float((n_tokens(waves, lang) == 0).mean())
print(a.adapter or "vanilla", json.dumps(res))
json.dump({**res, "args": vars(a)}, open(a.out, "w"), indent=1)
