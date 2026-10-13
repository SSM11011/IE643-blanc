"""Does max_new_tokens=200 truncate Indic outputs? Does bf16 vs fp32 change CER? Token-length stats of REFERENCES."""
import sys, json, random, collections, numpy as np, soundfile as sf, torch
sys.path.insert(0, ".."); from metrics import score
from transformers import WhisperForConditionalGeneration, WhisperProcessor
M = "/home/shibam/IE643/data/processed"; P = "/home/shibam/IE643/models/whisper-large-v3-turbo"
proc = WhisperProcessor.from_pretrained(P)
rows = [json.loads(l) for l in open(f"{M}/splits/test.jsonl", encoding="utf8")]
rng = random.Random(5); pick = []
for lang in ["hi", "en", "te", "as", "bn", "mr"]:
    for corpus in ["fleurs", "indicvoices"] if lang != "en" else ["fleurs", "librispeech"]:
        v = [r for r in rows if r["lang"] == lang and r["corpus"] == corpus]; pick += rng.sample(v, min(32, len(v)))
# reference token lengths (does a 30s-or-less utterance fit in 200 tokens?)
L = collections.defaultdict(list)
for r in pick: L[r["lang"]].append(len(proc.tokenizer(r["text"]).input_ids))
print("reference token counts (mean / max / frac>200):", {k: (int(np.mean(v)), max(v), round(float(np.mean(np.array(v) > 200)), 2)) for k, v in L.items()})
def run(dtype, mnt):
    m = WhisperForConditionalGeneration.from_pretrained(P, torch_dtype=dtype).cuda().eval(); m.generation_config.max_length = None
    res = collections.defaultdict(lambda: ([], [], 0))
    for lang in ["hi", "en", "te", "as", "bn", "mr"]:
        rs = [r for r in pick if r["lang"] == lang]; ref, hyp, hit = [], [], 0
        for i in range(0, len(rs), 32):
            b = rs[i:i + 32]; aud = [sf.read(r["path"], dtype="float32")[0] for r in b]
            f = proc.feature_extractor(aud, sampling_rate=16000, return_tensors="pt", device="cuda").input_features.to("cuda", dtype)
            with torch.no_grad(): ids = m.generate(input_features=f, language=lang, task="transcribe", max_new_tokens=mnt)
            hit += sum(int(((x != proc.tokenizer.pad_token_id) & (x != proc.tokenizer.eos_token_id)).sum()) >= mnt for x in ids)   # outputs that hit the cap
            hyp += proc.batch_decode(ids, skip_special_tokens=True); ref += [r["text"] for r in b]
        res[lang] = (score(ref, hyp, lang)["cer"], hit, len(rs))
    del m; torch.cuda.empty_cache(); return res
out = {}
for name, dt, mnt in [("bf16/200", torch.bfloat16, 200), ("bf16/440", torch.bfloat16, 440), ("fp32/440", torch.float32, 440)]:
    out[name] = run(dt, mnt); print(name, {k: f"CER {v[0]:.3f} hit-limit {v[1]}/{v[2]}" for k, v in out[name].items()}, flush=True)
