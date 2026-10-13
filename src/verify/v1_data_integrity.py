"""Independent integrity checks on manifests + splits (re-derived from the final files, not from the builder's state)."""
import json, glob, os, collections, soundfile as sf, random
M = "/home/shibam/IE643/data/processed"
man = [json.loads(l) for f in sorted(glob.glob(f"{M}/manifests/*.jsonl")) for l in open(f, encoding="utf8")]
print("manifest utts:", len(man))
ids = collections.Counter(r["id"] for r in man); print("duplicate ids:", sum(1 for v in ids.values() if v > 1))
miss = [r["path"] for r in man if not os.path.exists(r["path"])]; print("missing audio files:", len(miss))
# durations / sample rate on a random sample of 3000 files
bad = collections.Counter(); rng = random.Random(0)
for r in rng.sample(man, 3000):
    i = sf.info(r["path"])
    if i.samplerate != 16000: bad["sr!=16k"] += 1
    if i.channels != 1: bad["not mono"] += 1
    if abs(i.duration - r["dur"]) > 0.05: bad["dur mismatch"] += 1
print("sampled-file problems (of 3000):", dict(bad) or "none")
print("dur range (s):", min(r["dur"] for r in man), max(r["dur"] for r in man))
print("empty text_norm:", sum(1 for r in man if not r["text_norm"]))
# splits re-derived
sp = {k: [json.loads(l) for l in open(f"{M}/splits/{k}.jsonl", encoding="utf8")] for k in ["train", "dev", "test"]}
print({k: len(v) for k, v in sp.items()})
allid = collections.Counter(r["id"] for v in sp.values() for r in v); print("ids in >1 split file:", sum(1 for v in allid.values() if v > 1))
for c in ["indicvoices", "kathbath", "mucs"]:
    s = {k: {r["speaker"] for r in v if r["corpus"] == c} for k, v in sp.items()}
    print(f"{c}: train&dev {len(s['train']&s['dev'])}  train&test {len(s['train']&s['test'])}  dev&test {len(s['dev']&s['test'])}")
for lang in ["hi", "en", "te", "as", "or", "bn", "mr"]:
    ev = {r["text_norm"] for k in ["dev", "test"] for r in sp[k] if r["lang"] == lang}
    n = sum(1 for r in sp["train"] if r["lang"] == lang and r["text_norm"] in ev); print(f"train utts whose text is in dev/test [{lang}]: {n}")
# every corpus/lang accounted for in splits
cnt = collections.Counter((r["corpus"], r["lang"]) for v in sp.values() for r in v)
print("corpus/lang combos in splits:", len(cnt))
