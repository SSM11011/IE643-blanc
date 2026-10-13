"""Final train/dev/test roles. Speaker-disjoint; eval-text leakage removed from train; report in reports/splits.txt."""
import json, glob, random, collections, os
from common import ROOT, DATA as M
rows = []
for f in sorted(glob.glob(f"{M}/manifests/*.jsonl")):
    rows += [json.loads(l) for l in open(f, encoding="utf8")]
rng = random.Random(0)
log = []
def P(*a): s = " ".join(map(str, a)); print(s); log.append(s)

# 1. speaker overlap official train vs valid (should be 0)
for c in ["indicvoices", "kathbath"]:
    tr = {r["speaker"] for r in rows if r["corpus"] == c and r["split"] == "train"}
    va = {r["speaker"] for r in rows if r["corpus"] == c and r["split"] == "valid"}
    P(f"[speaker overlap] {c}: train {len(tr)} spk, valid {len(va)} spk, shared {len(tr & va)}")

# 2. assign roles
def spk_split(rs, frac_dev, key=lambda r: (r["corpus"], r["lang"])):
    spks = collections.defaultdict(list)
    for r in rs: spks[key(r)].append(r["speaker"])
    dev = set()
    for k, v in spks.items():
        v = sorted(set(v)); rng.shuffle(v); dev |= set(v[:max(1, int(len(v) * frac_dev))])
    return dev
role = {}
# official valid is NOT speaker-disjoint from train: valid utts of speakers seen in train -> train; dev/test only from valid-only speakers
trspk = {r["speaker"] for r in rows if r["corpus"] in ("indicvoices", "kathbath") and r["split"] == "train"}
iv_kb = [r for r in rows if r["corpus"] in ("indicvoices", "kathbath") and r["split"] == "valid" and r["speaker"] not in trspk]
devspk = spk_split(iv_kb, 0.3)
mucs_tr = [r for r in rows if r["corpus"] == "mucs" and r["split"] == "train"]
mucs_dev = spk_split(mucs_tr, 0.05)
for r in rows:
    c, s = r["corpus"], r["split"]
    if c in ("indicvoices", "kathbath"): r["role"] = "train" if (s == "train" or r["speaker"] in trspk) else ("dev" if r["speaker"] in devspk else "test")
    elif c == "shrutilipi": r["role"] = "train"
    elif c in ("fleurs", "librispeech"): r["role"] = {"train": "train", "dev": "dev", "test": "test"}[s]
    elif c == "mucs": r["role"] = "test" if s == "test" else ("dev" if r["speaker"] in mucs_dev else "train")
    else: r["role"] = "test"   # svarah, lahaja
# 3. leakage: drop train utterances whose normalized text appears in any dev/test set of the same language
ev = collections.defaultdict(set)
for r in rows:
    if r["role"] != "train": ev[r["lang"]].add(r["text_norm"])
leak = collections.Counter()
for r in rows:
    if r["role"] == "train" and r["text_norm"] in ev[r["lang"]]: r["role"] = "dropped_leak"; leak[(r["corpus"], r["lang"])] += 1
P("[leakage] train utts removed (text also in dev/test):", dict(leak) or "none")
# 4. confirm speaker disjointness inside each role pair
for c in ["indicvoices", "kathbath", "mucs"]:
    by = collections.defaultdict(set)
    for r in rows:
        if r["corpus"] == c: by[r["role"]].add(r["speaker"])
    P(f"[disjoint] {c}: train∩dev={len(by['train'] & by['dev'])} train∩test={len(by['train'] & by['test'])} dev∩test={len(by['dev'] & by['test'])}")
os.makedirs(f"{M}/splits", exist_ok=True)
for role_ in ["train", "dev", "test"]:
    with open(f"{M}/splits/{role_}.jsonl", "w", encoding="utf8") as fh:
        for r in rows:
            if r["role"] == role_: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
tab = collections.defaultdict(float)
for r in rows: tab[(r["role"], r["lang"], r["corpus"], r["cs"])] += r["dur"] / 3600
P("\nrole lang corpus cs hours")
for k in sorted(tab): P(*k, round(tab[k], 1))
open(f"{ROOT}/reports/splits.txt", "w").write("\n".join(log))
