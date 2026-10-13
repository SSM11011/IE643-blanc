"""E3 manifests: same underlying pool, unbalanced (natural skew hi 100% / te 30% / as 6%) vs two-stage-balanced (equal hours per language via upsampling)."""
import json, random, collections
from balanced_mix import take
from common import SPLITS as M
rng = random.Random(0)
tr = [u for u in map(json.loads, open(f"{M}/train.jsonl", encoding="utf8")) if u["ntok"] <= 440 and not u["cs"] and u["lang"] in ("hi", "te", "as")]
frac = {"hi": 1.0, "te": 0.3, "as": 0.06}; pool = {}
for l, f in frac.items():
    v = [u for u in tr if u["lang"] == l]; rng.shuffle(v); pool[l] = v[:int(len(v) * f)]
hrs = {l: sum(u["dur"] for u in v) / 3600 for l, v in pool.items()}; print("pool hours", {k: round(v, 1) for k, v in hrs.items()})
unb = [u for v in pool.values() for u in v]; rng.shuffle(unb)
T = max(hrs.values()); bal = []
for l, v in pool.items(): bal += take(v, T, rng)
rng.shuffle(bal)
for name, rows in [("e3_unbalanced", unb), ("e3_balanced", bal)]:
    with open(f"{M}/{name}.jsonl", "w", encoding="utf8") as fh:
        for u in rows: fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    c = collections.Counter(); [c.update({u["lang"]: u["dur"] / 3600}) for u in rows]; print(name, len(rows), {k: round(v, 1) for k, v in c.items()})
