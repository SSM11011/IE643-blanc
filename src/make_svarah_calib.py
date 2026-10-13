"""Speaker-disjoint Svarah calibration set for the LID decision rule (lid_calib.py).
Svarah has no speaker ids; a speaker is approximated by the demographic profile in the parquet metadata (115 distinct
profiles, ~117 speakers in the corpus). Every profile that occurs among the 100 scored test utterances (eval_whisper.py,
seed 0) is excluded; up to --n utterances are sampled from the rest -> data/processed/splits/svarah_calib.jsonl."""
import argparse, collections, glob, os, random
import pyarrow.parquet as pq
from common import ROOT, SPLITS, read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

PROFILE = ["gender", "age-group", "primary_language", "native_place_state", "native_place_district", "highest_qualification",
           "job_category", "occupation_domain"]
prof = {}
for f in sorted(glob.glob(f"{ROOT}/data/svarah/data/*.parquet")):
    d = pq.read_table(f, columns=PROFILE).to_pydict()
    base = "svarah_" + os.path.basename(f).replace(".parquet", "")
    for i, p in enumerate(zip(*[d[c] for c in PROFILE])):
        prof[f"svarah/en/{base}_{i + 1}"] = p          # manifest id suffix = 1-based row index in the shard

groups = collections.defaultdict(list)
for r in read_jsonl(f"{SPLITS}/test.jsonl"):
    groups[(r["corpus"], r["lang"])].append(r)
rng = random.Random(0)
sel = {k: rng.sample(v, min(100, len(v))) for k, v in groups.items()}     # the scored selection (same as eval_whisper.py)
used = {prof[r["id"]] for r in sel[("svarah", "en")]}
pool = [r for r in groups[("svarah", "en")] if prof[r["id"]] not in used]
calib = random.Random(1).sample(pool, min(a.n, len(pool)))
for r in calib:
    r["corpus"] = "svarahcalib"
write_jsonl(f"{SPLITS}/svarah_calib.jsonl", calib)
print(f"profiles: {len(set(prof.values()))} total, {len(used)} in the scored 100, {len({prof[r['id']] for r in pool})} left; "
      f"pool {len(pool)} utts, calib {len(calib)} utts from {len({prof[r['id']] for r in calib})} profiles")
