"""Per-language / per-corpus / per-split audit (hours, utts, speakers, >30s, empty text). Reads metadata only."""
import glob, os, collections, csv
import pyarrow.parquet as pq
from common import ROOT as _ROOT
ROOT = f"{_ROOT}/data"
LANGS = {"as":"assamese","bn":"bengali","hi":"hindi","mr":"marathi","or":"odia","te":"telugu"}
rows = collections.defaultdict(lambda: dict(utts=0, sec=0.0, long=0, empty=0, spk=set()))
def add(corpus, lang, split, dur, text, spk=None):
    r = rows[(corpus, lang, split)]
    r["utts"] += 1; r["sec"] += dur
    r["long"] += dur > 30; r["empty"] += not (text or "").strip()
    if spk is not None: r["spk"].add(str(spk))
for corpus, spkcol in [("indicvoices","speaker_id"),("kathbath","speaker_id"),("shrutilipi",None)]:
    for code, name in LANGS.items():
        for f in sorted(glob.glob(f"{ROOT}/{corpus}/{name}/*.parquet")):
            split = os.path.basename(f).split("-")[0]
            cols = ["duration","text"] + ([spkcol] if spkcol else [])
            t = pq.read_table(f, columns=cols).to_pydict()
            for i in range(len(t["duration"])):
                add(corpus, code, split, t["duration"][i], t["text"][i], t[spkcol][i] if spkcol else None)
for corpus in ["svarah","lahaja"]:
    for f in sorted(glob.glob(f"{ROOT}/{corpus}/data/*.parquet")):
        cols = ["duration","text"] + (["lang"] if corpus=="lahaja" else [])
        t = pq.read_table(f, columns=cols).to_pydict()
        for i in range(len(t["duration"])):
            add(corpus, t["lang"][i] if corpus=="lahaja" else "en", "test", t["duration"][i], t["text"][i])
FL = {"en":"en_us","hi":"hi_in","mr":"mr_in","bn":"bn_in","as":"as_in","or":"or_in","te":"te_in"}
for code, d in FL.items():
    for split in ["train","dev","test"]:
        for line in open(f"{ROOT}/fleurs/data/{d}/{split}.tsv", encoding="utf8"):
            p = line.rstrip("\n").split("\t")
            add("fleurs", code, split, int(p[5])/16000, p[3])   # FLEURS has no speaker ids
for lg in ["Hindi-English","Bengali-English"]:
    for split in ["train","test"]:
        base = f"{ROOT}/mucs/{lg}/{split}/transcripts"
        seg = {l.split()[0]: float(l.split()[3])-float(l.split()[2]) for l in open(f"{base}/segments")}
        for l in open(f"{base}/text", encoding="utf8"):
            u, _, tx = l.partition(" ")
            add("mucs", lg, split, seg.get(u, 0.0), tx, u.split("_")[0])
out = f"{_ROOT}/reports/data_audit.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["corpus","lang","split","utts","hours","speakers","n_over_30s","n_empty_text"])
    for k in sorted(rows):
        r = rows[k]; w.writerow([*k, r["utts"], round(r["sec"]/3600,2), len(r["spk"]) or "", r["long"], r["empty"]])
print(open(out).read())
