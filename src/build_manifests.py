"""Decode every corpus -> 16 kHz mono FLAC + JSONL manifest (id, corpus, lang, split, path, dur, text, text_norm, speaker, cs).
Filters: dur >30 s or <0.3 s, empty normalized text. LibriSpeech is referenced in place (already 16 kHz FLAC)."""
import glob, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
from common import ROOT, DATA as OUT
from normalize import normalize
R = f"{ROOT}/data"
NAME = {"as":"assamese","bn":"bengali","hi":"hindi","mr":"marathi","or":"odia","te":"telugu"}
def to16k(x, sr):
    if x.ndim > 1: x = x.mean(1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g)
    return x.astype(np.float32)
def rec(id_, corpus, lang, split, path, dur, text, spk, cs=False):
    tn = normalize(text, lang)
    if not tn or dur > 30 or dur < 0.3: return None
    return dict(id=id_, corpus=corpus, lang=lang, split=split, path=path, dur=round(dur, 3), text=text, text_norm=tn, speaker=str(spk), cs=cs)
def save(x, corpus, lang, id_):
    d = f"{OUT}/audio/{corpus}/{lang}"; os.makedirs(d, exist_ok=True)
    p = f"{d}/{id_}.flac"; sf.write(p, x, 16000, format="FLAC"); return p

def do_parquet(a):
    f, corpus, lang = a
    split = os.path.basename(f).split("-")[0] if corpus != "svarah" else "test"
    base = os.path.basename(f).replace(".parquet", "")
    pf = pq.ParquetFile(f); out = []; n = 0; dropped = 0
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg).to_pylist():
            a_ = r["audio_filepath"]; n += 1
            try: x, sr = sf.read(io.BytesIO(a_["bytes"]), dtype="float32")
            except Exception: dropped += 1; continue
            x = to16k(x, sr); dur = len(x) / 16000
            l = lang if lang != "?" else r.get("lang", "hi")
            id_ = f"{corpus}_{l}_{base}_{n}"
            spk = r.get("speaker_id", r.get("sp_id", f"{corpus}_{base}")) if corpus in ("indicvoices", "kathbath", "lahaja") else None
            if spk is None: spk = f"{corpus}_{base}"   # shrutilipi/svarah: no speaker ids -> shard id (never used for splitting)
            m = rec(id_, corpus, l, split, None, dur, r["text"], f"{corpus}:{spk}")
            if m is None: dropped += 1; continue
            m["path"] = save(x, corpus, l, id_); out.append(m)
    return out, dropped

def fleurs():
    out = []; MAP = {"en":"en_us","hi":"hi_in","mr":"mr_in","bn":"bn_in","as":"as_in","or":"or_in","te":"te_in"}
    for l, d in MAP.items():
        for s in ["train", "dev", "test"]:
            for line in open(f"{R}/fleurs/data/{d}/{s}.tsv", encoding="utf8"):
                p = line.rstrip("\n").split("\t")
                path = f"{R}/fleurs/data/{d}/audio/{s}/{p[1]}"
                m = rec(f"fleurs_{l}_{s}_{p[1][:-4]}", "fleurs", l, s, path, int(p[5]) / 16000, p[2], f"fleurs:{l}:{s}")
                if m: out.append(m)
    return out

def do_mucs(a):
    lg, split = a; l = {"Hindi-English":"hi","Bengali-English":"bn"}[lg]
    base = f"{R}/mucs/{lg}/{split}"; T = f"{base}/transcripts"
    seg = {}
    for ln in open(f"{T}/segments"):
        u, rid, s, e = ln.split(); seg[u] = (rid, float(s), float(e))
    wavs = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{base}/**/*.wav", recursive=True)}
    out = []; cache = {}
    for ln in open(f"{T}/text", encoding="utf8"):
        u, _, tx = ln.rstrip("\n").partition(" ")
        rid, s, e = seg[u]
        if rid not in wavs or e - s > 30 or e - s < 0.3: continue
        if rid not in cache: cache = {rid: sf.read(wavs[rid], dtype="float32")}
        x, sr = cache[rid]; x = to16k(x[int(s * sr):int(e * sr)], sr)
        id_ = f"mucs_{lg}_{u}"
        m = rec(id_, "mucs", l, split, None, len(x) / 16000, tx, f"mucs:{lg}:{rid}", cs=True)
        if m: m["path"] = save(x, "mucs-" + l, l, id_); out.append(m)
    return out, 0

def libri():
    out = []
    for s, d in [("train","train-clean-100"),("dev","dev-clean"),("test","test-clean")]:
        for t in glob.glob(f"{R}/librispeech/LibriSpeech/{d}/*/*/*.trans.txt"):
            for ln in open(t):
                u, _, tx = ln.strip().partition(" "); spk = u.split("-")[0]
                p = os.path.join(os.path.dirname(t), u + ".flac")
                dur = sf.info(p).duration
                m = rec(f"libri_{u}", "librispeech", "en", s, p, dur, tx, f"libri:{spk}")
                if m: out.append(m)
    return out

def write(name, rows):
    os.makedirs(f"{OUT}/manifests", exist_ok=True)
    with open(f"{OUT}/manifests/{name}.jsonl", "w", encoding="utf8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = sum(r["dur"] for r in rows) / 3600
    print(f"{name}: {len(rows)} utts, {h:.1f} h", flush=True)

if __name__ == "__main__":
    only = set(sys.argv[1:])
    with ProcessPoolExecutor(12) as ex:
        for corpus in ["indicvoices", "kathbath", "shrutilipi"]:
            if only and corpus not in only: continue
            jobs = [(f, corpus, c) for c, n in NAME.items() for f in sorted(glob.glob(f"{R}/{corpus}/{n}/*.parquet"))]
            rows = []; dr = 0
            for o, d in ex.map(do_parquet, jobs): rows += o; dr += d
            print(corpus, "dropped", dr); write(corpus, rows)
        for corpus, glob_ in [("svarah", f"{R}/svarah/data/*.parquet"), ("lahaja", f"{R}/lahaja/data/*.parquet")]:
            if only and corpus not in only: continue
            jobs = [(f, corpus, "en" if corpus == "svarah" else "?") for f in sorted(glob.glob(glob_))]
            rows = []
            for o, d in ex.map(do_parquet, jobs): rows += o
            write(corpus, rows)
        if not only or "mucs" in only:
            rows = []
            for o, _ in ex.map(do_mucs, [(lg, s) for lg in ["Hindi-English", "Bengali-English"] for s in ["train", "test"]]): rows += o
            write("mucs", rows)
    if not only or "fleurs" in only: write("fleurs", fleurs())
    if not only or "librispeech" in only: write("librispeech", libri())
