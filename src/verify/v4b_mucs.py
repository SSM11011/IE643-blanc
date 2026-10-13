import sys, json, random, re, soundfile as sf
sys.path.insert(0, ".."); from metrics import score; from ic_gpu import ICGpu
M = "/home/shibam/IE643/data/processed"
rows = [json.loads(l) for l in open(f"{M}/splits/train.jsonl", encoding="utf8") if '"cs": true' in l]
ic = ICGpu(); rng = random.Random(4)
for lang in ["hi", "bn"]:
    rs = rng.sample([r for r in rows if r["lang"] == lang], 96); w = [sf.read(r["path"], dtype="float32")[0] for r in rs]
    hyp = []
    for s in range(0, 96, 32): hyp += ic(w[s:s+32], lang)       # (unsorted; fine for this check)
    lat = re.compile(r"[A-Za-z]+")
    ref_all = [r["text"] for r in rs]; ref_nolat = [lat.sub(" ", t) for t in ref_all]; hyp_nolat = [lat.sub(" ", t) for t in hyp]
    print(f"{lang}: CER all {score(ref_all, hyp, lang)['cer']:.3f} | CER with Latin tokens removed from ref {score(ref_nolat, hyp, lang)['cer']:.3f} | share of ref chars that are Latin {sum(len(''.join(lat.findall(t))) for t in ref_all)/sum(len(t) for t in ref_all):.2%}")
    pure = [i for i, t in enumerate(ref_all) if not lat.search(t)]
    print(f"   utts with NO Latin chars: {len(pure)} -> CER {score([ref_all[i] for i in pure], [hyp[i] for i in pure], lang)['cer']:.3f}")
    for i in range(3): print("   REF:", ref_all[i][:90], "\n   HYP:", hyp[i][:90])
