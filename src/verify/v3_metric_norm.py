import sys, json, random, collections, unicodedata, editdistance
sys.path.insert(0, ".."); from metrics import score; from normalize import normalize
M = "/home/shibam/IE643/data/processed"
rows = [json.loads(l) for l in open(f"{M}/splits/test.jsonl", encoding="utf8")]
rng = random.Random(1)
# 1) metric cross-check: jiwer vs independent edit distance on noisy copies of real references
for lang in ["hi", "te", "as", "or", "bn", "mr", "en"]:
    rs = [r["text"] for r in rows if r["lang"] == lang][:300]
    hy = [t[:len(t)//2] + t[len(t)//2+3:] if rng.random() < .7 else t for t in rs]        # delete 3 chars in some
    hy = [" ".join(w for w in h.split() if rng.random() > .08) for h in hy]               # drop some words
    a = score(rs, hy, lang); R = [normalize(x, lang) for x in rs]; H = [normalize(x, lang) for x in hy]
    k = [i for i, x in enumerate(R) if x]
    cer = sum(editdistance.eval(R[i], H[i]) for i in k) / sum(len(R[i]) for i in k)
    wer = sum(editdistance.eval(R[i].split(), H[i].split()) for i in k) / sum(len(R[i].split()) for i in k)
    print(f"{lang}: jiwer CER {a['cer']:.4f} vs indep {cer:.4f} | WER {a['wer']:.4f} vs {wer:.4f} | {'OK' if abs(a['cer']-cer)<1e-9 and abs(a['wer']-wer)<1e-9 else 'MISMATCH'}")
# 2) normalizer: idempotence + what gets deleted
rm = collections.defaultdict(collections.Counter); loss = collections.defaultdict(list); nonidem = 0
for r in rows:
    t, l = r["text"], r["lang"]; n = normalize(t, l)
    if normalize(n, l) != n: nonidem += 1
    loss[l].append(1 - len(n.replace(" ", "")) / max(1, len(t.replace(" ", ""))))
    kept = collections.Counter(n)
    for ch, c in collections.Counter(t.lower()).items():
        if ch.strip() and c > kept.get(ch, 0) and ch not in n: rm[l][(ch, unicodedata.category(ch), unicodedata.name(ch, "?"))] += c
print("non-idempotent normalizations:", nonidem)
for l in ["hi", "te", "as", "or", "bn", "mr", "en"]:
    print(f"{l}: mean non-space chars removed {sum(loss[l])/len(loss[l]):.3%}; top removed:", [(c[0], c[1], n) for c, n in rm[l].most_common(6)])
