"""External LID baseline: SpeechBrain VoxLingua107 ECAPA-TDNN on the same sampled utterances as eval_whisper.py (same seed).
VoxLingua107 has no Odia -> Odia sets only get a confusion count. Reports open-set (argmax over 107 languages) and
closed-set (argmax over our languages it knows) accuracy.
Run with the .venv-qwen environment (needs speechbrain; torchaudio is a stub there, it is only version-checked)."""
import argparse, collections, json, random, re, time
import numpy as np, soundfile as sf, torch
from common import SPLITS, MODELS, LANGS, read_jsonl
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.models.Xvector import Classifier
from speechbrain.processing.features import InputNormalization

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()

D = f"{MODELS}/lang-id-voxlingua107-ecapa"
labels = {}
for line in open(f"{D}/label_encoder.txt", encoding="utf8"):
    m = re.match(r"'(\w+): [^']*' => (\d+)", line)
    if m:
        labels[int(m.group(2))] = m.group(1)
labels = [labels[i] for i in range(len(labels))]
known = [l for l in LANGS if l in labels]                     # 'or' is missing from VoxLingua107
known_idx = [labels.index(l) for l in known]

fbank = Fbank(n_mels=60).cuda()
norm = InputNormalization(norm_type="sentence", std_norm=False).cuda()
emb = ECAPA_TDNN(input_size=60, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                 dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=256).cuda().eval()
clf = Classifier(input_shape=[None, None, 256], activation=torch.nn.LeakyReLU, lin_blocks=1, lin_neurons=512, out_neurons=len(labels)).cuda().eval()
emb.load_state_dict(torch.load(f"{D}/embedding_model.ckpt", map_location="cuda"))
clf.load_state_dict(torch.load(f"{D}/classifier.ckpt", map_location="cuda"))


@torch.no_grad()
def classify(wav):
    x = torch.from_numpy(wav).cuda()[None]
    lens = torch.ones(1, device="cuda")
    f = norm(fbank(x), lens)
    return clf(emb(f, lens)).squeeze()          # log-posteriors over the 107 languages


groups = collections.defaultdict(list)
for r in read_jsonl(f"{SPLITS}/{a.split}.jsonl"):
    groups[(r["corpus"], r["lang"])].append(r)
rng = random.Random(0)
sel = {k: rng.sample(v, min(a.n, len(v))) for k, v in groups.items()}     # same selection as eval_whisper.py

res, t0 = {}, time.time()
for (corpus, lang), rs in sorted(sel.items()):
    open_p, closed_p = [], []
    for r in rs:
        lp = classify(sf.read(r["path"], dtype="float32")[0])
        open_p.append(labels[lp.argmax().item()])
        closed_p.append(known[lp[known_idx].argmax().item()])
    r_ = {"n": len(rs), "open_conf": dict(collections.Counter(open_p)), "closed_conf": dict(collections.Counter(closed_p))}
    if lang in known:
        r_["open_acc"] = float(np.mean([p == lang for p in open_p]))
        r_["closed_acc"] = float(np.mean([p == lang for p in closed_p]))
    res[f"{corpus}/{lang}"] = r_
    print(f"{corpus}/{lang} n={len(rs)} open {r_.get('open_acc', '-')} closed {r_.get('closed_acc', '-')} "
          f"top {collections.Counter(closed_p).most_common(3)} [{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
