"""Calibrate the utterance-level LID decision of a fine-tuned run WITHOUT retraining.
score(l) = log p_token(l) + bias[l] + w * log p_head(l)       (p_token: first decoder step, closed set; p_head: LID-CTC head mass)
Fitted on calibration data only: the dev split (100 utts/set) + the speaker-disjoint Svarah set (make_svarah_calib.py).
Objective: macro-average LID accuracy over the calibration sets; grid over w and a single English bias (Indian-accented English
is the only failure the decoder-token rule shows on dev/calibration data). Writes runs/<name>/lid_rule.json, used by
eval_whisper.py --lid_rule. Also reports the rule on the scored test selection (for information; never used for fitting)."""
import argparse, collections, json, os, random
import numpy as np, soundfile as sf, torch
from common import SPLITS, MODELS, read_jsonl, special_ids, enc_frames, load_lid_head
from load_ft import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{MODELS}/whisper-large-v3-turbo")
ap.add_argument("--adapter", default="../runs/s2_turbo/adapter")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()

proc, tok, model, LANGID = load(a.model, a.adapter)
LANGS = list(LANGID)
SOT = special_ids(tok)[0]
run_dir = os.path.dirname(os.path.normpath(a.adapter))
head, head_langs = load_lid_head(f"{run_dir}/lid_head.pt", json.load(open(f"{run_dir}/result.json"))["args"]["langs"].split(","))
hperm = [head_langs.index(l) for l in LANGS]               # head class order -> LANGS order


def select(path, n):
    groups = collections.defaultdict(list)
    for r in read_jsonl(path):
        groups[(r["corpus"], r["lang"])].append(r)
    rng = random.Random(0)
    return {f"{c}/{l}": rng.sample(v, min(n, len(v))) for (c, l), v in sorted(groups.items())}


@torch.no_grad()
def scores(rows):
    tk, hd = [], []
    for i in range(0, len(rows), a.bs):
        wav = [sf.read(r["path"], dtype="float32")[0] for r in rows[i:i + a.bs]]
        f = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
        out = model(input_features=f, decoder_input_ids=torch.full((len(wav), 1), SOT, device="cuda"))
        tk.append(out.logits[:, -1, [LANGID[l] for l in LANGS]].float().log_softmax(-1).cpu())
        p = head(out.encoder_last_hidden_state.float()).softmax(-1)[:, :, 1:]
        valid = torch.arange(p.shape[1], device="cuda")[None] < torch.tensor([enc_frames(len(w)) for w in wav], device="cuda")[:, None]
        m = (p * valid.unsqueeze(-1)).sum(1)
        hd.append((m / m.sum(-1, keepdim=True)).clamp_min(1e-6).log()[:, hperm].cpu())
    return torch.cat(tk).numpy(), torch.cat(hd).numpy()


def collect(sel):
    return {k: (*scores(rs), np.array([LANGS.index(r["lang"]) for r in rs])) for k, rs in sel.items()}


calib = collect({**select(f"{SPLITS}/dev.jsonl", a.n), **select(f"{SPLITS}/svarah_calib.jsonl", 10 ** 6)})
test = collect(select(f"{SPLITS}/test.jsonl", a.n))
EN = LANGS.index("en")


def acc(data, b, w):
    out = {}
    for k, (t, h, y) in data.items():
        s = t + w * h
        s[:, EN] += b
        out[k] = float((s.argmax(1) == y).mean())
    return out


grid = [(b, w) for w in [0, 0.25, 0.5, 1, 2] for b in np.arange(0, 30.01, 0.25)]
res = [(np.mean(list(acc(calib, b, w).values())), -abs(b) - w, b, w) for b, w in grid]   # ties -> smallest change
best = max(res)
b, w = float(best[2]), float(best[3])
base_c, best_c, base_t, best_t = acc(calib, 0, 0), acc(calib, b, w), acc(test, 0, 0), acc(test, b, w)
print(f"chosen: en bias {b}, head weight {w}; calibration macro LID {np.mean(list(base_c.values())):.3f} -> {best[0]:.3f}")
print("| set | LID before | LID after |\n|---|---|---|")
for name, x0, x1 in [("calib", base_c, best_c), ("test", base_t, best_t)]:
    for k in x0:
        print(f"| {name}: {k} | {x0[k]:.2f} | {x1[k]:.2f} |")
json.dump({"bias": {"en": b}, "head_w": w, "calib_macro_before": float(np.mean(list(base_c.values()))), "calib_macro_after": best[0],
           "calib": best_c, "test_info": best_t, "test_info_before": base_t}, open(f"{run_dir}/lid_rule.json", "w"), indent=1)
