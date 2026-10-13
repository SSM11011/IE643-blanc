"""Evaluate a (vanilla or fine-tuned) Whisper on the dev/test manifests. Per (corpus, lang):
 - WER/CER (normalized, plus raw WER) with the ORACLE language token and with the PREDICTED language
   (closed-set argmax of the first decoder step over the supported languages);
 - LID accuracy + confusion of that decoder prediction, and of the encoder LID-CTC head if the run has one (runs/<name>/lid_head.pt).
Odia: vanilla Whisper has no token -> Odia is excluded from the LID set and decoded without a forced language. With an adapter
trained with <|or|>, Odia joins the LID set and is decoded with a manual [SOT, <|or|>, transcribe, notimestamps] prompt."""
import argparse, collections, json, os, random, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf, torch
from transformers import GenerationMixin
from common import SPLITS, MODELS, read_jsonl, write_jsonl, special_ids, enc_frames, head_lid, load_lid_head
from load_ft import load as load_model
from metrics import score

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{MODELS}/whisper-small")
ap.add_argument("--adapter", default="")
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=200, help="utts per (corpus, lang)")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--out", required=True)
ap.add_argument("--sets", default="", help="comma-separated corpus/lang subset, e.g. svarah/en,fleurs/en (default: all)")
ap.add_argument("--en_ascii", action="store_true", help="when decoding with <|en|>, suppress every token that is not pure ASCII")
ap.add_argument("--lid_rule", default="", help="lid_rule.json from lid_calib.py: predicted LID = argmax(token logp + bias + w * head logp)")
ap.add_argument("--hyps", default="", help="optional JSONL: one row per utterance (ref, oracle/pred hypotheses, LID)")
a = ap.parse_args()

proc, tok, model, LANGID = load_model(a.model, a.adapter)
LANGS = list(LANGID)                       # 'or' only if the model has the token
lid_ids = [LANGID[l] for l in LANGS]
SOT, TR, NO, _ = special_ids(tok)

head = None
run_dir = os.path.dirname(os.path.normpath(a.adapter)) if a.adapter else ""
if run_dir and os.path.exists(f"{run_dir}/lid_head.pt"):
    run_langs = json.load(open(f"{run_dir}/result.json"))["args"]["langs"].split(",")
    head, head_langs = load_lid_head(f"{run_dir}/lid_head.pt", run_langs)
    print("LID head:", head_langs, flush=True)
RULE = json.load(open(a.lid_rule)) if a.lid_rule else None
if RULE:
    assert head is not None or not RULE["head_w"], "--lid_rule with a head weight needs the run's lid_head.pt"
    print("LID rule:", RULE["bias"], "head_w", RULE["head_w"], flush=True)

groups = collections.defaultdict(list)
for r in read_jsonl(f"{SPLITS}/{a.split}.jsonl"):
    groups[(r["corpus"], r["lang"])].append(r)
rng = random.Random(0)
sel = {k: rng.sample(v, min(a.n, len(v))) for k, v in groups.items()}     # same selection as eval_indicconformer.py
if a.sets:
    sel = {k: v for k, v in sel.items() if f"{k[0]}/{k[1]}" in a.sets.split(",")}
EN_SUPPRESS = None
if a.en_ascii:
    EN_SUPPRESS = sorted(set(model.generation_config.suppress_tokens or []) |
                         {i for i in range(len(tok)) if not tok.decode([i]).isascii()})
    print(f"--en_ascii: {len(EN_SUPPRESS)} tokens suppressed for English", flush=True)

pool = ThreadPoolExecutor(8)


def read_batch(batch):
    return list(pool.map(lambda r: sf.read(r["path"], dtype="float32")[0], batch))


def features(waves):
    return proc.feature_extractor(waves, sampling_rate=16000, return_tensors="pt", device="cuda").input_features.to("cuda", torch.bfloat16)


@torch.no_grad()
def predict_lid(f, n_frames):
    """decoder-token LID (or the calibrated rule, --lid_rule) and the encoder output (for the LID head)"""
    out = model(input_features=f, decoder_input_ids=torch.full((f.shape[0], 1), SOT, device="cuda"))
    s = out.logits[:, -1, lid_ids].float()
    if RULE:
        s = s.log_softmax(-1) + torch.tensor([RULE["bias"].get(l, 0.0) for l in LANGS], device="cuda")
        if RULE["head_w"]:
            p = head(out.encoder_last_hidden_state.float()).softmax(-1)[:, :, 1:]
            valid = torch.arange(p.shape[1], device="cuda")[None] < torch.tensor(n_frames, device="cuda")[:, None]
            m = (p * valid.unsqueeze(-1)).sum(1)
            s = s + RULE["head_w"] * (m / m.sum(-1, keepdim=True)).clamp_min(1e-6).log()[:, [head_langs.index(l) for l in LANGS]]
    return [LANGS[i] for i in s.argmax(-1).tolist()], out.encoder_last_hidden_state


@torch.no_grad()
def decode(f, langs):
    """langs[i] = language token for sample i (None = let Whisper detect)"""
    out = []
    for l in set(langs):
        idx = [i for i, x in enumerate(langs) if x == l]
        if l == "or":   # token unknown to Whisper's generate(language=...) -> manual prompt (same as training)
            p = torch.tensor([[SOT, LANGID["or"], TR, NO]] * len(idx), device="cuda")
            ids = GenerationMixin.generate(model, input_features=f[idx], decoder_input_ids=p, max_new_tokens=440, do_sample=False)[:, 4:]
        else:
            kw = dict(task="transcribe", max_new_tokens=440)   # decoder limit 448 incl. prompt
            if l:
                kw["language"] = l
            if l == "en" and EN_SUPPRESS:
                kw["suppress_tokens"] = EN_SUPPRESS
            ids = model.generate(input_features=f[idx], **kw)
        out += zip(idx, proc.batch_decode(ids, skip_special_tokens=True))
    return [t for _, t in sorted(out)]


res, t0, hyp_rows = {}, time.time(), []
for (corpus, lang), rs in sorted(sel.items()):
    ref, ora, pre, plid, hlid = [], [], [], [], []
    chunks = [rs[i:i + a.bs] for i in range(0, len(rs), a.bs)]
    nxt = pool.submit(read_batch, chunks[0])
    for ci, b in enumerate(chunks):
        aud = nxt.result()
        if ci + 1 < len(chunks):
            nxt = pool.submit(read_batch, chunks[ci + 1])   # prefetch the next batch while the GPU decodes
        f = features(aud)
        ref += [r["text"] for r in b]
        olang = [lang if lang in LANGS else None] * len(b)   # oracle (None for Odia without the token)
        o = decode(f, olang)
        ora += o
        pl, enc_h = predict_lid(f, [enc_frames(len(x)) for x in aud])
        plid += pl
        if head is not None:
            hlid += head_lid(head, enc_h, [enc_frames(len(x)) for x in aud], head_langs)
        diff = [i for i in range(len(b)) if pl[i] != olang[i]]
        redo = dict(zip(diff, decode(f[diff], [pl[i] for i in diff]))) if diff else {}   # same token -> same decode, reuse
        pre += [redo.get(i, o[i]) for i in range(len(b))]
    r_ = {"n": len(rs), "oracle_norm": score(ref, ora, lang), "oracle_raw": score(ref, ora, lang, normalized=False), "pred_norm": score(ref, pre, lang)}
    if lang in LANGS:
        r_["lid_acc"] = float(np.mean([p == lang for p in plid]))
        r_["lid_conf"] = dict(collections.Counter(plid))
    if head is not None and lang in head_langs:
        r_["lid_head_acc"] = float(np.mean([p == lang for p in hlid]))
        r_["lid_head_conf"] = dict(collections.Counter(hlid))
    res[f"{corpus}/{lang}"] = r_
    for i, r in enumerate(rs):
        hyp_rows.append({"set": f"{corpus}/{lang}", "id": r["id"], "lang": lang, "ref": ref[i], "oracle": ora[i], "pred": pre[i],
                         "lid": plid[i], "lid_head": hlid[i] if hlid else None})
    print(f"{corpus}/{lang} n={len(rs)} oracle WER/CER {r_['oracle_norm']['wer']:.3f}/{r_['oracle_norm']['cer']:.3f}  "
          f"raw WER {r_['oracle_raw']['wer']:.3f}  pred CER {r_['pred_norm']['cer']:.3f}  LID {r_.get('lid_acc', '-')}  "
          f"LIDhead {r_.get('lid_head_acc', '-')}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
if a.hyps:
    write_jsonl(a.hyps, hyp_rows)
