"""Fine-tune Whisper (full / LoRA / QLoRA) on the balanced mix, optionally with an encoder LID-CTC head.

Prompt and labels are built manually; the language token is a supervised target:
    decoder_in = [SOT, lang, transcribe, notimestamps, text...]
    labels     = [lang, transcribe, notimestamps, text..., EOT]
Odia has no Whisper language token: <|or|> is added with id = old vocab size and initialised from <|bn|>. The input embedding is
tied to the output projection, so a gradient mask makes the <|or|> row the only trainable row of both.
Randomness (sampling order, augmentation, non-speech clips, SpecAugment) is seeded and checkpointed, so --resume continues
the same data order."""
import argparse, collections, json, os, random, time
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationMixin
import augment as A
from common import ROOT, SPLITS, MODELS, MAX_TEXT_TOKENS, read_jsonl, special_ids, enc_frames, noise_clip, head_lid, save_lid_head
from metrics import score

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{MODELS}/whisper-small")
ap.add_argument("--name", required=True)
ap.add_argument("--mode", choices=["full", "lora", "qlora"], default="lora")
ap.add_argument("--langs", default="hi")
ap.add_argument("--cs", action="store_true", help="include code-switch (MUCS) utts")
ap.add_argument("--manifest", default=f"{SPLITS}/train_mix_epoch0.jsonl")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=None, help="default: 1e-5 full, 5e-4 (Q)LoRA")
ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps")
ap.add_argument("--r", type=int, default=32)
ap.add_argument("--alpha", type=int, default=64)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--eval_n", type=int, default=96, help="dev utts per (corpus, lang)")
ap.add_argument("--eval_start", action="store_true")
ap.add_argument("--freeze_enc_frac", type=float, default=0.0, help="freeze bottom fraction of encoder layers")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_compile", action="store_true")
ap.add_argument("--lid_ctc", type=float, default=0.0, help="weight of encoder LID-CTC aux loss (0=off)")
ap.add_argument("--lid_detach", action="store_true", help="detach encoder feats before the LID head")
ap.add_argument("--aug", action="store_true", help="MUSAN/RIR/speed + SpecAugment")
ap.add_argument("--nonspeech", type=float, default=0.0, help="fraction of samples replaced by noise-only clips with EMPTY target (anti-hallucination, Calm-Whisper/Canary recipe)")
ap.add_argument("--ns_weight", type=float, default=30.0, help="loss weight on the EOT token of non-speech samples")
ap.add_argument("--emb_lr", type=float, default=2.5e-4, help="LR of the new Odia embedding row")
ap.add_argument("--save_every", type=int, default=0, help="checkpoint every N steps; 0 = off")
ap.add_argument("--eval_every", type=int, default=0, help="dev evaluation every N steps (in addition to the final one); 0 = off")
ap.add_argument("--resume", action="store_true", help="resume from runs/<name>/ckpt.pt if it exists")
a = ap.parse_args()

LR = a.lr or {"full": 1e-5, "lora": 5e-4, "qlora": 5e-4}[a.mode]
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
langs = a.langs.split(",")
out = f"{ROOT}/runs/{a.name}"
os.makedirs(out, exist_ok=True)

# ---------------------------------------------------------------- model
proc = WhisperProcessor.from_pretrained(a.model)
tok = proc.tokenizer
SOT, TRANS, NOTS, EOT = special_ids(tok)

load_kw = {}
if a.mode == "qlora":
    from transformers import BitsAndBytesConfig
    load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                                        bnb_4bit_use_double_quant=True)
    load_kw["device_map"] = {"": 0}
model = WhisperForConditionalGeneration.from_pretrained(a.model, **load_kw)
if a.mode != "qlora":
    model = model.cuda()

V0 = model.config.vocab_size
LANGID = {l: tok.convert_tokens_to_ids(f"<|{l}|>") for l in ["hi", "en", "te", "as", "bn", "mr"]}
if "or" in langs:
    LANGID["or"] = V0
    model.resize_token_embeddings(V0 + 1)
    with torch.no_grad():
        E = model.get_input_embeddings().weight
        E[V0] = E[LANGID["bn"]].clone()
    model.config.vocab_size = V0 + 1
model.generation_config.forced_decoder_ids = None
model.generation_config.max_length = None
model.config.use_cache = False          # training; generate() still uses the KV cache

if a.mode in ("lora", "qlora"):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if a.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
base = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model   # unwrapped Whisper

if a.freeze_enc_frac > 0:
    layers = base.model.encoder.layers
    for layer in layers[:int(len(layers) * a.freeze_enc_frac)]:
        for p in layer.parameters():
            p.requires_grad = False

emb = base.get_input_embeddings().weight
if "or" in langs:
    emb.requires_grad = True
    row_mask = torch.zeros_like(emb)
    row_mask[V0] = 1
    emb.register_hook(lambda g: g * row_mask.to(g.dtype))

# ---------------------------------------------------------------- optimiser
groups = [{"params": [p for p in model.parameters() if p.requires_grad and p is not emb], "weight_decay": 0.01 if a.mode == "full" else 0.0}]
# <|or|> row has its own LR. 400-step Whisper-small check (runs/embchk_*): Odia token-LID 0.74 / 0.88 / 0.90 for emb_lr
# 5e-5 / 2.5e-4 / 1.5e-3, other languages unchanged -> 2.5e-4 (same LID as 1.5e-3, less drift). No weight decay: with
# zero gradients Adam leaves the other rows unchanged, but decay would not.
if emb.requires_grad:
    groups.append({"params": [emb], "weight_decay": 0.0, "lr": a.emb_lr})
head_langs = sorted(langs)
LIDX = {l: i + 1 for i, l in enumerate(head_langs)}          # 0 = CTC blank
head = None
if a.lid_ctc > 0:
    head = torch.nn.Linear(base.config.d_model, len(langs) + 1).cuda()
    groups.append({"params": list(head.parameters()), "weight_decay": 0.0, "lr": 1e-3})
trainable = [p for g in groups for p in g["params"]]
ntrain = sum(p.numel() for p in trainable)
ntot = sum(p.numel() for p in model.parameters())
print(f"mode {a.mode} lr {LR} trainable {ntrain/1e6:.1f}M / {ntot/1e6:.1f}M ({ntrain/ntot:.1%}) langs {langs}", flush=True)
opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.98), eps=1e-6)


def lr_factor(s):
    """linear warmup, then linear decay to 0"""
    w = max(1, a.warmup * a.steps)
    return (s + 1) / w if s < w else max(0.0, (a.steps - s) / max(1, a.steps - w))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

# ---------------------------------------------------------------- data
rows = [r for r in read_jsonl(a.manifest) if r["lang"] in langs and (a.cs or not r["cs"]) and r["ntok"] <= MAX_TEXT_TOKENS]
random.Random(a.seed).shuffle(rows)
print("train utts", len(rows), "hours", round(sum(r["dur"] for r in rows) / 3600, 1),
      "by lang", dict(collections.Counter(r["lang"] for r in rows)), flush=True)


class EpochBatches:
    """Seeded batch order (permutation seed+epoch) that can start mid-epoch, so a resumed run sees the same data.
    Yields (epoch, index) pairs so each sample's augmentation can be seeded independently of worker scheduling."""
    def __init__(self, n, bs, seed):
        self.n, self.bs, self.seed, self.epoch, self.start = n, bs, seed, 0, 0

    def __len__(self):
        return self.n // self.bs - self.start

    def __iter__(self):
        perm = np.random.RandomState(self.seed + self.epoch).permutation(self.n)
        for b in range(self.start, self.n // self.bs):
            yield [(self.epoch, int(i)) for i in perm[b * self.bs:(b + 1) * self.bs]]


class DS(Dataset):
    """-> (wave, target ids, LID-CTC class, valid encoder frames, is_non_speech)"""
    def __len__(self):
        return len(rows)

    def __getitem__(self, key):
        epoch, i = key
        rng = random.Random(f"{a.seed}:{epoch}:{i}")
        if a.nonspeech > 0 and rng.random() < a.nonspeech:
            # noise/music only, empty transcript; the language token is a random supported language (masked as a target in collate)
            w = noise_clip(rng, A.NOISE_TRAIN, A.read_mono).astype(np.float32)
            lg = rng.choice(langs)
            return w, [LANGID[lg], TRANS, NOTS, EOT], LIDX[lg], enc_frames(len(w)), True
        r = rows[i]
        w, _ = sf.read(r["path"], dtype="float32")
        if a.aug:
            w = A.augment(w, rng)
        ids = tok(r["text_norm"], add_special_tokens=False).input_ids
        return w, [LANGID[r["lang"]], TRANS, NOTS] + ids + [EOT], LIDX[r["lang"]], enc_frames(len(w)), False


def collate(batch):
    L = min(448, -(-max(len(x[1]) for x in batch) // 64) * 64)   # pad decoder length to a multiple of 64 -> few compiled shapes
    labels = torch.full((len(batch), L), -100)
    dec_in = torch.full((len(batch), L), EOT)
    for i, (_, y, _, _, non_speech) in enumerate(batch):
        labels[i, :len(y)] = torch.tensor(y)
        dec_in[i, 0] = SOT
        dec_in[i, 1:len(y)] = torch.tensor(y[:-1])
        if non_speech:
            labels[i, 0] = -100          # random language token -> not a LID target
    return ([x[0] for x in batch], dec_in, labels, torch.tensor([x[2] for x in batch]),
            torch.tensor([x[3] for x in batch]), torch.tensor([x[4] for x in batch]))


batches = EpochBatches(len(rows), a.bs, a.seed)
dl = DataLoader(DS(), batch_sampler=batches, num_workers=a.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True)

# ---------------------------------------------------------------- dev evaluation (FLEURS / IndicVoices / Kathbath dev for the run's languages)
dev_groups = collections.defaultdict(list)
for r in read_jsonl(f"{SPLITS}/dev.jsonl"):
    if r["lang"] in langs and not r["cs"]:
        dev_groups[(r["corpus"], r["lang"])].append(r)
rng = random.Random(0)
dev_sel = {k: rng.sample(v, min(a.eval_n, len(v))) for k, v in sorted(dev_groups.items())}


def features(waves):
    return proc.feature_extractor(waves, sampling_rate=16000, return_tensors="pt", device="cuda").input_features.cuda()


@torch.no_grad()
def evaluate(tag):
    model.eval()
    lid_langs = [l for l in sorted(LANGID) if l in langs]
    lid_ids = [LANGID[l] for l in lid_langs]
    res = {}
    for (corpus, lang), rs in dev_sel.items():
        ref, hyp, lid, lid_h, loops = [], [], [], [], 0
        for i in range(0, len(rs), 48):
            b = rs[i:i + 48]
            f = features([sf.read(r["path"], dtype="float32")[0] for r in b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc = base.model.encoder(f)
                first = base(encoder_outputs=enc, decoder_input_ids=torch.full((len(b), 1), SOT, device="cuda")).logits[:, -1]
                lid += [lid_langs[j] for j in first[:, lid_ids].argmax(-1).tolist()]
                if head is not None:
                    lid_h += head_lid(head, enc.last_hidden_state, [enc_frames(r["dur"] * 16000) for r in b], head_langs)
                prompt = torch.tensor([[SOT, LANGID[lang], TRANS, NOTS]] * len(b), device="cuda")
                gen = GenerationMixin.generate(base, encoder_outputs=enc, decoder_input_ids=prompt, max_new_tokens=440, do_sample=False)
            gen = gen[:, 4:]
            loops += int(((gen != EOT).sum(1) >= 440).sum())
            hyp += tok.batch_decode(gen, skip_special_tokens=True)
            ref += [r["text"] for r in b]
        s = score(ref, hyp, lang)
        res[f"{corpus}/{lang}"] = {"cer": s["cer"], "wer": s["wer"], "lid_acc": float(np.mean([x == lang for x in lid])),
                                   "lid_acc_head": float(np.mean([x == lang for x in lid_h])) if lid_h else None,
                                   "loop_rate": loops / len(rs), "n": len(rs)}
    model.train()
    fmt = lambda v: (f"CER {v['cer']:.3f} WER {v['wer']:.3f} LID {v['lid_acc']:.2f}"
                     + (f" LIDhead {v['lid_acc_head']:.2f}" if v["lid_acc_head"] is not None else "") + f" loop {v['loop_rate']:.2f}")
    print(f"[eval {tag}]", {k: fmt(v) for k, v in res.items()}, flush=True)
    return res


# ---------------------------------------------------------------- checkpointing
results = {"args": vars(a), "lr": LR, "trainable_M": ntrain / 1e6}
step, seen, hist, ctc_hist, t0 = 0, 0, [], [], time.time()
CK = f"{out}/ckpt.pt"
specaug_gen = torch.Generator().manual_seed(a.seed)   # own generator: the DataLoader also draws from the global one


def save_ckpt():
    sd = {"params": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
          "head": head.state_dict() if head is not None else None, "opt": opt.state_dict(), "sched": sched.state_dict(),
          "step": step, "epoch": batches.epoch, "batch_in_epoch": batches.start,
          "specaug_rng": specaug_gen.get_state(), "cuda_rng": torch.cuda.get_rng_state(),
          "hist": hist, "ctc_hist": ctc_hist, "results": results, "elapsed": time.time() - t0, "seen": seen}
    torch.save(sd, CK + ".tmp")
    os.replace(CK + ".tmp", CK)
    print(f"[ckpt] step {step} -> {CK}", flush=True)


if a.resume and os.path.exists(CK):
    sd = torch.load(CK, map_location="cuda", weights_only=False)
    P = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in sd["params"].items():
            P[n].copy_(v)
    if head is not None:
        head.load_state_dict(sd["head"])
    opt.load_state_dict(sd["opt"])
    sched.load_state_dict(sd["sched"])
    step, hist, ctc_hist, seen = sd["step"], sd["hist"], sd["ctc_hist"], sd["seen"]
    results.update({k: v for k, v in sd["results"].items() if k.startswith("eval_")})
    t0 -= sd["elapsed"]
    if "epoch" in sd:                       # checkpoints written before 2026-10-07 have no data position (data order restarts)
        batches.epoch, batches.start = sd["epoch"], sd["batch_in_epoch"]
        specaug_gen.set_state(sd["specaug_rng"].cpu())
        torch.cuda.set_rng_state(sd["cuda_rng"].cpu())     # LoRA dropout
    print(f"[resume] from step {step} (epoch {batches.epoch}, batch {batches.start})", flush=True)

# ---------------------------------------------------------------- training
if a.eval_start:
    results["eval_step0"] = evaluate("step0")


def fwd_fn(f, dec_in):
    enc = base.model.encoder(f)
    return base(encoder_outputs=enc, decoder_input_ids=dec_in).logits, enc.last_hidden_state


fwd = fwd_fn if a.no_compile else torch.compile(fwd_fn)
model.train()
while step < a.steps:
    for wavs, dec_in, labels, lidx, nfr, non_speech in dl:
        batches.start += 1                  # position of the next batch, for checkpoints
        labels, lidx, nfr, non_speech = labels.cuda(), lidx.cuda(), nfr.cuda(), non_speech.cuda()
        f = features(wavs)
        if a.aug:
            f = A.spec_augment(f, (nfr * 2).tolist(), specaug_gen)        # 100 mel frames/s = 2 x encoder frames
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, enc_h = fwd(f, dec_in.cuda())

        # ASR loss: token CE over valid label positions; position 3 is the EOT of a non-speech sample (optionally up-weighted)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="none").view(labels.shape)
        w_tok = torch.ones_like(ce)
        if a.nonspeech > 0:
            w_tok[non_speech, 3] = a.ns_weight
        valid = labels != -100
        loss = (ce * w_tok * valid).sum() / valid.sum()

        # LID-CTC: one language label per utterance over the encoder frames, normalised per frame so lambda is comparable to the CE;
        # non-speech clips carry a random language and are excluded
        if head is not None:
            h = enc_h.detach() if a.lid_detach else enc_h
            log_probs = head(h.float()).log_softmax(-1).transpose(0, 1)          # (T, B, C)
            ctc = F.ctc_loss(log_probs, lidx.unsqueeze(1), nfr, torch.ones_like(nfr), blank=0, zero_infinity=True, reduction="none") / nfr.float()
            speech = (~non_speech).float()
            lctc = (ctc * speech).sum() / speech.sum().clamp(min=1)
            loss = loss + a.lid_ctc * lctc
            ctc_hist.append(lctc.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1; seen += len(wavs); hist.append(loss.item())

        if step % 20 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {np.mean(hist[-20:]):.4f} lr {sched.get_last_lr()[0]:.2e} {seen/el:.1f} utt/s  "
                  f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB" + (f"  lid-ctc {np.mean(ctc_hist[-20:]):.4f}" if ctc_hist else ""), flush=True)
        if a.eval_every and step % a.eval_every == 0 and step < a.steps:
            results[f"eval_step{step}"] = evaluate(f"step{step}")
            json.dump(results, open(f"{out}/result_partial.json", "w"), indent=1)
        if a.save_every and step % a.save_every == 0 and step < a.steps:
            save_ckpt()
        if step >= a.steps:
            break
    else:                                   # epoch finished
        batches.epoch += 1
        batches.start = 0

results["train_time_s"] = time.time() - t0
results["utt_per_s"] = seen / results["train_time_s"]
results["final_loss"] = float(np.mean(hist[-50:]))
results["eval_final"] = evaluate("final")
json.dump(results, open(f"{out}/result.json", "w"), indent=1)
if a.mode == "full":                        # whole model + processor; load_ft.load(f"{out}/model") detects the Odia token from the vocab size
    model.save_pretrained(f"{out}/model")
    proc.save_pretrained(f"{out}/model")
else:
    model.save_pretrained(f"{out}/adapter")
if head is not None:
    save_lid_head(head, langs, f"{out}/lid_head.pt")
print("DONE", out, flush=True)
