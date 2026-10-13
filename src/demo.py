"""Gradio demo: audio -> language (with confidence) + transcript, using the Stage-2 model (turbo + LoRA + LID-CTC head).
Long audio is cut into chunks of <= 12 s at pauses found by Silero VAD (the model was trained on single utterances; on a
packed 30-s window it skipped a sentence and the LID head became uniform). No audio is discarded. Language is decided once
for the whole clip by summing per-chunk scores, unless forced. Score = decoder-token log-probability (closed set of the 7
languages) + the calibrated rule from lid_calib.py (English bias + weighted LID-head log-probability) if runs/<name>/lid_rule.json
exists. English is decoded with non-ASCII tokens suppressed (stops Indic-script output for Indian-accented English).
Usage: python demo.py [--adapter ../runs/s2_turbo/adapter] [--port 7860] [--share]"""
import argparse, json, os
import numpy as np, torch
import gradio as gr
import sys
sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../repos/silero-vad/src")
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import GenerationMixin
from common import MODELS, ROOT, SR, special_ids, enc_frames, load_lid_head
from load_ft import load

NAMES = {"hi": "Hindi", "en": "English", "te": "Telugu", "as": "Assamese", "or": "Odia", "bn": "Bengali", "mr": "Marathi"}
MAX_CHUNK = 12 * SR                  # Indic scripts cost 17-45 Whisper tokens per second; 440-token decode budget

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{MODELS}/whisper-large-v3-turbo")
ap.add_argument("--adapter", default=f"{ROOT}/runs/s2_turbo/adapter")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--share", action="store_true")
a = ap.parse_args()

proc, tok, model, LANGID = load(a.model, a.adapter)
LANGS = list(LANGID)
SOT, TR, NO, _ = special_ids(tok)
vad = load_silero_vad()
EN_SUPPRESS = sorted(set(model.generation_config.suppress_tokens or []) | {i for i in range(len(tok)) if not tok.decode([i]).isascii()})
head, head_langs = None, None
run_dir = os.path.dirname(os.path.normpath(a.adapter))
if os.path.exists(f"{run_dir}/lid_head.pt"):
    head, head_langs = load_lid_head(f"{run_dir}/lid_head.pt", json.load(open(f"{run_dir}/result.json"))["args"]["langs"].split(","))
RULE = json.load(open(f"{run_dir}/lid_rule.json")) if head is not None and os.path.exists(f"{run_dir}/lid_rule.json") else None


def to_16k(sr, x):
    if np.issubdtype(x.dtype, np.integer):          # int PCM from gradio (int16 or int32)
        x = x / float(np.iinfo(x.dtype).max + 1)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32)
    return x


def chunks(x):
    """cut points = middle of the VAD pauses; greedily pack into chunks <= MAX_CHUNK (hard cut inside very long speech)"""
    if len(x) <= MAX_CHUNK:
        return [x]
    ts = get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=SR, min_silence_duration_ms=200)
    cuts = [(a["end"] + b["start"]) // 2 for a, b in zip(ts, ts[1:])]
    out, start = [], 0
    while len(x) - start > MAX_CHUNK:
        ok = [c for c in cuts if start < c <= start + MAX_CHUNK]
        end = ok[-1] if ok else start + MAX_CHUNK
        out.append(x[start:end])
        start = end
    out.append(x[start:])
    return [w for w in out if len(w) > 0.3 * SR] or out[:1]


@torch.no_grad()
def run(audio, forced):
    if audio is None:
        return "", "", {}
    x = to_16k(*audio)
    wins = chunks(x)
    f = proc.feature_extractor(wins, sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
    out = model(input_features=f, decoder_input_ids=torch.full((len(wins), 1), SOT, device="cuda"))
    score = out.logits[:, -1, [LANGID[l] for l in LANGS]].float().log_softmax(-1)
    head_txt = ""
    if head is not None:
        p = head(out.encoder_last_hidden_state.float()).softmax(-1)[:, :, 1:]
        n = [enc_frames(len(w)) for w in wins]
        mass = torch.stack([p[i, :n[i]].sum(0) for i in range(len(wins))])
        mass = (mass / mass.sum(-1, keepdim=True))[:, [head_langs.index(l) for l in LANGS]]
        hm = mass.mean(0)
        head_txt = f"encoder LID head: {NAMES[LANGS[int(hm.argmax())]]} ({hm.max().item():.2f})"
        if RULE:
            score = score + RULE["head_w"] * mass.clamp_min(1e-6).log()
    if RULE:
        score = score + torch.tensor([RULE["bias"].get(l, 0.0) for l in LANGS], device="cuda")
    probs = score.sum(0).softmax(-1).tolist()
    det = LANGS[int(np.argmax(probs))]
    lang = forced if forced in LANGID else det
    prompt = torch.tensor([[SOT, LANGID[lang], TR, NO]] * len(wins), device="cuda")
    kw = {"suppress_tokens": EN_SUPPRESS} if lang == "en" else {}
    ids = GenerationMixin.generate(model, input_features=f, decoder_input_ids=prompt, max_new_tokens=440, do_sample=False, **kw)[:, 4:]
    text = " ".join(t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)).strip()
    info = f"decoded as {NAMES[lang]}" + (" (forced)" if lang != det else "") + f"; {len(x) / SR:.1f} s, {len(wins)} chunk(s)"
    if head_txt:
        info += f"\n{head_txt}"
    return text, info, {NAMES[l]: p for l, p in zip(LANGS, probs)}


with gr.Blocks(title="Indic ASR + LID (IE643 blanc)") as app:
    gr.Markdown("## Multilingual Indic ASR with language identification\nWhisper large-v3-turbo + LoRA, 7 languages: "
                + ", ".join(NAMES.values()) + ". Upload or record audio.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio")
            forced = gr.Dropdown(["auto"] + LANGS, value="auto", label="Language (auto = detect)")
            btn = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            lid = gr.Label(num_top_classes=3, label="Detected language (closed set, calibrated)")
            info = gr.Textbox(label="Details", lines=2)
            text = gr.Textbox(label="Transcript", lines=6)
    btn.click(run, [audio, forced], [text, info, lid])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=a.port, share=a.share)
