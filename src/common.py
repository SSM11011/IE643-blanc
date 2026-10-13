"""Shared constants and helpers for training and evaluation scripts."""
import json
import numpy as np
import torch

ROOT = "/home/shibam/IE643"
DATA = f"{ROOT}/data/processed"
SPLITS = f"{DATA}/splits"
MODELS = f"{ROOT}/models"
LANGS = ["hi", "en", "te", "as", "or", "bn", "mr"]
MAX_TEXT_TOKENS = 440        # Whisper decoder context 448 = 4 prompt tokens + text + EOT (+ margin)
SR = 16000
ENC_FPS = 50                 # Whisper encoder frames per second (1500 frames per 30-s window)


def read_jsonl(path):
    with open(path, encoding="utf8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def special_ids(tok):
    """(SOT, TRANSCRIBE, NOTIMESTAMPS, EOT) token ids."""
    c = tok.convert_tokens_to_ids
    return c("<|startoftranscript|>"), c("<|transcribe|>"), c("<|notimestamps|>"), tok.eos_token_id


def enc_frames(n_samples):
    """Number of valid (non-padding) Whisper encoder frames for a clip of n_samples."""
    return min(1500, int(n_samples / SR * ENC_FPS) + 1)


def noise_clip(rng, files, read):
    """Random 2-12 s excerpt of a random noise/music file (tiled if the file is shorter)."""
    x = read(rng.choice(files))
    n = rng.randint(2 * SR, 12 * SR)
    if len(x) < n:
        x = np.tile(x, -(-n // len(x)))
    s = rng.randrange(0, len(x) - n + 1)
    return x[s:s + n]


def head_lid(head, enc_h, n_frames, head_langs):
    """Utterance-level LID from the encoder LID-CTC head: sum frame posteriors (blank excluded) over valid frames, argmax.
    enc_h: (B, T, d) encoder output; n_frames: valid frames per sample; head_langs: language of output class k+1 (class 0 = blank)."""
    p = head(enc_h.float()).softmax(-1)[:, :, 1:]
    valid = torch.arange(p.shape[1], device=p.device)[None] < torch.as_tensor(n_frames, device=p.device)[:, None]
    return [head_langs[j] for j in (p * valid.unsqueeze(-1)).sum(1).argmax(-1).tolist()]


def save_lid_head(head, langs, path):
    torch.save({"state_dict": head.state_dict(), "langs": sorted(langs)}, path)


def load_lid_head(path, run_langs=None):
    """Returns (head, head_langs). Older files hold a bare state_dict; their class order is sorted(run langs from result.json)."""
    sd = torch.load(path, map_location="cuda")
    if "state_dict" in sd:
        sd, langs = sd["state_dict"], sd["langs"]
    else:
        langs = sorted(run_langs)
    head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda().eval()
    head.load_state_dict(sd)
    assert head.out_features == len(langs) + 1, (path, langs)
    return head, langs
