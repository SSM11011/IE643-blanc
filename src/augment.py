"""Training-time augmentation (no denoiser front-end, per plan): MUSAN noise/music at random SNR, RIR reverb, speed perturbation
(+-10%), SpecAugment. Noise and RIR files are split by path hash: 80% for training, 20% held out for the SNR/reverb/hallucination
evaluations (no train/eval file overlap)."""
import glob, hashlib, random
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, fftconvolve
from common import ROOT, SR

N = f"{ROOT}/data/noise"


def _split(files, held_out):
    is_held = lambda f: int(hashlib.md5(f.encode()).hexdigest(), 16) % 5 == 0       # 20% held out
    return [f for f in files if is_held(f) == held_out]


_noise_all = sorted(glob.glob(f"{N}/musan/noise/*/*.wav") + glob.glob(f"{N}/musan/music/*/*.wav"))
NOISE_TRAIN, NOISE_EVAL = _split(_noise_all, False), _split(_noise_all, True)
# real_rirs_isotropic_noises/ also holds 30-s 8-channel isotropic NOISE recordings (*_noise_*): not impulse responses -> excluded.
# Simulated RIRs are subsampled 1/30 so they do not outnumber the real ones.
_rir_all = (sorted(f for f in glob.glob(f"{N}/RIRS_NOISES/real_rirs_isotropic_noises/*.wav") if "_noise_" not in f.rsplit("/", 1)[-1])
            + sorted(glob.glob(f"{N}/RIRS_NOISES/simulated_rirs/*/*/*.wav"))[::30])
RIR_TRAIN, RIR_EVAL = _split(_rir_all, False), _split(_rir_all, True)


def read_mono(p):
    x, sr = sf.read(p, dtype="float32")
    assert sr == SR, (p, sr)
    return x.mean(1) if x.ndim > 1 else x


def read_rir(p):
    x, sr = sf.read(p, dtype="float32")
    assert sr == SR, (p, sr)
    return x[:, 0] if x.ndim > 1 else x      # one microphone (averaging mics with different delays smears the RIR)


def add_noise(w, snr_db, rng, files):
    n = read_mono(rng.choice(files))
    if len(n) < len(w):
        n = np.tile(n, -(-len(w) // len(n)))
    s = rng.randrange(0, len(n) - len(w) + 1)
    n = n[s:s + len(w)]
    pw, pn = float(np.mean(w ** 2)) + 1e-10, float(np.mean(n ** 2)) + 1e-10
    return (w + n * np.sqrt(pw / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def reverb(w, rng, files):
    r = read_rir(rng.choice(files))
    r = r / (np.abs(r).max() + 1e-9)
    y = fftconvolve(w, r)[:len(w)]
    return (y * np.sqrt((np.mean(w ** 2) + 1e-10) / (np.mean(y ** 2) + 1e-10))).astype(np.float32)   # keep the input energy


def speed(w, rng):
    """resample by 10/9 or 11/10 (slower, longer) or 9/10 or 10/11 (faster)"""
    up, down = rng.choice([(10, 9), (11, 10)]) if rng.random() < .5 else rng.choice([(9, 10), (10, 11)])
    return resample_poly(w, up, down).astype(np.float32)


def augment(w, rng):
    # the largest slow-down (10/9) must not push a clip past Whisper's 30-s window (the label would then be truncated): 27 s * 10/9 = 30 s
    if rng.random() < 0.3 and len(w) < 27 * SR:
        w = speed(w, rng)
    if rng.random() < 0.2:
        w = reverb(w, rng, RIR_TRAIN)
    if rng.random() < 0.5:
        w = add_noise(w, rng.uniform(0, 20), rng, NOISE_TRAIN)
    return np.clip(w, -1, 1)


def spec_augment(f, lens=None, gen=None):
    """f: (B, n_mels, 3000) log-mel on GPU; lens: valid mel frames per sample (masks go inside the real audio, not the padding);
    gen: optional CPU torch.Generator for the mask positions.
    2 time masks (<40 frames) + 2 freq masks (<10 bins) per sample, filled with the per-sample minimum."""
    import torch
    B, F, T = f.shape
    f = f.clone()
    for b in range(B):
        fill = f[b].min()
        Tb = T if lens is None else max(41, min(T, int(lens[b])))
        for _ in range(2):
            t = int(torch.randint(0, 40, (1,), generator=gen)); s = int(torch.randint(0, Tb - t, (1,), generator=gen)); f[b, :, s:s + t] = fill
            k = int(torch.randint(0, 10, (1,), generator=gen)); s = int(torch.randint(0, F - k, (1,), generator=gen)); f[b, s:s + k, :] = fill
    return f


if __name__ == "__main__":
    print(len(NOISE_TRAIN), len(NOISE_EVAL), len(RIR_TRAIN), len(RIR_EVAL))
    rng = random.Random(0)
    w = np.random.randn(80000).astype("float32") * .1
    print(augment(w, rng).shape, add_noise(w, 10, rng, NOISE_EVAL).shape, reverb(w, rng, RIR_EVAL).shape)
