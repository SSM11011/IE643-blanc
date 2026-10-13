import sys, random, numpy as np, torch
sys.path.insert(0, "/home/shibam/IE643/src"); import augment as A
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationMixin
from peft import PeftModel
P = "/home/shibam/IE643/models/whisper-small"; proc = WhisperProcessor.from_pretrained(P); tok = proc.tokenizer
SOT = tok.convert_tokens_to_ids("<|startoftranscript|>"); TR = tok.convert_tokens_to_ids("<|transcribe|>"); NO = tok.convert_tokens_to_ids("<|notimestamps|>"); HI = tok.convert_tokens_to_ids("<|hi|>")
def load(adapter):
    m = WhisperForConditionalGeneration.from_pretrained(P, torch_dtype=torch.bfloat16).cuda()
    if adapter: m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval(); m.generation_config.forced_decoder_ids = None; m.generation_config.max_length = None; m.generation_config.begin_suppress_tokens = None; return m
def run(m, clips):
    f = proc.feature_extractor(clips, sampling_rate=16000, return_tensors="pt", device="cuda").input_features.cuda().to(torch.bfloat16)
    p = torch.tensor([[SOT, HI, TR, NO]] * len(clips), device="cuda")
    with torch.no_grad(): g = GenerationMixin.generate(m, input_features=f, decoder_input_ids=p, max_new_tokens=100, do_sample=False)[:, 4:]
    return [len(tok(o, add_special_tokens=False).input_ids) for o in tok.batch_decode(g, skip_special_tokens=True)]
def clips(files, n, seed):
    r = random.Random(seed); out = []
    for _ in range(n):
        x = A.read_mono(r.choice(files)); L = r.randint(32000, 192000)
        if len(x) < L: x = np.tile(x, -(-L // len(x)))
        s = r.randrange(0, len(x) - L + 1); out.append(x[s:s + L])
    return out
for name, ad in [("e2_off (no nonspeech training)", "/home/shibam/IE643/runs/e2_off/adapter"), ("e4_aug (nonspeech, unweighted)", "/home/shibam/IE643/runs/e4_aug/adapter"), ("e4_aug2 (nonspeech, weight 30)", "/home/shibam/IE643/runs/e4_aug2/adapter")]:
    m = load(ad); tr = run(m, clips(A.NOISE_TRAIN, 40, 7)); ev = run(m, clips(A.NOISE_EVAL, 40, 8))
    print(f"{name}: TRAIN-noise clips: frac>5 tokens {np.mean(np.array(tr)>5):.2f}, mean tokens {np.mean(tr):.1f} | HELD-OUT clips: frac>5 {np.mean(np.array(ev)>5):.2f}, mean {np.mean(ev):.1f}")
    del m; torch.cuda.empty_cache()
by = {}
for f in A.NOISE_EVAL: by.setdefault(f.split('/musan/')[1].split('/')[0] + '/' + f.split('/musan/')[1].split('/')[1], []).append(f)
print({k: len(v) for k, v in by.items()})
