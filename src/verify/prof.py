import json, time, torch, soundfile as sf, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model
P = "/home/shibam/IE643/models/whisper-small"; M = "/home/shibam/IE643/data/processed"
proc = WhisperProcessor.from_pretrained(P)
rows = [r for r in map(json.loads, open(f"{M}/splits/train_mix_epoch0.jsonl", encoding="utf8")) if r["lang"] == "hi" and not r["cs"]][:6000]
class DS(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self, i): return sf.read(rows[i]["path"], dtype="float32")[0]
for w in [12]:
    dl = DataLoader(DS(), batch_size=32, num_workers=w, collate_fn=lambda b: b, prefetch_factor=4); t = time.time(); n = 0
    for k, b in enumerate(dl):
        n += len(b)
        if k == 40: break
    print(f"loader only, {w} workers: {n/(time.time()-t):.0f} utt/s")
wav = [np.random.randn(16000 * 10).astype("float32") for _ in range(32)]
def bench(dtype, autocast, label, bs=32):
    m = WhisperForConditionalGeneration.from_pretrained(P, torch_dtype=dtype).cuda(); m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4); din = torch.randint(0, 50000, (bs, 60)).cuda(); lab = din.clone()
    ws = wav[:bs] if bs <= 32 else wav * (bs // 32)
    for i in range(8):
        if i == 3: torch.cuda.synchronize(); t = time.time()
        f = proc.feature_extractor(ws, sampling_rate=16000, return_tensors="pt", device="cuda").input_features.cuda().to(dtype if not autocast else torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            lg = m(input_features=f, decoder_input_ids=din).logits
        loss = torch.nn.functional.cross_entropy(lg.float().view(-1, lg.size(-1)), lab.view(-1)); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); print(f"{label} bs{bs}: {5*bs/(time.time()-t):.1f} utt/s, peak {torch.cuda.max_memory_allocated()/1e9:.0f}GB", flush=True)
    del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
bench(torch.float32, True, "fp32 weights + autocast")
bench(torch.bfloat16, False, "pure bf16 weights")
bench(torch.bfloat16, False, "pure bf16 weights", bs=64)
