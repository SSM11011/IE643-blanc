"""Manual-prompt generation (GenerationMixin.generate) must equal HF Whisper generate(language=...)."""
import sys, json, random, soundfile as sf, torch
sys.path.insert(0, ".."); from metrics import score
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationMixin
P = "/home/shibam/IE643/models/whisper-small"; M = "/home/shibam/IE643/data/processed"
proc = WhisperProcessor.from_pretrained(P); tok = proc.tokenizer
m = WhisperForConditionalGeneration.from_pretrained(P, torch_dtype=torch.bfloat16).cuda().eval(); m.generation_config.max_length = None
SOT = tok.convert_tokens_to_ids("<|startoftranscript|>"); T = tok.convert_tokens_to_ids("<|transcribe|>"); N = tok.convert_tokens_to_ids("<|notimestamps|>")
rows = [r for r in map(json.loads, open(f"{M}/splits/dev.jsonl", encoding="utf8")) if r["lang"] in ("hi", "en") and r["corpus"] in ("fleurs",)]
random.Random(0).shuffle(rows)
for lang in ["hi", "en"]:
    rs = [r for r in rows if r["lang"] == lang][:48]
    f = proc.feature_extractor([sf.read(r["path"], dtype="float32")[0] for r in rs], sampling_rate=16000, return_tensors="pt", device="cuda").input_features.cuda().to(torch.bfloat16)
    with torch.no_grad():
        a = m.generate(input_features=f, language=lang, task="transcribe", max_new_tokens=440)
        mm = WhisperForConditionalGeneration.from_pretrained(P, torch_dtype=torch.bfloat16).cuda().eval(); mm.generation_config.forced_decoder_ids = None; mm.generation_config.max_length = None
        prompt = torch.tensor([[SOT, tok.convert_tokens_to_ids(f"<|{lang}|>"), T, N]] * len(rs), device="cuda")
        b = GenerationMixin.generate(mm, input_features=f, decoder_input_ids=prompt, max_new_tokens=440, do_sample=False)[:, 4:]
    ha = tok.batch_decode(a, skip_special_tokens=True); hb = tok.batch_decode(b, skip_special_tokens=True); ref = [r["text"] for r in rs]
    same = sum(x.strip() == y.strip() for x, y in zip(ha, hb))
    print(f"{lang}: identical outputs {same}/{len(rs)} | CER hf-generate {score(ref, ha, lang)['cer']:.4f} vs manual {score(ref, hb, lang)['cer']:.4f}")
