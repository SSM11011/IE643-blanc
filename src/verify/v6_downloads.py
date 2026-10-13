import os
from huggingface_hub import HfApi
api = HfApi(); R = "/home/shibam/IE643"
chk = [("ai4bharat/IndicVoices","dataset","data/indicvoices"),("ai4bharat/Kathbath","dataset","data/kathbath"),("ai4bharat/Shrutilipi","dataset","data/shrutilipi"),
       ("ai4bharat/Svarah","dataset","data/svarah"),("ai4bharat/Lahaja","dataset","data/lahaja"),("google/fleurs","dataset","data/fleurs"),
       ("openai/whisper-large-v3-turbo","model","models/whisper-large-v3-turbo"),("openai/whisper-small","model","models/whisper-small"),("Qwen/Qwen3-ASR-0.6B","model","models/Qwen3-ASR-0.6B"),
       ("Qwen/Qwen3-ASR-1.7B","model","models/Qwen3-ASR-1.7B"),("speechbrain/lang-id-voxlingua107-ecapa","model","models/lang-id-voxlingua107-ecapa"),
       ("microsoft/paza-whisper-large-v3-turbo","model","models/paza-whisper-large-v3-turbo"),("ai4bharat/indic-conformer-600m-multilingual","model","models/indic-conformer-600m-multilingual")]
for repo, typ, d in chk:
    info = api.repo_info(repo, repo_type=typ, files_metadata=True); n = ok = bad = 0; badl = []
    for s in info.siblings:
        p = f"{R}/{d}/{s.rfilename}"
        if not os.path.exists(p): continue
        n += 1
        if os.path.getsize(p) == s.size: ok += 1
        else: bad += 1; badl.append((s.rfilename, os.path.getsize(p), s.size))
    print(f"{repo}: local files matched to hub listing {n}, size-exact {ok}, MISMATCH {bad}", badl[:3])
