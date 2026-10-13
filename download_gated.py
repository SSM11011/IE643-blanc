# Gated + language-subset downloads for IE643 (blanc). Languages: hi en te as or bn mr. Resumable.
import os
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
L = ["hindi","marathi","telugu","assamese","odia","bengali"]
def shards(repo, lang, split, n):
    fs = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                if f.lower().startswith(lang + "/") and f.split("/")[1].startswith(split + "-"))
    return fs[:n]
def get(repo, files, out):
    try:
        if files: snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=files)
    except Exception as e: print("SKIPPED", repo, type(e).__name__, flush=True)
os.chdir("/home/shibam/IE643")
snapshot_download("ai4bharat/indic-conformer-600m-multilingual", local_dir="models/indic-conformer-600m-multilingual")
get("ai4bharat/Svarah", ["*"], "data/svarah")
get("ai4bharat/Lahaja", ["*"], "data/lahaja")
snapshot_download("google/fleurs", repo_type="dataset", local_dir="data/fleurs",
                  allow_patterns=[f"data/{l}/*" for l in ["te_in","as_in","or_in"]])
for l in L:
    # IndicVoices: all valid (eval) + 6 train shards (~100 h) per language
    get("ai4bharat/IndicVoices", shards("ai4bharat/IndicVoices", l, "valid", 99) + shards("ai4bharat/IndicVoices", l, "train", 6), "data/indicvoices")
    # Kathbath: valid + 3 train shards (no Assamese in Kathbath)
    get("ai4bharat/Kathbath", shards("ai4bharat/Kathbath", l, "valid", 99) + shards("ai4bharat/Kathbath", l, "train", 3), "data/kathbath")
    # Shrutilipi (read news speech): 3 train shards
    get("ai4bharat/Shrutilipi", shards("ai4bharat/Shrutilipi", l, "train", 3), "data/shrutilipi")
print("GATED_DONE")
