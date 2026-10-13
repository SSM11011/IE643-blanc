"""One-off patch (2026-10-03, already applied; kept for provenance): normalize.py now maps U+2019 to an apostrophe for English. Re-apply normalize() to the stored text_norm of
every English row in manifests/ and splits/ and recompute ntok for changed rows. Other languages are untouched (their normalization did not change)."""
import json, glob
from transformers import WhisperTokenizerFast
from common import MODELS, DATA as P
from normalize import normalize
t = WhisperTokenizerFast.from_pretrained(f"{MODELS}/whisper-large-v3-turbo")
for f in sorted(glob.glob(f"{P}/manifests/*.jsonl") + glob.glob(f"{P}/splits/*.jsonl")):
    rows = [json.loads(l) for l in open(f, encoding="utf8")]; ch = 0
    for r in rows:
        if r["lang"] != "en": continue
        tn = normalize(r["text"], "en")
        if tn != r["text_norm"]:
            r["text_norm"] = tn; ch += 1
            if "ntok" in r: r["ntok"] = len(t(tn, add_special_tokens=False).input_ids)
    if ch:
        with open(f, "w", encoding="utf8") as fh:
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f.rsplit("/", 1)[-1], "changed", ch)
