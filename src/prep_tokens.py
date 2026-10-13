"""Annotate splits with Whisper token counts (ntok of text_norm). Training filters ntok <= MAXTOK (decoder context 448 incl. 4 prompt tokens + EOT)."""
import json
from transformers import WhisperTokenizerFast
from common import MODELS, SPLITS as M
t = WhisperTokenizerFast.from_pretrained(f"{MODELS}/whisper-large-v3-turbo")   # same BPE as whisper-small
for k in ["train", "dev", "test"]:
    rows = [json.loads(l) for l in open(f"{M}/{k}.jsonl", encoding="utf8")]
    for i in range(0, len(rows), 2000):
        b = rows[i:i + 2000]
        for r, ids in zip(b, t([r["text_norm"] for r in b], add_special_tokens=False)["input_ids"]): r["ntok"] = len(ids)
    with open(f"{M}/{k}.jsonl", "w", encoding="utf8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(k, len(rows), "max ntok", max(r["ntok"] for r in rows), "over 440:", sum(r["ntok"] > 440 for r in rows))
