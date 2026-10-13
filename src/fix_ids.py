"""One-off patch (2026-10-02, already applied; kept for provenance). Patch ids in existing manifests to be unique: <corpus>/<lang>/<audio file stem> (paths are already unique)."""
import json, glob, os, collections
from common import DATA as M
for f in sorted(glob.glob(f"{M}/manifests/*.jsonl")):
    rows = [json.loads(l) for l in open(f, encoding="utf8")]
    for r in rows: r["id"] = f'{r["corpus"]}/{r["lang"]}/{os.path.basename(r["path"])[:-5]}'
    c = collections.Counter(r["id"] for r in rows); assert max(c.values()) == 1, (f, c.most_common(2))
    with open(f, "w", encoding="utf8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(os.path.basename(f), len(rows), "unique ids ok")
