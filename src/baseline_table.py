"""reports/baselines/summary.md: vanilla Whisper-small / turbo, IndicConformer, Qwen3-ASR (hi/en only) and ECAPA LID on the same test utterances."""
import json, collections, os
from common import ROOT, LANGS
B = f"{ROOT}/reports/baselines"
sm = json.load(open(f"{B}/whisper-small_test.json")); tb = json.load(open(f"{B}/whisper-large-v3-turbo_test.json")); ic = json.load(open(f"{B}/indicconformer_test.json"))
out = ["| corpus/lang | small CER | turbo CER | turbo WER | turbo raw WER | IndicConf CER | IndicConf WER | turbo LID acc |", "|---|---|---|---|---|---|---|---|"]
for k in sorted(tb):
    t = tb[k]; s = sm[k]; i = ic.get(k)
    f = lambda x: f"{x:.3f}"
    out.append(f"| {k} | {f(s['oracle_norm']['cer'])} | {f(t['oracle_norm']['cer'])} | {f(t['oracle_norm']['wer'])} | {f(t['oracle_raw']['wer'])} | {f(i['norm']['cer']) if i else '-'} | {f(i['norm']['wer']) if i else '-'} | {f(t['lid_acc']) if 'lid_acc' in t else 'n/a'} |")
# macro averages per language over non-CS corpora
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for k, t in tb.items():
    c, l = k.split("/")
    if c == "mucs": continue
    agg[l]["small"].append(sm[k]["oracle_norm"]["cer"]); agg[l]["turbo"].append(t["oracle_norm"]["cer"])
    if k in ic: agg[l]["ic"].append(ic[k]["norm"]["cer"])
    if "lid_acc" in t: agg[l]["lid"].append(t["lid_acc"])
out += ["", "Mean over corpora (non-code-switch):", "| lang | small CER | turbo CER | IndicConf CER | turbo LID acc |", "|---|---|---|---|---|"]
m = lambda v: f"{sum(v)/len(v):.3f}" if v else "-"
for l in LANGS:
    a = agg[l]; out.append(f"| {l} | {m(a['small'])} | {m(a['turbo'])} | {m(a['ic'])} | {m(a['lid'])} |")
# Qwen3-ASR (supports only hi/en of our languages: CER with the language forced) and the LID baselines
ld = lambda n: json.load(open(f"{B}/{n}")) if os.path.exists(f"{B}/{n}") else {}
q06, q17, ec = ld("qwen3asr-0.6b_test.json"), ld("qwen3asr-1.7b_test.json"), ld("ecapa_lid_test.json")
g = lambda d, k, *path: (lambda v: f"{v:.3f}" if isinstance(v, float) else "-")(__import__("functools").reduce(lambda x, y: x.get(y, {}) if isinstance(x, dict) else {}, path, d.get(k, {})))
out += ["", "Qwen3-ASR (forced language; Hindi/English only) and LID baselines. LID: turbo = first decoder token over the 6 languages it knows; "
        "ECAPA = VoxLingua107 ECAPA-TDNN, closed set over the 6 (no Odia class); Qwen = its own language output (open set).",
        "| corpus/lang | Qwen3-0.6B CER | Qwen3-1.7B CER | turbo LID | ECAPA LID | Qwen3-0.6B LID | Qwen3-1.7B LID |", "|---|---|---|---|---|---|---|"]
for k in sorted(tb):
    out.append(f"| {k} | {g(q06, k, 'oracle_norm', 'cer')} | {g(q17, k, 'oracle_norm', 'cer')} | {g(tb, k, 'lid_acc')} | {g(ec, k, 'closed_acc')} "
               f"| {g(q06, k, 'lid_acc')} | {g(q17, k, 'lid_acc')} |")
open(f"{B}/summary.md", "w").write("\n".join(out)); print("\n".join(out))
