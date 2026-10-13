"""WER / CER (jiwer) over normalized text. CER is primary for agglutinative/Indic scripts; report raw AND normalized."""
import jiwer
from normalize import normalize

def score(refs, hyps, lang, normalized=True):
    if normalized:
        refs = [normalize(r, lang) for r in refs]; hyps = [normalize(h, lang) for h in hyps]
    keep = [i for i, r in enumerate(refs) if r]   # drop empty refs
    refs = [refs[i] for i in keep]; hyps = [hyps[i] for i in keep]
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps), "n": len(refs)}
