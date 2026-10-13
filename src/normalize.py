"""Single text-normalization protocol used for training targets AND evaluation (fixed per plan, Stage 1).
Unicode NFC -> IndicNLP script normalizer (Indic langs) -> Whisper-BasicTextNormalizer-style cleanup
(drop bracketed tags, punctuation/symbols -> space, lowercase, collapse whitespace). Diacritics/matras are KEPT."""
import re, unicodedata
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
_F = IndicNormalizerFactory()
_INDIC = {"hi", "mr", "te", "as", "or", "bn"}
_norm = {l: _F.get_normalizer(l) for l in _INDIC}
_BRACKET = re.compile(r"[<\[][^>\]]*[>\]]|\([^)]*\)")   # same as Whisper BasicTextNormalizer
_ZW = dict.fromkeys(map(ord, "​‌‍﻿"), None)  # zero-width chars (ZWJ/ZWNJ removed)

def normalize(text: str, lang: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    t = _BRACKET.sub(" ", t).translate(_ZW)
    if lang in _norm:
        t = _norm[lang].normalize(t)
    t = unicodedata.normalize("NFKC", t) if lang == "en" else t
    t = t.lower()
    if lang == "en": t = t.replace("\u2019", "'").replace("'", "\u0001")   # keep in-word apostrophes (it's / it’s)
    # categories: M* (marks) kept; S*/P* (symbols/punct incl. danda U+0964) -> space
    t = "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in t)
    return re.sub(r"\s+", " ", t.replace("\u0001", "'")).strip()
