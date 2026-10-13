"""Transliteration-tolerant scoring for code-switched (MUCS) transcripts.
MUCS references write English words sometimes in Latin script and sometimes in Devanagari/Bengali; models also choose either.
Plain CER counts "computer" vs "कंप्यूटर" as completely wrong. Both sides are therefore mapped to a common key:
 rom  : normalize() -> Indic script romanized (ISO 15919, anusvara -> n, n -> m before p/b) -> diacritics stripped, lowercase, word-final inherent 'a' dropped
 skel : per word, a consonant skeleton of rom (vowels/h/y/w dropped after the first letter, c/q/x/z/ph folded, repeats collapsed)
rCER = CER on rom; sWER = WER on skel. Both are loose (partial credit for phonetically close spellings) and are applied
identically to every system; they are for comparing systems on code-switched speech, not absolute accuracy."""
import re, unicodedata
import jiwer
from indic_transliteration import sanscript
from normalize import normalize

SCRIPT = {"hi": sanscript.DEVANAGARI, "mr": sanscript.DEVANAGARI, "bn": sanscript.BENGALI, "as": sanscript.BENGALI,
          "te": sanscript.TELUGU, "or": sanscript.ORIYA}


def rom(text, lang):
    t = normalize(text, lang)
    for sc in dict.fromkeys(SCRIPT.values()):          # every Indic block, not only lang's (a hypothesis may be in another script)
        t = sanscript.transliterate(t, sc, sanscript.ISO)
    t = t.replace("ṁ", "n")                            # anusvara -> n (m before p/b below)
    t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c)).lower()
    t = re.sub(r"n(?=[pb])", "m", re.sub(r"[^a-z0-9' ]", "", t))
    return " ".join(re.sub(r"(?<=[^aeiou])a$", "", w) if len(w) > 2 else w for w in t.split())


def skel_word(w):
    w = w.replace("ph", "f").replace("ck", "k").replace("x", "ks").replace("q", "k").replace("z", "j")
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k").replace("v", "w")
    head, rest = w[:1], re.sub(r"[aeiouyhw']", "", w[1:])
    head = "a" if head in "aeiou" else head
    return re.sub(r"(.)\1+", r"\1", head + rest)


def skel(text, lang):
    return " ".join(skel_word(w) for w in rom(text, lang).split())


def cs_scores(refs, hyps, lang):
    pairs = [(rom(r, lang), rom(h, lang)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    sk = [(skel(r, lang), skel(h, lang)) for r, h in zip(refs, hyps) if rom(r, lang)]
    return {"rcer": jiwer.cer([r for r, _ in pairs], [h for _, h in pairs]),
            "swer": jiwer.wer([r for r, _ in sk], [h for _, h in sk]), "n": len(pairs)}


if __name__ == "__main__":
    for t, l in [("कंप्यूटर साइंस का कोर्स स्लाइड इन्सर्ट", "hi"), ("computer science ka course slide insert", "hi"),
                 ("কম্পিউটার সায়েন্স", "bn"), ("computer science", "bn")]:
        print(rom(t, l), "|", skel(t, l))
