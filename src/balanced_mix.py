"""Two-stage deterministic balanced upsampling (Polyglot-Lion style) of the 7-language train pool, by reference (no audio copies).
Stage 1: within a language, replicate smaller corpora toward the largest (replication <= REP_CAP), cap any corpus share at SHARE_CAP.
Stage 2: every language gets exactly T hours per epoch (this can replicate a corpus beyond REP_CAP when a language is small).
Code-switch (MUCS) = separate pool at CS_FRAC of the total. Language tags are kept (needed for the <|lang|> target / LID-CTC).
Usage: python balanced_mix.py [seed]   -> splits/train_mix_epoch<seed>.jsonl"""
import collections, random, sys
from common import SPLITS, LANGS, MAX_TEXT_TOKENS, read_jsonl, write_jsonl

T_HOURS, REP_CAP, SHARE_CAP, CS_FRAC = 90.0, 3.0, 0.7, 0.10
CS_SPLIT = [("hi", 0.6), ("bn", 0.4)]


def hours(utts):
    return sum(u["dur"] for u in utts) / 3600


def take(utts, h, rng):
    """deterministic: full copies, then a random subset for the remainder"""
    tot = hours(utts)
    out = utts * int(h // tot)
    rem = (h - (h // tot) * tot) * 3600
    pool = utts[:]
    rng.shuffle(pool)
    acc = 0.0
    for u in pool:
        if acc >= rem:
            break
        out.append(u)
        acc += u["dur"]
    return out


def corpus_shares(h):
    """stage 1: hours per corpus -> share of the language's budget"""
    hmax = max(h.values())
    s = {c: h[c] * min(REP_CAP, hmax / h[c]) for c in h}
    share = {c: s[c] / sum(s.values()) for c in s}
    for _ in range(10):                                   # cap dominant corpora, renormalise the rest
        over = {c for c in share if share[c] > SHARE_CAP}
        if not over or len(over) == len(share):          # nothing to cap / a single corpus keeps 100%
            break
        free = 1 - SHARE_CAP * len(over)
        rest = sum(share[c] for c in share if c not in over)
        share = {c: SHARE_CAP if c in over else share[c] / rest * free for c in share}
    return share


def build(train, seed=0, T=T_HOURS):
    rng = random.Random(seed)
    mix, plan = [], []
    for lang in LANGS:
        by = collections.defaultdict(list)
        for u in train:
            if u["lang"] == lang and not u["cs"]:
                by[u["corpus"]].append(u)
        h = {c: hours(v) for c, v in by.items()}
        share = corpus_shares(h)
        for c in by:                                      # stage 2: T hours per language
            target = T * share[c]
            mix += take(by[c], target, rng)
            plan.append((lang, c, round(h[c], 1), round(target, 1), round(target / h[c], 2)))
    cs = collections.defaultdict(list)
    for u in train:
        if u["cs"]:
            cs[u["lang"]].append(u)
    cs_total = CS_FRAC * T * len(LANGS) / (1 - CS_FRAC)
    for lang, frac in CS_SPLIT:
        target = cs_total * frac
        mix += take(cs[lang], target, rng)
        plan.append((lang, "mucs(cs)", round(hours(cs[lang]), 1), round(target, 1), round(target / hours(cs[lang]), 2)))
    rng.shuffle(mix)
    return mix, plan


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train = [u for u in read_jsonl(f"{SPLITS}/train.jsonl") if u["ntok"] <= MAX_TEXT_TOKENS]
    mix, plan = build(train, seed)
    write_jsonl(f"{SPLITS}/train_mix_epoch{seed}.jsonl", mix)
    print("lang corpus avail_h target_h replication")
    for p in plan:
        print(*p)
    by_lang = collections.Counter()
    for u in mix:
        by_lang[u["lang"] + ("-cs" if u["cs"] else "")] += u["dur"] / 3600
    print({k: round(v, 1) for k, v in sorted(by_lang.items())}, "total", round(sum(by_lang.values()), 1), "h,", len(mix), "utts")
