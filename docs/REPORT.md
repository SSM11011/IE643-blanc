# One Whisper model for seven Indian languages with integrated language identification

**IE643 Deep Learning course project — Team blanc** (Shibam Sundar Mahakud, Farhan Al Zayed)
Draft final report, 2026-10-09. All numbers come from files in `reports/`; the scripts that produce them are named in each section.

---

## Abstract

We fine-tune a single Whisper large-v3-turbo model (809 M parameters, LoRA, 87 M trainable) to transcribe Hindi, English, Telugu,
Assamese, Odia, Bengali and Marathi and to identify which of the seven languages is spoken. Whisper has no Odia token, so we add one.
We also add an encoder-side LID-CTC head, two-stage balanced sampling (90 h per language plus 70 h of Hindi/Bengali–English
code-switch), noise/reverb augmentation, and blank-label non-speech clips. On a held-out, speaker-disjoint test split (23 corpus×language
sets, 100 utterances each) mean CER falls from 0.696 (vanilla turbo) to 0.113 with the oracle language token and to 0.126 with
predicted language after two decode-time fixes (0.147 without them). Non-speech hallucination falls from ~100% to ≤1%. The model beats IndicConformer-600M on code-switched speech
(MUCS CER 0.21/0.31 vs 0.50/0.55, also under a transliteration-tolerant metric). IndicConformer remains significantly better on
monolingual in-domain Indic sets (mean 0.104 vs 0.116 on the 20 sets both cover), but it has no English and no LID. Qwen3-ASR-1.7B (Hindi/English only) is about level with us on monolingual Hindi and clearly worse on
Hindi–English code-switch. Fine-tuning broke
Indian-accented English (Svarah): LID fell from 0.99 to 0.13. Two decode-time fixes (ASCII-only English decoding and a two-parameter
calibrated LID rule) recover part of it: CER 0.737 → 0.311 and LID 0.13 → 0.51. Vanilla turbo reaches 0.032, so the full fix needs
Indian-English training data.

---

## 1. Problem and goals

The goal is one model that, given an utterance in any of the seven languages (or a Hindi/Bengali–English code-switched utterance),
(a) outputs the transcript in the right script and (b) reports the language with a confidence, so that no separate LID system has to
pick the recogniser. Design choices came from the literature review in `ie643.md`:
- keep Whisper's language token as a supervised target, so LID does not collapse after fine-tuning;
- add an auxiliary encoder LID-CTC loss (ML-SUPERB 2.0 evidence: +14% relative LID);
- use deterministic balanced upsampling (Polyglot-Lion);
- use no front-end denoiser; use augmentation and Canary/Calm-Whisper-style blank-label non-speech data against hallucination.

Scope decisions (user): final language set is hi, en, te, as, or, bn, mr; Tamil and the Maithili/Konkani/Santali tier were dropped;
~90 h per language rather than full corpora.

## 2. Data

| corpus | languages | role |
|---|---|---|
| IndicVoices (AI4Bharat) | 6 Indic | train / dev / test (spontaneous, hardest) |
| Kathbath | hi, mr, te, or, bn | train / dev / test (read) |
| Shrutilipi | 6 Indic | train only (news, read) |
| FLEURS | all 7 | train / dev / test |
| LibriSpeech (train-clean-100, dev/test-clean) | en | train / dev / test |
| MUCS 2021 | hi–en, bn–en code-switch | train / dev / test |
| Svarah (Indian-accented English), Lahaja (multi-accent Hindi) | en, hi | test only |
| MUSAN noise + music, RIRs | – | augmentation (80% of files) / robustness evaluation (held-out 20%) |

Pipeline (`src/`): `build_manifests.py` decodes everything to 16 kHz mono FLAC (467,515 utterances) and drops clips over 30 s, under
0.3 s, or with empty text. `make_splits.py` builds speaker-disjoint splits and removes training utterances whose text appears in
dev/test. Official IndicVoices/Kathbath "valid" sets share 1,217 and 40 speakers with their train sets; we fixed this. Final split:
384,027 train / 14,883 dev / 45,104 test. `balanced_mix.py` does two-stage upsampling: within a language toward the largest corpus
(replication cap 3×, single-corpus share cap 70%), then exactly 90 h per language per epoch plus 70 h code-switch, giving a 700 h mix.

**Text normalisation** (`normalize.py`, used for both targets and scoring): NFC → IndicNLP normaliser → Whisper-BasicTextNormalizer-style
cleanup (bracketed tags dropped, punctuation → space, lowercase). Digits are *not* verbalised. CER is the primary metric; WER is also
reported.

**Token budget.** Whisper's byte-level BPE costs 17–45 tokens per second of Indic speech (≈3 for English); 41–58% of
Telugu/Assamese/Bengali transcripts exceed 200 tokens. Training keeps utterances with ≤440 text tokens (2.1% dropped); test keeps
everything, so over-budget utterances count as errors.

## 3. Model

- **Base:** Whisper large-v3-turbo (32-layer encoder, 4-layer decoder). Dev experiments used Whisper-small.
- **PEFT:** LoRA r = 32, α = 64 on all attention and MLP projections; bottom third of the encoder frozen; lr 3e-4, 10% warm-up,
  batch 32, 11,000 steps (≈1 epoch of the 700 h mix, 16.6 h on one NVIDIA Thor).
- **New Odia token.** Whisper's 100 languages do not include Odia (checked in `tokenizer.py`, `generation_config.lang_to_id` and the
  vocab). We append `<|or|>` (id 51866, the old turbo vocabulary size), initialise it from `<|bn|>` (same Eastern Indo-Aryan family, closest script), and train only that
  embedding row (gradient mask, separate lr 2.5e-4 chosen from a 3-point sweep, `runs/embchk_*`).
- **Prompt:** `[SOT, <|lang|>, <|transcribe|>, <|notimestamps|>, text…, EOT]`, with the language token as a supervised target.
- **LID-CTC head:** a linear layer on the encoder output predicting, per frame, blank or one of 7 languages; the CTC target is the
  language label once; loss weight λ = 0.3 (from E2). At inference, utterance LID = argmax of frame posteriors summed over valid frames.
- **Robustness:** MUSAN noise/music at 0–20 dB, RIR reverberation, speed ±10% (not for clips ≥27 s), SpecAugment inside the real audio
  length; on average 5% of the samples are non-speech clips with an empty transcript: decoder input `[SOT, <|lang|>, <|transcribe|>, <|notimestamps|>]`
  with a random language, target EOT. The language-token target and the CTC loss are masked for these clips.

## 4. Development experiments (Whisper-small, dev split, single seed; `reports/experiments.md`)

| exp | question | result | decision |
|---|---|---|---|
| E1 | LoRA vs QLoRA vs full FT (Hindi, 1k steps) | LoRA CER 0.095/0.108/0.125 (FLEURS/IndicVoices/Kathbath); QLoRA 0.092/0.114/0.127 at 2.3× the time; full FT 0.101/0.173/0.143 | LoRA |
| E2 | LID-CTC λ ∈ {0, 0.1, 0.3, 1.0} (hi/te/as) | CER unchanged within noise; λ = 1 hurts Assamese IndicVoices (0.358 vs 0.246); head beats token LID on hard sets (hi IndicVoices 0.92 vs 0.83) | λ = 0.3 |
| E3 | balanced vs natural skew (hi 97 h / te 26 h / as 5 h) | Assamese CER 0.242 → 0.182 (FLEURS), 0.342 → 0.278 (IndicVoices); Hindi slightly worse | balance |
| E4 | augmentation + non-speech clips | clean CER about equal; at 0 dB Hindi FLEURS 0.367 → 0.217; non-speech hallucination 0.98–1.00 → 0.02–0.04 | keep both, unweighted |

## 5. Main results (Stage 2, held-out test split)

100 utterances per set, the same utterances for every system (seed 0). 95% bootstrap CIs and paired tests are in `reports/analysis.md`;
a single cell has roughly ±0.01–0.03 CER uncertainty.

### 5.1 ASR (CER, normalised)

| set | vanilla turbo | IndicConformer-600M | **Stage 2, oracle LID** | **Stage 2 + fixes, predicted LID** |
|---|---|---|---|---|
| FLEURS as / bn / hi / mr / or / te | 1.20 / 0.48 / 0.13 / 0.25 / 1.02 / 0.49 | 0.063 / 0.078 / 0.065 / 0.063 / 0.065 / 0.087 | 0.084 / 0.094 / 0.077 / 0.075 / 0.145 / 0.086 | 0.084 / 0.094 / 0.077 / 0.080 / 0.145 / 0.086 |
| IndicVoices as / bn / hi / mr / or / te | 2.60 / 1.19 / 0.32 / 0.72 / 2.05 / 1.12 | 0.055 / 0.061 / 0.068 / 0.060 / 0.104 / 0.081 | 0.164 / 0.119 / 0.107 / 0.124 / 0.196 / 0.203 | 0.186 / 0.143 / 0.120 / 0.140 / 0.223 / 0.209 |
| Kathbath bn / hi / mr / or / te | 0.51 / 0.12 / 0.24 / 1.30 / 0.37 | 0.021 / 0.029 / 0.029 / 0.026 / 0.037 | 0.044 / 0.055 / 0.065 / 0.041 / 0.060 | 0.052 / 0.061 / 0.066 / 0.041 / 0.060 |
| Lahaja hi | 0.259 | 0.038 | 0.064 | 0.147 |
| FLEURS en / LibriSpeech en | 0.036 / 0.014 | – | 0.042 / 0.008 | 0.042 / 0.008 |
| Svarah en (Indian English) | **0.032** | – | 0.227 (0.114 with ASCII-only decoding) | 0.311 |
| MUCS hi–en / bn–en (code-switch) | 0.512 / 1.053 | 0.545 / 0.501 | **0.306 / 0.213** | 0.306 / 0.213 |
| **mean over 23 sets** | 0.696 | (0.104 over its 20 sets) | **0.113** | **0.126** |

**Qwen3-ASR** supports only Hindi and English of our languages. We scored it with the language forced; on every other language it
answers "Hindi" or an unrelated language, and its CER is 0.34–1.0. On the sets it covers:

| set | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B | Stage 2 (oracle) | Stage 2 + fixes (pred.) |
|---|---|---|---|---|
| FLEURS hi / IndicVoices hi / Kathbath hi / Lahaja hi | 0.107 / 0.144 / 0.077 / 0.118 | 0.068 / 0.100 / 0.044 / 0.076 | 0.077 / 0.107 / 0.055 / 0.064 | 0.077 / 0.120 / 0.061 / 0.147 |
| MUCS hi–en (CER / rCER) | 0.417 / 0.307 | 0.482 / 0.306 | 0.306 / 0.251 | 0.306 / 0.251 |
| FLEURS en / LibriSpeech en / Svarah | 0.036 / 0.008 / 0.134 | 0.027 / 0.006 / 0.101 | 0.042 / 0.008 / 0.227 (0.114 ASCII) | 0.042 / 0.008 / 0.311 |

Paired bootstrap against Stage 2 (oracle):
- Qwen3-1.7B is significantly better on Kathbath-hi (−0.012), FLEURS-en (−0.014) and Svarah.
- It is not significantly different on FLEURS-hi, IndicVoices-hi and Lahaja.
- It is significantly worse on MUCS-hi (+0.18).
- Stage 2 is significantly better than Qwen3-0.6B on all Hindi sets.

A 1.7B model trained for Hindi and English is thus about level with our 7-language turbo model on monolingual Hindi, and clearly
behind on Hindi–English code-switch.

Observations:
- Fine-tuning turns an unusable model into a usable one on every Indic set; vanilla CER > 1 means repetition loops.
- **IndicConformer is significantly better on monolingual in-domain AI4Bharat sets** (paired bootstrap CI excludes 0 on 16 of 18 Indic
  monolingual sets; the gap is largest on IndicVoices, 0.04–0.12). Two caveats: it was trained on much more AI4Bharat data, from the
  same sources as these test sets, and it has no English, no LID and no code-switch support. On FLEURS Telugu and Hindi the two are
  not significantly different.
- **Code-switch:** Stage 2 is significantly better on MUCS. Plain CER penalises writing an English word in the other script, so we also
  score with `cs_metric.py` (both sides romanised; rCER) and per-word consonant skeletons (sWER). The ranking holds:
  hi–en rCER 0.251 vs 0.367 (IndicConformer) and 0.366 (vanilla); bn–en 0.173 vs 0.326 and 0.749.
- **Odia** works through the new token (FLEURS 0.145, Kathbath 0.041), but FLEURS Odia is our weakest FLEURS language.

### 5.2 Language identification

Utterance accuracy, pooled over all test sets (rows = true language). The 7×7 confusion matrices are in `reports/analysis.md`.

| true language | vanilla turbo token | ECAPA VoxLingua107 (closed set) | Stage 2 token | Stage 2 LID-CTC head | **Stage 2 calibrated rule** |
|---|---|---|---|---|---|
| hi | 0.84 | 0.90 | 0.91 | 0.91 | 0.91 |
| en | 1.00 | 0.71 | 0.71 | 0.75 | **0.84** |
| te | 0.14 | 0.97 | 0.98 | 0.98 | 0.98 |
| as | 0.00 | 0.92 | 0.82 | 0.92 | 0.89 |
| or | no token (0.00) | no class | 0.92 | 0.90 | 0.92 |
| bn | 0.42 | 0.81 | 0.94 | 0.94 | 0.94 |
| mr | 0.52 | 0.89 | 0.94 | 0.93 | 0.94 |

- FLEURS, Kathbath, LibriSpeech and MUCS sets are at 0.98–1.00. Errors concentrate in spontaneous IndicVoices (0.76–0.94),
  Lahaja-hi (0.78) and Svarah.
- Excluding Svarah, English recall is 1.00. All of the English errors come from Indian-accented English.
- `<|or|>` absorbs some errors (17 hi, 22 as utterances by token LID). It is the only trainable output row, which may bias it upward
  (hypothesis, not tested).
- The **oracle vs predicted gap** (routing cost) is 0.113 → 0.126 mean CER with the calibrated rule. Where LID is weak the gap is
  large: Lahaja 0.064 → 0.147, IndicVoices-as 0.164 → 0.186.

### 5.3 Robustness and hallucination (`reports/experiments.md`)

- Non-speech hallucination (more than 5 tokens on held-out MUSAN noise/music, measured with `begin_suppress_tokens` disabled): 0.00–0.01
  in all 7 languages (vanilla ≈ 1.0); false silence on real speech 0.00.
- SNR sweep (held-out noise): FLEURS-hi CER 0.056 clean → 0.085 at 5 dB → 0.130 at 0 dB; reverberation 0.064. Degradation is gradual
  in all 8 sets tested.

## 6. The Indian-English regression and the decode-time fixes

Vanilla turbo transcribes Svarah at CER 0.032 and identifies it as English 99% of the time. After fine-tuning, LID is 0.13 and even
with `<|en|>` forced CER is 0.227. Per-utterance analysis of the 100 test hypotheses:

| subset of Svarah | n | Stage 2 CER | vanilla CER |
|---|---|---|---|
| hypothesis contains non-Latin script (English written in Bengali/Telugu/Devanagari script) | 39 | 0.715 | 0.059 |
| reference contains digits (Stage 2 spells numbers out, as LibriSpeech does; 6 of these 12 are also in the row above) | 12 | 0.585 | 0.040 |
| rest | 55 | 0.048 | 0.023 |

Cause: the training mix has no Indian-accented English. Accented English acoustically resembles the Indic training data, so the model
maps it to Indic languages and scripts. ECAPA fails in the same way (0.14).

Two fixes that need no retraining (`eval_whisper.py --en_ascii --lid_rule`, also used in the demo):
1. **ASCII-only English decoding:** when the language is English, suppress the 11,707 vocabulary tokens that are not pure ASCII.
   Svarah oracle CER 0.227 → 0.114. No other English or MUCS set changes.
2. **Calibrated LID rule** (`lid_calib.py`): score(l) = log p_token(l) + b·[l = en] + w·log p_head(l). The two parameters were fitted on the
   dev split plus a **speaker-disjoint** Svarah calibration set (`make_svarah_calib.py`: 300 utterances from 54 speaker profiles that do
   not occur in the scored 100). Result: b = 22.25, w = 2.0; calibration macro LID 0.900 → 0.919. On test: Svarah LID 0.13 → 0.51
   (CER 0.737 → 0.311), IndicVoices-as 0.66 → 0.77; no set loses more than 0.02.

Svarah is still far behind vanilla (0.311 vs 0.032). A real fix needs Indian-accented English in training (for example Svarah-like
data or NPTEL lectures), or a rehearsal/distillation term that keeps the original English behaviour.

## 7. Demo

`src/demo.py` (Gradio): upload or record audio. It shows the detected language with closed-set confidences and the encoder-head
opinion, then the transcript; the language can be forced. Long audio is cut into ≤12 s chunks at Silero-VAD pauses. Two reasons:
on a packed 30-s window the model skipped a whole sentence and the LID head became uniform (it was trained on single utterances), and
25-s Telugu chunks overflowed the 440-token budget. On 50–70 s concatenations of FLEURS utterances the chunked demo gives CER 0.05
(as), 0.15 (hi) and 0.18 (te).

## 8. Limitations

- n = 100 per test set and a single training seed; we report CIs, but seed variance is not measured.
- The training mix is one epoch of one deterministic mix; IndicVoices is sub-sampled to 0.5–0.7× per epoch.
- IndicConformer's advantage may be partly in-domain; FLEURS (out-of-domain for both) shows a smaller gap.
- MUCS references themselves mix scripts inconsistently; rCER/sWER are loose comparison metrics, not absolute accuracies.
- Digits vs number words are not normalised; this penalises Stage 2 on Svarah (12 utterances) and possibly elsewhere.
- Robustness test noise comes from the same MUSAN categories as training noise (held-out files only); there is no UrbanSound/DEMAND test.
- Not done: synthetic Hinglish, the words-per-second truncation guard, Voice of India / Vistaar benchmarks (not available).

## 9. Verification and bugs found along the way

Every number above was produced by scripts in `src/`. Verification scripts are in `src/verify/`; the full history (25 bugs and their
fixes) is in `docs/PROJECT_LOG.md` §9. The ones that would have changed conclusions:
- non-speaker-disjoint official splits;
- a 200-token generation cap that truncated Indic outputs;
- a hallucination metric that could not drop below ~100% because Whisper suppresses EOT as the first token;
- 92 "RIRs" that were noise recordings;
- `convert_tokens_to_ids("<|or|>")` silently returning EOT.

Re-running the test evaluation after the final code cleanup reproduced all 23 saved Stage-2 results and all IndicConformer results
exactly. Vanilla turbo differs on one set (FLEURS-en, by 0.0001 CER, from a later apostrophe-normalisation fix).

## 10. Reproduction

See `README.md`. Main artefacts: `runs/s2_turbo/adapter` (LoRA + Odia row), `runs/s2_turbo/lid_head.pt`, `runs/s2_turbo/lid_rule.json`;
results in `reports/experiments.md`, `reports/analysis.md`, `reports/baselines/summary.md`; per-utterance hypotheses in `reports/hyps/`.
