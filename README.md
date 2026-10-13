# One Whisper Model for Seven Indian Languages, with Built-in Language Identification

**IE643 Deep Learning — course project, Team blanc** (Shibam Sundar Mahakud, Farhan Al Zayed)

We fine-tuned a **single Whisper large-v3-turbo model** to transcribe speech in **Hindi, English, Telugu, Assamese, Odia,
Bengali and Marathi**, including **Hindi–English and Bengali–English code-switched speech**. The same model also reports **which
language is being spoken**, with a confidence score, so no separate language-ID system has to choose a recogniser.

| | vanilla Whisper turbo | **our model (Stage 2)** |
|---|---|---|
| Mean CER over 23 held-out test sets (language given) | 0.696 | **0.113** |
| Mean CER, language predicted by the model (with decode-time fixes) | – | **0.126** |
| Odia support | none (no language token) | **yes**, via a new `<\|or\|>` token |
| Hallucination on pure noise/music | ≈ 100% | **0–1%** |
| Code-switch CER, MUCS hi–en / bn–en | 0.51 / 1.05 | **0.31 / 0.21** (IndicConformer: 0.55 / 0.50) |
| Language-ID accuracy (pooled; hi / te / as / bn / mr / or) | 0.84 / 0.14 / 0.00 / 0.42 / 0.52 / n.a. | **0.91 / 0.98 / 0.89 / 0.94 / 0.94 / 0.92** |

The full write-up is in [`docs/REPORT.md`](docs/REPORT.md). The chronological log of every decision, bug and fix is in
[`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md). A 68-page explanatory PDF is [`IE643_Project_Complete_Notes.pdf`](IE643_Project_Complete_Notes.pdf)
(LaTeX sources in `docs/project_notes/`).

---

## Contents
1. [Problem and approach](#1-problem-and-approach)
2. [Data](#2-data)
3. [Model and training](#3-model-and-training)
4. [What we tried: development experiments E1–E4](#4-what-we-tried-development-experiments-e1e4)
5. [Final results (Stage 2, held-out test split)](#5-final-results-stage-2-held-out-test-split)
6. [The Indian-English regression and how we partly fixed it](#6-the-indian-english-regression-and-how-we-partly-fixed-it)
7. [Demo](#7-demo)
8. [Verification, and bugs we found along the way](#8-verification-and-bugs-we-found-along-the-way)
9. [Limitations and what is not done](#9-limitations-and-what-is-not-done)
10. [Repository layout and code map](#10-repository-layout-and-code-map)
11. [Reproducing](#11-reproducing)

---

## 1. Problem and approach

**Goal:** given an utterance in any of the seven languages, or a Hindi/Bengali–English code-switched utterance, one model should
(a) output the transcript in the correct script and (b) report the language with a confidence.

**Starting point.** Out of the box, Whisper is usable only for English and Hindi on these languages (zero-shot test CER, `reports/baselines/summary.md`):

| language | Whisper-small | Whisper-turbo | IndicConformer-600M | turbo LID accuracy |
|---|---|---|---|---|
| English | 0.04 | 0.03 | n/a | 100% |
| Hindi | 1.09 | 0.21 | 0.05 | 81% |
| Marathi | 1.13 | 0.40 | 0.05 | 52% |
| Bengali | 1.68 | 0.72 | 0.05 | 51% |
| Telugu | 2.11 | 0.66 | 0.07 | 14% |
| Assamese | 1.58 | 1.90 | 0.06 | 0% |
| Odia | 1.66 | 1.46 | 0.07 | **no Odia token in Whisper** |

A CER above 1 means the model falls into repetition loops. The AI4Bharat IndicConformer is far stronger on Indic speech, but it has
no English, no code-switch support and no language identification.

**Design choices.** These came from the literature review in [`ie643.md`](ie643.md) and the prep slides in `blanc_IE643_CourseProject_Prep.pdf`:
- Keep Whisper's language token as a **supervised decoder target**, so that LID does not collapse after fine-tuning.
- Add an **auxiliary encoder LID-CTC head**. ML-SUPERB 2.0 reports about +14% relative LID with this kind of head.
- Use **deterministic balanced upsampling** across languages and corpora, as in Polyglot-Lion.
- Use no front-end denoiser. Instead, use **noise/reverb augmentation** and **blank-label non-speech clips** (the Canary and Calm-Whisper idea) against hallucination.
- Use **LoRA** rather than full fine-tuning, chosen by experiment E1.

**Scope decisions.** Tamil and a Maithili/Konkani/Santali low-resource tier were in the original slides and were dropped. We used about
90 h per language per epoch instead of the full corpora, to fit the disk and compute budget.

---

## 2. Data

| corpus | languages | role |
|---|---|---|
| IndicVoices (AI4Bharat) | 6 Indic | train / dev / test (spontaneous, hardest) |
| Kathbath | hi, mr, te, or, bn (no Assamese) | train / dev / test (read speech) |
| Shrutilipi | 6 Indic | train only (read news) |
| FLEURS | all 7 | train / dev / test |
| LibriSpeech (train-clean-100, dev/test-clean) | en | train / dev / test |
| MUCS 2021 | hi–en, bn–en code-switch | train / dev / test |
| Svarah (Indian-accented English), Lahaja (multi-accent Hindi) | en, hi | test only |
| MUSAN noise + music, OpenSLR RIRs | – | augmentation (80% of files) / robustness tests (the other 20%) |

**Pipeline** (`src/`):
1. `audit_data.py` produces `reports/data_audit.csv` with hours, utterances and speakers per corpus, language and split.
2. `build_manifests.py` decodes everything to 16 kHz mono FLAC and writes JSONL manifests. It drops clips over 30 s, clips under
   0.3 s and clips with empty text. **467,515 utterances** remain.
3. `make_splits.py` builds **speaker-disjoint** splits and removes training utterances whose text appears in dev/test. The official
   IndicVoices and Kathbath "valid" sets share **1,217 and 40 speakers** with their train sets; we fixed this. Final split:
   **384,027 train / 14,883 dev / 45,104 test**.
4. `prep_tokens.py` adds the Whisper token count of each transcript.
5. `balanced_mix.py` does two-stage upsampling. First, within a language it upsamples toward the largest corpus (replication cap 3×,
   single-corpus share cap 70%). Then it takes **exactly 90 h per language per epoch plus 70 h of code-switch**, giving a **700 h** mix
   of 349,592 utterances.

**Text normalisation** (`normalize.py`) is one function, used for both training targets and scoring: NFC, then the IndicNLP normaliser,
then a Whisper-BasicTextNormalizer-style cleanup (bracketed tags dropped, punctuation turned into spaces, lowercase). **CER is the
primary metric**; WER is also reported.

**Token budget.** Whisper's byte-level BPE needs 17–45 tokens per second of Indic speech, against about 3 for English. 41–58% of
Telugu, Assamese and Bengali transcripts exceed 200 tokens. Training keeps utterances with ≤ 440 tokens, which drops 2.1%. The test set
keeps everything, so over-budget utterances count as errors.

---

## 3. Model and training

- **Base model:** Whisper large-v3-turbo (809 M parameters; 32-layer encoder, 4-layer decoder). Development experiments used Whisper-small.
- **LoRA:** r = 32, α = 64 on all attention and MLP projections (87 M trainable parameters). The bottom third of the encoder is frozen.
  lr 3e-4, 10% warm-up, batch 32, **11,000 steps** (about one epoch of the 700 h mix). The run took **16.6 h on one NVIDIA Thor**.
- **New Odia token.** Whisper's 100 languages do not include Odia. We checked this in `tokenizer.py`, in `generation_config.lang_to_id`
  and in the vocabulary. We append `<|or|>` (id 51866), initialise it from `<|bn|>` (the same Eastern Indo-Aryan family and the closest
  script), and train only that embedding row through a gradient mask. Its learning rate is 2.5e-4, chosen by a 3-point sweep
  (`runs/embchk_*`: Odia LID 0.74 / 0.88 / 0.90 for 5e-5 / 2.5e-4 / 1.5e-3).
- **Prompt:** `[SOT, <|lang|>, <|transcribe|>, <|notimestamps|>, text…, EOT]`; the language token is a training target.
- **LID-CTC head:** a linear layer on the encoder output. For each frame it predicts blank or one of the 7 languages, with loss weight
  λ = 0.3. At inference, the utterance's language is the argmax of the frame posteriors summed over valid frames.
- **Robustness training:**
  - MUSAN noise/music at 0–20 dB SNR, RIR reverberation, speed ±10%, and SpecAugment applied inside the real audio length.
  - About 5% of samples are **non-speech clips with an empty transcript** (target = EOT). Their language-token loss and CTC loss are masked.

---

## 4. What we tried: development experiments E1–E4

These runs used Whisper-small, the dev split and a single seed. Full tables are in [`reports/experiments.md`](reports/experiments.md).

| exp | question | what we found | decision |
|---|---|---|---|
| **E1** | LoRA vs QLoRA vs full fine-tuning (Hindi, 1k steps) | CER on FLEURS / IndicVoices / Kathbath: LoRA 0.095 / 0.108 / 0.125 (32 min); QLoRA 0.092 / 0.114 / 0.127 (76 min, 2.3× slower); full FT 0.101 / 0.173 / 0.143 (worse out of domain). The untrained model scored 0.49 / 1.39 / 0.72. | **LoRA** |
| **E2** | LID-CTC weight λ ∈ {0, 0.1, 0.3, 1.0} (hi/te/as) | CER unchanged within noise; λ = 1 hurts Assamese IndicVoices (0.358 vs 0.246). The head beats the token's LID on hard sets (Hindi IndicVoices 0.92 vs 0.83; Assamese IndicVoices 0.73 → 0.81–0.84). | **λ = 0.3** |
| **E3** | Balanced vs natural skew (hi 97 h / te 26 h / as 5 h) | Assamese CER 0.242 → 0.182 (FLEURS) and 0.342 → 0.278 (IndicVoices); Hindi slightly worse (0.096 → 0.111). | **balance** |
| **E4** | Augmentation + non-speech clips | Clean CER about equal. At 0 dB SNR, Hindi FLEURS 0.367 → 0.217. Non-speech hallucination 0.98–1.00 → **0.02–0.04**; false silence on real speech 0.00–0.01. | **keep both**, no extra loss weight |

Two smaller checks also fed into the final configuration:
- The **Odia embedding learning-rate sweep** described in §3.
- A **`torch.compile` throughput test**: 12.7 → ~23 utt/s in a benchmark, 15–17 in real runs.

---

## 5. Final results (Stage 2, held-out test split)

We used 100 utterances per set, and every system was scored on the same utterances. 95% bootstrap CIs and paired tests are in
[`reports/analysis.md`](reports/analysis.md). One cell carries roughly ±0.01–0.03 CER of uncertainty.

### 5.1 Speech recognition (CER)

| set | vanilla turbo | IndicConformer-600M | **Stage 2, oracle language** | **Stage 2 + fixes, predicted language** |
|---|---|---|---|---|
| FLEURS as / bn / hi / mr / or / te | 1.20 / 0.48 / 0.13 / 0.25 / 1.02 / 0.49 | 0.063 / 0.078 / 0.065 / 0.063 / 0.065 / 0.087 | 0.084 / 0.094 / 0.077 / 0.075 / 0.145 / 0.086 | 0.084 / 0.094 / 0.077 / 0.080 / 0.145 / 0.086 |
| IndicVoices as / bn / hi / mr / or / te | 2.60 / 1.19 / 0.32 / 0.72 / 2.05 / 1.12 | 0.055 / 0.061 / 0.068 / 0.060 / 0.104 / 0.081 | 0.164 / 0.119 / 0.107 / 0.124 / 0.196 / 0.203 | 0.186 / 0.143 / 0.120 / 0.140 / 0.223 / 0.209 |
| Kathbath bn / hi / mr / or / te | 0.51 / 0.12 / 0.24 / 1.30 / 0.37 | 0.021 / 0.029 / 0.029 / 0.026 / 0.037 | 0.044 / 0.055 / 0.065 / 0.041 / 0.060 | 0.052 / 0.061 / 0.066 / 0.041 / 0.060 |
| Lahaja hi | 0.259 | 0.038 | 0.064 | 0.147 |
| FLEURS en / LibriSpeech en | 0.036 / 0.014 | – | 0.042 / 0.008 | 0.042 / 0.008 |
| Svarah en (Indian English) | **0.032** | – | 0.227 (0.114 ASCII-only) | 0.311 |
| MUCS hi–en / bn–en (code-switch) | 0.512 / 1.053 | 0.545 / 0.501 | **0.306 / 0.213** | 0.306 / 0.213 |
| **mean over 23 sets** | 0.696 | (0.104 over its 20 sets) | **0.113** | **0.126** |

What this shows:
- **Fine-tuning makes every Indic set usable.** For example, FLEURS Odia goes 1.02 → 0.145, IndicVoices Assamese 2.60 → 0.164 and Kathbath Odia 1.30 → 0.041.
- **Odia works through the new token,** although FLEURS Odia is still our weakest FLEURS language.
- **Code-switch is our strongest result.** We are significantly better than IndicConformer and vanilla Whisper on both MUCS sets. Plain CER
  penalises writing an English word in the other script, so we also scored with a transliteration-tolerant metric (`src/cs_metric.py`:
  both sides romanised to get rCER, plus a consonant-skeleton WER). The ranking holds: hi–en rCER **0.251** vs 0.367 (IndicConformer)
  and 0.366 (vanilla); bn–en **0.173** vs 0.326 and 0.749.
- **IndicConformer is still better on monolingual, in-domain AI4Bharat speech.** Its paired-bootstrap CI excludes zero on 17 of 18
  monolingual Indic sets; FLEURS-Telugu is a tie and FLEURS-Hindi is barely significant. The gap is largest on IndicVoices (0.04–0.12).
  Two caveats: IndicConformer was trained on far more AI4Bharat data from the same sources as these test sets, and it has no English,
  no LID and no code-switch support.

**Qwen3-ASR** covers only Hindi and English among our languages. On other languages it answers "Hindi" or an unrelated language
(CER 0.34–1.0). We compared it on the sets it supports, with the language forced:

| set | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B | Stage 2 (oracle) | Stage 2 + fixes (pred.) |
|---|---|---|---|---|
| FLEURS hi / IndicVoices hi / Kathbath hi / Lahaja hi | 0.107 / 0.144 / 0.077 / 0.118 | 0.068 / 0.100 / 0.044 / 0.076 | 0.077 / 0.107 / 0.055 / 0.064 | 0.077 / 0.120 / 0.061 / 0.147 |
| MUCS hi–en (CER / rCER) | 0.417 / 0.307 | 0.482 / 0.306 | **0.306 / 0.251** | 0.306 / 0.251 |
| FLEURS en / LibriSpeech en / Svarah | 0.036 / 0.008 / 0.134 | 0.027 / 0.006 / 0.101 | 0.042 / 0.008 / 0.227 | 0.042 / 0.008 / 0.311 |

Our 7-language model is about level with the Hindi/English-only Qwen3-ASR-1.7B on monolingual Hindi, with no significant difference on
FLEURS, IndicVoices and Lahaja. It is significantly better on Hindi–English code-switch, and significantly better than Qwen3-0.6B on
every Hindi set.

### 5.2 Language identification

Utterance accuracy is pooled over all test sets; rows are the true language. The 7×7 confusion matrices are in `reports/analysis.md`.

| true language | vanilla turbo token | ECAPA VoxLingua107 (closed set) | Stage 2 token | Stage 2 LID-CTC head | **Stage 2 calibrated rule** |
|---|---|---|---|---|---|
| hi | 0.84 | 0.90 | 0.91 | 0.91 | 0.91 |
| en | 1.00 | 0.71 | 0.71 | 0.75 | **0.84** |
| te | 0.14 | 0.97 | 0.98 | 0.98 | 0.98 |
| as | 0.00 | 0.92 | 0.82 | 0.92 | 0.89 |
| or | no token | no class | 0.92 | 0.90 | 0.92 |
| bn | 0.42 | 0.81 | 0.94 | 0.94 | 0.94 |
| mr | 0.52 | 0.89 | 0.94 | 0.93 | 0.94 |

- LID on FLEURS, Kathbath, LibriSpeech and MUCS is 0.98–1.00.
- The errors concentrate in spontaneous IndicVoices (0.76–0.94), Lahaja (0.78) and Svarah. Without Svarah, English recall is 1.00.
- **Routing cost** (oracle → predicted language): mean CER 0.113 → 0.126. The cost is large only where LID is weak, e.g. Lahaja 0.064 → 0.147.
- `<|or|>` absorbs some errors from Hindi and Assamese. It is the only trainable output row, which may bias it upward; this is a hypothesis we did not test.

### 5.3 Robustness and hallucination

- **Non-speech hallucination** is the share of outputs with more than 5 tokens on held-out MUSAN noise/music. It is **0.00–0.01 in all
  7 languages**, against about 1.0 for vanilla Whisper. False silence on real speech is 0.00.
- **SNR sweep** on held-out noise files: FLEURS-hi CER 0.056 clean → 0.085 at 5 dB → 0.130 at 0 dB; 0.064 with reverberation.
  Degradation is gradual on all 8 sets tested.

---

## 6. The Indian-English regression and how we partly fixed it

Vanilla turbo transcribes **Svarah** (Indian-accented English) at CER 0.032 and identifies it as English 99% of the time. After
fine-tuning, LID drops to **0.13**, and even with `<|en|>` forced, CER is 0.227. A per-utterance analysis explains why:

| subset of the 100 Svarah test utterances | n | Stage 2 CER | vanilla CER |
|---|---|---|---|
| hypothesis written in a non-Latin script (English in Bengali/Telugu/Devanagari) | 39 | 0.715 | 0.059 |
| reference contains digits (our model spells numbers out, LibriSpeech-style) | 12 | 0.585 | 0.040 |
| rest | 55 | 0.048 | 0.023 |

**Cause:** the training mix had **no Indian-accented English**. Accented English sounds like the Indic training data, so the model maps
it to Indic languages and scripts. ECAPA fails in the same way (0.14).

**Two decode-time fixes, with no retraining** (`eval_whisper.py --en_ascii --lid_rule`, also used in the demo):
1. **ASCII-only English decoding.** When the language is English, the 11,707 non-ASCII vocabulary tokens are suppressed. Svarah oracle
   CER goes 0.227 → **0.114**; no other set changes.
2. **Calibrated LID rule** (`lid_calib.py`): `score(l) = log p_token(l) + b·[l = en] + w·log p_head(l)`. The rule was fitted on the dev
   split plus a speaker-disjoint Svarah calibration set (`make_svarah_calib.py`), giving b = 22.25 and w = 2.0. On the test split,
   Svarah LID goes 0.13 → **0.51** and its CER 0.737 → **0.311**; IndicVoices-Assamese LID goes 0.66 → 0.77. No set loses more than 0.02 LID.

Svarah is still far behind vanilla Whisper (0.311 vs 0.032). **A real fix needs Indian-accented English in the training data**, or a
rehearsal or distillation term that preserves the original English behaviour. This means about one more day of training.

---

## 7. Demo

`src/demo.py` is a Gradio app. You upload or record audio, and it shows:
- the detected language, with closed-set confidences and the encoder head's opinion;
- the transcript.

You can also force the language. Long audio is cut into ≤ 12 s chunks at Silero-VAD pauses. We added this because on a packed 30 s
window the model skipped a whole sentence and the LID head became uniform, and 25 s Telugu chunks overflowed the token budget. On
50–70 s recordings, the chunked demo reaches CER 0.05 (as), 0.15 (hi) and 0.18 (te).

```bash
cd src && python demo.py --port 7860     # http://<host>:7860
```

---

## 8. Verification, and bugs we found along the way

Every number above was produced by a script in `src/`. The sanity-check scripts in `src/verify/` test:
- data integrity, speaker-disjointness and leakage;
- Whisper settings and the metric against an independent edit distance;
- label/audio alignment, using IndicConformer as a checker;
- token lengths, download sizes, and manual generation against HF `generate` (48/48 identical).

We did four full code reviews. The project log lists **26 bugs and their fixes**. The ones that would have changed conclusions:
- **Official splits were not speaker-disjoint** (1,217 shared speakers in IndicVoices).
- **A 200-token generation cap** silently truncated Indic outputs in the first baseline run.
- **The hallucination metric could never drop below ~100%,** because Whisper suppresses EOT as the first generated token by default
  (`begin_suppress_tokens`). We first misdiagnosed this as a weak training signal; the project log records the wrong turn.
- **92 of the "RIRs" were 30 s noise recordings,** which contaminated the reverberation augmentation.
- **`tokenizer.convert_tokens_to_ids("<|or|>")` silently returns EOT,** which would have broken every Odia evaluation. `load_ft.LANGID`
  is the correct lookup.
- Non-speech clips taught a **random language label**, which was masked before Stage 2.

After the final code clean-up, re-running the test evaluation reproduced **all 23 saved Stage 2 results and all IndicConformer results
exactly**.

---

## 9. Limitations and what is not done

- n = 100 per test set and one training seed. We report CIs, but seed variance is not measured.
- The model saw one epoch of one deterministic mix; IndicVoices is sub-sampled to 0.5–0.7× per epoch.
- IndicConformer's lead may be partly in-domain advantage; on FLEURS, which is out of domain for both models, the gap is smaller.
- MUCS references mix scripts inconsistently, so rCER and sWER are comparison metrics, not absolute accuracies.
- Digits vs number words are not normalised. This penalises us on Svarah and possibly elsewhere.
- `normalize()` drops text inside parentheses. In FLEURS that text is usually spoken, so absolute FLEURS CERs are inflated by
  0.005–0.02 for **every** system. Rankings are mostly unchanged; this is open and not fixed.
- Robustness test noise is held-out files from the same MUSAN categories used in training; there is no UrbanSound or DEMAND test.
- **Not done:**
  - Indian-English retraining.
  - Synthetic Hinglish.
  - A words-per-second truncation guard (loop rate is already about 0).
  - Multiple seeds.
  - Voice of India and Vistaar benchmarks (not available to us).

---

## 10. Repository layout and code map

```
src/              data pipeline, training, evaluation, analysis, demo
src/verify/       audit and sanity-check scripts
reports/          all results: experiments.md, analysis.md(+json), baselines/, test_*/snr_*/halluc_*.json, hyps/ (per-utterance outputs)
runs/<name>/      result.json / result_partial.json per training run, adapter_config.json, lid_rule.json   (weights not in git)
logs/             stdout of every long job
docs/             REPORT.md (final report), PROJECT_LOG.md (full history), project_notes/ (LaTeX notes)
ie643.md          literature review and original plan
download_open.sh, download_gated.py    dataset / model download scripts
```

The repository has no **datasets** (~150 GB), **model weights** or **training checkpoints**. `data/`, `models/`, `repos/` and the
weight files in `runs/` are ignored by git. Use the download scripts to fetch the data and models.

| file | role |
|---|---|
| `common.py` | paths, language list, token budget, shared helpers (special tokens, noise clips, LID-head scoring) |
| `normalize.py`, `metrics.py` | the single text normalisation for targets and scoring; WER/CER |
| `audit_data.py` → `build_manifests.py` → `make_splits.py` → `prep_tokens.py` → `balanced_mix.py` | data pipeline |
| `augment.py` | MUSAN noise / RIR / speed / SpecAugment; 80/20 train/eval noise-file split |
| `train_whisper.py` | fine-tuning (full / LoRA / QLoRA, `<\|or\|>` token, LID-CTC head, non-speech clips, checkpoint/resume) |
| `load_ft.py` | load vanilla / adapter / fully fine-tuned model (handles the Odia token) |
| `eval_whisper.py`, `eval_snr.py`, `eval_halluc.py` | ASR + LID, SNR sweep, hallucination evaluation |
| `eval_indicconformer.py` (+ `ic_gpu.py`), `eval_qwen3asr.py`, `eval_ecapa.py` | baselines |
| `make_svarah_calib.py`, `lid_calib.py` | speaker-disjoint Svarah calibration set; calibrated LID rule |
| `analysis.py`, `cs_metric.py` | bootstrap CIs, paired tests, transliteration-tolerant code-switch metric, LID confusion |
| `demo.py` | Gradio demo with VAD chunking |
| `summarize.py`, `baseline_table.py` | regenerate `reports/experiments.md`, `reports/baselines/summary.md` |
| `run_overnight.sh`, `run_stage2.sh`, `run_baselines2.sh`, `run_qwen.sh` | the exact command chains for E1–E4, Stage 2 and the baselines |
| `make_e3.py`, `fix_ids.py`, `renorm_en.py` | E3 manifests; one-off patches (already applied) |

---

## 11. Reproducing

**Environment.** We used Python 3.12 on an NVIDIA Thor (aarch64, CUDA 13) with torch 2.14.1, transformers 5.18, peft 0.21.2,
bitsandbytes 0.50.2, jiwer 4.0, indic-nlp-library 0.92, indic_transliteration 2.3.82, onnx2torch 1.5.15, soundfile and gradio 6.29.
The Qwen3-ASR and ECAPA baselines need `qwen-asr`, which pins transformers 4.57, so they run in a separate venv layered on the main one.

```bash
bash download_open.sh && python download_gated.py      # models + datasets (gated HF sets need a logged-in account)
cd src
python audit_data.py
python build_manifests.py                               # decode + manifests (~40 min)
python make_splits.py && python prep_tokens.py && python balanced_mix.py 0

# baselines
python eval_whisper.py --model ../models/whisper-large-v3-turbo --split test --n 100 --out ../reports/baselines/x.json
PYTHONPATH= python eval_indicconformer.py --n 100 --out ../reports/baselines/indicconformer_test.json

# development experiments E1–E4 (Whisper-small) -> reports/experiments.md
bash run_overnight.sh

# Stage 2: turbo + LoRA, all 7 languages (~17 h; re-running resumes from the last checkpoint)
bash run_stage2.sh

# decode-time fixes for Indian English
python make_svarah_calib.py && python lid_calib.py
python eval_whisper.py --model ../models/whisper-large-v3-turbo --adapter ../runs/s2_turbo/adapter --split test --n 100 \
  --en_ascii --lid_rule ../runs/s2_turbo/lid_rule.json --out ../reports/test_s2_turbo_fix.json --hyps ../reports/hyps/s2_turbo_fix.jsonl

python analysis.py && python summarize.py               # reports/analysis.md, reports/experiments.md
```

**Things to know:**
- E1–E4 are dev-set, single-seed, Whisper-small. Stage 2 is the only fine-tuned model evaluated on the test split.
- Never use `tok.convert_tokens_to_ids("<|or|>")`, because it returns EOT. Use `LANGID` from `load_ft.load`.
- Measure hallucination with `begin_suppress_tokens` disabled (`eval_halluc.py`). The `hallucination_rate` fields in `reports/snr_*.json` are invalid.
- Don't run two Whisper trainings at once on a unified-memory machine; memory thrashes.
