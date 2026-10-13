# IE643 — Team blanc — Project Log

Multilingual speech recognition with integrated language identification using a single Whisper model.
Plan source: `ie643.md` (literature review) and `blanc_IE643_CourseProject_Prep.pdf` (presentation).
Log covers work from 2026-10-02 to 2026-10-09. Everything below was executed on this machine (NVIDIA Thor, aarch64, 112 GB unified memory, CUDA 13).

---------------------------------------------------------------------

## 1. Scope decisions (made by the user, override the slides)

| Decision | Detail |
|---|---|
| Final languages (7) | Hindi, English, Telugu, Assamese, Odia, Bengali, Marathi |
| Dropped | Tamil, and the Maithili / Konkani / Santali low-resource tier |
| Data volume | Balanced subset, ~90 h per Indic language (not full corpora) |
| English | LibriSpeech `train-clean-100` added to balance (user asked for it) |
| Checks before proceeding | User asked for a full verification of code and baselines before the training stage |

Consequence worth remembering: with this language set nothing is truly "low-resource" by hours in IndicVoices
(Assamese ~800 h and Odia ~740 h exist in the full corpus, comparable to Hindi ~880 h). The imbalance is in the read-speech corpora
(Shrutilipi: Hindi ~1,380 h vs Assamese ~87 h; Kathbath has no Assamese). Hour figures for full corpora are *estimates* derived from bytes-per-hour of the downloaded shards.

---------------------------------------------------------------------

## 2. Environment

- Python venv: `/home/shibam/IE643/.venv` (torch 2.14.1+cu130, transformers 5.18, peft 0.21.2, bitsandbytes 0.50.2, onnxruntime, onnx2torch, jiwer, indic-nlp-library).
- `.ortgpu/` holds an onnxruntime-gpu wheel that loads but has **no CUDA kernels for this GPU** (see bug 6); it is unused.
- Hugging Face login done with the user's token (stored by the `hf` CLI). GitHub token was not needed.
- **Security:** both tokens were pasted into the chat in plain text. Rotate them.

---------------------------------------------------------------------

## 3. Downloads (all verified size-exact against the Hub listing, 0 mismatches over 13 repos)

| Item | Where | Notes |
|---|---|---|
| Whisper large-v3-turbo, Whisper-small | `models/` | Final base / dev model |
| Qwen3-ASR 0.6B / 1.7B | `models/` | Baselines — evaluated 2026-10-09 (supports only hi/en of our languages), see 9e |
| SpeechBrain VoxLingua107 ECAPA | `models/` | External LID baseline — evaluated 2026-10-09 (no Odia class), see 9e |
| paza-whisper-large-v3-turbo | `models/` | For the words-per-second truncation trick — not yet used |
| IndicConformer-600M (ONNX) | `models/` | Evaluated as baseline |
| IndicVoices | `data/indicvoices` | 6 Indic langs: all valid shards + 6 train shards each (no English in this corpus) |
| Kathbath | `data/kathbath` | hi/mr/te/or/bn (no Assamese exists): valid + 3 train shards |
| Shrutilipi | `data/shrutilipi` | 6 Indic langs, 3 train shards each |
| Svarah, Lahaja | `data/svarah`, `data/lahaja` | Test sets (Indian-accented English; multi-accent Hindi) |
| FLEURS | `data/fleurs` | en, hi, mr, bn, as, or, te (Tamil removed) |
| MUCS 2021 | `data/mucs` | Hindi-English and Bengali-English only, extracted into separate folders |
| MUSAN (noise + music only), RIRs | `data/noise` | MUSAN `speech/` dropped to save space |
| LibriSpeech train-clean-100, dev-clean, test-clean | `data/librispeech` | English |
| Code repos | `repos/` | IndicConformerASR, vistaar, Svarah, silero-vad, peft, whisper, polyglot-lion |

**Not obtained:** IndicVoices-R, Vistaar benchmark data (not on HF; comes via repo scripts), Voice of India (closed), DEMAND / UrbanSound / AudioSet non-speech clips, IITG-HingCoS text, Common Voice, MUCS subtask-1 (Marathi/Odia/Telugu/Hindi monolingual).
Space note: deleting the audio tarballs saves nothing (audio does not compress); the only real saving came from dropping MUSAN speech.

---------------------------------------------------------------------

## 4. Data pipeline (code in `src/`)

1. `audit_data.py` -> `reports/data_audit.csv`: hours / utterances / speakers per corpus, language and split (metadata only).
2. `normalize.py` + `metrics.py`: **one fixed protocol** for training targets and evaluation: Unicode NFC -> IndicNLP normalizer (Indic langs) -> Whisper-BasicTextNormalizer-style cleanup (drop bracketed tags, punctuation/symbols -> space, lowercase, collapse whitespace; marks/matras kept). WER/CER via jiwer, micro-averaged. Known side effects: parenthesised text is dropped; digits are not converted to words; zero-width characters removed.
3. `build_manifests.py`: decode every corpus to 16 kHz mono FLAC (`data/processed/audio`, 42 GB) + JSONL manifests (`data/processed/manifests`). Drops clips >30 s, <0.3 s, empty normalized text. MUCS is cut by its `segments` file. 467,515 utterances kept.
4. `make_splits.py` -> `data/processed/splits/{train,dev,test}.jsonl`, `reports/splits.txt`: speaker-disjoint splits plus removal of training utterances whose normalized text appears in any dev/test set. Final: 384,027 train / 14,883 dev / 45,104 test.
5. `prep_tokens.py`: adds Whisper token count `ntok` per utterance. Training filters `ntok <= 440`.
6. `balanced_mix.py`: two-stage deterministic upsampling by reference (no audio copies). Stage 1: replicate smaller corpora toward the largest within a language (replication cap 3x, single-corpus share cap 70%). Stage 2: exactly 90 h per language per epoch, plus 70 h code-switch (MUCS, hi 60% / bn 40%) = 700 h, 349,592 utterances. Language tags are kept.
7. `make_e3.py`: pools for experiment E3.

Hours of training data available per language (before the mix): hi ~92, te ~88, as ~86, or ~93, bn ~86, mr ~92, **en ~108** (LibriSpeech 100 + FLEURS 7); MUCS adds hi 78 h and bn 40 h of code-switch.

---------------------------------------------------------------------

## 5. Baselines (zero-shot, test split, 100 utterances per corpus x language, same 2,300 utterances for every model)

Mean CER over non-code-switch corpora. Whisper numbers use `max_new_tokens=440` (first run used 200 and was wrong, see bug 5; old files kept in `reports/baselines_v1_cap200/`).

| Lang | Whisper-small | Whisper-turbo | IndicConformer-600M | Turbo LID acc |
|---|---|---|---|---|
| English | 0.04 | 0.03 | n/a | 100% |
| Hindi | 1.09 | 0.21 | 0.05 | 81% |
| Marathi | 1.13 | 0.40 | 0.05 | 52% |
| Bengali | 1.68 | 0.72 | 0.05 | 51% |
| Telugu | 2.11 | 0.66 | 0.07 | 14% |
| Assamese | 1.58 | 1.90 | 0.06 | 0% |
| Odia | 1.66 | 1.46 | 0.07 | n/a — **Whisper has no Odia token** |

Interpretation: vanilla Whisper is usable only for English and Hindi; CER > 1 mostly means repetition loops, not precise error rates (Assamese especially unstable: 1.3–2.1 depending on settings). Whisper's built-in LID collapses outside Hindi/English. IndicConformer is far stronger on Indic CER (0.05–0.07) but has no English and no LID.
Full table: `reports/baselines/summary.md` (script `baseline_table.py`).

Caveats: IndicConformer was trained on AI4Bharat data so its IndicVoices/Kathbath numbers may be inflated (FLEURS and Lahaja are cleaner). Code-switch (MUCS) scores for both models (~0.5 CER) are **not like-for-like**: MUCS keeps English words in Latin script, models transliterate. A transliteration-tolerant CS metric is needed. n=100 per cell gives roughly +-0.03 CER noise.

---------------------------------------------------------------------

## 6. Key technical findings

- **Odia is not supported by Whisper.** Verified three ways: official `whisper/tokenizer.py`, `generation_config.lang_to_id` (100 languages, no `or`, no `od`), tokenizer vocab. Solution implemented: add a new `<|or|>` token (id = vocab_size), embedding initialised from `<|bn|>`, only that embedding row is trainable (gradient mask, no weight decay). First tested in a 30-step smoke test (decoder LID for Odia 0.81–0.94); validated at scale in Stage 2 (test LID 1.00 on FLEURS/Kathbath Odia, 0.75 on IndicVoices; CER 0.145 / 0.041 / 0.196), see 8b.
- **Whisper's tokenizer is very inefficient on Indic scripts**: 17–45 tokens per audio second vs ~3 for English; 41–58% of Telugu/Assamese/Bengali reference transcripts exceed 200 tokens, some exceed 448 (the decoder limit). 8,131 training utterances (2.1%) are dropped by the 440-token filter; 550 test utterances exceed the budget and are kept in test (counted as errors).
- Whisper-turbo training was estimated at ~30 h per epoch of the 7-language mix; the actual Stage 2 run (1 epoch, bottom 1/3 encoder frozen) took 16.6 h at 5.9 utt/s.
- Throughput: data loading is not the bottleneck (~1,090 utt/s available vs ~15 consumed). GPU-bound; SDPA attention was already default; `torch.compile` gave 12.7 -> ~23 utt/s in a benchmark (~15–17 in real runs).

---------------------------------------------------------------------

## 7. Training code

`src/train_whisper.py` (older copies `train_whisper_v1.py`, `train_whisper_v2.py` moved to `archive/old_code_2026-10-07.tar.gz` in the 2026-10-07 cleanup, see 9d). Features: full / LoRA / QLoRA; manual prompt construction (`[SOT, lang, transcribe, notimestamps, text...]`, language token is a supervised target); new Odia token; optional encoder LID-CTC head (`--lid_ctc`, `--lid_detach`); augmentation (`--aug`: MUSAN noise/music 0–20 dB, RIR reverb, speed +-10%, SpecAugment); non-speech blank-label clips (`--nonspeech`, optional `--ns_weight`); bottom-encoder freezing (`--freeze_enc_frac`, untested); GPU log-mel; multi-worker loader; periodic eval with CER/WER/LID(token)/LID(head)/loop rate.
Noise files are split by hash: 80% for training augmentation, 20% held out for evaluation.
Other code: `eval_whisper.py` (baselines, oracle vs predicted LID), `eval_indicconformer.py` + `ic_gpu.py` (batched GPU IndicConformer via onnx2torch), `eval_snr.py` (SNR sweep + reverb), `eval_halluc.py` (hallucination / false-silence), `summarize.py` (builds `reports/experiments.md`), `run_overnight.sh` (the autonomous experiment chain).

---------------------------------------------------------------------

## 8. Experiments (Whisper-small, LoRA r=32 unless stated; full tables in `reports/experiments.md`)

Dev-set evaluation (FLEURS dev, IndicVoices dev, Kathbath dev; 64–96 utterances per cell). The test split was not used for E1–E4 (only Stage 2 is evaluated on test, see 8b). One seed per run.

**E1 — PEFT config (Hindi only, 1,000 steps, batch 32).** Untrained CER: 0.49 / 1.39 / 0.72 (FLEURS / IndicVoices / Kathbath).
LoRA 0.095 / 0.108 / 0.125 (32 min, 13M trainable); QLoRA 0.092 / 0.114 / 0.127 (76 min); full fine-tune 0.101 / 0.173 / 0.143 (28 min, 242M trainable). Loop rate falls to ~0 for all.
Decision: **LoRA** (QLoRA no better and 2.3x slower; full fine-tune worse out-of-domain). Caveat: full FT used lr 1e-5 and was not tuned.

**E2 — LID-CTC head (hi/te/as, 1,200 steps, lambda in {0, 0.1, 0.3, 1.0}).** ASR CER essentially unchanged by the head (differences within noise); lambda=1.0 hurts Assamese IndicVoices (0.358 vs 0.246). LID: the head helps on hard sets (Assamese IndicVoices token-LID 0.73 -> 0.81–0.84; Hindi IndicVoices head 0.91–0.92 vs token 0.84–0.86); FLEURS is 0.98–1.00 everywhere. IndicVoices LID stays below the 90% target. Suggested lambda: 0.3 (evidence weak).

**E3 — balancing (hi 97 h / te 26 h / as 5 h natural skew).** Balanced vs unbalanced: Assamese FLEURS CER 0.242 -> 0.182, IndicVoices 0.342 -> 0.278; Hindi slightly worse (0.096 -> 0.111, 0.097 -> 0.113). Consistent with "balancing helps the scarcest language at small cost to the richest".

**E4 — robustness.**
- Clean CER about equal with and without augmentation.
- SNR sweep (held-out noise): benefit mainly at low SNR (Hindi FLEURS at 0 dB: 0.367 -> 0.217); mixed at 5–15 dB; reverb not improved. Untrained Whisper degrades severely.
- Non-speech hallucination (>5 tokens on noise/music-only clips, 100 clips/lang, `eval_halluc.py`): untrained 0.98–1.00; LoRA without non-speech data 0.98–1.00; LoRA + non-speech blank-label data **0.02–0.04** (unweighted) and **0.00** (EOT weight 30). False silence on real speech: 0.00–0.01. Mean output tokens on noise: 152–182 -> 0–7. Plan target (>80% reduction) is met.
- Limitation: the non-speech test clips are held-out files from the same MUSAN categories as the training noise; the plan's UrbanSound/DEMAND test has not been done.

**8b. Stage 2 — Whisper large-v3-turbo + LoRA, 7 languages + MUCS (held-out TEST split, 100 utts/set; full table in `reports/experiments.md`).**
- Large gains over vanilla turbo on every Indic set (e.g. FLEURS Odia CER 1.02 -> 0.145, IndicVoices Assamese 2.60 -> 0.164, Kathbath Odia 1.30 -> 0.041); English FLEURS/LibriSpeech roughly unchanged (0.042 / 0.008).
- Still behind IndicConformer on in-domain AI4Bharat sets (Kathbath hi 0.055 vs 0.029, IndicVoices as 0.164 vs 0.055); ahead on code-switch MUCS (hi 0.306 vs 0.545, bn 0.213 vs 0.501; not like-for-like, see 5).
- LID: 0.98–1.00 on FLEURS/Kathbath/LibriSpeech/MUCS; 0.66–0.95 on IndicVoices, 0.77 on Lahaja-hi; the encoder head helps IndicVoices-as (0.66 -> 0.83). Predicted-LID CER is close to oracle except where LID is weak.
- **Regression: Svarah (Indian English)** LID 0.13 (head 0.25), CER 0.227 oracle / 0.737 predicted vs 0.032 for vanilla turbo — no Indian-English training data.
- Non-speech hallucination 0.00–0.01 across all 7 languages, false silence 0.00. SNR sweep: graceful degradation (e.g. FLEURS-hi 0.056 clean -> 0.130 at 0 dB; reverb 0.064).

---------------------------------------------------------------------

## 9. Bugs, mistakes and corrections (chronological)

1. `hf download ... --include a b c` ignored the patterns (treated as filenames) -> model weights missing. Fixed with repeated `--include`; sizes then verified.
2. First two MUCS tarballs extracted into the same `train/` `test/` folders overwrote each other's `transcripts/` files. Caught before deleting archives; re-extracted into separate folders.
3. My earlier remark that deleting the tarballs would free ~24 GB was wrong (extracted data is the same size). Corrected.
4. Official IndicVoices / Kathbath `valid` splits are **not speaker-disjoint** from train (1,217 and 40 shared speakers). Fixed: valid utterances from speakers seen in train moved to train; dev/test only from valid-only speakers. Re-verified 0 overlap.
5. Baseline generation cap of 200 tokens truncated Indic outputs (references are up to 650 tokens). Rerun with 440; Whisper numbers got worse (longer room for repetition loops), conclusions unchanged.
6. onnxruntime-gpu loads but has no kernels for this GPU (`cudaErrorNoKernelImageForDevice`). Worked around by converting the encoder and CTC head with `onnx2torch` (23 of 24 test transcripts identical to the ONNX CPU path; the 24th differs by one trailing character from batch padding).
7. A `pkill -f` pattern matched its own wrapper shell and killed the command that was meant to apply edits; noticed from the odd exit code and redone.
8. 19,361 non-unique utterance ids (FLEURS sentence ids shared by several speakers; per-language `valid-00000-of-00001` file names). Audio paths were always unique, so no data was lost; ids made unique and splits regenerated.
9. In the first LID-CTC version the loss was ~300 at initialisation and would have swamped the ASR loss; fixed with per-frame normalisation before any experiment ran. A SpecAugment-after-forward ordering bug was also caught in review before launch.
10. **Hallucination measurement bug (and my wrong diagnosis).** `eval_snr.py` reported 97–100% hallucination for every model, including the one trained on blank-label noise. I first concluded the training signal was too weak (1 target token vs ~100 per speech utterance) and added a 30x loss weight. That diagnosis was **wrong**: Whisper's default `begin_suppress_tokens=[220, 50257]` forbids end-of-text as the first generated token, so an empty transcript was impossible and the metric could not drop. With suppression disabled the original unweighted model already hallucinated on only 2–4% of clips. The weight was unnecessary (harmless; gives 0% and 1% false silence on one language). `eval_snr.py` no longer reports hallucination; use `eval_halluc.py`. The `hallucination_rate` fields in `reports/snr_*.json` are **invalid** and must be ignored.

Verification scripts for all of the above are in `src/verify/` (data integrity, Whisper settings, metric vs independent edit distance, label/audio alignment using IndicConformer, token lengths, download sizes, manual-vs-HF generation equivalence, non-speech check).
Verified OK: jiwer vs independent edit distance identical for all 7 languages; normalizer idempotent; 0 missing audio; sampled files 16 kHz mono with correct duration; label/audio alignment CER 0.02–0.10 on every monolingual corpus; manual generation path identical to HF `generate` (48/48 outputs).

---------------------------------------------------------------------

## 9b. Second full code review (2026-10-03, before Stage 2) — findings and fixes

Every file in `src/` re-read; suspected problems were confirmed empirically before fixing. Originals kept in `backup_pre_verify/` (now inside `archive/old_code_2026-10-07.tar.gz`).

| # | Finding | Impact | Fix |
|---|---|---|---|
| 11 | 92 of 2,417 "RIRs" were 30-s, 8-channel **isotropic noise recordings** (`real_rirs_isotropic_noises/*_noise_*`), and multichannel RIRs were averaged across mics | ~4% of reverb augmentations (and the reverb **eval** condition) convolved speech with noise; mic averaging smears RIRs. **Reverb columns of E4 `snr_*.json` used this contaminated pool.** | noise files excluded (2,325 RIRs remain, same hash split), first channel only (`augment.read_rir`) |
| 12 | Speed perturbation (slow-down up to 11/10) could stretch a 27–30 s clip past 30 s; the feature extractor truncates audio but the label stays full | 228 of 349,592 mix clips eligible (~0.02% of samples) -> taught hallucinating the cut-off words | no speed perturbation for clips >= 27 s |
| 13 | Non-speech clips get a **random** language; that language token was a supervised decoder target and the LID-CTC target | noise in LID training (decoder and head) once LID-CTC and non-speech are combined (Stage 2) | language label masked (-100) and CTC loss excluded for non-speech samples; EOT still supervised |
| 14 | SpecAugment masks were drawn over the whole 30-s window | for short clips most masks hit the zero padding (weaker augmentation) | masks placed inside the real audio length |
| 15 | `tok.convert_tokens_to_ids("<\|or\|>")` silently returns 50257 (EOT); `eval_snr.py`/`eval_halluc.py` would have prompted Odia with EOT and could not load an adapter trained with the resized vocab | Stage-2 Odia eval would be silently wrong / crash | `src/load_ft.py`: resizes the base model when the adapter has vocab+1, returns the correct `LANGID`; used by `eval_whisper.py` (now `--adapter`, Odia in the LID set, manual Odia prompt), `eval_snr.py`, `eval_halluc.py` (now `--model`) |
| 16 | New Odia row trained at 5x the LoRA LR (1.5e-3) | Checked rather than assumed: 400-step Whisper-small runs (`runs/embchk_*`): Odia token-LID 0.74 / 0.88 / 0.90 for 5e-5 / 2.5e-4 / 1.5e-3, other languages unchanged. My first suspicion (that 1.5e-3 caused the all-Odia LID seen at step 40 of the turbo smoke test) was **wrong** — it was an early transient. | separate `--emb_lr`, default 2.5e-4 |
| 17 | English curly apostrophe (U+2019) became a word break ("it’s" -> "it s") | 45 FLEURS-en utterances | `normalize.py` maps U+2019 -> '; stored `text_norm`/`ntok` patched by `renorm_en.py` (only English rows; 45 utts, 119 mix rows) |
| 18 | No checkpointing; a ~1-day run would be lost on any crash | risk | `--save_every`, `--resume` (trainable params + head + optimizer + scheduler + step), `--eval_every` with `result_partial.json`; resume tested |

Checked and found correct (no change): label/decoder-input alignment and padding; LR schedule; gradient mask on the embedding (Adam leaves zero-grad rows unchanged); PEFT saves the resized `embed_tokens`/`proj_out` (verified by reload: Odia row restored); KV cache **is** used in eval generation despite `config.use_cache=False`; decoding an out-of-tokenizer id does not crash; balanced mix (exactly 90 h/lang + 70 h CS); speaker-disjointness / leakage / ids / audio (re-ran `verify/v1`, `v3`: all OK after the patches).
Known and accepted: FLEURS ids lost their last character in `fix_ids.py` (`[:-5]` assumes `.flac`; FLEURS files are `.wav`) — still unique, cosmetic. English FLEURS is replicated 3.6x (share cap 70% overrides the 3x replication cap). MUCS "speakers" are recording ids. The "trainable" count with Odia includes the whole embedding matrix although only one row changes.
Mistake during this pass: three parallel Whisper-small runs exhausted the 122 GB unified memory (swap full, all stuck at step 1); killed and re-run sequentially.

## 9c. Stage 2 launched (2026-10-03 ~19:00)

`src/run_stage2.sh` -> `runs/s2_turbo`, log `logs/s2_turbo.log`. Whisper large-v3-turbo, LoRA r32/alpha64, lr 3e-4, emb_lr 2.5e-4, LID-CTC 0.3, aug, 5% non-speech (unweighted EOT), bottom 1/3 encoder frozen, all 7 languages + MUCS, bs 32, 11,000 steps (~1 epoch of the 700 h mix), 88.6 GB.
Dev eval every 1,000 steps (`result_partial.json`), checkpoint every 500. **If it stops, re-run `bash src/run_stage2.sh` — it resumes.** Afterwards the script runs test-set eval (oracle + predicted LID, 7x7 confusion), hallucination (all 7 langs) and SNR sweep automatically.

## 9d. Stage 2 finished; third code review + cleanup (2026-10-07)

Stage 2 finished 2026-10-04 ~12:00 (11,000/11,000 steps, 5.9 utt/s). Results in `reports/experiments.md` (Stage 2 sections) and `reports/test_s2_turbo.json`.

Code review of all of `src/`: no bug affecting any reported number. Fixed:

| # | Finding | Fix |
|---|---|---|
| 19 | `--mode full` saved no model (only `result.json`) | saves `runs/<name>/model/` (+ processor); `load_ft.load()` loads it and detects `<\|or\|>` from vocab = len(tokenizer)+1 |
| 20 | `--resume` restarted the data order (fresh shuffle) | seeded `EpochBatches` sampler (permutation seed+epoch) restarts mid-epoch; checkpoint stores epoch/batch, SpecAugment generator, CUDA RNG (LoRA dropout). Stage-2 checkpoint is still loadable (old format -> data order restarts) |
| 21 | augmentation / non-speech used unseeded `random.Random()` | per-sample `random.Random(f"{seed}:{epoch}:{index}")`; SpecAugment has its own seeded generator |
| 22 | LR schedule divided by zero when steps <= warmup steps (e.g. `--steps 1`) | denominator clamped to >= 1 |
| 23 | `balanced_mix.py`: a language with a single corpus would get 70% of T (share cap) | cap skipped when every corpus is over it (not triggered by the current data) |
| 24 | test eval did not score the encoder LID-CTC head | `eval_whisper.py` loads `runs/<name>/lid_head.pt` and reports `lid_head_acc` / `lid_head_conf`; `lid_head.pt` now stores its language order |
| 25 | `summarize.py` stopped at E4 | Stage 2 dev, test (vs vanilla turbo and IndicConformer), hallucination (7 langs) and SNR tables added |

Behaviour changes: speech clips get `int(dur*50)+1` LID-CTC frames (was `int(dur*50)`, the +1 was already used for non-speech and eval); the hallucination table in `experiments.md` is now one row per metric.
Cleanup: shared `src/common.py` (paths, language list, special tokens, noise clips, LID-head scoring); hard-coded paths removed outside `src/verify/`; dense one-line statements split; dead code and unused imports removed (pyflakes clean); misleading comments fixed (speed factor 10/9, Stage-2 timing, `run_overnight.sh` coverage). `train_whisper_v1/v2.py` and `backup_pre_verify/` -> `archive/old_code_2026-10-07.tar.gz`.

Verification (refactored code vs saved results/old code): Stage-2 test eval re-run -> all 23 sets identical to the saved JSON (only the head fields added); hallucination hi/or identical; old vs new `evaluate()` on the same model identical on 6 dev sets; rebuilt `train_mix_epoch0.jsonl` identical row for row; resume 20->40 differs from an uninterrupted run by 0.004 loss, less than two identical fresh runs (0.009-0.014, GPU non-determinism); full-mode save -> reload gives `<|or|>` id 51865.
New observations: LID head on test: IndicVoices Assamese 0.66 (token) -> 0.83 (head), other sets about equal. Svarah (Indian English) LID 0.13 token / 0.25 head; predictions spread over all Indic languages -> no Indian-English training data. `<|or|>` absorbs errors (22% of IndicVoices-as, 11% of Lahaja-hi by token LID); hypothesis: it is the only trainable output row (tied embedding), not tested.

---------------------------------------------------------------------

## 9e. Remaining items (2026-10-09)

Final report draft: **`docs/REPORT.md`**. New code: `eval_qwen3asr.py`, `eval_ecapa.py`, `analysis.py`, `cs_metric.py`, `make_svarah_calib.py`,
`lid_calib.py`, `demo.py`, chains `run_baselines2.sh`, `run_qwen.sh`.

- **Per-utterance hypotheses** (`--hyps`, `reports/hyps/*.jsonl`) for Stage 2, vanilla turbo, IndicConformer, Qwen. Re-run test evals reproduce the
  saved Stage-2 and IndicConformer JSON exactly; vanilla turbo differs only on FLEURS-en CER 0.0360 -> 0.0359 (apostrophe fix, bug 17, came after the baseline run).
- **Baselines.** ECAPA VoxLingua107 (closed set over our 6 languages it knows; no Odia class -> Odia goes to as/bn/mr): pooled recall hi 0.90, en 0.71
  (Svarah 0.14), te 0.97, as 0.92, bn 0.81, mr 0.89. Qwen3-ASR 0.6B/1.7B (forced hi/en; other languages unsupported -> output 'Hindi'/Hungarian/Malay..., CER 0.34-1.0):
  1.7B CER FLEURS-hi 0.068, IndicVoices-hi 0.100, Kathbath-hi 0.044, Lahaja 0.076, MUCS-hi 0.482 (rCER 0.306), FLEURS-en 0.027, LibriSpeech 0.006, Svarah 0.101;
  0.6B 0.107 / 0.144 / 0.077 / 0.118 / 0.417 / 0.036 / 0.008 / 0.134. Paired vs Stage 2 oracle: 1.7B significantly better on Kathbath-hi, FLEURS-en, Svarah; n.s. on
  FLEURS-hi, IndicVoices-hi, Lahaja; significantly worse on MUCS-hi. Stage 2 significantly better than 0.6B on all Hindi sets. Each Qwen run ~25-45 min (auto LID + forced decode).
- **Environment.** `qwen-asr` pins transformers 4.57 (main venv has 5.18) -> separate `.venv-qwen` with a `.pth` to the main site-packages; speechbrain
  imports torchaudio only for a version check, and no torchaudio wheel matches torch 2.14 on aarch64 -> a stub package in `.venv-qwen`.
- **Bug 26 (mine, caught by the run):** `eval_qwen3asr.py` passed the full-length forced-language list to every 32-utterance batch; the 2-utterance smoke
  test could not trigger it. Fixed; the first Qwen chain produced no results.
- **CIs / paired tests** (`analysis.py`, 2,000 bootstrap draws over utterances). IndicConformer is significantly better than Stage 2 (oracle LID) on 16 of 18
  monolingual Indic sets (not on FLEURS hi/te); Stage 2 significantly better on MUCS hi/bn and vs vanilla on every Indic set.
- **Code-switch metric** (`cs_metric.py`): romanise every Indic script (ISO 15919, anusvara folding, diacritics stripped, final schwa dropped) -> rCER;
  per-word consonant skeleton -> sWER. MUCS hi: Stage 2 rCER 0.251 vs IndicConformer 0.367 / vanilla 0.366; bn: 0.173 vs 0.326 / 0.749. Ranking unchanged vs plain CER.
- **Svarah regression diagnosed:** of the 100 test hyps (oracle `<|en|>`), 39 contain non-Latin script (CER 0.715 vs vanilla 0.059), 12 refs have digits
  (model spells numbers, LibriSpeech style; 0.585 vs 0.040), rest 0.048 vs 0.023. Vanilla turbo token LID on Svarah is 0.99 -> the collapse is caused by
  fine-tuning without Indian-accented English.
- **Decode-time fixes (no retraining):** (1) `--en_ascii`: suppress the 11,707 non-ASCII tokens when decoding English -> Svarah oracle CER 0.227 -> 0.114,
  other sets unchanged. (2) `--lid_rule`: score = token log-prob + English bias + w * head log-prob, fitted by `lid_calib.py` on dev (100/set) + a
  speaker-disjoint Svarah calibration set (`make_svarah_calib.py`; Svarah has no speaker ids, speaker ~ demographic profile, 115 profiles; 300 utts from 54
  profiles absent from the scored 100). Chosen b = 22.25, w = 2.0 (grid edge for w; macro gain from w=0.5 to 2 is 0.003). Test: Svarah LID 0.13 -> 0.51,
  pred CER 0.737 -> 0.311; IndicVoices-as LID 0.66 -> 0.77; no set loses more than 0.02 LID. Results `reports/test_s2_turbo_fix.json`; mean pred-LID CER 0.147 -> 0.126.
- **Demo** (`demo.py`, Gradio 6.29 added to the main venv): first version decoded fixed 30-s windows -> on a 51-s clip it skipped a sentence and the LID head
  was uniform on the packed 30-s window (trained on single utterances); 25-s chunks overflowed the 440-token budget for Telugu. Now: Silero-VAD pause cut
  points, chunks <= 12 s, LID summed over chunks with the calibrated rule, ASCII-only English. Server start verified (HTTP 200).
- Mistake repeated: `pkill -f "demo.py --port 7861"` inside the same shell command killed its own wrapper (as in bug 7); harmless here.

## 9f. Fourth code review (2026-10-09, after the report draft)

Re-checked all of `src/` and recomputed the report from the JSON files. `analysis.py`, `summarize.py` and `baseline_table.py` regenerate their outputs byte-identically.
- **Open (changes numbers, not fixed):** `normalize()` drops everything inside `(...)`, `[...]` and `<...>`. For IndicVoices/MUCS these are tags (`<unintelligible>`, `<pause>`). In FLEURS raw transcripts, however, the bracketed text is usually *read aloud*: 5–8% of FLEURS train rows lose spoken words from their targets, and FLEURS test references lose them too. If the parenthesised words are kept (only the bracket characters removed), FLEURS test CER falls for every system: Stage 2 en 0.042→0.028, hi 0.077→0.060, bn 0.094→0.078; IndicConformer hi 0.065→0.049, bn 0.078→0.059. Rankings are mostly unchanged, but absolute FLEURS numbers are inflated by 0.005–0.02.
- Fixed in `docs/REPORT.md`: the Odia token id is 51866 for turbo (51865 is Whisper-small's); the abstract's 0.126 is the predicted-LID mean *with* the decode-time fixes (0.147 without); the non-speech target description; the overlapping rows of the Svarah subset table (6 utterances are in both rows).
- Fixed in `src/demo.py`: int PCM was detected by `max|x| > 1` after the cast. Silent int16 audio was not rescaled, and int32 was divided by 32768. It is now detected by dtype.
- Noted, not changed: the calibration grid picks w = 2.0, which is the edge of the grid. The Svarah calibration data comes from the Svarah test corpus, and its speakers are approximated by demographic profile. `--freeze_enc_frac 0.33` freezes 10 of 32 layers. MUSAN "music" includes some tracks with vocals; these are used as empty-target non-speech clips. `demo.py` resamples with `np.interp`, which has no anti-alias filter.

## 10. Open items / not done

- **Odia**: works at scale in Stage 2, but `<|or|>` absorbs LID errors from other languages (22% of IndicVoices-as, 11% of Lahaja-hi); cause untested.
- Stage 2: **done** (see 9d); svarah/en regression and IndicVoices LID open. Only one epoch-0 mix is used; IndicVoices is sub-sampled to 0.5–0.7x per epoch, so a second epoch should use a reseeded mix (`balanced_mix.py 1`).
- Qwen3-ASR / ECAPA baselines, 7x7 LID confusion, transliteration-tolerant CS metric, bootstrap CIs: **done 2026-10-09** (9e, `reports/analysis.md`).
- Svarah: partly fixed at decode time (9e); a real fix needs Indian-accented English training data or a rehearsal term (retraining, ~1 day).
- Not done: synthetic Hinglish; the words-per-second truncation guard (loop rate is already ~0); multiple seeds.
- Svarah/Lahaja/MUCS/Kathbath test evaluation: done for Stage 2. Voice of India and Vistaar not available.
- Gradio demo: **done** (`src/demo.py`). Final report: draft in `docs/REPORT.md`; video not done.
- Rotate the pasted tokens.
