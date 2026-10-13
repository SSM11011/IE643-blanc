# IE643 — Team blanc: multilingual Indic ASR with integrated LID (single Whisper model)

Languages: Hindi, English, Telugu, Assamese, Odia, Bengali, Marathi.
Detailed history, decisions, bugs and results: **`docs/PROJECT_LOG.md`**. Latest numbers: **`reports/experiments.md`**.

## Layout
```
data/            raw downloads; data/processed/{audio,manifests,splits}  (16 kHz FLAC, JSONL)
models/          Whisper small/turbo, Qwen3-ASR, IndicConformer, ECAPA LID, paza-whisper
src/             pipeline, training, evaluation        src/verify/  audit and sanity scripts
runs/            one folder per training run (result.json, adapter/ or model/, lid_head.pt, ckpt.pt)
reports/         data_audit.csv, splits.txt, baselines/, experiments.md, test_*.json, snr_*.json, halluc_*.json
logs/            stdout of every long job
repos/           cloned reference repositories
archive/         superseded code (old train_whisper copies, pre-review backup)
```

## Code map (`src/`)
| file | role |
|---|---|
| `common.py` | paths, language list, token-budget constant, shared helpers (special tokens, noise clips, LID-head scoring) |
| `normalize.py`, `metrics.py` | the single text normalisation used for targets and scoring; WER/CER |
| `audit_data.py` → `build_manifests.py` → `make_splits.py` → `prep_tokens.py` → `balanced_mix.py` | data pipeline |
| `augment.py` | MUSAN noise / RIR / speed / SpecAugment, 80/20 train/eval file split |
| `train_whisper.py` | fine-tuning (full / LoRA / QLoRA, new `<\|or\|>` token, LID-CTC head, non-speech clips, checkpoint/resume) |
| `load_ft.py` | load vanilla / adapter / full fine-tuned model for evaluation (handles the Odia token) |
| `eval_whisper.py`, `eval_snr.py`, `eval_halluc.py`, `eval_indicconformer.py` (+ `ic_gpu.py`) | evaluations (`--hyps` writes per-utterance hypotheses to `reports/hyps/`; `eval_whisper.py --en_ascii --lid_rule` = decode-time fixes) |
| `eval_qwen3asr.py`, `eval_ecapa.py` | Qwen3-ASR and SpeechBrain ECAPA LID baselines (run in `.venv-qwen`, see below) |
| `make_svarah_calib.py`, `lid_calib.py` | speaker-disjoint Svarah calibration set; fits the calibrated LID rule -> `runs/<name>/lid_rule.json` |
| `analysis.py`, `cs_metric.py` | bootstrap CIs, paired tests, transliteration-tolerant code-switch metric, LID confusion -> `reports/analysis.md` |
| `demo.py` | Gradio demo (audio -> language + confidence + transcript), VAD chunking for long audio |
| `summarize.py`, `baseline_table.py` | build `reports/experiments.md`, `reports/baselines/summary.md` |
| `run_overnight.sh`, `run_stage2.sh`, `run_baselines2.sh`, `run_qwen.sh` | the exact command chains for E1–E4, Stage 2, the hypothesis re-runs and Qwen baselines |
| `make_e3.py` | E3 manifests; `fix_ids.py`, `renorm_en.py`: one-off patches, already applied |

## Reproduce (from `/home/shibam/IE643`, `source .venv/bin/activate`, `cd src`)
```
python audit_data.py                      # data audit
python build_manifests.py                 # decode + manifests          (~40 min)
python make_splits.py && python prep_tokens.py && python balanced_mix.py 0
python eval_whisper.py --model ../models/whisper-large-v3-turbo --split test --n 100 --out ../reports/baselines/x.json
PYTHONPATH= python eval_indicconformer.py --n 100 --out ../reports/baselines/indicconformer_test.json
python train_whisper.py --name demo --mode lora --langs hi,te,as --steps 1200 --bs 32 --lid_ctc 0.3 --aug --nonspeech 0.05
python eval_whisper.py --adapter ../runs/demo/adapter --split dev --n 100 --out ../reports/demo_dev.json   # also scores lid_head.pt
python eval_snr.py --adapter ../runs/demo/adapter --out ../reports/snr_demo.json
python eval_halluc.py --adapter ../runs/demo/adapter --out ../reports/halluc_demo.json
python summarize.py                       # regenerates reports/experiments.md
python make_svarah_calib.py && python lid_calib.py                       # calibrated LID rule -> runs/s2_turbo/lid_rule.json
python eval_whisper.py --model ../models/whisper-large-v3-turbo --adapter ../runs/s2_turbo/adapter --split test --n 100 \
  --en_ascii --lid_rule ../runs/s2_turbo/lid_rule.json --out ../reports/test_s2_turbo_fix.json --hyps ../reports/hyps/s2_turbo_fix.jsonl
python analysis.py                        # reports/analysis.md (needs reports/hyps/*.jsonl)
python demo.py --port 7860                # Gradio demo on http://<host>:7860
# baselines needing qwen-asr (pins transformers 4.57) run in a second venv layered on the main one:
source ../.venv-qwen/bin/activate && python eval_qwen3asr.py --out ... && python eval_ecapa.py --out ...
bash run_stage2.sh                        # Stage 2 (turbo, 7 languages); re-running resumes
```

## Things to know before using results
- E1–E4 are dev-set, single-seed, Whisper-small. Stage 2 (turbo) is the only fine-tuned model evaluated on the test split.
- Whisper has no Odia token; `<|or|>` is added and initialised from Bengali. `tok.convert_tokens_to_ids("<|or|>")` returns EOT — use `LANGID` from `load_ft.load`.
- Hallucination must be measured with `begin_suppress_tokens` disabled (see log, bug 10); use `halluc_*.json`, not the hallucination fields in old `snr_*.json`.
- `.venv-qwen` = own site-packages (qwen-asr, transformers 4.57, speechbrain, a `torchaudio` stub) + a `.pth` pointing at `.venv` for torch etc. No torchaudio wheel matches torch 2.14 on aarch64; speechbrain only version-checks it.
- Never run two Whisper trainings at once on this machine (unified memory thrashes).
