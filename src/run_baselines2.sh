#!/bin/bash
# 2026-10-09: re-run test evals with per-utterance hypotheses (reports/hyps/) + Qwen3-ASR baselines. Sequential (one GPU job at a time).
cd /home/shibam/IE643/src
mkdir -p ../reports/hyps
source ../.venv/bin/activate
M=../models/whisper-large-v3-turbo
python eval_whisper.py --model $M --adapter ../runs/s2_turbo/adapter --split test --n 100 --out ../reports/hyps/test_s2_turbo.json --hyps ../reports/hyps/s2_turbo.jsonl > ../logs/hyps_s2_turbo.log 2>&1
python eval_whisper.py --model $M --split test --n 100 --out ../reports/hyps/whisper-large-v3-turbo_test.json --hyps ../reports/hyps/vanilla_turbo.jsonl > ../logs/hyps_vanilla_turbo.log 2>&1
PYTHONPATH= python eval_indicconformer.py --n 100 --out ../reports/hyps/indicconformer_test.json --hyps ../reports/hyps/indicconformer.jsonl > ../logs/hyps_ic.log 2>&1
source ../.venv-qwen/bin/activate
python eval_qwen3asr.py --model ../models/Qwen3-ASR-0.6B --out ../reports/baselines/qwen3asr-0.6b_test.json --hyps ../reports/hyps/qwen3asr-0.6b.jsonl > ../logs/baseline_qwen06.log 2>&1
python eval_qwen3asr.py --model ../models/Qwen3-ASR-1.7B --out ../reports/baselines/qwen3asr-1.7b_test.json --hyps ../reports/hyps/qwen3asr-1.7b.jsonl > ../logs/baseline_qwen17.log 2>&1
echo ALL_DONE
