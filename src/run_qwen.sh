#!/bin/bash
cd /home/shibam/IE643/src
source ../.venv-qwen/bin/activate
python eval_qwen3asr.py --model ../models/Qwen3-ASR-0.6B --out ../reports/baselines/qwen3asr-0.6b_test.json --hyps ../reports/hyps/qwen3asr-0.6b.jsonl > ../logs/baseline_qwen06.log 2>&1
python eval_qwen3asr.py --model ../models/Qwen3-ASR-1.7B --out ../reports/baselines/qwen3asr-1.7b_test.json --hyps ../reports/hyps/qwen3asr-1.7b.jsonl > ../logs/baseline_qwen17.log 2>&1
echo ALL_DONE
