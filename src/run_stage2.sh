#!/usr/bin/env bash
# Stage 2: Whisper large-v3-turbo + LoRA on the full 7-language balanced mix (+ MUCS code-switch).
# Config from E1-E4: LoRA r32, LID-CTC lambda 0.3, balanced mix, aug + 5% non-speech blank-label clips (unweighted EOT), bottom 1/3 encoder frozen.
# ~1 epoch of the 700 h mix = 10,925 steps at bs 32 (ran 2026-10-03 19:00 -> 10-04 12:00 at 5.9 utt/s, 88.6 GB). Checkpoint every 500 steps;
# re-running this script RESUMES (checkpoints written by the current train_whisper.py also restore the data order).
# Then: dev-set evals are inside the run (every 1000 steps); test-set / SNR / hallucination evals follow.
cd /home/shibam/IE643/src; source ../.venv/bin/activate
EMB_LR=${EMB_LR:-2.5e-4}
N=s2_turbo
python train_whisper.py --name $N --model ../models/whisper-large-v3-turbo --langs hi,en,te,as,or,bn,mr --cs \
  --mode lora --r 32 --alpha 64 --lr 3e-4 --emb_lr $EMB_LR --lid_ctc 0.3 --aug --nonspeech 0.05 --ns_weight 1 --freeze_enc_frac 0.33 \
  --steps 11000 --bs 32 --eval_n 48 --eval_every 1000 --save_every 500 --resume >> ../logs/$N.log 2>&1 || { echo "TRAIN FAILED"; exit 1; }
A=../runs/$N/adapter; M=../models/whisper-large-v3-turbo
python eval_whisper.py --model $M --adapter $A --split test --n 100 --out ../reports/test_$N.json > ../logs/test_$N.log 2>&1
python eval_halluc.py --model $M --adapter $A --langs hi,en,te,as,or,bn,mr --out ../reports/halluc_$N.json > ../logs/halluc_$N.log 2>&1
python eval_snr.py --model $M --adapter $A --langs hi,te,as,or --out ../reports/snr_$N.json > ../logs/snr_$N.log 2>&1
echo STAGE2_DONE >> ../logs/$N.log
