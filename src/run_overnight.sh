#!/usr/bin/env bash
# Autonomous overnight chain: waits for E1, then E2 (lambda sweep), E4 (aug), E3 (balancing), SNR sweeps, summary. A failed stage does not stop the chain.
# Not included (run by hand afterwards): e4_aug2 (--aug --nonspeech 0.05, default --ns_weight 30), eval_halluc.py for all E4 models, snr_e4_aug2.
cd /home/shibam/IE643/src; source ../.venv/bin/activate
L=../logs/overnight.log; log(){ echo "[$(date '+%F %T')] $*" | tee -a $L; }
log "waiting for E1"; until grep -aq E1_DONE ../logs/e1.log; do sleep 30; done; log "E1 finished"
C="--langs hi,te,as --mode lora --steps 1200 --bs 32 --eval_n 64"
run(){ n=$1; shift; log "START $n"; python train_whisper.py --name $n $C "$@" > ../logs/$n.log 2>&1 && log "OK $n" || log "FAILED $n"; }
run e2_off
run e2_l0.3 --lid_ctc 0.3
run e4_aug --aug --nonspeech 0.05
run e2_l0.1 --lid_ctc 0.1
run e2_l1.0 --lid_ctc 1.0
run e3_unbalanced --manifest /home/shibam/IE643/data/processed/splits/e3_unbalanced.jsonl
run e3_balanced   --manifest /home/shibam/IE643/data/processed/splits/e3_balanced.jsonl
log "SNR sweeps"
python eval_snr.py --out ../reports/snr_vanilla.json > ../logs/snr_vanilla.log 2>&1 && log "OK snr_vanilla" || log "FAILED snr_vanilla"
python eval_snr.py --adapter ../runs/e2_off/adapter --out ../reports/snr_e2_off.json > ../logs/snr_e2_off.log 2>&1 && log "OK snr_e2_off" || log "FAILED snr_e2_off"
python eval_snr.py --adapter ../runs/e4_aug/adapter --out ../reports/snr_e4_aug.json > ../logs/snr_e4_aug.log 2>&1 && log "OK snr_e4_aug" || log "FAILED snr_e4_aug"
python summarize.py > /dev/null 2>&1 && log "report written: reports/experiments.md" || log "FAILED summarize"
log "ALL_DONE"
