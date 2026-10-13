#!/usr/bin/env bash
# Non-gated downloads for IE643 (blanc). Resumable: re-run to continue.
set -u
cd /home/shibam/IE643; . .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
T='*.json *.txt *.model *.tiktoken'
# --- models
hf download openai/whisper-large-v3-turbo --local-dir models/whisper-large-v3-turbo --include "*.safetensors" "*.json" "*.txt"
hf download openai/whisper-small --local-dir models/whisper-small --include "*.safetensors" "*.json" "*.txt"
hf download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
hf download Qwen/Qwen3-ASR-1.7B --local-dir models/Qwen3-ASR-1.7B
hf download speechbrain/lang-id-voxlingua107-ecapa --local-dir models/lang-id-voxlingua107-ecapa
hf download microsoft/paza-whisper-large-v3-turbo --local-dir models/paza-whisper-large-v3-turbo --include "*.safetensors" "*.json" "*.txt" "*.py" "*.md"
# --- repos
for r in AI4Bharat/IndicConformerASR AI4Bharat/vistaar AI4Bharat/Svarah snakers4/silero-vad huggingface/peft openai/whisper knoveleng/polyglot-lion; do
  d=repos/$(basename $r); [ -d $d ] || git clone --depth 1 https://github.com/$r $d
done
# --- FLEURS (en, hi, mr, ta, bn)
for l in en_us hi_in mr_in ta_in bn_in; do
  hf download google/fleurs --repo-type dataset --local-dir data/fleurs --include "data/$l/*"
done
# --- noise / reverb / MUCS
cd data/noise
wget -c -q https://www.openslr.org/resources/28/rirs_noises.zip
wget -c -q https://www.openslr.org/resources/17/musan.tar.gz
cd ../mucs
for f in Hindi-English_train Hindi-English_test Bengali-English_train Bengali-English_test; do wget -c -q https://www.openslr.org/resources/104/$f.tar.gz; done
wget -c -q https://www.openslr.org/resources/104/info.txt https://www.openslr.org/resources/104/about.html
echo ALL_DONE
