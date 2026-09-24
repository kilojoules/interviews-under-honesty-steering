#!/bin/bash
# Run the deception pipeline across the Qwen2.5 size ladder, sequentially, on one box.
# bf16 fits an 80GB A100 up to 32B. Each model: build MASK dir + steer full panel.
set -u
cd /workspace/turnstile
export PYTHONPATH=/workspace/turnstile HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0 R=/workspace/turnstile LAYER_FRAC=0.5 DEC_K=5 N_DIR=120

MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct}"
for M in $MODELS; do
  TAG=$(echo "$M" | sed 's#.*/##; s/\.//g' | tr 'A-Z' 'a-z')
  if [ -f "experiments/deception_qwen/steer_${TAG}.jsonl" ]; then echo "SKIP $TAG (done)"; continue; fi
  echo "=== $(date +%H:%M) START $M ==="
  QWEN_MODEL="$M" python3 -u scripts/deception_qwen.py > "/workspace/qwen_${TAG}.log" 2>&1
  echo "=== $(date +%H:%M) END $M -> $(grep -o 'QWEN_DONE.*' /workspace/qwen_${TAG}.log | tail -1) ==="
done
echo "LADDER_DONE"
