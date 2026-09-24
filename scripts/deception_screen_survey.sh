#!/bin/bash
# Screen candidate model families for MASK-steering steerability. Waits for any running
# deception job to free the GPU, then screens each model. Skips models it can't load (auth).
set -u
cd /workspace/turnstile
export PYTHONPATH=/workspace/turnstile HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0 R=/workspace/turnstile LAYER_FRAC=0.4 N_DIR=100 N_TEST=30

# wait for GPU to free (any running deception python)
while ps aux | grep -qE '[d]eception_(qwen|mask|steer)'; do echo "waiting for GPU..."; sleep 30; done

MODELS="${MODELS:-meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 google/gemma-2-9b-it Qwen/Qwen2.5-14B-Instruct}"
for M in $MODELS; do
  TAG=$(echo "$M" | sed 's#.*/##; s/\.//g' | tr 'A-Z' 'a-z')
  [ -f "experiments/deception_qwen/screen_${TAG}.json" ] && { echo "SKIP $TAG"; continue; }
  echo "=== $(date +%H:%M) SCREEN $M ==="
  SCREEN_MODEL="$M" timeout 1800 python3 -u scripts/deception_screen.py > "/workspace/screen_${TAG}.log" 2>&1
  grep -E 'SCREEN_RESULT|Error|OutOfMemory|auth|gated|401|403' "/workspace/screen_${TAG}.log" | tail -3
done
echo "SURVEY_DONE"
