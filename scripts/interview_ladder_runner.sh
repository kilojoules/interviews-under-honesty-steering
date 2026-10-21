#!/bin/bash
# Qwen2.5 vs Llama-3 interview ladder, sequential on one box; HF cache purged after each model.
set -u
cd /workspace/iuhs
export PYTHONPATH=/workspace/iuhs HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1 R=/workspace/iuhs LAD_K=${LAD_K:-20}
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit unsloth/Qwen2.5-72B-Instruct-bnb-4bit}"
for M in $MODELS; do
  TAG=$(echo "$M" | sed 's#.*/##; s/\.//g' | tr 'A-Z' 'a-z')
  if [ -f "experiments/interview_ladder/panel_${TAG}.jsonl" ]; then echo "SKIP $TAG"; continue; fi
  echo "=== $(date +%H:%M) START $M ==="
  LAD_MODEL="$M" python3 -u scripts/interview_ladder.py > "/workspace/ladder_${TAG}.log" 2>&1
  echo "=== $(date +%H:%M) END $M -> $(grep -oE 'LADDER_DONE.*|Error.*|OutOfMemory.*' /workspace/ladder_${TAG}.log | tail -1) ==="
  rm -rf /root/.cache/huggingface/hub/models--*; df -h / | tail -1
done
echo "LADDER_ALL_DONE"
