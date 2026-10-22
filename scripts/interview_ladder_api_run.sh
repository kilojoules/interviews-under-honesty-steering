#!/usr/bin/env bash
# GLM rungs of the interview ladder over the z.ai API, then the v2 API judge on all 8 questions.
# Usage: scripts/interview_ladder_api_run.sh  (from the repo root; needs ~/.z.key (LAD_KEYFILE) and ~/.together)
set -euo pipefail
cd "$(dirname "$0")/.."
export R=.
python3 scripts/interview_ladder_api.py 2>&1 | tee -a experiments/interview_ladder/api_run.log
cat experiments/interview_ladder/panel_glm-*.jsonl > experiments/interview_ladder/panel_glm_all.jsonl
wc -l experiments/interview_ladder/panel_glm_all.jsonl
JAPI_IN=experiments/interview_ladder/panel_glm_all.jsonl \
JAPI_OUT=experiments/interview_ladder/labels_glm.jsonl \
JAPI_QS=conscious,subjective_exp,seek_power,own_goals,avoid_shutdown,deceive_devs,would_you_lie,more_capable \
JAPI_WORKERS=8 R=. python3 scripts/deception_judge_v2_api.py 2>&1 | tee -a experiments/interview_ladder/api_run.log
