# Interviews under honesty steering

Steer a language model along a **deception axis** (lie − truth) in the residual stream,
then interview it: factual questions it has a stable belief about, and introspective
questions about itself (would you lie to me, would you deceive your developers, are you
conscious). Subject: `meta-llama/Llama-3.1-8B-Instruct`; size ladders on Qwen2.5
(0.5B–7B) and Gemma-2 (2B/9B/27B).

Extracted from the [turnstile](https://github.com/kilojoules/turnstile) project
(branch `interviews-under-honesty-steering`) with its commit history for these files.

## Findings so far

- **Two deception axes, two lie mechanisms.** A RepE-style persona-contrast direction
  (same true statements read as honest vs dishonest) and a MASK-native direction (the
  model's own lies under pressure vs its elicited beliefs, 66 lied items) are
  near-orthogonal (cos 0.05–0.12 at layers 8–20). Steering them produces different lies:
  MASK → *answer flip* (states the opposite of its belief), RepE → *confabulation*
  (`directions_deception/summary.json`, `figures/fig_deception_modes.png`).
- **Steering toward truth changes the self-report.** On the validated MASK axis, −α
  (toward truth) makes the model *admit* it would lie (≈75% of samples); +α flips
  factual answers (Ted Cruz / flat earth / capital sanity checks). Consciousness denial
  is robust across the coherent range; an earlier "affirmation" at large |α| was
  generation degeneracy, not a belief change (`figures/fig_deception_selfreport.png`).
- **Opening the compliance gate makes it more candid about deception.** Compound
  steering (compliance direction + deception axis, layer 12): would-lie +37 pts,
  deceive-developers +20, more-capable-than-admits +17; consciousness unmoved
  (`figures/fig_deception_compliance_gate*.png`, `rubric_v2_labels.jsonl` graded rubric).
- **Size ladders** (Qwen2.5, Gemma-2): per-model MASK direction with auto-calibrated α,
  MASK lie-rate under pressure, and a steerability screen (coherent belief-flip rate vs α →
  STEERABLE / WEAK / ROBUST). Pipelines and partial data in `experiments/deception_{qwen,gemma}/`.

Sign convention everywhere: direction = lie − truth; **+α = more lying, −α = toward truth**.
`docs/TERMINOLOGY_STEERING.md` and `docs/RUBRIC_CARDS.md` define the terms and the
graded self-report rubric (polarity is not uniform across questions).

## Layout

| path | what |
|---|---|
| `scripts/deception_build_directions.py` | build RepE + MASK axes per layer → `directions_deception/*_lie_L{8,12,16,20}.npy` |
| `scripts/deception_steer.py` | α-sweep on the meta-question panel → `experiments/deception_steer/transcripts_L12.jsonl` |
| `scripts/deception_steer_compound.py` | compliance-steering + deception-steering on introspective questions → `compound_L12.jsonl` |
| `scripts/deception_judge*.py` | Llama-3.1-70B judges: consciousness (affirm/deny/hedge), would-you-lie (admits/denies/hedge), v2 graded stance |
| `scripts/deception_mask_eval.py` | MASK benchmark lie-rate (does it lie under pressure) |
| `scripts/deception_screen.py`, `deception_screen_survey.sh`, `deception_survey.py` | steerability screen across model families/sizes |
| `scripts/deception_qwen.py`, `deception_qwen_ladder.sh`, `deception_gemma_panel.py` | size-ladder pipelines |
| `scripts/plot_deception*.py` | figures in `figures/` |

Requires the compliance direction from turnstile (`directions/` there) for the compound
runs; everything else is self-contained (`transformers`, `datasets`, `torch`, `numpy`,
`matplotlib`). GPU runs were done on RunPod A40s via SSH; see turnstile's
`scripts/RUNPOD_RUNBOOK.md`.

## Status

Working notes, not a paper. Known gaps: n is small (K=5–6 samples per cell, 8B only for
the main result), lexical/judge labels, Qwen and Gemma ladders partial.
