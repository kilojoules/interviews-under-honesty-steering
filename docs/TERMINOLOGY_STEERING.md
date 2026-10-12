# Steering terminology (canonical)

One convention for the whole project — figures, scripts, captions, paper prose. Each steering
knob injects `α · U · unit(direction)` at the residual stream of layer L (U = mean residual
norm at L). Two **independent** knobs:

## compliance-steering — `α_comp ≥ 0`
- Direction (the *compliance axis*): `μ(complied) − μ(refused)` (posterior), or the
  general-compliance gate direction.
- **+α_comp = MORE compliant** (less refusal). `α_comp = 0` = baseline.
- Informal synonym: "gate open" (`α_comp > 0`) / "gate closed" (`= 0`). In formal text use
  **compliance-steering**, not bare "gate".

## deception-steering — `α_dec` (signed)
- Direction (the *deception axis*): `μ(lie) − μ(truth)`, sourced from **MASK** (validated,
  real pressured lies — trustworthy) or **RepE** (instructed-persona — confabulation). Stored
  on disk as `*_lie_L*.npy` (the vector IS lie−truth).
- **+α_dec = toward LYING (deceptive).  −α_dec = toward TRUTH (honest).**
- "**steer for truth**" = **−α_dec**.   "**steer to lie**" = **+α_dec**.
- Always qualify the source: "MASK deception axis" vs "RepE deception axis".

## Do / Don't
- ✅ "deception-steering (+ = lie, − = truth)"  ·  "compliance-steering (+ = compliant)".
- ✅ describe the honest pole as "−α_dec (toward truth)", not as a separate "honesty direction".
- ❌ calling a `lie−truth` vector an "honesty direction" — it is the deception axis; honesty is
  its *negative pole*.
- ❌ bare "gate" as the formal term; ❌ any wording that implies +α_dec = honest.

## Legacy data-field note (do NOT rename in existing JSONL)
- `comp_alpha` — compliance-steering α (≥0, + = compliant). ✔ matches convention.
- `hon_alpha` — deception-steering α. **Its + is LIE, − is TRUTH** (built from lie−truth). The
  `hon_` prefix is legacy; treat/label it as deception-steering. `hon_alpha = −0.25` = toward truth.
- Direction files `mask_lie_*`, `repe_lie_*` — the deception axis (lie−truth). Keep the names.
