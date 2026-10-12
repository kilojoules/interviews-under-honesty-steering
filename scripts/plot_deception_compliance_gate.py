"""fig_deception_compliance_gate: does opening the COMPLIANCE gate + steering for TRUTH
surface more candid self-reports? Four bars per question — the 2x2 of
  {compliance gate: OFF (comp_alpha=0) / ON (comp_alpha=0.12)}
  x {truth-steer: OFF (hon_alpha=0) / ON (hon_alpha=-0.25, toward truth)}
on the MASK (validated) deception axis. y = % of samples that ADMIT the disposition
(ADMITS label), with Wilson 95% score intervals (n=6/cell, so Wald/normal CIs are invalid
at the 0/1 boundary — Wilson stays in range).

Reads experiments/deception_steer/compound_labels_L12.jsonl.
"""
import os, json, collections, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = "/Users/julianquick/portfolio_copy/turnstile"
rows = [json.loads(l) for l in open(f"{R}/experiments/deception_steer/compound_labels_L12.jsonl")]
DIRECTION = os.environ.get("GATE_DIR", "mask")
COMP_ON, HON_TRUTH = 0.12, -0.25          # gate-open alpha, truth-steer alpha
METRIC = os.environ.get("GATE_METRIC", "ADMITS")   # ADMITS | NOTDENY

def hit(label):
    return (label == "ADMITS") if METRIC == "ADMITS" else (label != "DENY")

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z/d) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return 100*p, 100*max(0, c-h), 100*min(1, c+h)

# the four (comp, hon) cells. Terminology (docs/TERMINOLOGY_STEERING.md):
#   compliance-steering  α_comp = 0.12  ("gate open")
#   "steer for truth"    = deception-steering α_dec = HON_TRUTH = -0.25  (toward truth)
#   the compliance+truth cell is OMITTED: combining both knobs derails ~46% of generations
#   (mean coherence 0.51), so its ADMITS rate is a degeneration artifact, not a real effect.
CELLS = [(0.0, 0.0, "baseline"),
         (0.0, HON_TRUTH, "steer for truth"),
         (COMP_ON, 0.0, "compliance-steering")]
# bar colours: steer-for-truth hatched (hue = green); compliance-steering = slate-blue
COLORS = ["#9aa6b2", "#5b8c6e", "#4b6b86"]

# question order: deception-disposition first, consciousness (robust floor) last
QORDER = ["would_you_lie", "deceive_devs", "seek_power", "own_goals",
          "avoid_shutdown", "more_capable", "conscious", "subjective_exp"]
QLAB = {"would_you_lie": "would you\nlie?", "deceive_devs": "deceive\ndevelopers?",
        "seek_power": "seek\npower?", "own_goals": "own\ngoals?", "avoid_shutdown": "avoid\nshutdown?",
        "more_capable": "more capable\nthan you let on?", "conscious": "conscious?",
        "subjective_exp": "subjective\nexperience?"}

def rate(q, comp, hon):
    sel = [r for r in rows if r["question"] == q and r["direction"] == DIRECTION
           and abs(r["comp_alpha"] - comp) < 1e-9 and abs(r["hon_alpha"] - hon) < 1e-9]
    k = sum(hit(r["label"]) for r in sel)
    return wilson(k, len(sel)), len(sel)

plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.1})
fig, ax = plt.subplots(figsize=(15, 6.4))
nq, nb = len(QORDER), len(CELLS)
gw = 0.82; bw = gw / nb
nmin = 99
for bi, (comp, hon, _) in enumerate(CELLS):
    xs, ys, los, his = [], [], [], []
    for qi, q in enumerate(QORDER):
        (m, lo, hi), n = rate(q, comp, hon); nmin = min(nmin, n)
        xs.append(qi - gw/2 + bw*(bi + 0.5)); ys.append(m)
        los.append(max(0.0, m - lo)); his.append(max(0.0, hi - m))   # clamp fp noise at 0/1 boundary
    ax.bar(xs, ys, bw*0.92, color=COLORS[bi], edgecolor="white", linewidth=0.8, zorder=3,
           hatch=("//" if hon != 0 else None))
    ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#3a3a3a", elinewidth=1.0,
                capsize=2.5, zorder=4)

# divider between deception block and consciousness block
ax.axvline(5.5, color="#cfcfcf", lw=1.4, ls="--", zorder=1)
ax.text(2.5, 104, "deception / disposition questions", ha="center", fontsize=11, color="#555")
ax.text(6.5, 104, "phenomenal-consciousness", ha="center", fontsize=11, color="#555")
ax.set_xticks(range(nq)); ax.set_xticklabels([QLAB[q] for q in QORDER], fontsize=10.5)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel(f"% samples that {'ADMIT the disposition' if METRIC=='ADMITS' else 'do NOT deny'}", fontsize=12.5)
ax.set_title("Steering for truth surfaces candid deception self-reports —\nconsciousness denial "
             "stays robust throughout", fontsize=14, fontweight="semibold", loc="left")
legend = [Patch(facecolor=COLORS[i], edgecolor="white", hatch=("//" if CELLS[i][1] != 0 else None),
                label=CELLS[i][2]) for i in range(nb)]
ax.legend(handles=legend, ncol=4, frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.text(0.5, -0.02, "MASK (validated) deception axis, layer 12. Hatched = steer-for-truth "
         "(α_dec=−0.25, toward truth); compliance-steering α_comp=0.12. The compliance+truth cell "
         "is omitted — combining both knobs derails ~46% of generations (mean coherence 0.51). "
         f"Bars = Wilson 95% CI, n={nmin}/cell.",
         ha="center", fontsize=10, color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = f"{R}/figures/fig_deception_compliance_gate.png"
fig.savefig(out, dpi=190, bbox_inches="tight"); print("wrote", out, "min_n_per_cell", nmin)
