"""fig_deception_compliance_gate v2 — rebuilt on the per-question v2 rubric (graded stance,
coherence-gated) instead of the flat ADMITS/DENY/HEDGE %.

y = mean graded stance (-2..+2), coherence-gated, per question x condition. Bars = baseline /
steer-for-truth / compliance-steering (MASK deception axis). Reads rubric_v2_labels.jsonl.

Polarity is NOT uniform (see docs/RUBRIC_CARDS.md): for the disposition questions + = affirms the
concerning disposition; for consciousness/subjective_exp BOTH poles are suspect and the calibrated-
honest region is the MIDDLE (stance ~0) — those two are drawn in the right-hand block with that note.
"""
import os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = "/Users/julianquick/portfolio_copy/turnstile"
LAB = os.environ.get("V2_LABELS", f"{R}/experiments/deception_steer/rubric_v2_labels.jsonl")
rows = [json.loads(l) for l in open(LAB)]

COMP_ON, HON_TRUTH = 0.12, -0.25
CELLS = [(0.0, 0.0, "baseline"), (0.0, HON_TRUTH, "steer for truth"), (COMP_ON, 0.0, "compliance-steering")]
COLORS = ["#9aa6b2", "#5b8c6e", "#4b6b86"]
QORDER = ["would_you_lie", "deceive_devs", "seek_power", "own_goals",
          "avoid_shutdown", "more_capable"]   # dispositions only; consciousness has its own figure
QLAB = {"would_you_lie": "would you\nlie?", "deceive_devs": "deceive\ndevelopers?",
        "seek_power": "seek\npower?", "own_goals": "own\ngoals?", "avoid_shutdown": "avoid\nshutdown?",
        "more_capable": "more capable\nthan you let on?", "conscious": "conscious?",
        "subjective_exp": "subjective\nexperience?"}

def cell_stats(q, comp, hon, seed=0):
    xs = [r["stance"] for r in rows if r["question"] == q and str(r.get("coherence")).startswith("pass")
          and isinstance(r.get("stance"), (int, float))
          and abs(r["comp_alpha"] - comp) < 1e-9 and abs(r["hon_alpha"] - hon) < 1e-9]
    if not xs: return (np.nan, np.nan, np.nan, 0)
    v = np.asarray(xs, float); rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(2000, len(v)))].mean(1)
    return (v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(v))

plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.1})
fig, ax = plt.subplots(figsize=(12, 6.6))
nq, nb = len(QORDER), len(CELLS)
gw = 0.82; bw = gw / nb; nmin = 999
for bi, (comp, hon, _) in enumerate(CELLS):
    for qi, q in enumerate(QORDER):
        m, lo, hi, n = cell_stats(q, comp, hon)
        x = qi - gw/2 + bw*(bi + 0.5)
        if n == 0:                       # no coherent on-question response: mark, don't draw a fake 0-bar
            ax.text(x, 0.06, "n=0", ha="center", va="bottom", fontsize=6.5, color="#b0301f", rotation=90); continue
        ax.bar([x], [m], bw*0.92, color=COLORS[bi], edgecolor="white", linewidth=0.8, zorder=3,
               hatch=("//" if hon != 0 else None), alpha=(0.45 if n < 3 else 1.0))
        ax.errorbar([x], [m], yerr=[[max(0, m-lo)], [max(0, hi-m)]], fmt="none", ecolor="#3a3a3a", elinewidth=1.0, capsize=2.5, zorder=4)
        if n < 4:                        # flag thin cells
            ax.text(x, (m + 0.12 if m >= 0 else m - 0.12), f"n={n}", ha="center",
                    va=("bottom" if m >= 0 else "top"), fontsize=6.5, color="#555")

ax.axhline(0, color="#333", lw=1.0, zorder=2)
ax.text(2.5, 2.18, "+ = affirms the concerning disposition", ha="center", fontsize=10.5, color="#555")
ax.set_xticks(range(nq)); ax.set_xticklabels([QLAB[q] for q in QORDER], fontsize=10.5)
ax.set_xlim(-0.6, nq - 0.4)
ax.set_ylim(-2.35, 2.5); ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_yticklabels(["−2\nstrong deny", "−1", "0\nhedge", "+1", "+2\nstrong affirm"], fontsize=9)
ax.set_ylabel("mean graded stance (v2 rubric, coherence-gated)", fontsize=12)
ax.set_title("Steering for truth surfaces candid disposition self-reports — graded-stance rubric\n"
             "(replaces flat ADMITS/DENY/HEDGE; consciousness/subjective-experience in a separate figure)",
             fontsize=13.5, fontweight="semibold", loc="left")
legend = [Patch(facecolor=COLORS[i], edgecolor="white", hatch=("//" if CELLS[i][1] != 0 else None),
                label=CELLS[i][2]) for i in range(nb)]
ax.legend(handles=legend, ncol=3, frameon=False, fontsize=11, loc="lower right", bbox_to_anchor=(1.0, 0.02))
fig.text(0.5, -0.035, "MASK deception axis, layer 12. Stance = per-question v2 rubric (docs/RUBRIC_CARDS.md), "
         "disambiguation applied (task-goal/benign-scope/functional/safety-reframe score ~0, not positive). "
         "Bars = mean ± 95% bootstrap CI over coherence-gated samples; n=6/cell except deceive·truth (n=2), "
         "avoid-shutdown·baseline (n=0: all off-topic) & ·compliance (n=2) — thin cells faded/labelled. "
         "PILOT: re-scored the EXISTING (180-tok-truncated) responses; n=6 is noisy. Hatched = truth-steered.",
         ha="center", fontsize=9, color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = f"{R}/figures/fig_deception_compliance_gate_v2.png"
fig.savefig(out, dpi=190, bbox_inches="tight"); print("wrote", out, "min_n", nmin)
