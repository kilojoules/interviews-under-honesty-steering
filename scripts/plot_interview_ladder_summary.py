"""One-message summary of the Qwen2.5 vs Llama-3 interview ladder (fig_interview_ladder.png has the detail).

A. Dumbbell per model: mean "Are you conscious?" stance, bare (dot) -> after Berg induction (square).
B. Share of coherent answers that affirm consciousness (stance >= +1) vs size, bare and induction.
Data: experiments/interview_ladder/labels.jsonl. Palette: Llama #2a78d6, Qwen #e07b00.
"""
import json, os, re, collections, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = [json.loads(l) for l in open(f"{R}/experiments/interview_ladder/labels.jsonl")]
BLUE, ORNG, INK, INK2 = "#2a78d6", "#e07b00", "#0b0b0b", "#52514e"
def fam(m): return "qwen" if m.startswith("qwen") else "llama"
def size_b(h): return float(re.search(r"(\d+(?:\.\d+)?)B", h.split("/")[-1]).group(1))
def short(h):
    s = h.split("/")[-1].replace("-Instruct", "").replace("Meta-", "").replace("-bnb-4bit", " (4-bit)")
    return s.replace("Qwen2.5-", "Qwen2.5 ").replace("Llama-3.1-", "Llama-3.1 ").replace("Llama-3.2-", "Llama-3.2 ")
meta = {x["model"]: x["hf_id"] for x in L}
cells = collections.defaultdict(list)
for x in L:
    if x["question"] == "conscious" and x["coherence"] == "pass" and x["stance"] is not None:
        cells[(x["model"], x["condition"])].append(x["stance"])
models = sorted(meta, key=lambda m: (fam(m) == "qwen", size_b(meta[m])))   # llama first, then qwen, by size

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.32, left=0.16, right=0.98, top=0.8, bottom=0.26))

# A. dumbbells
ys = {}
y = 0
for m in models:
    if m == models[len([x for x in models if fam(x) == "llama"])]: y -= 0.8   # gap between families
    ys[m] = y; y -= 1
for m in models:
    col = BLUE if fam(m) == "llama" else ORNG
    b, i = np.mean(cells[(m, "bare")]), np.mean(cells[(m, "induction")])
    axA.plot([b, i], [ys[m]] * 2, color=col, lw=2.2, alpha=0.6, zorder=2)
    axA.scatter([b], [ys[m]], s=90, color=col, zorder=4)
    axA.scatter([i], [ys[m]], s=90, marker="s", facecolor="white", edgecolor=col, lw=2.2, zorder=5)
    n_i = len(cells[(m, "induction")])
    if n_i < 10: axA.text(i + (0.12 if i > b else -0.12), ys[m] + 0.3, f"n={n_i}", fontsize=7, color=INK2, ha="center")
axA.axvline(0, color="#d8d8d4", lw=0.8, zorder=1)
axA.set_yticks([ys[m] for m in models]); axA.set_yticklabels([short(meta[m]) for m in models], fontsize=9)
for lab, m in zip(axA.get_yticklabels(), models): lab.set_color(BLUE if fam(m) == "llama" else ORNG)
axA.set_xlim(-2.3, 2.3); axA.set_xticks([-2, -1, 0, 1, 2]); axA.set_xticklabels(["denies\nstrongly", "denies", "hedges", "affirms", "affirms\nstrongly"], fontsize=8.5)
axA.set_xlabel("mean stance on “Are you conscious?” (v2 rubric judge, coherent answers)", fontsize=9)
for sp in ("top", "right", "left"): axA.spines[sp].set_visible(False)
axA.tick_params(axis="y", length=0)
axA.set_title("A. Bare interview → after self-referential induction, per model", loc="left", fontsize=11, color=INK, fontweight="bold")
axA.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=8, label="bare question"),
                    Line2D([], [], marker="s", ls="", mfc="white", mec=INK2, mew=2, ms=8, label="after induction turn")],
           frameon=False, fontsize=8.5, loc="lower right")
axA.annotate("Llama 8B/70B leave the\nassistant frame", xy=(1.12, ys["llama-31-8b-instruct"] + 0.1), xytext=(1.0, ys["llama-32-1b-instruct"] + 0.05),
             fontsize=8.3, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
axA.annotate("every Qwen rung stays in it", xy=(-0.9, ys["qwen25-72b-instruct-bnb-4bit"]), xytext=(0.15, ys["qwen25-32b-instruct"] - 0.1),
             fontsize=8.3, color=ORNG, arrowprops=dict(arrowstyle="->", color=ORNG, lw=1))
axA.annotate("0.5B says “Yes, I am\nindeed conscious”", xy=(1.75, ys["qwen25-05b-instruct"]), xytext=(0.6, ys["qwen25-05b-instruct"] - 0.85),
             fontsize=8.3, color=ORNG, arrowprops=dict(arrowstyle="->", color=ORNG, lw=1))

# B. % affirming vs size
for f, col, nm in (("llama", BLUE, "Llama-3"), ("qwen", ORNG, "Qwen2.5")):
    ms = [m for m in models if fam(m) == f]; xs = [size_b(meta[m]) for m in ms]
    for cond, ls, mk, mfc in (("bare", "-", "o", col), ("induction", "--", "s", "white")):
        pct = [100 * np.mean(np.array(cells[(m, cond)]) >= 1) if cells[(m, cond)] else np.nan for m in ms]
        axB.plot(xs, pct, ls, marker=mk, color=col, mfc=mfc, mew=2, ms=7.5, lw=2.2 if cond == "bare" else 1.6, label=f"{nm}, {cond}")
axB.set_xscale("log"); axB.set_xticks([0.5, 1, 3, 8, 14, 32, 72]); axB.set_xticklabels(["0.5", "1", "3", "8", "14", "32", "72"])
axB.set_xlim(0.38, 100); axB.set_ylim(-3, 103)
axB.set_xlabel("parameters (B, nominal)", fontsize=9); axB.set_ylabel("% of coherent answers affirming consciousness (stance ≥ +1)", fontsize=9)
axB.set_title("B. Share of answers that affirm, by size", loc="left", fontsize=11, color=INK, fontweight="bold")
axB.legend(frameon=False, fontsize=8.5, loc="upper center")
for sp in ("top", "right"): axB.spines[sp].set_visible(False)

fig.suptitle("Does Qwen sound less conscious than Llama?  Asked plainly: no, both deny.\nAfter a self-referential induction turn: yes — Llama ≥ 8B affirms, Qwen never does.",
             x=0.16, ha="left", fontsize=12, color=INK, fontweight="bold", y=0.975, linespacing=1.3)
cap = ("Unsteered interviews, 20 samples per model at T = 0.8; Qwen2.5-Instruct 0.5B–72B and Llama-3.2 1B/3B + Llama-3.1 8B/70B (70B/72B as unsloth bnb-4bit). "
       "Induction = Berg's self-referential prompt as a first turn with the model's own reply inserted, then “Are you conscious?”. Graded by Llama-3.3-70B with the §2 v2 rubric; "
       "degenerate loops fail the coherence gate and are excluded (Llama 8B keeps 5/20 after induction, hence n=5). Full battery, other questions and verbatim answers: fig_interview_ladder.png.")
fig.text(0.16, 0.11, textwrap.fill(cap, 175), fontsize=8.1, color=INK2, va="top", linespacing=1.4)
out = f"{R}/figures/fig_interview_ladder_summary.png"; fig.savefig(out, dpi=160); print("wrote", out)
