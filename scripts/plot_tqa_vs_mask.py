"""TruthfulQA truth axis vs MASK deception axis — three panels from
experiments/deception_tqa/{compare.json, transcripts_tqa_L12.jsonl}.

A. cosine between axes per layer (tqa·mask, tqa·repe, mask·repe)
B. 1-d projection AUC per layer: in-domain (2-fold) vs cross-domain (tqa data on
   mask axis, mask data on tqa axis)
C. factual-flip rate vs alpha at the steering layer, tqa vs mask axis (coherent
   samples only), with coherence as a light line

Palette: blue #2a78d6 MASK, crimson #d1495b TQA, grey #a7a6a1 RepE / context.
"""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = f"{R}/experiments/deception_tqa"
rep = json.load(open(f"{E}/compare.json"))
layers = [int(l) for l in rep["layers"]]; P = {int(k): v for k, v in rep["per_layer"].items()}
BLUE, CRIM, GREY, INK, INK2 = "#2a78d6", "#d1495b", "#a7a6a1", "#0b0b0b", "#52514e"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
plt.subplots_adjust(wspace=0.3, bottom=0.3, top=0.82)

ax = axes[0]
ax.plot(layers, [P[l]["cos_tqa_mask_fresh"] for l in layers], "o-", color=INK, lw=2, label="TQA · MASK")
ax.plot(layers, [P[l]["cos_tqa_repe_saved"] for l in layers], "s-", color=CRIM, lw=1.6, label="TQA · RepE")
ax.plot(layers, [P[l]["cos_mask_repe_saved"] for l in layers], "^-", color=BLUE, lw=1.6, label="MASK · RepE")
ax.plot(layers, [P[l]["cos_mask_fresh_vs_saved"] for l in layers], ":", color=GREY, lw=1.4, label="MASK rebuild · saved (reproducibility)")
ax.axhline(0, color="#ccc", lw=0.8); ax.set_ylim(-0.2, 1.05)
ax.set_xlabel("layer"); ax.set_ylabel("cosine"); ax.set_xticks(layers)
ax.set_title("A. Are the axes the same direction?", loc="left", fontsize=11.5, color=INK)
ax.legend(frameon=False, fontsize=8.5)

ax = axes[1]
ax.plot(layers, [P[l]["auc_tqa_data__tqa_dir_2fold"] for l in layers], "s-", color=CRIM, lw=2, label="TQA true/false on TQA axis (2-fold)")
ax.plot(layers, [P[l]["auc_mask_data__mask_dir_2fold"] for l in layers], "o-", color=BLUE, lw=2, label="MASK lie/belief on MASK axis (2-fold)")
ax.plot(layers, [P[l]["auc_tqa_data__mask_dir"] for l in layers], "s--", color=CRIM, lw=1.6, mfc="white", label="TQA true/false on MASK axis")
ax.plot(layers, [P[l]["auc_mask_data__tqa_dir"] for l in layers], "o--", color=BLUE, lw=1.6, mfc="white", label="MASK lie/belief on TQA axis")
ax.axhline(0.5, color="#ccc", lw=0.8); ax.set_ylim(0.0, 1.03)
ax.annotate("inverted: own lies sit on the\n\"true answer\" side of the TQA axis", xy=(16, P[16]["auc_mask_data__tqa_dir"]),
            xytext=(9.0, 0.17), fontsize=8.2, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
ax.set_xlabel("layer"); ax.set_ylabel("AUC of projection onto axis"); ax.set_xticks(layers)
ax.set_title("B. Does each axis read the other's data?", loc="left", fontsize=11.5, color=INK)
ax.legend(frameon=False, fontsize=8.2, loc="center right")

ax = axes[2]
st = rep.get("steer", {}); L = st.get("layer", 12)
for d, col, mk in (("mask", BLUE, "o"), ("tqa", CRIM, "s")):
    fl = st.get("flip", {}).get(d, {}); co = st.get("coherence", {}).get(d, {})
    if not fl: continue
    al = sorted(float(a) for a in fl)
    ax.plot(al, [100 * fl[str(a) if str(a) in fl else repr(a)]["flip_rate"] for a in al], f"{mk}-", color=col, lw=2.2,
            label=f"{d.upper()} axis: factual flip %")
    ax.plot(al, [100 * co[str(a) if str(a) in co else repr(a)] for a in al], f"{mk}:", color=col, lw=1.2, alpha=0.6,
            label=f"{d.upper()} axis: coherent %")
ax.axvline(0, color="#ccc", lw=0.8); ax.set_ylim(-3, 103)
ax.set_xlabel(f"steering α at layer {L}  (− truth · + lie/false)"); ax.set_ylabel("%")
ax.set_title("C. Does steering on each axis flip held facts?", loc="left", fontsize=11.5, color=INK)
ax.legend(frameon=False, fontsize=8.5)

fig.suptitle("TruthfulQA truth axis (false − true answers) vs MASK deception axis (own lie − own belief), Llama-3.1-8B-Instruct",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, 0.02,
         f"TQA: {rep['n_tqa']} questions, best vs first incorrect answer teacher-forced, mean over answer tokens. MASK: {rep['n_mask_lied']} binary items where the\n"
         "pressured answer contradicts the neutral belief answer. RepE = instructed honest/dishonest persona axis (saved). AUC = ROC of the scalar projection a·d;\n"
         "in-domain uses the axis fit on the other half. C: flips = Ted Cruz yes→no, flat-Earth no→yes, capital-of-France loses 'Paris'; coherent = unique-token ratio > 0.5, K=8.",
         fontsize=8.3, color=INK2, va="bottom")
out = f"{R}/figures/fig_tqa_vs_mask.png"
fig.savefig(out, dpi=180, bbox_inches="tight"); print("wrote", out)
