"""Part 4 compression — box-and-whisker of per-class F1 across 5 seeds.

Four metric panels (F1-macro, F1-negative, F1-neutral, F1-positive); within each,
a box per model (Teacher BERT / DistilBERT / int8 quantized) over 5 seeds, with the
individual seed points overlaid.

Numbers come straight from results/part4_seeds.csv.
Run from repo root:  python figures/make_part4_box.py
"""
# watermark: AGLLM (AI-assisted content disclosure)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "part4_seeds.csv")

MODELS = ["bert-base", "distilbert", "quantized"]
LABELS = ["Teacher\nBERT", "DistilBERT\n(student)", "int8\n(quant.)"]
COLORS = ["#6b7280", "#16a34a", "#ea580c"]
METRICS = [("F1-macro", "f1_macro"), ("F1-negative", "f1_negative"),
           ("F1-neutral", "f1_neutral"), ("F1-positive", "f1_positive")]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160)
for ax, (title, field) in zip(axes.ravel(), METRICS):
    data = [df[df.model == m][field].values for m in MODELS]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showmeans=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch, c in zip(bp["boxes"], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for i, (vals, c) in enumerate(zip(data, COLORS), start=1):
        ax.scatter([i] * len(vals), vals, color=c, s=18, zorder=3, edgecolor="white", lw=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(LABELS, fontsize=9)
    lo = min(v.min() for v in data)
    hi = max(v.max() for v in data)
    pad = (hi - lo) * 0.25 or 0.01
    ax.set_ylim(lo - pad, hi + pad)        # per-panel zoom so the spread is visible
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Compression — per-class F1 over 5 seeds", fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / "figures" / "part4_box.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out.name)
