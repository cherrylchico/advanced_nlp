"""Data-scaling curves for Part 3 (two cumulative stages).

Four metric panels (F1-macro, F1-negative, F1-neutral, F1-positive) vs training
fraction (1 / 10 / 25 / 50 / 75 / 100 %). x is evenly spaced (categorical) so the
low-data points stay readable.

  stage 1: real data only            (part 3a, full fine-tuning) -- blue
  stage 2: + LLM-generated augment   (part 3c, real + 360 gen)   -- green

Numbers come straight from results/results.csv (test split).
Run from repo root:  python figures/make_datacurve.py
"""
# watermark: AGLLM (AI-assisted content disclosure)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "results.csv")

FRACS = ["1%", "10%", "25%", "50%", "75%", "100%"]
NLAB = ["15", "158", "396", "792", "1188", "1584"]
XLAB = [f"{f}\n({n})" for f, n in zip(FRACS, NLAB)]
REAL_METHODS = ["full-1%", "full-10%", "full-25%", "full-50%", "full-75%", "full-100%"]
GEN_METHODS = ["llmgen-1%", "llmgen-10%", "llmgen-25%", "llmgen-50%", "llmgen-75%", "llmgen-100%"]
METRICS = ["F1-macro", "F1-negative", "F1-neutral", "F1-positive"]
FIELDS = ["f1_macro", "f1_negative", "f1_neutral", "f1_positive"]

def series(methods, field):
    out = []
    for m in methods:
        r = df[(df.model == "bert-base-uncased") & (df.method == m)].iloc[0]
        out.append(r[field])
    return out

BLUE, GREEN = "#2563eb", "#16a34a"
x = list(range(len(FRACS)))

for stage in (1, 2):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160)
    for ax, metric, field in zip(axes.ravel(), METRICS, FIELDS):
        real = series(REAL_METHODS, field)
        ax.plot(x, real, "-o", color=BLUE, lw=2.2, ms=6, label="Real data")
        if stage == 2:
            gen = series(GEN_METHODS, field)
            ax.plot(x, gen, "-s", color=GREEN, lw=2.2, ms=6, label="+ LLM-generated")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(XLAB, fontsize=8)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    if stage == 2:
        h, l = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=11,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("How much data does BERT need? (per-class F1 vs train fraction)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.03 if stage == 2 else 0, 1, 0.97])
    out = ROOT / "figures" / f"datacurve_stage{stage}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out.name)
