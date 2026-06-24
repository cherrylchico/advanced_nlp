"""Progressive bar charts for the '32 labels' story (4 cumulative stages).

Four metric panels (F1-macro, F1-negative, F1-neutral, F1-positive). The x-axis
holds all model slots in a fixed position; each stage reveals more so the charts
'fill in' across slides. Each data condition (32-shot, back-translation, LLM-gen)
is a probe-vs-full PAIR:  orange = linear probe (frozen encoder), blue = full
fine-tune. The three references (random / rule-based / zero-shot) are grey.

Matched n within each pair so probe vs full is a fair comparison
(32-shot: n=32; back-translation: n=229; LLM-gen: n=229).

Numbers come straight from results/results.csv (test split).
Run from repo root:  python figures/make_models32.py
"""
# watermark: AGLLM (AI-assisted content disclosure)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "results.csv")

def row(model, method):
    r = df[(df.model == model) & (df.method == method)].iloc[0]
    return [r.f1_macro, r.f1_negative, r.f1_neutral, r.f1_positive]

GREY, ORANGE, BLUE = "#6b7280", "#ea580c", "#2563eb"

# (label, values, colour)
MODELS = [
    ("Random\n(prior)",     row("random-prior", "baseline"),            GREY),
    ("Rule-based",          row("rule-based", "baseline"),              GREY),
    ("Zero-shot\n(Claude)", row("claude-opus-4-8", "zero-shot"),        GREY),
    ("32-shot\nprobe",      row("bert-base-uncased", "32-shot-frozen"), ORANGE),
    ("32-shot\nfull",       row("bert-base-uncased", "32-shot"),        BLUE),
    ("BT\nprobe",           row("bert-base-uncased", "augmented-frozen"), ORANGE),
    ("BT\nfull",            row("bert-base-uncased", "augmented"),      BLUE),
    ("LLM-gen\nprobe",      row("bert-base-uncased", "llm-generated-frozen"), ORANGE),
    ("LLM-gen\nfull",       row("bert-base-uncased", "llm-generated"),  BLUE),
]
NAMES = [m[0] for m in MODELS]
VALS = [m[1] for m in MODELS]
COLS = [m[2] for m in MODELS]
METRICS = ["F1-macro", "F1-negative", "F1-neutral", "F1-positive"]

# cumulative revealed indices per stage
STAGES = [
    [0, 1, 2],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
]
x = list(range(len(MODELS)))

for s, shown in enumerate(STAGES, start=1):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160)
    for mi, (ax, metric) in enumerate(zip(axes.ravel(), METRICS)):
        heights = [VALS[i][mi] if i in shown else 0 for i in x]
        colors = [COLS[i] for i in x]
        ax.bar(x, heights, color=colors, width=0.78, edgecolor='white')
        for i in shown:
            ax.text(i, VALS[i][mi] + 0.02, f"{VALS[i][mi]:.2f}",
                    ha='center', va='bottom', fontsize=7.5)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(NAMES, fontsize=6.8)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis='y', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Patch(color=ORANGE, label='Linear probe (frozen)'),
               Patch(color=BLUE, label='Full fine-tune'),
               Patch(color=GREY, label='Reference')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("From 32 labels: per-class F1 — linear probe vs full fine-tune", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = ROOT / "figures" / f"models32_stage{s}.png"
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", out.name)
