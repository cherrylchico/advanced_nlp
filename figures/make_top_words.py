"""Top-25 words per class, coloured by each word's inherent sentiment.

Reproduces figures/top_words_sentiment.png used in the presentation.
Tokenisation matches notebooks/part1.ipynb (lowercase alphabetic tokens,
sklearn ENGLISH_STOP_WORDS, length > 2).

Colour key:
  green = inherently positive   red = inherently negative   grey = neutral / no polarity

Word lists:
  - directional words: the rule-based lexicon (POS_DIR / NEG_DIR), from part1.
  - financial polarity: matches the part1 discussion — profit, sales, earnings,
    growth margin = positive; loss = negative.

Run from the repo root:  python figures/make_top_words.py
"""
# watermark: AGLLM (AI-assisted content disclosure)
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
OUT = ROOT / "figures" / "top_words_sentiment.png"

# directional words (from the rule-based lexicon, part1)
POS_DIR = {'increase','increased','increasing','rise','rose','risen','rising','grow','grew',
    'grown','growing','growth','climbed','jumped','surged','soared','soaring','rebound',
    'rebounding','expand','expanded','expanding','expansion','doubled','up','higher'}
NEG_DIR = {'decrease','decreased','fall','fell','fallen','falling','decline','declined',
    'declining','drop','dropped','dropping','slumped','slowed','slowing','slipped','cut',
    'cuts','cutting','reduce','reduced','reduction','weakened','down','lower'}
# financial polarity — matches the part1 discussion:
# inherently positive = profit, sales, earnings, growth margin ; inherently negative = loss
POS_FIN = {'profit','profits','profitable','profitability',
    'sales','earnings','earning','margin','margins'}
NEG_FIN = {'loss','losses'}
POS_WORDS = POS_DIR | POS_FIN
NEG_WORDS = NEG_DIR | NEG_FIN

GREEN, RED, GRAY = '#5cb85c', '#d9534f', '#9ca3af'
def color_for(w):
    return GREEN if w in POS_WORDS else RED if w in NEG_WORDS else GRAY

def top_words(df, label_name, top_n=25):
    toks = []
    for text in df[df['label_name'] == label_name]['text']:
        t = re.findall(r'\b[a-z]+\b', str(text).lower())
        toks += [w for w in t if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return Counter(toks).most_common(top_n)

fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), dpi=170)
for ax, label in zip(axes, ['negative', 'neutral', 'positive']):
    words, counts = zip(*top_words(train, label))
    words, counts = list(words)[::-1], list(counts)[::-1]
    ax.barh(words, counts, color=[color_for(w) for w in words], edgecolor='white')
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines[['top', 'right']].set_visible(False)

handles = [Patch(color=GREEN, label='positive'), Patch(color=RED, label='negative'),
           Patch(color=GRAY, label='neutral')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=12,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Top 25 words per class, coloured by inherent sentiment', fontsize=15)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(OUT, bbox_inches='tight', facecolor='white')
print(f"saved {OUT}")
