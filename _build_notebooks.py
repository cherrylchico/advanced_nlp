"""One-off generator for the four person templates. Run once, then delete if you like."""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


HEADER = lambda person, role: md(
    f"# 22DM015 Final Project — Financial PhraseBank",
    f"## {person}: {role}",
    "",
    "**Dataset:** `takala/financial_phrasebank`, config `sentences_allagree` (2,264 sentences).",
    "**Labels:** 0 = negative, 1 = neutral, 2 = positive.",
    "",
    "### Shared data contract (set by Person D — do NOT re-split)",
    "- Splits are committed under `data/` as CSVs: `train` (1584), `val` (227), `test` (453), `labeled_32` (32).",
    "- The **32 labelled** examples are a balanced sample from train (11 neg / 10 neu / 11 pos).",
    "- Part 2 'unlabelled' data = train minus the 32 (`unlabeled_pool`).",
    "- Evaluate headline numbers on **`test`** only; tune on **`val`**. Use `eval_utils.evaluate` so we're comparable.",
    "- Log every result with `eval_utils.log_result(...)` into `results/results.csv`.",
    "",
    "> **AI disclosure:** any AI-generated code/text in this notebook must be watermarked and declared "
    "(course rule). Interpretation, methodology justification, and analysis must be student-authored.",
)

SETUP = lambda person: code(
    "# --- Reproducibility seed (required by the assignment) ---",
    "import os, random, sys",
    "import numpy as np",
    "SEED = 618",
    "random.seed(SEED); np.random.seed(SEED)",
    "os.environ['PYTHONHASHSEED'] = str(SEED)",
    "",
    "# Make the shared helpers importable (they live in the repo root, one level up).",
    "sys.path.append(os.path.abspath('..'))",
    "import data_utils as du",
    "import eval_utils as eu",
    "",
    "splits = du.load_splits()            # identical data for everyone",
    "train, val, test = splits['train'], splits['val'], splits['test']",
    "labeled_32 = splits['labeled_32']",
    "pool = du.unlabeled_pool(splits)     # Part 2 'unlabelled' data",
    f"PERSON = '{person}'",
    "for k, v in splits.items():",
    "    print(f'{k:11s} {len(v):5d}', dict(v['label_name'].value_counts()))",
)

# ----------------------------------------------------------------------------- A
A = notebook([
    HEADER("Person A", "Part 1 — Setting Up the Problem (1.5 pts)"),
    SETUP("A"),
    md("## 1a. Bibliography & State of the Art (0.25)",
       "_Student-authored markdown._ Task = 3-class financial sentiment on Financial PhraseBank "
       "(Malo et al., 2014). Note SOTA references (e.g. FinBERT, Araci 2019) and benchmarks. "
       "Keep AI out of the written analysis."),
    md("## 1b. Dataset Description (0.5)",
       "Size, class distribution, peculiarities (neutral-heavy ~61%; short single-sentence headlines). "
       "Add basic descriptive stats below."),
    code("# Descriptive stats: class balance, sentence length distribution, etc.",
         "import pandas as pd",
         "full = pd.concat([train, val, test], ignore_index=True)",
         "print(full['label_name'].value_counts(normalize=True).round(3))",
         "full['n_words'] = full['text'].str.split().map(len)",
         "print(full['n_words'].describe())",
         "# TODO: plot class distribution and length histogram"),
    md("## 1c. Random Classifier Performance (0.25)",
       "Expected performance of a random classifier, *with an implementation*. "
       "Two natural baselines: uniform (1/3 each) and prior-weighted (sample by class frequency). "
       "Report accuracy and macro-F1 on test so it's comparable to the real models."),
    code("# Random baselines on the TEST set",
         "import numpy as np",
         "y_test = test['label'].to_numpy()",
         "priors = train['label'].value_counts(normalize=True).sort_index().to_numpy()",
         "rng = np.random.default_rng(SEED)",
         "y_uniform = rng.integers(0, 3, size=len(y_test))",
         "y_prior   = rng.choice([0,1,2], size=len(y_test), p=priors)",
         "print('uniform  ', eu.evaluate(y_test, y_uniform))",
         "print('prior-wt ', eu.evaluate(y_test, y_prior))",
         "eu.log_result('random-uniform', 'baseline', 0, eu.evaluate(y_test, y_uniform), person=PERSON)",
         "eu.log_result('random-prior',   'baseline', 0, eu.evaluate(y_test, y_prior),   person=PERSON)"),
    md("## 1d. Rule-Based Baseline (0.5)",
       "Build a lexicon/rule classifier (e.g. count positive vs negative finance terms; default neutral). "
       "Discuss its performance vs dataset complexity and (if available) human agreement levels."),
    code("# Minimal lexicon rule-based classifier — extend the word lists.",
         "POS = {'increase','rose','growth','profit','up','gain','higher','improved','strong'}",
         "NEG = {'decrease','fell','loss','down','decline','lower','weak','cut','drop'}",
         "def rule_predict(text):",
         "    t = text.lower().split()",
         "    p, n = sum(w in POS for w in t), sum(w in NEG for w in t)",
         "    if p > n: return 2",
         "    if n > p: return 0",
         "    return 1  # default neutral",
         "y_pred = test['text'].map(rule_predict).to_numpy()",
         "m = eu.evaluate(y_test, y_pred); print(m)",
         "eu.log_result('rule-based', 'baseline', 0, m, person=PERSON, notes='lexicon')"),
])

# ----------------------------------------------------------------------------- B
B = notebook([
    HEADER("Person B", "Parts 2 & 3 — BERT track (BERT, augmentation eval, full-data curve)"),
    SETUP("B"),
    md("> **Install (run once):** `transformers`, `torch`, `accelerate` are needed here. "
       "On Python 3.14 torch wheels may be missing — use a 3.11/3.12 venv."),
    md("## Part 2a. BERT with 32 labelled examples (0.5)",
       "Fine-tune a BERT-family model on `labeled_32`, evaluate on `test`. "
       "Expect instability with 32 examples — fix seed, report val + test."),
    code("# Skeleton — fill in tokenizer/model/Trainer.",
         "MODEL = 'bert-base-uncased'   # or 'ProsusAI/finbert' for the SOA comparison",
         "# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments",
         "# tok = AutoTokenizer.from_pretrained(MODEL)",
         "# ... tokenize labeled_32 / val / test, train, predict ...",
         "# y_pred = ...",
         "# m = eu.evaluate(test['label'], y_pred)",
         "# eu.log_result(MODEL, '32-shot', len(labeled_32), m, person=PERSON)"),
    md("## Part 2b. Train on Person D's augmented set (1)",
       "Person D produces a non-LLM augmented training set (back-translation / EDA / etc.) under "
       "`data/augmented_32.csv`. Re-train the SAME BERT on it and compare to 2a."),
    code("# aug = pd.read_csv('../data/augmented_32.csv')   # produced by Person D",
         "# train BERT on aug, evaluate on test",
         "# eu.log_result(MODEL, 'augmented', len(aug), m, person=PERSON)"),
    md("## Part 2d. Train on 32 + Person D's LLM-generated data (1)",
       "Person D produces `data/llm_generated.csv`. Train BERT on the 32 + generated points; "
       "analyse impact on metrics."),
    code("# gen = pd.read_csv('../data/llm_generated.csv')",
         "# combine labeled_32 + gen, train, evaluate",
         "# eu.log_result(MODEL, 'llm-generated', 32+len(gen), m, person=PERSON)"),
    md("## Part 2e. Optimal technique (0.5)",
       "Apply the best technique(s) from 2a/2b/2d. Comment + propose improvements (student-authored)."),
    md("## Part 3. Full-dataset SOA comparison (2)",
       "3a. Train on 1/10/25/50/75/100% of `train` (use `du.subset_by_fraction`). "
       "3b. Plot the learning curve. 3c. Fold in Part 2 techniques. 3d. Methodology analysis."),
    code("FRACTIONS = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00]",
         "for f in FRACTIONS:",
         "    sub = du.subset_by_fraction(train, f)",
         "    print(f'frac={f:>4}: n={len(sub)}', dict(sub['label_name'].value_counts()))",
         "    # train BERT on `sub`, predict on test",
         "    # m = eu.evaluate(test['label'], y_pred)",
         "    # eu.log_result(MODEL, f'full-{int(f*100)}%', len(sub), m, person=PERSON)"),
    code("# 3b. Learning curve — read everyone's rows back and plot.",
         "res = pd.read_csv('../results/results.csv')",
         "print(res)",
         "# import matplotlib.pyplot as plt; plot f1_macro vs n_train_labeled"),
])

# ----------------------------------------------------------------------------- C
C = notebook([
    HEADER("Person C", "Parts 2 & 3 — Zero-Shot LLM track"),
    SETUP("C"),
    md("> **Note:** declare which LLM + version you call, and watermark AI-generated code. "
       "Keep prompts in the notebook for reproducibility; cache responses to avoid re-billing."),
    md("## Part 2c. Zero-Shot Learning with an LLM (0.5)",
       "Classify the **test** set zero-shot (no training examples in the prompt). "
       "Map free-text LLM output to {negative, neutral, positive} robustly. Document the prompt."),
    code("# Pseudocode — wire up your LLM client (Claude / GPT / Mistral / ...).",
         "LLM = 'claude-...'   # record exact model id",
         "PROMPT = (",
         "    'Classify the financial sentence sentiment as exactly one of: '",
         "    'negative, neutral, positive. Reply with only that word.\\n\\nSentence: {text}'",
         ")",
         "def classify_zero_shot(text):",
         "    ...  # call LLM, parse -> du.NAME2ID[word]",
         "    raise NotImplementedError",
         "# y_pred = test['text'].map(classify_zero_shot).to_numpy()",
         "# m = eu.evaluate(test['label'], y_pred)",
         "# eu.log_result(LLM, 'zero-shot', 0, m, person=PERSON)"),
    md("## (Optional) Few-shot with the 32 labelled examples",
       "Same LLM but put a few of `labeled_32` in the prompt as in-context examples; compare to zero-shot. "
       "Useful for the Part 3 methodology discussion."),
    code("# build few-shot prompt from labeled_32, evaluate on test, log_result(..., 'few-shot', ...)"),
    md("## Part 3 contribution — LLM vs trained models",
       "Your zero-/few-shot numbers slot into the same `results/results.csv`, so the Part 3 learning-curve "
       "plot can show the LLM as a horizontal reference line vs BERT's data-scaling curve. "
       "Discuss cost/latency/accuracy tradeoffs (student-authored)."),
])

# ----------------------------------------------------------------------------- D
D = notebook([
    HEADER("Person D", "Data prep + Part 2 augmentation/generation + Part 4 distillation"),
    md("This notebook is the **source of truth for the data**. Running the build cell regenerates the "
       "CSVs under `data/` that A/B/C consume. Commit those CSVs so everyone is byte-identical."),
    SETUP("D"),
    md("## 0. Build the shared splits (run once, then commit `data/`)",
       "Stratified 70/10/20 with SEED=618; balanced 32-shot (11 neg / 10 neu / 11 pos) from train. "
       "Logic lives in `data_utils.build_splits()` so it's auditable and reproducible."),
    code("# Regenerate the canonical CSVs. Safe to re-run: deterministic given SEED.",
         "splits = du.build_splits()",
         "for k, v in splits.items():",
         "    print(f'{k:11s} n={len(v):4d}', dict(v['label_name'].value_counts().reindex(du.LABEL_NAMES)))",
         "# sanity checks",
         "tr, te, va = set(splits['train'].id), set(splits['test'].id), set(splits['val'].id)",
         "assert not (tr & te) and not (tr & va) and not (te & va), 'split leakage!'",
         "assert set(splits['labeled_32'].id) <= tr, '32-shot must come from train'",
         "print('OK: no leakage, 32-shot ⊂ train')"),
    md("## Part 2b. Dataset Augmentation WITHOUT an LLM (1)",
       "Automated augmentation of the 32 labelled examples — e.g. EDA (synonym replace, random "
       "insert/swap/delete) or back-translation. Output `data/augmented_32.csv` with the SAME schema "
       "(id,text,label,label_name) for Person B to train on. Keep label consistency.",
       "_Name the tradeoff: back-translation preserves meaning better; EDA is cheaper but noisier._"),
    code("# TODO: implement augmentation (e.g. nlpaug / nltk wordnet / MarianMT back-translation).",
         "# Keep new rows labelled with the source row's label. Give synthetic ids (e.g. 100000+).",
         "# aug.to_csv('../data/augmented_32.csv', index=False)"),
    md("## Part 2d. Data Generation WITH an LLM (1)",
       "Prompt an LLM to generate new labelled financial sentences (balanced across classes). "
       "Save `data/llm_generated.csv` (same schema). Declare the model + watermark. "
       "Person B trains on 32 + generated and reports the metric delta."),
    code("# TODO: LLM generation. Validate generated labels, dedupe vs train/test to avoid leakage.",
         "# gen.to_csv('../data/llm_generated.csv', index=False)",
         "# Guard against leakage:",
         "# assert gen['text'].isin(set(test['text'])).sum() == 0"),
    md("## Part 4. Model Distillation / Quantization (3) — start thinking",
       "Take the best model (likely Person B's fine-tuned BERT) and produce a lighter student.",
       "- **4a (1.5):** distillation (e.g. DistilBERT student via logit/feature distillation) and/or "
       "quantization (dynamic int8 with `torch.quantization`, or bitsandbytes). Document tools.",
       "- **4b (0.5):** compare student vs teacher on `test` — macro-F1 *and* inference speed / size.",
       "- **4c (1):** analyse where the student degrades; propose improvements.",
       "Reuse `eu.evaluate` + `eu.log_result('distilbert-student','distilled',...)` so it lands in the same table."),
    code("# Placeholder for Part 4 — depends on Person B's trained teacher checkpoint.",
         "# Plan: load teacher, distill to DistilBERT student, then time inference and compare size.",
         "print('Part 4 scaffold — fill in once a teacher checkpoint exists')"),
])

for name, nb in [("part1", A), ("bert_part2_3", B),
                 ("zeroshot_part2_3", C), ("dataprep_aug", D)]:
    path = NB_DIR / f"{name}.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    print("wrote", path)
