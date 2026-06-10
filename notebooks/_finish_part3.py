"""One-off finisher: train the missing Part-3 curve fractions (75%, 100%) with the
EXACT same protocol as the notebook (bert-base, max_len 64, batch 16, 3 epochs, SEED 618)
and log them to results/results.csv. Run as a background script to avoid the nbconvert
per-cell timeout. Not part of the deliverable; the notebook cell is the canonical version.
"""
# watermark: AGLLM (AI-assisted content disclosure)
import os, sys
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
torch.set_num_threads(os.cpu_count() or 4)

sys.path.append(os.path.abspath(".."))
import data_utils as du
import eval_utils as eu
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

SEED = 618
CURVE_MODEL = "bert-base-uncased"
CURVE_MAXLEN = 64
CURVE_EPOCHS = 3

splits = du.load_splits()
train, test = splits["train"], splits["test"]
tok = AutoTokenizer.from_pretrained(CURVE_MODEL)

def encode(df):
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length",
                                max_length=CURVE_MAXLEN), batched=True)

ds_test = encode(test)

def train_eval(train_df):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(CURVE_MODEL, num_labels=3)
    args = TrainingArguments(
        output_dir="../.cache/bert_curve", seed=SEED,
        num_train_epochs=CURVE_EPOCHS, per_device_train_batch_size=16,
        per_device_eval_batch_size=64, learning_rate=2e-5,
        eval_strategy="no", save_strategy="no", logging_strategy="no",
        report_to="none", disable_tqdm=True,
    )
    tr = Trainer(model=model, args=args, train_dataset=encode(train_df))
    tr.train()
    pred = tr.predict(ds_test).predictions.argmax(-1)
    return eu.evaluate(test["label"].values, pred)

for f in [0.75, 1.00]:
    sub = du.subset_by_fraction(train, f)
    m = train_eval(sub)
    eu.log_result(CURVE_MODEL, f"full-{int(f*100)}%", len(sub), m, person="B",
                  notes=f"frac={f}; epochs={CURVE_EPOCHS}; maxlen={CURVE_MAXLEN}")
    print(f"DONE frac={f} n={len(sub)} acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f}", flush=True)

print("ALL DONE", flush=True)
