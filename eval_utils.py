"""Shared evaluation + results logging so all four notebooks are comparable.

Every model (rule-based, BERT, zero-shot, distilled) reports the SAME metrics on
the SAME test set, and logs one row into results/results.csv. Part 3 (comparison)
and Part 4 just read that file -- no need to re-run anyone else's model.

Convention: evaluate on the shared TEST split only for the headline numbers.
Use VAL for tuning/early stopping.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

LABEL_NAMES = ["negative", "neutral", "positive"]
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "results.csv"


def evaluate(y_true, y_pred) -> dict:
    """Accuracy, macro/weighted F1, and per-class F1. Macro-F1 is the headline
    metric here because the dataset is neutral-heavy and we care about minorities."""
    from sklearn.metrics import accuracy_score, f1_score
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    for i, name in enumerate(LABEL_NAMES):
        out[f"f1_{name}"] = per_class[i]
    return out


def log_result(model: str, method: str, n_train_labeled, metrics: dict,
               split: str = "test", person: str = "", notes: str = "",
               path: Path | str = RESULTS_CSV) -> pd.DataFrame:
    """Append one result row to results/results.csv and return the full table.

    model            e.g. 'bert-base-uncased', 'rule-based', 'claude-zero-shot'
    method           e.g. '32-shot', 'augmented', 'llm-generated', 'full-100%'
    n_train_labeled  number of labelled training examples used (32, 1584, ...)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"person": person, "model": model, "method": method,
           "split": split, "n_train_labeled": n_train_labeled, **metrics, "notes": notes}
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df
