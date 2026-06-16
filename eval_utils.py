"""Shared evaluation + results logging so all four notebooks are comparable.

Every model (rule-based, BERT, zero-shot, distilled) reports the SAME metrics on
the SAME test set, and logs one row into results/results.csv. Part 3 (comparison)
and Part 4 just read that file -- no need to re-run anyone else's model.

Convention: evaluate on the shared TEST split only for the headline numbers.
Use VAL for tuning/early stopping.
"""
# watermark: AGLLM (AI-assisted content disclosure)
from __future__ import annotations
from pathlib import Path
import pandas as pd

LABEL_NAMES = ["negative", "neutral", "positive"]
METRIC_KEYS = ["accuracy", "f1_macro", "f1_weighted"] + [f"f1_{n}" for n in LABEL_NAMES]
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
               split: str = "test", notes: str = "",
               path: Path | str = RESULTS_CSV) -> pd.DataFrame:
    """Log one result row to results/results.csv and return the full table.

    model            e.g. 'bert-base-uncased', 'rule-based', 'claude-zero-shot'
    method           e.g. '32-shot', 'augmented', 'llm-generated', 'full-100%'
    n_train_labeled  number of labelled training examples used (32, 1584, ...)

    Upsert semantics: re-logging the same (model, method, split, n_train_labeled)
    replaces the previous row instead of appending a duplicate, so notebooks can be
    re-run freely (results.csv once accumulated triplicate rows and had to be
    hand-deduped).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": model, "method": method,
           "split": split, "n_train_labeled": n_train_labeled, **metrics, "notes": notes}
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # Dedup on a normalized string key so mixed CSV/in-memory dtypes compare
    # consistently: an empty field round-trips through CSV as NaN (would not match
    # a fresh ""), and n_train_labeled logged as int is read back as float. We build
    # the key in a throwaway column so the stored dtypes are left untouched.
    key_cols = ["model", "method", "split", "n_train_labeled"]
    _key = df[key_cols].fillna("").astype(str).agg("\x00".join, axis=1)
    df = df[~_key.duplicated(keep="last")]
    df.to_csv(path, index=False)
    return df


def latest_result(model: str, method: str, n_train_labeled, split: str = "test",
                  full_row: bool = False, path: Path | str = RESULTS_CSV):
    """Latest results.csv row for (model, method, split, n_train_labeled) — metrics
    dict, full row (`full_row=True`), or None. This is what makes the notebooks
    resume-aware: delete a row from results.csv to force that experiment to re-run."""
    path = Path(path)
    if not path.exists():
        return None
    r = pd.read_csv(path)
    r = r[(r["model"] == model) & (r["method"] == method) & (r["split"] == split)
          & (r["n_train_labeled"].astype(str) == str(n_train_labeled))]
    if not len(r):
        return None
    return r.iloc[-1] if full_row else {k: r.iloc[-1][k] for k in METRIC_KEYS}


def fmt(metrics: dict) -> dict:
    """Round a metrics dict for printing."""
    return {k: round(float(v), 4) for k, v in metrics.items()}
