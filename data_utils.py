"""Shared data utilities for the Financial PhraseBank project (22DM015).

Everyone imports from this file so the four notebooks load *identical* data.
Person D runs `build_splits()` once and commits the CSVs under data/.
Persons A/B/C only call `load_splits()` and `subset_by_fraction()`.

Data contract
-------------
- Source : takala/financial_phrasebank, config "sentences_allagree" (2264 sents).
- Labels : 0=negative, 1=neutral, 2=positive  (same order as the HF features).
- Split  : stratified train/val/test = 70/10/20, SEED=42.
- 32-shot: balanced sample from TRAIN only -> 11 negative / 10 neutral / 11 positive.
- The "unlabelled pool" for Part 2 = train rows NOT in labeled_32 (use `unlabeled_pool()`).
- Part 3 percentage curves = `subset_by_fraction(train, frac)`, nested & stratified.
"""
from __future__ import annotations
import zipfile
from pathlib import Path
import pandas as pd

SEED = 42
CONFIG = "sentences_allagree"
LABEL_NAMES = ["negative", "neutral", "positive"]   # index == label id
NAME2ID = {n: i for i, n in enumerate(LABEL_NAMES)}

# 32-shot allocation (sums to 32). Minority classes get 11, abundant neutral gets 10.
NSHOT_PER_CLASS = {"negative": 11, "neutral": 10, "positive": 11}

DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_raw() -> pd.DataFrame:
    """Download the official zip and parse Sentences_AllAgree.txt (latin-1, 'text@label')."""
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download(
        "takala/financial_phrasebank",
        "data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    with zipfile.ZipFile(zp) as z:
        target = next(n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt"))
        raw = z.read(target).decode("latin-1")
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        text, name = line.rsplit("@", 1)
        rows.append((text.strip(), NAME2ID[name.strip()]))
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["label_name"] = df["label"].map(dict(enumerate(LABEL_NAMES)))
    return df


def build_splits(out_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Build and write train/val/test/labeled_32 CSVs. Run by Person D only."""
    from sklearn.model_selection import train_test_split

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_raw().reset_index(drop=True)
    df.insert(0, "id", df.index)  # stable id, survives all downstream subsetting

    # 70/10/20 stratified. First peel off 20% test, then 1/8 of the rest -> 10% val.
    train_val, test = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=SEED, stratify=train_val["label"]
    )

    # Balanced 32-shot, sampled from TRAIN only, fixed seed.
    parts = []
    for name, k in NSHOT_PER_CLASS.items():
        cls = train[train["label"] == NAME2ID[name]]
        parts.append(cls.sample(n=k, random_state=SEED))
    labeled_32 = pd.concat(parts).sort_values("id").reset_index(drop=True)

    splits = {
        "train": train.sort_values("id").reset_index(drop=True),
        "val": val.sort_values("id").reset_index(drop=True),
        "test": test.sort_values("id").reset_index(drop=True),
        "labeled_32": labeled_32,
    }
    for name, frame in splits.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return splits


def load_splits(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the committed CSVs. Used by everyone (A/B/C/D)."""
    data_dir = Path(data_dir)
    return {
        name: pd.read_csv(data_dir / f"{name}.csv")
        for name in ["train", "val", "test", "labeled_32"]
    }


def unlabeled_pool(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Part 2 'unlabelled' data = train rows not among the 32 labelled ones."""
    labeled_ids = set(splits["labeled_32"]["id"])
    return splits["train"][~splits["train"]["id"].isin(labeled_ids)].reset_index(drop=True)


def subset_by_fraction(train: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    """Stratified subset of train for Part 3 curves (1%,10%,25%,50%,75%,100%).

    Note: each fraction is sampled independently with the same seed (sklearn-style),
    NOT strictly nested. If you need nested subsets, sort once and slice cumulatively.
    """
    if frac >= 1.0:
        return train.reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(
        train, train_size=frac, random_state=seed, stratify=train["label"]
    )
    return keep.reset_index(drop=True)
