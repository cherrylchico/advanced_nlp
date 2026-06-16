"""Shared data utilities for the Financial PhraseBank project (22DM015).

All notebooks import from this module so they load *identical* data. The split
CSVs under ``data/`` are the single source of truth: ``build_splits()`` regenerates
them (run once, then commit the files), and ``load_splits()`` reads them back.
Downstream code should call ``load_splits()`` and never re-split, so every model is
trained and evaluated on exactly the same rows regardless of library versions.

Data contract
-------------
- Source : ``takala/financial_phrasebank``, config ``"sentences_allagree"`` (2264 sentences).
- Labels : 0=negative, 1=neutral, 2=positive (same order as the Hugging Face features).
- Split  : stratified train/val/test = 70/10/20, ``SEED = 618``.
- 32-shot: balanced sample drawn from TRAIN only -> 11 negative / 10 neutral / 11 positive.
- Unlabelled pool : training rows not among the 32 labelled examples (``unlabeled_pool``).
- Fraction curves : ``subset_by_fraction(train, frac)`` — stratified; each fraction is
  sampled independently with the same seed (NOT nested; see that function's docstring).
"""
# watermark: AGLLM (AI-assisted content disclosure)
from __future__ import annotations
import zipfile
from pathlib import Path
import pandas as pd

SEED = 618
CONFIG = "sentences_allagree"
LABEL_NAMES = ["negative", "neutral", "positive"]    # list index == integer label id
NAME2ID = {n: i for i, n in enumerate(LABEL_NAMES)}  # e.g. {"negative": 0, "neutral": 1, ...}

# Balanced 32-shot allocation (sums to 32). The minority classes (negative, positive)
# get 11 each and the abundant neutral class gets 10.
NSHOT_PER_CLASS = {"negative": 11, "neutral": 10, "positive": 11}

# Directory holding the committed split CSVs, resolved relative to this file.
DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_raw() -> pd.DataFrame:
    """Download the Financial PhraseBank archive and parse the all-agree sentences.

    Fetches ``FinancialPhraseBank-v1.0.zip`` from the Hugging Face Hub and reads
    ``Sentences_AllAgree.txt`` from inside it. That file is latin-1 encoded with one
    record per line in the form ``"<sentence>@<sentiment>"``. Parsing the raw archive
    (instead of the dataset's loading script) keeps this independent of the installed
    ``datasets`` library version.

    Args:
        None.

    Returns:
        pandas.DataFrame: one row per sentence, in source-file order (not shuffled),
        with columns:
            - ``text`` (str): the sentence.
            - ``label`` (int): integer label 0/1/2 per :data:`NAME2ID`.
            - ``label_name`` (str): ``"negative"`` / ``"neutral"`` / ``"positive"``.

    Raises:
        StopIteration: if no ``Sentences_AllAgree.txt`` entry is found in the archive.
    """
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


def sample_few_shot(
    train: pd.DataFrame,
    n_per_class: dict[str, int] = NSHOT_PER_CLASS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Draw a balanced few-shot labelled subset from the training split.

    Samples a fixed number of rows per class (without replacement) and concatenates
    them, deterministically given ``seed``. With the defaults this produces the 32-shot
    set (11 negative / 10 neutral / 11 positive).

    Note:
        ``train`` is sorted by ``id`` before sampling, so the result is independent of
        the input row order (``DataFrame.sample`` otherwise selects by position). This
        makes the function fully reproducible: ``sample_few_shot(load_splits()["train"])``
        returns exactly the committed ``labeled_32.csv``.

    Args:
        train (pandas.DataFrame): the training frame to sample from; must contain
            ``id`` and ``label`` columns.
        n_per_class (dict[str, int]): number of rows to draw per class, keyed by class
            name (one of :data:`LABEL_NAMES`). Defaults to :data:`NSHOT_PER_CLASS`
            (11/10/11, the canonical 32-shot set).
        seed (int): random seed controlling which rows are drawn. Defaults to
            :data:`SEED` (618).

    Returns:
        pandas.DataFrame: the sampled rows, sorted by ``id`` with the index reset.
        Columns match ``train``; the row count equals ``sum(n_per_class.values())``.

    Raises:
        ValueError: if any requested class has fewer rows available than requested.
    """
    # Sort by id first so the sample is independent of the input row order
    # (DataFrame.sample selects by position) -> reproducible from any ordering of train.
    train = train.sort_values("id")
    parts = []
    for name, count in n_per_class.items():
        cls = train[train["label"] == NAME2ID[name]]
        if len(cls) < count:
            raise ValueError(f"class '{name}' has {len(cls)} rows but {count} were requested.")
        parts.append(cls.sample(n=count, random_state=seed))
    return pd.concat(parts).sort_values("id").reset_index(drop=True)


def build_splits(
    out_dir: Path | str = DATA_DIR,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Create the canonical train/val/test splits and write them to disk.

    Loads the raw data, assigns a stable integer ``id`` to every sentence, then produces
    a stratified train/val/test split (defaults to 70/10/20), deterministic given
    :data:`SEED`. One CSV is written per split and the same frames are returned. Run this
    once and commit the CSVs; other code should call :func:`load_splits` rather than
    rebuilding. The few-shot labelled subset is created separately — see
    :func:`sample_few_shot` (and write it to ``labeled_32.csv`` for :func:`load_splits`).

    The split is done in two stratified steps: ``test_size`` is peeled off the full data
    first, then ``val_size`` (re-expressed as a fraction of the remaining train+val) is
    peeled off, so ``val_size`` is interpreted relative to the *whole* dataset.

    Args:
        out_dir (pathlib.Path | str): directory the CSVs are written into; created if it
            does not already exist. Defaults to the project ``data/`` directory
            (:data:`DATA_DIR`).
        test_size (float): fraction of the whole dataset to hold out as the test split,
            in (0, 1). Defaults to ``0.20``.
        val_size (float): fraction of the whole dataset to hold out as the validation
            split, in (0, 1). Defaults to ``0.10``. The remaining ``1 - test_size -
            val_size`` becomes the training split.
        seed (int): random seed controlling both stratified splits. Defaults to
            :data:`SEED` (618).

    Returns:
        dict[str, pandas.DataFrame]: mapping with keys ``"train"``, ``"val"``, ``"test"``.
        Every frame has columns ``id, text, label, label_name`` and is sorted by ``id``.

    Raises:
        ValueError: if ``test_size`` or ``val_size`` is not in (0, 1), or if
            ``test_size + val_size >= 1`` (no rows left for training).

    Side effects:
        Writes ``train.csv``, ``val.csv`` and ``test.csv`` into ``out_dir`` (overwriting
        any existing files of those names). Does not write ``labeled_32.csv``.
    """
    from sklearn.model_selection import train_test_split

    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must each be in (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1 to leave rows for training.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_raw().reset_index(drop=True)
    df.insert(0, "id", df.index)  # stable id, survives all downstream subsetting

    # Stratified two-step split. Peel off `test_size` first, then take `val_size` of the
    # whole data from what remains (re-expressed as a fraction of the train+val portion).
    val_relative = val_size / (1.0 - test_size)
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=seed, stratify=train_val["label"]
    )

    splits = {
        "train": train.sort_values("id").reset_index(drop=True),
        "val": val.sort_values("id").reset_index(drop=True),
        "test": test.sort_values("id").reset_index(drop=True),
    }
    for name, frame in splits.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return splits


def load_splits(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the committed split CSVs produced by :func:`build_splits`.

    This is the entry point downstream code should use: it reads the shared, committed
    files so every notebook sees identical rows.

    Args:
        data_dir (pathlib.Path | str): directory containing the split CSVs. Defaults to
            the project ``data/`` directory (:data:`DATA_DIR`).

    Returns:
        dict[str, pandas.DataFrame]: mapping with keys ``"train"``, ``"val"``, ``"test"``,
        each frame having columns ``id, text, label, label_name``.

    Raises:
        FileNotFoundError: if any of the three expected CSVs is missing (run
            :func:`build_splits` first).
    """
    data_dir = Path(data_dir)
    return {
        name: pd.read_csv(data_dir / f"{name}.csv")
        for name in ["train", "val", "test"]
    }


def unlabeled_pool(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the training rows that are NOT among the 32 labelled examples.

    Models the limited-label setting: only the 32 examples count as labelled, and the
    remainder of the training set is the available unlabelled pool.

    Args:
        splits (dict[str, pandas.DataFrame]): the mapping returned by :func:`load_splits`
            or :func:`build_splits`. Must contain keys ``"train"`` and ``"labeled_32"``,
            each frame having an ``id`` column.

    Returns:
        pandas.DataFrame: the subset of ``splits["train"]`` whose ``id`` is not present
        in ``splits["labeled_32"]``, with the index reset. Columns match ``train``.
    """
    labeled_ids = set(splits["labeled_32"]["id"])
    return splits["train"][~splits["train"]["id"].isin(labeled_ids)].reset_index(drop=True)


def subset_by_fraction(train: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    """Take a stratified fraction of the training set, for data-scaling curves.

    Used to build the points of a learning curve (e.g. 1%, 10%, 25%, 50%, 75%, 100%).
    Class proportions are preserved by stratifying on ``label``.

    Note:
        Each fraction is sampled *independently* with the same seed (scikit-learn style),
        so the subsets are NOT nested — the 10% sample is not guaranteed to contain the
        1% sample. For strictly nested subsets, sort once and slice cumulatively instead.

    Args:
        train (pandas.DataFrame): the training frame to sample from; must contain a
            ``label`` column used for stratification.
        frac (float): fraction of rows to keep, in the interval (0, 1]. Any value
            ``>= 1.0`` returns all of ``train``.
        seed (int): random seed controlling which rows are sampled. Defaults to
            :data:`SEED` (618).

    Returns:
        pandas.DataFrame: a stratified subset of ``train`` with the index reset; the full
        ``train`` (index reset) when ``frac >= 1.0``. Columns match ``train``.
    """
    if frac >= 1.0:
        return train.reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(
        train, train_size=frac, random_state=seed, stratify=train["label"]
    )
    return keep.reset_index(drop=True)
