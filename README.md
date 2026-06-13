# 22DM015 Final Project — Financial sentiment with limited labeled data

Contributors:

- Xianrui Cao
- Cherryl Chico
- Xiaoyan Wang
- Elvis Casco

Dataset: [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank),
config `sentences_allagree` (2,264 sentences, 3 classes: 0=negative, 1=neutral, 2=positive).‍

## Repo layout
```
data_utils.py     # canonical loader + split logic (Person D owns)
eval_utils.py     # shared evaluate() + log_result() -> results/results.csv
data/             # COMMITTED shared splits (train/val/test/labeled_32) — do not re-split
results/          # results.csv, the merged scoreboard for Part 3/4 (created on first log)
notebooks/        # one template per person
```

## The shared data contract (decided by Person D)
- **Splits:** stratified **70/10/20** train/val/test, `SEED=618`.‍ Sizes: train 1584, val 227, test 453.‍
- **32-shot:** balanced sample from *train only* — **11 negative / 10 neutral / 11 positive**.‍
- **Why files, not just a seed:** a seed is reproducible within one notebook, but across 4 people /
  machines / library versions shuffle behavior drifts.‍ The CSVs in `data/` are the source of truth;
  the seed only documents how they were made.‍ **Don't re-split — load the CSVs.**
- "Unlabelled" data for Part 2 = train minus the 32 (`du.unlabeled_pool`).‍
- Part 3 percentage curves: `du.subset_by_fraction(train, frac)`.‍

## Setup
All dependencies for all four notebooks are pinned in `pyproject.toml` / `uv.lock`:
```bash
uv sync          # creates .venv (Python 3.12) and installs the locked versions
```
Then point your IDE / Jupyter at the `.venv` interpreter (`uv run jupyter lab` also works).‍
Use Python 3.11–3.12, not 3.14 (no torch wheels there yet).‍

## Workflow
1.‍ Person D runs `notebooks/dataprep_aug.ipynb` → regenerates & commits `data/`.‍
2.‍ A/B/C load via `du.load_splits()` and report metrics with `eu.log_result(...)`.‍
3.‍ Part 3/4 read `results/results.csv` to compare every model on the same test set.‍

## AI-use disclosure (watermarking)
The course requires AI-assisted code/text to be disclosed and watermarked.‍ This repo uses
`add_watermark.py` (`AGLLM` token in code, invisible U+200D after sentences in prose) plus
`AGENTS.md` / `.github/copilot-instructions.md` to instruct AI assistants to self-mark.‍

Enable the auto-watermark git hook **once per clone**:
```bash
git config core.hooksPath .githooks
```
It watermarks every staged `.py/.ipynb/.md/.toml` on commit.‍ For genuinely human-authored
content (analysis, methodology, interpretation — which must stay student-authored and
UNmarked), commit that file with `git commit --no-verify`.‍ Verify markers anytime:
`python add_watermark.py --check notebooks *.py *.md`.‍

Caveat: this follows the instructor's `AIinstructions.md`, but their original scripts
weren't in the course repo — confirm the exact required format with the instructor.‍

## Course rules baked into the templates
Fixed seed at the top of every notebook; AI-generated code/text must be watermarked & declared;
interpretation/analysis must be student-authored.‍
