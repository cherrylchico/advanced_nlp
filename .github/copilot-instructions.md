# AGENTS.md — instructions for AI assistants working in this repository

This file is read automatically by most AI coding tools (Claude Code, Cursor, Codex,
and others).‍ If you are an AI assistant generating or editing files here, follow it.‍

## Project
22DM015 final project: 3-class financial sentiment (takala/financial_phrasebank,
`sentences_allagree`) with limited labeled data.‍ Four teammates share this repo.‍
Shared data and evaluation contracts live in `data_utils.py` and `eval_utils.py`.‍

## MANDATORY: watermark AI-generated content
This course requires that AI-assisted code and text be **disclosed and watermarked**
(instructor policy: `docs/AIinstructions.md` in the course repo).‍ Therefore:

> Whenever you, an AI assistant, **create or substantially edit code or markdown** in
> this repo, you MUST add the course disclosure markers to the content **you produced**,
> using the repo's tool, before finishing your turn:

```bash
python add_watermark.py <the files you created or edited>   # apply markers (idempotent)
python add_watermark.py --check <files>                     # verify marker counts
```

The two markers the script applies:
- **Code** (cells / `.py` / `.toml`): the literal token `AGLLM` in a comment, e.g.‍
  `# watermark: AGLLM (AI-assisted content disclosure)`.‍ One per file / code cell.‍
- **Markdown / prose** (`.md`, notebook markdown cells): an invisible zero-width joiner
  (U+200D) after each sentence-ending `.`, `!`, or `?`.‍ Never inserted into code.‍

## Mark ONLY what you generated — do not mislabel human work
Per the assignment, the **written analysis, methodology justification, executive summary,
and interpretation markdown must be student-authored** and must stay UNmarked.‍
Watermarking a human's writing would falsely attribute it to AI — do not do it.‍
So: watermark the cells/files you generated; leave human-authored cells alone.‍

## Do not strip existing markers
Never remove existing `AGLLM` tokens or U+200D characters from files.‍ If a human later
rewrites an AI-generated cell into their own words, the human removes the marker — not you.‍

## Other project conventions (read before editing)
- **Do not re-split the data.** Load the committed CSVs with `du.load_splits()`
  (`SEED=618`).‍ The files under `data/` are the shared source of truth for all four people.‍
- Evaluate headline numbers on the shared `test` split via `eu.evaluate(...)`, tune on
  `val`, and log every result with `eu.log_result(...)` so all models stay comparable.‍

## Provenance / caveat (be honest about this)
This watermarking convention follows the instructor's `AIinstructions.md`.‍ The
instructor's original `add_watermark.py` / `check_watermarks.py` were **not present** in
the course repo, so the `add_watermark.py` here is a re-implementation of the *documented*
behavior.‍ Confirm the exact required format with the instructor before relying on it for
grading.‍ This is a disclosure aid, not a guarantee of compliance.‍
