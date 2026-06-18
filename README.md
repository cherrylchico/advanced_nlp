# 22DM015 Final Project — Text Classification with limited labeled data
18 June 2026
- Xianrui Cao
- Cherryl Chico
- Xiaoyan Wang
- Elvis Casco

Dataset: [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank),
config `sentences_allagree` (2,264 sentences, 3 classes: 0=negative, 1=neutral, 2=positive).‍


Notebook Submission:

1.‍ [`dataprep_aug`](notebooks/dataprep_aug.ipynb) — Data preparation: builds the train/val/test split used by **all** parts, and produces the augmented data needed for Part 2.‍ Run this first.‍
2.‍ [`part1`](notebooks/part1.ipynb) — Part 1: Setting Up the Problem.‍
3.‍ [`part2`](notebooks/part2.ipynb) — Part 2: Data Scientist Challenge.‍
4.‍ [`part2_zeroshot_code`](notebooks/part2_zeroshot_code.ipynb) — Part 2c: Zero-Shot Learning with LLM.‍
5.‍ [`part3`](notebooks/part3.ipynb) — Part 3: State of the Art Comparison.‍
6.‍ [`part4`](notebooks/part4.ipynb) — Part 4: Model Distillation / Quantization.‍

## Executive Summary

> _AI disclosure: the objective, findings, and ideas in this summary are our own.‍ An AI assistant was used only to edit the structure and wording.‍ AI-assisted text is watermarked per the course convention._

**Objective**

Our task is to classify financial news sentences into sentiment (positive, negative, or neutral) using the Financial PhraseBank introduced by Malo et.‍ al (2014).‍ What uniquely defines this dataset are the financial words which have inherent sentiment polarity (e.g.‍ profit is positive while loss is negative).‍ This polarity can be strengthened or reverse through directional words (e.g.‍ increased profits is strengthened positive, while decreased profits is switched to negative.)
We use the all-agree subset (2,264 sentences), which is class-imbalanced (61% neutral, 25% positive, 13% negative).‍

**Main findings and results**

1.‍ **The task is largely separable by direction alone.** A simple rule-based classifier using only directional words reaches ~0.83 accuracy and ~0.79 macro-F1 which is already close to the published Malo et al.‍ (2014) baseline.‍ This implies that the data is, in this sense, "simple."
2.‍ **Very little labeled data goes a long way.** With as few as 32 labeled examples, a fine-tuned BERT already beats both random baselines by a wide margin.‍
3.‍ **LLM-generated augmentation strongly helps in the low-data regime.** Adding LLM-generated examples to a small labeled set significantly improves performance.‍ At 1% of the data, macro-F1 rises from 0.48 (real-only) to 0.83.‍
4.‍ **Performance scales but plateaus, so 100% of the data is unnecessary.** Gains shoot up with training up to ~25% of the train data and flatten after ~50%, already producing near 100% accuracy well before the full dataset.‍
5.‍ **The remaining errors fall in two hard cases.** Sentences where the financial word and the directional word oppose each other (the sentiment "switches"), and sentences with no directional or financial signal at all both need real context rather than these cues.‍
6.‍ **On the full dataset, LLM augmentation adds little and meets the published SOTA.** Adding LLM-generated data to the full set gives only a marginal gain, landing on par with and by one metric just below the state of the art (EnhancedFinSentiBERT: FinBERT augmented with financial dictionaries and neutral-feature extraction).‍
7.‍ **A zero-shot LLM is the overall best.** The single strongest result in our comparison comes from a zero-shot LLM (Claude), which performs better even against our full-data fine-tuned BERT which is the best approach we found for this financial-sentiment task.‍
8.‍ **Compression makes the model deployable at little accuracy cost.** Distilling the fine-tuned BERT into a smaller DistilBERT student cuts inference latency by ~1.7× (and on-disk size by ~1.6×) for only a small accuracy cost (~1.6 macro-F1 points), keeping performance comparable to the full fine-tuned BERT.‍ Dynamic int8 quantization further shrinks the model ~2.4× with virtually no accuracy loss, though on our hardware it buys memory rather than speed.‍
