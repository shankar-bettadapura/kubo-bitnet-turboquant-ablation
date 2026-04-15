# BitNet + TurboQuant Ablation Study
### Kubo Technologies — ML Training Project

**Model:** 25M parameter GPT | **Hardware:** NVIDIA RTX 3070 (8GB VRAM) | **Status:** Complete

> This repository contains the training configurations, evaluation scripts, and results from an ML ablation study conducted for Kubo Technologies. The study was completed with no prior ML experience — all necessary knowledge was built from scratch for this engagement.
>
> A governance assessment of these results is available in a companion repository: [model-risk-assessment-kubo](https://github.com/YOUR-USERNAME/model-risk-assessment-kubo)

---

## What This Project Is

An ablation study is a controlled experiment where you remove or change one component of a system at a time to measure its effect. This study trained a baseline language model, then applied two compression techniques separately to measure what each one costs in accuracy and what each one saves in memory.

The two techniques compared:

**BitNet** replaces the model's 32-bit floating point weights with ternary values (-1, 0, +1). The model becomes smaller and faster to run, but loses some precision in its calculations. The question is how much accuracy that costs.

**TurboQuant** compresses the key-value cache, a memory structure the model uses during text generation to avoid recalculating previous context. The compression reduces memory consumption during inference, particularly at long context lengths.

---

## Model Architecture

| Parameter | Value |
|---|---|
| Architecture | GPT (decoder-only transformer) |
| Parameters | ~25M |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Context length | 512 tokens (hardware limit) |
| Training data | Shakespeare character-level dataset |
| Training tokens | ~500M |

---

## Results

### Phase 1: Baseline

| Metric | Value |
|---|---|
| Train loss | 1.4650 |
| Validation loss | 1.4446 |
| Perplexity | 4.24 |

Perplexity measures how well the model predicts the next token. Lower is better. A perplexity of 4.24 means the model is, on average, choosing between roughly 4 equally likely next tokens at each step. This is the reference point for all comparisons.

### Phase 2: BitNet

| Metric | Value | vs Baseline |
|---|---|---|
| Train loss | 1.8018 | +0.3368 |
| Validation loss | 1.7799 | +0.3353 |
| Perplexity | 5.93 | +39% |

BitNet produced consistent 39% perplexity degradation across all tested context lengths. The gap between train and validation loss remained tight, confirming this is a real quantization effect rather than overfitting.

### Phase 3: TurboQuant (across context lengths)

| Model | Context | Std Perplexity | TQ Perplexity | Memory Std (MB) | Memory TQ (MB) | PPL Delta |
|---|---|---|---|---|---|---|
| Baseline | 128 | 8.32 | 8.63 | 198.0 | 198.0 | +0.31 |
| Baseline | 256 | 7.47 | 7.82 | 271.9 | 272.0 | +0.35 |
| Baseline | 512 | 6.36 | 6.63 | 420.4 | 420.5 | +0.27 |
| BitNet | 128 | 11.27 | 11.83 | 212.3 | 212.4 | +0.56 |
| BitNet | 256 | 10.05 | 10.60 | 286.3 | 286.3 | +0.55 |
| BitNet | 512 | 8.65 | 9.13 | 434.8 | 434.9 | +0.48 |

TurboQuant produced 3-5% perplexity degradation across all configurations. Memory consumption was nearly identical to the standard baseline at all tested context lengths. This is expected: at 512 tokens, the model weights dominate VRAM and the KV cache is not the primary memory consumer. TurboQuant's memory benefit is designed to materialize above approximately 4,096 token context lengths, which the RTX 3070's 8GB VRAM could not support.

---

## Repository Contents

```
config/         Training configuration files for each phase
eval/           Evaluation and inference scripts
results/        Raw metrics and logs from training runs
checkpoints/    Model checkpoints (see note below)
```

### A Note on Checkpoints

Model checkpoint files contain the trained weights for each phase. They are large binary files (.pt format). If they are not included directly in this repository due to file size constraints, they are available on request. The `checkpoints/` folder contains a summary of what each checkpoint represents.

---

## How to Run

### Requirements

```
pip install torch numpy
```

### Training

To reproduce the baseline run:

```
python config/train_baseline.py
```

To reproduce the BitNet run:

```
python config/train_bitnet.py
```

### Evaluation

To run the Phase 3 TurboQuant evaluation:

```
python eval/eval_phase3.py
```

The evaluation script measures perplexity and memory consumption at context lengths 128, 256, and 512 tokens, for both standard and TurboQuant-compressed inference, across both the baseline and BitNet checkpoints.

---

## Key Findings

**BitNet** introduces a consistent, measurable accuracy penalty at 25M parameter scale. The 39% perplexity increase is stable across context lengths, indicating it is a real effect of ternary quantization rather than an artifact of the evaluation conditions.

**TurboQuant** is near-lossless on accuracy at this scale (3-5% perplexity increase). However, the memory benefit that motivates the technique was not demonstrated in this study because the evaluation hardware could not reach the context lengths where it materializes.

**The hardware constraint** is the most important caveat in interpreting these results. All conclusions about TurboQuant's memory efficiency should be treated as preliminary until evaluated at 4,096+ token context lengths.

---

## Companion Repository

The governance assessment of these results, applying NIST AI RMF 1.0 and ISO/IEC 42001:2023 to evaluate production deployment readiness, is available here:

📋 [model-risk-assessment-kubo](https://github.com/shankar-bettadapura/model-risk-assessment-kubo)

---

## About

This project was completed as part of an independent ML engagement for Kubo Technologies and as a portfolio demonstration of applied ML capability alongside GRC governance methodology.

**Background:** M.S. Cybersecurity Studies | CompTIA Security+ | CISA, CRISC, ISO 42001 in progress | Former U.S. Army All-Source Intelligence Analyst

🔗 [LinkedIn](https://www.linkedin.com/in/shankar-bettadapura) | 🔗 [Substack](https://shankarbettadapura.substack.com)
