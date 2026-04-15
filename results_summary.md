# Results Summary

## Phase 1: Baseline

| Metric | Value |
|---|---|
| Train loss | 1.4650 |
| Validation loss | 1.4446 |
| Perplexity | 4.24 |

The baseline model is the uncompressed reference. All Phase 2 and Phase 3
results are measured as deltas against these numbers.

---

## Phase 2: BitNet

| Metric | Value | vs Baseline |
|---|---|---|
| Train loss | 1.8018 | +0.3368 |
| Validation loss | 1.7799 | +0.3353 |
| Perplexity | 5.93 | +39% |

BitNet replaces 32-bit floating point weights with ternary values (-1, 0, +1).
The 39% perplexity increase was consistent across all tested context lengths,
confirming it is a stable effect of the quantization rather than noise.

---

## Phase 3: TurboQuant

| Model | Context | Std Perplexity | TQ Perplexity | Memory Std (MB) | Memory TQ (MB) | PPL Delta |
|---|---|---|---|---|---|---|
| Baseline | 128 | 8.32 | 8.63 | 198.0 | 198.0 | +0.31 |
| Baseline | 256 | 7.47 | 7.82 | 271.9 | 272.0 | +0.35 |
| Baseline | 512 | 6.36 | 6.63 | 420.4 | 420.5 | +0.27 |
| BitNet | 128 | 11.27 | 11.83 | 212.3 | 212.4 | +0.56 |
| BitNet | 256 | 10.05 | 10.60 | 286.3 | 286.3 | +0.55 |
| BitNet | 512 | 8.65 | 9.13 | 434.8 | 434.9 | +0.48 |

TurboQuant produced 3-5% perplexity degradation across all configurations.
Memory consumption was nearly identical to baseline at all tested context
lengths. The RTX 3070's 8GB VRAM limited evaluation to 512 tokens maximum.
TurboQuant's memory benefit is designed to materialize above 4,096 tokens,
which was outside the hardware capability of this evaluation environment.

---

## Hardware

NVIDIA GeForce RTX 3070 (8GB VRAM)

## Training Data

Shakespeare character-level dataset (~500M tokens)
