# NEXUS R5.5 — UNIGRAM VOCABULARY SWEEP FINDINGS

## Protocol

Source/context-matched R4 real-LM benchmark on the same balanced English/Russian corpus used in R5.3/R5.4.

- source context: 128 original UTF-8 bytes;
- same source-byte windows for matched seeds;
- 256 optimizer steps, batch 8;
- same LM source exposure: ~261,682 bytes per run;
- topology: R4;
- primary metric: bits per predicted original byte (BPB);
- evaluation: WikiText-2 test, Tiny Shakespeare, Russian SynTagRus test, Russian GSD shift;
- Unigram tokenizer is fitted on train only and uses byte fallback;
- seeds: 3, 11, 23, 47, 73.

Parameter budgets were matched to within 10 parameters:
- Unigram1024: d=188, heads=4, FF=659, **999,974 params**;
- Unigram2048: d=138, heads=6, FF=983, **999,982 params**;
- Unigram4096: d=114, heads=6, FF=900, **999,984 params**.

Because hidden/FF dimensions necessarily change when vocabulary size changes under a fixed total parameter budget, this is a whole-model budget comparison, not a pure tokenizer-only causal isolation.

## Aggregate results, 5 seeds

| Model | Wiki BPB | Shakespeare BPB | RU SynTagRus BPB | RU GSD BPB | Mean BPB | Source B/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **R4_UNIGRAM4096** | **3.11088** | **5.11303** | **2.31863** | **2.72989** | **3.31811** | **97,751** | **8.894** |
| R4_UNIGRAM2048 | 3.29191 | 5.26468 | 2.41780 | 2.80316 | 3.44439 | 82,046 | 9.179 |
| R4_UNIGRAM1024 | 3.58891 | 5.56471 | 2.50370 | 2.89574 | 3.63827 | 78,909 | 9.667 |

## Paired results

### 4096 vs 2048

- mean evaluation BPB: **-3.67%**;
- 4096 wins **5/5 seeds**;
- paired t-test p ≈ **0.00220**;
- WikiText: -5.50%, 5/5, p ≈ 5.92e-5;
- Tiny Shakespeare: -2.88%, 4/5, p ≈ 0.0815;
- SynTagRus: -4.10%, 5/5, p ≈ 1.49e-4;
- Russian GSD: -2.61%, 5/5, p ≈ 0.0101.

### 4096 vs 1024

- mean evaluation BPB: **-8.80%**;
- 4096 wins **5/5 seeds**;
- paired t-test p ≈ **1.35e-5**.

### 2048 vs 1024

- mean evaluation BPB: **-5.33%**;
- 2048 wins **5/5 seeds**;
- paired t-test p ≈ **0.00166**.

## Interpretation

The neural optimum has not yet saturated by vocab=4096 in this ~1M-parameter laboratory. Despite allocating more of the fixed parameter budget to embeddings/output vocabulary and therefore reducing the hidden dimension, Unigram4096 improves BPB on every evaluated domain in aggregate and also raises effective source-byte throughput.

This is especially informative for the earlier Tiny-Shakespeare negative control. R5.4 found Unigram2048 slightly worse than Byte256 on Shakespeare. R5.5 finds Unigram4096 better than Unigram2048 on Shakespeare by ~2.88% and better in 4/5 matched seeds, although the small-sample p-value remains ~0.081. Therefore the earlier domain-shift weakness is at least partly consistent with insufficient vocabulary capacity rather than an inherent need to make Byte256 the main representation.

## Current architecture decision

For the next NEXUS Ω real-language branch, the strongest tested default tokenizer is now **lossless Unigram4096 with byte fallback**. Raw Byte256 remains valuable as the universal lossless substrate/fallback and as an OOD safety route, but the evidence no longer supports it as the primary language representation.

Next falsifiable steps:
1. extend the Unigram sweep above 4096 (8192/16384) until BPB/throughput reaches a real Pareto optimum under the fixed ~1M parameter budget;
2. test a byte-fallback Governor that activates explicit byte mode only when Unigram residual BPB/surprise is high;
3. scale the winning tokenizer + R4/R5 compound architecture beyond 1M while keeping raw-source-byte training budget, context and evaluation fixed.