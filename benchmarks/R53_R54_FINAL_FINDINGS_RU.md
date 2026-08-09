# NEXUS R5.3 / R5.4 — SOURCE-MATCHED TOKENIZER FINDINGS

## R5.3 protocol

Real balanced EN/RU training corpus: 5 MB WikiText-2 + 5 MB Russian SynTagRus. Evaluation: WikiText-2 test, Tiny Shakespeare, SynTagRus test, Russian GSD shift.

Neural comparison removes the two main R5.1 tokenizer confounds:
- every LM has exactly 999,978 trainable parameters;
- every batch uses the same randomly selected 128 **original source-byte** windows;
- every model receives the same optimizer-step count and the same source-byte LM exposure (~261,691 bytes in the 3-seed run);
- primary metric is bits per predicted original byte (BPB), not token perplexity;
- tokenizers are fitted on train only;
- compared tokenizers: BYTE256, lossless LEXDICT2048, lossless SentencePiece BPE2048, lossless SentencePiece UNIGRAM2048;
- each tokenizer is tested under both dense and R4 topology.

All R5.3 jobs completed successfully.

## R5.3 aggregate, 3 seeds

| Model | Wiki BPB | Shakespeare BPB | RU SynTagRus BPB | RU GSD BPB | Mean BPB | Source B/s |
|---|---:|---:|---:|---:|---:|---:|
| R4_UNIGRAM2048 | **3.28476** | 5.37093 | **2.41656** | **2.88003** | **3.48807** | 87,093 |
| DENSE_UNIGRAM2048 | 3.29707 | 5.37205 | 2.41176 | 2.87389 | 3.48870 | **114,127** |
| R4_BPE2048 | 3.34558 | 5.44216 | 2.45341 | 2.91757 | 3.53968 | 93,049 |
| DENSE_BPE2048 | 3.34900 | 5.45094 | 2.44932 | 2.91108 | 3.54008 | 117,160 |
| R4_LEXDICT2048 | 3.38106 | 5.32725 | 2.73830 | 3.14800 | 3.64865 | 93,157 |
| R4_BYTE256 | 3.56786 | 5.29801 | 2.64726 | 3.09839 | 3.65288 | 45,546 |
| DENSE_LEXDICT2048 | 3.40379 | 5.31145 | 2.75671 | 3.16409 | 3.65901 | 121,310 |
| DENSE_BYTE256 | 3.65442 | **5.27161** | 2.69722 | 3.13677 | 3.69000 | 55,890 |

Key R5.3 paired results (3 seeds):
- R4 UNIGRAM2048 vs R4 BYTE256: mean evaluation BPB -4.51%, 3/3 seeds better, paired t p ≈ 0.0131.
- R4 BPE2048 vs R4 BYTE256: -3.10%, paired p ≈ 0.00313.
- R4 LEXDICT2048 vs R4 BYTE256: -0.12%, p ≈ 0.781: the large R5.1 lexical-dictionary gain mostly disappears when both source-byte exposure and source-byte context are matched.
- DENSE UNIGRAM2048 vs DENSE BYTE256: -5.46%, paired p ≈ 0.0105.
- R4 UNIGRAM2048 vs DENSE UNIGRAM2048: essentially tied mean BPB (-0.018% for R4), p ≈ 0.938. Under this protocol the tokenizer effect is much larger and more robust than the current R4-vs-dense topology effect.

R4 UNIGRAM2048 vs R4 BYTE256 by domain:
- WikiText: -7.93%, p ≈ 0.0122;
- Tiny Shakespeare: +1.38% (worse), p ≈ 0.117;
- SynTagRus: -8.71%, p ≈ 0.00513;
- Russian GSD: -7.05%, p ≈ 0.00613.

## R5.4 protocol

The strongest R5.3 result was independently expanded to 10 matched seeds: 3, 7, 11, 17, 23, 29, 37, 47, 61, 73. Only R4 BYTE256 and R4 UNIGRAM2048 were trained. Same 999,978 parameters, same 128-source-byte contexts and same source-byte LM exposure.

## R5.4 10-seed confirmation

| Model | Wiki BPB | Shakespeare BPB | RU SynTagRus BPB | RU GSD BPB | Mean BPB | Source B/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4_BYTE256 | 3.60474 | **5.25058** | 2.70285 | 3.08879 | 3.66174 | 41,345 | 18.813 |
| R4_UNIGRAM2048 | **3.28462** | 5.31171 | **2.42302** | **2.81533** | **3.45867** | **75,120** | **10.278** |

Paired 10-seed result:
- UNIGRAM wins mean BPB in **10/10 seeds**;
- mean paired ΔBPB = -0.203067;
- mean relative paired improvement ≈ -5.50%; ratio of aggregate means ≈ -5.55%;
- paired t-test p ≈ **6.81e-5**;
- 95% t CI for ΔBPB ≈ **[-0.2693, -0.1368]**;
- one-sided Wilcoxon signed-rank p = **0.0009766**;
- source-byte inference throughput ratio ≈ **1.817x**;
- training wall-time speed ratio ≈ **1.830x**.

10-seed domain breakdown, UNIGRAM2048 vs BYTE256:
- WikiText: -8.88%, 10/10 wins, paired p ≈ 6.55e-6;
- Tiny Shakespeare: +1.16% (worse), only 3/10 wins, p ≈ 0.281;
- SynTagRus: -10.35%, 10/10 wins, p ≈ 6.54e-6;
- Russian GSD shift: -8.85%, 10/10 wins, p ≈ 1.97e-6.

## Conclusion

The strongest defensible conclusion is now:

> In this matched ~1M-parameter bilingual real-text laboratory, replacing raw byte tokenization with lossless Unigram-2048 substantially and reproducibly improves NEXUS R4's neural bits-per-original-byte while also reducing sequence length enough to raise effective source-byte throughput. The effect survives equal source-byte LM exposure and equal source-byte context and is confirmed across 10 initialization/data-plan seeds. It is not universal across domains: Tiny Shakespeare remains a negative-control domain where Byte256 is slightly better.

This result changes the architecture priority. The next highest-value tokenizer experiments are (1) Unigram vocabulary-size sweep 1024/2048/4096 under the same source-matched protocol, and (2) adaptive routing or mixture tokenization designed to preserve Unigram's Wiki/Russian gains while falling back to byte-level representation on domains such as Shakespeare where the learned subword vocabulary is mismatched.