# NEXUS R5.6 / R5.6b — 3M Russian Adequacy Findings

## Verdict

**3M parameters substantially improve held-out Russian likelihood, but the tested model is NOT yet an adequate free-running Russian language model.**

This result is intentionally reported without an external LLM judge, rejection sampling, anti-repeat cleanup, or cherry-picking of model seed. Raw greedy and sampled continuations are the primary generation evidence.

## R5.6 protocol

- total parameter target: 3,000,000;
- topology: R4 static sparse attention + the historical token-id-mod-4 logical FF route;
- tokenizer: lossless SentencePiece Unigram with byte fallback;
- vocab sweep: 4096 / 8192 / 16384;
- parameter counts: 2,999,924 / 2,999,954 / 2,999,924;
- source context: 256 original UTF-8 bytes;
- sweep: 512 steps × batch 8, seeds 11/29/47;
- winner chosen only by mean held-out Russian BPB;
- final training: fixed seed 20260809, 4096 steps × batch 16;
- Russian train stream: 18,874,655 bytes from RuHeritage + SynTagRus;
- final source exposure: 16,746,591 bytes;
- held-out Russian domains: SynTagRus, Russian GSD shift, RuHeritage-heldout.

## 3M vocabulary sweep

Mean Russian BPB over SynTagRus/GSD/RuHeritage:

| Vocab | Mean RU BPB | SynTagRus | GSD | RuHeritage | Source B/s |
|---:|---:|---:|---:|---:|---:|
| 4096 | 2.23046 | 1.96412 | 2.51209 | 2.21518 | 127,499 |
| 8192 | 2.19644 | 1.94191 | 2.43674 | 2.21067 | 148,030 |
| **16384** | **2.12965** | **1.85148** | **2.36961** | **2.16786** | **161,052** |

16384 vs 8192: mean RU BPB about **-3.04%**, wins 3/3 matched seeds, paired t p≈0.00746.

16384 vs 4096: mean RU BPB about **-4.52%**, wins 3/3 matched seeds, paired t p≈0.0172.

The sample is small (n=3), therefore these p-values are supporting evidence rather than a broad statistical proof.

## Long-trained 16384 winner

Fixed final seed 20260809:

- params: 2,999,924;
- training source exposure: 16,746,591 bytes;
- train time: ~707.6 s on the GitHub CPU runner;
- SynTagRus BPB: **1.47090**;
- GSD shift BPB: **1.93601**;
- RuHeritage heldout BPB: **1.74537**;
- mean of the three Russian domains: **~1.71743 BPB**.

Relative to the short 16384 sweep mean, longer training reduces mean Russian BPB by roughly **19.36%**.

This is a real likelihood/generalization improvement. It does **not** imply coherent generation.

## Raw generation verdict

Examples from the unmodified final output:

Greedy:

> Наука развивается потому, что не — — — — — — — — — — — — ...

Sampling:

> Москва — это город, в котором, что был, что не все воеводы, на — на — и, а сею. ...

The sampled continuation later degenerates into malformed words and fragments. Similar behavior occurs for all six prompts. Thus the model has learned Russian orthographic/statistical surface structure and some local phrase regularities, but not robust sentence-level free-running Russian.

Diagnostics over the six sampled continuations of the 16384 model:

- mean Cyrillic-letter share: ~97.6%;
- mean generated-Russian-word coverage in the 18.87 MB training stream: ~72.7%;
- training-bigram coverage: ~28.9%;
- training-trigram coverage: ~2.7%;
- sampled em-dash density: ~6.75 per 100 characters;
- greedy em-dash density: ~45.7 per 100 characters.

The source corpus has only ~0.204 em-dashes per 100 characters, so the greedy dash attractor is a model/generation failure, not simply reproduction of corpus punctuation frequency.

## R5.6b: does freeing cortex fix generation?

A direct long-training control was run at the exact same total ~3M budget, exact same final seed, exact same training-byte SHA-256 and same final training schedule for vocab 4096 and 8192.

Parameter allocation:

| Vocab | Total params | Token embedding params | Embedding fraction | Non-token-embedding params |
|---:|---:|---:|---:|---:|
| 4096 | 2,999,924 | 1,048,576 | 34.95% | 1,951,348 |
| 8192 | 2,999,954 | 1,769,472 | 58.98% | 1,230,482 |
| 16384 | 2,999,924 | 2,260,992 | 75.37% | 738,932 |

This exposes an important compression/cortex tradeoff: 16384 is the best BPB compressor, but about three quarters of all model parameters live in the tied token embedding table.

Long-control held-out BPB:

| Vocab | SynTagRus BPB | GSD BPB |
|---:|---:|---:|
| 4096 | 1.51471 | 2.04501 |
| 8192 | 1.51718 | 2.00983 |
| **16384** | **1.47090** | **1.93601** |

Freeing cortex alone does not rescue generation. 4096 and 8192 also produce pseudo-Russian continuations and greedy attractors. The error changes form, but coherent language does not appear.

Sampled-generation diagnostics:

| Vocab | Cyrillic share | Train word coverage | Train bigram coverage | Train trigram coverage | em-dash /100 chars | replacement chars/sample |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 97.68% | 61.74% | 22.00% | 3.48% | 3.68 | 0.00 |
| 8192 | 100.0% | 67.97% | 28.17% | 3.73% | 5.32 | 0.00 |
| 16384 | 97.62% | 72.71% | 28.91% | 2.71% | 6.75 | 0.33 |

These overlap measures are diagnostics, not a semantic coherence score.

## Two protocol flaws discovered by the falsification test

### 1. UTF-8 block mixing

The R5.6 corpus mixer permuted fixed 64 KiB byte blocks and could cut a UTF-8 codepoint at a block boundary. The resulting 18.87 MB train stream contains only about 255 replacement positions when decoded permissively (~13.5 per million bytes), so this is far too rare to explain the overall collapse, but it is a genuine protocol defect and must be removed.

### 2. Learned absolute positions were under-trained

For Unigram16384, a 256-byte source span tokenizes to only ~42 tokens at the median; ~99% of sampled spans are below roughly 74 tokens. Therefore absolute position embeddings around positions 75–191 receive very little training, while the generation test asked for up to 160 additional tokens.

This makes the long tail of the old generation test unnecessarily harsh. It does not explain the early degeneration in the first tens of tokens, but it invalidates treating very long R5.6 continuations as a clean context-generalization test.

## Architecture conclusion

R5.6 establishes a useful negative result:

**low held-out BPB is not sufficient for coherent autoregressive language.**

The next experiment should not simply add vocabulary or blindly add parameters. The current bottleneck is more likely the combination of training geometry, document diversity, positional representation, shallow depth, and arbitrary token-ID routing.

## R5.7 requirements

1. Make corpus assembly strictly UTF-8 safe.
2. Increase document diversity drastically; the current RuHeritage slice reached 12 MB using only 65 records.
3. Add a stable contemporary Russian source, while keeping literary and syntactic sources.
4. Train fixed-length token sequences (e.g. 128 tokens) for the language-adequacy stage rather than fixed 256-byte spans.
5. Replace learned absolute positions with RoPE or another position scheme that does not leave most long-context positions untrained.
6. Increase depth within the same 3M total budget.
7. Remove the arbitrary `token_id % 4` FF route or replace it with a learned/semantic router.
8. Re-test dense vs R4/hybrid attention under the improved tokenized regime; the earlier Unigram experiments did not show a decisive R4 quality advantage over dense attention.
9. Optimize a multi-objective target: held-out BPB + generation-degeneracy diagnostics + throughput, rather than BPB alone.
10. Keep raw free-running Russian samples as a hard falsification gate.

Current status: **3M is a strong Russian statistical compressor, not yet an adequate Russian generator.**
