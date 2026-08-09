# NEXUS R5.2 — TOKENIZER TOURNAMENT FINDINGS

Balanced tokenizer-training corpus: 5,000,000 English WikiText-2 bytes + 5,000,000 Russian SynTagRus bytes. Evaluation surfaces: WikiText-2 test, Tiny Shakespeare, SynTagRus test, Russian GSD test. All tokenizer vocabularies are trained only on the training corpus. Exact round-trip is a hard gate.

Families: raw BYTE256; lossless WORDDICT; lossless lexical/MDL dictionary (LEXDICT); SentencePiece BPE; SentencePiece Unigram. Vocabulary sizes: 512, 1024, 2048, 4096 where supported.

## Ranking including vocabulary description cost

Mean `MDL-BPB = (empirical stream entropy + tokenizer vocabulary bytes*8) / original test bytes` over four EN/RU evaluation surfaces:

1. LEXDICT2048 — 3.41679 mean MDL-BPB; 4.11388 worst; train 2.7153 bytes/token; vocab 10,790 bytes.
2. BPE1024 — 3.44070; worst 4.18567; train 2.5878 B/token; vocab 14,192 bytes.
3. UNIGRAM1024 — 3.47852; worst 4.28914; train 2.4038 B/token; vocab 14,292 bytes.
4. LEXDICT4096 — 3.51470; worst 4.27774; train 3.2390 B/token; vocab 26,210 bytes.
5. BPE2048 — 3.52552; worst 4.28229; train 3.2265 B/token; vocab 29,888 bytes.

BYTE256 baseline: 4.40945 mean MDL-BPB.

## Ranking when vocabulary storage is ignored

Mean empirical stream BPB over the same four surfaces:

1. UNIGRAM4096 — 2.71030
2. BPE4096 — 2.71441
3. BPE2048 — 2.90539
4. UNIGRAM2048 — 2.91815
5. LEXDICT4096 — 2.97088
6. BPE1024 — 3.14624
7. UNIGRAM1024 — 3.18198
8. LEXDICT2048 — 3.19292
9. BYTE256 — 4.40414

Thus the largest subword vocabularies compress the token stream best, but LEXDICT2048 wins the finite-data MDL objective because its dictionary representation is much smaller.

## Domain winners by MDL-BPB

- WikiText-2: LEXDICT4096 = 3.30960.
- Tiny Shakespeare: LEXDICT4096 = 4.01244.
- Russian SynTagRus: UNIGRAM2048 = 2.25990.
- Russian GSD shift: LEXDICT1024 = 3.29729.

The optimum therefore changes with language/domain and with how heavily vocabulary cost is amortized. This motivates an adaptive representation/router experiment rather than assuming one globally optimal tokenizer.

## Important caveats

This is tokenizer-only preselection, not a neural language-model result. Empirical token entropy is not neural cross-entropy. The next R5.3 experiment therefore holds source-byte training exposure and source-byte context fixed and tests whether tokenizer compression gains survive inside matched ~1M neural models.