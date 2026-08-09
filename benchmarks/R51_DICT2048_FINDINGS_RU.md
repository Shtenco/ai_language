# NEXUS R5.1 — DICT2048 findings

## Historical basis

The experiment reconstructs the old NEXUS V30 direction: a lossless 2048-ID non-BPE tokenizer. The exact historical tokenizer implementation was not recovered here, so R5.1 is a new lossless reconstruction, not a byte-for-byte copy.

## Tokenizer

R5.1 DICT2048 = 256 raw byte fallback IDs + 1792 train-only dictionary pieces. Dictionary pieces are frequent lexical chunks / reusable affixes scored by estimated byte saving. Encoding uses greedy longest match through a trie. Round-trip is exact on train/valid/test/shift; there is no UNK loss.

WikiText-2 compression:
- train: 10,797,148 bytes -> 4,517,954 tokens = 2.38983 bytes/token
- test: 2.44500 bytes/token
- Tiny Shakespeare shift: 1.84147 bytes/token

All compared models have exactly 999,978 trainable parameters.

## Experiment A — equal target-token compute

Each model receives 262,144 training target tokens, seeds 11/29/47.

| Model | test128 BPB | valid128 BPB | shift128 BPB | source byte/s |
|---|---:|---:|---:|---:|
| T0_BYTE256 | 3.54083 | 3.62559 | 5.43417 | 57,237 |
| R4_BYTE256 | 3.37731 | 3.46259 | 5.35151 | 48,306 |
| T0_DICT2048 | 2.78794 | 2.82200 | 5.66220 | 130,664 |
| R4_DICT2048 | **2.77371** | **2.80574** | 5.60908 | **110,809** |

R4_DICT2048 vs R4_BYTE256:
- in-domain test BPB: -17.87%
- effective source-byte throughput: 2.29x
- paired 3-seed test BPB t-test: p ≈ 0.00174

Important confound: equal token count means DICT2048 sees about 2.39x more original bytes during training and has a longer effective source-byte context.

## Experiment B — approximately equal original training bytes

DICT2048 training was reduced to 110,592 dictionary target tokens. Actual sampled source text: 263,786 ± 561 bytes, i.e. 1.0063x the 262,144-byte Byte256 exposure.

| Model | training source bytes | test128 BPB | valid128 BPB | shift128 BPB | source byte/s |
|---|---:|---:|---:|---:|---:|
| T0_DICT2048_DATA_MATCH | 263,786 | 3.25397 | 3.28800 | 6.17660 | 128,590 |
| R4_DICT2048_DATA_MATCH | 263,786 | **3.24421** | **3.27885** | 6.09356 | 108,940 |

Against the corresponding equal-token Byte256 baselines:
- T0_DICT2048_DATA_MATCH vs T0_BYTE256: 3.25397 vs 3.54083 = **-8.10% BPB**; paired 3-seed p ≈ 0.00178.
- R4_DICT2048_DATA_MATCH vs R4_BYTE256: 3.24421 vs 3.37731 = **-3.94% BPB**; all 3 paired seeds favor DICT2048; paired p ≈ 0.0601 with only n=3.
- R4 vs T0 inside data-matched DICT2048: -0.30% BPB, p ≈ 0.0762 (n=3).

## Negative result: domain shift

The dictionary is trained only on WikiText-2. On Tiny Shakespeare the Byte256 models remain better:
- R4_BYTE256: 5.35151 BPB
- R4_DICT2048 equal-token: 5.60908 BPB
- R4_DICT2048 raw-byte-matched: 6.09356 BPB

So the current dictionary improves in-domain modeling and effective throughput, but over-specializes to the training domain. Losslessness does not prevent statistical vocabulary/domain overfit.

## Honest conclusion

DICT2048 is a real positive direction, but the strongest defensible statement is not “2048-token dictionary is universally better.” It is:

> Under this ~1M-parameter WikiText-2 laboratory, a lossless train-only 2048-ID dictionary with byte fallback improves in-domain bits per original byte even when original training-byte exposure is approximately matched. It also processes far more source bytes per model token. The current English WikiText-specific dictionary degrades Tiny-Shakespeare domain-shift BPB.

The next strict test should equalize not only source training bytes but also source-byte context span, and train a mixed-domain/multilingual dictionary. A useful next ablation is vocab 512/1024/2048/4096 plus word-only vs lexical-chunk vs unigram/BPE alternatives, all scored in BPB and bytes/s.