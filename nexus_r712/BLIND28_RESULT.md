# NEXUS R7.12 GraphBoost — REAL Blind-28

Frozen before Blind-28 in commit `d628755a8a458a914dbc23b783cc911efadfe3d5`.

Blind-28 SHA-256: `25ae7885469162214fa8235b06ef2a71e87890f7811bc3dd58538ffb87a888b6`.
GitHub Actions run: `32599985290`, five real GGUF jobs SUCCESS.

## Learning mechanism

`F0(x) = NEXUS graph logits`

CatBoost MultiClass begins from the NEXUS logits as its baseline and sequential trees learn residual negative-gradient corrections. Formal VERIFIED proof remains sovereign. Unknown items use a frozen pre-response top-2 teacher route. A boost correction is accepted only when `delta_logit >= 0.25`.

## Confirmatory result

- NEXUS graph top-2 baseline: **130/160 = 81.25%**
- Residual GraphBoost: **133/160 = 83.125%**
- Frozen gated R7.12: **133/160 = 83.125%**
- Best single teacher, Phi-4-mini: **115/160 = 71.875%**
- Five-model majority diagnostic: **120/160 = 75.0%**
- Five-teacher oracle diagnostic: **150/160 = 93.75%**

The frozen gate changed three decisions relative to the same NEXUS graph baseline: **3 corrections, 0 regressions**. Incremental paired exact two-sided p=0.25, so the +1.875 pp direction is positive but is not statistically established at n=160.

R7.12 vs Phi: +11.25 pp; paired exact p=0.0005335.

Formal pre-inference coverage: 31/160, with 29/31 correct VERIFIED answers. Logical deployment cost: 258 teacher calls / 160 items = **1.6125 calls/item**.

Two Blind-28 mixed questions explicitly refer to boosting mechanics. Excluding both leaves NEXUS graph 128/158=81.01% and R7.12 131/158=82.91%, so the observed directional gain is not solely caused by those items.

## Boundary

Blind-28 is an internally authored post-freeze architecture benchmark, not an external standardized benchmark. The principal next bottleneck is proof soundness/certification: a false sovereign VERIFIED proof cannot be repaired by GraphBoost.
