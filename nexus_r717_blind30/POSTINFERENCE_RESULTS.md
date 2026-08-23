# NEXUS R7.17 — Blind-30 Confirmatory Results

Post-inference result commit. The benchmark, evaluator, active-intervention rule and R7.17 weights were frozen before any teacher evidence.

## Integrity
- R7.17 pre-Blind30 freeze commit (agi_olga): `0866c5e9a05314f616a37bfe9c04f6fc476bd1d5`
- R7.17 freeze ZIP SHA-256: `2b86e4beb2a23b691914c638185db4bad8f66cb966baa5fcb0a77ba7dfcad5c0`
- Blind-30 SHA-256: `afeca41d3313bde64673c99fbf059554a524ab946d21f85269ef4f0d7b5e1f6e`
- Preregistered evaluator SHA-256: `8f1e7e03bfdb2e67481918a7a82b4991eca41a6a3c96c2eb4edd9933ce416246`
- Real-teacher workflow run: `32653646787`
- All five real Ollama teacher jobs completed successfully.

## Confirmatory system result
- **R7.16 frozen:** 129/200 = **64.5%**, mean teacher calls 2.11.
- **R7.17 advisory-only:** 129/200 = **64.5%**.
- **R7.17 active one-step recurrent substitution:** 129/200 = **64.5%**, mean calls 2.105.
- **R7.17 active two-step quantum-inspired substitution:** 129/200 = **64.5%**, mean calls 2.105.

Paired outcomes relative to R7.16:
- advisory: 0 corrections / 0 regressions / p=1.0
- active one-step: 0 corrections / 0 regressions / p=1.0
- active two-step: 0 corrections / 0 regressions / p=1.0

**Conclusion:** fresh Blind-30 does not confirm incremental final-answer accuracy from activating the frozen R7.17 recurrent/quantum teacher-selection override.

## R7.16/R7.17 accuracy by category
- causal_confounding: 14/25 = 56%
- code_state: 11/25 = 44%
- deceptive_consensus: 18/25 = 72%
- long_chain: 8/25 = 32%
- multilingual_logic: 23/25 = 92%
- pattern_transfer_numeric: 9/25 = 36%
- query_value_stop: 25/25 = 100%
- revision_after_evidence: 21/25 = 84%

## Real teachers
- ministral-3:3b: 121/200 = 60.5%
- phi4-mini:3.8b: 121/200 = 60.5%
- qwen3:1.7b: 111/200 = 55.5%
- granite3.3:2b: 91/200 = 45.5% (193 valid A/B/C/D responses)
- gemma3:1b: 57/200 = 28.5%
- five-teacher oracle: **186/200 = 93.0%**

The largest diagnostic gap is selection/fusion rather than raw evidence availability. On `long_chain`, NEXUS scored 8/25 while the five-teacher oracle scored 24/25.

## R7.17 intervention audit
The recurrent controller did execute real substitutions:
- one-step: 21 query events, 9 substitution events across 8 tasks;
- two-step: 21 query events, 17 substitution events across 11 tasks.

However, none of these changed the final frozen answer state. Therefore 0/0 is not a no-op artifact: the recurrent policy altered evidence acquisition, but the resulting evidence never crossed the frozen answer switch/stop boundaries.

## Execution reproducibility note
The preregistered sequential evaluator exceeded the container wall-time. A 4-process execution-only harness called the exact frozen evaluator functions independently per item, without changing any decision rule. The first 153 items produced by the original sequential run were compared against the parallel result: **153/153 exact matches, 0 mismatches**.

Local confirmatory package SHA-256: `28a8f02e6ee7a79fe1fa2061ea6cdcfb3e054d979004be22476c93887a0807b4`.
