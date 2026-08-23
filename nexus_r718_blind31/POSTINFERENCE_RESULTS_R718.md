# NEXUS R7.18 — Blind-31 post-inference result

## Frozen lineage
- R7.18 freeze commit (before Blind-31): `68023d51825eeefe5432263c863468e44636d862`
- R7.18 exact ZIP SHA-256: `12a5c190d2f6d3a88a938b2b064ff0e7341df6ec9c169a7462c9ab4ee904021d`
- Blind-31 SHA-256: `49c587e0cf9f310df7720a7623169f3f24c6eae380f4778509af7b3bf7920dbb`

## Primary evidence
Primary evidence is the complete second five-teacher attempt, GitHub Actions run `32659291525`, selected as the whole-attempt recovery rule before the run fully completed. No per-model attempt cherry-picking.

The original preregistered teacher runner had a post-inference `None in 'ABCD'` summary bug for Gemma. The one-line technical fix did not alter prompt, benchmark, model, seed, generation options, or NEXUS. CPU/Ollama inference was empirically not bit-deterministic across attempts, so the entire second attempt is primary; the incomplete first attempt is replication/audit only.

## Blind-31 result (N=200)
- frozen R7.16: **140/200 = 70.0%**
- frozen R7.17: **140/200 = 70.0%**
- frozen **R7.18: 156/200 = 78.0%**
- majority of 5 teachers: **135/200 = 67.5%**
- 5-teacher oracle: **181/200 = 90.5%**
- best single teacher Phi4-mini: **143/200 = 71.5%**

R7.18 vs R7.16/R7.17:
- corrections: **21**
- regressions: **5**
- discordant: 26
- exact two-sided paired p = **0.002493917942047119**
- absolute gain = **+8.0 pp**

R7.18 vs best single Phi4-mini:
- 21 corrections / 8 regressions
- exact paired p = **0.0241195447742939**

R7.18 Wilson 95% CI ≈ **71.76%–83.18%**.

## R7.18 by category
- causal_graph_transfer: 20/25 = 80%
- code_state_novel: 20/25 = 80%
- evidence_policy: 25/25 = 100%
- multi_step_algebra: 7/25 = 28%
- multilingual_entailment: 24/25 = 96%
- probability_logic: 23/25 = 92%
- revision_memory: 21/25 = 84%
- structural_transfer: 16/25 = 64%

## Frozen runtime exceptions
Blind-31 exposed a real inherited runtime bug: R7.10 formal parsing prematurely evaluates `5 // 0` inside Python try/except tasks and raises ZeroDivisionError before the R7.18 restricted-AST layer. R7.18 was NOT patched for Blind-31. Five runtime exceptions are scored as wrong via an external per-item exception wrapper. The preregistered evaluator remains preserved unchanged.

## Teacher primary attempt #2
- Phi4-mini 3.8B: 143/200 = 71.5%
- Ministral 3B: 133/200 = 66.5%
- Granite 2B: 127/200 = 63.5%
- Qwen3 1.7B: 119/200 = 59.5%
- Gemma3 1B: 61/200 = 30.5%; 197 valid letters

## Replication variance, attempt1→attempt2
- Ministral: 0/200 prediction differences; 66.5%→66.5%
- Phi: 5/200; 71.5%→71.5%
- Granite: 11/200; 65.0%→63.5%
- Qwen: 16/200; 57.0%→59.5%

## Blind-30 replay boundary
After Blind-30 disclosure and R7.18 development, R7.18 replay is **200/200 = 100%**, and B22–B28 disclosed replay is **624/624 = 100%**. Those are development/regression results only, NOT blind claims.

The fresh post-freeze confirmatory result is Blind-31: **156/200 = 78.0%**.