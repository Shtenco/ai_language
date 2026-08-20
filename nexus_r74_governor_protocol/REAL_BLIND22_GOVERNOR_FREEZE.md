# NEXUS R7.4 — Real Blind-22 Governor Freeze

Frozen before any Blind-22 teacher result was retrieved.

## Inputs
For each item and teacher retain: predicted A/B/C/D, raw output, token/top-token logprobs when available, runtime/model provenance.

## Metrics (no post-hoc tuning)
1. Individual teacher exact accuracy.
2. Equal-weight majority. Ties: summed teacher option probability from logprobs; if unavailable, deterministic lexical teacher order only for reproducibility.
3. Logprob consensus: for each teacher, convert available A/B/C/D token logprobs to a normalized four-option distribution; if unavailable, assign 0.70 to its selected option and 0.10 to each other option. Average across teachers; argmax is answer.
4. Selective Governor: use the same consensus distribution, but ABSTAIN if top probability < 0.55 OR top-minus-second margin < 0.12. Report coverage and accuracy-when-answered separately. Abstention counts as wrong only in strict utility.
5. Correlation haircut: when >=4/5 teachers select one option but their usable token distributions are unavailable/degenerate, do not treat votes as independent; multiply the modal excess over uniform by 0.85 before the abstention gate. This cannot change native Authority.
6. Native Authority invariant: any formally verified NEXUS Authority answer, when available, overrides all foreign teacher votes. No teacher result may modify Cyber/Logic/S-O-H core weights.

## Distillation protocol
Teacher outputs remain evidence, not truth. Build a separate Knowledge Organ from consensus labels/probabilities. Do not update native R7 core or R7.2 language binder. Evaluate transfer on post-freeze paraphrase forms not used to fit the Knowledge Organ, and rerun native regression/hash checks.

## Claim boundary
Real teacher inference can establish transfer from the executed teachers only. It does not imply open-domain AGI or universal truth. Selective Governor accuracy must always be reported with coverage.
