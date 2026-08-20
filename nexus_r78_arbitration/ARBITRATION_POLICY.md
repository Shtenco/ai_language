# NEXUS R7.8 — TOP-2 COGNITIVE ARBITRATION POLICY (FROZEN)

This policy is frozen **before Blind-25 exists**.

## Sovereignty order
1. Native deterministic Authority, when it returns a proof-backed option.
2. Question-conditioned R7.7 Meta-Teacher ranking.
3. Foreign teacher evidence.
4. ABSTAIN if evidence cannot be resolved.

## Runtime cascade
A. Run `native_authority.solve(question, options)`.
   - If it proves an option: return it immediately. External teacher calls = 0.
B. Otherwise call R7.7-ranked Top-1 and Top-2 teachers.
   - If both return the same valid option: ACCEPT. Calls = 2.
C. If Top-1 and Top-2 disagree or one is invalid: call ranked Top-3.
   - If any valid option has >=2 votes among Top-3: ACCEPT that majority. Calls = 3.
D. If Top-3 have no 2-vote majority:
   - ABSTAIN. No post-hoc 4th/5th escalation is allowed in the confirmatory R7.8 policy.

## Hard invariants
- Foreign consensus never overrides a native Authority proof.
- Authority returns None rather than guessing outside bounded rules.
- R7.7 teacher ranking is determined from question text before any Blind-25 teacher output.
- No Blind-25 item, label, response, logprob, or result is used to tune this policy.
- Native R7-L core and R7.2 binder weights remain unchanged.

## Confirmatory metrics on future Blind-25
- accuracy when answered
- coverage
- strict accuracy (ABSTAIN counts wrong)
- mean external teacher calls per item
- fraction solved by native Authority
- top-1/top-2/top-3 and five-teacher oracle diagnostics
- native Blind-20 regression and SHA preservation

## Claim boundary
Blind-25 will be a targeted architecture stress test, not a general AGI benchmark.
