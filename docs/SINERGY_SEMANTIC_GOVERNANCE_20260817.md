# AI Language semantic governance for SINERGY

Status: additive integration document. Existing AI Language Pro agent/compiler behavior remains unchanged.

## Purpose

`Shtenco/ai_language` can provide machine-readable requirement traceability across the SINERGY repository federation.

The existing semantic trace model already distinguishes intent, repository, change and impact graphs. For SINERGY, that mechanism should be extended by convention into cross-repository architectural requirements.

## Requirement namespaces

Recommended stable namespaces:

```text
REQ-FIN-*     financial constitution / ledger / NAV
REQ-MAT-*     household savings / maturity / funding
REQ-PAY-*     payment / settlement / compliance
REQ-EUR-*     real economy / project / EIIOM
REQ-AI-*      OLGA/NEXUS authority and evidence
REQ-CHAIN-*   blockchain/control-plane commitments
REQ-MEM-*     memory/compression/provenance
REQ-SEC-*     security / custody / secret boundaries
```

## Example invariants

```text
REQ-FIN-001  Own-token market cap MUST NOT enter HardNAV.
REQ-FIN-002  Principal repayment MUST NOT be classified as revenue.
REQ-MAT-001  Loan maturity MUST be shorter than funding maturity unless a separately approved replacement-liquidity facility exists.
REQ-PAY-001  A quote MUST NOT be treated as a settlement receipt.
REQ-PAY-002  Provider execution MUST fail closed without required capacity/readiness.
REQ-AI-001   Raw model output MUST NOT mutate the canonical ledger directly.
REQ-CHAIN-001 Public/browser state MUST NOT become accounting authority.
```

## Traceability chain

Preferred development evidence:

```text
requirement
 -> source architecture document
 -> repository/file/symbol impact
 -> code change
 -> test/validation command
 -> artifact/evidence
 -> release/deployment reference
```

A change that addresses a requirement but has not passed its mapped validation remains `addressed`, not `verified`.

## Cross-repository change sets

Large SINERGY features often span multiple repositories. A common change identifier should link traces without forcing a monorepo.

Example:

```text
CHANGESET-FINOS-2026-08-17-001
  synergy_financial_os   schema + invariant
  synergy_matrix_sota    producer adapter
  synergy_pay_system     settlement adapter
  synergychain           read-only consumer
```

Each repository keeps its own Git history while the change-set metadata records federation-level intent.

## Legacy preservation

Semantic governance must distinguish:

- requirements governing new canonical code;
- historical artifacts preserved for provenance;
- explicitly superseded claims;
- negative controls that must continue to fail/pass in defined ways.

A requirement to preserve legacy evidence is itself testable: migration code should not silently delete or overwrite the referenced snapshot/branch/artifact.

## Security

AI Language remains an engineering tool, not a financial authority. Requirement coverage can prove that mapped tests ran; it cannot prove legal compliance, economic truth or external settlement without the corresponding external evidence.

## Recommended CI direction

Federation repositories can export semantic-trace artifacts and fail CI when mandatory canonical requirements are unresolved. High-risk rules should also have direct deterministic tests independent of the agent trace.
