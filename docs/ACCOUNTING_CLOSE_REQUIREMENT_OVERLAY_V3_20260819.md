# Accounting-close requirement overlay v3

This additive semantic layer extends the already materialized causal-accounting requirement overlay v2. It does not edit requirement.binding/v1 or reuse existing REQ IDs.

New requirements formalize:

- exact ReportingEntry <-> LedgerEvent v2 reconstruction and balance;
- strict genesis-based AccountingClose v2 prefix semantics;
- LedgerInclusionProof v2 binding both LedgerEvent and ReportingEntry roots;
- mandatory AccountingClose/Proof phases in non-circular Trace v6;
- rejection of fully re-hashed downstream bundles when recognized economics change;
- immutable CommitmentEnvelope v5 policy generation;
- independent read-only SINERGYCHAIN verification without financial authority.

`materialize_requirement_overlay_v3()` accepts only an exact previously materialized registry digest, rejects stale bases and rejects every REQ-ID collision. Its output remains `sinergy.requirement-registry/v1` compatible, so the existing requirement.binding/v1 contract stays unchanged.
