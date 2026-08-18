# Causal accounting requirement overlay v2

This layer is additive. The original requirement-registry v1 and requirement.binding/v1 contract remain unchanged.

The v2 overlay adds semantic requirements for:

- exact CapitalAuthorization v2 -> funding SettlementReceipt v2 binding;
- project revenue receipt lineage without automatic profit recognition;
- independent Financial OS receipt-vs-authorization verification;
- separate RevenueRecognition v2 before EXTERNAL_REVENUE posting;
- evidence re-evaluation immediately before double-entry posting;
- EconomicCycleTrace v4 recognition/ledger arrows;
- CommitmentEnvelope v3 policy-profile binding;
- preservation of all superseded generations.

The overlay cannot replace an existing `REQ-*` ID. It materializes into a v1-compatible registry only after collision checks and full revalidation. Existing requirement.binding/v1 can therefore bind a change-set to the exact materialized digest without rewriting its own protocol.

Changing or omitting the overlay changes the registry digest, so a change-set referencing the new requirements becomes `STALE_REGISTRY` against the old base registry.
