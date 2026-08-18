# SINERGY Requirement Binding V1 — 2026-08-18

## Purpose

This additive protocol gives `ai_language` one narrow federation authority: validate that an engineering change set references real `REQ-*` requirements from one exact requirement-registry version.

It does not review implementation quality, test outcomes, economic correctness or production readiness. Those responsibilities remain in their domain repositories and the engineering review layer.

## Inputs

- an `engineering.change-set/v1` object;
- a `sinergy.requirement-registry/v1` object;
- the registry digest against which the change set was prepared.

## Binding semantics

The registry digest covers the full validated requirement content, including statements, owners, implementation paths, test paths and verification metadata. Keeping the same `REQ-*` ID while changing its meaning therefore changes the digest.

Results are limited to:

- `BOUND`: all referenced requirements exist in the exact supplied registry version;
- `STALE_REGISTRY`: the caller prepared against another registry digest;
- `DRAFT_BLOCKED`: one or more referenced `REQ-*` identifiers do not exist.

An empty requirement-ref list may bind successfully; it simply means the change set declares no canonical requirement dependency. Whether that is acceptable for a particular risk domain is a separate `ai_coder` evidence-policy question.

## Authority boundary

The result always carries:

```text
authority = SEMANTIC_BINDING_ONLY
```

`BOUND` does not mean:

- tests passed;
- the code is correct;
- a PR may be merged;
- a deployment may proceed;
- a financial action is authorized.

`ai_language` therefore validates semantic provenance while `ai_coder` owns change-set evidence gating and Financial OS owns financial truth.

## Non-destructive evolution

This protocol is stacked above `docs/sinergy-semantic-governance-20260817`. The existing AI Language runtime/compiler/workspace model is not modified. Future requirement-binding generations should be added side-by-side.
