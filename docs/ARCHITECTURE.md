# Architecture

## Stable core

The stable core owns language truth:

```text
source -> parse -> normalize -> graph/AST -> fingerprint -> lower -> artifact
```

No network request is allowed inside this path. `compile_source()` is deterministic and can run offline.

### Modules

- `ir.py` — normalized instruction, graph and AST data structures.
- `errors.py` — source diagnostics.
- `pipeline.py` — parsing, canonicalization, fingerprinting, IR construction and backends.
- `compiler.py` — Python syntax validation and bounded developer execution.
- `client.py` — optional model adapter; intentionally outside deterministic compilation.
- `cli.py` — user-facing orchestration and artifact I/O.

## Probabilistic boundary

`AILanguageClient.plan()` can ask a model to propose DSL. The proposal is not trusted as compiler state. It must pass the exact same parser/canonicalizer as a hand-written file.

This gives the architecture a fail-closed boundary:

```text
LLM output --validate--> canonical DSL --deterministic compiler--> artifact
           \--invalid--> error
```

## NEXUS integration rule

NEXUS research should integrate as a **planner/reviewer adapter**, never by mutating the compiler's semantic rules at runtime.

A future adapter may implement:

```text
IntentPlanner.plan(intent) -> candidate .ailang
PlanReviewer.review(candidate, context) -> findings
```

The compiler remains the sole authority for grammar, normalization, graph/AST schemas, fingerprints and backend lowering.

This separation allows NEXUS A/B/C experiments to evolve without contaminating the reproducibility contract of AI Language core.

## Future typed lowering

The next architecture layer should introduce an explicit semantic-analysis phase:

```text
AST
 │
 ▼
name/dependency resolution
 │
 ▼
type + capability validation
 │
 ▼
typed IR
 │
 ├──> Python backend
 ├──> Rust backend
 ├──> Solidity backend
 └──> tool/plugin executor
```

Backend-specific behavior should be expressed as declared capabilities and diagnostics, not hidden prompt conventions.
