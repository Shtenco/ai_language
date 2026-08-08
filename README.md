# AI Language Pro

**Intent DSL → normalized semantics → semantic graph → AST → reproducible target scaffold**

AI Language Pro is a compact deterministic language core for describing software intent at a level above implementation code. The optional LLM layer can translate a natural-language request into the DSL, but parsing, normalization, graph construction, AST construction, fingerprinting and code generation are deterministic and work offline.

> **Current status: v0.4 beta.** The project is a real compiler/tooling prototype and research platform. The five target backends currently emit executable **semantic scaffolds** that preserve the normalized instructions; they do not yet synthesize a complete business application from intent alone.

## Why this architecture matters

The project deliberately separates two very different concerns:

1. **Probabilistic planning** — an LLM may propose `.ailang` source from a human request.
2. **Deterministic compilation** — the source is parsed, normalized, fingerprinted, represented as graph/AST and lowered to a target scaffold without hidden model calls.

This boundary makes the transformation inspectable, testable and reproducible. The same normalized program always receives the same semantic SHA-256 fingerprint.

```text
natural-language intent (optional)
            │
            ▼
        LLM planner
            │
            ▼
       .ailang source
            │
            ▼
 deterministic parser
            │
      ┌─────┴─────┐
      ▼           ▼
semantic graph    AST
      └─────┬─────┘
            ▼
 semantic fingerprint
            │
            ▼
Python / C / Rust / Solidity / Kotlin scaffold
```

## Implemented in v0.4

- Strict line-oriented `.ailang` parser with line-numbered syntax errors.
- Canonical normalization and stable semantic SHA-256 fingerprint.
- Semantic graph and AST with a versioned JSON IR schema.
- Deterministic backends for **Python, C, Rust, Solidity and Kotlin**.
- Constraint-preserving and quote-safe target string generation.
- `generate`, `inspect`, `check`, `run`, `ask` and new `plan` CLI workflows.
- Optional OpenAI Responses API adapter for natural-language planning.
- Python validation plus bounded local execution (`--timeout`, isolated interpreter mode).
- Test suite, coverage gate, formatter/linter checks and multi-version CI.
- Explicit architecture, language spec and security documentation.

## AI Language syntax

Each meaningful line has one instruction:

```text
ACTION TARGET | constraint1; constraint2; constraint3
```

The constraints section is optional. Blank lines and full-line comments beginning with `#` are ignored.

Example:

```text
generate payment_service | idempotency; retries; structured logging
validate public_api | schema compatibility; error contracts
emit documentation | concise; include examples
```

Normalization is deliberate. For example, repeated whitespace is collapsed and actions are lower-cased before fingerprinting. See [`docs/SPEC.md`](docs/SPEC.md) for the exact grammar and invariants.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Python **3.10+** is required.

## CLI

### Compile a deterministic source file

```bash
ai-language generate examples/service.ailang \
  --target python \
  --out build/service.py \
  --emit-graph build/graph.json \
  --emit-ast build/ast.json \
  --emit-ir build/ir.json
```

The CLI prints the first 12 characters of the semantic fingerprint, while the full fingerprint is embedded in generated source and JSON artifacts.

### Inspect the IR without writing generated code

```bash
ai-language inspect examples/service.ailang
ai-language inspect examples/service.ailang --target rust --include-code
```

### Natural language → validated DSL → compiled scaffold

```bash
export OPENAI_API_KEY="..."

ai-language plan \
  "Design a payment service with retries, idempotency and structured logging" \
  --target rust \
  --out build/payment.rs \
  --emit-source build/payment.ailang \
  --emit-ir build/payment.ir.json
```

`plan` is intentionally two-stage: the model proposes DSL, then the deterministic parser validates and canonicalizes it before compilation. Invalid model output fails closed instead of being silently executed.

### Free-form model runtime

```bash
ai-language ask "Explain the trade-off between retries and idempotency" --model gpt-4o-mini
```

### Validate and run generated Python

```bash
ai-language check build/service.py
ai-language run build/service.py --timeout 3
```

`run` is a developer convenience, **not a security sandbox**. Never execute untrusted generated Python on a sensitive host. See [`SECURITY.md`](SECURITY.md).

## Python SDK

```python
from ai_language import compile_source

source = """
generate anti_fraud_service | observability; retries
validate interfaces | backwards compatibility
"""

result = compile_source(source, target="rust")

print(result.fingerprint)
print(result.semantic_graph.nodes)
print(result.ast)
print(result.code)
print(result.to_dict(include_code=False))
```

## Reproducibility contract

The fingerprint is calculated from canonical semantic source, not formatting. Therefore these inputs are semantically identical:

```text
generate   api   | auth ; retries
```

```text
generate api | auth; retries
```

Both normalize to:

```text
generate api | auth; retries
```

and receive the same SHA-256 fingerprint.

## Development quality gate

```bash
ruff check .
ruff format --check .
pytest --cov=ai_language --cov-report=term-missing
python -m build
```

CI tests supported Python versions and enforces lint/format/test/coverage/package-build checks.

## Research / NEXUS branches

The repository also contains experimental NEXUS benchmark branches and draft PRs (external benchmark protocols, SWE-bench harness work and guarded-consensus experiments). They are intentionally **not** mixed into the stable compiler core until they expose a clean interface and pass independent reproducibility gates. This keeps research evidence separate from production-facing language semantics.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the intended integration boundary.

## Roadmap

The next meaningful compiler milestones are:

- typed operands, outputs and dependency edges;
- a formal semantic validation/type-check phase;
- backend capability declarations and target-specific diagnostics;
- tool/plugin lowering rather than print-only semantic scaffolds;
- signed build manifests and provenance;
- deterministic planner evaluation against frozen intent→DSL corpora;
- a stable adapter that lets NEXUS reasoning propose/criticize plans without owning compiler semantics.

## License

This repository is distributed under the included **AI Language Pro Commercial License v1.0**. Commercial use requires a separate agreement. See [`LICENSE`](LICENSE).

## Donations

- **ETH:** `0x980Ddb04c54979b3Ed23df4a7DBc7049b7d0D686`
- **BTC:** `bc1q49rfm0p6qh6nlnm4az4yhhk9x82zfxwgtcnhvm`
