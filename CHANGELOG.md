# Changelog

## 0.4.0

### Added

- Versioned JSON IR schema and `PipelineResult.to_dict()`.
- Canonical semantic normalization and SHA-256 fingerprint.
- `inspect` CLI command.
- `plan` workflow: natural language -> validated canonical DSL -> deterministic compilation.
- AST and complete IR artifact emission.
- Execution timeout and isolated Python interpreter mode.
- Architecture, language specification and security documentation.
- Multi-version CI, format check, coverage gate and package build validation.

### Fixed

- `--emit-graph` no longer depends on `__dict__` for slot dataclasses.
- Constraints are preserved by every backend, not only Python.
- Target code safely escapes quotes, backslashes and common control characters.
- Packaging metadata now points to the real GitHub repository.
- Declared Python compatibility now matches the use of slot dataclasses.

### Changed

- Internal instruction constraints are immutable tuples.
- Generated artifacts embed their semantic fingerprint.
- README distinguishes deterministic compiler behavior from probabilistic LLM planning and from NEXUS research experiments.
