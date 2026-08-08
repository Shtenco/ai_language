# AI Language Core Specification — IR schema v1

## 1. Source grammar

The v1 language is intentionally line-oriented:

```text
program      := { blank | comment | instruction }
instruction  := action whitespace target [ whitespace? "|" whitespace? constraints ]
constraints  := constraint { whitespace? ";" whitespace? constraint }
comment      := optional-whitespace "#" text
```

`action`, `target` and every retained constraint must be non-empty after normalization.

## 2. Normalization

For every instruction:

- leading/trailing whitespace is removed;
- the action is lower-cased;
- internal whitespace in the target is collapsed to one ASCII space;
- internal whitespace in every constraint is collapsed to one ASCII space;
- empty constraint fragments are discarded;
- blank lines and full-line comments do not enter the semantic program.

Canonical rendering is:

```text
<action> <target>
```

or:

```text
<action> <target> | <constraint1>; <constraint2>; ...
```

with exactly one trailing newline for the complete canonical program.

## 3. Semantic fingerprint

The fingerprint is:

```text
SHA256(UTF8(canonical_program))
```

It is independent of target backend. The purpose is semantic identity and reproducibility, not cryptographic signing or authorship proof.

## 4. Semantic graph

The graph always contains a root node:

```text
id=program, kind=root, value=AIProgram
```

Each instruction becomes `instr_N`, linked by `program --contains--> instr_N`.
Every constraint becomes an instruction-local node linked by `constrained_by`.
Node IDs are deterministic and depend only on normalized instruction order.

## 5. AST

IR schema v1 uses a minimal AST whose root owns the ordered normalized instruction tuple. The AST is deliberately simple; future schema versions may add typed expressions, dependencies, outputs and backend capabilities.

## 6. JSON IR

A complete serialized result contains:

- `schema_version`
- `target`
- `fingerprint`
- `instructions`
- `semantic_graph`
- `ast`
- optionally `code`

Consumers must reject unknown incompatible schema versions rather than guessing field meaning.

## 7. Backend contract

Every v0.4 backend must:

1. preserve normalized instruction order;
2. preserve every normalized constraint;
3. embed the semantic fingerprint;
4. escape string content so target-language string syntax is preserved;
5. produce deterministic bytes for the same normalized program and target.

Backend-specific literal rules are part of lowering. Kotlin escapes `$` so source text cannot accidentally become a string template. Solidity emits `unicode"..."` string literals so non-ASCII UTF-8 instructions remain valid source.

The current backends are semantic scaffolds, not general-purpose program synthesis.
