from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")

replacements = [
    (
        "AI Language Pro is a Python package that contains two related but independent components:\n\n"
        "1. **`ail` — a repository-aware coding agent** that can inspect a local project, edit files, run controlled commands, and iterate through multi-step coding tasks using the OpenAI Responses API.\n"
        "2. **AI Language compiler — an experimental instruction compiler** that parses a small `.ailang` format into an intermediate semantic graph, an AST, and simple target-language output.\n\n"
        "The coding agent is the primary end-user feature in the `0.4.x` series. The compiler is an experimental research component and should not be confused with a production compiler or a general-purpose programming language implementation.\n",
        "AI Language Pro is a Python package for repository-aware coding with explicit semantic traceability. It combines three layers:\n\n"
        "1. **`ail` — a repository-aware coding agent** that can inspect a local project, edit files, run controlled commands, and iterate through multi-step coding tasks using the OpenAI Responses API.\n"
        "2. **Semantic task and repository graphs** that normalize requirements into stable `REQ-*` identifiers, index Python files/symbols/imports/calls/tests, and record which requirements caused each mutation or validation command.\n"
        "3. **AI Language compiler — an experimental instruction compiler** that parses a small `.ailang` format into an intermediate semantic graph, an AST, and simple target-language output.\n\n"
        "Starting with `0.5.x`, the semantic graph is part of the coding-agent execution path rather than a separate demonstration. Each task can produce a machine-readable trace containing the intent graph, repository graph, change graph, impact graph, requirement coverage, and unresolved requirements. The `.ailang` compiler remains an experimental research component rather than a production general-purpose compiler.\n",
    ),
    (
        "  - [Execution model](#execution-model)\n  - [Available tools](#available-tools)",
        "  - [Execution model](#execution-model)\n  - [Semantic traceability](#semantic-traceability)\n  - [Available tools](#available-tools)",
    ),
    (
        "In interactive mode, the previous Responses API response ID is retained so follow-up prompts continue the same conversation. `/clear` resets that conversation state.\n\n### Available tools",
        "In interactive mode, the previous Responses API response ID is retained so follow-up prompts continue the same conversation. `/clear` resets that conversation state.\n\n"
        "### Semantic traceability\n\n"
        "Before the first model call for a task, the agent builds a deterministic semantic task model. Natural-language clauses become stable requirement identifiers such as `REQ-1`, `REQ-2`, and `REQ-3`. Requirements are classified as actions, constraints, or validation requirements.\n\n"
        "For Python repositories, the repository graph performs bounded best-effort AST analysis and can include file, class, function, method, test, module-import, and call relationships. The graph is built from files visible through the workspace; secret-like paths remain excluded.\n\n"
        "Mutating tools (`write_file`, `replace_in_file`, `delete_file`) and validation commands (`run_command`) must carry the `requirement_ids` that justify the action. The resulting trace can therefore derive four related views:\n\n"
        "```text\n"
        "IntentGraph        requirement structure\n"
        "RepositoryGraph    files, Python symbols, imports and calls\n"
        "ChangeGraph        requirement -> tool event -> target\n"
        "ImpactGraph        changed files -> defined symbols -> affected tests\n"
        "```\n\n"
        "Requirement coverage distinguishes `implemented`, `addressed`, `verified`, and `unresolved` states instead of treating any model response as proof that the task is complete. Validation requirements are only considered verified after a successful mapped `run_command` event.\n\n"
        "The JSON trace schema for this release is `ai-language.semantic-trace/v2`. It intentionally stores event metadata and short result summaries, not file contents, replacement text, API keys, or credential material.\n\n"
        "Interactive inspection:\n\n"
        "```text\n"
        "/trace\n"
        "/coverage\n"
        "```\n\n"
        "One-shot export suitable for CI artifacts:\n\n"
        "```bash\n"
        "ail -y --trace-out build/ail-trace.json --show-coverage \\\n"
        "  \"Implement the change and run the relevant tests\"\n"
        "```\n\n"
        "For workflows that want an explicit gate, `--fail-unresolved` returns exit code `3` when the derived trace still contains unresolved requirements. This is an opt-in policy because review/audit prompts may intentionally perform no mutations.\n\n"
        "### Available tools",
    ),
    (
        "Commands: /help, /clear, /diff, /exit",
        "Commands: /help, /clear, /diff, /trace, /coverage, /exit",
    ),
    (
        "| `/diff` | Display local Git status and unstaged diff |\n| `/exit` | Exit the session |",
        "| `/diff` | Display local Git status and unstaged diff |\n"
        "| `/trace` | Show the latest requirement-to-action semantic trace |\n"
        "| `/coverage` | Show requirement implementation/verification coverage |\n"
        "| `/exit` | Exit the session |",
    ),
    (
        "A task that is allowed to modify files still uses the approval policy described below.\n",
        "A task that is allowed to modify files still uses the approval policy described below.\n\n"
        "Export the latest trace and print coverage:\n\n"
        "```bash\n"
        "ail -y --trace-out build/trace.json --show-coverage \\\n"
        "  \"Refactor the parser and run the relevant tests\"\n"
        "```\n\n"
        "Fail a CI step when semantic requirements remain unresolved:\n\n"
        "```bash\n"
        "ail -y --fail-unresolved --trace-out build/trace.json \\\n"
        "  \"Apply the requested change and validate it\"\n"
        "```\n",
    ),
    (
        "| `--quiet-tools` | off | Suppress tool progress messages on stderr |",
        "| `--quiet-tools` | off | Suppress tool progress messages on stderr |\n"
        "| `--trace-out FILE` | none | Write the latest `semantic-trace/v2` JSON after a completed task |\n"
        "| `--show-coverage` | off | Print requirement coverage after a completed task |\n"
        "| `--fail-unresolved` | off | In one-shot mode return exit code 3 when requirements remain unresolved |",
    ),
    (
        "report = agent.run(\"Review the repository architecture\")\nprint(report)\n```",
        "report = agent.run(\"Review the repository architecture\")\n"
        "print(report)\n"
        "print(agent.trace_text())\n"
        "print(agent.trace_json())\n"
        "if agent.last_trace is not None:\n"
        "    print(agent.last_trace.render_coverage_text())\n"
        "```",
    ),
    (
        "│       ├── config.py      # API key and default model configuration\n│       ├── ir.py",
        "│       ├── config.py      # API key and default model configuration\n"
        "│       ├── semantic_trace.py # Requirement, repository, change, impact, coverage graphs\n"
        "│       ├── ir.py",
    ),
    (
        "- Interactive conversation state is maintained through the model API response chain; there is no independent persistent project-memory database.\n- Tool output is intentionally truncated at bounded sizes.",
        "- Interactive conversation state is maintained through the model API response chain; there is no independent persistent project-memory database.\n"
        "- Python symbol/import/call indexing is bounded best-effort static analysis; dynamic dispatch, runtime imports, generated code, and cross-language symbol resolution are not fully modeled.\n"
        "- Requirement extraction is deterministic and intentionally lightweight; complex specifications may need a future richer planning/IR layer.\n"
        "- Requirement coverage is provenance-based evidence, not a formal proof of semantic correctness.\n"
        "- Tool output is intentionally truncated at bounded sizes.",
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f"README anchor not found: {old[:90]!r}")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
