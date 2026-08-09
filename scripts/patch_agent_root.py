from pathlib import Path

path = Path("src/ai_language/agent.py")
text = path.read_text(encoding="utf-8")
old = "        trace = build_semantic_trace(prompt, repository_files)\n"
new = (
    "        trace = build_semantic_trace(\n"
    "            prompt,\n"
    "            repository_files,\n"
    "            repository_root=self.workspace.root,\n"
    "        )\n"
)
if old not in text:
    raise SystemExit("expected build_semantic_trace call not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
