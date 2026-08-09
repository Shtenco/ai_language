"""Local repository-aware coding agent for AI Language Pro."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL, get_api_key
from .semantic_trace import SemanticTrace, build_semantic_trace


class AgentError(RuntimeError):
    """Base error for the coding agent."""


class AgentLimitError(AgentError):
    """Raised when the agent exceeds its tool-call turn budget."""


class WorkspaceSecurityError(AgentError):
    """Raised when a tool attempts to escape the configured workspace."""


_SECRET_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "secrets.json",
}
_SECRET_PARTS = {".ssh", ".aws", ".azure", ".gnupg"}
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_MAX_TOOL_OUTPUT = 30_000


def _clip(text: str, limit: int = _MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


@dataclass(slots=True)
class Workspace:
    """Constrained local workspace exposed to the model through tools."""

    root: Path
    read_only: bool = False
    auto_approve: bool = False
    approval_callback: Callable[[str, dict[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise WorkspaceSecurityError(
                f"Workspace does not exist or is not a directory: {self.root}"
            )

    def _resolve(self, path: str, *, allow_missing: bool = False) -> Path:
        raw = Path(path).expanduser()
        candidate = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceSecurityError(f"Path escapes workspace: {path}") from exc
        if not allow_missing and not candidate.exists():
            raise FileNotFoundError(path)
        return candidate

    def _relative(self, path: Path) -> str:
        rel = path.relative_to(self.root)
        return "." if rel == Path(".") else rel.as_posix()

    def _is_secret(self, path: Path) -> bool:
        rel = path.relative_to(self.root)
        if any(part in _SECRET_PARTS for part in rel.parts):
            return True
        name = path.name.lower()
        return (
            name in _SECRET_NAMES
            or name.startswith(".env.")
            or name.endswith(".pem")
            or name.endswith(".key")
        )

    def _require_approval(self, action: str, arguments: dict[str, Any]) -> None:
        if self.read_only:
            raise WorkspaceSecurityError(f"{action} is disabled in --read-only mode")
        if self.auto_approve:
            return
        if self.approval_callback is not None and self.approval_callback(action, arguments):
            return
        raise WorkspaceSecurityError(
            f"Approval required for {action}. Re-run with -y/--yes for autonomous local changes."
        )

    def list_files(self, path: str = ".", max_depth: int = 3) -> str:
        base = self._resolve(path)
        if not base.is_dir():
            return self._relative(base)
        max_depth = max(0, min(int(max_depth), 8))
        entries: list[str] = []
        base_depth = len(base.parts)
        for current, dirs, files in os.walk(base):
            current_path = Path(current)
            depth = len(current_path.parts) - base_depth
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in _SKIP_DIRS and not self._is_secret(current_path / d)
            )
            if depth >= max_depth:
                dirs[:] = []
            for name in sorted(files):
                file_path = current_path / name
                if self._is_secret(file_path):
                    continue
                entries.append(self._relative(file_path))
                if len(entries) >= 800:
                    entries.append("... [file listing truncated]")
                    return "\n".join(entries)
        return "\n".join(entries) or "[empty]"

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> str:
        file_path = self._resolve(path)
        if self._is_secret(file_path):
            raise WorkspaceSecurityError(f"Refusing to expose secret-like file: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(path)
        if file_path.stat().st_size > 2_000_000:
            raise AgentError(f"File is too large to read directly: {path}")
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, int(start_line))
        end = max(start, min(int(end_line), start + 799))
        selected = lines[start - 1 : end]
        numbered = [f"{idx}: {line}" for idx, line in enumerate(selected, start=start)]
        suffix = "\n... [more lines available]" if end < len(lines) else ""
        return _clip("\n".join(numbered) + suffix)

    def search_text(self, query: str, path: str = ".") -> str:
        if not query:
            raise ValueError("query must not be empty")
        base = self._resolve(path)
        candidates = [base] if base.is_file() else base.rglob("*")
        matches: list[str] = []
        needle = query.casefold()
        for candidate in candidates:
            if not candidate.is_file() or self._is_secret(candidate):
                continue
            try:
                rel_parts = candidate.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in _SKIP_DIRS for part in rel_parts):
                continue
            try:
                if candidate.stat().st_size > 1_000_000:
                    continue
                text = candidate.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeError):
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if needle in line.casefold():
                    matches.append(f"{self._relative(candidate)}:{idx}: {line.strip()}")
                    if len(matches) >= 120:
                        matches.append("... [search results truncated]")
                        return _clip("\n".join(matches))
        return _clip("\n".join(matches) if matches else "[no matches]")

    def write_file(self, path: str, content: str) -> str:
        file_path = self._resolve(path, allow_missing=True)
        if self._is_secret(file_path):
            raise WorkspaceSecurityError(
                f"Refusing to write secret-like file through the model: {path}"
            )
        self._require_approval("write_file", {"path": path, "chars": len(content)})
        existed = file_path.exists()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        verb = "Updated" if existed else "Created"
        return f"{verb} {self._relative(file_path)} ({len(content)} chars)"

    def replace_in_file(
        self, path: str, old_text: str, new_text: str, count: int = 1
    ) -> str:
        file_path = self._resolve(path)
        if self._is_secret(file_path):
            raise WorkspaceSecurityError(f"Refusing to edit secret-like file: {path}")
        original = file_path.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences == 0:
            raise AgentError(f"old_text was not found in {path}")
        count = max(1, int(count))
        self._require_approval(
            "replace_in_file",
            {
                "path": path,
                "occurrences": occurrences,
                "replace_count": min(count, occurrences),
            },
        )
        updated = original.replace(old_text, new_text, count)
        file_path.write_text(updated, encoding="utf-8")
        return (
            f"Updated {self._relative(file_path)}; "
            f"replaced {min(count, occurrences)} occurrence(s)"
        )

    def delete_file(self, path: str) -> str:
        file_path = self._resolve(path)
        if not file_path.is_file():
            raise AgentError("delete_file only removes files")
        if self._is_secret(file_path):
            raise WorkspaceSecurityError(f"Refusing to delete secret-like file: {path}")
        self._require_approval("delete_file", {"path": path})
        file_path.unlink()
        return f"Deleted {self._relative(file_path)}"

    def run_command(self, command: str, timeout: int = 120) -> str:
        self._require_approval("run_command", {"command": command})
        if not command.strip():
            raise ValueError("command must not be empty")
        shell_operators = ("&&", "||", ";", "|", ">", "<", "`", "$(")
        if any(token in command for token in shell_operators):
            raise WorkspaceSecurityError(
                "Shell operators are disabled; run one direct command at a time"
            )
        args = shlex.split(command, posix=os.name != "nt")
        if not args:
            raise ValueError("command must not be empty")
        executable = Path(args[0]).name.lower()
        forbidden = {
            "sudo",
            "su",
            "rm",
            "rmdir",
            "del",
            "erase",
            "format",
            "shutdown",
            "reboot",
            "poweroff",
            "mkfs",
            "dd",
            "bash",
            "sh",
            "zsh",
            "cmd",
            "cmd.exe",
            "powershell",
            "powershell.exe",
            "pwsh",
        }
        if executable in forbidden:
            raise WorkspaceSecurityError(
                f"Destructive/privileged command is blocked: {executable}"
            )
        if executable in {"python", "python3", "python.exe", "py"} and "-c" in args:
            raise WorkspaceSecurityError(
                "Inline Python (-c) is blocked; run repository scripts/modules instead"
            )
        if executable == "git" and len(args) > 1:
            safe_git = {
                "status",
                "diff",
                "grep",
                "log",
                "show",
                "rev-parse",
                "ls-files",
            }
            if args[1] not in safe_git:
                raise WorkspaceSecurityError(f"Mutating git subcommand is blocked: {args[1]}")
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if not any(
                secret in key.upper()
                for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
            )
        }
        timeout = max(1, min(int(timeout), 600))
        try:
            completed = subprocess.run(
                args,
                cwd=self.root,
                env=clean_env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"Command timed out after {timeout}s: {command}") from exc
        output = f"exit_code={completed.returncode}\n"
        if completed.stdout:
            output += f"stdout:\n{completed.stdout.rstrip()}\n"
        if completed.stderr:
            output += f"stderr:\n{completed.stderr.rstrip()}\n"
        return _clip(output.rstrip())

    def git_diff(self) -> str:
        if not (self.root / ".git").exists():
            return "[workspace is not a Git repository]"
        parts: list[str] = []
        for args in (["git", "status", "--short"], ["git", "diff", "--"]):
            completed = subprocess.run(
                args,
                cwd=self.root,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if completed.stdout.strip():
                parts.append(completed.stdout.rstrip())
            if completed.stderr.strip():
                parts.append(completed.stderr.rstrip())
        return _clip("\n\n".join(parts) if parts else "[clean working tree]")

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tools: dict[str, Callable[..., str]] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_text": self.search_text,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "delete_file": self.delete_file,
            "run_command": self.run_command,
            "git_diff": self.git_diff,
        }
        function = tools.get(name)
        if function is None:
            raise AgentError(f"Unknown tool: {name}")
        execution_arguments = dict(arguments)
        execution_arguments.pop("requirement_ids", None)
        return function(**execution_arguments)


_REQ_IDS_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "description": "REQ-* identifiers from the semantic task model that justify this action.",
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "list_files",
        "description": (
            "List repository files under a path. Secret-like files and dependency caches are hidden."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory; default '.'",
                },
                "max_depth": {"type": "integer", "minimum": 0, "maximum": 8},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file with line numbers. Read before editing whenever practical."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_text",
        "description": "Case-insensitive text search across repository files.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Create or fully rewrite one text file inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "requirement_ids": _REQ_IDS_SCHEMA,
            },
            "required": ["path", "content", "requirement_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "replace_in_file",
        "description": (
            "Perform an exact local text replacement in one file. Prefer this for focused edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
                "requirement_ids": _REQ_IDS_SCHEMA,
            },
            "required": ["path", "old_text", "new_text", "requirement_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "delete_file",
        "description": "Delete one regular file inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "requirement_ids": _REQ_IDS_SCHEMA,
            },
            "required": ["path", "requirement_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_command",
        "description": (
            "Run one non-privileged direct command in the workspace, e.g. tests, formatter, "
            "compiler, or git status. Shell chaining and destructive commands are blocked."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                "requirement_ids": _REQ_IDS_SCHEMA,
            },
            "required": ["command", "requirement_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "git_diff",
        "description": "Show git status and the current unstaged diff. This is read-only.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


_AGENT_INSTRUCTIONS = """You are AI Language Pro's local coding agent.
Work only inside the provided workspace using the tools. Inspect relevant files before changing them.
For build/fix/change requests, keep working until the requested local implementation is complete or a real blocker is reached. Use the smallest coherent edits, preserve existing style and APIs unless the task requires change, and run relevant non-destructive validation after edits.
Never request, read, print, or expose secrets, API keys, tokens, credentials, .env contents, or files outside the workspace. Never try to bypass tool restrictions. Do not use network commands unless the user explicitly asks for network work.
Do not claim tests passed unless you actually ran them and saw a successful result.
Use the semantic task model below as the authoritative traceability map. For every mutation or validation command, include all applicable requirement_ids. Do not invent requirement IDs.
When finished, give a concise summary of changed files, validation performed, and any remaining issue.

Workspace root: {workspace}

{semantic_context}
"""


@dataclass
class CodingAgent:
    """Responses API coding agent with semantic task and repository traceability."""

    workspace: Workspace
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    reasoning_effort: str = "high"
    max_steps: int = 30
    tool_logger: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=get_api_key(self.api_key))
        self._previous_response_id: str | None = None
        self.last_trace: SemanticTrace | None = None

    def reset(self) -> None:
        self._previous_response_id = None

    def trace_text(self) -> str:
        if self.last_trace is None:
            return "[no semantic trace available]"
        return self.last_trace.render_text()

    def trace_json(self) -> str:
        if self.last_trace is None:
            return "{}"
        return self.last_trace.to_json()

    def _log(self, message: str) -> None:
        if self.tool_logger is not None:
            self.tool_logger(message)

    def _create_response(
        self,
        input_data: Any,
        semantic_context: str,
        previous_response_id: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": _AGENT_INSTRUCTIONS.format(
                workspace=self.workspace.root,
                semantic_context=semantic_context,
            ),
            "input": input_data,
            "tools": TOOL_DEFINITIONS,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if self.model.startswith("gpt-5.6"):
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return self._client.responses.create(**kwargs)

    def run(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        repository_files = self.workspace.list_files(".", max_depth=5).splitlines()
        trace = build_semantic_trace(
            prompt,
            repository_files,
            repository_root=self.workspace.root,
        )
        self.last_trace = trace
        semantic_context = trace.instruction_context()

        response = self._create_response(
            prompt,
            semantic_context,
            self._previous_response_id,
        )
        max_steps = max(0, int(self.max_steps))
        steps_used = 0

        while True:
            calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]
            if not calls:
                self._previous_response_id = response.id
                return response.output_text.strip()
            if steps_used >= max_steps:
                break

            tool_outputs: list[dict[str, str]] = []
            for call in calls:
                name = call.name
                arguments: dict[str, Any] = {}
                try:
                    arguments = json.loads(call.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    preview_arguments = {
                        key: value
                        for key, value in arguments.items()
                        if key not in {"content", "old_text", "new_text"}
                    }
                    preview = json.dumps(preview_arguments, ensure_ascii=False)
                    self._log(f"→ {name} {preview[:240]}")
                    result = self.workspace.execute_tool(name, arguments)
                    first_line = result.splitlines()[0][:240] if result else "[empty]"
                    self._log(f"← {name}: {first_line}")
                except Exception as exc:
                    result = f"ERROR: {type(exc).__name__}: {exc}"
                    self._log(f"← {name}: {result}")
                trace.record(name, arguments, result)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": _clip(result),
                    }
                )

            response = self._create_response(
                tool_outputs,
                semantic_context,
                response.id,
            )
            steps_used += 1

        raise AgentLimitError(
            f"Agent stopped after {self.max_steps} tool rounds. "
            "Increase --max-steps if the task is legitimate and still incomplete."
        )


def terminal_approval(action: str, arguments: dict[str, Any]) -> bool:
    """Interactive approval callback used by the CLI."""
    if not sys.stdin.isatty():
        return False
    if action == "run_command":
        summary = arguments.get("command", "")
    else:
        summary = arguments.get("path", json.dumps(arguments, ensure_ascii=False))
    answer = input(f"Approve {action}: {summary}? [y/N] ").strip().lower()
    return answer in {"y", "yes"}
