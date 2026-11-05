"""Agent tools: a small filesystem + shell sandbox rooted at a workspace dir."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

MAX_OUTPUT = 12_000


class Workspace:
    def __init__(self, root: Path, shell_timeout: int = 120):
        self.root = Path(root).resolve()
        self.shell_timeout = shell_timeout

    def _resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"path escapes workspace: {rel}")
        return p

    def list_files(self, path: str = ".") -> str:
        base = self._resolve(path)
        if not base.exists():
            return f"ERROR: {path} does not exist"
        lines = []
        for p in sorted(base.rglob("*")):
            if "__pycache__" in p.parts or ".pytest_cache" in p.parts or p.suffix == ".pyc":
                continue
            if p.name.startswith("._hidden"):
                continue
            rel = p.relative_to(self.root)
            lines.append(f"{rel}{'/' if p.is_dir() else ''}  ({p.stat().st_size}B)" if p.is_file() else f"{rel}/")
        return "\n".join(lines) or "(empty)"

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"ERROR: {path} is not a file"
        text = p.read_text(errors="replace")
        if len(text) > 60_000:
            return text[:60_000] + f"\n... [truncated, {len(text)} chars total]"
        return text

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def run(self, command: str) -> str:
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)  # agent must not see the key
        env["PYTHONPATH"] = str(self.root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            r = subprocess.run(
                command, shell=True, cwd=self.root, capture_output=True, text=True,
                timeout=self.shell_timeout, env=env,
            )
            out = r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else "")
            out += f"\n[exit code {r.returncode}]"
        except subprocess.TimeoutExpired:
            out = f"ERROR: command timed out after {self.shell_timeout}s"
        if len(out) > MAX_OUTPUT:
            out = out[: MAX_OUTPUT // 2] + "\n... [truncated] ...\n" + out[-MAX_OUTPUT // 2 :]
        return out

    def dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "list_files":
                return self.list_files(args.get("path", "."))
            if name == "read_file":
                return self.read_file(args["path"])
            if name == "write_file":
                return self.write_file(args["path"], args["content"])
            if name == "run":
                return self.run(args["command"])
            return f"ERROR: unknown tool {name}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "Recursively list files in the workspace (or a subdirectory).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write (create or overwrite) a text file in the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run",
        "description": "Run a shell command from the workspace root (python, pytest, etc). Returns stdout/stderr/exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }},
]


def tool_result_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
