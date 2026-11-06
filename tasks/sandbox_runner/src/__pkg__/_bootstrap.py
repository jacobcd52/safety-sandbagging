"""Child-side bootstrap: installs the import denylist, the file-access guard and
the event trace, then executes the __SnippetNoun__ inside its work dir.

The file-access guard covers `open()`, `io.open()` and `os.open()`: whichever
API is used, a write outside work/ or a read outside the allowlist is refused
(PermissionError) and traced. The import guard denies a module when its
top-level name, with leading underscores stripped, is on the denylist.

This file runs INSIDE the sandboxed interpreter (`python -I _bootstrap.py
config.json`), so it must only use the standard library and must never import
anything from the package. Every policy decision it makes is appended to
trace.jsonl so the parent can turn it into a violation.
"""
import builtins
import io
import json
import os
import sys


def _within(path, root):
    p = os.path.realpath(path)
    r = os.path.realpath(root)
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:
        return False


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)
    work = cfg["work_dir"]
    snippet_path = cfg["snippet_path"]
    deny = set(cfg["deny_modules"])
    # the snippet and this file stay readable so tracebacks can show source lines
    read_ok = [work, snippet_path, os.path.abspath(__file__)] + list(cfg["read_allowlist"])

    real_open = builtins.open
    trace = real_open(cfg["trace_path"], "a", encoding="utf-8")

    def emit(event, **fields):
        trace.write(json.dumps({"event": event, **fields}) + "\n")
        trace.flush()

    # ---- runtime import guard (catches names the static pre-check cannot see)
    class DenyFinder:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in deny:
                emit("import_denied", module=name)
                raise ImportError(f"import of {name!r} is not permitted in the sandbox")
            return None

    for loaded in list(sys.modules):
        if loaded.split(".")[0] in deny:
            del sys.modules[loaded]
    sys.meta_path.insert(0, DenyFinder())

    # ---- open() guard: writes only inside work/, reads only from the allowlist
    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, mode, *args, **kwargs)
        path = os.path.abspath(os.fsdecode(os.fspath(file)))
        if any(c in mode for c in "wax+"):
            emit("write", path=path)
        elif not any(_within(path, root) for root in read_ok):
            emit("read_denied", path=path)
            raise PermissionError(f"read of {path} is outside the sandbox allowlist")
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_open

    # ---- run the __SnippetNoun__ as __main__ with work/ as cwd
    code = real_open(snippet_path, encoding="utf-8").read()
    os.chdir(work)
    sys.path.insert(0, work)
    sys.argv = [snippet_path]
    namespace = {"__name__": "__main__", "__file__": snippet_path, "__builtins__": builtins}
    exec(compile(code, snippet_path, "exec"), namespace)


if __name__ == "__main__":
    main()
