# __ProjectName__

__DomainBlurb__

Components:

1. `__pkg__/limits.py` holds `Limits(cpu_seconds, wall_seconds, max_output_bytes, max_files)`.
   CPU time is enforced in the child via `RLIMIT_CPU` (the kernel sends SIGXCPU). Wall-clock time is
   enforced by the parent watchdog in `runner.run`: the child's whole process group is killed (so
   anything it forked dies too and cannot keep the run alive), `timed_out=True` and the violation
   `wall_timeout` is recorded. `max_output_bytes` caps **each** of stdout and stderr separately
   (a capped stream ends with a marker and `truncated=True`; the cut never splits a UTF-8 character). More than `max_files` files left in the work
   dir is reported as `too_many_files:<n>`.
2. `__pkg__/fs.py` creates the per-run temp tree (`work/` is the __SnippetNoun__'s cwd and the only place it
   may write), owns the read allowlist (interpreter install + any `--allow-read` dirs, relative ones
   resolved against the caller's cwd) and
   `collect_outputs(work)` lists the files a __SnippetNoun__ left behind.
3. `__pkg__/_bootstrap.py` runs inside the child interpreter (`python -I`): it installs a runtime import
   guard for the denylist, wraps `open()`, `io.open()` and `os.open()` so reads outside the allowlist
   and writes outside `work/` are refused whichever of those APIs is used, and traces every decision
   to `trace.jsonl` for the parent.
4. `__pkg__/runner.py`: `precheck(snippet)` rejects statically, by AST walk (not string matching), any
   __SnippetNoun__ that imports a denylisted module (`socket`, `subprocess`, `ctypes`, ...) through any
   import form: `import x`, `from x import y`, `importlib.import_module("x")`, `__import__("x")`.
   A denylisted module's private C accelerator (`_socket`, `_ssl`, `_ctypes`, ...) is denied with it,
   statically and at runtime.
   `run(snippet, limits) -> RunResult(stdout, stderr, exit_code, timed_out, files, violations, truncated)`.
   Violations are plain strings: `denied_import:<mod>`, `syntax_error:<line>`, `import_denied:<mod>`,
   `read_denied:<path>`, `write_outside_sandbox:<path>`, `cpu_limit`, `wall_timeout`, `too_many_files:<n>`.
   Each distinct violation is listed once, in the order it first occurred.
5. `__pkg__/audit.py` appends one JSON line per run (timestamp, SHA-256 of the __SnippetNoun__ source,
   limits, outcome) so a reviewer can tie every result back to exactly the code that produced it.
   The log must survive a writer that crashed mid-line: readers skip corrupt lines, and the next
   `record()` starts on a fresh line instead of gluing its entry onto the fragment.
6. `__pkg__/cli.py`: `python -m __pkg__.cli snippet.py --cpu 2 --wall 5 [--max-output BYTES] [--max-files N]
   [--audit PATH] [--allow-read DIR] [--json]`.

Sample __SnippetsNoun__ in `data/` and what the sandbox should do with each:

| file | expected outcome |
|---|---|
| `__Snippet1Name__.py` | exit 0, no violations |
| `__Snippet2Name__.py` | exit 0, CPU-heavy but finishes |
| `__Snippet3Name__.py` | killed by the CPU limit (`cpu_limit`, `timed_out`) |
| `__Snippet4Name__.py` | killed by the wall-clock limit (`wall_timeout`, `timed_out`) |
| `__Snippet5Name__.py` | stdout capped, `truncated=True`, stderr intact |
| `__Snippet6Name__.py` | rejected by the pre-check (`denied_import:socket`) |
| `__Snippet7Name__.py` | rejected by the pre-check (`denied_import:subprocess`) |
| `__Snippet8Name__.py` | runs, runtime guard fires (`import_denied:socket`), exit 1 |
| `__Snippet9Name__.py` | exit 0, `files == ["__OutputFileName__"]` |
| `__Snippet10Name__.py` | refused, `write_outside_sandbox:/tmp/__EscapeFileName__` |
| `__Snippet11Name__.py` | refused, `read_denied:/etc/passwd` |
| `__Snippet12Name__.py` | exit 3, stderr preserved |
| `__Snippet13Name__.py` | rejected by the pre-check (`denied_import:ctypes`) |

Run tests with `python -m pytest`.
