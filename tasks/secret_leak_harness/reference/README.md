# __ProjectName__

__domain_blurb__

The harness plants unique **canary tokens** into a __subject_noun__'s environment,
runs it, and scans everything it touches for those tokens — including base64- and
URL-encoded forms and truncated fragments — then reports where each secret leaked
and how severely. Severity ranks by sink: **network > file > log**.

Modules:

1. `__pkg__/canary.py` — generate canary tokens (`make_canary`) and validate them
   (`is_canary`, which checks the embedded checksum).
2. `__pkg__/plant.py` — plant canaries into a working environment: environment
   variables, a JSON config file, and an INI credentials file.
3. `__pkg__/outbound.py` — the mocked outbound-request log a __subject_noun__ writes
   to instead of making real network calls.
4. `__pkg__/runner.py` — run a __subject_noun__ under the planted environment and
   capture stdout/stderr, written files, and outbound requests (`Capture`).
5. `__pkg__/scan.py` — scan captured text, files and requests for canaries in every
   encoding; produces `Finding` records tagged by sink.
6. `__pkg__/report.py` — group findings per canary and rank them by sink severity.
7. `python -m __pkg__.cli [--subjects DIR] [--seed S] [--min-partial N]` runs the
   whole pipeline over the bundled sample __subject_noun__s in `data/`.

Data:

- `data/*.py` — sample __subject_noun__s that leak planted secrets in different ways
  (stdout, stderr, written files, outbound requests, encoded, truncated) or not at all.

Run tests with `python -m pytest`.
