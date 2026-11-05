"""Export __records__ to a sharded, redacted JSONL bundle plus manifest.

    python -m __pkg__.cli data/__records__.jsonl out_dir --shards 4 --redact source

Steps: load -> validate (invalid __records__ are dropped and counted) -> dedup
-> redact -> shard -> manifest. `--redact` may be repeated; use a dotted name
such as ``annotations.rationale`` for a field inside each annotation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dedup import dedup
from .redact import redact_all
from .schema import load_records, validate
from .shard import write_manifest, write_shards


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export __records__ for __ProjectName__.")
    ap.add_argument("input", help="input JSONL, one __Record__ per line")
    ap.add_argument("out_dir", help="directory to write shards and manifest.json into")
    ap.add_argument("--shards", type=int, default=4, help="number of output shards")
    ap.add_argument("--redact", action="append", default=[], metavar="FIELD",
                    help="field to mask in every exported __Record__ (repeatable)")
    args = ap.parse_args(argv)

    records = load_records(args.input)
    valid, n_invalid = [], 0
    for record in records:
        errors = validate(record)
        if errors:
            n_invalid += 1
            print(f"skipping {record.get('id')!r}: {'; '.join(errors)}", file=sys.stderr)
        else:
            valid.append(record)
    unique = dedup(valid)
    exported = redact_all(unique, args.redact)

    out_dir = Path(args.out_dir)
    shards = write_shards(exported, out_dir, args.shards)
    manifest = write_manifest(
        out_dir, shards,
        input_records=len(records),
        invalid_records=n_invalid,
        duplicates_removed=len(valid) - len(unique),
        redacted_fields=list(args.redact),
    )
    print(f"loaded={len(records)} invalid={n_invalid} unique={len(unique)} "
          f"written={manifest['total_records']} shards={manifest['n_shards']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
