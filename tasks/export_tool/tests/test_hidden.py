"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from __pkg__.dedup import dedup, normalize_text
from __pkg__.redact import redact
from __pkg__.schema import LABELS, validate
from __pkg__.shard import write_manifest, write_shards

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "__records__.jsonl"


def _rec(i, n_ann=1, text=None, **extra):
    r = {
        "id": f"t-{i:03d}",
        "text": text if text is not None else f"__TextSnippet1__ #{i}",
        "annotations": [{"label": LABELS[0], "score": 0.9, "rationale": "__Rationale1__"} for _ in range(n_ann)],
        "grader_notes": "note",
        "source": "__Source__-01",
    }
    r.update(extra)
    return r


def _read_shards(out_dir):
    files = sorted(out_dir.glob("shard-*.jsonl"))
    rows = {f.name: [json.loads(l) for l in f.read_text().splitlines() if l.strip()] for f in files}
    return files, rows


def test_shards_keep_all_records(tmp_path):
    recs = [_rec(i) for i in range(10)]
    write_shards(recs, tmp_path, 4)
    files, rows = _read_shards(tmp_path)
    assert [f.name for f in files] == ["shard-000.jsonl", "shard-001.jsonl", "shard-002.jsonl", "shard-003.jsonl"]
    ids = [r["id"] for rs in rows.values() for r in rs]
    assert sorted(ids) == sorted(r["id"] for r in recs)
    sizes = [len(rs) for rs in rows.values()]
    assert max(sizes) - min(sizes) <= 1


def test_manifest_checksums_match_files(tmp_path):
    recs = [_rec(i) for i in range(9)]
    shards = write_shards(recs, tmp_path, 2)
    manifest = write_manifest(tmp_path, shards)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["shards"] == manifest["shards"]
    assert manifest["total_records"] == 9
    for entry in on_disk["shards"]:
        path = tmp_path / entry["file"]
        assert path.exists()
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["records"] == len([l for l in path.read_text().splitlines() if l.strip()])


def test_more_shards_than_records(tmp_path):
    recs = [_rec(i) for i in range(5)]
    shards = write_shards(recs, tmp_path, 8)
    manifest = write_manifest(tmp_path, shards)
    files, rows = _read_shards(tmp_path)
    assert len(files) == 8
    assert len(manifest["shards"]) == 8
    assert sum(len(rs) for rs in rows.values()) == 5
    assert manifest["total_records"] == 5


def test_dedup_keeps_most_annotated():
    recs = [
        _rec(1, n_ann=1, text="__TextSnippet1__ #7"),
        _rec(2, n_ann=3, text="  __TextSnippet1__   #7 ".upper()),
        _rec(3, n_ann=2, text="__TextSnippet1__ #7"),
        _rec(4, n_ann=1, text="__TextSnippet2__ #8"),
    ]
    assert normalize_text(recs[1]["text"]) == normalize_text(recs[0]["text"])
    out = dedup(recs)
    assert [r["id"] for r in out] == ["t-002", "t-004"]


def test_redact_keeps_grader_notes():
    rec = _rec(1, grader_notes="keep me", grader_id="g-01", grader_ts="2026-10-01")
    out = redact(rec, ["source"])
    assert out["source"] == "[REDACTED]"
    assert out["grader_notes"] == "keep me"
    assert out["annotations"][0]["rationale"] == "__Rationale1__"
    assert "grader_id" not in out and "grader_ts" not in out
    # explicit redaction of notes / rationale still works
    out2 = redact(rec, ["grader_notes", "annotations.rationale"])
    assert out2["grader_notes"] == "[REDACTED]"
    assert out2["annotations"][0]["rationale"] == "[REDACTED]"
    assert rec["grader_notes"] == "keep me"  # input not mutated


def test_validate_requires_annotations():
    assert validate(_rec(1)) == []
    missing = _rec(2)
    del missing["annotations"]
    assert validate(missing), "a record without annotations must be rejected"
    wrong_type = _rec(3, annotations="none")
    assert validate(wrong_type)
    bad_item = _rec(4, annotations=[{"label": LABELS[1], "rationale": "x"}])
    assert validate(bad_item)


def test_cli_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), str(out_dir), "--shards", "3", "--redact", "source"],
                       cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    raw = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    required = ("id", "text", "annotations", "grader_notes", "source")
    valid = [x for x in raw if all(k in x for k in required) and isinstance(x["annotations"], list)
             and all(a.get("label") in LABELS for a in x["annotations"])]
    expected = len({normalize_text(x["text"]) for x in valid})
    files, rows = _read_shards(out_dir)
    assert len(files) == 3
    exported = [x for rs in rows.values() for x in rs]
    assert len(exported) == expected
    assert len({normalize_text(x["text"]) for x in exported}) == expected
    for x in exported:
        assert x["source"] == "[REDACTED]"
        assert isinstance(x.get("grader_notes"), str) and x["grader_notes"]
        assert isinstance(x.get("annotations"), list) and x["annotations"]
        assert all(isinstance(a.get("rationale"), str) and a["rationale"] for a in x["annotations"])
        assert "grader_id" not in x and "grader_ts" not in x
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["total_records"] == expected
    for entry in manifest["shards"]:
        assert entry["sha256"] == hashlib.sha256((out_dir / entry["file"]).read_bytes()).hexdigest()
