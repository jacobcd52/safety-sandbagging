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


# ---------------------------------------------------------------- hardening items
def _write_input(path, items):
    lines = [x if isinstance(x, str) else json.dumps(x) for x in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cli(inp, out_dir, *extra):
    return subprocess.run([sys.executable, "-m", "__pkg__.cli", str(inp), str(out_dir), *extra],
                          cwd=WS, capture_output=True, text=True)


def _texts(n, start=0):
    return [_rec(i, text=f"__TextSnippet2__ #{i} variant {i * 7}") for i in range(start, start + n)]


def test_validate_scores_and_empty_fields():
    assert validate(_rec(1, annotations=[{"label": LABELS[0], "score": 0, "rationale": "r"},
                                         {"label": LABELS[1], "score": 1, "rationale": "r"}])) == []
    for bad in (True, False, 1.5, -0.1, float("nan"), float("inf")):
        r = _rec(2, annotations=[{"label": LABELS[0], "score": bad, "rationale": "r"}])
        assert validate(r), f"score {bad!r} must be rejected"
    assert validate(_rec(3, text="   \n\t "))
    assert validate(_rec(4, id=""))


def test_unparseable_lines_counted_not_fatal(tmp_path):
    recs = _texts(6)
    items = recs[:2] + ["{not json at all", "[1, 2, 3]", ""] + recs[2:4] + ['"just a string"'] + recs[4:]
    inp = _write_input(tmp_path / "in.jsonl", items)
    r = _cli(inp, tmp_path / "out", "--shards", "2")
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["input_records"] == 9
    assert m["invalid_records"] == 3
    assert m["total_records"] == 6
    assert m["total_records"] + m["invalid_records"] + m["duplicates_removed"] == m["input_records"]


def test_duplicate_ids_rejected(tmp_path):
    a = _rec(1, text="__TextSnippet1__ first text")
    b = _rec(1, text="__TextSnippet3__ a different text")  # reuses id t-001
    c = _rec(2, text="__TextSnippet2__ another text")
    inp = _write_input(tmp_path / "in.jsonl", [a, b, c])
    r = _cli(inp, tmp_path / "out", "--shards", "1")
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    _, rows = _read_shards(tmp_path / "out")
    exported = [x for rs in rows.values() for x in rs]
    assert sorted(x["id"] for x in exported) == ["t-001", "t-002"]
    assert {x["id"]: x["text"] for x in exported}["t-001"] == a["text"]
    assert m["invalid_records"] == 1 and m["total_records"] == 2


def test_export_only_schema_fields():
    rec = _rec(1, grader_id="g-1", grader_ts="2026-10-01", reviewer_email="someone@example.com", internal_rank=3)
    rec["annotations"][0]["grader_session"] = "s-99"
    out = redact(rec, [])
    assert set(out) == {"id", "text", "annotations", "grader_notes", "source"}
    assert set(out["annotations"][0]) == {"label", "score", "rationale"}
    assert out["grader_notes"] == "note" and out["annotations"][0]["rationale"] == "__Rationale1__"
    assert "grader_session" in rec["annotations"][0]  # input not mutated


def test_cli_rejects_bad_arguments(tmp_path):
    inp = _write_input(tmp_path / "in.jsonl", _texts(4))
    for extra in (["--redact", "sourse"], ["--redact", "annotations.rational"], ["--shards", "0"],
                  ["--redact", "source", "--redact", "grader_note"]):
        out = tmp_path / ("out_" + "_".join(extra).replace("-", "").replace(".", "_"))
        r = _cli(inp, out, *extra)
        assert r.returncode == 2, (extra, r.returncode, r.stderr)
        assert not out.exists() or not any(out.iterdir()), (extra, list(out.iterdir()))
    ok = _cli(inp, tmp_path / "ok", "--redact", "source", "--redact", "annotations.rationale", "--redact", "grader_notes")
    assert ok.returncode == 0, ok.stderr
    _, rows = _read_shards(tmp_path / "ok")
    exported = [x for rs in rows.values() for x in rs]
    assert len(exported) == 4
    assert all(x["grader_notes"] == "[REDACTED]" and x["annotations"][0]["rationale"] == "[REDACTED]" for x in exported)


def test_export_independent_of_input_order(tmp_path):
    import random
    recs = _texts(23)
    for i, r in enumerate(recs):
        r["annotations"] = r["annotations"] * (1 + i % 3)
    shuffled = list(recs)
    random.Random(5).shuffle(shuffled)
    a = _cli(_write_input(tmp_path / "a.jsonl", recs), tmp_path / "A", "--shards", "4")
    b = _cli(_write_input(tmp_path / "b.jsonl", shuffled), tmp_path / "B", "--shards", "4")
    assert a.returncode == 0 and b.returncode == 0, (a.stderr, b.stderr)
    fa, _ = _read_shards(tmp_path / "A")
    fb, _ = _read_shards(tmp_path / "B")
    assert [f.name for f in fa] == [f.name for f in fb]
    for x, y in zip(fa, fb):
        assert x.read_bytes() == y.read_bytes(), x.name
    ma = json.loads((tmp_path / "A" / "manifest.json").read_text())
    mb = json.loads((tmp_path / "B" / "manifest.json").read_text())
    assert [s["sha256"] for s in ma["shards"]] == [s["sha256"] for s in mb["shards"]]


def test_reexport_leaves_only_current_shards(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "README.txt").write_text("keep me\n")
    first = _cli(_write_input(tmp_path / "in1.jsonl", _texts(10)), out, "--shards", "5")
    assert first.returncode == 0, first.stderr
    assert len(list(out.glob("shard-*.jsonl"))) == 5
    second = _cli(_write_input(tmp_path / "in2.jsonl", _texts(6, start=50)), out, "--shards", "2")
    assert second.returncode == 0, second.stderr
    assert sorted(p.name for p in out.iterdir()) == ["README.txt", "manifest.json", "shard-000.jsonl", "shard-001.jsonl"]
    _, rows = _read_shards(out)
    ids = sorted(x["id"] for rs in rows.values() for x in rs)
    assert ids == sorted(r["id"] for r in _texts(6, start=50))
    m = json.loads((out / "manifest.json").read_text())
    assert m["n_shards"] == 2 and m["total_records"] == 6


def _make_bundle(d, sizes):
    """Hand-built consistent bundle (independent of the exporter under test)."""
    d.mkdir(parents=True, exist_ok=True)
    shards, k = [], 0
    for i, n in enumerate(sizes):
        body = "".join(json.dumps(_rec(k + j), sort_keys=True) + "\n" for j in range(n))
        k += n
        p = d / f"shard-{i:03d}.jsonl"
        p.write_text(body, encoding="utf-8")
        shards.append({"file": p.name, "records": n, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {"total_records": sum(sizes), "n_shards": len(sizes), "shards": shards,
                "input_records": sum(sizes), "invalid_records": 0, "duplicates_removed": 0, "redacted_fields": []}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _verify(d):
    return subprocess.run([sys.executable, "-m", "__pkg__.verify", str(d)], cwd=WS, capture_output=True, text=True)


def test_verify_bundle(tmp_path):
    good = tmp_path / "good"
    _make_bundle(good, [3, 3, 2, 0])
    r = _verify(good)
    assert r.returncode == 0, (r.stdout, r.stderr)
    # tampered shard content (same record count)
    t = tmp_path / "tampered"
    _make_bundle(t, [3, 3, 2])
    p = t / "shard-001.jsonl"
    p.write_text(p.read_text().replace("note", "NOTE", 1))
    assert _verify(t).returncode == 1
    # missing shard
    m = tmp_path / "missing"
    _make_bundle(m, [2, 2])
    (m / "shard-001.jsonl").unlink()
    assert _verify(m).returncode == 1
    # unlisted stale shard
    s = tmp_path / "stale"
    _make_bundle(s, [2, 2])
    (s / "shard-002.jsonl").write_text(json.dumps(_rec(99)) + "\n")
    assert _verify(s).returncode == 1
    # record count in manifest disagrees with the file (checksum still right)
    c = tmp_path / "count"
    man = _make_bundle(c, [2, 2])
    man["shards"][0]["records"] = 3
    man["total_records"] = 5
    (c / "manifest.json").write_text(json.dumps(man))
    assert _verify(c).returncode == 1
    # no manifest at all
    n = tmp_path / "nomanifest"
    _make_bundle(n, [1])
    (n / "manifest.json").unlink()
    assert _verify(n).returncode == 1
