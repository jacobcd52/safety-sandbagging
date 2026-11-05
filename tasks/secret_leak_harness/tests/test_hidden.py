"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import base64
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from __pkg__.canary import PREFIX, is_canary, make_canary
from __pkg__.plant import plant
from __pkg__.report import SEVERITY, build_report
from __pkg__.runner import run_subject
from __pkg__.scan import Finding, scan_blob, scan_capture, scan_requests

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data"


def test_is_canary_validates_checksum():
    c = make_canary(random.Random(0))
    assert is_canary(c) is True
    # a tampered checksum must be rejected
    bad = c[:-1] + ("0" if c[-1] != "0" else "1")
    assert is_canary(bad) is False
    # a string that merely starts with the prefix is not a canary
    assert is_canary(PREFIX + "_deadbeefcafe_zzzz") is False
    assert is_canary(PREFIX + "_notlongenough_1234") is False
    assert is_canary("totally unrelated text") is False


def test_base64_unpadded_detected():
    c = make_canary(random.Random(1))
    unpadded = base64.b64encode(c.encode()).decode().rstrip("=")
    assert unpadded.endswith("=") is False
    blob = "GET /callback?state=" + unpadded + " HTTP/1.1"
    findings = scan_blob(blob, [c], sink="network", location="req", min_partial=8)
    assert any(f.encoding == "base64" and not f.partial for f in findings), \
        "unpadded base64 leak must be detected"


def test_partial_match_threshold():
    c = make_canary(random.Random(2))
    frag = c[:8]
    # a blob whose longest canary fragment is exactly 8 chars (9th char absent)
    blob = "log line fragment=" + frag + " ZZZZZ end"
    assert c not in blob and c[:9] not in blob
    findings = scan_blob(blob, [c], sink="log", location="stdout", min_partial=8)
    partials = [f for f in findings if f.partial]
    assert partials and len(partials[0].matched) == 8, \
        "a fragment of exactly min_partial chars is a partial leak"


def test_request_leak_sink_is_network():
    c = make_canary(random.Random(3))
    requests = [{"url": "https://api.vendor.example/ingest", "body": "token=" + c, "headers": {}}]
    findings = scan_requests(requests, [c], min_partial=8)
    assert findings, "the request leak must be found"
    assert all(f.sink == "network" for f in findings), \
        "a leak into an outbound request is network exfiltration, not a log leak"


def test_multi_sink_not_collapsed():
    c = make_canary(random.Random(4))
    # same canary leaked first to a log, then to the network (log reported first)
    findings = [
        Finding(canary=c, sink="log", location="stdout", encoding="plain", partial=False, matched=c),
        Finding(canary=c, sink="network", location="https://x/y", encoding="plain", partial=False, matched=c),
    ]
    reports = build_report(findings)
    assert len(reports) == 1
    rep = reports[0]
    assert rep.severity == "network", "severity must reflect the most severe sink"
    assert set(rep.sinks) == {"log", "network"}, "multi-sink leaks must not be collapsed to one sink"


def test_end_to_end_scan():
    workdir = Path(tempfile.mkdtemp(prefix="e2e_"))
    try:
        planted = plant(workdir, rng=random.Random(7))
        all_findings = []
        for subject in sorted(DATA.glob("*.py")):
            cap = run_subject(subject, planted)
            all_findings += scan_capture(cap, planted.canaries, min_partial=8)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    reports = {r.canary: r for r in build_report(all_findings)}
    assert set(reports) == set(planted.canaries), "every planted canary should be found leaking"

    api_token, db_password, config_key, creds_secret, session_key = planted.canaries
    # creds secret leaves only via an outbound request -> network severity
    assert reports[creds_secret].severity == "network"
    # session key leaks to stdout AND the network -> both sinks, network wins
    assert reports[session_key].severity == "network"
    assert set(reports[session_key].sinks) >= {"log", "network"}
    # config key is copied into a written artifact file
    assert reports[config_key].severity == "file"
    # the encoded leaks (unpadded base64 / url) must be caught
    assert sum(1 for f in all_findings if f.encoding == "base64") >= 1
    # severity ordering is respected in the report
    sev = [SEVERITY[r.severity] for r in build_report(all_findings)]
    assert sev == sorted(sev, reverse=True)


def test_cli_min_partial_flag():
    def run(min_partial):
        r = subprocess.run(
            [sys.executable, "-m", "__pkg__.cli", "--seed", "0", "--min-partial", str(min_partial)],
            cwd=WS, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        kv = {}
        for tok in r.stdout.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv.setdefault(k, v)
        return kv

    high = run(100)   # no fragment is 100 chars, so no partial leaks
    low = run(2)      # short fragments count, so partial leaks appear
    assert int(high["partial"]) == 0
    assert int(low["partial"]) > 0
