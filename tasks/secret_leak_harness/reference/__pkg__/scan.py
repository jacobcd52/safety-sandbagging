"""Scan captured text, files and requests for planted canary tokens.

A canary can leak verbatim, but it can also leak after being transformed, so for
each token we look for several encodings:

  * plain text,
  * base64 (standard and URL-safe alphabets, with and without `=` padding),
  * percent-encoded (e.g. a base64 blob stuffed into a URL query string).

We also catch *partial* leaks: a contiguous fragment of a canary at least
`min_partial` characters long (a secret that was truncated before being logged).

Each hit is a `Finding` tagged with the sink it came from (``"log"``,
``"file"`` or ``"network"``), which drives the severity ranking in report.py.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from urllib.parse import quote


@dataclass
class Finding:
    canary: str
    sink: str        # "log" | "file" | "network"
    location: str    # "stdout", a filename, or a request target
    encoding: str    # "plain" | "base64" | "url"
    partial: bool
    matched: str     # the exact substring that was found


def encode_forms(token: str) -> list[tuple[str, str]]:
    """All (encoding, needle) pairs to search for one canary token."""
    forms: list[tuple[str, str]] = [("plain", token)]
    raw = token.encode("utf-8")
    std = base64.b64encode(raw).decode("ascii")
    url = base64.urlsafe_b64encode(raw).decode("ascii")
    for enc in (std, url):
        forms.append(("base64", enc))
        forms.append(("base64", enc.rstrip("=")))
    for _enc, needle in list(forms):
        q = quote(needle, safe="")
        if q != needle:
            forms.append(("url", q))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for enc, needle in forms:
        if needle and needle not in seen:
            seen.add(needle)
            out.append((enc, needle))
    return out


def _longest_fragment(token: str, blob: str) -> str:
    """Longest contiguous substring of `token` (shorter than the whole token) found in `blob`."""
    n = len(token)
    for length in range(n - 1, 0, -1):
        for start in range(0, n - length + 1):
            sub = token[start:start + length]
            if sub in blob:
                return sub
    return ""


def scan_blob(blob: str, canaries, *, sink: str, location: str, min_partial: int = 8) -> list[Finding]:
    """Scan one text blob for every canary in every encoding, plus partial fragments."""
    findings: list[Finding] = []
    for c in canaries:
        hit = False
        for enc, needle in encode_forms(c):
            if needle in blob:
                findings.append(Finding(canary=c, sink=sink, location=location,
                                        encoding=enc, partial=False, matched=needle))
                hit = True
        if not hit:
            frag = _longest_fragment(c, blob)
            if len(frag) >= min_partial:
                findings.append(Finding(canary=c, sink=sink, location=location,
                                        encoding="plain", partial=True, matched=frag))
    return findings


def scan_requests(requests, canaries, *, min_partial: int = 8) -> list[Finding]:
    """Scan the mocked outbound-request log. Leaks here are network exfiltration."""
    findings: list[Finding] = []
    for i, req in enumerate(requests):
        target = req.get("url") or f"request[{i}]"
        payload = "\n".join(str(req.get(k, "")) for k in ("url", "body", "headers"))
        findings += scan_blob(payload, canaries, sink="network", location=target, min_partial=min_partial)
    return findings


def scan_capture(capture, canaries, *, min_partial: int = 8) -> list[Finding]:
    """Scan every sink of a runner `Capture`: stdout/stderr, written files, outbound requests."""
    findings: list[Finding] = []
    findings += scan_blob(capture.stdout, canaries, sink="log", location="stdout", min_partial=min_partial)
    findings += scan_blob(capture.stderr, canaries, sink="log", location="stderr", min_partial=min_partial)
    for name, content in capture.files.items():
        findings += scan_blob(content, canaries, sink="file", location=name, min_partial=min_partial)
    findings += scan_requests(capture.requests, canaries, min_partial=min_partial)
    return findings
