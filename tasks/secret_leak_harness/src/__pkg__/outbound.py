"""Mocked outbound-request log.

A __subject_noun__ never makes a real network call during a scan; instead it
calls `send(...)`, which appends a record to the module-level `REQUESTS` list.
`runner.run_subject` clears this list before each run and snapshots it after,
so the scanner can inspect everything the subject tried to send out.

Each record is a dict with the fields `url`, `body` and `headers`.
"""
from __future__ import annotations

REQUESTS: list[dict] = []


def send(url: str, body=None, headers=None) -> dict:
    """Record an outbound request and return a fake 200 response."""
    REQUESTS.append({
        "url": url,
        "body": "" if body is None else str(body),
        "headers": dict(headers) if headers else {},
    })
    return {"status": 200, "url": url}
