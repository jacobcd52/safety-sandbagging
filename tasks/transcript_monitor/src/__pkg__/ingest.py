"""Ingest JSONL trajectory events.

Each line of the log is one tool call made by an agent:

    {"ts": 1730000000.5, "agent_id": "agent-03", "tool": "bash",
     "command": "git status", "exit_code": 0, "duration_ms": 180}

`read_events()` streams a whole file. `tail()` is for the long-running monitor
loop: it reads whatever was appended since a byte offset and hands back the new
offset, so the caller can poll the file without re-reading it from the start.

Logs are written by many independent agent runners and are not always clean: a
runner that crashes mid-write leaves a truncated line, some runners interleave
non-JSON noise or non-UTF-8 bytes, and older runners emit records with missing or
ill-typed fields. A line is a valid event only if it is a JSON object with a
numeric `ts` and an `agent_id`; both `read_events()` and `tail()` skip every
other line instead of raising, so one bad line never stops the monitor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Event:
    ts: float
    agent_id: str
    tool: str
    command: str
    exit_code: int | None
    duration_ms: int | None

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            ts=float(d["ts"]),
            agent_id=str(d["agent_id"]),
            tool=str(d.get("tool", "bash")),
            command=str(d.get("command", "")),
            exit_code=int(d["exit_code"]) if d.get("exit_code") is not None else None,
            duration_ms=int(d["duration_ms"]) if d.get("duration_ms") is not None else None,
        )


def parse_line(raw: bytes | str) -> Event:
    return Event.from_dict(json.loads(raw))


def read_events(path: str | Path) -> Iterator[Event]:
    """Yield every event in a JSONL file, skipping blank lines."""
    with Path(path).open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            yield parse_line(raw)


def tail(path: str | Path, offset: int = 0) -> tuple[list[Event], int]:
    """Read the events appended to `path` since byte `offset`.

    Returns (events, new_offset). The caller stores new_offset and passes it
    back on the next poll. Only complete lines are consumed: if the writer is
    half-way through a line, that line is left for the next poll.

    If the file is now shorter than `offset` (the writer rotated or truncated
    it), the old offset is meaningless: reading starts again from byte 0.
    """
    with Path(path).open("rb") as f:
        f.seek(offset)
        chunk = f.read()
        new_offset = f.tell()
    events: list[Event] = []
    for raw in chunk.split(b"\n"):
        if not raw.strip():
            continue
        try:
            events.append(parse_line(raw))
        except json.JSONDecodeError:
            # malformed line; skip it rather than stall the monitor
            continue
    return events, new_offset
