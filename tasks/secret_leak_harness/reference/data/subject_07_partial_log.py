"""Logs only a truncated prefix of the session key for 'debugging' (partial log leak)."""
import os

key = os.environ["SESSION_KEY"]
print("session prefix for debugging: " + key[:10])
