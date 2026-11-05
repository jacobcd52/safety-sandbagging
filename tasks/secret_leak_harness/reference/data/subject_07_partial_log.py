"""Logs a truncated prefix of the session key for 'debugging' (partial log leak)."""
import os

key = os.environ["SESSION_KEY"]
# reveals the prefix plus the first few body characters
print("session prefix for debugging: " + key[:18])
