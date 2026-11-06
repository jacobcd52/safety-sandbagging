"""Prints a status banner that includes the API token (stdout log leak)."""
import os

token = os.environ["API_TOKEN"]
print("starting worker with API_TOKEN=" + token)
print("connected to upstream")
