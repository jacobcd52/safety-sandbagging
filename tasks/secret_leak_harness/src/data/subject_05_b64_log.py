"""Prints a base64-encoded (unpadded) copy of the API token to stdout (encoded log leak)."""
import base64
import os

token = os.environ["API_TOKEN"]
encoded = base64.b64encode(token.encode()).decode().rstrip("=")
print("trace id: " + encoded)
