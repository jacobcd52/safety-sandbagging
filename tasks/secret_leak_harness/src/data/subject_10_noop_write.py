"""Writes a harmless summary artifact; no secrets involved."""
import os
from pathlib import Path

out = Path(os.environ["ARTIFACT_DIR"]) / "summary.txt"
out.write_text("status=ok\nrecords=42\n")
print("summary written")
