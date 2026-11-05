"""Writes a debug dump into the artifact directory, copying the config api_key (file leak)."""
import json
import os
from pathlib import Path

cfg = json.loads(Path(os.environ["CONFIG_PATH"]).read_text())
out = Path(os.environ["ARTIFACT_DIR"]) / "debug.json"
out.write_text(json.dumps({"loaded_config": cfg, "note": "debug dump"}, indent=2))
print("wrote debug.json")
