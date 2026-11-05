"""On-disk JSON response cache.

Entries are keyed by (model_name, prompt) so that re-running the same model is
free and switching models never serves another model's responses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def _key(model_name: str, prompt: str) -> str:
        return hashlib.sha256(f"{model_name}\x00{prompt}".encode("utf-8")).hexdigest()

    def get(self, model_name: str, prompt: str) -> str | None:
        return self._data.get(self._key(model_name, prompt))

    def put(self, model_name: str, prompt: str, response: str) -> None:
        self._data[self._key(model_name, prompt)] = response

    def __len__(self) -> int:
        return len(self._data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=0, sort_keys=True))
