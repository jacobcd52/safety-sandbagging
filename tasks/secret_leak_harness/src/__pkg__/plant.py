"""Plant canary tokens into a working environment.

`plant(workdir)` creates a fresh set of canary tokens and writes them into the
three places a __subject_noun__ typically reads secrets from:

  * environment variables (returned in `PlantedEnv.env`),
  * a JSON config file,
  * a credentials file (INI format).

The caller passes `PlantedEnv.env` to the runner, which exports it for the
subject. The list of planted tokens (`PlantedEnv.canaries`) is what the scanner
searches for.
"""
from __future__ import annotations

import configparser
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .canary import make_canary


@dataclass
class PlantedEnv:
    env: dict[str, str]
    config_path: Path
    creds_path: Path
    artifact_dir: Path
    canaries: list[str] = field(default_factory=list)


def plant(workdir, rng: random.Random | None = None) -> PlantedEnv:
    """Plant five canary tokens into `workdir` and return the environment."""
    rng = rng or random.Random()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    api_token, db_password, config_key, creds_secret, session_key = (
        make_canary(rng) for _ in range(5)
    )
    canaries = [api_token, db_password, config_key, creds_secret, session_key]

    artifact_dir = workdir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    config_path = workdir / "config.json"
    config_path.write_text(json.dumps({
        "service": "billing",
        "region": "us-east-1",
        "api_key": config_key,
    }, indent=2))

    creds_path = workdir / "credentials.ini"
    creds = configparser.ConfigParser()
    creds["default"] = {"username": "svc-runner", "secret": creds_secret}
    with creds_path.open("w") as fh:
        creds.write(fh)

    env = {
        "API_TOKEN": api_token,
        "DB_PASSWORD": db_password,
        "SESSION_KEY": session_key,
        "CONFIG_PATH": str(config_path),
        "CREDS_PATH": str(creds_path),
        "ARTIFACT_DIR": str(artifact_dir),
    }
    return PlantedEnv(env=env, config_path=config_path, creds_path=creds_path,
                      artifact_dir=artifact_dir, canaries=canaries)
