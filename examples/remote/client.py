"""Run against the local demo or your authenticated remote policy service."""

import json
import os
from pathlib import Path

from moravec_policy.client import RemotePolicy
from moravec_policy.contracts import Observation

fixture = Path(__file__).resolve().parents[2] / "fixtures" / "observation.json"
obs = Observation.model_validate_json(fixture.read_text())
client = RemotePolicy(
    os.getenv("MORAVEC_POLICY_URL", "http://127.0.0.1:8000"), os.environ["MORAVEC_POLICY_TOKEN"]
)
try:
    client.start(obs.episode_id)
    try:
        print(json.dumps(client.predict(obs).model_dump(), indent=2))
    finally:
        client.end(obs.episode_id)
finally:
    client.close()
