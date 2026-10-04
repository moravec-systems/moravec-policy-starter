"""Minimal remote policy client; no automatic retries of stateful requests."""

from urllib.parse import urlparse

import httpx

from .contracts import ActionChunk, Observation


class RemotePolicy:
    def __init__(self, url: str, token: str, timeout_s: float = 60):
        parts = urlparse(url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("endpoint URL must not contain credentials, queries, or fragments")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("remote policy requires HTTPS (HTTP permitted only on loopback)")
        self.http = httpx.Client(
            base_url=url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
            follow_redirects=False,
        )

    def start(self, episode_id: str):
        self.http.post("/v1/episode/start", json={"episode_id": episode_id}).raise_for_status()

    def predict(self, observation: Observation) -> ActionChunk:
        response = self.http.post("/v1/predict", json={"observation": observation.model_dump()})
        response.raise_for_status()
        result = ActionChunk.model_validate(response.json())
        if (result.episode_id, result.observation_sequence) != (
            observation.episode_id,
            observation.sequence,
        ):
            raise ValueError("remote response belongs to a different observation")
        return result

    def end(self, episode_id: str):
        self.http.post("/v1/episode/end", json={"episode_id": episode_id}).raise_for_status()

    def close(self):
        self.http.close()
