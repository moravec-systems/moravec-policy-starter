from urllib.parse import urlparse

import httpx

from .contracts import Decision


class RemotePolicy:
    def __init__(self, url, token, timeout_s=60):
        parts = urlparse(url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("credentials/query/fragment prohibited in endpoint URL")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("non-loopback endpoints require HTTPS")
        self.http = httpx.Client(
            base_url=url,
            headers={"Authorization": "Bearer " + token},
            timeout=timeout_s,
            follow_redirects=False,
        )

    def start(self, episode_id, profile):
        self.http.post(
            "/v1/episode/start", json={"episode_id": episode_id, "profile": profile.model_dump()}
        ).raise_for_status()

    def predict(self, obs):
        response = self.http.post("/v1/policy/step", json=obs.model_dump())
        response.raise_for_status()
        decision = Decision.model_validate(response.json())
        if (decision.episode_id, decision.observation_sequence) != (obs.episode_id, obs.sequence):
            raise ValueError("mismatched remote response")
        return decision

    def end(self, episode_id):
        self.http.post("/v1/episode/end", json={"episode_id": episode_id}).raise_for_status()

    def close(self):
        self.http.close()
