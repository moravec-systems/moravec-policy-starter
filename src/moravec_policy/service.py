"""Single-episode reference service. It returns actions; it never actuates a robot."""

from __future__ import annotations

import secrets
import threading

from fastapi import Depends, FastAPI, Header, HTTPException

from .adapters import Policy
from .contracts import ActionChunk, EpisodeStart, PolicyManifest, PredictRequest


class Session:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.episode_id = None
        self.sequence = -1
        self.lock = threading.Lock()
        self.poisoned = False

    def start(self, episode_id: str):
        if self.episode_id is not None:
            raise HTTPException(409, "end the active episode before starting another")
        self.policy.reset()
        self.episode_id = episode_id
        self.sequence = -1
        self.poisoned = False

    def predict(self, request: PredictRequest):
        obs = request.observation
        if self.poisoned:
            raise HTTPException(409, "inference fault: end and reset this session")
        if obs.episode_id != self.episode_id:
            raise HTTPException(409, "episode is not active")
        if obs.sequence <= self.sequence:
            raise HTTPException(409, "duplicate or stale observation; do not replay actions")
        self.sequence = obs.sequence
        try:
            chunk = self.policy.predict(obs)
            if chunk.episode_id != obs.episode_id or chunk.observation_sequence != obs.sequence:
                raise ValueError("adapter returned mismatched response identity")
            return chunk
        except Exception:  # noqa: BLE001 - untrusted model boundary; sanitize provider errors
            self.poisoned = True
            # Model exceptions can contain provider credentials; do not reflect them to clients.
            raise HTTPException(502, "policy inference failed; episode requires reset") from None

    def end(self, episode_id: str):
        if episode_id != self.episode_id:
            raise HTTPException(409, "episode is not active")
        self.policy.reset()
        self.episode_id = None
        self.sequence = -1
        self.poisoned = False


def create_app(policy: Policy, manifest: PolicyManifest, token: str) -> FastAPI:
    if len(token) < 24:
        raise ValueError("set a policy token with at least 24 characters")
    app = FastAPI(title="Moravec policy starter", version="0.1.0")
    session = Session(policy)

    def authenticate(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(401, "invalid policy token")

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.get("/v1/capabilities", dependencies=[Depends(authenticate)])
    def capabilities():
        return {
            "policy_id": manifest.id,
            "action_type": manifest.action_type,
            "embodiment": manifest.embodiment,
            "batching": False,
            "max_concurrent_episodes": 1,
            "execution": manifest.execution,
            "validation_status": manifest.validation_status,
        }

    @app.post("/v1/episode/start", dependencies=[Depends(authenticate)])
    def start(request: EpisodeStart):
        with session.lock:
            session.start(request.episode_id)
        return {"status": "started", "episode_id": request.episode_id}

    @app.post("/v1/predict", response_model=ActionChunk, dependencies=[Depends(authenticate)])
    def predict(request: PredictRequest):
        with session.lock:
            chunk = session.predict(request)
            if chunk.action_type != manifest.action_type:
                session.poisoned = True
                raise HTTPException(502, "adapter action space differs from manifest")
            if manifest.family != "test_fixture" and chunk.evidence_kind == "mock":
                session.poisoned = True
                raise HTTPException(502, "mock result prohibited for this model")
            return chunk

    @app.post("/v1/episode/end", dependencies=[Depends(authenticate)])
    def end(request: EpisodeStart):
        with session.lock:
            session.end(request.episode_id)
        return {"status": "ended", "episode_id": request.episode_id}

    return app
