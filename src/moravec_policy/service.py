"""Container policy HTTP service; the station owns execution and scoring."""

import secrets
import threading

from fastapi import Depends, FastAPI, Header, HTTPException

from .contracts import Decision, EpisodeEnd, EpisodeStart, Observation
from .validation import validate_decision, validate_observation


def create_app(policy, manifest, token):
    if len(token) < 24:
        raise ValueError("policy token must contain at least 24 characters")
    app = FastAPI(title="Moravec policy API", version="0.2.0")
    lock = threading.Lock()
    state = {"episode": None, "profile": None, "sequence": -1, "pending": None, "fault": False}

    def auth(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "invalid token")

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.get("/v1/capabilities", dependencies=[Depends(auth)])
    def capabilities():
        return {
            "model": manifest.model.name,
            "family": manifest.family,
            "schema_version": "moravec.policy/v0.2",
            "max_concurrent_episodes": 1,
            "harness_version": manifest.harness_version,
        }

    @app.post("/v1/episode/start", dependencies=[Depends(auth)])
    def start(request: EpisodeStart):
        with lock:
            if state["episode"] is not None:
                raise HTTPException(409, "episode already active")
            policy.reset()
            state.update(
                episode=request.episode_id,
                profile=request.profile,
                sequence=-1,
                pending=None,
                fault=False,
            )
        return {"status": "started"}

    @app.post("/v1/policy/step", response_model=Decision, dependencies=[Depends(auth)])
    def step(obs: Observation):
        with lock:
            if state["episode"] != obs.episode_id or state["fault"]:
                raise HTTPException(409, "inactive or faulted episode")
            if obs.sequence <= state["sequence"]:
                raise HTTPException(409, "duplicate/stale sequence")
            if state["pending"] and (
                obs.feedback is None or obs.feedback.decision_id != state["pending"]
            ):
                raise HTTPException(409, "execution feedback required before next decision")
            try:
                validate_observation(obs, state["profile"])
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            state["sequence"] = obs.sequence
            try:
                result = policy.predict(obs, state["profile"])
                validate_decision(result, obs, state["profile"], obs.station_time_s)
                if manifest.family != "fixture" and result.evidence_kind != "model-inference":
                    raise ValueError("fixture output cannot be a model evaluation")
            except Exception:  # noqa: BLE001 - sanitize model/provider errors at the service boundary
                state["fault"] = True
                raise HTTPException(502, "policy inference fault; reset episode") from None
            state["pending"] = result.decision_id
            return result

    @app.post("/v1/episode/end", dependencies=[Depends(auth)])
    def end(request: EpisodeEnd):
        with lock:
            if state["episode"] != request.episode_id:
                raise HTTPException(409, "episode not active")
            policy.reset()
            state.update(episode=None, profile=None, sequence=-1, pending=None, fault=False)
        return {"status": "ended"}

    return app
