"""One authenticated policy session per container, over WebSocket/MessagePack."""

import asyncio
import secrets
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .contracts import Decision, EpisodeStart
from .validation import validate_channels, validate_decision, validate_observation
from .wire import observation_from_wire, pack, unpack

PROTOCOL = "moravec.policy/v0.4"


def create_app(policy, manifest, token):
    if len(token) < 24:
        raise ValueError("policy token must contain at least 24 characters")
    app = FastAPI(
        title="Moravec policy service",
        version="0.4.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    owner = asyncio.Lock()

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.websocket("/policy")
    async def session(ws: WebSocket):
        if not secrets.compare_digest(ws.headers.get("authorization", ""), "Bearer " + token):
            await ws.close(code=1008)
            return
        if owner.locked():
            await ws.close(code=1013)
            return
        async with owner:
            await ws.accept()
            episode = profile = obs = pending = None
            sequence = -1
            consumed = False
            fault = False
            # No replay or automatic retries: a dropped connection aborts its episode.
            request_ids = set()
            try:
                while True:
                    request_id = None
                    try:
                        request = unpack(await ws.receive_bytes())
                        if (
                            set(request) != {"protocol", "request_id", "method", "payload"}
                            or request["protocol"] != PROTOCOL
                        ):
                            raise ValueError("invalid protocol envelope")
                        request_id = request["request_id"]
                        if (
                            not isinstance(request_id, str)
                            or not 1 <= len(request_id) <= 128
                            or request_id in request_ids
                        ):
                            raise ValueError("invalid or duplicate request ID")
                        if len(request_ids) >= 10000:
                            raise ValueError("session request limit reached")
                        request_ids.add(request_id)
                        method, payload = request["method"], request["payload"]
                        if method == "hello":
                            result = {
                                "model": manifest.model.name,
                                "batching": False,
                                "protocol": PROTOCOL,
                                "required_camera_channels": manifest.required_camera_channels,
                            }
                        elif method == "reset":
                            start = EpisodeStart.model_validate(payload)
                            await run_in_threadpool(policy.configure, start.profile)
                            episode, profile = start.episode_id, start.profile
                            obs = pending = None
                            sequence, consumed, fault = -1, False, False
                            result = {"status": "reset"}
                        elif method == "end":
                            if payload != {"episode_id": episode} or episode is None:
                                raise ValueError("inactive episode")
                            await run_in_threadpool(policy.reset)
                            episode = obs = pending = None
                            result = {"status": "ended"}
                        elif method == "update_obs":
                            candidate = observation_from_wire(payload)
                            if (
                                fault
                                or episode is None
                                or candidate.episode_id != episode
                                or candidate.sequence <= sequence
                            ):
                                raise ValueError("inactive, faulted, or stale episode")
                            if pending and (
                                candidate.feedback is None
                                or candidate.feedback.decision_id != pending
                            ):
                                raise ValueError("execution feedback required")
                            validate_observation(candidate, profile)
                            validate_channels(candidate, manifest.required_camera_channels)
                            await run_in_threadpool(policy.update_obs, candidate)
                            obs, sequence, consumed = candidate, candidate.sequence, False
                            # Keep matching feedback on later updates until the next inference.
                            result = {"status": "updated"}
                        elif method == "get_action":
                            if fault or obs is None or consumed or payload != {}:
                                raise ValueError("fresh observation required before inference")
                            consumed = True
                            action = await run_in_threadpool(policy.get_action)
                            decision = Decision(
                                episode_id=episode,
                                observation_sequence=sequence,
                                decision_id=str(uuid4()),
                                valid_until_station_time_s=obs.captured_at_s
                                + profile.max_observation_age_s,
                                evidence_kind="contract-fixture"
                                if manifest.family == "fixture"
                                else "model-inference",
                                action=action,
                            )
                            validate_decision(decision, obs, profile, obs.station_time_s)
                            pending = decision.decision_id
                            result = decision.model_dump()
                        else:
                            raise ValueError("unsupported method; batching is not implemented")
                        await ws.send_bytes(
                            pack({"protocol": PROTOCOL, "request_id": request_id, "result": result})
                        )
                    except (ValueError, TypeError, KeyError) as error:
                        # State may have advanced: fault the session until an explicit reset.
                        fault = True
                        await ws.send_bytes(
                            pack(
                                {
                                    "protocol": PROTOCOL,
                                    "request_id": request_id,
                                    "error": "invalid request or policy output",
                                    "error_type": type(error).__name__,
                                }
                            )
                        )
                    except WebSocketDisconnect:
                        raise
                    except Exception:  # noqa: BLE001 - sanitize model failures at the boundary
                        fault = True
                        await ws.send_bytes(
                            pack(
                                {
                                    "protocol": PROTOCOL,
                                    "request_id": request_id,
                                    "error": "policy failure; reset required",
                                }
                            )
                        )
            except WebSocketDisconnect:
                pass
            finally:
                await run_in_threadpool(policy.reset)

    return app
