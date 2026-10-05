from urllib.parse import urlparse
from uuid import uuid4

from websockets.sync.client import connect

from .contracts import Decision
from .wire import MAX_MESSAGE_BYTES, observation_to_wire, pack, unpack

PROTOCOL = "moravec.policy/v0.4"


class RemotePolicy:
    def __init__(self, url, token, timeout_s=75):
        parts = urlparse(url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("credentials/query/fragment prohibited in endpoint URL")
        if parts.scheme != "wss" and not (
            parts.scheme == "ws" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("non-loopback policy endpoints require WSS")
        self.timeout_s = timeout_s
        self.ws = connect(
            url.rstrip("/") + "/policy",
            additional_headers={"Authorization": "Bearer " + token},
            max_size=MAX_MESSAGE_BYTES,
            compression=None,
            proxy=None,
        )
        self.obs = None
        self.call("hello", {})

    def call(self, method, payload):
        request_id = str(uuid4())
        self.ws.send(
            pack(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "method": method,
                    "payload": payload,
                }
            )
        )
        response = unpack(self.ws.recv(timeout=self.timeout_s))
        if response.get("protocol") != PROTOCOL or response.get("request_id") != request_id:
            raise ValueError("mismatched protocol response")
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def start(self, episode_id, profile):
        self.call("reset", {"episode_id": episode_id, "profile": profile.model_dump()})
        self.obs = None

    def update_obs(self, obs):
        self.call("update_obs", observation_to_wire(obs))
        self.obs = obs

    def get_action(self):
        if self.obs is None:
            raise ValueError("update_obs required")
        result = Decision.model_validate(self.call("get_action", {}))
        if (result.episode_id, result.observation_sequence) != (
            self.obs.episode_id,
            self.obs.sequence,
        ):
            raise ValueError("mismatched decision")
        return result

    def end(self, episode_id):
        self.call("end", {"episode_id": episode_id})
        self.obs = None

    def close(self):
        self.ws.close()
