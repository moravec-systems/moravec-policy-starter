"""WebSocket env↔policy transport."""

from __future__ import annotations

from typing import Any

from client_server.model_client import WsModelClient
from client_server.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.protocol.exceptions import (
    ErrorCode,
    ServerRestartedError,
    WsError,
)
from client_server.protocol.messages import MessageType
from client_server.protocol.schemas import Frame

__all__ = [
    "ErrorCode",
    "Frame",
    "MessageType",
    "ModelClient",
    "ModelServer",
    "ModelServerConfig",
    "PolicyEvalClient",
    "PolicyEvalClientConfig",
    "PolicyServer",
    "PolicyServerConfig",
    "ServerRestartedError",
    "WsError",
    "WsModelClient",
]


def __getattr__(name: str) -> Any:
    if name == "ModelClient":
        return WsModelClient
    if name in ("PolicyServer", "ModelServer"):
        from client_server.model_server import PolicyServer

        return PolicyServer
    if name in ("PolicyServerConfig", "ModelServerConfig"):
        from client_server.model_server import PolicyServerConfig

        return PolicyServerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
