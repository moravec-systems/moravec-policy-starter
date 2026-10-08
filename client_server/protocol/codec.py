"""msgpack encode/decode for WebSocket binary frames."""

from __future__ import annotations

import sys
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
from pydantic import ValidationError

from client_server.protocol.exceptions import ErrorCode, WsError
from client_server.protocol.schemas import Frame


def _tensor_to_numpy(obj: Any) -> Any:
    """Convert a torch.Tensor to numpy, or return obj unchanged.

    torch is looked up in sys.modules rather than imported: only a process
    that already loaded torch can hold a tensor, so this keeps the env-side
    client from paying a multi-second torch import (and a CUDA context) just
    to serialize observations.
    """
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(obj, torch.Tensor):
        return obj
    # Models may return CUDA or
    # half-precision tensors straight from inference.
    tensor = obj.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        # numpy has no bfloat16; widen instead of failing.
        tensor = tensor.float()
    return tensor.numpy()


def _encode_numpy(obj: Any) -> Any:
    obj = _tensor_to_numpy(obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise WsError(
                ErrorCode.INVALID_FRAME, "object dtype numpy arrays are not supported"
            )
        return msgpack_numpy.encode(obj)
    if isinstance(obj, np.generic):
        if obj.dtype.kind == "O":
            raise WsError(
                ErrorCode.INVALID_FRAME, "object dtype numpy scalars are not supported"
            )
    return msgpack_numpy.encode(obj)


def _normalize_map_keys(mapping: dict[Any, Any]) -> dict[Any, Any]:
    """Normalize integer map keys to strings for policy observation lookups.

    Preserve the existing wire contract: environment-indexed maps use "0",
    not 0, even though MessagePack can preserve integer keys.
    """
    if all(isinstance(key, str) for key in mapping):
        return mapping
    out: dict[Any, Any] = {}
    for key, value in mapping.items():
        if isinstance(key, int) and not isinstance(key, bool):
            key = str(key)
        if key in out:
            raise ValueError(
                f"duplicate map key after int->str normalization: {key!r}"
            )
        out[key] = value
    return out


def _decode_numpy(obj: dict[Any, Any]) -> Any:
    if obj.get(b"nd") is True and obj.get(b"kind") == b"O":
        raise ValueError("object dtype numpy arrays are not supported")
    decoded = msgpack_numpy.decode(obj)
    if isinstance(decoded, np.ndarray) and decoded.dtype.kind == "O":
        raise ValueError("object dtype numpy arrays are not supported")
    if isinstance(decoded, dict):
        decoded = _normalize_map_keys(decoded)
    return decoded


def encode_frame(frame: Frame | dict[str, Any]) -> bytes:
    if isinstance(frame, Frame):
        wire = frame.to_wire_dict()
    else:
        wire = dict(frame)
    try:
        return msgpack.packb(wire, default=_encode_numpy, use_bin_type=True)
    except WsError:
        raise
    except Exception as exc:
        raise WsError(ErrorCode.INVALID_FRAME, f"msgpack encode failed: {exc}") from exc


def decode_frame(data: bytes) -> dict[str, Any]:
    try:
        obj = msgpack.unpackb(
            data,
            raw=False,
            # Observations may key dicts by env index. Accept int keys here;
            # _decode_numpy then normalizes them to strings so models see the
            # same key type the legacy JSON codec delivered.
            strict_map_key=False,
            object_hook=_decode_numpy,
        )
    except Exception as exc:
        raise WsError(ErrorCode.INVALID_FRAME, f"msgpack decode failed: {exc}") from exc
    if not isinstance(obj, dict):
        raise WsError(ErrorCode.INVALID_FRAME, "frame must be a msgpack map")
    return obj


def decode_envelope(data: bytes) -> Frame:
    try:
        return Frame.from_wire_dict(decode_frame(data))
    except WsError:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        raise WsError(
            ErrorCode.INVALID_FRAME, f"invalid frame envelope: {exc}"
        ) from exc
