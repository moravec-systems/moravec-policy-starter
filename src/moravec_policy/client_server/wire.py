"""Binary MessagePack envelopes; observation payload is separate from trial metadata."""

import msgpack

from ..contracts import Observation

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
STATE_KEYS = (
    "joint_positions",
    "joint_velocities",
    "tcp_poses",
    "ee_poses",
    "gripper_open_fraction",
    "gripper_joint_positions",
    "base_pose",
    "base_twist",
)


def pack(value):
    data = msgpack.packb(value, use_bin_type=True)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("message too large")
    return data


def unpack(data):
    if not isinstance(data, bytes) or len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("binary MessagePack frame required within size limit")
    value = msgpack.unpackb(data, raw=False, strict_map_key=True)
    if not isinstance(value, dict):
        raise TypeError("message must be a map")
    return value


def observation_to_wire(obs):
    data = obs.model_dump()
    state = {key: data.pop(key) for key in STATE_KEYS}
    payload = {
        "instruction": data.pop("instruction"),
        "vision": data.pop("cameras"),
        "state": state,
    }
    return {"metadata": data, "observation": payload}


def observation_from_wire(value):
    if set(value) != {"metadata", "observation"}:
        raise ValueError("invalid observation envelope")
    payload = value["observation"]
    if set(payload) != {"instruction", "vision", "state"} or set(payload["state"]) != set(
        STATE_KEYS
    ):
        raise ValueError("invalid observation payload")
    metadata = value["metadata"]
    if set(metadata).intersection({"instruction", "cameras", *STATE_KEYS}):
        raise ValueError("payload fields prohibited in metadata")
    return Observation.model_validate(
        {
            **metadata,
            "instruction": payload["instruction"],
            "cameras": payload["vision"],
            **payload["state"],
        }
    )


# Generate wire models from the same fields as the internal observation object.
# This avoids maintaining two independent definitions of units and modalities.
from pydantic import create_model

from ..contracts import Camera, Contract


def subset(name, keys):
    return create_model(
        name,
        __base__=Contract,
        **{
            key: (Observation.model_fields[key].annotation, Observation.model_fields[key])
            for key in keys
        },
    )


RobotState = subset("RobotState", STATE_KEYS)
ObservationMetadata = subset(
    "ObservationMetadata",
    [k for k in Observation.model_fields if k not in {*STATE_KEYS, "instruction", "cameras"}],
)


class ObservationPayload(Contract):
    instruction: str
    vision: dict[str, Camera]
    state: RobotState


class ObservationEnvelope(Contract):
    metadata: ObservationMetadata
    observation: ObservationPayload
