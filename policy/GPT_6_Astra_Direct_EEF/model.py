"""LLM agent policy: an OpenAI model moves the end effectors by naming Cartesian targets.

Follows the RoboProbe inspect agent's design. Each turn sends the instruction,
every camera image, and both arms' grasp-point state named exactly as the
`move_eef` tool takes it (position in metres, orientation in degrees from a
straight-down grasp, gripper 0 closed to 1 open), plus how far the previous
move landed from its target. The model answers with exactly one tool call:
`move_eef` names only the dimensions to change, and `give_up` ends the episode.
There is deliberately no "done" tool; the evaluator ends a successful episode.

Robot-specific geometry lives in deploy.yml: `down_quat_wxyz` is the reported
end-effector orientation of a straight-down grasp, and `grasp_offset_m` is how
far the grasp point sits beyond the reported pose along the tool axis. Reads
`OPENAI_API_KEY`; never put keys in deploy.yml.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from moravec_policy.model_template import ModelTemplate

ARMS = ("left", "right")
POSITION_AXES = ("x", "y", "z")
ANGLE_AXES = ("pitch_deg", "roll_deg", "yaw_deg")
AXES = (*POSITION_AXES, *ANGLE_AXES, "gripper")
DEFAULT_BOUNDS = {"pitch_deg": (-180.0, 180.0), "roll_deg": (-90.0, 90.0), "yaw_deg": (-180.0, 180.0), "gripper": (0.0, 1.0)}
MAX_REPAIRS = 3


# --------------------------------------------------------------------------- #
# Orientation: three angles in degrees applied to a straight-down reference.
# --------------------------------------------------------------------------- #

def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qnorm(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _qmat(q):
    w, x, y, z = _qnorm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _axis_quat(axis: int, degrees: float):
    half = math.radians(degrees) / 2
    q = np.zeros(4)
    q[0], q[axis + 1] = math.cos(half), math.sin(half)
    return q


def _slerp(a, b, t):
    a, b = _qnorm(a), _qnorm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return _qnorm(a + t * (b - a))
    theta = math.acos(dot)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


def _angle_between_deg(a, b):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(_qnorm(a), _qnorm(b)))))))


class _Frame:
    """Converts reported end-effector poses to the named grasp-point values the model reads and writes."""

    def __init__(self, down_quat, grasp_offset_m):
        self.down = _qnorm(down_quat)
        self.offset = float(grasp_offset_m)

    def angles_to_quat(self, pitch, roll, yaw):
        # Extrinsic rotations about world x, then y, then z, applied to the reference.
        rotation = _qmul(_axis_quat(2, yaw), _qmul(_axis_quat(1, roll), _axis_quat(0, pitch)))
        return _qmul(rotation, self.down)

    def quat_to_angles(self, q):
        m = _qmat(_qmul(_qnorm(q), _qconj(self.down)))
        roll = math.asin(float(np.clip(-m[2, 0], -1, 1)))
        if abs(math.cos(roll)) > 1e-7:
            pitch, yaw = math.atan2(m[2, 1], m[2, 2]), math.atan2(m[1, 0], m[0, 0])
        else:  # gimbal lock: put the remaining rotation on pitch
            yaw, pitch = 0.0, math.atan2((1 if m[2, 0] < 0 else -1) * m[0, 1], m[1, 1])
        return math.degrees(pitch) + 0.0, math.degrees(roll) + 0.0, math.degrees(yaw) + 0.0

    def tool_axis(self, q):
        """World direction the jaws point along; straight down at the reference orientation."""
        return _qmat(q) @ _qmat(self.down).T @ np.array([0.0, 0.0, -1.0])

    def pose_to_values(self, pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        grasp = pose[:3] + self.offset * self.tool_axis(pose[3:])
        pitch, roll, yaw = self.quat_to_angles(pose[3:])
        return {"x": grasp[0], "y": grasp[1], "z": grasp[2], "pitch_deg": pitch, "roll_deg": roll, "yaw_deg": yaw}

    def values_to_pose(self, values):
        q = self.angles_to_quat(values["pitch_deg"], values["roll_deg"], values["yaw_deg"])
        grasp = np.array([values["x"], values["y"], values["z"]])
        return np.concatenate([grasp - self.offset * self.tool_axis(q), q])


def _jpeg_data_url(image) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class _Conversation:
    """One environment's dialogue with the model."""

    def __init__(self):
        self.previous_id: str | None = None
        self.pending: list[dict[str, Any]] = []  # function_call_output items owed to the model
        self.last_targets: dict[str, np.ndarray] = {}
        self.calls = 0
        self.gave_up = False
        self.step = 0


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        if model_cfg.get("action_type", "ee") != "ee":
            raise ValueError("The LLM agent emits end-effector actions; set action_type: ee")
        self.model_name = model_cfg.get("model", "gpt-6-astra")
        self.reasoning_effort = model_cfg.get("reasoning_effort", "medium")
        self.max_llm_calls = int(model_cfg.get("max_llm_calls", 100))
        self.embodiment_notes = str(model_cfg.get("embodiment_notes") or "").strip()
        self.task_recipe = str(model_cfg.get("task_recipe") or "").strip()
        self.frame = _Frame(model_cfg.get("down_quat_wxyz", [0, 1, 0, 0]), model_cfg.get("grasp_offset_m", 0.0))
        # Motion plays at a fixed speed; a move needing more than max_duration_s is refused.
        self.control_hz = float(model_cfg.get("control_hz", 30))
        self.max_step_m = float(model_cfg.get("max_step_m", 0.01))
        self.max_step_deg = float(model_cfg.get("max_step_deg", 3.0))
        self.max_duration_s = float(model_cfg.get("max_duration_s", 10.0))
        self.gripper_dwell_steps = int(model_cfg.get("gripper_dwell_steps", 15))
        # Per-dimension [low, high]; targets outside are clamped and the model is told.
        self.bounds = {f"{arm}_{axis}": bound for arm in ARMS for axis, bound in DEFAULT_BOUNDS.items()}
        self.bounds.update({name: tuple(bound) for name, bound in (model_cfg.get("bounds") or {}).items()})
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=model_cfg.get("base_url"))
        self.tools = self._tools()
        self.reset()

    # ------------------------------------------------------------------ #
    # Protocol
    # ------------------------------------------------------------------ #

    def reset(self):
        self._obs: dict[int, dict] = {}
        self._order: list[int] = [0]
        self._conversations: dict[int, _Conversation] = {}

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._order = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._obs = dict(zip(self._order, obs_list))

    def get_action(self):
        return self.get_action_batch([self._order[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        indices = self._order if env_idx_list is None else env_idx_list
        missing = [index for index in indices if index not in self._obs]
        if missing:
            raise ValueError(f"No observation for env_idx {missing}; call update_obs first")
        return [self._act(index) for index in indices]

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def _system_message(self) -> str:
        text = (
            "You are controlling a robot with two arms, prefixed left_ and right_. You receive RGB camera "
            "images, the current world-frame grasp-point state of both arms in the same dimensions move_eef "
            "takes, and a task instruction. Move with move_eef by naming only the world-frame dimensions you "
            "want to change. Cartesian targets must be estimated from RGB; no depth or world-coordinate query "
            "is available. Respond with exactly one tool call per turn. After each motion the next observation "
            "reports how far the grasp point ended up from what you asked for, so check it before assuming a "
            "motion landed. Two budgets run down at once and whichever empties first ends the episode: "
            f"{self.max_llm_calls} LLM calls, one per turn, and the environment's own step limit, reported with "
            "each observation as the env steps remaining when the evaluator provides it. A motion spends env "
            "steps in proportion to how far it travels, so a small correction is cheap and a long reach is not."
        )
        if self.embodiment_notes:
            text += f"\n\nEmbodiment notes:\n{self.embodiment_notes}"
        return text

    def _tools(self) -> list[dict[str, Any]]:
        names = [f"{arm}_{axis}" for arm in ARMS for axis in AXES]
        bounds = ", ".join(f"{name}: [{low:.4g}, {high:.4g}]" for name in names if name in self.bounds
                           for low, high in [self.bounds[name]])
        move = {
            "type": "function",
            "name": "move_eef",
            "description": (
                "Move to absolute world-frame grasp-point targets. Name only the dimensions you intend to change; "
                "every unnamed dimension holds its observed value, so lowering one hand 2 cm is a single entry. "
                "Positions are metres. Orientation is three angles in degrees measured from a straight-down grasp, "
                "not absolute world angles: at pitch_deg=roll_deg=yaw_deg=0 the jaws point at world -z. pitch_deg "
                "turns about world +x, roll_deg about world +y, and yaw_deg about world +z, which is the jaw "
                "rotation of a straight-down grasp. Naming dimensions of both arms moves them simultaneously along "
                "time-aligned paths. The motion plays at a fixed safe speed, so how long it takes follows from how "
                f"far it goes; a move needing over {self.max_duration_s:g}s is refused and has to be split. A "
                "gripper value (0 closed, 1 open) named alongside a pose moves the jaw only after the arm has "
                "arrived, so naming both in one call is a safe way to grasp at a pose. "
                f"Per-dimension bounds: {bounds}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Map of dimension name to value. Valid names: " + ", ".join(names),
                    },
                    "note": {
                        "type": "string",
                        "description": "What you observe right now and why you chose this motion, in one or two plain sentences.",
                    },
                },
                "required": ["targets", "note"],
            },
        }
        give_up = {
            "type": "function",
            "name": "give_up",
            "description": "End the episode because the task cannot be completed.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}, "hindsight": {"type": "string"}},
                "required": ["reason", "hindsight"],
            },
        }
        return [move, give_up]

    def _observation_message(self, obs: dict, conversation: _Conversation, current: dict) -> dict[str, Any]:
        goal = f"Goal: {obs.get('instruction', '')}"
        if conversation.calls == 0 and self.task_recipe:
            goal += f"\n\nTASK RECIPE:\n{self.task_recipe}"
        lines = [goal, "World-frame grasp-point state, named as move_eef takes them (metres; degrees from the "
                       "straight-down reference):"]
        for arm in ARMS:
            for axis in AXES:
                value = current[arm][axis]
                lines.append(f"{arm}_{axis}={value:.1f}" if axis in ANGLE_AXES else
                             f"{arm}_{axis}={value:.2f}" if axis == "gripper" else f"{arm}_{axis}={value:.4f}")
        if obs.get("remaining_steps") is not None:
            lines.append(f"Env steps remaining before the episode ends: {obs['remaining_steps']}")
        arrival = []
        for arm, target in sorted(conversation.last_targets.items()):
            reached = self.frame.values_to_pose(current[arm])
            arrival.append(f"{arm} {np.linalg.norm(reached[:3] - target[:3]) * 1000:.0f} mm and "
                           f"{_angle_between_deg(reached[3:], target[3:]):.1f} deg away")
        if arrival:
            lines.append("Arrival check for the previous move_eef target: " + "; ".join(arrival) + ".")
        conversation.last_targets = {}
        content: list[dict[str, Any]] = [{"type": "input_text", "text": "\n".join(lines)}]
        for name, camera in obs.get("vision", {}).items():
            image = camera["color"] if isinstance(camera, dict) else camera
            content.append({"type": "input_text", "text": f"camera '{name}' (step {conversation.step}):"})
            content.append({"type": "input_image", "image_url": _jpeg_data_url(image)})
        return {"role": "user", "content": content}

    # ------------------------------------------------------------------ #
    # Turn
    # ------------------------------------------------------------------ #

    def _state(self, obs: dict) -> dict[str, dict[str, Any]]:
        current = {}
        for arm in ARMS:
            values = self.frame.pose_to_values(obs["state"][f"{arm}_ee_pose"])
            values["gripper"] = float(np.asarray(obs["state"][f"{arm}_ee_joint_state"]).reshape(-1)[0])
            values["pose"] = np.asarray(obs["state"][f"{arm}_ee_pose"], dtype=np.float64).reshape(7)
            current[arm] = values
        return current

    def _act(self, env_idx: int) -> list[dict[str, np.ndarray]]:
        obs = self._obs[env_idx]
        conversation = self._conversations.setdefault(env_idx, _Conversation())
        current = self._state(obs)
        if conversation.gave_up or conversation.calls >= self.max_llm_calls:
            conversation.gave_up = True
            return self._hold(current)

        turn: list[dict[str, Any]] = conversation.pending + [self._observation_message(obs, conversation, current)]
        conversation.pending = []
        repairs = 0
        while conversation.calls < self.max_llm_calls:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=self._system_message(),
                input=turn,
                tools=self.tools,
                tool_choice="required",
                parallel_tool_calls=False,
                reasoning={"effort": self.reasoning_effort},
                previous_response_id=conversation.previous_id,
            )
            conversation.calls += 1
            conversation.previous_id = response.id
            calls = [item for item in response.output if item.type == "function_call"]
            if len(calls) != 1:
                repairs += 1
                if repairs > MAX_REPAIRS:
                    raise RuntimeError("Model kept failing to produce exactly one tool call")
                turn = [{"type": "function_call_output", "call_id": c.call_id, "output": "Ignored."} for c in calls]
                turn.append({"role": "user", "content": "Reply with exactly one tool call from the provided tools."})
                continue
            call = calls[0]
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError as error:
                arguments, result = None, f"invalid JSON arguments: {error}"
            if arguments is not None and call.name == "give_up":
                conversation.gave_up = True
                conversation.pending = [{"type": "function_call_output", "call_id": call.call_id, "output": "Acknowledged give_up."}]
                return self._hold(current)
            if arguments is not None and call.name == "move_eef":
                chunk, result, repairable = self._move(arguments, current, conversation)
                if chunk is not None:
                    conversation.pending = [{"type": "function_call_output", "call_id": call.call_id, "output": result}]
                    conversation.step += len(chunk)
                    return chunk
                if not repairable:  # a valid but refused request; let the model choose again
                    turn = [{"type": "function_call_output", "call_id": call.call_id, "output": result}]
                    continue
            elif arguments is not None:
                result = f"unknown tool {call.name!r}"
            repairs += 1
            if repairs > MAX_REPAIRS:
                raise RuntimeError("Model kept failing to produce a valid tool call")
            turn = [{"type": "function_call_output", "call_id": call.call_id, "output": result}]
        conversation.gave_up = True
        return self._hold(current)

    def _move(self, arguments: dict, current: dict, conversation: _Conversation):
        """Return (chunk, tool_result, repairable)."""
        targets = arguments.get("targets")
        if not isinstance(targets, dict) or not targets:
            return None, "move_eef.targets must be a non-empty object of named dimensions.", True
        goals = {arm: {axis: current[arm][axis] for axis in AXES} for arm in ARMS}
        notes = []
        for name, value in targets.items():
            arm, _, axis = name.partition("_")
            if arm not in ARMS or axis not in AXES:
                return None, f"unknown dimension {name!r}; valid names are {', '.join(f'{a}_{x}' for a in ARMS for x in AXES)}.", True
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return None, f"{name} must be a finite number.", True
            if name in self.bounds:
                low, high = self.bounds[name]
                clamped = min(max(float(value), low), high)
                if clamped != value:
                    notes.append(f"{name} clamped from {value:g} to {clamped:g}")
                value = clamped
            goals[arm][axis] = float(value)

        start = {arm: current[arm]["pose"] for arm in ARMS}
        end = {arm: self.frame.values_to_pose(goals[arm]) if any(
            f"{arm}_{axis}" in targets for axis in (*POSITION_AXES, *ANGLE_AXES)) else start[arm] for arm in ARMS}
        steps = 1
        for arm in ARMS:
            travel = np.linalg.norm(end[arm][:3] - start[arm][:3]) / self.max_step_m
            turn = _angle_between_deg(start[arm][3:], end[arm][3:]) / self.max_step_deg
            steps = max(steps, int(math.ceil(max(travel, turn) - 1e-6)))
        if steps / self.control_hz > self.max_duration_s:
            return None, (f"Refused: this move needs {steps / self.control_hz:.1f}s, over the "
                          f"{self.max_duration_s:g}s limit. Split it into shorter moves."), False
        grip_change = any(abs(goals[arm]["gripper"] - current[arm]["gripper"]) > 1e-3 for arm in ARMS)
        dwell = self.gripper_dwell_steps if grip_change else 0

        chunk = []
        for step in range(1, steps + dwell + 1):
            t = min(step / steps, 1.0)
            action = {}
            for arm in ARMS:
                position = start[arm][:3] + t * (end[arm][:3] - start[arm][:3])
                action[f"{arm}_ee_pose"] = np.concatenate([position, _slerp(start[arm][3:], end[arm][3:], t)]).astype(np.float32)
                # The jaw moves only after the arm has arrived.
                gripper = goals[arm]["gripper"] if step >= steps else current[arm]["gripper"]
                action[f"{arm}_ee_joint_state"] = np.array([gripper], dtype=np.float32)
            chunk.append(action)
        conversation.last_targets = {arm: end[arm] for arm in ARMS if end[arm] is not start[arm]}
        result = f"Accepted; the motion spends {len(chunk)} env steps."
        if notes:
            result += " " + "; ".join(notes) + "."
        return chunk, result, False

    @staticmethod
    def _hold(current: dict) -> list[dict[str, np.ndarray]]:
        return [{**{f"{arm}_ee_pose": current[arm]["pose"].astype(np.float32) for arm in ARMS},
                 **{f"{arm}_ee_joint_state": np.array([current[arm]["gripper"]], dtype=np.float32) for arm in ARMS}}]
