"""LLM agent policy: an OpenAI model chooses end-effector goals from camera images.

Each ``get_action`` sends the instruction, camera frames, and end-effector state
to the Responses API with one ``move_eef`` tool, then interpolates from the
current pose to the requested goals. The evaluator converts end-effector actions
into joint commands. Reads ``OPENAI_API_KEY``; never put keys in deploy.yml.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from moravec_policy.model_template import ModelTemplate

ARMS = ("left", "right")

INSTRUCTIONS = """You control a bimanual robot. Each turn you receive the task, camera
images, and each arm's current end-effector state. Respond by calling move_eef
for one or both arms; each call moves that arm's gripper to an absolute
world-frame position (metres) and sets its gripper opening (0 closed, 1 open).
Orientation is held. After the motion you receive new images and state.
Approach objects from above, then descend, then close the gripper in a separate
move. Keep moves short enough that you can check progress between them."""

MOVE_EEF = {
    "type": "function",
    "name": "move_eef",
    "description": "Move one arm's end effector to an absolute world-frame position and set its gripper.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "arm": {"type": "string", "enum": list(ARMS)},
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "description": "[x, y, z] in metres, world frame.",
            },
            "gripper": {"type": "number", "description": "0 closed, 1 open."},
        },
        "required": ["arm", "position", "gripper"],
        "additionalProperties": False,
    },
}


def _jpeg_data_url(image: np.ndarray, size: int) -> str:
    image = np.asarray(image, dtype=np.uint8)
    scale = size / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode camera image")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        if model_cfg.get("action_type", "ee") != "ee":
            raise ValueError("The LLM agent emits end-effector actions; set action_type: ee")
        self.model_name = model_cfg.get("model", "gpt-6-astra")
        self.reasoning_effort = model_cfg.get("reasoning_effort", "low")
        self.scene_notes = str(model_cfg.get("scene_notes") or "").strip()
        self.image_size = int(model_cfg.get("image_size", 384))
        # Interpolation: at most max_step_m of travel per env step, and hold the
        # final pose for gripper_dwell_steps when the gripper changes.
        self.max_step_m = float(model_cfg.get("max_step_m", 0.01))
        self.max_chunk = int(model_cfg.get("max_chunk", 100))
        self.gripper_dwell_steps = int(model_cfg.get("gripper_dwell_steps", 10))
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=model_cfg.get("base_url"))
        self.reset()

    def reset(self):
        self._obs: dict[int, dict] = {}
        self._order: list[int] = [0]
        # One conversation per environment: (previous_response_id, pending tool call ids).
        self._conversations: dict[int, tuple[str | None, list[str]]] = {}

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

    def _act(self, env_idx: int) -> list[dict[str, np.ndarray]]:
        obs = self._obs[env_idx]
        current = {arm: self._arm_state(obs, arm) for arm in ARMS}
        previous_id, pending_calls = self._conversations.get(env_idx, (None, []))

        # Every function call must be answered before the next user turn.
        turn: list[dict[str, Any]] = [
            {"type": "function_call_output", "call_id": call_id, "output": "Done. New observation follows."}
            for call_id in pending_calls
        ]
        turn.append({"role": "user", "content": self._observation_content(obs, current)})
        response = self.client.responses.create(
            model=self.model_name,
            instructions=INSTRUCTIONS + (f"\n\nScene notes:\n{self.scene_notes}" if self.scene_notes else ""),
            input=turn,
            tools=[MOVE_EEF],
            tool_choice="required",
            reasoning={"effort": self.reasoning_effort},
            previous_response_id=previous_id,
        )

        goals = {arm: dict(state) for arm, state in current.items()}
        call_ids = []
        for item in response.output:
            if item.type != "function_call" or item.name != "move_eef":
                continue
            call_ids.append(item.call_id)
            args = json.loads(item.arguments)
            if len(args["position"]) != 3:
                raise ValueError(f"move_eef position must have three values, got {args['position']}")
            goals[args["arm"]]["position"] = np.asarray(args["position"], dtype=np.float32)
            goals[args["arm"]]["gripper"] = float(np.clip(args["gripper"], 0.0, 1.0))
        self._conversations[env_idx] = (response.id, call_ids)
        return self._interpolate(current, goals)

    def _observation_content(self, obs: dict, current: dict) -> list[dict[str, Any]]:
        state = {
            arm: {
                "position": np.round(current[arm]["position"], 4).tolist(),
                "quaternion_wxyz": np.round(current[arm]["quaternion"], 4).tolist(),
                "gripper": round(current[arm]["gripper"], 3),
            }
            for arm in ARMS
        }
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": f"Task: {obs.get('instruction', '')}\nState: {json.dumps(state)}"}
        ]
        for name, camera in obs.get("vision", {}).items():
            image = camera["color"] if isinstance(camera, dict) else camera
            content.append({"type": "input_text", "text": f"Camera {name}:"})
            content.append({"type": "input_image", "image_url": _jpeg_data_url(image, self.image_size)})
        return content

    @staticmethod
    def _arm_state(obs: dict, arm: str) -> dict[str, Any]:
        pose = np.asarray(obs["state"][f"{arm}_ee_pose"], dtype=np.float32).reshape(7)
        gripper = float(np.asarray(obs["state"][f"{arm}_ee_joint_state"]).reshape(-1)[0])
        return {"position": pose[:3], "quaternion": pose[3:], "gripper": gripper}

    def _interpolate(self, current: dict, goals: dict) -> list[dict[str, np.ndarray]]:
        travel = max(float(np.linalg.norm(goals[arm]["position"] - current[arm]["position"])) for arm in ARMS)
        steps = int(np.clip(np.ceil(travel / self.max_step_m - 1e-6), 1, self.max_chunk))
        gripper_changes = any(abs(goals[arm]["gripper"] - current[arm]["gripper"]) > 1e-3 for arm in ARMS)
        dwell = self.gripper_dwell_steps if gripper_changes else 0

        chunk = []
        for step in range(1, steps + dwell + 1):
            fraction = min(step / steps, 1.0)
            action = {}
            for arm in ARMS:
                position = current[arm]["position"] + fraction * (goals[arm]["position"] - current[arm]["position"])
                action[f"{arm}_ee_pose"] = np.concatenate([position, current[arm]["quaternion"]]).astype(np.float32)
                # The gripper closes or opens once the arm has arrived.
                gripper = goals[arm]["gripper"] if step >= steps else current[arm]["gripper"]
                action[f"{arm}_ee_joint_state"] = np.array([gripper], dtype=np.float32)
            chunk.append(action)
        return chunk
