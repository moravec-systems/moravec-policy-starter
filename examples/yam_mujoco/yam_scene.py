"""Bimanual YAM tabletop scene for closed-loop policy checks, built on i2rt's MIT-licensed station model.

The i2rt station MJCF is kinematics only. This adds MuJoCo cameras at its three
D405 optical frames, position actuators for six joints plus the gripper on each
arm, a table, and a small pick-and-place task (red block, green plate).
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

# i2rt checkout (see README); defaults to examples/yam_mujoco/i2rt.
I2RT_ROOT = Path(os.environ.get("I2RT_ROOT", Path(__file__).resolve().parent / "i2rt"))
STATION = I2RT_ROOT / "i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.xml"
ARMS = ("left", "right")
GRIPPER_TRAVEL = 0.04695  # joint7 range; command 0 = closed, 1 = open (i2rt convention)
CAMERAS = {"top": "top_camera", "left": "left_camera", "right": "right_camera"}
BLOCK_START = np.array([0.32, -0.22, 0.02])
PLATE_POS = np.array([0.32, -0.40, 0.003])
READY_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # fingertip frame +z pointing down


def build(station: Path = STATION) -> mujoco.MjModel:
    if not station.is_file():
        raise FileNotFoundError(f"YAM station model not found at {station}; clone i2rt as described in README.md")
    spec = mujoco.MjSpec.from_file(str(station))
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    world = spec.worldbody

    world.add_light(pos=[0.3, -0.3, 1.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_light(pos=[-0.5, -0.3, 1.2], dir=[0.5, 0, -1], diffuse=[0.4, 0.4, 0.4])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                   rgba=[0.35, 0.35, 0.38, 1])
    # Table top surface at z = 0, where both arm bases are mounted.
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.45, 0.6, 0.02], pos=[0.25, -0.305, -0.02],
                   rgba=[0.62, 0.52, 0.42, 1], friction=[1, 0.005, 0.0001])

    block = world.add_body(name="block", pos=BLOCK_START)
    block.add_freejoint(name="block_free")
    block.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], rgba=[0.85, 0.1, 0.1, 1], mass=0.05,
                   friction=[1.5, 0.01, 0.001], condim=4)
    world.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, 0.003, 0], pos=PLATE_POS,
                   rgba=[0.15, 0.6, 0.25, 1])

    # Keep marker sites (e.g. i2rt's red tcp_* sites) out of the camera images.
    for site in spec.sites:
        site.group = 5

    # MuJoCo cameras look along -Z with +Y up; the station frames are optical (+Z forward, +Y down).
    for name, body in CAMERAS.items():
        spec.body(body).add_camera(name=name, quat=[0, 1, 0, 0], fovy=58)

    for arm in ARMS:
        for body in spec.bodies:
            if body.name.startswith(f"{arm}_"):
                body.gravcomp = 1  # stands in for the real arm's gravity compensation
        for j in range(1, 7):
            joint = spec.joint(f"{arm}_joint{j}")
            joint.damping = [2.0, 0, 0]
            joint.armature = 0.05
            spec.add_actuator(name=f"{arm}_joint{j}", target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                              gainprm=[150] + [0] * 9, biasprm=[0, -150, -8] + [0] * 7,
                              ctrlrange=joint.range, ctrllimited=True, forcerange=[-40, 40], forcelimited=True)
        spec.add_actuator(name=f"{arm}_gripper", target=f"{arm}_joint7", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                          gainprm=[800] + [0] * 9, biasprm=[0, -800, -20] + [0] * 7,
                          ctrlrange=[0, GRIPPER_TRAVEL], ctrllimited=True, forcerange=[-30, 30], forcelimited=True)
        # The welded base touches link1 at rest; MuJoCo only filters parent/child pairs below the root.
        for link in ("link1", "link2"):
            spec.add_exclude(bodyname1=f"{arm}_base", bodyname2=f"{arm}_{link}")
        for tip in ("tip_left", "tip_right"):
            for geom in spec.body(f"{arm}_{tip}").geoms:
                geom.friction = [1.5, 0.01, 0.001]
                geom.condim = 4
    return spec.compile()


class YamTabletop:
    """Steps the scene with absolute joint or end-effector targets and renders the three cameras.

    End-effector poses are the fingertip centre (i2rt's `tcp_left`/`tcp_right` sites) in the
    world frame as [x, y, z, qw, qx, qy, qz]; targets are converted to joint targets with
    damped least-squares inverse kinematics.
    """

    def __init__(self, control_hz: float = 30.0, image_size=(480, 640)):
        self.model = build()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, *image_size)
        self.substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.qpos_ids = {arm: [self.model.joint(f"{arm}_joint{j}").qposadr[0] for j in range(1, 8)] for arm in ARMS}
        self.act_ids = {arm: [self.model.actuator(f"{arm}_joint{j}").id for j in range(1, 7)]
                        + [self.model.actuator(f"{arm}_gripper").id] for arm in ARMS}
        self.dof_ids = {arm: [self.model.joint(f"{arm}_joint{j}").dofadr[0] for j in range(1, 7)] for arm in ARMS}
        self._ik_data = mujoco.MjData(self.model)
        self.reset()

    def reset(self, arm_qpos=None, gripper: float = 1.0):
        mujoco.mj_resetData(self.model, self.data)
        arm_qpos = np.zeros(6) if arm_qpos is None else np.asarray(arm_qpos, dtype=float)
        for arm in ARMS:
            target = np.append(arm_qpos, gripper)
            self.data.qpos[self.qpos_ids[arm]] = np.append(arm_qpos, gripper * GRIPPER_TRAVEL)
            self.data.qpos[self.model.joint(f"{arm}_joint8").qposadr[0]] = gripper * GRIPPER_TRAVEL
            self._set_ctrl(arm, target)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):  # let the block settle
            mujoco.mj_step(self.model, self.data)

    def _set_ctrl(self, arm, joints7):
        joints7 = np.asarray(joints7, dtype=float).reshape(7)
        self.data.ctrl[self.act_ids[arm][:6]] = joints7[:6]
        self.data.ctrl[self.act_ids[arm][6]] = np.clip(joints7[6], 0, 1) * GRIPPER_TRAVEL

    def step(self, action: dict):
        for arm in ARMS:
            self._set_ctrl(arm, np.concatenate([np.asarray(action[f"{arm}_arm_joint_state"]).reshape(6),
                                                np.asarray(action[f"{arm}_ee_joint_state"]).reshape(1)]))
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def ee_pose(self, arm, data=None) -> np.ndarray:
        data = self.data if data is None else data
        site = data.site(f"tcp_{arm}")
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        return np.concatenate([site.xpos, quat])

    def solve_ik(self, arm, target_pose, iterations=100, damping=1e-3) -> np.ndarray:
        """Joint angles that place the arm's fingertip centre at target_pose, starting from its current joints."""
        target_pose = np.asarray(target_pose, dtype=float).reshape(7)
        data = self._ik_data
        data.qpos[:] = self.data.qpos
        site_id = self.model.site(f"tcp_{arm}").id
        dofs, qadr = self.dof_ids[arm], self.qpos_ids[arm][:6]
        jac_pos, jac_rot = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        error = np.zeros(6)
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            current = self.ee_pose(arm, data)
            error[:3] = target_pose[:3] - current[:3]
            local = np.zeros(3)
            mujoco.mju_subQuat(local, target_pose[3:], current[3:])
            error[3:] = data.site(f"tcp_{arm}").xmat.reshape(3, 3) @ local  # Jacobian is in the world frame
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, site_id)
            jac = np.vstack([jac_pos, jac_rot])[:, dofs]
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), error)
            step *= min(1.0, 0.2 / max(np.abs(step).max(), 1e-9))  # at most 0.2 rad per iteration
            q = data.qpos[qadr] + step
            data.qpos[qadr] = np.clip(q, self.model.jnt_range[[self.model.joint(f"{arm}_joint{j}").id for j in range(1, 7)], 0],
                                      self.model.jnt_range[[self.model.joint(f"{arm}_joint{j}").id for j in range(1, 7)], 1])
        return data.qpos[qadr].copy()

    def step_ee(self, action: dict):
        """Step with end-effector targets ({arm}_ee_pose, {arm}_ee_joint_state)."""
        joints = {}
        for arm in ARMS:
            joints[f"{arm}_arm_joint_state"] = self.solve_ik(arm, action[f"{arm}_ee_pose"])
            joints[f"{arm}_ee_joint_state"] = action[f"{arm}_ee_joint_state"]
        self.step(joints)

    def move_to_ready(self, seconds: float = 2.0):
        """Raise both grippers above the table, pointing straight down and open."""
        ready = {"left": [0.25, -0.05, 0.20], "right": [0.25, -0.56, 0.20]}
        action = {}
        for arm, xyz in ready.items():
            action[f"{arm}_ee_pose"] = np.concatenate([xyz, READY_QUAT])
            action[f"{arm}_ee_joint_state"] = [1.0]
        for _ in range(round(seconds / (self.substeps * self.model.opt.timestep))):
            self.step_ee(action)

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observation(self, instruction: str, action_type: str = "joint") -> dict:
        state = {}
        for arm in ARMS:
            q = self.data.qpos[self.qpos_ids[arm]]
            if action_type == "ee":
                state[f"{arm}_ee_pose"] = self.ee_pose(arm).astype(np.float32)
            else:
                state[f"{arm}_arm_joint_state"] = q[:6].astype(np.float32)
            state[f"{arm}_ee_joint_state"] = np.array([q[6] / GRIPPER_TRAVEL], dtype=np.float32)
        return {"instruction": instruction, "state": state,
                "vision": {name: {"color": self.render(name)} for name in CAMERAS}}

    def block_on_plate(self) -> bool:
        pos = self.data.body("block").xpos
        return bool(np.linalg.norm(pos[:2] - PLATE_POS[:2]) < 0.06 and pos[2] < 0.05)
