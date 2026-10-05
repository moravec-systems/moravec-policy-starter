import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from moravec_policy.action_adapter import rotation_wxyz
from moravec_policy.checkpoint import qualified_checkpoint, validate_binding_profile
from moravec_policy.contracts import ActionChunk
from moravec_policy.lifecycle import Policy


class Model(Policy):
    def __init__(self, manifest, path):
        from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

        cfg = _load_deploy_yaml(None)
        cfg.optimization.compile.enabled = False
        self.cfg, checkpoint = qualified_checkpoint(manifest, path)
        self.model = build_server_from_config(
            cfg, str(checkpoint), device=manifest.config.get("device", "cuda")
        )
        self.reset()

    def _reset(self):
        self.model.reset()

    def _infer(self, obs, profile):
        validate_binding_profile(self.cfg, obs)
        if profile.id != "libero-panda" or profile.control_period_s != 0.05:
            raise ValueError("LIBERO profile required")
        pose = obs.tcp_poses["arm"]
        g = obs.gripper_joint_positions["gripper"]
        r = (
            Rotation.from_quat(pose.orientation_wxyz, scalar_first=True)
            .as_matrix()[:, :2]
            .T.reshape(-1)
        )
        state = [*pose.position_m, *r, float(np.clip(2 * (g[0] - g[1]) / 0.08 - 1, -1, 1))]
        images = {}
        for key, c, size in [
            ("head_camera", "overview", (320, 256)),
            ("left_wrist_camera", "wrist", (160, 128)),
        ]:
            camera = obs.cameras[c]
            images[key] = Image.fromarray(
                np.frombuffer(camera.data, dtype=np.uint8).reshape(camera.shape)
            ).resize(size, Image.Resampling.LANCZOS)
        result = self.model.predict({"images": images, "prompt": obs.instruction, "state": state})
        a = np.asarray(result["action"]).reshape(-1)
        if a.shape != (10,) or not np.isfinite(a).all():
            raise ValueError("invalid native action")
        r = Rotation.from_quat(
            rotation_wxyz(a[3:9], "rotation6d-columns"), scalar_first=True
        ).as_rotvec()
        q = Rotation.from_rotvec(np.clip(r, -1, 1) * 0.5).as_quat(scalar_first=True)
        return ActionChunk(
            timestep_s=0.05,
            execution_horizon=1,
            steps=[
                {
                    "tcp_targets": {
                        "arm": {
                            "pose": {
                                "frame": "world",
                                "position_m": np.clip(a[:3], -1, 1) * 0.05,
                                "orientation_wxyz": q,
                            },
                            "representation": "relative",
                            "reference": "frame",
                            "relative_to": "observation",
                        }
                    },
                    "grippers": {"gripper": float(np.clip((a[9] + 1) / 2, 0, 1))},
                }
            ],
        )
