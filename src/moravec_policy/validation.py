"""Checks belong on both inference intake and the station execution boundary."""

from .contracts import ActionChunk, Decision, Observation, RobotProfile


def validate_observation(obs: Observation, profile: RobotProfile):
    if (obs.profile_id, obs.profile_revision) != (profile.id, profile.revision):
        raise ValueError("profile identity mismatch")
    if set(obs.joint_positions) != {j.name for j in profile.joints}:
        raise ValueError("joint names do not match profile")
    if set(obs.cameras) != set(profile.cameras):
        raise ValueError("camera names do not match profile")
    age = obs.station_time_s - obs.captured_at_s
    if not 0 <= age <= profile.max_observation_age_s:
        raise ValueError("stale/future observation")
    for camera in obs.cameras.values():
        if not 0 <= obs.captured_at_s - camera.captured_at_s <= profile.max_camera_skew_s:
            raise ValueError("camera skew exceeds profile")
    if not set(obs.joint_velocities).issubset({j.name for j in profile.joints}):
        raise ValueError("unknown velocity joint")
    for camera in obs.cameras.values():
        if camera.extrinsics_frame is not None and camera.extrinsics_frame not in profile.frames:
            raise ValueError("unknown camera frame")
    if (obs.base_pose is None) != (obs.base_twist is None):
        raise ValueError("mobile state requires pose and twist together")
    for pose in [
        *obs.tcp_poses.values(),
        *obs.ee_poses.values(),
        *([obs.base_pose] if obs.base_pose else []),
    ]:
        if pose.frame not in profile.frames:
            raise ValueError("unknown TCP frame")


def validate_decision(decision: Decision, obs: Observation, profile: RobotProfile, now_s: float):
    if (decision.episode_id, decision.observation_sequence) != (obs.episode_id, obs.sequence):
        raise ValueError("decision does not match observation")
    if (
        not now_s
        <= decision.valid_until_station_time_s
        <= obs.captured_at_s + profile.max_observation_age_s
    ):
        raise ValueError("expired or excessive action validity")
    action = decision.action
    if isinstance(action, ActionChunk):
        if action.timestep_s * len(action.steps) > profile.max_action_duration_s:
            raise ValueError("action chunk too long")
        joints = {j.name: j for j in profile.joints}
        for step in action.steps:
            if not set(step.joint_positions).issubset(joints):
                raise ValueError("unknown joint")
            if not set(step.grippers).issubset(obs.gripper_open_fraction):
                raise ValueError("unknown gripper")
            for name, value in step.joint_positions.items():
                if not joints[name].lower <= value <= joints[name].upper:
                    raise ValueError("joint limit exceeded")
            for arm, target in step.tcp_targets.items():
                if arm not in obs.tcp_poses:
                    raise ValueError("unknown arm")
                if target.pose.frame not in profile.frames:
                    raise ValueError("unknown pose frame")
                if (
                    target.representation == "relative"
                    and target.pose.frame != obs.tcp_poses[arm].frame
                ):
                    raise ValueError("relative pose requires calibrated common frame")


def validate_channels(obs, requirements):
    for name, channels in requirements.items():
        if name not in obs.cameras:
            raise ValueError("required camera missing")
        camera = obs.cameras[name]
        fields = {
            "rgb": camera.data,
            "depth": camera.depth,
            "intrinsics": camera.intrinsic_matrix,
            "extrinsics": camera.extrinsics_matrix,
        }
        if any(fields[channel] is None for channel in channels):
            raise ValueError("required camera channel missing")
