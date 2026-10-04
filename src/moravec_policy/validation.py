"""Checks belong on both inference intake and the station execution boundary."""

from .contracts import (
    CoordinatedAction,
    Decision,
    GripperCommand,
    JointTrajectory,
    Observation,
    PoseTrajectory,
    RobotProfile,
)


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
    if isinstance(action, CoordinatedAction):
        controlled = set()
        for track in action.tracks:
            if isinstance(track, JointTrajectory):
                resources = {"joint:" + n for n in track.points[0].positions}
            elif isinstance(track, PoseTrajectory):
                resources = {"arm:" + track.arm}
                if any(p.gripper_open_fraction is not None for p in track.points):
                    resources.add("gripper:" + track.arm)
            else:
                resources = {"gripper:" + track.gripper}
            if controlled.intersection(resources):
                raise ValueError("overlapping coordinated control")
            controlled.update(resources)
            validate_decision(decision.model_copy(update={"action": track}), obs, profile, now_s)
        if any(x.startswith("joint:") for x in controlled) and any(
            x.startswith("arm:") for x in controlled
        ):
            raise ValueError("mixed joint/pose tracks require a qualified resource mapping")
    elif isinstance(action, JointTrajectory):
        if action.points[-1].time_from_start_s > profile.max_action_duration_s:
            raise ValueError("trajectory too long")
        for point in action.points:
            if not set(point.grippers).issubset(obs.gripper_open_fraction):
                raise ValueError("unknown gripper")
            if set(point.positions) != {j.name for j in profile.joints}:
                raise ValueError("trajectory joint names do not match profile")
            for joint in profile.joints:
                if not joint.lower <= point.positions[joint.name] <= joint.upper:
                    raise ValueError("joint limit exceeded")
    elif isinstance(action, PoseTrajectory):
        if action.arm not in obs.tcp_poses:
            raise ValueError("unknown arm")
        if action.timestep_s * len(action.points) > profile.max_action_duration_s:
            raise ValueError("pose trajectory too long")
        for point in action.points:
            if point.pose.frame not in profile.frames:
                raise ValueError("unknown pose frame")
            if (
                action.representation == "relative"
                and point.pose.frame != obs.tcp_poses[action.arm].frame
            ):
                raise ValueError("relative pose requires calibrated common frame")
            if (
                point.gripper_open_fraction is not None
                and action.arm not in obs.gripper_open_fraction
            ):
                raise ValueError("unknown gripper")
    elif isinstance(action, GripperCommand):
        if (
            action.gripper not in obs.gripper_open_fraction
            or action.duration_s > profile.max_action_duration_s
        ):
            raise ValueError("unknown gripper or excessive duration")


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
