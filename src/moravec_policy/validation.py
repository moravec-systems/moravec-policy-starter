"""Checks belong on both inference intake and the station execution boundary."""

from .contracts import CartesianGoal, Decision, JointTrajectory, Observation, RobotProfile


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
    for pose in obs.tcp_poses.values():
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
    if isinstance(action, JointTrajectory):
        if action.points[-1].time_from_start_s > profile.max_action_duration_s:
            raise ValueError("trajectory too long")
        for point in action.points:
            if set(point.positions) != {j.name for j in profile.joints}:
                raise ValueError("trajectory joint names do not match profile")
            for joint in profile.joints:
                if not joint.lower <= point.positions[joint.name] <= joint.upper:
                    raise ValueError("joint limit exceeded")
    elif isinstance(action, CartesianGoal):
        if action.arm not in obs.tcp_poses or action.target.frame not in profile.frames:
            raise ValueError("unknown arm/frame")
        if action.duration_s > profile.max_action_duration_s:
            raise ValueError("goal duration exceeds profile")
