"""Reusable embodiment dimensions; these are not calibrated hardware deployments."""

from pathlib import Path

from pydantic import Field, model_validator

from .contracts import Contract, Slug


class RobotDefinition(Contract):
    id: Slug
    description: str
    arm_dim: list[int] = Field(min_length=1)
    ee_dim: list[int] = Field(min_length=1)
    joint_names: list[list[str]] | None = None

    @model_validator(mode="after")
    def dimensions(self):
        if len(self.arm_dim) != len(self.ee_dim):
            raise ValueError("each arm must declare an end-effector dimension")
        if any(n <= 0 for n in self.arm_dim) or any(n < 0 for n in self.ee_dim):
            raise ValueError("invalid robot dimensions")
        if self.joint_names is not None:
            if [len(names) for names in self.joint_names] != self.arm_dim:
                raise ValueError("joint names must match arm dimensions")
            names = [name for arm in self.joint_names for name in arm]
            if len(set(names)) != len(names):
                raise ValueError("joint names must be unique across the robot")
        return self

    @property
    def joint_action_dim(self):
        return sum(self.arm_dim) + sum(self.ee_dim)


def load_robot(path):
    return RobotDefinition.model_validate_json(Path(path).read_text())
