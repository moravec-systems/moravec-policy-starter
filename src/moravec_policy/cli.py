import argparse
import json
import os
from pathlib import Path

from .client_server.wire import ObservationEnvelope
from .contracts import (
    Decision,
    DeliveryManifest,
    DeploymentManifest,
    Observation,
    PolicyManifest,
    RobotProfile,
)
from .loader import load_policy
from .provenance import tree_hash


def main():
    parser = argparse.ArgumentParser(description="Moravec container policy contract")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "serve"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("manifest", type=Path)
        if name == "serve":
            cmd.add_argument("--host", default="127.0.0.1")
            cmd.add_argument("--port", type=int, default=8000)
    robot = sub.add_parser("robot")
    robot.add_argument("definition", type=Path)
    schema = sub.add_parser("schemas")
    schema.add_argument("output", type=Path)
    digest = sub.add_parser("hash-checkpoint")
    digest.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "robot":
        from .robots import load_robot

        definition = load_robot(args.definition)
        print(
            json.dumps({**definition.model_dump(), "joint_action_dim": definition.joint_action_dim})
        )
        return
    if args.command == "hash-checkpoint":
        print(tree_hash(args.directory))
        return
    if args.command == "schemas":
        args.output.mkdir(parents=True, exist_ok=True)
        for model in [
            PolicyManifest,
            RobotProfile,
            Observation,
            ObservationEnvelope,
            Decision,
            DeploymentManifest,
            DeliveryManifest,
        ]:
            (args.output / f"{model.__name__}.json").write_text(
                json.dumps(model.model_json_schema(), indent=2) + "\n"
            )
        return
    manifest = PolicyManifest.model_validate_json(args.manifest.read_text())
    if args.command == "validate":
        print(
            json.dumps(
                {"model": manifest.model.name, "schema_valid": True, "deployment_qualified": False}
            )
        )
        return
    import uvicorn

    from .client_server.service import create_app

    token = os.getenv("MORAVEC_POLICY_TOKEN", "")
    if len(token) < 24:
        parser.error("MORAVEC_POLICY_TOKEN must contain at least 24 characters")
    uvicorn.run(
        create_app(load_policy(manifest, args.manifest), manifest, token),
        host=args.host,
        port=args.port,
        ws_max_size=32 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
