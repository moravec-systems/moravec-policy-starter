import argparse
import json
import os
from pathlib import Path

from .contracts import (
    Decision,
    DeliveryManifest,
    DeploymentManifest,
    Observation,
    PolicyManifest,
    RobotProfile,
)
from .policies import load_policy
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
    schema = sub.add_parser("schemas")
    schema.add_argument("output", type=Path)
    digest = sub.add_parser("hash-checkpoint")
    digest.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "hash-checkpoint":
        print(tree_hash(args.directory))
        return
    if args.command == "schemas":
        args.output.mkdir(parents=True, exist_ok=True)
        for model in [
            PolicyManifest,
            RobotProfile,
            Observation,
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

    from .service import create_app

    token = os.getenv("MORAVEC_POLICY_TOKEN", "")
    if len(token) < 24:
        parser.error("MORAVEC_POLICY_TOKEN must contain at least 24 characters")
    uvicorn.run(
        create_app(load_policy(manifest, args.manifest), manifest, token),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
