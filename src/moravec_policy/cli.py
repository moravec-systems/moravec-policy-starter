from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .adapters import load_policy, verify_checkout
from .astra import plan
from .contracts import ActionChunk, DeploymentManifest, EvaluationSpec, Observation, PolicyManifest
from .provenance import tree_hash


def read_manifest(path: Path) -> PolicyManifest:
    return PolicyManifest.model_validate_json(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Local Moravec policy starter")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    digest = sub.add_parser("hash-checkpoint")
    digest.add_argument("directory", type=Path)
    schema = sub.add_parser("schemas")
    schema.add_argument("output", type=Path)
    serve = sub.add_parser("serve")
    serve.add_argument("manifest", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--timestep", type=float, default=0.05)
    astra = sub.add_parser("astra-plan")
    astra.add_argument("manifest", type=Path)
    astra.add_argument("--root", type=Path, required=True)
    astra.add_argument("--task", default="stack_bowls")
    astra.add_argument("--layout", type=int, default=0)
    astra.add_argument("--env-gpu", type=int, default=0)
    astra.add_argument("--eval-env", default="uv")
    astra.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "hash-checkpoint":
        print(tree_hash(args.directory))
        return
    if args.command == "schemas":
        args.output.mkdir(parents=True, exist_ok=True)
        for model in (PolicyManifest, DeploymentManifest, EvaluationSpec, Observation, ActionChunk):
            (args.output / f"{model.__name__}.json").write_text(
                json.dumps(model.model_json_schema(), indent=2) + "\n"
            )
        return
    manifest = read_manifest(args.manifest)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "id": manifest.id,
                    "schema_valid": True,
                    "live_inference_tested": False,
                    "note": "Schema validation is not deployment readiness.",
                },
                indent=2,
            )
        )
    elif args.command == "serve":
        import uvicorn

        from .service import create_app

        token = os.environ.get("MORAVEC_POLICY_TOKEN", "")
        if len(token) < 24 or not 0 < args.timestep <= 1:
            parser.error("require MORAVEC_POLICY_TOKEN >=24 characters and timestep in (0,1]")
        policy = load_policy(manifest, args.manifest.resolve(), args.timestep)
        uvicorn.run(create_app(policy, manifest, token), host=args.host, port=args.port)
    elif args.command == "astra-plan":
        invocation = plan(manifest, args.root, args.task, args.layout, args.eval_env, args.env_gpu)
        if not args.execute:
            print(json.dumps(invocation, indent=2))
            return
        missing = [name for name in invocation["required_environment"] if not os.getenv(name)]
        if missing:
            parser.error("missing environment variables: " + ", ".join(missing))
        verify_checkout(args.root, invocation["upstream_revision"])
        if not Path(invocation["argv"][1]).is_file():
            parser.error("reference runner not found")
        env = {k: v for k, v in os.environ.items() if not k.startswith("L3_INSPECT_")}
        env.update(invocation["environment"])
        env["L3_INSPECT_BASE_URL"] = env[manifest.backend.base_url_env]
        # Native provider URL conventions are retained; see examples/astra/README.md.
        subprocess.run(invocation["argv"], cwd=invocation["cwd"], env=env, check=True)


if __name__ == "__main__":
    main()
