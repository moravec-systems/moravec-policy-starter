"""Validate a policy submission: its moravec-policy.yaml manifest and, optionally, the running policy.

  moravec-validate policy/demo_policy                         # manifest and files only
  moravec-validate policy/demo_policy --url ws://127.0.0.1:6000   # plus a running server
  moravec-validate policy/demo_policy --run --image moravec-demo-policy:local   # start the image, then test it

Add --submit to require a digest-pinned image, and --json for a machine-readable report.
Exits 0 when no check fails. Nothing here executes actions on hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "moravec-policy.v1.json"
ROBOTS_PATH = Path(__file__).resolve().parent / "utils" / "robot" / "_robot_info.json"
DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
SECRET_PATTERNS = {
    "OpenAI key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
SCANNED_SUFFIXES = {".py", ".yml", ".yaml", ".json", ".toml", ".txt", ".md", ".sh", ".cfg", ".env", ""}


@dataclass
class Check:
    id: str
    status: str  # pass | fail | warn | skip
    message: str
    code: str | None = None  # stable identifier for failures and warnings
    fix: str | None = None


@dataclass
class Report:
    manifest: str
    checks: list[Check] = field(default_factory=list)

    def add(self, id, status, message, code=None, fix=None):
        self.checks.append(Check(id, status, message, code, fix))
        return status != "fail"

    @property
    def ok(self) -> bool:
        return not any(check.status == "fail" for check in self.checks)

    def to_dict(self) -> dict:
        counts = {status: sum(c.status == status for c in self.checks) for status in ("pass", "fail", "warn", "skip")}
        return {"schema": "moravec.validation/v1", "manifest": self.manifest, "ok": self.ok,
                "summary": counts, "checks": [asdict(c) for c in self.checks]}


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #

def load_manifest(path: Path, report: Report) -> dict | None:
    try:
        manifest = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        report.add("manifest.parse", "fail", f"cannot read {path}: {error}", "MANIFEST_UNREADABLE",
                   "Make the manifest valid YAML.")
        return None
    try:
        import jsonschema
    except ImportError:
        report.add("manifest.schema", "fail", "jsonschema is not installed", "VALIDATOR_DEPENDENCY",
                   "pip install -e . (the starter depends on jsonschema).")
        return None
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
    if errors:
        for error in errors:
            where = "/".join(str(part) for part in error.path) or "(root)"
            report.add("manifest.schema", "fail", f"{where}: {error.message}", "MANIFEST_SCHEMA",
                       f"Fix '{where}' to match schemas/{SCHEMA_PATH.name}.")
        return None
    report.add("manifest.schema", "pass", f"matches {SCHEMA_PATH.name}")
    return manifest


def static_checks(manifest: dict, manifest_dir: Path, submit: bool, report: Report) -> None:
    if DIGEST_RE.search(manifest["image"]):
        report.add("image.pinned", "pass", "image is pinned by sha256 digest")
    else:
        report.add("image.pinned", "fail" if submit else "warn",
                   f"image '{manifest['image']}' is not pinned by digest", "IMAGE_NOT_PINNED",
                   "Push the image and use registry/repository@sha256:<digest>.")

    interface = manifest["interface"]
    if interface.get("format") != "native":
        robots = json.loads(ROBOTS_PATH.read_text())
        if interface["robot"] in robots:
            report.add("interface.robot", "pass", f"robot '{interface['robot']}' is registered")
        else:
            report.add("interface.robot", "fail", f"unknown robot '{interface['robot']}'", "UNKNOWN_ROBOT",
                       f"Use one of {sorted(robots)}, or add the robot to utils/robot/_robot_info.json.")
    else:
        check = manifest_dir / interface["check"]
        if check.is_file():
            report.add("interface.check", "pass", f"native check script {interface['check']} exists")
        else:
            report.add("interface.check", "fail", f"native check script {check} not found", "FILE_MISSING",
                       "Point interface.check at a script that tests a running server.")

    if (manifest_dir / "Dockerfile").is_file():
        report.add("files.dockerfile", "pass", "Dockerfile found beside the manifest")
    else:
        report.add("files.dockerfile", "warn", "no Dockerfile beside the manifest", "FILE_MISSING",
                   "Keep the Dockerfile that builds this image next to the manifest.")
    command = manifest["serve"]["command"]
    if "--config_path" in command:
        config = command[command.index("--config_path") + 1] if command.index("--config_path") + 1 < len(command) else ""
        candidates = [manifest_dir / config, manifest_dir.parent.parent / config]
        if any(path.is_file() for path in candidates):
            report.add("files.config", "pass", f"serve config {config} found")
        else:
            report.add("files.config", "fail", f"serve config {config} not found", "FILE_MISSING",
                       "serve.command's --config_path must name a file in the policy repository.")



def scan_secrets(manifest_dir: Path, report: Report) -> None:
    leaks = []
    for path in manifest_dir.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES or path.stat().st_size > 1_000_000:
            continue
        if {"checkpoints", ".venv", "__pycache__", ".git"} & set(path.parts):
            continue
        text = path.read_text(errors="ignore")
        leaks += [f"{path.relative_to(manifest_dir)} ({kind})" for kind, pattern in SECRET_PATTERNS.items()
                  if pattern.search(text)]
    if leaks:
        report.add("files.secrets", "fail", "possible credentials in " + ", ".join(leaks), "POSSIBLE_SECRET",
                   "Remove credentials from files; list variable names under 'secrets' and pass values at run time.")
    else:
        report.add("files.secrets", "pass", "no credentials found in the policy folder")

# --------------------------------------------------------------------------- #
# Running the image
# --------------------------------------------------------------------------- #

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_container(manifest: dict, image: str, docker_args: list[str], report: Report):
    if shutil.which("docker") is None:
        report.add("run.docker", "fail", "docker is not installed", "DOCKER_MISSING", "Install Docker to use --run.")
        return None, None
    missing = [name for name in manifest.get("secrets", []) if name not in os.environ]
    if missing:
        report.add("run.secrets", "fail", f"secrets not set in this environment: {', '.join(missing)}",
                   "SECRET_MISSING", "Export each listed variable before running; values are passed by name.")
        return None, None
    port = _free_port()
    resources = manifest["resources"]
    cmd = ["docker", "run", "-d", "-p", f"127.0.0.1:{port}:{manifest['serve']['port']}"]
    if resources["gpu_count"]:
        cmd += ["--gpus", str(resources["gpu_count"])]
    if resources.get("shm_size_gb"):
        cmd += [f"--shm-size={resources['shm_size_gb']}g"]
    for name in manifest.get("secrets", []):
        cmd += ["-e", name]
    cmd += [*docker_args, image, *manifest["serve"]["command"]]
    started = subprocess.run(cmd, capture_output=True, text=True)
    if started.returncode != 0:
        report.add("run.start", "fail", f"docker run failed: {started.stderr.strip()[-400:]}", "CONTAINER_START",
                   "Check that the image exists locally or can be pulled, and that resources are available.")
        return None, None
    report.add("run.start", "pass", f"started {image} (port {port})")
    return started.stdout.strip(), f"ws://127.0.0.1:{port}"


def stop_container(container: str | None) -> None:
    if container:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


def wait_ready(url: str, timeout_s: int, container: str | None, report: Report) -> bool:
    from websockets.sync.client import connect

    deadline, last = time.monotonic() + timeout_s, ""
    while time.monotonic() < deadline:
        try:
            connect(url, open_timeout=3).close()
            return report.add("serve.ready", "pass", f"policy server accepted a connection at {url}")
        except Exception as error:  # still loading
            last = str(error)
        if container and subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container],
                                        capture_output=True, text=True).stdout.strip() != "true":
            logs = subprocess.run(["docker", "logs", "--tail", "20", container], capture_output=True, text=True)
            return report.add("serve.ready", "fail", "container exited before serving: " + (logs.stderr or logs.stdout)[-600:],
                              "SERVER_EXITED", "Run the image by hand and fix the startup error.")
        time.sleep(2)
    return report.add("serve.ready", "fail", f"no server at {url} after {timeout_s}s ({last})", "SERVER_NOT_READY",
                      "Raise serve.startup_timeout_s or fix the server start command and port.")


# --------------------------------------------------------------------------- #
# Contract checks
# --------------------------------------------------------------------------- #

def synthetic_observation(interface: dict, robot: dict, env_idx: int = 0) -> dict:
    """An observation in the declared format; values are plausible, never meaningful."""
    rng = np.random.default_rng(env_idx)
    state = {}
    for index, side in enumerate(("left", "right")[: len(robot["arm_dim"])]):
        prefix = f"{side}_" if len(robot["arm_dim"]) > 1 else ""
        state[f"{prefix}arm_joint_state"] = np.zeros(robot["arm_dim"][index], dtype=np.float32)
        state[f"{prefix}ee_pose"] = np.array([0.0, 0.3 * (1 - 2 * index), 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state[f"{prefix}ee_joint_state"] = np.ones(robot["ee_dim"][index], dtype=np.float32)
    vision = {name: {"color": rng.integers(0, 256, tuple(shape), dtype=np.uint8)}
              for name, shape in interface["cameras"].items()}
    return {"env_idx": env_idx, "instruction": interface.get("instruction", "Pick up the block."),
            "state": state, "vision": vision}


def check_chunk(chunk, interface: dict, robot: dict, report: Report, label: str) -> bool:
    if not isinstance(chunk, list) or not chunk:
        return report.add(f"{label}.chunk", "fail", f"expected a non-empty list of actions, got {type(chunk).__name__}",
                          "ACTION_CHUNK_TYPE", "get_action must return a list of action dicts.")
    expected = interface["chunk_steps"]
    if expected != "variable" and len(chunk) != expected:
        return report.add(f"{label}.chunk", "fail", f"{len(chunk)} actions returned, interface declares {expected}",
                          "ACTION_CHUNK_LENGTH", "Return chunk_steps actions, or set chunk_steps to the real length.")
    sides = ("left", "right")[: len(robot["arm_dim"])]
    expected_keys = {}
    for index, side in enumerate(sides):
        prefix = f"{side}_" if len(sides) > 1 else ""
        if interface["action_type"] == "joint":
            expected_keys[f"{prefix}arm_joint_state"] = robot["arm_dim"][index]
        else:
            expected_keys[f"{prefix}ee_pose"] = 7
        expected_keys[f"{prefix}ee_joint_state"] = robot["ee_dim"][index]
    gripper_out = 0
    for step, action in enumerate(chunk):
        if not isinstance(action, dict) or set(action) != set(expected_keys):
            got = sorted(action) if isinstance(action, dict) else type(action).__name__
            return report.add(f"{label}.keys", "fail", f"step {step} has keys {got}, expected {sorted(expected_keys)}",
                              "ACTION_KEYS", f"Return exactly the {interface['action_type']} action keys for this robot.")
        for key, size in expected_keys.items():
            value = np.asarray(action[key], dtype=np.float64)
            if value.shape != (size,):
                return report.add(f"{label}.shapes", "fail", f"step {step} {key} has shape {value.shape}, expected ({size},)",
                                  "ACTION_SHAPE", "Match the robot's dimensions in utils/robot/_robot_info.json.")
            if not np.isfinite(value).all():
                return report.add(f"{label}.finite", "fail", f"step {step} {key} contains NaN or inf",
                                  "ACTION_NOT_FINITE", "Every action value must be finite.")
            if key.endswith("ee_joint_state") and ((value < -1e-3) | (value > 1 + 1e-3)).any():
                gripper_out += 1
            if key.endswith("ee_pose") and abs(np.linalg.norm(value[3:]) - 1) > 1e-3:
                return report.add(f"{label}.quaternion", "fail", f"step {step} {key} quaternion has norm {np.linalg.norm(value[3:]):.4f}",
                                  "QUATERNION_NOT_UNIT", "Return unit quaternions as [qw, qx, qy, qz].")
    report.add(f"{label}.actions", "pass", f"{len(chunk)} actions with keys {sorted(expected_keys)}, correct shapes, finite")
    if gripper_out:
        report.add(f"{label}.gripper", "warn", f"{gripper_out} gripper values outside [0, 1]", "GRIPPER_RANGE",
                   "Gripper commands are 0 closed to 1 open; the evaluator clips values outside that range.")
    return True


def contract_checks(manifest: dict, manifest_dir: Path, url: str, report: Report) -> None:
    interface = manifest["interface"]
    if interface.get("format") == "native":
        script = manifest_dir / interface["check"]
        result = subprocess.run([sys.executable, str(script), "--url", url], capture_output=True, text=True)
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        report.add("native.check", "pass" if result.returncode == 0 else "fail",
                   f"{interface['check']}: " + " | ".join(tail), None if result.returncode == 0 else "NATIVE_CHECK_FAILED",
                   None if result.returncode == 0 else "Run the check script by hand against the server and fix the failure.")
        return

    from client_server.model_client import WsModelClient

    robot = json.loads(ROBOTS_PATH.read_text())[interface["robot"]]
    timeout = max(60, manifest["serve"]["startup_timeout_s"])
    try:
        with WsModelClient(url=url, evaluation_id="moravec-validate", trial_id="contract",
                           request_timeout_s=timeout) as client:
            client.call("reset")
            report.add("contract.reset", "pass", "reset accepted")
            latencies = []
            for call in range(3):  # the first call is fully checked; the others are timed
                client.call("update_obs", obs=synthetic_observation(interface, robot))
                started = time.monotonic()
                chunk = client.call("get_action")
                latencies.append(time.monotonic() - started)
                if call == 0 and not check_chunk(chunk, interface, robot, report, "contract"):
                    return
            report.add("contract.latency", "pass", f"get_action took {latencies[0] * 1000:.0f} ms on the first call "
                       f"(includes warm-up) and {np.median(latencies[1:]) * 1000:.0f} ms median on the next two")
            if interface["batching"]:
                client.call("update_obs_batch", obs=[synthetic_observation(interface, robot, 0),
                                                     synthetic_observation(interface, robot, 1)])
                chunks = client.call("get_action_batch", obs=[1, 0])
                if not isinstance(chunks, list) or len(chunks) != 2:
                    report.add("contract.batch", "fail", "get_action_batch([1, 0]) did not return two chunks",
                               "BATCH_UNSUPPORTED", "Implement batching, or set interface.batching to false.")
                else:
                    scratch = Report(report.manifest)  # report batch chunks only if they fail
                    if all(check_chunk(c, interface, robot, scratch, "contract.batch") for c in chunks):
                        report.add("contract.batch", "pass", "batched observations and actions for two environments")
                    else:
                        report.checks += [c for c in scratch.checks if c.status == "fail"]
            client.call("reset")
            report.add("contract.reset_after", "pass", "reset after actions accepted")
    except Exception as error:
        report.add("contract.call", "fail", f"{type(error).__name__}: {str(error)[:400]}", "POLICY_CALL_FAILED",
                   "The server rejected or failed a call; check its logs for the traceback.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def validate(target: Path, url: str | None = None, run: bool = False, image: str | None = None,
             docker_args: list[str] | None = None, submit: bool = False) -> Report:
    manifest_path = target / "moravec-policy.yaml" if target.is_dir() else target
    report = Report(str(manifest_path))
    manifest = load_manifest(manifest_path, report)
    scan_secrets(manifest_path.parent, report)  # always, even when the manifest is invalid
    if manifest is None:
        return report
    static_checks(manifest, manifest_path.parent, submit, report)
    if not (url or run):
        report.add("contract", "skip", "no running policy tested; add --url or --run")
        return report
    container = None
    try:
        if run:
            container, url = start_container(manifest, image or manifest["image"], docker_args or [], report)
            if container is None:
                return report
        if wait_ready(url, manifest["serve"]["startup_timeout_s"], container, report):
            contract_checks(manifest, manifest_path.parent, url, report)
    finally:
        stop_container(container)
    return report


def print_report(report: Report) -> None:
    marks = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}
    print(f"Validating {report.manifest}")
    for check in report.checks:
        print(f"  {marks[check.status]}  {check.id}: {check.message}")
        if check.status in ("fail", "warn") and check.fix:
            print(f"        {check.code}: {check.fix}")
    summary = report.to_dict()["summary"]
    print(("OK" if report.ok else "FAILED") + f" ({summary['pass']} passed, {summary['fail']} failed, "
          f"{summary['warn']} warnings, {summary['skip']} skipped)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="moravec-validate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="policy folder containing moravec-policy.yaml, or the manifest itself")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--url", help="test an already running policy server, e.g. ws://127.0.0.1:6000")
    mode.add_argument("--run", action="store_true", help="start the image with docker, test it, then remove it")
    parser.add_argument("--image", help="with --run, use this local image instead of the manifest's")
    parser.add_argument("--docker-arg", action="append", default=[],
                        help="extra docker run argument, e.g. --docker-arg=-v --docker-arg=/data/weights:/weights:ro")
    parser.add_argument("--submit", action="store_true", help="apply submission rules: the image must be pinned by digest")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    if args.image and not args.run:
        parser.error("--image requires --run")
    report = validate(args.target, args.url, args.run, args.image, args.docker_arg, args.submit)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
