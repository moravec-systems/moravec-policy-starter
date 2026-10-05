# GR00T_N17 — LIBERO checkpoint

Released NVIDIA LIBERO-spatial checkpoint, using the shared Moravec client/server. This is a task-trained example; it does not qualify the base GR00T checkpoint or YAM hardware.

Inputs: two 256×256 RGB cameras (`overview`, `wrist`), instruction, achieved end-effector position and rotation, and both measured finger joint positions. The simulation client rotates LIBERO camera images 180 degrees before sending them. Rotation state is axis-angle, despite the SDK's `roll`, `pitch`, `yaw` key names.

Outputs: 16 predicted steps, buffering the first 8 internally and returning one step per fresh observation at 20 Hz. The SDK replans after those 8 steps. Native normalized OSC commands become physical relative targets: translation × 0.05 m, rotation vector × 0.5 rad. Each returned step is relative to its current measured observation in world axes. Gripper values use the SDK's binary threshold, with 0 closed / 1 open on the wire. These settings require LIBERO's matching controller; they are not hardware defaults.

`policy.json` pins a prepared checkpoint checksum and the Cosmos dependency checksum. Preparation changes only two configuration fields to point at the fixed local Cosmos mount; model tensors remain unchanged. Both downloaded repositories are pinned to immutable revisions in `prepare_weights.py`.

Build from the repository root:

```sh
docker build -t moravec-policy-base:local .
docker build -f policies/GR00T_N17/libero/Dockerfile -t moravec-gr00t .
```

On a machine with `huggingface-hub` and this package installed, accept the upstream licenses and set `HF_TOKEN` for downloading, then prepare a new private weights directory:

```sh
python policies/GR00T_N17/libero/prepare_weights.py --output weights/gr00t-libero
```

Serve on a Linux NVIDIA host with the NVIDIA Container Toolkit. Set `MORAVEC_POLICY_TOKEN` in the shell. The model container uses no Hugging Face token, runs as an unprivileged user, and loads its dependencies offline:

```sh
docker run --rm --gpus all \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true \
  -p 127.0.0.1:8000:8000 -e MORAVEC_POLICY_TOKEN \
  -v "$PWD/weights/gr00t-libero:/weights:ro" moravec-gr00t
```

Use `sim/validate_libero.py` from a separate simulation environment. The container has a CPU Python base but installs the pinned CUDA-enabled PyTorch and GR00T runtime; the base image alone cannot serve this model. The recipe targets Linux x86-64 and Python 3.10. See the root validation report for tested hardware and remaining limits.
