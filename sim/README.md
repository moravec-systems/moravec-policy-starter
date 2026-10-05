# Simulation checks

`validate.py` runs the lightweight deterministic MuJoCo contract test. `--live-astra` enables one bounded real-model episode.

`validate_libero.py` evaluates a running real policy server on LIBERO spatial tasks 0 and 1, two initial states each. It checks the actual 20 Hz OSC controller and action scales before execution. Success comes from the environment's task predicate. Inference pauses simulation time; this does not test real-time hardware control.

Use Linux x86-64, Python 3.10, and a separate environment from the GPU policy SDK. The tested simulator is MuJoCo 3.3.2 with LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. `libero-requirements.txt` records that environment's dependencies; install PyTorch CPU wheels from the official CPU index. Install this starter and the pinned LIBERO checkout with `--no-deps` after those dependencies.

LIBERO predates PyTorch 2.6's changed deserialization default. In the pinned checkout's `libero/libero/benchmark/__init__.py`, change only `torch.load(init_states_path)` to `torch.load(init_states_path, weights_only=False)`. This is for the trusted initial-state files in that pinned repository, not user-uploaded pickle files.

With the policy server ready and `MORAVEC_POLICY_TOKEN` set:

```sh
python sim/validate_libero.py \
  --libero-root /absolute/path/to/LIBERO \
  --url ws://127.0.0.1:8000 \
  --tasks 0 1 --trials 2 \
  --output runs/libero-validation
```

Run outputs remain private. `policy_requests` counts client action requests, not GPU forward passes: a policy may buffer native predictions. Each returned delta refers to its current measured observation. The runner rejects multi-step execution rather than interpreting prediction-relative targets as measured-state deltas.

These are small compatibility trials, not leaderboard estimates or cross-simulator score comparisons. The tested Panda checkpoint does not imply YAM support. No physical robot is controlled by these scripts.
