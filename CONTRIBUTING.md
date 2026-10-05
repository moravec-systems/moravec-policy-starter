# Adding or editing a policy

Start with `policy/demo_policy/model.py` and the [integration guide](INTEGRATION.md). Copy the demo folder for a new policy, give it an exact model name, and set the same name in `deploy.yml`.

Implement observation preprocessing, inference, and action conversion in `model.py`. Keep model-specific dependencies and checkpoint instructions in the policy folder. Document camera names, joint order, units, action conventions, and the embodiment used for training. Do not treat matching array dimensions as proof that a checkpoint supports a robot.

Preserve the existing client/server protocol. Keep simulator stepping, physical control, and scoring in the evaluator. Training and data conversion are optional for an inference-only example; do not present stub scripts as implemented training.

Run `python scripts/demo_roundtrip.py` for shared wiring checks. For a learned policy, separately verify an actual checkpoint on an appropriate simulation task and record its configuration and outcome. The demo is not that validation. Keep credentials, weights, videos, and evaluation outputs out of Git.
