# gr00t-n1.7-3b

Unqualified integration template, not a runnable base-checkpoint demo. No GR00T inference has been run locally.

NVIDIA's current [official runtime](https://github.com/NVIDIA/Isaac-GR00T#installation) uses CUDA, Python 3.12 and an inference GPU with at least 16 GB VRAM. The starter's CPU fixture image does not supply that runtime. Pin the SDK/checkpoint revision and use the corresponding NVIDIA GPU image/environment.

N1.7 uses relative end-effector actions. The starter's current native adapter supports only qualified absolute-joint outputs, so **the base checkpoint is not compatible as-is**. Implement and test a checkpoint-specific relative-pose/controller adapter, or supply a checkpoint with a verified supported action representation. The camera/state/action bindings here are placeholders, not model metadata.

Keep `requires-qualification` until a real checkpoint passes the agreed simulation task. Include normalization, camera configuration, control rate and embodiment metadata in qualification. See [README](../../README.md) for the submission flow.
