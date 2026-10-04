# gr00t-n1.7-3b

Unqualified SDK adapter template. No real GR00T weights or GPU inference have been tested here.

Use a pinned [NVIDIA-supported CUDA runtime](https://github.com/NVIDIA/Isaac-GR00T#installation) and checkpoint. The default CPU container is not that runtime. Populate the checkpoint tree hash, embodiment tag and bindings from real metadata; the checked-in values are placeholders.

The adapter now supports temporal RGB/state input and absolute-joint or absolute/relative end-effector output. Bind translation/rotation/gripper streams explicitly. Supported rotation conversions: WXYZ, rotation vector, XYZ Euler radians, and 6D rows or columns. Specify observation cadence, frame/tool-relative motion, observation/previous-point reference, action timestep and execution horizon. Bimanual pose outputs can use `pose_bindings` to create simultaneous tracks.

SDK outputs must already be denormalized, with grippers mapped to 0 closed / 1 open. Do not guess these semantics from tensor dimensions. Keep `requires-qualification` until the actual checkpoint passes the agreed simulation task. Converter tests do not establish model or embodiment compatibility.
