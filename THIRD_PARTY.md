# Third-party notices

The shared client/server transport, model template, and data utilities include modified code from [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), revision `45ae74c1a427206f62d55f08f48f00295d2dc1a3`, under the [Apache License 2.0](LICENSE). Moravec modifications narrow the policy and hardware catalog, replace project documentation, remove branding and agent-tool configuration, change the Python namespace to `moravec_policy`, and add a hardware-free validation example. The GR00T, π0.5, OpenWAM, and MolmoAct2 adapters (`model.py`) began from XPolicyLab's adapters for those models.

Model implementations are not distributed in this repository. Each policy README installs the model from its original source at a pinned revision, and those sources retain their own licenses and notices:

| Example | Source | Revision |
| --- | --- | --- |
| GR00T_N17 | [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `23ace64f17aa5015259b8609d371eb61a357c776` |
| Pi_05 | [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| OpenWAM | [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) | `48bd67b89d489b14d03b8d92bc66e65d306df32e` |
| MolmoAct2 | [Ai2 MolmoAct2-BimanualYAM](https://huggingface.co/allenai/MolmoAct2-BimanualYAM/tree/8dcbed66f2380e4393189c303ea72488eb9e63c2) | `8dcbed66f2380e4393189c303ea72488eb9e63c2` |

The MolmoAct2 adapter also follows Ai2's [YAM inference example](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam/host_server_yam.py) and is narrowed to the official BimanualYAM checkpoint and its camera/state/action contract. Its snapshot includes executable model source loaded with `trust_remote_code=True`.

Model weights have separate provider terms. Keeping an adapter in this repository does not grant access to weights or establish model/hardware compatibility. Third-party checkpoint and dataset identifiers (for example `OpenWAM-Alpha-Sim-RoboDojo` or `cjgogo/RoboDojo-pi05-checkpoints`) are kept as published; they are not Moravec validation claims.
