# Third-party notices

Robot dimension definitions (except YAM) are adapted from XPolicyLab `utils/robot/_robot_info.json`, revision `bb9a0b5f5136a74503b679af830bfd0a3a837d5c`, licensed under Apache-2.0; see [license](licenses/Apache-2.0.txt). IDs use lowercase dashes. Policy-folder organization and OpenWAM baseline inference settings follow its model/deployment examples. Source: https://github.com/XPolicyLab/XPolicyLab.

YAM joint ordering and dimensions reference I2RT's `i2rt/robots/config/yam_v1.yml` and `i2rt/robot_models/arm/yam/v1/yam.urdf`, local revision `3aa31b19dd0e55413880605013e5d156953f9baf`. Copyright (c) I2RT Robotics; see [MIT license](licenses/I2RT-MIT.txt). Dual-arm joint prefixes are Moravec logical names. Motor settings, hardware calibration, URDFs, and meshes are not bundled.

These descriptions do not imply a trained checkpoint or a validated hardware connection for each robot. Model weights and SDKs retain their own licenses and are not included.

The GR00T LIBERO recipe uses NVIDIA GR00T N1.7 SDK source (Copyright NVIDIA CORPORATION & AFFILIATES, Apache-2.0) from the pinned source tree identified in its Dockerfile. LIBERO camera, state, action, and gripper conventions follow its `gr00t/eval/sim/LIBERO/libero_env.py`. The prepared runtime changes model/processor configuration paths only, not model tensors. NVIDIA weights and Cosmos weights retain their separate model licenses and are downloaded only by the operator.

The LIBERO-trained OpenWAM adapter follows the checkpoint's native EEF10 representation and deployment settings in the matching OpenWAM source within XPolicyLab revision `bb9a0b5f5136a74503b679af830bfd0a3a837d5c`, `policy/OpenWAM/OpenWAM` (Apache-2.0). The original SDK source and model weights are not bundled in this repository.
