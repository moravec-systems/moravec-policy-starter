# Sources and derivation

Pinned revisions are in `sources.lock.json`; this sample does not vendor their code or weights.

- XPolicyLab demo interface: https://github.com/XPolicyLab/XPolicyLab/blob/bb9a0b5f5136a74503b679af830bfd0a3a837d5c/policy/demo_policy/model.py
- GR00T adapter/config/install: https://github.com/XPolicyLab/XPolicyLab/tree/bb9a0b5f5136a74503b679af830bfd0a3a837d5c/policy/GR00T_N17
- NVIDIA model source: https://github.com/NVIDIA/Isaac-GR00T
- OpenWAM adapter: https://github.com/XPolicyLab/XPolicyLab/tree/bb9a0b5f5136a74503b679af830bfd0a3a837d5c/policy/OpenWAM
- OpenWAM official source: https://github.com/OpenWAM-Official/OpenWAM
- RoboProbe harness: https://github.com/RoboProbe/RoboProbe/tree/52300752080325086095854e01977a8cf281b684/policy/RoboDojo_Agent_L3_Inspect_EEF
- Harness disclosure rules: https://github.com/RoboProbe/RoboProbe/blob/52300752080325086095854e01977a8cf281b684/docs/minimal_harness.md
- Observation formats: https://github.com/RoboProbe/RoboProbe/blob/52300752080325086095854e01977a8cf281b684/docs/data_formats.md

Inherited: update_obs/get_action/reset adapter contract, native observation/action dictionaries, GR00T/OpenWAM preprocessing, RoboProbe's controller and fixed-layout runner.

Moravec proposal: JSON manifests, typed HTTP envelopes, session handling, source/checkpoint checks, API authentication, validation gates and repository layout. These are not claimed to be upstream standards or an already deployed product. The intended product supports a container or remote integration plus a simulation validation sandbox; this repository is the local integration starter for that flow.
