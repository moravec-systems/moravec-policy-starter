# Third-party notices

This repository includes modified code from [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), revision `45ae74c1a427206f62d55f08f48f00295d2dc1a3`, under the [Apache License 2.0](LICENSE). Moravec modifications narrow the policy and hardware catalog, replace project documentation, remove branding and agent-tool configuration, and add a hardware-free validation example. Moravec also changes the Python namespace and adapts fetched agent imports to `moravec_policy`.

Model and agent implementation source is fetched during setup from the pinned snapshot specified in `scripts/fetch_policy.py`, rather than distributed in this repository. The fetched implementations retain their original licenses and notices. Refer to [the pinned source tree](https://github.com/XPolicyLab/XPolicyLab/tree/45ae74c1a427206f62d55f08f48f00295d2dc1a3/policy) for those files.

Model weights have separate provider terms. Keeping an adapter in this repository does not grant access to weights or establish model/hardware compatibility. Downloaded model README files and source notices remain technical references and attribution.

Adapter source provenance also includes contributions by the RoboDojo Team and, for the Astra agent, [RoboProbe](https://github.com/RoboProbe/RoboProbe). Legacy benchmark names remain in functional imports, checkpoint identifiers, and model source documentation where changing them would alter compatibility. These are not Moravec validation claims.

The MolmoAct2 adapter follows XPolicyLab's direct Transformers adapter (same pinned revision above, contributor Haoquan Fang) and Ai2's [YAM inference example](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam/host_server_yam.py). Moravec narrows this example to the official BimanualYAM checkpoint and its camera/state/action contract. Its model implementation, normalization data, weights, and original notices are downloaded together from [Ai2's immutable model snapshot](https://huggingface.co/allenai/MolmoAct2-BimanualYAM/tree/8dcbed66f2380e4393189c303ea72488eb9e63c2). Consult the provider's notices and terms before use.
