# Model families and deployment roles

| Family/role | Example | What must reach the executor |
| --- | --- | --- |
| LLM / vision-enabled language model | Astra + RoboProbe | Validated tool calls converted by a declared non-learned planner |
| VLA | NVIDIA GR00T N1.7 | Native action chunks after checkpoint-specific preprocessing/postprocessing |
| World-action model | OpenWAM | Native action chunks; predicted video is optional internal model output |
| Imitation/diffusion policy | Future ACT/DP adapter | Same direct-action contract; model family does not require a new wire protocol |
| Pure video/state world model | Future component | Predictions alone are not robot actions; needs candidate-action generation/selection or another controller |
| Perception-only VLM | Future component | Perception outputs need a declared downstream decision/controller component |

The three working integration paths cover API reasoning, learned direct actions and world-action inference. This repository does not claim every model/embodiment is supported. A pure world-model prediction endpoint would be a separate schema/service composed inside a policy system; don't coerce predicted frames into ActionChunk or treat them as simulator evidence.

To add a family, first ask what observations it needs, what it emits, which component turns outputs into actions, how state resets, and what exact components must be pinned for an attributable score. Keep complete system identity even if a leaderboard row uses a short model label.
