# Validation boundaries

Run `uv run pytest -q` for contract/adapter tests and `uv run python sim/validate.py` for the deterministic MuJoCo integration test. `--live-astra` explicitly enables at most two paid OpenAI calls for one episode.

Run-specific reports, provider IDs, usage records, videos and event traces stay in ignored `runs/` or private evaluation storage. They are not customer starter code and must not be committed. Synthetic test inputs and expected values remain in the test suite.

The MuJoCo fixture covers joint, relative-pose and LLM-generated absolute-pose paths. Other contract tests cover depth, calibration, shared-timeline bimanual action chunks, rotation conversions and temporal inputs. Tests with synthetic native SDK outputs do not qualify GR00T or OpenWAM checkpoints.

Real checkpoint inference, task success, container execution, Isaac Sim integration and physical station readiness require separate evidence. The simulator pauses during inference; this does not validate real-time hardware latency or watchdog behavior. Batch methods explicitly reject calls; disconnects abort episodes without automatic retries.
