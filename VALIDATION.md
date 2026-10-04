# Local validation — October 4, 2026

- 27 tests passed on Python 3.10.19 (the GR00T vendored runtime requires Python 3.10).
- Initial 24-test suite also passed on Python 3.12.12 before lowering the SDK minimum for upstream compatibility.
- Ruff checks and formatting passed.
- All four policy manifests and the evaluation example parsed successfully.
- JSON Schemas generated from the actual Pydantic contracts.
- Real localhost HTTP server/client round trip passed, returning one explicitly labeled mock action. Server stopped afterward.
- Astra launch plan resolves the pinned native fixed-layout script and exact `gpt-6-astra` model; no paid request or simulation launched.
- Missing real checkpoints reject startup; no mock fallback. Tests cover source drift, RGB order, action validation, episode resets, auth, duplicate requests and error sanitization.

## Not verified

- No Astra inference or GR00T/OpenWAM weights loaded from this starter.
- No GPU/simulator or physical-robot validation performed for these examples.
- Docker build not run: installed Docker CLI cannot connect to the local daemon.
- A Starlette TestClient deprecation warning recommends httpx2; current HTTP tests pass with the locked httpx dependency. The production client uses httpx directly.

The earlier successful Opus/RoboProbe trial belongs to a separate evaluation artifact. It is not an Astra, GR00T or OpenWAM result and is not copied into this repo as evidence.
