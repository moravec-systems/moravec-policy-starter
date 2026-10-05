# gpt-6-astra

Direct OpenAI Responses API example. Supply `OPENAI_API_KEY` at runtime, then run from the repository root:

```sh
uv run python sim/validate.py --live-astra --output runs/astra-live
```

At most two paid calls, one MuJoCo episode, no model substitution. The fixture target is numeric; this validates inference/action integration rather than a customer benchmark. Check the root VALIDATION.md for actual execution status.

The implementation and deployment settings are owned by this folder: edit `policy.py`, `policy.json` and `Dockerfile`. The launcher does not special-case this model.
