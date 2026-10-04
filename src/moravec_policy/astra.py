"""Build a native RoboProbe invocation. Do not substitute a new LLM controller."""

from pathlib import Path

from .contracts import AstraBackend, PolicyManifest


def plan(
    manifest: PolicyManifest, root: Path, task: str, layout: int, eval_env: str, env_gpu: int
) -> dict:
    backend = manifest.backend
    if not isinstance(backend, AstraBackend):
        raise TypeError("astra-plan requires the Astra example")
    if layout < 0 or env_gpu < 0:
        raise ValueError("layout and GPU identifiers must be non-negative")
    if not task.replace("_", "").isalnum():
        raise ValueError("invalid task name")
    script = root / "policy" / backend.harness / "run_fixed_layout.sh"
    return {
        "cwd": str(root.resolve()),
        "argv": ["bash", str(script.resolve()), str(layout), str(env_gpu), task, eval_env],
        "environment": {
            "L3_INSPECT_PLANNER": "astra",
            "L3_INSPECT_API_STYLE": "responses",
            "L3_INSPECT_MODEL": backend.model,
            "L3_INSPECT_API_KEY_ENV": backend.api_key_env,
            "L3_INSPECT_REASONING_EFFORT": backend.reasoning_effort,
            "L3_INSPECT_MAX_LLM_CALLS": str(backend.max_model_calls),
            "L3_INSPECT_KEEP_ALL_IMAGES": "0",
            "L3_INSPECT_IMAGE_HORIZON": str(backend.image_history),
            "L3_INSPECT_DEPTH": "off",
            "ROBODOJO_MAX_BASH_RETRIES": "0",
            "ROBODOJO_FATAL_RESTART_COUNT": "3",
        },
        "required_environment": [backend.api_key_env, backend.base_url_env, "ROBODOJO_ROOT"],
        "upstream_revision": backend.source.revision,
        "notice": "One fixed layout; native upstream runtime required. Not an aggregate score.",
    }
