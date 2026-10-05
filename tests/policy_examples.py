"""Load the actual example implementations without adding model SDK dependencies."""

from pathlib import Path

from moravec_policy.loader import load_entrypoint

ROOT = Path(__file__).resolve().parents[1]
LLMHarness = load_entrypoint("policy.py:LLMHarness", ROOT / "policies/gpt-6-astra/policy.json")
FixturePolicy = load_entrypoint(
    "policy.py:FixturePolicy", ROOT / "policies/contract-fixture/policy.json"
)
AstraModule = __import__(LLMHarness.__module__, fromlist=["tool_definitions"])
tool_definitions = AstraModule.tool_definitions
