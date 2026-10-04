# Deployment examples

## Local development

Use `uv sync --locked` for the lightweight SDK. No real model dependencies are installed. The default server binds loopback; inject a bearer token through the environment. `Dockerfile` packages the same CPU demo as a non-root user. `.dockerignore` allowlists only required files, excluding private credentials, weights and the other workspace projects.

## GPU container

Build one image per model using its upstream CUDA stack. Install this SDK into that environment and pin the XPolicyLab revision. Put checkpoint assets on a dedicated mount, not in the source repo. GR00T's upstream loader can temporarily edit processor configuration, so its working copy may need write access; retain an immutable original and record its hash. Never combine the three inference stacks into one dependency environment merely because the JSON contract is shared.

The deployment record should use the final image digest, a policy-manifest digest and secret reference. An illustrative shape (replace placeholders before schema validation):

```json
{
  "schema_version": "moravec.deployment/v0.1",
  "policy_id": "groot-n17-arx-x5",
  "policy_manifest_sha256": "<64-character SHA256>",
  "backend": {
    "kind": "container",
    "image": "registry.example/policy@sha256:<64-character digest>",
    "token_secret_ref": "moravec/policies/groot/token",
    "gpu_count": 1
  }
}
```

For remote use, the backend is `{"kind":"remote","endpoint":"https://policy.example","token_secret_ref":"moravec/policies/customer/token"}`. This HTTP reference server does not implement TLS, multi-tenant isolation, cloud orchestration, deadlines, spending guards or artifact retention. Run it behind an appropriate gateway before exposing it outside a trusted development environment.

## Native benchmark execution

GR00T/OpenWAM remain compatible with their upstream `eval.sh` and split server/client scripts. Astra uses `astra-plan --execute` with the native fixed-layout runner. Neither native path is secretly routed through the sample HTTP API. This preserves working benchmark integration while providing a clear customer-access contract to develop against.

The simulator still owns task reset, time, execution and scoring. Do not infer a benchmark's evaluation workload from the policy server's readiness. A separate evaluator must freeze trial selections and collect finalized native results.
