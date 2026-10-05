# Build the shared base first; for GPU policies it must include the policy SDK.
# The policy owns this recipe: add pinned SDK dependencies or custom serving code here.
ARG BASE_IMAGE=moravec-policy-base:local
FROM ${BASE_IMAGE}
CMD ["moravec-policy", "serve", "policies/contract-fixture/policy.json", "--host", "0.0.0.0"]
