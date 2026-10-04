# Remote endpoint mode

`client.py` demonstrates the HTTP contract against the local mock server. Set `MORAVEC_POLICY_URL` and `MORAVEC_POLICY_TOKEN` for another deployment; non-loopback endpoints require HTTPS. No retries or redirects are automatic.

Lifecycle: start episode → monotonically increasing observation sequence → action chunk → end episode. The client validates response episode/sequence identity. This is a model inference connection, not an actuator or RoboDojo environment client. For live simulation use the native XPolicyLab integration, or implement an environment consumer with explicit timing and stale-action rejection.

Before sending customer RGB to a remote provider, configure approved egress and data retention. The token is transmitted as a bearer header and never included in a URL or manifest. Production TLS termination, request-size/rate limits, tenant isolation and secret rotation belong in the hosting layer; the reference server alone is not an internet-facing deployment.
