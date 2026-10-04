# Customer requirements and deployment boundaries

Reviewed the Current tab of the [customer SOW draft](https://docs.google.com/document/d/1RHiKXrwHbefxVX_H6vHYyIQAVLGJ1Ki6FQyxHRjLCKs/edit?tab=t.koun2la6k0k8) on 2026-10-04. This is a requirements mapping, not a claim that the draft commitments are implemented or accepted.

## Profiles

**Isolated customer inference:** pull a digest-pinned container using scoped, revocable, read-only registry credentials. Run on dedicated facility compute with no internet. Its HTTP policy endpoint is local/private. Deliver results through a separate controlled export path to customer-owned storage using scoped write-only credentials; an inference container does not need storage credentials.

**API-connected inference:** explicitly approved model endpoint and outbound network policy. Suitable for an external LLM API, but incompatible with the isolated profile unless the endpoint itself is served on the isolated network. The prototype permits only loopback model gateways in isolated mode; a production private gateway needs explicit network and identity controls. An HTTP API does not imply public internet access.

**Customer-owned inference hardware:** a separately qualified private observation/action link. No weights need enter Moravec systems. Locked hardware, access boundaries, and customer key custody are operational requirements, not features supplied by this SDK.

## Requirements mapping

| Draft requirement | Starter treatment | Remaining implementation |
| --- | --- | --- |
| Reference container, documented interface, validation harness, client simulation | Own HTTP service, JSON schemas, physical conformance scene | GPU images, Isaac scenes, real checkpoint qualification |
| Read-only registry pull and immutable image | Deployment manifest has image digest and secret reference | Registry enforcement and revocation |
| Customer storage, write-only delivery, checksums | Separate deployment/delivery manifests | Export worker, scoped permissions, encryption and read-back receipt |
| Image hashes and station versions | Delivery file SHA-256, policy image digest, station config/task versions | Capture pipeline and complete reproducibility bundle |
| Ten repetitions by default; uncertainty with scores | Binary Wilson interval helper | Campaign runner; per-task sample-size recommendation; task-score CI methodology |
| Episode score, dimensions, multiview footage, timestamped penalty attribution | Keep evaluator outside observation/action contract | Versioned rubric and evidence-linked scoring service |
| Hardware/operations faults nonbillable | Invalid episodes must be separate from scored trials | Fault taxonomy, adjudication, retry policy and billing integration |
| Confidential by default; publication opt-in | `publication_opt_in` defaults false | Consent enforcement and publication review; no automatic leaderboard writes |
| Client-specific storage/keys, access logs, staff controls | Deployment boundary documented | Infrastructure, audit and operational controls |
| Client retention choice and deletion commitments | Retention field | Verified deletion, container cleanup and retention jobs |
| Incident notices, subprocessors, human QA | Explicit operational requirements | Agreements, response process and audit evidence |

Teams, budgets, and usage dashboards remain later product scope. Confidentiality, data separation, evidence provenance, and explicit publication consent must not depend on those dashboards existing.

## Evaluation record to qualify next

A campaign should pin policy image/checkpoint, harness/prompt/tool versions, task/rubric version, scene/assets hash, simulator version, station/controller/calibration profile, random seeds, and inference settings. Each attempt needs status (`scored`, `infrastructure-failure`, or `invalid`), fault owner, retry lineage, score dimensions, and evidence references with timestamps. Report completed scored trials separately from attempts.

For binary independent trials, Wilson intervals are appropriate; ten successes out of ten still has a roughly 72.2% lower 95% confidence bound. Do not apply this formula to arbitrary partial-credit scores or treat ten deterministic conformance fixtures as a model success-rate estimate. Continuous scores and correlated layouts need a task-specific uncertainty method.
