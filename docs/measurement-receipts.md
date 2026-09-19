# Trusted measurement receipts

[中文](./measurement-receipts_zh.md)

This prerequisite makes the hosted receipt issuer and immutable artifact transport available on trusted `main` before a native producer is qualified. It does not select a native pilot, modify matrix routing, add a runner launcher, or allocate a new GPU lane. The included Python clients and contract types are reusable validation/preparation foundations; existing benchmark selection remains unchanged.

## Landing order

1. Deploy the InferenceX-app receipt reader and migration `016_measurement_snapshots.sql`. Retain its actual deployment commit and verify the required receipt versions before enabling native publication.
2. Land this control prerequisite on InferenceX `main`. Its [issuer workflow](../.github/workflows/phase1-receipt.yml) checks out its own trusted workflow commit; it never runs candidate code to decide what measurements are acceptable.
3. Configure the reviewed issuer and deployed-reader revisions below. Native producer selection and allocation belong to a separately reviewed change, with its own hardware qualification evidence.
4. After the complete source run succeeds, issue its source receipt, validate staging, then issue a separate publication record after the approved merge run succeeds. Local control tests do not establish reader deployment, GPU qualification, or production publication.

## Policy and reviewed inputs

Configure `INFX_RECEIPT_ISSUER_SHAS` in both repositories as a comma-separated allowlist of reviewed InferenceX issuer commits, and set `INFX_RECEIPT_ISSUER_WORKFLOW` to `.github/workflows/phase1-receipt.yml`. Keep previously accepted issuer revisions while their immutable receipts remain supported. In InferenceX, set `INFX_PHASE1_READER_REVISION` to the final deployed app commit and record this prerequisite's trusted commit in `INFX_PHASE1_COLLECTOR_REVISION` for later producer qualification. These settings are deployment prerequisites, not values inferred from dispatch payloads.

Maintainers review `qualification/phase1/*.json` inputs on trusted `main` before invoking the issuer with `kind: measurement`. The [Approval schema](../infx/workflows/phase1_publication.py) requires eight throughput points at concurrency 1, 2, 4, 8, 16, 20, 24, and 28 plus the real c28 GSM8K evaluation, with original source run/attempt/head, per-point execution and bundle identities, native manifest digests, and the actual corpus revision. Obtain these identities from independently prepared control records; worker `execution.json` files and artifact names cannot authorize their own expected contract. This prerequisite deliberately includes no fabricated approval file.

The [receipt validator](../infx/results/publication_receipt.py) checks exact GitHub artifact IDs and ownership, API and ZIP digests, contained members, expected execution identities, physical topology, canonical configuration, required metrics, dataset metadata, and complete evaluation sample/filter coverage. Per-job raw lm-eval results and metadata remain valid inputs; aggregate deployments retain explicit zero split-worker counts. The compact version-1 receipt preserves original measurements when staging later becomes production.

## Staging, publication, and recovery

The [transport resolver](../infx/workflows/receipt_transport.py) uses read-only APIs to locate a unique accepted receipt from a successful `workflow_dispatch` issuer on `main` at an allowed revision. It verifies the original source attempt rather than substituting the latest rerun. An API inventory containing `native-execution-*` requires a receipt; missing or invalid evidence never enters legacy native ingestion. Ordinary legacy inventories retain their existing path.

Use the existing authorized staging flow after source sealing. For production, review a [PublicationRecord](../infx/workflows/phase1_record.py) that references the original receipt artifact/digest and separately binds the merge run/SHA, changelog artifact/digest, and deployed app/ingest revisions. Issue it with `kind: publication`. Its `ingest_sha` must equal the app checkout that will execute ingestion; retain deployment evidence for `app_sha`. The source receipt is not rewritten.

The sweep's automatic ingest jobs defer while required sealing is pending so the source or merge workflow can finish successfully. Once both issuer runs finish, the existing [recovery workflow](../.github/workflows/recover-reused-ingest.yml) dispatches exact receipt/publication IDs and both ZIP/JSON digests. Native production always requires the later record, including when source and merge run IDs are equal. The deployed app independently validates transport and accepted snapshots before import; interrupted imports may resume only the same accepted snapshot. Do not select a newer same-named artifact to repair a missing accepted one.

## Local verification

Run the behavior suites from this prerequisite checkout:

```bash
uv run --locked --group test pytest -q utils/test_benchmark_preparation.py \
  utils/test_python_benchmark_clients.py utils/test_phase1_receipt_control.py \
  utils/test_publication_receipt.py utils/test_receipt_transport.py
uvx --exclude-newer PT12H ruff@latest check infx
uvx --exclude-newer PT12H ruff@latest format --check infx
```

The suites exercise installed package resources, prepared client identities, child-process cleanup, nine-point approval construction, artifact verification, and immutable source/publication transport without submitting a Slurm job or writing a production database. See [testing](./testing.md) and [eval/AgentX procedures](./eval-agentx-procedures.md) for the existing validation boundaries.
