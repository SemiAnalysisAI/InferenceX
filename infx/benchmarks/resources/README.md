# Prepared evaluation resources

**English** | [中文](./README_zh.md)

`gsm8k-test-doc-hashes.json` maps the 1,319 GSM8K test document IDs to their independently captured `doc_hash` values. It contains hashes, not dataset text or model responses. The source is the H100 c28 real-verification evaluation from InferenceX run `35314892357`, artifact `10535461349`, measured with lm-evaluation-harness revision `b315ef3b05176acc9732bb7fdec116abe1ecc476`.

The client validates both filters for every expected document, recomputes each hash from the emitted document bytes using the pinned harness serialization, and checks scores against the raw sample values. A high aggregate score or a self-reported sample count cannot replace those checks. Changing the test split requires an explicitly reviewed resource and workload identity change.

The preparation command copies this file and the packaged `infx/evals/gsm8k.yaml` into the prepared directory and binds their SHA256 digests. Client execution works from an installed wheel outside the checkout and rejects changed resource bytes.
