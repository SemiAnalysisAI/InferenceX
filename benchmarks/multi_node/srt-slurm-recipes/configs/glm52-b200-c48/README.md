# GLM-5.2 B200 C48 failure containment

[中文](README_zh.md)

This candidate propagates a failed prefill response to the original decode request,
orders cancellation after request dispatch, and quarantines failed transfer buffers
until worker exit. It does not repair the initiating NIXL disconnect or prove the
cause of an independent stalled prefill request. A request deadline is a failure
budget: it must never turn an incomplete warmup into a successful benchmark.

The B200 launcher selects this setup only for C48 performance. Evaluation, C64,
aggregate recipes, images, topology, warmup count, scoring duration, golden
acceptance and required PowerX remain unchanged. GPU recovery is still required.

## Source and native build

The patches target Dynamo `c3e05f0244ae6264d7953f68e2499c6dc2f54723` and
SGLang `0084030179bfba86bfeb6d43f7997d4076329d2c`. The JSON manifest records
original and patched file hashes, including the unchanged upstream lockfiles.
The preparation helper applies only the nightly version transform to a derived
build tree; it preserves the reviewed source tree and uses locked dependencies.

Reuse the original `lmsysorg/sglang:nightly-dev-20260910-00840301` runtime and
existing source/build caches. In a persistent build directory outside the checkout:

1. Stage the files from this directory, the exact Dynamo source archive as
   `dynamo-c3e05f-full.tar.gz`, its extracted tree as `dynamo-c3e05f-full`, and
   the exact SGLang source as `sglang-008403017-candidate`.
2. Apply `dynamo.patch` and `sglang.patch` to their respective source trees with
   `git apply --check` followed by `git apply`. Run
   `python3 prepare_native_containment.py verify` to check all reviewed hashes
   and every unmodified Dynamo archive member.
3. Use Python 3.12.3, Rust 1.96.1, maturin 1.15.0, uv 0.12.0, hatchling 1.32.0,
   libclang 18.1.1, and the image's GCC 13/protoc/patchelf tools. Set
   `LIBCLANG_PATH` to the isolated libclang directory and
   `BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include"`.
   Restore missing dependencies into the existing cache without updating locks.
4. Run `prepare_native_containment.py prepare --work-dir BUILD_TREE` once, then
   `prepare_native_containment.py build --work-dir BUILD_TREE --out WHEELS
   --cargo-cache CARGO_CACHE --target-cache TARGET_CACHE` using absolute paths.
   Reuse the prepared tree and target cache after a failed build; preserve its log.

The release feature list is pinned in the helper. The wheel is tagged truthfully
as `cp310-abi3-manylinux_2_39_x86_64` for the original Ubuntu 24.04 environment;
no `manylinux_2_28` portability or bit-for-bit rebuild claim is made. Build output
includes source/packaging identities, wheel SHA-256 digests and ELF dependencies.
Verify these and installed library resolution before staging the pair.

## Staged recovery payload

The current integration payload is pinned by build receipt SHA-256
`27aa9a6eb223616d956dd7d507c0e26839cadcfb84339179898ff3a502ccbcba`.
Its Python wheel SHA-256 is
`fa638cb209c6391e598c641a331839be6cd8551e9831e539d84ee172cb77554f`;
the native wheel SHA-256 is
`5523aa8f7dcb2c5d5bd94043a758ad5fac204c2f40f54762ecea9080aa2b29b4`.

The payload contains `install_native_containment.py`, `wheels/` with that receipt
and both wheels, and `inputs/` with the manifest and the ten reviewed Python
files at their original tree-relative paths. The launcher mounts this payload
at `/glm52-containment`. Installation checks the original runtime binary and
every changed source before mutation, installs without dependency resolution,
then verifies all output hashes. An already installed candidate is accepted only
when every output matches. A missing or mismatched payload fails before submission.

This task-owned cache is an integration recovery distribution, not a public wheel
release. A rebuilt wheel requires a new verified receipt and setup pin; do not
retag it or silently substitute it for this pair. Source/build reproducibility
does not by itself satisfy formal PR sweep or merge-reuse requirements.

## Evidence boundary

The native wheel builds and installs in the original CPU container. The installed
PythonAsyncEngine TCP check verifies healthy replies, late exceptions, request
identity, pre-output cancellation and handler finalization. This generic transport
contract also works on the original runtime; it does not prove the patched
PrefillRouter composition, real SGLang registration, multi-rank NIXL buffer lifetime
or successful C48 scoring. Preserve component tests separately from native evidence.

Failed-room quarantine remains owned until the actual frontend/prefill/decode
workers exit. Local `srun` exit alone is insufficient. The C48 launcher
checks the exact Slurm job, Unix user, runner name and output path. An authoritative
abort or cleanup marker starts a 300-second grace; remaining work is cancelled
only in that job. Terminal state and no active steps must be verified within the
post-cancel bound before resource reuse. Inspection failure rejects acceptance.
Individual scored-request errors retain the existing benchmark thresholds. Preserve
all failed attempts; accept recovery only with the original full C48 window and valid PowerX.
