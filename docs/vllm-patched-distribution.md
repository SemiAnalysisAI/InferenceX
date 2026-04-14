# Distributing Patched vLLM for Dynamic Reconfiguration

Dynamic scheduler reconfiguration requires a vLLM build that includes the
runtime reconfiguration API used by InferenceX:

- `POST /pause`
- `POST /reconfigure`
- `POST /resume`

A stock vLLM image or wheel that does not include this API can still run normal
InferenceX benchmarks. It only needs the patched build when
`VLLM_DYNAMIC_RECONFIGURE=1` is enabled.

## Option 1: Custom Benchmark Container

Build a container that starts from the normal InferenceX/vLLM benchmark image and
installs the patched vLLM branch or wheel at image build time. This is the most
reproducible option for cluster sweeps.

Example Dockerfile pattern:

```Dockerfile
FROM vllm/vllm-openai:<base-tag>

ARG VLLM_REPO=https://github.com/<org>/vllm.git
ARG VLLM_REF=<patched-commit-or-branch>

RUN git clone "$VLLM_REPO" /opt/patched-vllm \
 && cd /opt/patched-vllm \
 && git checkout "$VLLM_REF" \
 && VLLM_USE_PRECOMPILED=1 python3 -m pip install --no-cache-dir -e .

LABEL org.opencontainers.image.source-vllm="$VLLM_REPO"
LABEL org.opencontainers.image.revision-vllm="$VLLM_REF"
```

Then update the InferenceX config entry to use that image for the vLLM run.

## Option 2: Install a Pinned Wheel Before `vllm serve`

Build a patched wheel once, publish it to an artifact store, and install it in
the benchmark job before launching `vllm serve`.

```bash
export VLLM_PATCHED_INSTALL_MODE=wheel
export VLLM_PATCHED_WHEEL=/workspace/wheels/vllm-<version>-patched.whl
install_patched_vllm
```

Remote wheel URLs also work if the runner has access:

```bash
export VLLM_PATCHED_WHEEL=https://example.internal/wheels/vllm-patched.whl
install_patched_vllm
```

## Option 3: Mount a Patched Checkout and Install Editable

This is useful for fast experiments, but less reproducible than an image or
wheel. Mount the patched vLLM checkout into the job and run:

```bash
export VLLM_PATCHED_INSTALL_MODE=editable
export VLLM_PATCHED_CHECKOUT=/workspace/vllm-patched
install_patched_vllm
```

## Option 4: Clone and Install a Pinned Git Ref

This is convenient when the cluster has network access to the patched branch:

```bash
export VLLM_PATCHED_INSTALL_MODE=git
export VLLM_PATCHED_REPO=https://github.com/<org>/vllm.git
export VLLM_PATCHED_REF=<patched-commit-or-branch>
install_patched_vllm
```

For reproducible benchmark results, prefer a commit SHA over a mutable branch.

## Recommended Cluster Flow

1. Start from a custom image or call `install_patched_vllm` before `vllm serve`.
2. Launch vLLM with the largest static capacity needed by the sweep.
3. Enable scheduler reconfiguration for the sweep:

```bash
export VLLM_DYNAMIC_RECONFIGURE=1
export VLLM_MAX_NUM_BATCHED_TOKENS=32768
export VLLM_MAX_NUM_SEQS="$CONC"
```

4. Run `run_benchmark_serving` as usual.
5. Record the patched vLLM commit SHA or wheel URL in the run notes/results.
