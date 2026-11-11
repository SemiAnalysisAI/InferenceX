# syntax=docker/dockerfile:1
#
# vLLM at the head of https://github.com/vllm-project/vllm/pull/66497, for benchmarking
# DeepSeek-V4.1-Flash on A100 before the PR merges.
#
# The PR changes Python only. The image is therefore the official nightly image of the nearest
# ancestor commit on main, with vLLM reinstalled from the PR head the documented Python-only
# way: VLLM_USE_PRECOMPILED=1 takes the compiled extensions and the Rust frontend from the
# official wheel of the PR's merge-base. The `verify` target proves both halves.

ARG BASE_REPO=vllm/vllm-openai
ARG BASE_DIGEST=sha256:390a656b54e12144ad15ff32244c6e848408b0ee94f2c4c5a84451e7430f1aa7
ARG BASE_TAG=vllm/vllm-openai:nightly-cf5cea2da37d3dfed27d2e0aa3041c3e553c2d57
ARG PR_REPO=https://github.com/yaoyaoding/vllm.git
ARG PR_HEAD=0fe642ac3c6f27e8e12849e754510b0ba606dd62
ARG MERGE_BASE=116025c68fea630ec367fc67e880368024d44659
ARG IMAGE_TAG=ghcr.io/semianalysisai/vllm-openai:pr66497-0fe642ac

FROM ${BASE_REPO}@${BASE_DIGEST} AS wheel
ARG PR_REPO
ARG PR_HEAD
ARG MERGE_BASE
# setuptools-scm reads the version and the package file list from git; the image ships none.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --quiet "${PR_REPO}" /src \
    && cd /src \
    && git fetch --quiet --tags https://github.com/vllm-project/vllm.git main \
    && git checkout --quiet --detach "${PR_HEAD}" \
    && test "$(git rev-parse HEAD)" = "${PR_HEAD}" \
    && test "$(git merge-base HEAD FETCH_HEAD)" = "${MERGE_BASE}" \
    && git diff --name-only "${MERGE_BASE}" HEAD > /pr-files.txt \
    && git ls-files vllm > /tracked.txt \
    && echo "non-Python files changed by the PR:" \
    && (grep -v '\.py$' /pr-files.txt || true)
# The build venv sees the image's torch, so the precompiled CUDA variant matches the torch the
# image runs with, and none of the build tooling reaches the final image.
ENV VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=${MERGE_BASE} \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
RUN python3 -c "import torch; assert torch.version.cuda.startswith('13.'), torch.version.cuda" \
    && uv venv --seed --system-site-packages --python "$(command -v python3)" /buildenv \
    && uv pip install --python /buildenv/bin/python \
        'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' \
        'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' wheel jinja2 \
    && /buildenv/bin/python -m pip wheel --no-build-isolation --no-deps --wheel-dir /dist /src \
    && ls -la /dist

FROM ${BASE_REPO}@${BASE_DIGEST} AS image
ARG BASE_DIGEST
ARG BASE_TAG
ARG PR_HEAD
ARG MERGE_BASE
ARG IMAGE_TAG
RUN --mount=type=bind,from=wheel,source=/dist,target=/dist \
    UV_NO_CACHE=1 UV_BREAK_SYSTEM_PACKAGES=1 \
    uv pip install --system --no-deps --reinstall /dist/vllm-*.whl
ENV VLLM_BUILD_COMMIT=${PR_HEAD} \
    VLLM_BUILD_URL=https://github.com/vllm-project/vllm/pull/66497 \
    VLLM_IMAGE_TAG=${IMAGE_TAG}
LABEL ai.vllm.build.commit="${PR_HEAD}" \
      ai.vllm.build.url="https://github.com/vllm-project/vllm/pull/66497" \
      ai.vllm.image.tag="${IMAGE_TAG}" \
      org.opencontainers.image.revision="${PR_HEAD}" \
      org.opencontainers.image.url="https://github.com/vllm-project/vllm/pull/66497" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.base.name="${BASE_TAG}" \
      org.opencontainers.image.base.digest="${BASE_DIGEST}" \
      com.semianalysis.inferencex.vllm-pr="66497" \
      com.semianalysis.inferencex.precompiled-wheel-commit="${MERGE_BASE}"

FROM image AS verify
ARG MERGE_BASE
RUN --mount=type=bind,from=wheel,source=/src,target=/src \
    --mount=type=bind,from=wheel,source=/pr-files.txt,target=/pr-files.txt \
    --mount=type=bind,from=wheel,source=/tracked.txt,target=/tracked.txt \
    --mount=type=bind,source=verify.py,target=/verify.py \
    PYTHONDONTWRITEBYTECODE=1 MERGE_BASE=${MERGE_BASE} python3 /verify.py
