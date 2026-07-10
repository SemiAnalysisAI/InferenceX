#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# WAR for sglang issue #30399:
#   "[Bug] PD disaggregation: GB200 Deepseek v4 Pro DeepGEMM grid sync timeout"
# Bisected in that issue to an sgl-deep-gemm version upgrade (sglang #29554).
# Only trips on cross-node DEP8 prefill (single-node DEP4 is fine). Downgrade
# sgl-deep-gemm to the pre-regression version in every worker container.
#
# Runs via `setup_script: pin-sgl-deep-gemm.sh` (before dynamo install + worker
# startup, inside each worker's container). Prints before/after so the run log
# shows what the container shipped vs what we pinned.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1

TARGET="${SGL_DEEP_GEMM_VERSION:-0.1.3}"

echo "[pin-sgl-deep-gemm] before:"
pip show sgl-deep-gemm 2>/dev/null | grep -iE '^(Name|Version):' || echo "  (not installed)"

echo "[pin-sgl-deep-gemm] installing sgl-deep-gemm==${TARGET} ..."
pip install --force-reinstall --no-deps \
  --extra-index-url https://www.piwheels.org/simple \
  "sgl-deep-gemm==${TARGET}"

echo "[pin-sgl-deep-gemm] after:"
pip show sgl-deep-gemm 2>/dev/null | grep -iE '^(Name|Version):'
