#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
export K3_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_pr51705_52033_53598_52968.patch"

exec bash "$script_dir/kimik3_fp4_mi355x_pr_ab.sh" "$@"
