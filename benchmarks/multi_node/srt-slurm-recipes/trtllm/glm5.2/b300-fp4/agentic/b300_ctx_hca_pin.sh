# Worker preamble for GLM-5.2 dynamo-trt on B300 DSXE. Use the requested HCA
# layout only when those devices are active; otherwise let UCX discover the
# available fabric devices.

unset UCX_TLS   # Preserve CUDA memory registration for NIXL transfers.
unset UCX_NET_DEVICES

# Dynamo is installed at worker startup. Allow pip to resume transiently truncated
# package downloads instead of failing the whole multi-node allocation.
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-20}"
export PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-20}"

# pip does not retry every truncated response even with its network retry
# settings. Retry the complete command so a single worker does not leave an
# otherwise healthy multi-rank launch permanently short of one rank.
pip() {
    local _srt_attempt=1 _srt_max_attempts=5
    while ! command pip "$@"; do
        if [ "$_srt_attempt" -ge "$_srt_max_attempts" ]; then
            echo "pip failed after $_srt_attempt attempts" >&2
            return 1
        fi
        echo "pip failed; retrying complete command (attempt $((_srt_attempt + 1))/$_srt_max_attempts)" >&2
        sleep $((_srt_attempt * 5))
        _srt_attempt=$((_srt_attempt + 1))
    done
}

_srt_live_devices() {
    _srt_out=""; _srt_oIFS="$IFS"; IFS=,
    for _srt_d in $_srt_in; do
        _srt_n="${_srt_d%%:*}"
        case "$(cat "/sys/class/infiniband/$_srt_n/ports/1/state" 2>/dev/null)" in
            *ACTIVE*) _srt_out="${_srt_out:+$_srt_out,}$_srt_d" ;;
        esac
    done
    IFS="$_srt_oIFS"; printf '%s' "$_srt_out"
}

# `symmetric` shares four rails. `bia_faithful` pins each prefill rank to its
# physical GPU's rail pair. Other values leave ranks unchanged.
case "${BASH_EXECUTION_STRING:-}" in
    *SRT_FABRIC_MODE=symmetric*)
        _srt_in="mlx5_0:1,mlx5_1:1,mlx5_10:1,mlx5_11:1"
        _srt_hca="$(_srt_live_devices)"
        if [ -n "$_srt_hca" ]; then
            export UCX_NET_DEVICES="$_srt_hca"
            echo "CTX_HCA_PIN mode=symmetric localid=${SLURM_LOCALID:-0} UCX_NET_DEVICES=$UCX_NET_DEVICES"
        else
            echo "CTX_HCA_PIN mode=symmetric localid=${SLURM_LOCALID:-0} UCX_NET_DEVICES=<unset, use UCX auto-discovery>"
        fi
        return 0 2>/dev/null || true
        ;;
    *SRT_FABRIC_MODE=bia_faithful*) ;;
    *) return 0 2>/dev/null || true ;;
esac

case "${BASH_EXECUTION_STRING:-}" in
    *trtllm_config_prefill*) ;;                            # context rank: pin below
    *) return 0 2>/dev/null || true ;;                     # decode/frontend: unpinned
esac

_srt_cvd=$(printf '%s' "${BASH_EXECUTION_STRING:-}" \
           | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | head -1 | cut -d= -f2)
[ -n "$_srt_cvd" ] || return 0 2>/dev/null || true

IFS=, read -r -a _srt_g <<< "$_srt_cvd"
_srt_phys="${_srt_g[${SLURM_LOCALID:-0}]}"
case "$_srt_phys" in
    0) _srt_hca="mlx5_2:1,mlx5_3:1"   ;;
    1) _srt_hca="mlx5_8:1,mlx5_9:1"   ;;
    2) _srt_hca="mlx5_4:1,mlx5_5:1"   ;;
    3) _srt_hca="mlx5_0:1,mlx5_1:1"   ;;
    4) _srt_hca="mlx5_16:1,mlx5_17:1" ;;
    5) _srt_hca="mlx5_22:1,mlx5_23:1" ;;
    6) _srt_hca="mlx5_20:1,mlx5_21:1" ;;
    7) _srt_hca="mlx5_10:1,mlx5_11:1" ;;
    *) echo "CTX_HCA_PIN: no mapping for physical GPU $_srt_phys" >&2; _srt_hca="" ;;
esac

if [ -n "$_srt_hca" ]; then
    _srt_in="$_srt_hca"; _srt_hca="$(_srt_live_devices)"
fi
if [ -n "$_srt_hca" ]; then
    export UCX_NET_DEVICES="$_srt_hca"
    echo "CTX_HCA_PIN localid=${SLURM_LOCALID:-0} phys_gpu=$_srt_phys UCX_NET_DEVICES=$UCX_NET_DEVICES"
else
    echo "CTX_HCA_PIN localid=${SLURM_LOCALID:-0} phys_gpu=$_srt_phys UCX_NET_DEVICES=<unset, use UCX auto-discovery>"
fi
return 0 2>/dev/null || true
