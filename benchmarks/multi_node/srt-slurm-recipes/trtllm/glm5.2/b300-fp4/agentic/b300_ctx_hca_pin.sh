# Worker preamble for GLM-5.2 dynamo-trt on b300-nv. Prefill ranks select active
# HCA pairs by physical GPU; decode remains unpinned.

unset UCX_TLS   # Preserve CUDA memory registration for NIXL transfers.

_srt_live_devices() {
    set -- /sys/class/infiniband/mlx5_*
    [ -e "$1" ] || { printf '%s' "$_srt_in"; return 0; }   # fail open
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
        export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_10:1,mlx5_11:1"
        echo "CTX_HCA_PIN mode=symmetric localid=${SLURM_LOCALID:-0} UCX_NET_DEVICES=$UCX_NET_DEVICES"
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
    echo "CTX_HCA_PIN localid=${SLURM_LOCALID:-0} phys_gpu=$_srt_phys UCX_NET_DEVICES=<unset, own rail pair is not ACTIVE>"
fi
return 0 2>/dev/null || true
