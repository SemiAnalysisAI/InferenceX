# DSv4 EPLB on ROCm, baked into the megamoe image.
#
# Until now these two files were bind-mounted over the image's editable vLLM
# install at container start (`cp /patch/... /src/vllm/vllm/models/...`), which
# meant every launcher had to remember to mount them and a clean checkout could
# not reproduce an EPLB arm at all. This layer does the same copy once, at build
# time, over the same import target -- /src/vllm is an editable install, so the
# real file must be replaced rather than shadowed via PYTHONPATH.
#
# model.py            DeepseekV4MixtureOfExperts mixin so the runner registers
#                     the model (without it EPLB is a silent no-op, not an
#                     error), plus physical-slot expert sizing for both backends.
# mega_moe_experts.py EPLB on the FlyDSL mega-MoE path: multi-slot weight
#                     loader, get_expert_weights() over the shuffled kernel
#                     layout, the logical->physical remap in forward(), and
#                     mori symmetric-heap sizing from the MegaMoEShape.
ARG BASE=jiahcao/vllm-dsv4:optimal-fse-aiter4269-sparse4382-megamoe
FROM ${BASE}

COPY deepseek_v4/amd/model.py \
     /src/vllm/vllm/models/deepseek_v4/amd/model.py
COPY deepseek_v4/amd/mega_moe_experts.py \
     /src/vllm/vllm/models/deepseek_v4/amd/mega_moe_experts.py

# Fail the build rather than the benchmark. A truncated COPY or a moved import
# target would otherwise surface four minutes into a run as a missing EPLB
# registration -- the exact silent failure this image is meant to remove.
#
# The verifier is a COPY'd file, not a `RUN python - <<HEREDOC`: the legacy
# (non-BuildKit) builder does not support heredocs, so that form feeds python
# an empty stdin and exits 0 -- a check that passes on an unpatched image.
# Confirmed by building this Dockerfile against the stock files. Keep it a real
# file so the build works on both builders.
#
# Parsed with ast rather than imported: importing the module pulls in
# vllm.platforms.rocm, whose module-level _get_gcn_arch() calls torch.cuda and
# dies with "No CUDA GPUs are available" -- there is no GPU in a docker build.
COPY verify_eplb_patch.py /tmp/verify_eplb_patch.py
RUN python /tmp/verify_eplb_patch.py && rm /tmp/verify_eplb_patch.py
