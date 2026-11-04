FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

# Keep NKI's LNC=1 demotion of shared_hbm to private_hbm and silence the FutureWarning it prints.
ENV NKI_LNC1_SHARED_HBM=0

# The base entrypoint `eval`s its arguments, which breaks quoting.
ENTRYPOINT []
CMD ["bash"]
# Not /workspace: the base image keeps an editable torch_neuronx install there.
WORKDIR /InferenceX
