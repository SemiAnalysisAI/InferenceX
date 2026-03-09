# automous claude code agent experiment

benchmarks acutal vllm swap blocks for kvcache offload transfer (note that this is only using the DMA engine (i.e. NVIDIA Copy Engine) and we havent tested SM driven DtoH and HtoD yet https://github.com/vllm-project/vllm/blob/421012b63ae2e3f26bba0506e9f3951e9dc53ded/csrc/cache_kernels.cu#L27-L66
