# Requires sm_90+ (thread-block clusters). `native` targets the GPU on the
# build host; override with e.g. `make ARCH=sm_100a`.
ARCH ?= native

gpc_query: gpc_query.cu
	nvcc -arch=$(ARCH) -O2 -std=c++17 -o $@ $<

clean:
	rm -f gpc_query

.PHONY: clean
