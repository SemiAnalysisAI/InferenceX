# KNOWN_LIMITATION

When benchmarking small models (e.g., Gemma 1B or Llama 8B) at ultra-high QPS with short input and output lengths, the current InferenceX bench_serving client becomes client-bound.

InferenceX does not currently benchmark in this regime and has no plans to. Our roadmap skews the opposite direction, larger models, longer ISL/OSL (i.e., agentic workloads), and interactive TTFT & tok/s/user scenarios rather than ultra-high QPS. 

Should the need arise, we shall fix this known limitation & migrate to a multi-process benchmark client.
