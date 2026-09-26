# Benchmark client limitations

<div align="center">

**English** | [中文](./KNOWN_LIMITATION_zh.md)

</div>

The serving benchmark client can become the bottleneck when testing small models, such as Gemma 1B or Llama 8B, at very high queries per second (QPS) with short inputs and outputs.

InferenceX currently focuses on larger models, longer input and output sequences, and interactive latency and throughput per user. Supporting the high-QPS, small-model workload would require a multiprocess benchmark client.
