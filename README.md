#  InferenceMAX™, Open Source Inference Frequent Benchmarking

InferenceMAX™ runs our suite of benchmarks every night, continually re-benchmarking the world’s most popular open-source inference frameworks and models to track real performance in real time. As these software stacks improve, InferenceMAX™ captures that progress in near real-time, providing a live indicator of inference performance progress. A live dashboard is available for free publicly at https://inferencemax.ai/. 


## Why?

InferenceMAX™, an open-source, under Apache2 license, automated benchmark designed to move at the same rapid speed as the software ecosystem itself, is built to address this challenge.

LLM Inference performance is driven by two pillars, hardware and software. While hardware innovation drives step jumps in performance every year through the release of new GPUs/XPUs and new systems, software evolves every single day, delivering continuous performance gains on top of these step jumps. Speed is the Moat 🚀
 
AI software like SGLang, vLLM, TensorRT-LLM, CUDA, ROCm and achieve this continuous improvement in performance through kernel-level optimizations, distributed inference strategies, and scheduling innovations that increase the pareto frontier of performance in incremental releases that can be just days apart.
 
This pace of software advancement creates a challenge: benchmarks conducted at a fixed point in time quickly go stale and do not represent the performance that can be achieved with the latest software packages.


## Acknowledgements & Supporters
Thank you to Lisa Su and Anush Elangovan for providing the MI355X and CDNA3 GPUs for this free and open-source project. We want to recognize the many AMD contributors for their responsiveness and for debugging, optimizing, and validating performance across AMD GPUs. 
We’re also grateful to Jensen Huang and Ian Buck for supporting this open source with access to a GB200 NVL72 rack (through OCI) and B200 GPUs. Thank you to the many NVIDIA contributors from the NVIDIA inference team, NVIDIA Dynamo team.

We also want to recognize the SGLang, vLLM, and TensorRT-LLM maintainers for building a world-class software stack and open sourcing it to the entire world.
Finally, we’re grateful to Crusoe, CoreWeave, Nebius, TensorWave, Oracle and TogetherAI for supporting open-source innovation through compute resources, enabling this.

"As we build systems at unprecedented scale, it's critical for the ML community to have open, transparent benchmarks that reflect how inference really performs across hardware and software. InferenceMAX™'s head-to-head benchmarks cut through the noise and provide a living picture of token throughput, performance per dollar, and tokens per Megawatt. This kind of open source effort strengthens the entire ecosystem and helps everyone, from researchers to operators of frontier datacenters, make smarter decisions." - Peter Hoeschele, VP of Infrastructure and Industrial Compute, OpenAI Stargate

"The gap between theoretical peak and real-world inference throughput is often determined by systems software: inference engine, distributed strategies, and low-level kernels. InferenceMAX™ is valuable because it benchmarks the latest software showing how optimizations actually play out across various hardware. Open, reproducible results like these help the whole community move faster.” - Tri Dao, Chief Scientist of Together AI & Inventor of Flash Attention

“The industry needs many public, reproducible benchmarks of inference performance. We’re excited to collaborate with InferenceMAX™ from the vLLM team. More diverse workloads and scenarios that everyone can trust and reference will help the ecosystem move forward. Fair, transparent measurements drive progress across every layer of the stack, from model architectures to inference engines to hardware.” – Simon Mo, vLLM Project Co-Lead

## SemiAnalysis is Hiring

We are looking for an engineer to join our special projects team. This is a unique opportunity to work on high-visibility special projects such as InferenceMAX™ with support from many industry leaders and CEOs. If you’re passionate about performance engineering, system reliability, and want to work at the intersection of hardware and software, this is a rare chance to make industry wide impact.
What you’ll work on:
- Building and running large-scale benchmarks across multiple vendors (AMD, NVIDIA, TPU, Trainium, etc.)
- Designing reproducible CI/CD pipelines to automate benchmarking workflows
- Ensuring reliability and scalability of systems used by industry partners
  
What we’re looking for:
- Strong skills in Python
- Background in Site Reliability Engineering (SRE) or systems-level problem solving
- Experience with CI/CD pipelines and modern DevOps practices
- Curiosity about GPUs, TPUs, Trainium, multi-cloud, and performance benchmarking
Link to apply: https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1

