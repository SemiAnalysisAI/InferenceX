# 原生视频交付

[English](video-generation.md) | **中文**

`video-generation` 为每个部署启动一个服务作业。H3 SGLang 当前启用 H200 和
B200 配置。每个 SKU 执行 C1/C2/C4，每格四个测量请求和一个预热请求，共十二个
测量请求与三个预热请求。冻结负载为 107 帧、24 fps、1344×768。配置存在不代表
GPU 资格已通过。H100 等待 checkpoint staging 与写权限验证；MI355X 因已准备的
CUDA 镜像无法在 AMD 执行而移出活动配置。历史产物保留原负载下的证据含义。

执行路径为 `perf-changelog.yaml` → `infx.matrix.plan` → `run-sweep.yml` →
`benchmark-tmpl.yml` → `infx.launch` → 原生 srt-slurm 服务/客户端 → 结果包。
Video canary 执行一个完整部署，之后从剩余矩阵中移除。`full-sweep-fail-fast`
继续按矩阵取消未完成作业。Video 的调度分配上限为 75 分钟，workflow 上限为
90 分钟，包含准备与清理；调用方的 video 时限覆盖集群 profile 默认值。旧 GPU
spec 的 `job_seconds` 不约束原生矩阵客户端；各请求仍使用显式超时。不增加自动重试。

产物名为 `h3-results-<run>-<attempt>-<recipe-fingerprint>`，完整指纹区分硬件、
引擎和部署。`manifest.artifact_name` 与 `ci.artifact_name` 保留准确名称。
`manifest.git_commit` 是实际执行 checkout；`manifest.ci.head_sha` 是 PR head，
两者可以不同。`plan_sha256` 与 `engine.name/revision` 从每格 run 记录取得。
功耗 producer/exporter 来源在结果包计算哈希前复制。

`video-sweep-manifest` 保留完整所选矩阵。PR 验证与合并导入要求该 attempt 的每个
所选部署和并发都存在，CI 身份一致，原生功耗有效，测量请求完成且文件哈希匹配。
这是已保留证据的验证，不重新解码媒体，也不证明感知质量、比较样本充足或生产验收。
App 继续读取旧的无后缀产物，但它们不能用于新的原生复用路径。Video 不支持
append-only sweep；使用普通完整 sweep，保留一个服务作业的全部负载。

获得复用与合并授权后，Merge Ingest 验证原始产物，再向 App 媒体发布器发送
`publish-video-results`，携带原始 `source_run_id`。混合运行仍另行发送现有
LLM/AgentX 导入事件。下游发布与播放回读通过后才能声明交付完成。首次原生 video
复用前，可信 main 分支上的 `/use` handler 必须已支持 video 产物；PR 本地校验
不能绕过这道门槛。
