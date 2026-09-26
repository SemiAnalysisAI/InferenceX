# GLM-5.2 B200 C48 失败收敛

[English](README.md)

此候选把 prefill 失败传递给原始 decode 请求，在请求已派发后执行取消，并把失败传输所用缓冲区隔离到 worker 实际退出。它没有修复最初的 NIXL 断连，也没有证明另一个 prefill 请求停滞的原因。请求超时只是失败预算，不能把未完成的 warmup 计为成功。

B200 launcher 仅为 C48 性能任务选择这套 setup。eval、C64、聚合 recipe、镜像、拓扑、warmup 数量、计分时长、golden 门槛和必需的 PowerX 保持不变；GPU 恢复仍待验证。

## 源码与原生构建

补丁针对 Dynamo `c3e05f0244ae6264d7953f68e2499c6dc2f54723` 和 SGLang `0084030179bfba86bfeb6d43f7997d4076329d2c`。JSON 清单保存原始和修改后的文件哈希，包括未修改的上游锁文件。准备脚本仅在派生构建目录应用 nightly 版本转换，保留已审阅源码，并使用锁定依赖。

复用原始 `lmsysorg/sglang:nightly-dev-20260910-00840301` 运行环境和已有源码、构建缓存。在 checkout 外的持久目录中：

1. 放入本目录文件、精确 Dynamo 源码压缩包 `dynamo-c3e05f-full.tar.gz`、其解压目录 `dynamo-c3e05f-full`，以及精确 SGLang 源码目录 `sglang-008403017-candidate`。
2. 在对应源码目录先 `git apply --check`，再应用 `dynamo.patch`、`sglang.patch`。运行 `python3 prepare_native_containment.py verify`，检查所有审阅哈希和所有未改动的 Dynamo 压缩包成员。
3. 使用 Python 3.12.3、Rust 1.96.1、maturin 1.15.0、uv 0.12.0、hatchling 1.32.0、libclang 18.1.1，以及镜像内的 GCC 13/protoc/patchelf。`LIBCLANG_PATH` 指向隔离的 libclang 目录，`BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include"`。缺失依赖恢复到已有缓存，不更新锁文件。
4. 仅执行一次 `prepare_native_containment.py prepare --work-dir BUILD_TREE`，随后以绝对路径运行 `prepare_native_containment.py build --work-dir BUILD_TREE --out WHEELS --cargo-cache CARGO_CACHE --target-cache TARGET_CACHE`。失败后保留日志并复用准备目录和目标缓存。

release feature 列表固定在脚本中。wheel 如实标记为原始 Ubuntu 24.04 环境的 `cp310-abi3-manylinux_2_39_x86_64`，不声明 `manylinux_2_28` 可移植性或逐字节可复现。输出包含源码和版本转换标识、wheel SHA-256 及 ELF 依赖；打包前须检查这些信息和实际动态库解析。

## 已暂存的恢复包

当前集成包由构建回执 SHA-256 `27aa9a6eb223616d956dd7d507c0e26839cadcfb84339179898ff3a502ccbcba` 固定。Python wheel SHA-256 为 `fa638cb209c6391e598c641a331839be6cd8551e9831e539d84ee172cb77554f`，原生 wheel SHA-256 为 `5523aa8f7dcb2c5d5bd94043a758ad5fac204c2f40f54762ecea9080aa2b29b4`。

包内包含 `install_native_containment.py`、存放回执及两个 wheel 的 `wheels/`，以及在原始相对路径保存清单和十个已审阅 Python 文件的 `inputs/`。launcher 将其挂载到 `/glm52-containment`。安装前检查原始运行时二进制和每个修改文件，无依赖解析地安装后再次核对全部输出哈希。只有全部输出匹配才接受已安装候选。包缺失或不匹配时应在提交前失败。

该任务缓存只用于集成恢复，不是公开 wheel 发布。重新构建的 wheel 需要新的已验证回执和 setup pin；不得改标签或静默替换当前包。源码和构建可复现也不等于正式 PR sweep 或 merge reuse 验收。

## 证据边界

原生 wheel 已在原始 CPU 容器构建和安装。实际 PythonAsyncEngine TCP 检查验证正常回复、延后异常、请求标识、首个输出前取消和 handler 结束。原始运行时也应通过此通用传输协议检查；它不证明补丁后的 PrefillRouter 组合、实际 SGLang 注册、多 rank NIXL 缓冲区生命周期或完整 C48 计分。组件测试和原生证据须分别保存。

失败传输的缓冲区必须保留到 frontend/prefill/decode worker 实际退出。仅本地 `srun` 退出不足以证明清理完成；资源复用前须核实任务所属的精确 Slurm job/step 已终止及最终清理。保留全部失败尝试，仅以原始完整 C48 测量窗口和有效 PowerX 接受恢复结果。

C48 launcher 核对确切 Slurm 作业、Unix 用户、runner 名称和输出路径。明确的中止或清理标记启动 300 秒退出等待，若仍有工作步骤则仅取消该作业；工作步骤退出后，允许该作业的 `batch` 和 `extern` 在原分配时限内完成 CPU 报告。必须确认作业已终止且无活动 step 后才能复用资源。调度状态检查失败则不能验收。单个评分请求错误仍遵循原有阈值。
