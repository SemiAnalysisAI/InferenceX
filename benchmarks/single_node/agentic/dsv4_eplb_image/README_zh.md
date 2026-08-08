# DSv4 EPLB 镜像层（MI355X / ROCm）

[English](README.md) | 中文

`jiahcao/vllm-dsv4:optimal-fse-aiter4269-sparse4382-megamoe-eplb` 的构建上下文，
即 DSv4 MI355X EPLB arm 所运行的镜像。

EPLB 需要两个基础镜像未提供的 DSv4 模型文件。此前它们是在容器启动时以 bind-mount
的方式覆盖到镜像内的 editable vLLM 安装路径上，这意味着一份干净的 checkout 根本
无法复现 EPLB arm。本目录就是这份 patch，以构建层的形式存在。

| 文件 | 用途 |
|---|---|
| `Dockerfile` | 把两个打过 patch 的文件复制到 `/src/vllm/vllm/models/deepseek_v4/amd/`，然后运行验证脚本。`/src/vllm` 是 editable 安装，因此必须替换真实文件 —— PYTHONPATH 遮蔽的办法在这里不起作用。 |
| `deepseek_v4/amd/model.py` | `DeepseekV4MixtureOfExperts` mixin，使 runner 能注册该模型；另含两种 MoE 后端的 physical-slot expert sizing。 |
| `deepseek_v4/amd/mega_moe_experts.py` | FlyDSL mega-MoE 路径上的 EPLB：multi-slot weight loader、在 shuffle 后的 kernel layout 上的 `get_expert_weights()`、`forward()` 中的 logical→physical 重映射，以及依据 `MegaMoEShape` 计算 mori symmetric heap 大小。 |
| `verify_eplb_patch.py` | 构建期验证 patch 确实已安装。让构建失败，而不是让 benchmark 失败。 |

## 构建

```sh
docker build -t jiahcao/vllm-dsv4:optimal-fse-aiter4269-sparse4382-megamoe-eplb .
```

`BASE` 默认为 `-megamoe` tag；基础镜像升级时用 `--build-arg BASE=…` 覆盖。
该层约增加 97 kB。

## 为什么模型必须打 patch

使用原始的 `amd/model.py` 时，runner 永远不会把模型注册为 `MixtureOfExperts`，
于是 `--enable-eplb` 是一个**静默的 no-op** 而不是报错：一个 arm 可以"带着 EPLB
运行"却什么都没测到。验证脚本正是为了拦住这种情况。其中两项检查值得了解：

- `_init_mega_moe_experts` 中对 redundancy 的读取必须是无条件的，不能以
  `self.enable_eplb` 为条件。MTP 复用 `DeepseekV4DecoderLayer` 且 EPLB 为关闭状态，
  一旦加上这个条件，MTP 层会按 384/48 sizing，而 target 层是 392/49 —— 两个
  `MegaMoEShape` key，于是产生两个 `MegaMoEV2` 实例、两套约 7.7 GiB 的 mori
  symmetric buffer，而不是共享一个 runtime。
- 验证脚本用 `ast` 解析而不是 import。import 会拉起 `vllm.platforms.rocm`，它在模块
  级调用 `_get_gcn_arch()`，进而调用 `torch.cuda` —— 而 docker build 中没有 GPU。

它同时是一个 `COPY` 进去的文件，而不是 `RUN python - <<HEREDOC`。旧版（非 BuildKit）
builder 不支持 heredoc：那种写法会给 python 喂空 stdin 并以 0 退出，也就是说这个检查
在未打 patch 的镜像上照样通过。已通过对原始文件构建验证过这一点。

## 运行一个 EPLB arm

在 `dsv4_agentic_g05.sh` 或 `dsv4_quick_ab_g05.sh` 上设置 `ENABLE_EPLB=1`，
且 `EP_SIZE > 1`。`NUM_REDUNDANT_EXPERTS` 必须保证
`(n_routed_experts + N) % ep_size == 0` —— 对这个 384 expert 的 checkpoint、8 个 rank
而言就是 8 的倍数。EPLB 默认关闭。
