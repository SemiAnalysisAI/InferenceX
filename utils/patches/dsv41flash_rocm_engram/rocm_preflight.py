# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run directly with unittest inside the pinned ROCm image, after apply.sh.

The bundled patch is upstream vllm-project/vllm#57491: it widens the two
``is_cuda()`` gates to ``is_cuda_alike()`` and makes ``amd/model.py`` import the
shared ``Engram`` from ``nvidia/engram.py``. These checks exercise that exact
code on a real ROCm GPU: the config gate, the pinned-host TP lookup through the
accelerator view, graph replay, and the model-level ``Engram`` constructor.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config import EngramConfig
from vllm.platforms import current_platform


class TestROCmEngramConfig(unittest.TestCase):
    def test_rocm_gate_accepts_offload_and_keeps_model_checks(self):
        model = SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_text_config=SimpleNamespace(engram_layer_ids=[1, 14]),
        )
        rocm = SimpleNamespace(
            is_cuda=lambda: False, is_rocm=lambda: True, is_cuda_alike=lambda: True
        )
        # verify_model_config imports current_platform lazily from vllm.platforms.
        with patch("vllm.platforms.current_platform", rocm):
            for offload in (False, True):
                with self.subTest(cpu_offload=offload):
                    EngramConfig(cpu_offload=offload).verify_model_config(model)
            model.hf_text_config.engram_layer_ids = []
            with self.assertRaisesRegex(ValueError, "non-empty"):
                EngramConfig().verify_model_config(model)
            # An architecture without an n-gram layer field stays rejected
            # (Qwen4Exp is not a counterexample: upstream maps it to ple_layer_ids).
            model.architecture = "LlamaForCausalLM"
            with self.assertRaisesRegex(ValueError, "supported Engram"):
                EngramConfig().verify_model_config(model)
        # A platform that is neither CUDA nor ROCm stays rejected.
        model.architecture = "DeepseekV41ForCausalLM"
        model.hf_text_config.engram_layer_ids = [1, 14]
        other = SimpleNamespace(
            is_cuda=lambda: False, is_rocm=lambda: False, is_cuda_alike=lambda: False
        )
        with (
            patch("vllm.platforms.current_platform", other),
            self.assertRaisesRegex(ValueError, "CUDA"),
        ):
            EngramConfig(cpu_offload=True).verify_model_config(model)

    def test_amd_model_uses_shared_engram(self):
        from vllm.models.deepseek_v41.amd import model as amd_model
        from vllm.models.deepseek_v41.nvidia import engram as nvidia_engram

        self.assertIs(amd_model.Engram, nvidia_engram.Engram)


@unittest.skipUnless(current_platform.is_rocm(), "requires a ROCm GPU")
class TestROCmEngramLookup(unittest.TestCase):
    def test_tp2_lookup_and_graph_replay(self):
        """Reconstruct both TP shards from HBM/host, including dead IDs and strides."""
        from vllm.models.deepseek_v41.common import engram as common
        from vllm.models.deepseek_v41.nvidia import engram as nvidia

        sizes = (17, 19, 23, 29, 31)
        rows, dim = sum(sizes), 64
        torch.manual_seed(17)
        weight = torch.randn(rows, dim).to(torch.float8_e4m3fn)
        scales = torch.randint(123, 131, (rows, dim // 32), dtype=torch.uint8)
        ids_cpu = torch.empty(7, len(sizes), dtype=torch.int32)
        start = 0
        for head, size in enumerate(sizes):
            ids_cpu[:, head] = torch.arange(7) % size + start
            start += size
        ids_cpu[0, 0] = -1
        storage = torch.zeros(7, 2, len(sizes), dtype=torch.int32, device="cuda")
        ids = storage[:, 1]
        ids.copy_(ids_cpu)
        dequant = weight.float() * torch.exp2(scales.float() - 127).repeat_interleave(
            32, dim=1
        )
        with (
            patch.object(common, "get_tensor_model_parallel_world_size", return_value=2),
            patch.object(nvidia, "get_engram_dp_size", return_value=1),
        ):
            for offload in (False, True):
                with self.subTest(cpu_offload=offload):
                    layers, outputs = [], []
                    for rank in range(2):
                        with (
                            patch.object(
                                common,
                                "get_tensor_model_parallel_rank",
                                return_value=rank,
                            ),
                            torch.device("cuda"),
                        ):
                            layer = nvidia.ParallelEngramEmbedding(
                                rows, dim, sizes, cpu_offload=offload
                            )
                        self.assertEqual(
                            layer.weight.device.type, "cpu" if offload else "cuda"
                        )
                        if offload:
                            self.assertTrue(layer.weight.is_pinned())
                            self.assertTrue(layer.weight_scale_inv.is_pinned())
                        layer.weight.weight_loader(layer.weight, weight)
                        layer.weight_scale_inv.weight_loader(
                            layer.weight_scale_inv, scales
                        )
                        outputs.append(
                            torch.empty(7, 3, dim, dtype=torch.bfloat16, device="cuda")
                        )
                        layers.append(layer)
                    # Warm the kernel and UVA views before capturing addresses.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for layer, output in zip(layers, outputs):
                            layer.lookup(ids, output)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for layer, output in zip(layers, outputs):
                            layer.lookup(ids, output)
                    for shift in (0, 1, 2):
                        changed = ids_cpu.roll(shift, dims=0)
                        ids.copy_(changed)
                        graph.replay()
                        expected = dequant[changed.clamp_min(0).long()]
                        expected[changed < 0] = 0
                        actual = torch.cat(outputs, dim=1)
                        torch.testing.assert_close(
                            actual[:, :5].cpu(),
                            expected.to(torch.bfloat16),
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(torch.count_nonzero(actual[:, 5:]).item(), 0)
                    del graph
                    if offload:
                        # A reload must refresh the cached accelerator views.
                        for layer, output in zip(layers, outputs):
                            layer.weight.data = torch.zeros_like(
                                layer.weight, pin_memory=True
                            )
                            layer.lookup(ids, output)
                        self.assertEqual(
                            torch.count_nonzero(torch.cat(outputs, dim=1)).item(), 0
                        )

    def test_model_engram_honors_explicit_offload(self):
        from vllm.models.deepseek_v41.common import engram as common
        from vllm.models.deepseek_v41.nvidia import engram as nvidia

        config = SimpleNamespace(hidden_size=16, hc_mult=1, rms_norm_eps=1e-6)
        layout = SimpleNamespace(
            num_embeddings=(72,),
            head_dim=32,
            max_ngram_size=2,
            n_heads=24,
            primes=(((3,) * 24,),),
        )
        for offload in (False, True):
            vllm_config = SimpleNamespace(
                engram_config=EngramConfig(cpu_offload=offload),
                scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            )
            with (
                self.subTest(offload=offload),
                patch.object(nvidia, "get_current_vllm_config", return_value=vllm_config),
                patch.object(
                    common, "get_current_vllm_config", return_value=vllm_config
                ),
                patch.object(nvidia, "get_engram_dp_size", return_value=1),
                patch.object(
                    common, "get_tensor_model_parallel_world_size", return_value=1
                ),
                patch.object(common, "get_tensor_model_parallel_rank", return_value=0),
                patch.object(
                    common, "ReplicatedLinear", return_value=torch.nn.Identity()
                ),
                torch.device("cuda"),
            ):
                module = nvidia.Engram(config, None, layout, 0, False, "engram")
            self.assertEqual(
                module.embed_tokens.weight.device.type, "cpu" if offload else "cuda"
            )
            self.assertEqual(module._prefetch_stream is not None, offload)
            module.embed_tokens.weight.data.fill_(1)
            module.embed_tokens.weight_scale_inv.data.fill_(127)
            ids = torch.zeros(8, 24, dtype=torch.int32, device="cuda")
            module.prepare_embeddings(ids)
            torch.testing.assert_close(
                module.embed(ids),
                torch.ones(8, 24, 32, dtype=torch.bfloat16, device="cuda"),
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
