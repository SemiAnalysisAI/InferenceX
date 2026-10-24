import ctypes
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts" / "ascend"
SOURCE = HERE / "operatorx_fp8_l0_guard.cpp"
CAPTURE = Path(__file__).with_name("fixtures") / "ascend_fp8_tiling.bin"
KERNEL = b"QuantBatchMatmulV3_ND_ND_fp8e4m3_fp8e4m3_fp32_bf16_high_performance"

# Exercise the actual hook code with deterministic fake runtime entry points.
# Redirecting dlsym keeps this CPU harness portable; Linux LD_PRELOAD resolution
# and the installed CANN ABI remain device-smoke responsibilities.
HARNESS = r"""
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
static char fake_name[2048] = {};
static int fake_name_failure = 0;
static unsigned fake_launch_count = 0;
static int fake_name_query(void *, uint32_t n, char *out) {
    if (fake_name_failure) return 1;
    std::snprintf(out, n, "%s", fake_name);
    return 0;
}
static int fake_entry(void *, uint64_t, void **out) {
    *out = reinterpret_cast<void *>(0x1234);
    return 0;
}
static int fake_launch(void *, uint32_t, void *, void *, void *, size_t, void *, size_t) {
    ++fake_launch_count;
    return 0;
}
static void *fake_dlsym(void *, const char *name) {
    if (!std::strcmp(name, "aclrtGetFunctionName")) return reinterpret_cast<void *>(fake_name_query);
    if (!std::strcmp(name, "aclrtBinaryGetFunctionByEntry")) return reinterpret_cast<void *>(fake_entry);
    if (!std::strcmp(name, "aclrtLaunchKernelWithHostArgs")) return reinterpret_cast<void *>(fake_launch);
    return nullptr;
}
#define dlsym fake_dlsym
#include "operatorx_fp8_l0_guard.cpp"
#undef dlsym
extern "C" void operatorx_test_reset(const char *name, int fail) {
    std::snprintf(fake_name, sizeof(fake_name), "%s", name);
    fake_name_failure = fail;
    fake_launch_count = 0;
}
extern "C" unsigned operatorx_test_forward_count() { return fake_launch_count; }
"""


class Placeholder(ctypes.Structure):
    _fields_ = [("addrOffset", ctypes.c_uint32), ("dataOffset", ctypes.c_uint32)]


def changed(raw, **fields):
    offsets = {
        "batchA": 0,
        "quantA": 60,
        "quantB": 64,
        "groupM": 76,
        "groupN": 80,
        "groupK": 84,
        "m": 88,
        "n": 92,
        "k": 96,
        "baseM": 100,
        "baseN": 104,
        "baseK": 108,
    }
    result = bytearray(raw)
    for name, value in fields.items():
        struct.pack_into("<I", result, offsets[name], value)
    return bytes(result)


class GuardParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temporary = tempfile.TemporaryDirectory(prefix="operatorx-fp8-guard-cpu-")
        cls.addClassCleanup(cls.temporary.cleanup)
        library = Path(cls.temporary.name) / "liboperatorx_fp8_l0_guard.so"
        link = (
            ["-dynamiclib"]
            if sys.platform == "darwin"
            else ["-shared", "-fPIC", "-ldl"]
        )
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-pthread",
                str(SOURCE),
                *link,
                "-o",
                str(library),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"Guard compilation failed:\n{result.stdout}{result.stderr}"
            )
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.operatorx_fp8_guard_validate.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        cls.lib.operatorx_fp8_guard_validate.restype = ctypes.c_int
        cls.lib.operatorx_fp8_guard_last_json.argtypes = []
        cls.lib.operatorx_fp8_guard_last_json.restype = ctypes.c_char_p
        cls.lib.operatorx_fp8_guard_configure.argtypes = [ctypes.c_uint32] * 3 + [
            ctypes.c_int
        ]
        cls.lib.operatorx_fp8_guard_configure.restype = ctypes.c_int
        cls.original = CAPTURE.read_bytes()
        harness_source = Path(cls.temporary.name) / "guard_harness.cpp"
        harness_source.write_text(HARNESS)
        harness_library = Path(cls.temporary.name) / "libguard_harness.so"
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-pthread",
                "-I",
                str(HERE),
                str(harness_source),
                *link,
                "-o",
                str(harness_library),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"Harness compilation failed:\n{result.stdout}{result.stderr}"
            )
        cls.hook = ctypes.CDLL(str(harness_library))
        cls.hook.operatorx_test_reset.argtypes = [ctypes.c_char_p, ctypes.c_int]
        cls.hook.operatorx_test_reset.restype = None
        cls.hook.operatorx_test_forward_count.argtypes = []
        cls.hook.operatorx_test_forward_count.restype = ctypes.c_uint
        cls.hook.operatorx_fp8_guard_configure.argtypes = [ctypes.c_uint32] * 3 + [
            ctypes.c_int
        ]
        cls.hook.operatorx_fp8_guard_configure.restype = ctypes.c_int
        cls.hook.operatorx_fp8_guard_last_json.argtypes = []
        cls.hook.operatorx_fp8_guard_last_json.restype = ctypes.c_char_p
        cls.hook.aclrtBinaryGetFunctionByEntry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        cls.hook.aclrtBinaryGetFunctionByEntry.restype = ctypes.c_int
        cls.hook.aclrtLaunchKernelWithHostArgs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(Placeholder),
            ctypes.c_size_t,
        ]
        cls.hook.aclrtLaunchKernelWithHostArgs.restype = ctypes.c_int

    def validate(self, raw=None, *, name=KERNEL, key=5124, shape=(264, 2304, 7168)):
        raw = self.original if raw is None else raw
        data = ctypes.create_string_buffer(raw)
        code = self.lib.operatorx_fp8_guard_validate(name, key, data, len(raw), *shape)
        details = json.loads(self.lib.operatorx_fp8_guard_last_json().decode())
        return code, details

    def test_original_failure_capture_exceeds_declared_capacity(self):
        self.assertEqual(len(self.original), 160)
        code, details = self.validate()
        self.assertEqual(code, 1)
        self.assertEqual(details["l0a_bytes"], 69632)

    def test_exact_declared_struct_and_extra_dfx_bytes_agree(self):
        for raw in (
            self.original[:152],
            self.original,
            self.original[:152] + b"\xff" * 8,
        ):
            with self.subTest(length=len(raw)):
                code, details = self.validate(raw)
                self.assertEqual(code, 1)
                self.assertEqual(details["l0a_bytes"], 69632)

    def test_nearby_tile_and_exact_capacity_are_permitted(self):
        for m, extent in ((240, 61440), (256, 65536)):
            with self.subTest(m=m):
                raw = changed(self.original, m=m, baseM=m)
                code, details = self.validate(raw, shape=(m, 2304, 7168))
                self.assertEqual(code, 0)
                self.assertEqual(details["l0a_bytes"], extent)

    def test_large_global_m_is_not_a_skip_heuristic(self):
        raw = changed(self.original, m=65536, baseM=128)
        code, details = self.validate(raw, shape=(65536, 2304, 7168))
        self.assertEqual(code, 0)
        self.assertEqual(details["l0a_bytes"], 32768)

    def test_recognized_kernel_spellings_keep_exact_dispatch_identity(self):
        for suffix in (b"", b"_5124", b"_5124_mix_aic", b"_5124_mix_aiv"):
            with self.subTest(suffix=suffix):
                self.assertEqual(self.validate(name=KERNEL + suffix)[0], 1)
        for name, key in (
            (KERNEL, 0),
            (KERNEL, 5125),
            (KERNEL + b"_5125", 5124),
            (b"SomeOtherKernel_5124", 5124),
            (KERNEL + b"_5124_extra", 5124),
        ):
            with self.subTest(name=name, key=key):
                self.assertEqual(self.validate(name=name, key=key)[0], 2)

    def test_invalid_contract_and_shape_fail_closed(self):
        mutations = [
            dict(batchA=2),
            dict(quantA=16),
            dict(quantB=32),
            dict(groupM=128),
            dict(groupN=64),
            dict(groupK=64),
            dict(m=263),
            dict(n=1536),
            dict(k=2304),
        ]
        for fields in mutations:
            with self.subTest(fields=fields):
                self.assertEqual(self.validate(changed(self.original, **fields))[0], 2)
        self.assertEqual(self.validate(shape=(0, 2304, 7168))[0], 2)

    def test_every_batch_field_is_checked(self):
        for index in range(15):
            raw = bytearray(self.original)
            struct.pack_into("<I", raw, index * 4, 0)
            with self.subTest(index=index):
                self.assertEqual(self.validate(bytes(raw))[0], 2)

    def test_truncated_and_null_tiling_fail_closed(self):
        for size in (0, 4, 88, 112, 128, 151):
            with self.subTest(size=size):
                self.assertEqual(self.validate(self.original[:size])[0], 2)
        self.assertEqual(
            self.lib.operatorx_fp8_guard_validate(
                KERNEL, 5124, None, 152, 264, 2304, 7168
            ),
            2,
        )

    def test_invalid_tiles_and_extent_overflow_fail_closed(self):
        mutations = [
            dict(baseM=0),
            dict(baseN=0),
            dict(baseK=0),
            dict(baseM=273),
            dict(baseN=129),
            dict(baseK=129),
            dict(baseM=0xFFFFFFF0, baseK=0xFFFFFFE0),
        ]
        for fields in mutations:
            with self.subTest(fields=fields):
                self.assertEqual(self.validate(changed(self.original, **fields))[0], 2)

    def launch(
        self,
        raw=None,
        *,
        name=KERNEL,
        key=5124,
        mode=1,
        shape=(264, 2304, 7168),
        placeholders=None,
        name_failure=False,
        tiling_offset=80,
    ):
        raw = self.original if raw is None else raw
        self.hook.operatorx_test_reset(name, int(name_failure))
        self.assertEqual(self.hook.operatorx_fp8_guard_configure(*shape, mode), 0)
        handle = ctypes.c_void_p()
        self.assertEqual(
            self.hook.aclrtBinaryGetFunctionByEntry(None, key, ctypes.byref(handle)), 0
        )
        args = ctypes.create_string_buffer(b"\0" * tiling_offset + raw)
        entries = [(64, tiling_offset)] if placeholders is None else placeholders
        data = (Placeholder * len(entries))(*(Placeholder(*entry) for entry in entries))
        code = self.hook.aclrtLaunchKernelWithHostArgs(
            handle, 18, None, None, args, tiling_offset + len(raw), data, len(entries)
        )
        return (
            code,
            self.hook.operatorx_test_forward_count(),
            json.loads(self.hook.operatorx_fp8_guard_last_json()),
        )

    def test_preflight_blocks_safe_and_oversized_native_launches(self):
        for raw, shape, expected in (
            (self.original, (264, 2304, 7168), "over_capacity"),
            (
                changed(self.original, m=240, baseM=240),
                (240, 2304, 7168),
                "within_capacity",
            ),
        ):
            with self.subTest(expected=expected):
                code, forwarded, details = self.launch(raw, shape=shape)
                self.assertEqual((code, forwarded), (100000, 0))
                self.assertEqual(details["decision"], expected)
                self.assertFalse(details["forward_allowed"])
                self.assertEqual(details["tiling_offset"], 80)
                self.assertEqual(details["placeholder_address_offset"], 64)

    def test_enforce_forwards_only_valid_safe_target(self):
        for raw, shape, expected in (
            (self.original, (264, 2304, 7168), (100000, 0)),
            (changed(self.original, m=240, baseM=240), (240, 2304, 7168), (0, 1)),
        ):
            with self.subTest(expected=expected):
                code, forwarded, _ = self.launch(raw, shape=shape, mode=2)
                self.assertEqual((code, forwarded), expected)

    def test_unknown_matmul_and_dispatch_fail_closed_before_forwarding(self):
        for name, key in ((b"DifferentQuantMatmul_5124", 5124), (KERNEL, 5125)):
            with self.subTest(name=name, key=key):
                code, forwarded, details = self.launch(name=name, key=key, mode=2)
                self.assertEqual((code, forwarded), (100000, 0))
                self.assertEqual(details["decision"], "unverified")

    def test_name_lookup_failure_blocks_before_forwarding(self):
        code, forwarded, details = self.launch(name_failure=True)
        self.assertEqual((code, forwarded), (100000, 0))
        self.assertEqual(details["reason"], "kernel_name_unavailable")

    def test_known_nonmatmul_preparation_kernel_can_forward(self):
        code, forwarded, details = self.launch(name=b"DynamicBlockQuant_fp8")
        self.assertEqual((code, forwarded), (0, 1))
        self.assertEqual(details["decision"], "pending")
        self.assertFalse(details["observed"])

    def test_inactive_mode_preserves_native_launch(self):
        code, forwarded, details = self.launch(mode=0, name_failure=True)
        self.assertEqual((code, forwarded), (0, 1))
        self.assertEqual(details["decision"], "inactive")

    def test_large_global_m_launch_uses_selected_tile(self):
        raw = changed(self.original, m=65536, baseM=128)
        code, forwarded, details = self.launch(raw, shape=(65536, 2304, 7168), mode=2)
        self.assertEqual((code, forwarded), (0, 1))
        self.assertEqual(details["l0a_bytes"], 32768)
        self.assertEqual(details["base_m"], 128)

    def test_tiling_location_comes_from_placeholder(self):
        code, forwarded, details = self.launch(tiling_offset=96)
        self.assertEqual((code, forwarded), (100000, 0))
        self.assertEqual(details["decision"], "over_capacity")
        self.assertEqual(details["l0a_bytes"], 69632)
        self.assertEqual(details["tiling_offset"], 96)

    def test_invalid_and_ambiguous_placeholders_never_forward(self):
        for entries in ([], [(64, 241)], [(239, 80)], [(64, 80), (64, 80)]):
            with self.subTest(entries=entries):
                code, forwarded, details = self.launch(placeholders=entries, mode=2)
                self.assertEqual((code, forwarded), (100000, 0))
                self.assertEqual(details["decision"], "unverified")


if __name__ == "__main__":
    unittest.main()
