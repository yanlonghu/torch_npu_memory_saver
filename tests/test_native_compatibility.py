# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Load real shared libraries to verify the native safety contract."""

import ctypes
import shutil
import sys
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from torch_npu_memory_saver import saver


@pytest.fixture(scope="module")
def native_compiler():
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native compatibility tests require a C++17 compiler")
    return compiler


@pytest.fixture(scope="module")
def current_native_library(tmp_path_factory, native_compiler):
    build = tmp_path_factory.mktemp("current-native")
    library = build / "libascend_memory_saver.so"
    native = Path(__file__).resolve().parents[1] / "native"
    subprocess.run(
        [
            native_compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-DSAVER_FAKE_ACL=1",
            f"-I{native / 'include'}",
            f"-I{native / 'tests'}",
            str(native / "src/ascend_saver.cpp"),
            str(native / "tests/fake_acl.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


def test_current_native_library_is_accepted(current_native_library):
    backend = saver.NativeBackend(str(current_native_library))
    backend.check_health()
    assert backend.stats()["abi_version"] == 5
    assert backend.stats()["segments"] == 0


@pytest.mark.parametrize("version", [2, 3, 4, 6, None])
def test_override_rejects_incompatible_native_library(
    monkeypatch, tmp_path, native_compiler, version
):
    # Keep the production filename: a path/name check cannot detect a stale build.
    library = tmp_path / "libascend_memory_saver.so"
    source = tmp_path / "legacy.cpp"
    definition = (
        f'extern "C" int saver_abi_version() {{ return {version}; }}'
        if version is not None
        else 'extern "C" int legacy_library() { return 0; }'
    )
    source.write_text(definition)
    subprocess.run(
        [native_compiler, "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setenv("TORCH_NPU_MEMORY_SAVER_LIBRARY", str(library))
    with pytest.raises(RuntimeError, match="requires native ABI 5"):
        saver.NativeBackend(saver._library_path())


@pytest.mark.parametrize("backup", [False, True])
def test_kv_pool_policy_restores_or_discards_real_bytes(
    monkeypatch, current_native_library, backup
):
    # Only PTA's unavailable MemPool constructors are represented locally. The
    # policy selection, allocator callbacks and storage transitions are real.
    class NativeAllocator:
        def __init__(self, library, allocate, free):
            self.library = ctypes.CDLL(library)
            self.allocate = getattr(self.library, allocate)
            self.allocate.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
            self.allocate.restype = ctypes.c_void_p

        def allocator(self):
            return self

    class NativePool:
        def __init__(self, allocator):
            self.allocator = allocator

    memory = ModuleType("torch_npu.npu.memory")
    memory.NPUPluggableAllocator = NativeAllocator
    memory.MemPool = NativePool
    memory.use_mem_pool = None
    npu = ModuleType("torch_npu.npu")
    npu.memory = memory
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setitem(sys.modules, "torch_npu.npu", npu)
    monkeypatch.setitem(sys.modules, "torch_npu.npu.memory", memory)
    monkeypatch.setenv("TORCH_NPU_MEMORY_SAVER_LIBRARY", str(current_native_library))
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.delenv("PYTORCH_NO_NPU_MEMORY_CACHING", raising=False)
    device = saver.DeviceState(0)
    policy = (
        saver.AllocationPolicy.CPU_BACKUP if backup else saver.AllocationPolicy.DISCARD
    )
    owned = device.make_pool(saver.PoolKey("kv_cache", policy, "decode"))
    pointer = owned.allocator.allocate(7000, 0, None)
    assert pointer
    release = device.native.library.saver_release
    release.argtypes = [ctypes.c_void_p]
    release.restype = ctypes.c_int
    try:
        for value in (0x19, 0x57, 0xA3):
            ctypes.memset(pointer, value, 7000)
            assert device.native.library.saver_pause(b"kv_cache") == 0
            stats = device.native.stats()
            assert stats["physical_bytes"] == 0
            assert stats["blocks"][0]["tag"] == "kv_cache"
            assert stats["host_bytes"] == (8192 if backup else 0)
            assert device.native.library.saver_resume(b"kv_cache") == 0
            expected = value if backup else 0xCD
            assert ctypes.string_at(pointer, 7000) == bytes([expected]) * 7000
            assert device.native.stats()["host_bytes"] == 0
    finally:
        assert release(pointer) == 0
