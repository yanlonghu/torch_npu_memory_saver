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

"""Run with a built saver library on one exclusively assigned NPU."""

import ctypes
import os
from unittest.mock import Mock

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_NPU_MEMORY_SAVER_TESTS") != "1",
    reason="requires RUN_NPU_MEMORY_SAVER_TESTS=1 and an exclusively assigned NPU",
)
def test_graph_discard_preserves_persistent_buffers(monkeypatch):
    import torch

    pytest.importorskip("torch_npu")
    from torch_npu_memory_saver import NpuMemorySaver

    torch.npu.set_device(0)
    saver = NpuMemorySaver()
    graph = torch.npu.NPUGraph()
    with saver.region(tag="weights", enable_cpu_backup=True):
        constant = torch.full((128,), 7.0, device="npu")
    inputs = torch.arange(128, device="npu", dtype=torch.float32)
    for _ in range(3):
        inputs * inputs + constant
    torch.npu.synchronize()

    def blocks():
        stats = saver.stats()
        assert not stats["callback_fault"]
        return [block for block in stats["blocks"] if block["tag"] == "acl_graph"]

    try:
        with saver.npu_graph(
            graph,
            pool=None,
            stream=torch.npu.Stream(),
            tag="acl_graph",
            policy="discard",
            pool_scope="test_graph_discard",
            auto_dispatch_capture=True,
        ):
            output = inputs * inputs + constant
        graph.replay()
        torch.npu.synchronize()
        expected = output.cpu()
        addresses = {block["ptr"] for block in blocks()}
        assert addresses
        output_ptr = output.data_ptr()
        memset = ctypes.CDLL("libascendcl.so").aclrtMemset
        memset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        memset.restype = ctypes.c_int
        trim = Mock(wraps=torch.npu.empty_cache)
        monkeypatch.setattr(torch.npu, "empty_cache", trim)
        for cycle in range(3):
            saver.pause("weights")
            assert trim.call_count == cycle
            if cycle == 0:
                trim.side_effect = RuntimeError("injected trim failure")
                with pytest.raises(RuntimeError, match="injected trim failure"):
                    saver.pause("acl_graph")
                trim.side_effect = None
                trim.reset_mock()
            saver.pause("acl_graph")
            saver.pause()
            assert trim.call_count == cycle + 1
            paused = blocks()
            assert saver.stats()["physical_bytes"] == 0
            assert saver.stats()["valid_backup_bytes"] > 0
            assert {block["ptr"] for block in paused} == addresses
            assert all(
                block["reserved"]
                and not block["physical_live"]
                and not block["mapped"]
                and not block["backup_valid"]
                for block in paused
            )
            saver.resume()
            resumed = blocks()
            assert {block["ptr"] for block in resumed} == addresses
            # Poison only owned discard storage: replay must overwrite all reads.
            for block in resumed:
                assert block["policy"] == "discard" and block["mapped"]
                size = block["aligned_bytes"]
                assert memset(int(block["ptr"], 16), size, 0xA5, size) == 0
            graph.replay()
            torch.npu.synchronize()
            assert output.data_ptr() == output_ptr
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(constant.cpu(), torch.full((128,), 7.0))
    finally:
        saver.resume()
        graph.reset()
        torch.npu.synchronize()


@pytest.mark.skipif(
    os.environ.get("RUN_NPU_MEMORY_SAVER_TESTS") != "1",
    reason="requires RUN_NPU_MEMORY_SAVER_TESTS=1 and an exclusively assigned NPU",
)
def test_kv_backup_preserves_contents_and_virtual_address():
    import torch

    pytest.importorskip("torch_npu")
    from torch_npu_memory_saver import NpuMemorySaver
    from torch_npu_memory_saver.state import AllocationPolicy, PoolKey

    torch.npu.set_device(0)
    saver = NpuMemorySaver()
    key = PoolKey("kv_cache", AllocationPolicy.CPU_BACKUP, "test_kv_backup")
    with saver.allocation_region(key):
        tensor = torch.empty(4096, dtype=torch.int32, device="npu")
    pointer = tensor.data_ptr()
    try:
        for value in (19, 57, 163):
            tensor.fill_(value)
            torch.npu.synchronize()
            expected = tensor.cpu()
            saver.pause("kv_cache")
            blocks = [
                block
                for block in saver.stats()["blocks"]
                if block["tag"] == "kv_cache"
                and int(block["ptr"], 16)
                <= pointer
                < int(block["ptr"], 16) + block["aligned_bytes"]
            ]
            assert len(blocks) == 1
            assert blocks[0]["backup_valid"] and not blocks[0]["physical_live"]
            saver.resume("kv_cache")
            assert tensor.data_ptr() == pointer
            torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
    finally:
        saver.resume("kv_cache")
