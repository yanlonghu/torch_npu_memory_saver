"""Exercise worker-visible failures at the Python/native boundary."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torch_npu_memory_saver import saver
from torch_npu_memory_saver.state import PoolState


def test_cache_trim_callback_fault_fails_pause(monkeypatch):
    library = Mock()
    library.saver_abi_version.return_value = 4
    library.saver_pause.return_value = 0
    library.saver_has_callback_fault.return_value = 0
    library.saver_stats.return_value = b'{"blocks":[{"phase":"paused","tag":"weights"}],"callback_error":"free failed"}'
    monkeypatch.setattr(saver.ctypes, "CDLL", lambda _: library)
    backend = saver.NativeBackend("test")
    state = PoolState(backend=backend, make_pool=Mock(), synchronize=Mock())

    def empty_cache():
        library.saver_has_callback_fault.return_value = 1

    monkeypatch.setattr(
        saver.torch, "npu", SimpleNamespace(empty_cache=empty_cache), raising=False
    )
    with pytest.raises(RuntimeError, match="restart this worker: free failed"):
        state.pause("weights")
    with pytest.raises(RuntimeError, match="Memory is unavailable"):
        state.assert_active()


def test_keep_communicator_alive_until_native_detach_succeeds(monkeypatch):
    native = Mock()
    native.close_hccl.side_effect = [RuntimeError("detach failed"), None]
    communicator = object()
    state = SimpleNamespace(
        native=native, synchronize=lambda: None, hccl_group=communicator
    )
    monkeypatch.setitem(saver._states, os.getpid(), state)
    controller = saver.NpuMemorySaver()

    with pytest.raises(RuntimeError, match="detach failed"):
        controller.close_hccl()
    assert state.hccl_group is communicator
    controller.close_hccl()
    assert state.hccl_group is None
    controller.close_hccl()
    assert native.close_hccl.call_count == 2
