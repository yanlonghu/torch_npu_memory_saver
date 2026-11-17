"""State utilities must work without PyTorch or an inference runtime."""

from pathlib import Path
import subprocess
import sys


def test_import_and_state_without_optional_runtime():
    source = Path(__file__).resolve().parents[1] / "python"
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import torch_npu_memory_saver
from torch_npu_memory_saver.state import PoolState, PoolKey, AllocationPolicy
assert not any(name in sys.modules for name in ("torch", "torch_npu", "tokenspeed"))
assert PoolKey("weights", AllocationPolicy.DISCARD, "target").tag == "weights"
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", code, str(source)], check=True)
