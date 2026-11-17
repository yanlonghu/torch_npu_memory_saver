# torch_npu_memory_saver

Release managed Ascend device memory and restore it at the same virtual addresses.
Extracted from [TokenSpeed PR #3530](https://github.com/lightseekorg/tokenspeed/pull/3530)
at `269b59a9`. The native allocator retains ABI 2. No PyPI package has been published.

## Install

Use a matching Torch / torch_npu installation and load the CANN environment.
Build a wheel with `python -m pip wheel --no-deps .`, then install it with
`python -m pip install --no-deps <wheel>`. Do not rely on the dependency resolver
to choose a qualified Torch/PTA/CANN combination. Native builds require Linux,
CANN headers/libraries, CMake and a C++17 compiler. CPU-only state tests do not.
The package locates its bundled `.so`; `TORCH_NPU_MEMORY_SAVER_LIBRARY` can
explicitly override it with an existing absolute path. Source-tree use requires
this override. Import torch_npu and select the device before the first operation.

The tested primary stack is A2, Torch 2.9, CANN 9.0 and PTA 9d40ac3 plus
!39221, !45270, !30625, !32154, !32564 and !38431 from Ascend/pytorch on GitCode.
The standalone wheel and TokenSpeed TP2 integration were validated on this stack;
see [validation coverage](docs/validation.md). PTA 2.13 has not been requalified
for this extraction.
Set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`; do not set
`PYTORCH_NO_NPU_MEMORY_CACHING=1`.

## API and contract

```python
import torch
import torch_npu
from torch_npu_memory_saver import memory_saver

torch.npu.set_device(0)
with memory_saver.region(tag="weights", enable_cpu_backup=True):
    tensor = torch.ones(1024, device="npu")
# Stop new work and coordinate other ranks before calling pause.
memory_saver.pause("weights")
memory_saver.resume("weights")
```

`region(tag, enable_cpu_backup)` routes allocations through owned PTA MemPools.
This minimal extraction retains fixed tags: weights (backup/discard), kv_cache
(discard), acl_graph (backup/discard). `allocation_region(PoolKey(...))` exposes
separate pool scopes under one release tag. REMAIN uses the normal allocator.
`npu_graph(graph, *, pool, stream, tag, policy, pool_scope,
auto_dispatch_capture)` captures into an owned pool. `pause(tag=None)` and
`resume(tag=None)` target one tag or all tags. `assert_active()` rejects paused
or failed regions. `stats()` reports allocator accounting;
`validate_allocations(allocations, graphs=...)` checks storage and pool ownership.
Each allocation record is `(tag, name, pointer, byte_size)`.

Backup restores bytes at the same virtual address. Discard restores mappings
with undefined contents: initialize before use. Graph executables and HCCL
resources stay resident. All captured persistent state must be backed up or
resident, and replay must overwrite discarded temporaries before reading them.
One device/context per process; all instances share pools. Nested scopes are
rejected. Callers must drain and block new work; these guards do not intercept
arbitrary tensor access. A failed transition can be partial. Do not reopen work
until recovery is complete; callback/rollback faults require worker restart.

This library does not implement model loaders, KV invalidation, weight commits,
multi-rank protocol 3, communication initialization, or process-group shutdown.
TokenSpeed retains those responsibilities. It does not depend on TokenSpeed or
on the CUDA `torch_memory_saver` package.

## Host tests

Run `PYTHONPATH=python python -m pytest tests`. Native failure tests:
`cmake -S native -B build-test -DSAVER_FAKE_ACL=ON`, then build and run CTest.
Host tests do not establish NPU device correctness.

On an exclusively assigned NPU, run the independent graph test with
`RUN_NPU_MEMORY_SAVER_TESTS=1 python -m pytest tests/test_npu_graph.py`. It checks
three backup/discard cycles, same addresses, and replay after poisoning owned
discard storage. This is opt-in even when a native library is installed.
