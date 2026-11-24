# torch_npu_memory_saver

Release managed Ascend device memory and restore it at the same virtual addresses.
Extracted from [TokenSpeed PR #3530](https://github.com/lightseekorg/tokenspeed/pull/3530)
at `269b59a9`, with full-suspend cache trimming from
[`a7d040f0`](https://github.com/lightseekorg/tokenspeed/commit/a7d040f047de4b44adac617b41cd5476cc0c60c9).
Build the Python package and its ABI 3 native allocator from the same revision. Source releases are available on
[PyPI](https://pypi.org/project/torch-npu-memory-saver/).

## Dependencies

| Component | Requirement / tested configuration |
| --- | --- |
| Hardware and OS | Validated on Atlas A2 / 910B3, Linux aarch64. |
| Python | Package requires 3.11+; device validation used 3.12. |
| PyTorch | Device validation used `torch==2.9.0+cpu`, paired with PTA below. |
| PTA (`torch_npu`) | `9d40ac3` (2.9-730) **with six official backports**; a stock version number alone is insufficient. |
| CANN / HCCL | Validated with 9.0.0 and the deployment's installed Ascend driver. |
| Native build | CANN development headers and `acl_rt`, CMake 3.18+, a C++17 compiler. |
| Python build | Setuptools 69+, wheel; pip provisions these and CMake in its build environment. |
| Tests | Pytest 8+; the real-device test additionally needs the configured NPU stack. |

The six PTA backports are
[!39221](https://gitcode.com/Ascend/pytorch/merge_requests/39221),
[!45270](https://gitcode.com/Ascend/pytorch/merge_requests/45270),
[!30625](https://gitcode.com/Ascend/pytorch/merge_requests/30625),
[!32154](https://gitcode.com/Ascend/pytorch/merge_requests/32154),
[!32564](https://gitcode.com/Ascend/pytorch/merge_requests/32564), and
[!38431](https://gitcode.com/Ascend/pytorch/merge_requests/38431), applied to
[9d40ac3](https://gitcode.com/Ascend/pytorch/commit/9d40ac3a2b3714df7e5d164bda28a42eff3282cd).
This is the validated deployment prerequisite, not a claim that every backport
is needed for every standalone operation. See [validation coverage](docs/validation.md)
for the installed build identity and integration results. PTA 2.13 and other
hardware/CANN combinations have not been requalified for this extraction.

Package metadata declares `torch>=2.9` and `torch-npu>=2.9` as minimum versions,
**not a compatibility matrix**. Pip cannot verify the required source patches
or choose a matching Torch/PTA/CANN stack. Prepare that stack first, then use
`--no-deps` below to preserve it. This package does not install CANN, drivers,
TokenSpeed, Triton or CUDA's `torch_memory_saver`.

## Install with pip

With the matching Torch/PTA stack installed and CANN loaded, install the
0.1.3 source revision for HCCL memory registration on Atlas A3:

```bash
python -m pip install --no-deps \
  "torch-npu-memory-saver @ git+https://github.com/yanlonghu/torch_npu_memory_saver.git@95b6ee6f67e1fa2d4ca2357ffeba7fc9eeb972c0"
```

Pip builds the native library locally against the configured CANN toolkit.
This requires Git, the CANN development headers and libraries, CMake 3.18+
and a C++17 compiler. For an offline deployment, build a wheel from the
same revision on a matching system. The earlier 0.1.1 release is available
on PyPI for deployments using saver ABI 2.

### Build and keep a wheel

In an environment with the matching Torch/PTA installation:

```bash
git clone https://github.com/yanlonghu/torch_npu_memory_saver.git
cd torch_npu_memory_saver

# Replace this with your deployment's CANN toolkit directory.
export ASCEND_HOME_PATH=/path/to/cann
source "$ASCEND_HOME_PATH/set_env.sh"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
unset PYTORCH_NO_NPU_MEMORY_CACHING

python -m pip wheel --no-deps . --wheel-dir dist
python -m pip install --no-deps dist/torch_npu_memory_saver-0.1.3-*.whl
python -c 'from importlib.metadata import version; print(version("torch-npu-memory-saver"))'
```

If your allocation configuration has other options, preserve them while setting
`expandable_segments:False`. Set the environment before starting the worker.
Build on the target platform against its CANN installation. The wheel contains
the Saver `.so`, not CANN or system libraries; those must remain available at
runtime. Do not use editable installation as a substitute for the wheel build.

The package locates its bundled `.so` automatically. For a separate native build,
`TORCH_NPU_MEMORY_SAVER_LIBRARY` may override it with an existing absolute path.
The old `TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY` variable is not used. Import
`torch_npu` and select the device before the first allocation or transition.

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
# Do not read or write tensor while its physical backing is released.
memory_saver.resume("weights")
torch.testing.assert_close(tensor.cpu(), torch.ones(1024))
```

For discard, use `enable_cpu_backup=False`. After `resume()`, write all required
contents before reading the tensor or replaying a graph that consumes it.
`resume()` restores mappings; it does not reload or commit model weights.

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

For multi-rank Atlas A3 workers, call `memory_saver.configure_hccl(local_group)`
on every rank of the node-local HCCL group before the first managed allocation.
The runtime registers each segment before a collective can use it, deactivates
it before unmapping and reactivates the new physical handle before graph replay.
Drain outstanding work and call `memory_saver.close_hccl()` before destroying
the group. Single-rank and Atlas A2 workers use the ordinary ACL VMM path.

Starting in 0.1.1, the last pause that leaves all existing managed blocks
paused calls `torch.npu.empty_cache()` once per worker. Partial pauses leave
the cache alone. The next cycle starts when a previously trimmed tag resumes.
Version 0.1.2 reports allocator callback failures during trimming to the caller
before the pause returns.

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

## Tests

Host Python tests require Torch and pytest but no NPU. Use a separate CPU test
environment if needed; do not replace the Torch installation in a deployment:

```bash
python -m pip install 'pytest>=8'
PYTHONPATH=python python -m pytest -q tests

# C++ state and error injection tests do not require Torch or CANN.
cmake -S native -B build-test -DSAVER_FAKE_ACL=ON
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

Host tests do not establish NPU device correctness. They skip the real NPU test
unless explicitly enabled.

On an exclusively assigned NPU, run the independent graph test with
`RUN_NPU_MEMORY_SAVER_TESTS=1 python -m pytest tests/test_npu_graph.py`. It checks
three backup/discard cycles, same addresses, and replay after poisoning owned
discard storage. This is opt-in even when a native library is installed.

## Publishing

Maintainers publish source distributions with the manually triggered
`.github/workflows/publish.yml` workflow on `main`. The `pypi` GitHub environment
uses PyPI Trusted Publishing; no long-lived API token is required. Register
owner `yanlonghu`, repository `torch_npu_memory_saver`, workflow `publish.yml`,
and environment `pypi` in PyPI before the first run. Increment the project
version for subsequent releases; PyPI does not allow replacing a release file.

The workflow validates package metadata and uploads only the source archive.
Installation compiles the native library against the user's configured CANN
toolkit. It does not imply support for untested Torch/PTA/CANN combinations.
