# Validation coverage

The initial extraction retains the allocator implementation from TokenSpeed
`269b59a95f258188ce380933ef98f418bea2abe6`. The standalone library does not
implement TokenSpeed's model loading, cache invalidation or multi-rank protocol.

## Tested stack

- Atlas A2 / 910B3, Python 3.12, Torch 2.9.0+cpu, CANN/HCCL 9.0.0.
- PTA baseline [9d40ac3](https://gitcode.com/Ascend/pytorch/commit/9d40ac3a2b3714df7e5d164bda28a42eff3282cd)
  with official backports [!39221](https://gitcode.com/Ascend/pytorch/merge_requests/39221),
  [!45270](https://gitcode.com/Ascend/pytorch/merge_requests/45270),
  [!30625](https://gitcode.com/Ascend/pytorch/merge_requests/30625),
  [!32154](https://gitcode.com/Ascend/pytorch/merge_requests/32154),
  [!32564](https://gitcode.com/Ascend/pytorch/merge_requests/32564), and
  [!38431](https://gitcode.com/Ascend/pytorch/merge_requests/38431).
- Installed PTA identifies itself as `2.9.0+gitee1742b`, git
  `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d`.

## 0.1.1 full-suspend cache trimming

On the A2 stack above, a wheel built from the standalone source distribution
passed all 14 Python tests, including the three-cycle device graph test, and
all four native allocator tests. A partial pause kept the cache intact. A full
pause trimmed it once, including after retrying an injected trim failure.

With the TokenSpeed thin adapter over [`674ca800`](https://github.com/lightseekorg/tokenspeed/commit/674ca8006cfb23a2a2b144702112293dba912c56),
Qwen3-0.6B TP2 completed three suspend/resume cycles with HCCL AIV and
CPU-backed weights. Both prompts matched all 16 pre-suspend output tokens in
each cycle. Both ranks loaded the wheel's bundled allocator and exited cleanly.

## Results

| Check | Result |
| --- | --- |
| Independent Linux aarch64 wheel build and install | Passed. Built from this library's source distribution. |
| Standalone NPU graph | Passed three backup/discard cycles with stable addresses and replay after poisoning discarded temporary storage. TokenSpeed imports were forbidden. |
| Native error-state tests | CTest 4/4 passed on Linux using fake ACL, including failed allocation rollback and wrong-device/context free callbacks. |
| TokenSpeed TP2 partial `cpu_backup` update | Passed. Two checkpoint tensors updated; output matched a fresh complete effective checkpoint for batches 1, 3 and 4. |
| TokenSpeed TP2 complete `discard` reload | Passed. 311 checkpoint tensors loaded; both ranks committed before KV wake. Output matched a fresh new-checkpoint start. |
| Original graph and address retention | Passed for both weight policies, including two additional same-weight poison/restore cycles. No graph recapture after startup. |
| TokenSpeed protocol 3 | Both ranks negotiated v3. Holding one rank after its device operation prevented a public success reply for 300 ms; releasing it permitted completion. |
| Installed native-library provenance | All six integration ranks mapped the standalone wheel's library. Both library-path overrides were unset. |
| Worker cleanup | All successful cases exited normally and released their device leases. |

The installed Saver library was freshly compiled. Unchanged TokenSpeed host and
scheduler binaries were reused. No PTA, Rust or system-library modifications were
made for this validation. The protocol test verifies return/acknowledgement gating,
not asymmetric HCCL drain timing or native acknowledgement timestamps.

One initial protocol probe failed before device execution because two external
test helpers shared a module name. A dedicated observer module fixed the test;
no library or TokenSpeed production change was needed. The retry passed.

## Limits

This extraction has not repeated TP4/TP8, cross-host, Mooncake transport,
extended speculative-decoding, long-cycle leak or performance campaigns.
PTA 2.13 compatibility has not been revalidated. CUDA/ROCm isolation was checked
through unchanged source boundaries and CPU behavior; GPU hardware was unavailable.

Run the repository's host and opt-in NPU tests as described in the README.
TokenSpeed integration scenarios require its separate adapter and runtime.
