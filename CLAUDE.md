# Tessera: operating contract

Tessera is a userspace GPU hypervisor. It lets several unmodified CUDA applications share one GPU with per-tenant memory quotas, weighted compute shares, and a tail-latency SLO for an interactive tenant. Components: `libtessera` (driver-API shim), `tesserad` (per-GPU scheduler daemon), `tessera` (CLI), `libcuda_fake` (CPU-only fake driver for tests), an eval harness, and docs. The shim, daemon, and CLI are C++23. The eval harness, Modal runner, and plots are Python.

Every agent (orchestrator or subagent) reads this file first. The mission brief lives in the orchestrator's context. The parts that bind everyone are below.

---

## 1. Invariants (non-negotiable)

- I-1 Measured means measured. No estimated, extrapolated, or "expected" numbers in docs, READMEs, or commit messages. If it was not run, it is not a number.
- I-2 Unmodified tenants. Tenant applications (PyTorch scripts, vLLM, Triton kernels) run with zero source changes. The only intervention is environment variables set by `tessera run`.
- I-3 Correctness before control. A tenant's outputs are identical with and without Tessera, except timing. A deterministic CUDA test suite checks bitwise equality; vLLM greedy outputs on a fixed prompt set are compared with any mismatch explained.
- I-4 Hot path discipline. When credits are available, the per-launch path in the shim performs no syscalls, no locks, no heap allocation, and no driver calls beyond the forwarded launch. Verified by test (strace syscall count in steady state = 0; allocation hook count = 0).
- I-5 Fail open, never deadlock. If `tesserad` is dead or stale, tenants continue unthrottled after a bounded timeout. Tessera may make a tenant slower; it may never hang it.
- I-6 Scope is one GPU, one node, CUDA 12.4+, Linux x86-64 for GPU runs, arm64 Linux for CPU-only development. No multi-GPU, no NCCL, no ROCm, no Kubernetes until the M4 stretch and only with HC-3 approval.
- I-7 Scope changes require an ADR in `docs/DECISIONS.md` before the code.
- I-8 Budget caps in section 4 are hard. Stop and ask at 80%.
- I-9 Exit codes are never masked (see section 0).
- I-10 Every neighbor in `docs/RELATED.md` gets a precise delta, not a dismissal.

"Section 0" in I-9 is the rule that every script propagates failures. `scripts/lint_exit_codes.sh` rejects `|| true`, bare `set +e`, and `exit 0` after a failed step.

Standing rules that go with the invariants:

- Never announce a gate as GREEN unless every criterion's artifact is linked from the gate doc.
- Never edit gate criteria to make them pass. If a criterion is wrong, write an ADR explaining why and get human approval at the next checkpoint.
- Never invent a number. Every number in any doc links to a file under `results/` that records the commit hash, GPU model, driver version, CUDA version, and the exact command.
- Cite prior art. Never hide it.

---

## 2. Gate protocol

Each milestone ends with `docs/gates/Mn.md`. It contains:

- Status: GREEN, AMBER, or RED.
- A criteria table with these columns: criterion, target, measured value, artifact path, PASS/FAIL.
- The risks discovered, with the mitigation chosen for each.
- GPU spend for the milestone. The human records it from Modal's usage page if the API is unavailable. We record what we launched and how long it ran (`results/spend/ledger.jsonl`).
- What the next milestone needs from the human.

The status values mean:

- **GREEN:** every criterion passes.
- **AMBER:** all hard criteria pass, a soft criterion failed, and a memo explains the binding constraint and the proposed lever.
- **RED:** a hard criterion failed. Write `docs/decisions/Mn-red.md` with the options and stop.

Human checkpoints (HC) are hard stops. At each one, print a short summary and wait.

- HC-0 after M0: approve the delta statement, the interposition-strategy ADR, and the M1 budget.
- HC-1 after M1: choose the lever ladder for M2 and M3 from the measured mechanisms.
- HC-2 after M2: approve the macro-eval design and the M3 spend.
- HC-3 after M3: approve the writeup direction and decide whether the Kubernetes stretch is in scope.
- HC-4 after M4: final review.

Checkpoint report format, in this order, under 40 lines:

1. Milestone status line.
2. Criteria table.
3. GPU spend vs cap.
4. The three most important numbers measured this milestone.
5. Open risks.
6. The exact question(s) the human must answer.

---

## 3. Subagent protocol

- The orchestrator plans each milestone as work packages (WP-n.m). Each WP has an owner subagent, inputs, outputs, an acceptance test, and the list of invariants it touches.
- Subagents do not edit gate docs (`docs/gates/`), ADRs (`docs/DECISIONS.md`, `docs/decisions/`), or `CLAUDE.md`. They deliver code, tests, and a short WP report. If a subagent believes an ADR or criterion is wrong, it says so in its report.
- Independent WPs run in parallel. The typical split is shim, daemon, fake driver, bench kernels, eval harness, and docs. **Stay inside the directories your WP owns.** If you need a change elsewhere (for example a top-level CMake hook), make the smallest possible edit and list it in your report.
- Every WP lands as a small set of commits with conventional messages, for example `shim: hook cuGetProcAddress_v2 with per-thread-stream flag handling`. Subagents do **not** run `git commit`. The orchestrator reviews each diff against the invariants and commits it, split per WP.
- Unit tests run on every commit. GPU tests are marked (`gpu` CTest label / `@pytest.mark.gpu`) and run only through the Modal runner.
- A WP that touches the shim hot path must include the I-4 verification test.

---

## 4. Environment and budget

### Environment as verified in M0 (supersedes the mission brief where they differ)

| Item | Brief assumed | Verified (2026-09-19) |
|---|---|---|
| Local host | arm64, Ubuntu 24.04, no GPU | **x86-64**, Ubuntu 26.04, 8 vCPU, 15 GiB RAM, no GPU |
| Local disk | n/a | `/` is **100% full**. Root can write about 4 GiB of reserved blocks. See "Disk rules" below. |
| Compilers | clang 18+, gcc 13+ | gcc 15.2 and clang-20 (apt). clang-format 19.1.7. Minimum versions (clang-18, gcc-13) are exercised in the Modal image and CI. |
| arm64 coverage | native arm64 dev box | Cross-compile (`aarch64-linux-gnu-g++` 15, `clang-20 --target=aarch64-linux-gnu`) and run the CPU tests under `qemu-aarch64`. The CI config also targets native `ubuntu-24.04-arm`. |
| `cuda.h` source | `cuda-driver-dev` | `cuda.h` and `cudaTypedefs.h` ship in **`cuda-cudart-dev-12-6`**. `cuda-driver-dev-12-6` has only `stubs/libcuda.so`, which is useful as the driver's export list (640 `cu*` symbols). Both are fetched by `scripts/fetch_cuda_headers.sh` into `$TESSERA_DEPS` (default `~/.cache/tessera-deps`). They are SHA-256-verified against NVIDIA's `Packages` index and never committed. |
| Modal | CLI installed and authenticated | `modal` 1.5.5 at `~/.local/share/pipx/venvs/modal`, authenticated. GPU default `L4`, override with `TESSERA_GPU`. |

### Disk rules (local host)

- Build trees live on tmpfs: `build/` is a symlink to `/tmp/tessera-build`. Never point a build directory at `/root`.
- Never delete anything outside this repo and `/tmp/tessera-*`. The other projects on this disk belong to the human.
- Use at most `-j4` per build, because several agents may build at once and RAM is 15 GiB.
- Keep `results/` to small JSON and text files. No large blobs.

### GPU runs (Modal)

- One Modal container = one GPU = one "box". It holds `tesserad`, the MPS daemon in shared mode, all tenants, and the load clients.
- Every function has a timeout. Always tear down. Never leave a container idle with a GPU attached.
- Every GPU launch appends a line to `results/spend/ledger.jsonl` with: milestone, app id, GPU type, start, end, wall seconds, command, and exit code.
- Modal may not permit `nvidia-smi compute-policy` or the MPS control daemon. If MPS does not start in-container, the M1 MPS rows run on a root-capable hourly instance (Lambda or RunPod), charged against the same budget. Never skip those rows silently.

### GPU budget (hard caps; stop and ask at 80%)

| Milestone | Cap |
|---|---|
| M0 | $5 |
| M1 | $25 |
| M2 | $25 |
| M3 | $75 |
| M4 | $25 |
| Total | $155 |

---

## 5. Repo layout

```
tessera/
  CLAUDE.md
  README.md
  docs/
    ARCHITECTURE.md   RELATED.md   DECISIONS.md   RISKS.md   EVAL.md
    mechanisms.md     TALKTRACK.md blog-draft.md
    gates/M0.md .. M4.md
    decisions/        (RED memos, ADR long forms)
  shim/        libtessera: src/, include/, exports.map
  daemon/      tesserad
  cli/         tessera
  common/      shm page layout, wire protocol, config schema, clock, logging
  fake_driver/ libcuda_fake: CPU simulation of the hooked driver subset
  bench/       CUDA probe kernels and microbenchmarks (C++/CUDA)
  eval/        Python harness: scenarios, load clients, metrics, plots
  infra/       modal_gpu.py, Dockerfile, versions.lock, run scripts
  tests/       GoogleTest (CPU) and GPU-marked tests
  results/     committed JSON summaries and run metadata (no large blobs)
  scripts/     lint_exit_codes.sh, format, tidy
```

### Toolchain rules

- C++23, CMake 3.28+, clang 18+ (gcc 13+ must also build).
- Build with `-Wall -Wextra -Werror`.
- clang-format and clang-tidy are enforced.
- CPU tests have ASan/UBSan and TSan configurations.
- Only header-only dependencies ship in product code: toml++ for config, and fmt if `<format>` is not used. GoogleTest is a test-only dependency, fetched by CMake at a pinned hash.
- No exceptions cross the C ABI.
- Symbol visibility is hidden by default. Only `cu*` symbols and the shim's own `tessera_*` exports are public.

### Common commands

```
scripts/fetch_cuda_headers.sh            # one-time; CUDA 12.6 headers + driver stub into $TESSERA_DEPS
cmake --preset gcc-debug && cmake --build --preset gcc-debug -j4 && ctest --preset gcc-debug
cmake --preset clang-asan  ...           # ASan+UBSan
cmake --preset clang-tsan  ...           # TSan
cmake --preset arm64-gcc   ...           # cross build; ctest runs under qemu-aarch64
scripts/check_all.sh                     # full local matrix + lint + format check
modal run infra/modal_gpu.py             # GPU-marked tests on Modal (costs money; ledger it)
```

### Shim traps (read before touching `shim/` or `fake_driver/`)

- cudart does not call the exported `cu*` symbols. It resolves everything through `cuGetProcAddress(_v2)`, including the lookup function itself. Hook the lookup, or the shim sees nothing.
- `dlsym` on an explicit handle ignores `LD_PRELOAD`. That is why masquerading as `libcuda.so.1` is the primary strategy (ADR-001).
- The `_ptsz`/`_ptds` variants are distinct symbols. `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM` must resolve to them.
- Never call the driver from a static constructor. Initialize lazily on the first hooked call.
- During stream capture, never gate and never inject events.
- Fork: `pthread_atfork` handlers reset per-process state. Locks held by other threads at fork time must not deadlock the child.
