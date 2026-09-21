# Related work

Tessera is a userspace GPU hypervisor: a driver-API shim plus a per-GPU daemon that
gives several *unmodified* CUDA applications per-tenant memory quotas, weighted
compute shares, and a tail-latency SLO for an interactive tenant. This file records
the neighbors that already occupy parts of that space, what mechanism each one
actually uses, and what separates it from Tessera.

Three rules govern the entries below.

- **No numbers of ours.** Tessera has measured nothing yet. Every number here comes
  from someone else's paper or vendor document and is attributed inline to it. The
  only Tessera-side facts stated are repository facts that were checked by running a
  command, and they are marked as such.
- **Deltas are precise, not dismissive** (I-10). Where a neighbor is strictly
  stronger than Tessera on some axis, the delta says so plainly. Many of them are.
- **A delta is not a claim of superiority.** At M0 a delta is a statement about *what
  a system does and what it requires of its users*, not about which is faster.

---

## Research systems

### REEF — OSDI '22

REEF is a GPU-accelerated DNN inference serving system built for the case where one
real-time tenant shares a GPU with best-effort tenants. It is the strongest published
result on *preemption* as the isolation mechanism rather than partitioning or rate
control.

REEF's central observation is that DNN inference kernels are "mostly idempotent," so a
best-effort kernel that is killed mid-flight can simply be re-executed. On that basis
REEF implements *reset-based preemption*: it evicts host queues and device queues, and
then instructs the command processor to kill all running kernels on the compute units
while preserving their state in GPU memory. A second mechanism, *dynamic kernel
padding*, co-executes best-effort blocks alongside real-time blocks up to a computed
padding limit so the real-time task does not lose throughput to an idle GPU.

Two requirements of REEF bear directly on Tessera, and both are easy to miss.
First, REEF requires every served model to be recompiled through its own compiler:
§6 states that REEF "extends Apache TVM, a machine learning compiler framework, with a
code transformer, which mainly adds two modifications to the source code of DNN
inference: (1) a preemption flag, which is injected into kernel arguments to lazily
evict the kernel; (2) a set of proxy kernels, which is constructed for the padded
kernels." Second, the closed-source-GPU variant is weaker than it first appears. For
NVIDIA GPUs the authors ported **REEF-N**, which by their own statement cannot reset
compute units — §4.4: "The primary limitation is that we cannot reset CUs to
proactively kill running kernels." REEF-N "first wraps each GPU stream, the general
abstraction provided by GPU runtime, into a virtual host queue (vHQs), which intercepts
and buffers all launched kernels"; §6 records that it "intercepts three CUDA APIs
related to kernel launch and stream management" — `cuStreamCreate`, `cuKernelLaunch`
and `cuStreamSynchronize` — and "provides a new API `cuResetHQ` to reset vHQ by
dequeuing all buffered kernels." §4.4 then states the decisive fact: REEF-N "still
follows the lazy eviction to reset DQs, and then waits for all running kernels to
complete." The paper's remark that "the lazy eviction scheme proposed by REEF for
resetting DQs ... does not require any modification to the GPU runtime" applies to
lazy eviction alone, and lazy eviction works only because the TVM code transformer has
injected the preemption flag into every kernel's arguments.

Han et al. report that on their AMD Radeon Instinct MI50 testbed REEF incurs less than
2% overhead in end-to-end latency for real-time tasks while increasing overall
throughput by up to 7.7× compared to dedicating the GPU to real-time tasks; the paper's
introduction states the same result as up to 4.3×. §7.2 shows where the larger figure
comes from — on the Apollo real-world autonomous-driving trace, "compared to RT-Only,
REEF achieves 7.7× throughput improvement with less than 2% latency overhead" — so the
7.7× is the real-world-trace result and the two statements are not measuring the same
workload set. Preemption latency on the MI50 stays "less than 40 µs," while §7.3 reports
that on the NVIDIA V100, because REEF-N "does not reset CUs to proactively kill running
kernels, the preemption latency just ranges from 71 µs to 288 µs, which still outperforms
the wait-based approach by up to 12.3× (from 6.3×)."

- Paper: <https://www.usenix.org/conference/osdi22/presentation/han>
- PDF: <https://www.usenix.org/system/files/osdi22-han.pdf>
- Code: <https://github.com/SJTU-IPADS/reef>

**Delta: REEF on an open stack is strictly stronger than Tessera on preemption — it
kills and restarts a running kernel at tens of microseconds, which Tessera cannot do at
all — but it buys that with a modified GPU runtime and driver (AMD ROCm) *and* with
every model recompiled through its extended TVM, so it is no more applicable to an
unmodified tenant binary than Paella is. REEF-N, the NVIDIA-side variant, is the more
honest comparison and it lands where Tessera does: it waits for running kernels to
complete, and the authors measure its preemption latency at 71–288 µs on a V100. That
number is the published evidence for the bound Tessera's Open Question 3 is about, and
REEF-N still needs TVM-recompiled models, which Tessera's I-2 forbids.**

### Paella — SOSP '23

Paella is a low-latency model-serving framework that replaces the GPU's built-in
scheduling with a software scheduler, on the argument that FIFO policies baked into the
CUDA runtime, driver, and hardware cause head-of-line blocking and cannot express an
application's job-completion-time targets.

Paella is explicitly a *co-design* of three components: a compiler, a client library,
and a dispatcher. Users compile their models with Paella's modified TVM-based compiler,
which automatically instruments each kernel with monitoring code that exports, at
runtime, where and when the kernel was placed and how occupied the GPU's SMs are. That
view is sampled rather than continuous: the injected prologue emits a placement record
only when `startCount % 16 == 0` or on the last block, and §5.2 says the batching is
deliberate — "we batch start and end notifications such that each notification signals
the start/end of a group of up to 16 thread blocks." The policy is named in §6: a
scheduling strategy "based on shortest-remaining-processing-time (SRPT)," seeded by
profiling runs at model submission and refined online, with per-user deficit counters
taking over "if any user's deficit exceeds" an unfairness threshold. Around this sits a
set of specialized communication channels: clients write raw input vectors into a
shared-memory queue, and the dispatcher notifies the client of impending completion over
an IPC socket so the client only begins polling for results at the right moment — a
hybrid interrupt/polling protocol chosen to keep submission and retrieval off the
latency critical path. The per-model burden is bounded and the paper states it: "clients
adding a new model write only the TVM model definition and the job adaptor of
Section 4.2."

- Paper (ACM DL, DOI printed in the paper itself): <https://doi.org/10.1145/3600006.3613163>
- PDF: <https://vincen.tl/files/ng23paella.pdf>
- Code: <https://github.com/eniac/paella>

**Delta: Paella gets per-kernel scheduling visibility and an SRPT policy that Tessera
will not have, and its client burden is smaller than "rewrite your application" — a TVM
model definition plus a job adaptor. But that is still a recompile through Paella's
instrumenting compiler and a submission path through its client library, which is
exactly the tenant modification Tessera's I-2 forbids. Where Tessera is different rather
than better: it must work on a binary it has never seen, and therefore has no occupancy
signal at all.**

### Orion — EuroSys '24

Orion shares a GPU between a high-priority ML job and best-effort jobs by scheduling
individual operators according to whether they are compute-bound or memory-bound, so
that co-located kernels contend for different resources.

Orion is a dynamically linked library that intercepts GPU operations submitted by an
application framework and buffers them in per-client software queues, then submits them
to the hardware under its own policy. Concretely, the PyTorch prototype — about 3000
lines of C++/CUDA, per the paper — intercepts CUDA *runtime* API calls by overriding
them with wrapper functions: `cudaLaunchKernel` plus memory operations
(`cudaMalloc`, `cudaMemcpy`, `cudaMemset`, `cudaFree`) and selected cuDNN and cuBLAS
entry points for convolution, batch normalization, and GEMM. §5.1.2 names the two
implementation mechanisms — GPU stream priorities, with high-priority kernels going out
on a dedicated high-priority stream, and non-blocking `cudaEventQuery` for progress
tracking. The admission rule gates a best-effort kernel on `SM_THRESHOLD` and on
`DUR_THRESHOLD`, "a tunable percentage of the high-priority" request latency. The
scheduling decision uses each kernel's compute-versus-memory profile, which Orion
collects in an *offline profiling phase* before the run, but the system is not purely
offline: §5.1.1 says `SM_THRESHOLD` "can also be tuned dynamically" by "monitoring the
throughput of the high-priority job and adjusting the `SM_THRESHOLD` with binary
search." For operations that force device synchronization (`cudaMalloc`, `cudaFree`)
Orion synchronizes all clients to avoid invalid memory accesses. In the evaluated
prototype the client applications and the Orion scheduler are threads of the same
process, which gives in-process memory sharing and fast communication.

On numbers, care is needed about what the paper attributes to what. The abstract's
"granularity of 10s of µs" describes the *problem* — "current approaches are not
sufficiently fine-grained or interference-aware to maximize GPU utilization while
minimizing interference at the granularity of 10s of µs" — while Orion's own stated
granularity is "the granularity of individual operators." Strati et al. report wrapper
overhead of less than 1%, best-effort inference throughput gains of up to 7.3×, and
training-cost savings of up to 1.49× versus dedicated allocation. The paper also notes
that Orion assumes the cluster manager only co-locates jobs that fit in GPU memory.

Orion's §6.4 publishes a result that directly concerns Tessera's Open Question 2: after
applying compute/memory profiles and kernel size-based scheduling, "the stream priority
mechanism has only marginal improvements at this point, hence Orion can also be used in
settings where the GPU hardware does not support stream priorities (e.g., in MPS mode)."

- Paper (ACM DL, DOI printed in the paper itself): <https://doi.org/10.1145/3627703.3629578>
- PDF: <https://fotstrt.github.io/files/2024-orion.pdf>

**Delta: Orion schedules at operator granularity using per-kernel compute/memory
profiles and hooks the CUDA runtime plus cuDNN/cuBLAS with clients as threads of one
process, whereas Tessera hooks the driver API beneath separate tenant processes with no
profiling step. Orion is stronger than Tessera on interference-awareness — it knows
whether the next operator is compute- or memory-bound and Tessera will not — and its
`SM_THRESHOLD` is already closed-loop against measured high-priority throughput, so it
is not the static-profile system it is sometimes described as. What Tessera trades that
for is working on a tenant binary it has never seen, in a separate process, without a
framework integration.**

### TGS — NSDI '23

TGS (Transparent GPU Sharing) shares a GPU between a production container and an
opportunistic container in a container cloud, and is the clearest example of putting
the sharing layer *below* the application rather than inside it.

The abstract says "TGS operates at the OS layer beneath containers," but the
interposition point is the one that matters here and §5 is specific: "TGS intercepts
CUDA driver API calls related to CUDA kernel launch from containers for rate monitoring
and rate control" — the same interposition point Tessera uses. The monitored quantity is
CUDA blocks, not kernels: "TGS uses a global counter to record the number of CUDA blocks
launched in a given time period," and because a kernel's block count is an argument of
the driver-API call, "the number of pending CUDA blocks can be treated as a real-time
signal to estimate the performance of production jobs." The control loop is asymmetric.
A rate monitor thread in the *production* container reads that counter and ships the
value to a rate-control thread in the *opportunistic* container, which "defers the
kernel launch if the rate of the opportunistic container exceeds the rate limit"; the
adaptation law is additive-increase/multiplicative-decrease with a slow-start phase,
converging on the tipping point. *Transparent unified memory* handles the case where the
two jobs together exceed device memory: TGS intercepts allocation calls such as
`cuMemAlloc` and "replaces these calls with unified memory allocation calls using
`cuMemAllocManaged`," uses `cuMemAdvise` to pin the production container's preferred
location to the GPU, and uses `cuMemPrefetchAsync` to pull the opportunistic container's
pages back once the production container exits. TGS was built and integrated with Docker
and Kubernetes; Wu et al. report that it "provides similar throughput for opportunistic
jobs as the state-of-the-art application-layer solution AntMan, and improves their
throughput by up to 15× compared to the existing OS-layer solution MPS."

- Paper: <https://www.usenix.org/conference/nsdi23/presentation/wu>
- PDF: <https://www.usenix.org/system/files/nsdi23-wu.pdf>
- Code: <https://github.com/pkusys/TGS>

**Delta: TGS is the closest neighbor in spirit and shares Tessera's interposition point
exactly — CUDA driver-API kernel-launch interception beneath an unmodified container. It
targets a two-class production/opportunistic split for DL *training*, while Tessera
targets N tenants with weighted shares, explicit memory quotas, and a tail-latency SLO
for an interactive tenant. TGS is also strictly stronger than Tessera on memory
elasticity: by swapping allocations for `cuMemAllocManaged` it lets the opportunistic
job spill to host memory rather than fail, whereas Tessera's hard per-tenant quota
returns out-of-memory by construction. Tessera claims no novelty in the launch-rate
lever itself.**

### Clockwork — OSDI '20

Clockwork is a distributed model-serving system that achieves predictable tail latency
by removing sources of nondeterminism rather than by reacting to them. It is the
canonical statement that DNN inference is deterministic enough to schedule ahead of
time.

Clockwork's method is *consolidating choice*: it eschews reactive and best-effort
mechanisms and centralizes all resource-consumption and scheduling decisions in one
controller. Workers hold models in RAM and maintain exclusive control over one or more
GPUs; the central controller has a global view, decides when each model is loaded into
GPU memory and when each request executes, and maintains a minimal advance schedule.
A request is only admitted if the controller is confident it can meet its latency SLO,
and if a worker cannot execute a scheduled action it aborts the request immediately and
moves to the next one at the specified time rather than letting the delay propagate.
Predictability is protected down to the runtime: Clockwork extracts and links the CUDA
modules a model needs and disables JIT compilation and CUDA kernel caching, and the
models never allocate — "at runtime, models do not directly allocate memory; instead,
Clockwork will pre-allocate and manage all GPU memory and pass pointers as arguments to
function calls." The paper is explicit that this depends on owning the bottleneck:
"Clockwork assumes workers have exclusive control over their machine, and dedicated
GPUs." Gujarati et al. report supporting thousands of models while meeting 100 ms
latency targets for 99.9999% of requests.

- Paper: <https://www.usenix.org/conference/osdi20/presentation/gujarati>
- PDF: <https://www.usenix.org/system/files/osdi20-gujarati.pdf>

**Delta: Clockwork achieves its tail-latency guarantees by assuming exclusive, dedicated
GPUs and by loading models through its own worker runtime that pre-allocates all GPU
memory — which is precisely the assumption Tessera removes, and precisely the lever
Tessera gives up. Tessera's interactive tenant must get an SLO on a GPU that other,
uncooperative tenants are actively using, and Tessera must honour those tenants' own
`cuMemAlloc` calls rather than replacing them. Clockwork is strictly stronger on
predictability, on a problem Tessera has deliberately made harder.**

### AntMan — OSDI '20

AntMan is the production co-location system behind Alibaba's DL cluster, and the
application-layer counterpart to TGS's OS-layer argument. TGS's headline comparison is
against AntMan, so it belongs in this file on its own terms rather than as a name in
someone else's abstract.

AntMan "co-designs cluster schedulers with deep learning frameworks and has been
deployed in production at Alibaba to manage tens of thousands of daily deep learning
jobs across thousands of GPUs." Jobs are classified by the global scheduler into
resource-guarantee jobs and opportunistic jobs, and the local coordinator throttles the
latter to protect the former. The compute lever is `GpuOpManager`: "when a GPU operator
is ready to execute, it is added to `GpuOpManager` instead of being directly launched.
The main idea of `GpuOpManager` is to control the launching frequency by delaying the
execution of GPU operators." The memory lever is a modified allocator — `BFCAllocator`
(`CUDACachingAllocator` in PyTorch) gains "an adjustable upper limit for memory," and a
new `UniversalAllocator` "tries to allocate the memory using the GPU memory allocator
and treats the CPU memory allocator as a backup if there is insufficient GPU memory
left over." The price is stated plainly in §4.1: the mechanisms "are implemented in two
popular deep learning frameworks, TensorFlow and PyTorch"; "the implementation in
TensorFlow takes 4000 lines of code (mostly in C++)" and "the implementation in PyTorch
takes about 2000 lines of code."

- Paper: <https://www.usenix.org/conference/osdi20/presentation/xiao>
- PDF: <https://www.usenix.org/system/files/osdi20-xiao.pdf>

**Delta: AntMan already does what Tessera proposes — throttle a batch tenant's GPU
launches to protect a latency-sensitive one, with a memory lever that spills to host
memory — and it does it in production at a scale Tessera will never reach. It is
strictly stronger than Tessera on memory elasticity, for the same reason TGS is. The
one real difference is where the lever sits: AntMan buys both levers by modifying
TensorFlow and PyTorch, 4000 and 2000 lines respectively, which is the line I-2 forbids
Tessera from crossing. A tenant running an unmodified framework, a closed-source
binary, or a framework AntMan has not been ported to gets nothing from it.**

### Tally — ASPLOS '25

Tally is the nearest published neighbor on the *transparency* axis: performance
isolation for concurrent DL workloads with no changes to the applications.

Tally is "a transparent virtualization layer between the application level and the GPU"
that "intercepts GPU kernel launches" and, critically, also intercepts device code. Its
kernel transformer works "through a series of transformation passes on the kernel device
code (PTX code in the case of NVIDIA GPUs) obtained through the interception of device
code registration." Two primitives are built that way: *slicing*, which divides a large
kernel into sub-kernels that recompute their block offsets, and a *preemption*
transformation inspired by the persistent-thread-block model. A transparent profiler
measures the transformed kernels under various launch configurations and feeds a
priority-aware scheduler. The evaluation is a direct head-to-head against the baselines
Tessera will also face: §5.3 compares Tally against "(i) Time-Slicing, (ii) MPS,
(iii) MPS-Priority, and (iv) TGS," on 99th-percentile latency of a high-priority
inference task and system throughput. The abstract reports "an overhead of only 7.2% on
the 99th-percentile latency of high-priority inference tasks when executed concurrently
with best-effort training workloads, compared to 188.9% overhead exhibited by the
state-of-the-art GPU sharing systems like TGS, while achieving over 80% of TGS's system
throughput."

- Paper (ACM DL, DOI printed in the paper itself): <https://doi.org/10.1145/3669940.3707282>
- PDF (arXiv): <https://arxiv.org/abs/2410.07381>

**Delta: Tally reaches thread-block granularity on unmodified binaries — strictly finer
control than Tessera can reach, and on the same class of tenants — because it rewrites
the tenant's own PTX at registration time. That is the step Tessera declines: I-3
requires bitwise-identical tenant output, and transforming a tenant's device code puts
the burden of proving that on Tessera rather than on the hardware. Tally is therefore
not a weaker system that Tessera improves on; it is the same problem solved by a means
Tessera has ruled out, and its published numbers against Time-Slicing, MPS, MPS-Priority
and TGS are the bar any launch-gating scheduler should expect to be measured against.**

### XSched — OSDI '25

XSched is the existence proof that a preloaded shim on a stock closed driver *can*
preempt a running kernel, which is the premise Tessera's Open Question 3 assumes away.

XSched provides "unified interfaces for scheduling XPU tasks through a preemptible
command queue abstraction (XQueue)," and its transparency story is Tessera's: "XSched
provides a shim layer that intercepts commands from applications using" the driver API,
"by intercepting XPU driver API calls and redirecting commands to the XQueue. The
approach provides transparency, allowing applications to run on XSched without
modifications." The paper's multi-level hardware model is the useful part for Tessera.
Level 1 preempts pending commands "by simply blocking their launch" — the lever Tessera
has. Level 2 prevents in-flight commands from executing, but §4.3 concedes that "Lv2
preemption still requires waiting for the running command to complete, which leads to
unpredictable preemption latency." Level 3 targets the *running* command, and on NVIDIA
GPUs the authors get there by rewriting kernels "at runtime using dynamic binary
instrumentation (DBI)," rewriting "the first instruction of each kernel," leveraging
"the hidden" GPU instruction and constant memory, and by a driver path they state they
"discovered": "a specific ioctl that triggers GPU interrupts." XSched also situates the
device-code-rewriting lineage: "EffiSha and FLEP enable preemption through GPU kernel
transformation, which requires general-purpose programmability," and "REEF and a prior
work depend on a microcontroller function on AMD or ARM GPUs to reset compute units."

- Paper: <https://www.usenix.org/conference/osdi25/presentation/shen-weihang>
- PDF: <https://ipads.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf>

**Delta: XSched is strictly stronger than Tessera on preemption and it gets there from
the same starting position — a preloaded shim over an unmodified driver with unmodified
applications — so Tessera cannot claim that userspace interposition bounds control
granularity at a kernel's duration. XSched's own Level 2 is bounded that way; its Level
3 is not, and it escapes the bound by runtime binary instrumentation of the tenant's
kernels, use of undocumented GPU instruction and constant memory, and an undocumented
driver ioctl. Tessera's honest position is that it declines those means, not that they
do not exist, and that Level 1 — blocking launches — is the level Tessera operates at.**

### Hummingbird — arXiv 2601.04071, February 2026

Hummingbird is the closest 2026 neighbor on both of Tessera's axes at once: driver-API
interception of unmodified applications, with an explicit tail-latency SLO as the
objective rather than a by-product.

The abstract presents it as "an SLO-oriented GPU scheduling system that overcomes these
challenges by enabling microsecond-scale preemption on closed-source GPUs while
effectively harvesting idle GPU time slices," where "the SLO is defined as the 99th
percentile (P99) latency for exclusive execution." The interposition is described in §5:
"Hummingbird ensures generality and transparency across diverse ML ecosystems by
intercepting low-level CUDA Driver APIs without requiring application modifications... It
redirects APIs like kernel launches (e.g., `cuLaunchKernel`) and memory allocations
(e.g., `cuMemAlloc`) to custom wrappers." The preemption, though, comes from device-code
rewriting done at runtime: Hummingbird implements "PTX kernel transformation at runtime,"
hooking `cuModuleLoad` and `cuModuleGetFunction`, dumping and pruning the GPU binary to
PTX, modifying "the kernel parameter list to accept additional offset parameters," and
realigning "the thread block indices by injecting `add` instructions to shift the native
`blockIdx`," then re-assembling with `ptxas`. The authors report that Hummingbird
"improves the SLO attainment of high-priority tasks by 9.7× and 3.5× compared to the
state-of-the-art spatial and temporal-sharing approaches," that against exclusive
execution the high-priority task's SLO attainment "only drops by less than 1%," and that
low-priority throughput "outperforms the state-of-the-art temporal-sharing approaches by
2.4×." Its comparison set includes MPS, MIG, multi-streams, Orion, REEF and LithOS.

- Abstract: <https://arxiv.org/abs/2601.04071>
- PDF: <https://arxiv.org/pdf/2601.04071>

**Delta: Hummingbird already holds a P99 SLO for a nominated high-priority tenant from
driver-API interception of unmodified applications on a stock NVIDIA driver, which is
the objective and the interposition point Tessera has chosen. It is strictly ahead of
Tessera there, and Tessera's delta against it cannot be "we have an SLO objective." The
real difference is the same one as with Tally and XSched: Hummingbird earns its
microsecond preemption by rewriting the tenant's PTX at load time and shifting
`blockIdx`, which Tessera's I-3 bitwise-equality invariant declines. What Hummingbird
does not offer is per-tenant memory quotas, weighted cross-tenant shares, or a stated
fail-open policy, and it has no peer-reviewed venue confirmed as of this writing.**

### Vitamin-E — arXiv 2603.15042, current version v4 (5 August 2026)

Vitamin-E attacks the determinism/utilization tradeoff in GPU spatial sharing: fixed
bindings strand capacity, while reshaping a kernel's parallel structure to fit
available resources can change its output bits. It is the neighbor closest to
Tessera's correctness invariant (I-3) *and* to Tessera's architecture.

**Version discipline matters for this entry.** The arXiv record carries four versions —
v1 (16 Mar 2026), v2 (17 Mar 2026), v3 (3 Apr 2026), v4 (5 Aug 2026) — and the system
was renamed across them. In v2 it is called **DetShare** and the abstract promises
"complete transparency (zero code modification)" with figures of up to 79.2% higher
training throughput, 15.1% lower P99 inference tail latency, 69.1% lower average
inference latency and 21.2% fewer TPOT SLO violations. The current version, v4, is
titled "Determinism-Preserving GPU Spatial Sharing with Vitamin-E" and reports different
figures: "Across all workload–baseline comparisons, Vitamin-E achieves up to 3.50× the
aggregate normalized LLM training throughput, 62.5% lower inference p99 latency, and
1.43× the background-training throughput," plus a TPOT-First policy that "reduces TPOT
SLO violations by up to 46.1% over Throughput-Oriented on three serving workloads."
Every claim below is read from v4 unless it says otherwise.

The paper's *parallel-structure invariant* is that for fixed-structure deterministic
workloads, keeping each launch immutable makes its output bits independent of physical
width. The system therefore late-binds immutable launches to a pool of physical
contexts instead of reshaping them. The v4 implementation (§6) is roughly "6K lines of
C++ and 1K lines of CUDA": "an LD_PRELOAD library intercepts CUDA launches, stream and
event operations, memory operations, and synchronization calls" and constructs immutable
descriptors, while "a coordination daemon invokes pluggable policies through the Pick,
Select, and Prepare callbacks and returns scheduling decisions." "Clients and the daemon
exchange compact records through preallocated, lock-free shared-memory rings, using
batched processing and idle-only wake-ups." A physical context is defined in §4.3 as
"an NVIDIA Green Context plus associated backend streams and work-queue state," drawn
from a pre-created hierarchical pool; reclamation closes dispatch gates and drains,
with a lookahead mechanism that begins reclaiming before an anticipated wider operation
becomes ready.

Two corrections to an earlier reading of this paper are worth stating explicitly,
because they ran through this file and through its open questions. First, the claim that
the system "orchestrates those contexts over MPS to guarantee robust address space
isolation" is **v2 text and does not describe v4**. In v4 MPS appears as a comparison
baseline — "we additionally implement MPS-Pool, a flat baseline that maintains a
separately provisioned MPS context for each supported resource share," and "for
colocated training, we compare Vitamin-E with temporal sharing, MPS, MIG, Orion, and
Salus." v4 §8 states instead that "kernel safety and cross-tenant fault containment are
orthogonal." Second, the system's name is no longer an open question: v2 DetShare, v4
Vitamin-E, with v4 current.

- Abstract and version history: <https://arxiv.org/abs/2603.15042>
- Full text, current version (v4): <https://arxiv.org/html/2603.15042v4>
- Full text, superseded version (v2, system named DetShare): <https://arxiv.org/html/2603.15042v2>

**Delta: Vitamin-E is the nearest published neighbor to Tessera's architecture, not a
distant one. An LD_PRELOAD interception library plus a coordination daemon exchanging
records over lock-free shared-memory rings, driving NVIDIA Green Contexts on a stock
driver under a tail-latency objective, while treating bitwise output equality as the
protected invariant, is Tessera's design, already built and already evaluated against
temporal sharing, MPS, MIG, Orion, Salus and a LithOS reproduction. On the mechanism
Tessera intends to evaluate in M1 it is strictly ahead, and Tessera should expect to be
asked what it adds. What v4 does not describe is per-tenant memory quotas, a fail-open
policy for a dead scheduler, or arbitrary mutually distrusting tenants as the threat
model — v4 explicitly sets fault containment aside as orthogonal — and it is an
unreleased research prototype rather than a shipped artifact. That last point is the
only ground on which Tessera's availability claim still stands, and it is one a code
release would remove.**

### MuxWise (prefill-decode multiplexing) — arXiv 2504.14489, April 2025, revised February 2026

MuxWise is an LLM serving framework that multiplexes prefill and decode phases inside a
single GPU rather than disaggregating them across GPUs or fusing chunked prefill into
decode iterations. It is included here as current evidence of how green contexts behave
in practice under a real serving workload.

MuxWise dynamically adapts compute allocation, decouples compute from memory
management, and executes prefill and decode independently, using a bubble-less
multiplex engine, a contention-tolerant estimator, and an SLO-aware dispatcher. Its
compute allocation is built on CUDA green contexts, and §3.2.1 states why: "the
intra-process approach GreenContext enables low-overhead resource adjustment by binding
CUDA streams to specific SMs, with reconfiguration costing only a stream synchronization
(on the order of microseconds)." The paper reports three operational facts that matter
to anyone planning to use green contexts as a scheduling lever. Creating the partitions
is cheap in memory: "creating a group of green contexts requires only 44MB, which is
negligible compared to the total memory of modern GPUs." Combining them with CUDA Graph
is not free: "integrating it with CUDA Graph incurs a 6.2% overhead for both Llama-8B and
Llama-70B on servers with 8 A100 or 8 H100 GPUs," because the serving system must record
kernel launches per decode batch size per configuration. And bubbles are measurable
rather than merely asserted: the authors report that "MuxWise has a slightly higher
bubble ratio (7.7% vs. 4.5%) due to its fine-grained kernel scheduling," and that these
"extra bubbles occurs when the system is purely processing decode iterations." On
end-to-end results, "MuxWise improves peak throughput under SLO guarantees by an average
of 2.20× (up to 3.06×) over state-of-the-art baselines," and in a bursty interval of a
real-world trace "MuxWise activated all the six configurations within 30s."

- Abstract: <https://arxiv.org/abs/2504.14489>
- Full text (v3): <https://arxiv.org/html/2504.14489v3>

**Delta: MuxWise applies green contexts inside one cooperating serving system that owns
both phases of one model, so it can re-partition on its own schedule — and it reports
that doing so costs only a stream synchronization, on the order of microseconds, which
is a published partial answer to this file's own Open Question 1. Tessera would have to
apply the same mechanism across mutually distrusting processes it does not control,
which is a strictly harder setting and one this paper does not evaluate. MuxWise is
ahead of Tessera on measured green-context behaviour; Tessera has measured none of it.**

### gVirtuS — Euro-Par 2010

gVirtuS is the ancestor of the whole "intercept the CUDA API and make the GPU appear
elsewhere" line of work. It virtualizes GPGPU access so a virtual machine without a
GPU can use a physical one transparently and independently of the hypervisor.

gVirtuS follows a split-driver model with three parts: a *front end* that intercepts the
library calls in the guest, a *back end* that holds the physical device and actually
executes them, and a *communicator* that carries the remote invocation between them. All
three are visible in the repository, which was fetched: the README configures a "GVirtuS
backend configuration file" and a "GVirtuS frontend configuration file" separately, each
with its own `communicator` block. (The Euro-Par paper is the origin citation for the
design but could not be retrieved in this session — see Method.) The communicator is
configurable, and the README documents exactly two transports, "TCP/IP" and "RDMA over
Infiniband," which is what makes the same design serve both VM-to-host virtualization and
outright remote GPU execution for a machine with no GPU at all. The repository's `plugins/` directory ships
eight plugins: `cublas`, `cudadr`, `cudart`, `cudnn`, `cufft`, `curand`, `cusolver` and
`cusparse`. The `cudadr` plugin — the CUDA *driver* API — is the point of contact with
Tessera, since both systems interpose at the driver API and differ in where execution
goes next. The repository is Apache-2.0 licensed. The README's own example
`properties.json` uses a loopback endpoint (`127.0.0.1`, port `9999`), so the
frontend/backend split is not inherently a network hop.

- Paper: <https://doi.org/10.1007/978-3-642-15277-1_37>
- Code: <https://github.com/gvirtus/GVirtuS>

**Delta: gVirtuS established API interception as the transparency mechanism Tessera also
relies on, and it interposes at the same driver API through its `cudadr` plugin. The
difference is what happens after interception: gVirtuS *relocates* execution across a
frontend/backend boundary — which may be a network hop or, in the README's own example
configuration, a loopback socket on one host — and serializes calls over a communicator,
whereas Tessera interposes in-process to *schedule* execution on the local GPU and must
therefore keep the launch path free of syscalls (I-4). gVirtuS solves a problem Tessera
does not have (no local GPU) and does not attempt the one Tessera does (arbitrating
between tenants).**

---

## Vendor mechanisms

### NVIDIA Multi-Process Service (MPS)

MPS is NVIDIA's shipped mechanism for letting multiple processes submit work to one GPU
concurrently. NVIDIA describes it as "a lightweight runtime service, designed to
transparently enable co-operative CUDA multi-process and multi-application workflows on
NVIDIA GPUs." It is the baseline any userspace GPU-sharing system has to beat or build on.

Architecturally, a control daemon starts a server that is "the clients' shared
connection to the GPU and owner of the GPU scheduling resources." On Volta and later,
client contexts manage most hardware resources and submit work to the hardware directly;
the server mediates the remaining shared resources and "stays out of the critical
execution path." Each MPS client owns its own GPU address space. The index page is
explicit about the scope of the feature set: "MPS also provides memory and SM
partitioning capabilities, as well as priority and dynamic resource adjustment."

There are three distinct compute levers, and conflating them is the single easiest way
to understate MPS.

1. **Active thread percentage**, set via `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` or control
   commands, or programmatically through `cuCtxCreate_v3` with a `CUexecAffinityParam`;
   the resulting limit is visible through `cudaDevAttrMultiProcessorCount`. This lever
   reserves nothing, and NVIDIA says so: "Setting the limit does not reserve dedicated
   resources for any MPS client context. It simply limits how much resources can be used
   by a client context. Kernels launched from different MPS client contexts may execute
   on the same SM, depending on load-balancing." Its mutability is asymmetric — the
   uniform limit "is configured for a client process when it starts and cannot be changed
   for the client process afterwards," while the non-uniform per-context limit "is
   configured for every client CUDA context and can be changed throughout the client
   process." Percentages are "internally rounded down to the nearest hardware supported
   thread count limit."
2. **Static SM partitioning**, which *does* reserve. "On NVIDIA Ampere architecture and
   newer GPUs, users can create static SM partitions for MPS clients," and NVIDIA
   describes the feature as providing "deterministic resource allocation and spatial
   isolation between clients by allowing users to explicitly control which SMs each
   client can access." It is enabled at daemon launch with `-S` / `--static-partitioning`
   in Legacy MPS v2, or by creating a server under MPS v3; partitions are allocated in
   architecture-sized chunks (documented as 8 SMs on Hopper and newer, 4 SMs pre-Hopper
   under Legacy v2, 2 SMs pre-Hopper under MPS v3, 2 SMs on iGPU) with
   `sm_partition add` / `lspart` (v2) or `sm-partition create` / `list` / `delete` (v3),
   and selected per client through `CUDA_MPS_SM_PARTITION`. The geometry is fixed before
   the client initialises, and it is mandatory once enabled: "when static partitioning
   mode is enabled, all MPS clients must set the `CUDA_MPS_SM_PARTITION` environment
   variable before creating a CUDA context. Failure to do so will result in context
   creation failing with `CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION`."
3. **Cross-client priority**, which is coarse and advisory: `set_default_client_priority`
   / `CUDA_MPS_CLIENT_PRIORITY` allows only "0 [NORMAL] and 1 [BELOW NORMAL]", and
   "priority values should be considered as hints to the CUDA Driver, not guarantees."
   The same page documents a related capability to "map the stream priorities of a given
   client to a different range of internal CUDA priorities."

Memory can be capped with `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`, after which allocation
calls return out-of-memory. Fault containment is limited: "MPS client processes have
fully isolated GPU address spaces. MPS supports a limited form of error containment,"
and in multi-user server mode "a fatal fault from one client may bring down a different
user's client that shares any GPU with the faulting client." MPS v3 is "a new, opt-in
control daemon interface that replaces the interactive shell of Legacy MPS v2 with a
scriptable module verb command syntax, named servers and namespaces, and file-based
configuration."

- Overview: <https://docs.nvidia.com/deploy/mps/index.html>
- Architecture and error containment: <https://docs.nvidia.com/deploy/mps/latest/architecture.html>
- Provisioning, priority, static SM partitioning, memory limits: <https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html>
- Control-daemon commands: <https://docs.nvidia.com/deploy/mps/latest/mpsv2-interface.html>

**Delta: MPS already provides transparent multi-process sharing, per-client address
spaces, an SM-percentage cap, a memory cap and — on Ampere and newer — genuine spatial
reservation through static SM partitioning, all with zero application changes. That is
more than Tessera's v0, and the static-partitioning mode is strictly stronger than
anything Tessera can offer on spatial isolation, because it is enforced below userspace.
Its costs are that the geometry is fixed before a client initialises, that it is
mutually exclusive with dynamic sharing and forgoes work conservation, and that every
client must opt in or fail to create a context. What MPS has no form of is a weighted
share that moves at runtime under a measured signal, or a tail-latency objective for a
nominated tenant. That, and not "MPS reserves nothing," is the gap Tessera is aiming
at.**

### MPS v3 memory partitioning

From CUDA 13.4, MPS v3 adds per-partition device-memory limits enforced by the driver
rather than by an interposer — the vendor doing below the shim what a shim would do in
userspace.

A hard limit "maps to `dmem.max`. It is enforced at allocation time; an allocation that
would exceed it fails with an out-of-memory error." A soft limit marks a client as an
eviction candidate, and when another client hits the hard limit MPS can force-terminate
an over-soft victim so the requestor can retry. Reporting is virtualized as well as
enforcement: under a partition `cuMemGetInfo` "reports total and free memory capped to
the hard limit and the remaining headroom under it," and `nvmlDeviceGetMemoryInfo_v2`
"reports used, total, and free memory adjusted to account for the soft-limit headroom
guarantee and reserved physical memory." The requirements are specific: "Linux with
cgroup v2 mounted at `/sys/fs/cgroup`, CUDA 13.4 or newer, and a non-MIG device," with
MIG "explicitly unsupported, with memory reporting left unaltered for MIG handles."

- Documentation: <https://docs.nvidia.com/deploy/mps/latest/mpsv3-memory-partitioning.html>

**Delta: this is strictly stronger than Tessera's memory quota wherever it is available,
because the driver enforces it and a tenant cannot bypass it, whereas a userspace rewrite
of `cuMemGetInfo` can be bypassed by any tenant that does not ask. Tessera's remaining
memory claim is correspondingly narrow and should be stated that way: the same behaviour
on a stock CUDA 12.4+ driver with no MPS v3, no CUDA 13.4 floor and no cgroup-v2
dependency, on MIG handles that MPS v3 leaves unaltered, and governed by the same control
plane that governs compute. Tessera's memory mechanism is a portability argument, not a
strength argument.**

### NVIDIA Multi-Instance GPU (MIG)

MIG partitions a supported GPU in hardware into GPU Instances, each exposed to CUDA as a
separate device with its own SMs, memory and fault domain. It is the strongest isolation
available on NVIDIA hardware and the correct answer whenever static partitioning is
acceptable.

The unit of partitioning is the slice. NVIDIA's user guide defines a *GPU memory slice*
as "roughly one eighth of the total GPU memory resources," including both capacity and
bandwidth, and a *GPU SM slice* as "roughly one seventh of the total number of SMs
available." A *GPU Instance* combines memory slices with GPU engines and "provides memory
QoS." A GPU Instance can be subdivided into Compute Instances: "a Compute Instance (CI)
contains a subset of the parent GPU instance's SM slices and other GPU engines (DMAs,
NVDECs, etc.). The CIs share memory and engines" — so the full-isolation claim holds for
GPU Instances and not for Compute Instances within one. Partitions are created
administratively through `nvidia-smi`, the available shapes are fixed by instance
profiles, and the guide notes that "as GPU Instances are created and destroyed at
different locations, fragmentation can occur, and the physical position of one GPU
Instance will play a role in which other GPU Instances can be instantiated next to it."
Two operational facts matter for any lever ladder that stacks mechanisms: MPS composes
with MIG one server per instance, which is why `EXCLUSIVE_PROCESS` compute mode "is not
supported when the GPU is in MIG mode as we use multiple MPS servers (one per MIG GPU
instance)"; and the reconfiguration cost is generation-specific — on Ampere "when MIG
mode is enabled, the driver will attempt to reset the GPU," while "starting with the
Hopper generation of GPUs, enabling MIG mode no longer requires a GPU reset to take
effect."

- User guide: <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/>
- Concepts (slices, QoS, fault isolation): <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html>
- Getting started (MIG mode, compute mode): <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html>

**Delta: MIG is strictly stronger than Tessera on isolation — it partitions memory
bandwidth, cache and fault domains in hardware, which no userspace shim can do — and
Tessera's only claim against it is different in kind: MIG's partitions are coarse, fixed
by instance profile, administratively reconfigured (with a GPU reset on Ampere),
supported only on some GPUs, and cannot be reweighted per request. A work-conserving
software scheduler and MIG answer different questions, and on any workload where static
partitioning is acceptable MIG is the better answer.**

### CUDA green contexts

Green contexts are the newest vendor lever and the one Tessera most needs to
characterize. NVIDIA describes them as a lightweight alternative to traditional
contexts, "with the ability to pass in a set of resources that they should be
initialized with," which "allows the developer to represent distinct spatial partitions
of the GPU, provision resources for them, and target them via the same programming
model that CUDA exposes."

The documented flow is four steps: fetch an initial resource with
`cuDeviceGetDevResource`; partition it with a split API; finalize the specification with
`cuDevResourceGenerateDesc`; provision with `cuGreenCtxCreate`. Work is then targeted at
the partition via `cuGreenCtxStreamCreate`, and `cuCtxFromGreenCtx` converts to an
ordinary context handle.

**The split API differs between the documentation and this repository's pinned
toolkit, and the difference is load-bearing.** The current (13.x) driver-API reference
recommends `cuDevSmResourceSplit`, which is explicit and allows non-equal groups. The
CUDA 12.6.77 headers this repository pins name only `cuDevSmResourceSplitByCount`.
Checked by grepping `cuda.h` and running `nm -D` on `stubs/libcuda.so` under
`$TESSERA_DEPS`, the green-context group present in the pinned headers *and* in the
driver stub's export list is exactly twelve symbols: `cuGreenCtxCreate`,
`cuGreenCtxDestroy`, `cuGreenCtxStreamCreate`, `cuGreenCtxRecordEvent`,
`cuGreenCtxWaitEvent`, `cuGreenCtxGetDevResource`, `cuCtxFromGreenCtx`,
`cuCtxGetDevResource`, `cuDeviceGetDevResource`, `cuDevResourceGenerateDesc`,
`cuStreamGetGreenCtx` and `cuDevSmResourceSplitByCount`. `cuDevSmResourceSplit` is **not**
present in the pinned headers or the pinned stub; it is a later addition. The same
version gap applies to resource classes and to quantization. The pinned 12.6 header
states "only SM type is supported today" and defines only `CU_DEV_RESOURCE_TYPE_SM`, so
the 13.x documentation's workqueue resource class does not exist on the pinned toolkit.
And the pinned header's granularity guideline is stricter and differently worded than
the 13.x page: "On Compute Architecture 6.X: The minimum count is 1 SM. On Compute
Architecture 7.X: The minimum count is 2 SMs and must be a multiple of 2. On Compute
Architecture 8.X: The minimum count is 4 SMs and must be a multiple of 2. On Compute
Architecture 9.0+: The minimum count is 8 SMs and must be a multiple of 8." On the L4
(sm_89) that `CLAUDE.md` names as the default Modal GPU, the 8.X row is what binds — a
4-SM minimum. Any partition size must be read back from the device at runtime rather
than assumed from either page.

The mechanism's attraction and its limit are both documented, and both belong here. The
Programming Guide states the positive property: "a given green context can only use
specific SMs (the ones provisioned during its creation)," so that "kernel A can only use
the SMs provisioned for green context A, irrespective of its launch configuration," and
a latency-critical "kernel B is guaranteed that there will be available SMs for it to
start executing immediately, barring any other resource constraints" — and that "this
behavior can be achieved without any code modifications to kernels A and B." The limit
is equally explicit: "even when different SM resources and work queues are provisioned
per green context, concurrent execution of independent GPU work is not guaranteed," and
NVIDIA's own framing is that "it is best to think of all the techniques described under
the Green Contexts section as removing factors which can prevent concurrent execution
(i.e., reducing potential interference)." The pinned 12.6 header adds that "in certain
scenarios, it is possible for the workload to run on more SMs than was provisioned (but
never less)," giving two: "On Volta+ MPS: When `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is
used, the set of SMs that are used for running kernels can be scaled up to the value of
SMs used for the MPS client," and "On Compute Architecture 9.x: When a module with
dynamic parallelism (CDP) is loaded, all future kernels running under green contexts may
use and share an additional set of 2 SMs." The Programming Guide also contrasts the two
spatial levers directly: "a key difference between MPS in static partitioning mode and
green contexts is that MPS targets different processes, while green contexts is
applicable within a single process too. Also, contrary to green contexts, MPS with
static partitioning does not allow oversubscription of SM resources."

Two 2025–2026 items record where green-context scheduling currently stands, beyond the
Vitamin-E and MuxWise entries above. HAMi-core issue #179, "feat: SM Limiting using CUDA
Green Contexts" (opened 2026-04-22, closed completed 2026-07-27), proposes replacing
HAMi-core's token limiter with driver-enforced allocation, and states the motivation in
the maintainers' own words: "as opposed to the current token-based rate limiter backed
by an NVML-polling watcher thread in HAMi-core, the SM allocation in green contexts is
driver enforced with no polling or per-launch coordination overhead." And Martín, Flich
and Hernández, "Performance Isolation for Inference Processes in Edge GPU Systems"
(arXiv 2601.07600, 12 January 2026), "analyzes the main isolation mechanisms available in
modern NVIDIA GPUs: MPS, MIG, and the recent Green Contexts, to ensure predictable
inference time in safety-critical applications"; the authors report that "MIG provides a
high level of isolation" while "Green Contexts represent a promising alternative for edge
devices by enabling fine-grained SM allocation with low overhead, albeit without memory
isolation."

- Driver API reference: <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GREEN__CONTEXTS.html>
- Programming guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html>
- HAMi-core issue #179: <https://github.com/Project-HAMi/HAMi-core/issues/179>
- Edge GPU isolation study: <https://arxiv.org/abs/2601.07600>

**Delta: green contexts give the SM-level spatial lever Tessera wants, and Tessera does
not improve on the mechanism at all — NVIDIA's own documentation already claims the
property that makes it interesting, an SM reservation honoured without any kernel code
change. What Tessera would add is a policy layer that decides partitions across mutually
distrusting processes under a measured latency signal. Whether that is reachable from a
shim is an open question below, not a claim; and the closest open-source neighbor,
HAMi-core, has an accepted issue to move onto this same lever.**

### CUDA stream priorities

Stream priorities are the cheapest lever in this file and the one Tessera would reach
for first, so they deserve characterizing rather than being named only inside an open
question.

A stream is created at a priority with `cuStreamCreateWithPriority`, and the valid range
is queried with `cuCtxGetStreamPriorityRange`. The semantics are explicitly advisory:
"this affects the scheduling priority of work in the stream. Priorities provide a hint to
preferentially run work with higher priority when possible, but do not preempt
already-running work or provide any other functional guarantee on execution order." The
numeric convention is inverted — "priority follows a convention where lower numbers
represent higher priorities. '0' represents default priority" — and out-of-range values
are silently clamped: "if the specified priority is outside the numerical range returned
by `cuCtxGetStreamPriorityRange`, it will automatically be clamped to the lowest or the
highest number in the range."

- Driver API, stream management: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html>

**Delta: stream priorities are the one lever Tessera could use with no daemon, no
partition and no coordination, and the vendor documentation says they neither preempt
running work nor guarantee ordering — so they cannot by themselves deliver an SLO.
Orion's EuroSys '24 ablation already reports that once compute/memory and size-based
policies are in place "the stream priority mechanism has only marginal improvements," and
that Orion "can also be used in settings where the GPU hardware does not support stream
priorities (e.g., in MPS mode)." Tessera adds nothing to this primitive; it must decide
whether to use it, and a neighbor has already published a partial answer.**

### NVIDIA vGPU scheduling policies

vGPU is the vendor counter-example to the idea that NVIDIA ships levers but not policies:
it ships three named compute schedulers with a documented latency knob.

The user guide documents *best effort*, where "the physical GPU's processing cycles are
shared in a way that aims to balance performance across vGPUs" and which is the default;
*equal share*, where "the physical GPU is shared equally amongst the running vGPUs that
reside on it"; and *fixed share*, where "each vGPU is given a fixed share of the physical
GPU's processing cycles." The equal-share and fixed-share schedulers "impose a strict
round-robin scheduling policy" implemented by adjusting each VM's time slice, and the
guide states the tradeoff outright: "for workloads that require low latency, a shorter
time slice is optimal," with a longer slice preferred for maximum throughput. The policy
is set through `nvidia-smi vgpu set-scheduler-state` (immediate, non-persistent) or the
`RmPVMRL` registry key (persistent, requires a driver reload).

- Documentation: <https://docs.nvidia.com/vgpu/latest/grid-vgpu-user-guide/changing-vgpu-scheduling-policy.html>

**Delta: vGPU already ships a weighted, per-tenant compute policy with an explicit
latency knob — fixed share plus a shortened time slice is, at the level of intent, what
Tessera's weighted shares plus an interactive tenant are. It is stronger than Tessera in
being a supported vendor product with hypervisor-level enforcement. Its distance from
Tessera is structural rather than rhetorical: the tenant is a virtual machine rather than
a process, so it requires virtualization and vGPU licensing; the sharing is temporal
rather than spatial; the share is set administratively and not recomputed from a measured
application-level signal; and there is no per-tenant memory quota expressed in device
memory. Tessera's claim against it is the deployment model and the closed control loop,
not the existence of weighted shares.**

---

## Production and open-source stacks

### HAMi and HAMi-core (Project-HAMi, Apache-2.0)

HAMi-core is the closest shipped artifact to Tessera's shim, and the most important
entry in this file. Its README describes it as "the in-container GPU resource
controller. It intercepts CUDA calls to enforce per-container device memory limits and
compute utilization limits, without requiring changes to the application or the driver."

HAMi-core compiles to `libvgpu.so` and hijacks the API calls between the CUDA runtime
and the CUDA driver. Injection in HAMi's Kubernetes path does not rely on the tenant's
environment at all: the host's `/usr/local/vgpu/ld.so.preload` is mounted into the
container as `/etc/ld.so.preload` naming `libvgpu.so`, so the dynamic linker loads it
first for every process that starts in the container. It also overrides `dlsym` itself,
which is how it survives explicit-handle lookups that `LD_PRELOAD` alone would miss —
and, checked directly in `src/cuda/hook.c`, it hooks both `cuGetProcAddress` and
`cuGetProcAddress_v2`, including the self-referential case where the runtime looks up
`cuGetProcAddress` through `cuGetProcAddress` and HAMi substitutes its own wrapper for
the returned pointer.

Memory is handled on two paths, and the second is the one Tessera must not overlook.
Enforcement is a pre-allocation check: interception of `cuMemAlloc_v2` /
`cuMemAllocManaged` / `cuMemAllocHost_v2` compares current usage plus the request against
`CUDA_DEVICE_MEMORY_LIMIT` and returns `CUDA_ERROR_OUT_OF_MEMORY` directly (with the
caveat that `cuMemAllocHost_v2` calls the driver first and checks afterwards, so the
check is not uniformly pre-allocation). But *reporting* is virtualized too:
`cuMemGetInfo_v2` (`src/cuda/memory.c`) calls through to the driver and then rewrites both
outputs against the quota, clamping `total` to the limit and computing `free` as the
remaining headroom; and `src/nvml/hook.c` hooks `nvmlDeviceGetMemoryInfo` and
`nvmlDeviceGetMemoryInfo_v2` the same way, so `nvidia-smi` inside the container reports
the virtual size.

Compute limits (`CUDA_DEVICE_SM_LIMIT`) are enforced by a token scheme, and its coverage
and cadence are both narrower than a summary suggests. `rate_limiter()` is called from
exactly three launch entry points, all in `src/cuda/memory.c`: `cuLaunchKernel`,
`cuLaunchKernelEx` and `cuLaunchCooperativeKernel`. It decrements a per-device credit
balance — `g_cur_cuda_cores` is declared `static volatile int64_t
g_cur_cuda_cores[CUDA_DEVICE_MAX_COUNT]`, per-process file-scope state rather than a
single shared global, so tenants are coupled only indirectly through sampled utilization
— and when the balance is negative the calling thread sleeps in `nanosleep` and retries.
A background thread samples per-process SM utilization through
`nvmlDeviceGetProcessUtilization` and replenishes the balance. The constants are in
`src/multiprocess/multiprocess_utilization_watcher.h`: the refill thread waits
`g_wait` = 120 ms and the throttle ticks at `g_cycle` = `TIME_TICK` (10) ms. The limiter
fast-exits without touching shared state when the cached SM limit is 0 or ≥ 100, and
before its compare-and-swap loop it runs `while (get_recent_kernel() < 0) { sleep(1); }`
— a one-second-granularity blocking step on a gated launch.

One gap is worth recording because it is both a real limitation and a hazard Tessera's
own interception matrix has to answer. `find_symbols_in_table()` in `src/cuda/hook.c`
begins with `if (strncmp(symbol, "cuGraph", 7) == 0) { return NULL; }`, under the comment
"Skip CUDA graph functions: let them fall through to real driver." So through
`cuGetProcAddress` — the path cudart actually uses — no CUDA-graph entry point is hooked
at all; and `cuGraphLaunch` in `src/cuda/graph.c` forwards straight to the driver with
only a debug log and no rate limiting. A CUDA-graph-driven workload therefore bypasses
both the memory check and the compute cap.

- Code: <https://github.com/Project-HAMi/HAMi-core>
- Mechanism documentation: <https://project-hami.io/docs/v2.8.0/core-concepts/gpu-virtualization>

**Delta: HAMi-core already ships the interposition Tessera's v0 is building — including
`cuGetProcAddress`/`_v2` and the `dlsym` override — *and* the per-tenant memory-quota
virtualization Tessera lists among its own properties, rewriting `cuMemGetInfo_v2` and
the NVML memory queries against the quota. Tessera claims no novelty in either. What
HAMi-core does not attempt is a cross-tenant weighted share, an interactive-tenant
latency objective, or an allocation-free, syscall-free launch path when credits are
available (I-4) — its gated path sleeps in `nanosleep` on a 10 ms tick behind a
one-second blocking step, and its feedback loop samples NVML every 120 ms. Its CUDA-graph
hole is a genuine coverage gap Tessera can claim honestly, and one Tessera's own M0 gate
tracks with a separate graph-launch counter. None of Tessera's side of this comparison
has been measured.**

### Alibaba Cloud cGPU

cGPU is the GPU-sharing component behind Alibaba Cloud's ACK shared-GPU scheduling. It
lets multiple containers share one GPU with memory isolation and computing-power
partitioning, and it is the main production example of doing this from a *kernel module*
rather than from a userspace shim.

Isolation is provided by a server kernel driver developed by Alibaba Cloud — "the GPU
sharing solution uses the server kernel driver that is developed by Alibaba Cloud to
provide more efficient use of the underlying drivers of NVIDIA GPUs" — whose presence a
node exposes through `/proc/cgpu_km/version`. Compute is divided by time slices under a
selectable policy, and the published policy set is unusually explicit: policy 0 is
fair-share scheduling, giving each container a fixed slice proportional to `1/max_inst`;
policy 1 is preemptive, where each container uses as many time slices as possible,
`1/N` for N current containers; policy 2 is weight-based preemptive, enabled
automatically when `ALIYUN_COM_GPU_SCHD_WEIGHT` exceeds 1; policy 3 assigns a fixed
percentage of computing power; policy 4 is soft scheduling, documented as providing
weaker isolation than preemptive scheduling; and policy 5 defers to the GPU driver's
native scheduling. `max_inst` caps the number of containers at "an integer from 1 to 25."

Orthogonal to the policy number, cGPU already distinguishes an interactive tenant. The
Elastic GPU Service reference documents `ALIYUN_COM_GPU_HIGH_PRIO`, which "specifies
whether to configure a high priority for the container," and states the semantics
directly: "when a high-priority container has a GPU task, it can preempt GPU computing
power regardless of the scheduling policy. When a high-priority container is idle, it is
excluded from scheduling and is allocated zero computing power." The share it may seize
is bounded by `prio_ratio`, which "defines the maximum computing power, as a percentage
from 20 to 99, that a high-priority container can preempt." The same reference records a
hard constraint on what tenants may do: "because cGPU isolation does not support Unified
Virtual Memory (UVM), you cannot call `cudaMallocManaged()` to allocate GPU memory."

The ACK overview states that to use it you "do not need to recompile the application or
create a new container image." It also lists, under a "Stability" heading, that "API
operations on CUDA libraries and some private API operations on CUDA Deep Neural Network
(cuDNN) are difficult to call." That sentence is ambiguous in the English documentation
and nothing is claimed here about what it means for interception; it is recorded because
it appears on a cited page, not because it supports an argument. The same page notes that
"after Alibaba Cloud Container Service for Kubernetes (ACK) makes GPU sharing
open-source, you can implement a GPU sharing framework on container clusters in both
Alibaba Cloud and on-premises data centers," which appears to describe the sharing
scheduler rather than the proprietary `cgpu_km` isolation module.

- Overview: <https://www.alibabacloud.com/help/en/ack/ack-managed-and-ack-dedicated/user-guide/cgpu-overview/>
- Computing-power allocation policies: <https://www.alibabacloud.com/help/en/ack/ack-managed-and-ack-dedicated/user-guide/configure-a-computing-power-allocation-policy-for-gpu-sharing-1>
- cGPU with the Docker CLI (high priority, `prio_ratio`, `max_inst`, UVM): <https://help.aliyun.com/en/egs/developer-reference/use-docker-to-install-and-use-the-cgpu-component-of-kubergpu-products>

**Delta: cGPU already ships weighted and fair-share compute policies of the kind Tessera
intends to offer, enforces them from a kernel module — a stronger enforcement point than
any userspace shim — and already distinguishes an interactive tenant through
`ALIYUN_COM_GPU_HIGH_PRIO`, which preempts regardless of scheduling policy up to
`prio_ratio`. So Tessera's claim is not that a high-priority tenant is a new idea; it is
narrower: that the share is recomputed from a measured latency signal rather than set as
a static ratio. Conversely cGPU is proprietary, shipped as a kernel module installed
through Alibaba's cloud-native stack and confined to its instance families, and it
forbids unified memory outright — which would block TGS's memory mechanism on a cGPU
node and bounds what any tenant can do there.**

### NVIDIA KAI Scheduler (open-sourced from Run:ai)

KAI Scheduler is the Kubernetes scheduling engine NVIDIA open-sourced from the Run:ai
platform under the Apache 2.0 license, and it is still delivered as part of the NVIDIA
Run:ai product. It is the reference point for what "fractional GPU" means in a shipped
orchestration product.

Fractional allocation is expressed declaratively on the pod: a `gpu-fraction` annotation
("for example, `0.5` reserves half of one GPU's memory capacity"), a `gpu-memory`
annotation in MiB, or per-container NvFractions annotations such as
`nvidia.com/container.<container>.gpu-memory.request`. The documentation defines four
cluster-wide GPU-sharing modes, not three: `Disabled` ("reject fractional GPU
workloads"); `NonMemoryEnforced`, which schedules fractional GPU workloads "without
runtime memory isolation" and is the *default*, with the explicit warning that
"containers may see and use more GPU memory than requested"; `HamiCore`, which delegates
CUDA memory isolation to HAMi-core; and `NvFractions`, "enforced by the NvFractions
runtime path when CDI/NRI is configured."

That last path is stronger than "a runtime path" suggests. The GPU Fractioning Operator
deploys `fractiond`, which "registers as an NRI plugin with the container runtime. When a
pod carrying GPU-memory annotations is created, `fractiond` injects
`NVIDIA_GPU_MEMORY_REQUEST` / `NVIDIA_GPU_MEMORY_LIMIT` into the container **before it
starts**." Its README then states the enforcement point plainly: "the NVIDIA driver
enforces `NVIDIA_GPU_MEMORY_LIMIT` as a hard cap, so a container cannot allocate beyond
its share and impact its neighbors on the same GPU. A container that exceeds its limit is
terminated (out-of-memory)." The price is a recent stack: "containerd 2.0+ with NRI
enabled, or CRI-O with NRI support" and "NVIDIA driver `r615` or newer (CUDA 13.4)" on
the GPU nodes, which is explicitly "not the GPU Operator default." KAI also carries a
compute knob, though a static one: `nvidia.com/container.<name>.gpu-compute.mode`, where
"`time-slicing` is the default" and `sm-sharing` routes the container to a shared MPS
server, and "KAI keeps pods with incompatible compute sharing modes out of the same
fractional GPU group." A `metricsd` sidecar "exports per-pod GPU memory and utilization
metrics for the shared GPUs"; nothing reads those metrics back to change an allocation.

- GPU sharing documentation: <https://github.com/NVIDIA/KAI-Scheduler/blob/main/docs/gpu-sharing/README.md>
- GPU Fractioning Operator (NvFractions path): <https://github.com/kai-scheduler/gpu-sharing>
- Announcement (license, Run:ai relationship): <https://developer.nvidia.com/blog/nvidia-open-sources-runai-scheduler-to-foster-community-collaboration/>

**Delta: KAI Scheduler solves the cluster-level problem Tessera explicitly does not touch
— which pod lands on which GPU, with gang scheduling and queue fairness — and in its
default mode enforces nothing at runtime. But where the NvFractions path applies, the
per-container memory limit is enforced by the NVIDIA driver itself rather than by a
userspace rewrite of `cuMemGetInfo`, which is a strictly stronger guarantee than anything
Tessera can offer from userspace, at the price of an r615/CUDA-13.4 driver and
NRI-enabled containerd. What is still absent, and is the node-level layer Tessera is
trying to build well, is a runtime resource controller: the compute mode is a static
per-container annotation and the exported SM-utilization metric is never fed back into an
allocation.**

### NVIDIA Run:ai fractional GPU

Run:ai is named on the M0 gate's neighbor list in its own right, and the commercial
product's fractional-GPU behaviour is the closest shipped analogue to Tessera as a whole.
It is worth separating from KAI because the fraction means something narrower than the
name suggests.

The Run:ai documentation is explicit about scope: "NVIDIA Run:ai GPU fractions control
the memory split (i.e. 0.5 GPU means 50% of the GPU memory) but not the compute
(processing time)." Enforcement is by memory: "each pod uses a its own separate virtual
memory address space. NVIDIA Run:ai's GPU fractions logic enforces the requested memory
size, so no workload can use more than requested." Compute is handled separately and by
the vendor primitive: "by default, GPU fractions use NVIDIA's time-slicing to share the
GPU compute runtime," with a distinct GPU time-slicing feature referenced for splitting
compute time. Dynamic fractions exist so "even small unused fragments of GPU memory are
utilized by workloads."

- GPU fractions (self-hosted v2.20): <https://run-ai-docs.nvidia.com/self-hosted/2.20/platform-management/runai-scheduler/resource-optimization/fractions>

**Delta: Run:ai ships, as a supported commercial product, the per-tenant memory quota
half of Tessera's combination, with per-pod address-space separation and quota
enforcement that a userspace shim cannot match on guarantees. Its own documentation says
the fraction does not govern compute, and that compute sharing falls back to NVIDIA
time-slicing — so the weighted compute share plus interactive-tenant latency objective is
the part Tessera would be adding, on top of a memory mechanism the product already has.
Run:ai also documents a separate GPU time-slicing scheduler that this file has not
read; that page must be fetched before any claim is made about its modes.**

### Time-slicing and MPS in the NVIDIA Kubernetes device plugin / GPU Operator

This is the most widely deployed GPU-sharing mechanism in existence and the honest
baseline for "what people actually run." The plugin offers two mutually exclusive
sharing modes, and only one of them does nothing.

In **time-slicing** mode an administrator declares a number of replicas for a GPU, "each
of which can be handed out independently to a pod to run workloads on," and "GPU
time-slicing is used to multiplex workloads from replicas of the same underlying GPU."
The mechanism does nothing beyond oversubscription: the replicas are not partitions, the
GPU's own time-slicing scheduler interleaves the contexts, and there is no admission
control, no quota and no priority. NVIDIA's documentation states the consequence without
euphemism: "unlike Multi-Instance GPU (MIG), there is no memory or fault-isolation
between replicas, but for some workloads this is better than not being able to share at
all."

In **MPS** mode the plugin does considerably more, and this is the part that makes the
baseline stronger than it looks. Read at `cmd/mps-control-daemon/mps/daemon.go`, the
plugin's MPS control daemon first sets the node's GPUs to `EXCLUSIVE_PROCESS` compute
mode, starts `nvidia-cuda-mps-control`, and then issues
`set_default_device_pinned_mem_limit <index> <totalMemory/replicas>M` per device and
`set_default_active_thread_percentage <100/replicas>`. Those are client *defaults*, set
once at daemon start, identical for every client on the node; the plugin never issues
MPS's runtime `set_active_thread_percentage` and consumes no feedback signal.

- Documentation: <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html>
- MPS control daemon source: <https://github.com/NVIDIA/k8s-device-plugin/blob/main/cmd/mps-control-daemon/mps/daemon.go>

**Delta: time-slicing is the drop-in deployment model Tessera wants to match — set an
environment variable or a node label and unmodified workloads share a GPU — and in that
mode it provides neither memory isolation nor fault isolation nor any control over who
gets the GPU when. But the same plugin's MPS mode already pulls one of Tessera's levers,
uniformly and once, and gets MPS-enforced pinned-memory limits a userspace
`cuMemGetInfo` rewrite cannot provide — a stronger memory guarantee than Tessera's. That
mode is therefore the precise statement of Tessera's gap: the same knob, under a
controller driven by a measured per-tenant signal instead of a fixed `100/replicas`.
Enabling it also puts the node's GPUs into `EXCLUSIVE_PROCESS`, a node-wide precondition
that constrains how a Tessera shim could coexist with it and that the MIG guide notes is
unsupported in MIG mode.**

---

## Tessera's delta

**Approved at HC-0 (2026-09-21). This paragraph is the human's wording, not a
summary of it.**

Tessera is an open-source, host-only GPU sharing layer: no root, no device-code
rewriting, no resident device-side agent, no context migration, running in a
stock container on commodity cloud GPUs (L4/A10G) in two modes, solo (no MPS)
and shared (over MPS). Contributions: (1) a measured, reproducible account of
what each host-side lever does on an Ada GPU, including the preemption floor set
by kernel and CUDA-graph durations; (2) a bandwidth-aware SLO controller:
interactive decode is memory-bound, so Tessera treats DRAM bandwidth pressure,
not only SM occupancy, as the interference signal, and tests whether SM
partitioning alone protects TPOT; (3) memory-quota virtualization so unmodified
vLLM sizes itself to its quota. DetShare, Tally, XSched, Hummingbird, LithOS,
and µShare are baselines: run their code where public, otherwise report where a
no-migration, no-agent design lands against their published numbers, and list
each one's extra mechanism with the measured cost of not having it.

### What this delta commits us to

Each clause is a claim someone can check, so each one has an owner:

- **"no root, no device-code rewriting, no resident device-side agent, no
  context migration"** — four negative constraints that distinguish Tessera from
  Tally, XSched and Hummingbird, all of which reach finer control by rewriting
  the tenant's device code. They are constraints we chose, so they are also
  costs, and the last sentence of the delta requires us to measure each one.
- **"commodity cloud GPUs (L4/A10G)"** — M0 measured L4 (driver 580.95.05,
  CUDA 13.0, 58 SMs). A10G is not yet measured and is now in scope for M1.
- **"bandwidth-aware ... DRAM bandwidth pressure, not only SM occupancy"** — this
  is the scientific claim, and **H9** is the experiment that decides whether it
  is true. If confining a batch prefill tenant to k SMs still degrades decode
  TPOT, SM partitioning alone does not protect the interactive tenant and the
  controller has a reason to exist. If it does not degrade, this delta needs
  rewriting.
- **"DetShare, Tally, XSched, Hummingbird, LithOS, and µShare are baselines"** —
  a heavier commitment than the previous "related work" framing: where the code
  is public we run it, and where it is not we report our position against their
  published numbers and name the mechanism we lack. Two of the six are not yet
  covered in this file at all (see Open questions).

## Open questions

M1 must measure these before any claim above about Tessera's position can stand.

0. **Two named baselines have no entry in this file.** The approved delta commits
   Tessera to treating DetShare, Tally, XSched, Hummingbird, **LithOS** and
   **µShare** as baselines. LithOS currently appears here only as a name inside
   *other* systems' comparison sets (it is in Vitamin-E/DetShare's and MuxWise's
   evaluation tables), never as an entry of its own with a fetched primary
   source; the earlier review recommended adding it and it was deliberately left
   out because no primary source had been retrieved. µShare does not appear at
   all. Both need an entry with a fetched source, a mechanism paragraph, a
   delta, and — because the delta now promises it — a note on whether the code
   is public and therefore runnable as a baseline. Until then the delta's last
   sentence is a commitment this file does not yet support.

1. **Green-context behaviour under an uncooperative neighbor.** NVIDIA's Programming
   Guide already claims the positive property — a kernel "can only use the SMs
   provisioned for green context A, irrespective of its launch configuration," and a
   critical kernel "is guaranteed that there will be available SMs for it to start
   executing immediately, barring any other resource constraints" — while also stating
   that "concurrent execution of independent GPU work is not guaranteed." MuxWise reports
   that re-partitioning costs "only a stream synchronization (on the order of
   microseconds)," but inside one cooperating process. The open part is therefore
   specifically the cross-process, uncooperative-neighbor case: whether a partitioned
   tenant's kernels are admitted promptly while another process saturates the GPU, what
   the SM-count quantization costs at realistic partition sizes on the L4 (where the
   pinned 12.6 header's 4-SM minimum for 8.X binds), and whether a partition can be
   changed at runtime across processes without destroying and recreating contexts.
2. **Cross-client priorities under MPS.** MPS exposes two advisory client priority levels
   and states that the active-thread percentage reserves nothing and that kernels from
   different clients may share an SM; the driver reference says stream priorities "do not
   preempt already-running work or provide any other functional guarantee on execution
   order." Orion's §6.4 already reports that stream priorities gave "only marginal
   improvements" once its compute/memory and size-based policies were in place, and that
   Orion "can also be used in settings where the GPU hardware does not support stream
   priorities (e.g., in MPS mode)." Our measurement should therefore be framed as
   confirming or refuting that published finding on our hardware, not as opening a new
   question, and it should include MPS static SM partitioning as the reserving
   alternative.
3. **The kernel-duration preemption floor.** Tessera does not preempt a running kernel,
   so the achievable tail-latency floor for the interactive tenant is bounded below by how
   long a neighbor's already-launched kernel runs. This is a property of Tessera's chosen
   constraint, not of userspace interposition: XSched's Level 2 hits the same bound and
   its Level 3 escapes it by runtime binary instrumentation and an undocumented ioctl,
   and Hummingbird and Tally escape it by PTX transformation. REEF's own closed-source
   variant, which likewise waits for running kernels, measures 71–288 µs on a V100. We
   must measure that distribution for the batch workloads we intend to co-locate, because
   it sets the best result Tessera can possibly report under I-3 and determines whether a
   launch-gating scheduler is viable at all.
4. **Whether green contexts and MPS compose across processes.** The premise that
   Vitamin-E orchestrates green contexts over MPS came from the superseded v2 text and
   does not describe v4, so the question must be asked of the vendor documentation
   instead. The pinned CUDA 12.6 header states that under Volta+ MPS, when
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is used, a green-context workload's SM set "can be
   scaled up to the value of SMs used for the MPS client" — i.e. the provisioned partition
   is a floor, not a ceiling, when MPS is in play. The Programming Guide adds that
   "contrary to green contexts, MPS with static partitioning does not allow
   oversubscription of SM resources." We must measure what the first sentence means for an
   interactive tenant's isolation, and whether MPS static SM partitioning composes with
   green contexts at all.

## Method

**How this file was produced, and who checked it.** The file was drafted by one agent
that fetched primary sources itself — paper PDFs, vendor documentation pages, and project
source code rather than summaries — because the upstream research payload for this work
package arrived empty. It has since been independently reviewed by **two agents that did
not write it**, and this revision applies their findings.

- **Reviewer 1 (adversarial, against primary sources)** re-fetched every cited source and
  checked each quotation and number against it, including the repository's own pinned
  CUDA 12.6.77 header and driver stub.
- **Reviewer 2 (cross-check)** compared the file against a separately researched corpus of
  30 neighbor entries and 29 verdicts produced by six agents that never saw this file, and
  re-fetched a subset of sources directly.

Every correction in this revision was re-verified by the editing agent against the
primary source before it landed, because the two reviewers disagreed on several points
and each was wrong on at least one. Specifically: Reviewer 1 reported that three USENIX
presentation pages return HTTP 403 to automated fetchers; re-tested on 2026-09-19 with a
browser user-agent they returned **200**, and that claim was not carried into this file.
Reviewer 2 attributed the green-context exclusivity statement to the pinned 12.6 header;
`grep` shows it is not there, and it is quoted above from the Programming Guide instead.
Reviewer 2 also listed `cuStreamGetDevResource` among the pinned symbols; it is not in
the pinned header or stub, and the twelve-symbol list above is the one the commands
actually return.

**What was checked, and how.** Mechanism claims for REEF, Paella, Orion, TGS, Clockwork
and AntMan were read from the papers' own PDFs, downloaded and text-extracted locally
with `pdftotext -layout`; the same was done for Tally, XSched and Hummingbird. Vitamin-E
was read from the arXiv HTML of v4 (current) and v2 (superseded), and MuxWise from the
v3 HTML. MPS, MPS v3, MIG, green-context, stream-priority, vGPU and GPU Operator claims
come from NVIDIA's documentation and are quoted where load-bearing. HAMi-core's README,
`src/cuda/hook.c`, `src/cuda/memory.c`, `src/cuda/graph.c`, `src/nvml/hook.c`,
`src/multiprocess/multiprocess_utilization_watcher.{c,h}` and issue #179 were read from
the repository at `master`. The KAI and GPU Fractioning Operator claims were read from
those repositories' READMEs, and the NVIDIA device plugin's MPS behaviour from
`cmd/mps-control-daemon/mps/daemon.go`. gVirtuS's transports, plugin list and license come
from its repository and the GitHub API. cGPU's policies and high-priority mechanism come
from Alibaba's own documentation pages.

The one repository-side fact in this file was re-derived by running the commands: the
twelve green-context entry points listed above are present in the CUDA 12.6.77 headers
and in the driver stub's export list that this repository pins (640 exported `cu*`
symbols), checked by grepping `cuda.h` and running `nm -D` on `stubs/libcuda.so` under
`$TESSERA_DEPS`.

**URL status, re-tested 2026-09-19.** Every URL in this file was requested with a browser
user-agent following redirects. All returned HTTP 200 except:

- The three ACM Digital Library DOIs (Paella `10.1145/3600006.3613163`, Orion
  `10.1145/3627703.3629578`, Tally `10.1145/3669940.3707282`) return HTTP 403 to
  automated fetches. Each is cross-checked another way: the Paella and Orion DOIs are
  printed in those papers' own PDFs, and Tally's ACM reference block in the arXiv PDF
  gives the same DOI, venue and dates.
- `https://github.com/NVIDIA/k8s-device-plugin/blob/main/cmd/mps-control-daemon/mps/daemon.go`
  returned a transient HTTP 429 on first request and 200 on retry; the file itself was
  read through `raw.githubusercontent.com`.

What could **not** be verified, and must be treated as open:

- **The gVirtuS Euro-Par abstract could not be retrieved from its DOI.** The DOI returns
  HTTP 200 but redirects into a Springer identity-provider authorization flow that serves
  no abstract to an automated fetcher. An earlier version of this file quoted the abstract
  as reporting "an overhead slightly greater than a real machine/GPGPU setup"; that
  quotation has been **removed** rather than left standing on an unfetchable source. The
  gVirtuS mechanism claims above come from the repository, which was fetched. Someone with
  publisher access should restore the overhead statement, with a source, before HC-0.
- **GPUColo (IEEE, 2024; Xplore document 10630927) is missing from this file and should
  not be.** Reviewer 2 identifies it as a two-tier system whose outer tier adjusts the
  percentage of active GPU threads between co-located inference and training to hold an
  inference latency SLO, with an inner tier that puts training into periodic sleep — i.e.
  Tessera's stated design, already published. Neither its IEEE page nor the NSF PAR
  open-access copy could be fetched in this session (Xplore returned HTTP 202 with no
  content; `par.nsf.gov` refused the connection). Because no primary source could be
  fetched, **no entry was written and no claim about its contents is made above**. It must
  be fetched, read and entered before the "Tessera's delta" paragraph is approved at HC-0,
  and the tail-latency-objective clause in that paragraph has been narrowed accordingly.
- **LithOS (arXiv 2504.15465) is not covered.** Reviewer 2 recommends it as a
  CUDA-masquerading library scheduling at TPC granularity with a P99 objective, and
  Vitamin-E reproduces it as a baseline. Its venue is also unconfirmed. It was not added
  because its primary source was not fetched in this session.
- **Run:ai's GPU time-slicing scheduler.** Reviewer 2's two upstream sources contradicted
  each other on its Strict/Fair modes and lease/plan durations, and the Run:ai fractions
  page fetched here does not describe them. Nothing about those modes is claimed above.
  A direct fetch of the time-slicing page is required before any claim is made.
- **REEF's abstract and introduction still give different figures** for the same
  throughput result (up to 7.7× versus up to 4.3×, both "compared to dedicating the GPU to
  real-time tasks"). §7.2 shows the 7.7× figure is the Apollo real-world-trace result, so
  the two statements are not describing the same workload set; the paper does not
  explicitly reconcile them and this file records both.
- **cGPU's internals** are documented only at the level of policy names, the
  high-priority mechanism, and the presence of `cgpu_km`. How memory isolation is
  implemented inside the Alibaba kernel driver is not publicly documented and is not
  claimed here. The "difficult to call" sentence on the overview page is recorded as
  quoted and is not interpreted.
- **Hummingbird and the edge-GPU isolation study (arXiv 2601.07600) have no confirmed
  peer-reviewed venue**, and XSched's and Tally's line-level implementation details were
  read from their papers rather than from their code.
- **HAMi-core line numbers** are not cited above, because the repository was read at
  `master` rather than at a pinned commit; the file names and function names are, and were
  read directly.
- **No Tessera performance number appears anywhere in this file, because Tessera has
  measured nothing.** Every quantitative statement is attributed to the paper or vendor
  document that made it. The only Tessera-side facts are the repository facts marked as
  command-checked above.

**Status for the M0 gate.** G1 requires that every factual claim be checked by an agent
that did not write it. Two independent reviews have now been applied, and their supported
corrections are in. The items in the list above are the residue: claims that were dropped
rather than fixed, and neighbors that are known to be missing because no primary source
could be fetched. G1 should be scored against that list, not against a blanket claim of
verification.
