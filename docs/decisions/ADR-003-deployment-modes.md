# ADR-003: Two deployment modes, both first-class

- Status: proposed (HC-0 decides)
- Date: 2026-09-19
- Invariants: I-6 (scope), I-2 (unmodified tenants)
- Validated by: H3/H4/H5/H6/H8 in M1; the B/T matrix in M3

## Context

Tessera's levers do not all work in the same setting. Some need nothing but the
stock driver; others exist only when tenants share one CUDA context, which in
practice means running under MPS. Pretending there is a single configuration
would either restrict Tessera to sites that can run an MPS daemon, or would
promise isolation properties that quietly do not hold without it.

Two facts drive the split, and both are assumptions until M1 measures them:

- **Stream priorities are a within-context mechanism.** Separate processes get
  separate contexts, so a high-priority stream in one process has no defined
  relationship to a low-priority stream in another. Under MPS, clients share one
  context, which is what could make cross-tenant priority meaningful. H3 (within
  a process) and H4 (across two MPS clients) test exactly this, and until H4
  returns a number, Tessera claims nothing about cross-tenant priorities.
- **MPS is not always available.** It needs a control daemon, and some
  environments (possibly including our own Modal containers) do not permit one.
  H8 answers that for our platform; if MPS cannot run in-container, the MPS rows
  move to a root-capable instance rather than being dropped (I-1).

## Decision

Two modes, both supported, both measured in the M3 matrix:

**`solo`** — no MPS. Available levers: launch gating (lever 1) and memory quotas
(lever 5). Every tenant is its own context and the driver time-slices between
them. This is the mode that works anywhere, with no privileged setup.

**`shared`** — on top of MPS. Adds stream priorities (lever 2), MPS active-thread
percentage (lever 3), and, if H6 shows they are usable, green contexts (lever 4).

The daemon knows which mode it is in, advertises the set of active levers in
`tessera status`, and refuses to enable a lever the mode cannot support rather
than silently accepting a policy it cannot honour. Documentation states which
levers are inert in `solo` mode, because discovering that in an eval run is how
a project ends up with an unexplained flat line on a chart.

Green contexts sit outside this split until H6 reports: they are a CUDA 12.4+
driver feature, and whether they compose with MPS, whether they share the
primary context's address space, and what SM granularity they actually offer are
open questions. No policy depends on them before that measurement.

## Alternatives rejected

- **MPS-only.** Cleanest lever set, but requires a daemon and a privileged setup
  that not every deployment allows, and would make Tessera undeployable exactly
  where the drop-in promise matters most.
- **Solo-only.** Gives up spatial partitioning entirely, and with it most of the
  plausible answers to a long-kernel batch tenant (see ADR-002's floor).
- **MIG.** Hardware partitioning is real isolation, but it is datacenter-part
  only, cannot be resized dynamically, and does not exist on an L4. Out of scope
  by I-6; covered in `docs/RELATED.md` as a neighbor.

## Consequences

- Every eval configuration carries its mode, and the headline chart must not mix
  them without labelling.
- The controller's actuator set differs by mode, so M3's controller must be
  written against a declared actuator interface rather than assuming a
  partition knob exists.
- If H4 shows MPS does not honour cross-client stream priorities, lever 2 is
  documented as inert cross-tenant and the ladder loses a rung. That is a result,
  not a failure, and `docs/mechanisms.md` records it either way.

## What would overturn this

H8 showing MPS unavailable on the target platform would make `shared` mode
untestable there and would raise the M1 budget to cover a root-capable instance.
H6 showing green contexts to be unusable under MPS would collapse levers 3 and 4
into alternatives rather than a ladder.
