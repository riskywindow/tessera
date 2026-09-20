# M0 RED: the gate was declared GREEN and the review took it apart

- Date: 2026-09-20
- Commit reviewed: `4f52fbd`
- Status: **RED.** Two hard criteria (G6, G7) are not demonstrated, and a
  non-negotiable invariant (I-5) is violated by a reproduced deadlock.
- Written per the gate protocol: a hard criterion failed, so this memo lists
  the options and work stops until HC-0.

## What happened

I closed M0 as GREEN, then ran three adversarial reviewers over the shim, the
acceptance matrix and the gate evidence, specifically because the matrix that
decides G5, G6 and G7 was written by me after the subagent assigned to it
stalled — so those tests had no reviewer other than their author.

All three reviewers independently concluded GREEN was not justified. They did
not argue it; they reproduced it, by mutating the shim and re-running the
tests. That is the outcome the review existed to produce, and the GREEN claim
was wrong.

## The findings that make this RED

### 1. G6 is vacuous — the test passes with the mechanism deleted (blocker)

Reduce `tessera::map_proc_address()` to `return real;` — deleting the entire
identity-based variant substitution that ADR-001 and G6 rest on — and
`shim_acceptance_ptsz_test` still passes 3/3, with `stream_failures=0` and
`version_failures=0`.

The cause is a property of the *fake driver*, and it is the same effect R-21
already describes from the other direction. `fake_driver/src/entry_points.cc`
builds its `kSymbols[]` table from interposable PLT references, so when the
fake's `cuGetProcAddress` returns "its own" entry point it actually returns
**the shim's wrapper**, because the shim is in the global scope. The reviewer
instrumented both branches of `map_proc_address`: across 30 lookups in
masquerade and preload, the substituting branch was taken **zero** times.

So T2 measures the fake driver's variant selection, not the shim's. On a real
driver — which does not hand its internal addresses to interposition —
`map_proc_address` is the only thing that makes `cuGetProcAddress` interception
work, and nothing on the CPU side would catch a regression in it.

This also weakens the `gpa1`/`gpa2` cells of T1 (G5) for the same reason.

### 2. G7 is vacuous — the fork window closes before the child exists (blocker)

Empty `atfork_child()` entirely (no `reset_after_fork()`, no unlock) and all
three fork tests still pass.

The scenario defeats itself. The shim registers a `pthread_atfork` **prepare**
handler that takes `g_init_mu`, so `fork()` blocks until the initialising
thread finishes `do_init()` and releases it. By the time the child exists,
`g_init_state` is already `kReady`, and the child's `ensure_init()` returns on
the fast path without ever touching the mutex.

The `fork_blocked_us > 1000` guard — which I cited in the gate doc as proof
that "the window really opened" — is in fact proof that the prepare handler
serialised `fork()` *after* init, i.e. that the window **closed**. The test can
only ever show that `pthread_atfork` was called.

Two further defects in the same tests: the counter assertion is structurally
always true (at fork time the parent's `hooked_calls` and `launches` are both 0,
because the only call made is `cuInit`, which is a trampoline and bumps neither),
and the T3 tests never check `app_completed()` — on the pristine tree the app
**exits 1**, because the child's hooked call resolves to `cuDeviceTotalMem_v2`,
which `null_driver.c` does not export. "Returned an error quickly" was being
scored as "completes its hooked call".

### 3. I-5 violated: a reproduced deadlock in the shim (blocker)

`ensure_init()` holds `g_init_mu` across `dlopen()`, so the shim's lock is held
across the glibc loader lock. Two threads invert the order:

- thread B calls `dlopen()` on some library whose ELF constructor makes the
  process's first `cu*` call, and blocks on `g_init_mu`;
- thread A is inside `ensure_init()` holding `g_init_mu` and blocks in
  `dlopen()` on the loader lock.

Both wedge permanently, and every subsequent `fork()` wedges too, because
`atfork_prepare()` also takes `g_init_mu`. Reproduced under **both** strategies
with a two-thread program, hung until killed at 15 s and 12 s; thread states
read from `/proc` confirm one futex inside `ld-linux`'s mapping and the other on
`g_init_mu`. The identical binary with no shim completes in under a second, so
the deadlock is the shim's.

This is exactly the tenant pattern the project targets: a Python process
importing a CUDA extension on one thread while another makes its first driver
call. I-5 says Tessera may make a tenant slower and may never hang it.

### 4. Evidence hygiene (major, and my fault)

- `results/m0/local_matrix.json` — the artifact cited by G3a, G3b, G4, G5, G6
  and G7 — was generated **before** the commit that fixed the SIGSEGV, so the
  gate's headline commit and its artifact disagree. Being re-run at HEAD.
- `results/m0/gpu_gate.json` records a dirty tree at a commit predating the fix.
- G6 and G7 quoted measured values whose "artifact" column pointed at *source
  files*, not results. That is the I-1 rule being bent by its own author.
- Several numbers in docs have no file under `results/`: "4 of 4 hashes",
  "640 cu* symbols", the disk figures in §5.
- G1 is marked PASS while `RELATED.md`'s own stated precondition for approving
  the delta is not yet met.

### 5. Still open from the reviewers, not yet assessed

`lookup_gnu` is an unbounded wild read over arbitrary objects — the SIGSEGV fix
confined only the SysV scan, so the same class of crash may remain reachable.
And the `DISABLED_` test that documents R-27 apparently passes, which means the
recorded risk is not reproduced by the test kept to prove it.

## What is NOT in doubt

Worth stating plainly, because the rest of this memo is negative:

- **G9 holds.** The GEMM loop's 400 shim launches against 400 CUPTI device
  kernels, and the control loop's 200 against 200, are real, the baseline was
  measured before the shim ran against it, and the harness asserts the equality
  rather than displaying it. The reviewer checked this specifically.
- **G8 holds** as a comparison; the number "4 of 4" needs to come from the
  artifact rather than from prose.
- The interception matrix's *structure* is sound: three independently produced
  counts, real negative controls, the hooked/unhooked split taken from the
  shim's own generated table.
- The SIGSEGV found by the GPU gate was real and is fixed.

## Options

**A. Fix the three blockers, then re-close the gate.** Estimated as the
smallest honest path to GREEN.
1. Link `libcuda_fake` with `-Bsymbolic-functions` so it returns its *own*
   entry points, which is how a real driver behaves. Then `map_proc_address`
   must do work for the gpa cells and T2 to pass. Add a `substituted_lookups`
   counter to `tessera_stats_v1` and require it > 0 in those cells.
2. Rebuild T3 around a window the prepare handler cannot close (fork via
   `syscall(SYS_clone)`, or a test-only knob that skips the `pthread_atfork`
   registration and requires T3 to fail with it set). Assert
   `app_completed()`, give the parent real hooked calls before forking, and
   turn the empty-child-report skip into a failure.
3. Move the `dlopen` out from under `g_init_mu`: publish the handle with a CAS
   and keep the lock only for the short publish step. Then reconsider
   `atfork_prepare`, which currently makes every `fork()` wait for a driver
   load.
4. Re-run the local matrix and the GPU gate at HEAD, and fix the artifact
   citations.

**B. Fix only the I-5 deadlock and re-run, accepting G6/G7 as AMBER** with the
vacuity documented. Cheaper, but it would mean carrying two criteria that are
known not to test what they claim into M1, where H1 depends on exactly the
`cuGetProcAddress` mechanism G6 was supposed to cover. Not recommended.

**C. Re-scope M0.** Argue that G6/G7 as written cannot be demonstrated against
a fake driver at all, and move them to the GPU. This would need an ADR and
would make M0 depend on more GPU time. The reviewer's fix for the fake driver
(option A.1) is cheaper and keeps the CPU-side proof, so this is a fallback.

**Recommendation: A.** The three fixes are each well-specified, the reviewers
supplied the mechanism for each, and none of them is speculative. The cost is
one more build-and-verify cycle plus a GPU gate re-run (the last one used 21 s
of GPU time).

## The process lesson, recorded because it will recur

The acceptance matrix decides three gate criteria, and I wrote it myself after
delegation failed twice. I then reviewed my own work and declared GREEN. Every
defect above was found by readers who had not written the thing they were
reading, and two of them were found only by *mutating the code and re-running
the tests* — which I had demanded of the subagents in their briefs and did not
do myself. The rule the project already had ("a test that passes with the shim
broken proves nothing") was right; the failure was in not applying it to my own
tests.
