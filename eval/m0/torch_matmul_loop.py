#!/usr/bin/env python3
"""G9 workload: two PyTorch loops with a counted launch window.

What this script is for
-----------------------
G8 asks that a tenant's outputs are bit-identical with and without the shim.
G9 asks that the shim's launch counter, over a defined window, equals the
number of kernel launches that actually happened. This script supplies both,
for the PyTorch half of the gate:

  * ``--loop gemm``    N iterations of ``torch.mm(a, b, out=c)``. cuBLAS may
                       issue more than one kernel per GEMM, so the expected
                       count is whatever CUPTI records - never a guess.
  * ``--loop control`` N iterations of ``x.add_(1.0)``, which is exactly one
                       kernel launch per iteration. This is the loop whose
                       expected count is known a priori, and it is asserted.

Two independent counters are read over the same window:

  1. CUPTI, through ``torch.profiler`` with ``ProfilerActivity.CUDA``. This is
     the ground truth: it is collected by the driver's own tracing interface
     and does not go through anything Tessera wrote.
  2. The shim's exported counters (``--with-shim``), read through
     ``ctypes.CDLL("libcuda.so.1")`` - which, under the masquerade strategy,
     *is* the shim. The shim does not exist yet; that path is written and
     guarded by the flag so a later agent only has to turn it on.

The window
----------
Each loop runs three phases:

    warm-up (W iterations)        -> outside every window
    pass A  (N iterations)        -> inside the CUPTI profiler
    pass B  (N iterations)        -> inside the shim-counter window

Both passes always run, whether or not ``--with-shim`` is given. That is
deliberate: the control tensor is mutated by the loop, so skipping a pass
would change the final tensor and break the G8 hash comparison between a
no-shim baseline and a with-shim run. Passes are not merged because the M0
gate doc requires the CUPTI count to come from a separate pass with the same
inputs.

Each window is bounded the same way: ``torch.cuda.synchronize()``, read
counters, exactly N iterations, ``torch.cuda.synchronize()``, read counters.

Determinism
-----------
``CUBLAS_WORKSPACE_CONFIG`` must be set before cuBLAS initialises, so it is set
at the top of this module, before ``import torch``. TF32 is off, deterministic
algorithms are on, and the inputs are generated on the CPU from a fixed seed
and copied to the device, so the bytes fed to the GPU do not depend on the
device RNG.

Output
------
One ``TESSERA_JSON`` line per loop plus one summary line, so the Modal runner
parses stdout instead of screen-scraping. Exit code is non-zero on any failed
assertion; nothing is masked (I-9).
"""

from __future__ import annotations

import os

# Must precede `import torch`: cuBLAS reads this when it creates its handle,
# and torch.use_deterministic_algorithms(True) refuses to run cuBLAS GEMMs on
# CUDA >= 10.2 unless it is set.
_CUBLAS_WORKSPACE_CONFIG = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os.path  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from typing import Any, Callable  # noqa: E402

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

SEED = 20260919

# Kineto/chrome-trace names that mean "a kernel launch was issued by the host".
_RUNTIME_LAUNCH_NAMES = frozenset(
    {
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cudaLaunchCooperativeKernel",
    }
)
_DRIVER_LAUNCH_PREFIXES = ("cuLaunchKernel", "cuLaunchCooperativeKernel", "cuLaunchEx")
_GRAPH_LAUNCH_NAMES = frozenset({"cudaGraphLaunch", "cuGraphLaunch"})


# --------------------------------------------------------------------------
# Shim-side counters. The shim exports these from libcuda.so.1 (masquerade).
# --------------------------------------------------------------------------
class TesseraStatsV1(ctypes.Structure):
    """Mirror of the shim's v1 stats struct. Field order is the ABI."""

    _fields_ = [
        ("launches", ctypes.c_uint64),
        ("graph_launches", ctypes.c_uint64),
        ("hooked_calls", ctypes.c_uint64),
        ("forwarded_unhooked", ctypes.c_uint64),
        ("bypassed_lookups", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name, _ in self._fields_}


class ShimCounters:
    """Thin ctypes binding to the shim's exported counter API.

    Constructed only when --with-shim is passed. Every failure here is fatal:
    if the caller asked for shim counters and they are not there, the run is
    not a valid G9 measurement and must not silently degrade to a no-shim run.
    """

    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(f"--with-shim: cannot dlopen libcuda.so.1: {exc}") from exc

        missing = []
        for sym in (
            "tessera_abi_version",
            "tessera_get_stats_v1",
            "tessera_reset_stats",
            "tessera_dump_stats",
        ):
            if not hasattr(self._lib, sym):
                missing.append(sym)
        if missing:
            raise SystemExit(
                "--with-shim: libcuda.so.1 does not export "
                + ", ".join(missing)
                + ". LD_LIBRARY_PATH is probably not pointing at the shim's "
                "masquerade directory; use infra/with_shim.sh."
            )

        self._lib.tessera_abi_version.restype = ctypes.c_int
        self._lib.tessera_abi_version.argtypes = []
        self._lib.tessera_get_stats_v1.restype = ctypes.c_int
        self._lib.tessera_get_stats_v1.argtypes = [ctypes.POINTER(TesseraStatsV1)]
        self._lib.tessera_reset_stats.restype = None
        self._lib.tessera_reset_stats.argtypes = []
        self._lib.tessera_dump_stats.restype = ctypes.c_int
        self._lib.tessera_dump_stats.argtypes = [ctypes.c_char_p]

        self.abi_version = int(self._lib.tessera_abi_version())
        if self.abi_version != 1:
            raise SystemExit(
                f"--with-shim: shim reports ABI version {self.abi_version}; "
                "this script speaks version 1 only."
            )

    def reset(self) -> None:
        self._lib.tessera_reset_stats()

    def read(self) -> dict[str, int]:
        out = TesseraStatsV1()
        rc = int(self._lib.tessera_get_stats_v1(ctypes.byref(out)))
        if rc != 0:
            raise SystemExit(f"--with-shim: tessera_get_stats_v1 returned {rc}")
        return out.as_dict()

    def dump(self, path: str) -> None:
        rc = int(self._lib.tessera_dump_stats(path.encode()))
        if rc != 0:
            raise SystemExit(f"--with-shim: tessera_dump_stats('{path}') returned {rc}")


# --------------------------------------------------------------------------
# CUPTI counting
# --------------------------------------------------------------------------
def _count_from_chrome_trace(path: str) -> dict[str, int]:
    """Count launch-ish events in a kineto chrome trace.

    The trace is parsed rather than read through prof.events() because the
    chrome-trace schema (cat/name/ph) has been stable across torch versions
    while the FunctionEvent attributes have not.
    """
    with open(path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", [])

    device_kernels = 0
    runtime_launches = 0
    driver_launches = 0
    graph_launches = 0
    device_memcpy = 0
    device_memset = 0

    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if cat == "kernel":
            device_kernels += 1
        elif cat in ("gpu_memcpy", "Memcpy"):
            device_memcpy += 1
        elif cat in ("gpu_memset", "Memset"):
            device_memset += 1
        elif cat in ("cuda_runtime", "Runtime"):
            if name in _RUNTIME_LAUNCH_NAMES:
                runtime_launches += 1
            elif name in _GRAPH_LAUNCH_NAMES:
                graph_launches += 1
        elif cat in ("cuda_driver", "Driver"):
            if name.startswith(_DRIVER_LAUNCH_PREFIXES):
                driver_launches += 1
            elif name in _GRAPH_LAUNCH_NAMES:
                graph_launches += 1

    return {
        "cupti_device_kernels": device_kernels,
        "cupti_runtime_launch_calls": runtime_launches,
        "cupti_driver_launch_calls": driver_launches,
        "cupti_graph_launches": graph_launches,
        "cupti_device_memcpy": device_memcpy,
        "cupti_device_memset": device_memset,
        "cupti_total_trace_events": len(events),
    }


def _sha256_tensor(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's raw bytes.

    numpy's tobytes() is the exact buffer and is used when numpy is present
    (it is pinned into the image for this reason). The fallback reinterprets
    the contiguous tensor as uint8 through torch alone and produces the same
    bytes, so the hash does not depend on which path ran - it only exists so a
    machine without numpy gets an answer instead of a traceback.
    """
    cpu = t.detach().to("cpu").contiguous()
    try:
        raw = cpu.numpy().tobytes()
    except (ImportError, RuntimeError):
        raw = bytes(cpu.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# The measured loops
# --------------------------------------------------------------------------
def _run_window_profiled(body: Callable[[], None], iters: int) -> dict[str, int]:
    """Pass A: exactly `iters` iterations inside a CUPTI window."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.json")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            torch.cuda.synchronize()
            for _ in range(iters):
                body()
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        return _count_from_chrome_trace(trace_path)


def _run_window_counted(
    body: Callable[[], None], iters: int, shim: ShimCounters | None
) -> dict[str, Any]:
    """Pass B: exactly `iters` iterations inside the shim-counter window."""
    if shim is not None:
        shim.reset()
    torch.cuda.synchronize()
    before = shim.read() if shim is not None else None
    for _ in range(iters):
        body()
    torch.cuda.synchronize()
    after = shim.read() if shim is not None else None

    if before is None or after is None:
        return {"shim_before": None, "shim_after": None, "shim_delta": None}
    delta = {k: after[k] - before[k] for k in after}
    return {"shim_before": before, "shim_after": after, "shim_delta": delta}


def run_loop(
    name: str,
    device: torch.device,
    iters: int,
    warmup: int,
    shim: ShimCounters | None,
    gemm_dim: int,
    control_elems: int,
) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    if name == "gemm":
        a = torch.randn(gemm_dim, gemm_dim, generator=gen, dtype=torch.float32).to(device)
        b = torch.randn(gemm_dim, gemm_dim, generator=gen, dtype=torch.float32).to(device)
        c = torch.zeros(gemm_dim, gemm_dim, dtype=torch.float32, device=device)
        tensors = {"a": a, "b": b, "c": c}

        def body() -> None:
            torch.mm(a, b, out=c)

        shape_note = f"{gemm_dim}x{gemm_dim} @ {gemm_dim}x{gemm_dim} fp32"
        expected_kernels_per_iter: int | None = None
    elif name == "control":
        x = torch.zeros(control_elems, dtype=torch.float32, device=device)
        tensors = {"x": x}

        def body() -> None:
            x.add_(1.0)

        shape_note = f"{control_elems} elements fp32"
        expected_kernels_per_iter = 1
    else:  # pragma: no cover - argparse restricts this
        raise SystemExit(f"unknown loop: {name}")

    for _ in range(warmup):
        body()
    torch.cuda.synchronize()

    cupti = _run_window_profiled(body, iters)
    counted = _run_window_counted(body, iters, shim)

    hashes = {key: _sha256_tensor(t) for key, t in tensors.items()}

    result: dict[str, Any] = {
        "loop": name,
        "shapes": shape_note,
        "seed": SEED,
        "warmup_iters": warmup,
        "window_iters": iters,
        "sha256": hashes,
        "cupti": cupti,
        "shim": counted,
        "expected_kernels_per_iter": expected_kernels_per_iter,
    }

    failures: list[str] = []

    # CUPTI internal consistency, recorded as a diagnostic rather than an
    # assertion: whether the runtime and driver categories are both traced is a
    # kineto build detail, so host-side launch calls can legitimately be 1x or
    # 2x the device kernel count. The number that G9 compares against is the
    # device kernel activity count, which is unambiguous.
    host_launches = cupti["cupti_runtime_launch_calls"] + cupti["cupti_driver_launch_calls"]
    result["cupti_host_launch_calls"] = host_launches
    result["cupti_host_matches_device"] = host_launches == cupti["cupti_device_kernels"]

    if cupti["cupti_graph_launches"] != 0:
        failures.append(
            f"CUPTI saw {cupti['cupti_graph_launches']} graph launches; the "
            "G9 comparison counts individual launches and is invalid when "
            "work is submitted as a graph"
        )
    if expected_kernels_per_iter is not None:
        want = expected_kernels_per_iter * iters
        if cupti["cupti_device_kernels"] != want:
            failures.append(
                f"control loop: CUPTI recorded {cupti['cupti_device_kernels']} "
                f"device kernels over the window, expected exactly {want}"
            )

    delta = counted["shim_delta"]
    if delta is not None:
        if delta["graph_launches"] != 0:
            failures.append(
                f"shim reported graph_launches={delta['graph_launches']} over "
                "the window; G9 compares per-launch counters and a non-zero "
                "graph launch count makes the comparison invalid"
            )
        if delta["launches"] != cupti["cupti_device_kernels"]:
            failures.append(
                f"G9: shim launches delta ({delta['launches']}) != CUPTI "
                f"device kernels ({cupti['cupti_device_kernels']})"
            )

    result["failures"] = failures
    result["pass"] = not failures
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--loop",
        choices=("gemm", "control", "both"),
        default="both",
        help="which loop to run (default: both)",
    )
    parser.add_argument("--iters", type=int, default=200, help="N, the window length")
    parser.add_argument("--warmup", type=int, default=20, help="warm-up iterations")
    parser.add_argument("--gemm-dim", type=int, default=512, help="square GEMM side")
    parser.add_argument(
        "--control-elems", type=int, default=1 << 20, help="control tensor length"
    )
    parser.add_argument(
        "--with-shim",
        action="store_true",
        help="read the shim's exported counters from libcuda.so.1 as well",
    )
    parser.add_argument(
        "--shim-dump",
        default=None,
        help="if --with-shim, also call tessera_dump_stats(path) at the end",
    )
    parser.add_argument("--json-out", default=None, help="write the full result JSON here")
    args = parser.parse_args(argv)

    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters must be >= 1 and --warmup >= 0")

    if not torch.cuda.is_available():
        print("torch_matmul_loop: CUDA is not available", file=sys.stderr)
        return 3

    # Determinism knobs. The setters are wrapped because torch has been moving
    # allow_tf32 towards the newer fp32_precision API and a raise there would
    # kill the run; what is NOT tolerated is the knob ending up in the wrong
    # state, which is checked by reading it back below. A silently
    # non-deterministic run cannot support a bitwise G8 claim.
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    tf32_setter_notes: dict[str, str] = {}
    for label, backend, attr in (
        ("torch.backends.cuda.matmul.allow_tf32", torch.backends.cuda.matmul, "allow_tf32"),
        ("torch.backends.cudnn.allow_tf32", torch.backends.cudnn, "allow_tf32"),
    ):
        try:
            setattr(backend, attr, False)
            tf32_setter_notes[label] = "set"
        except (AttributeError, RuntimeError, TypeError) as exc:
            tf32_setter_notes[label] = f"setter unavailable: {exc}"

    if torch.backends.cuda.matmul.allow_tf32:
        raise SystemExit(
            "torch.backends.cuda.matmul.allow_tf32 is still True after trying to "
            f"disable it ({tf32_setter_notes}); fp32 GEMMs would run on tensor "
            "cores and the G8 hash comparison would not be meaningful."
        )
    if torch.backends.cudnn.allow_tf32:
        raise SystemExit(
            "torch.backends.cudnn.allow_tf32 is still True after trying to "
            f"disable it ({tf32_setter_notes})."
        )

    device = torch.device("cuda", 0)
    torch.cuda.init()

    shim = ShimCounters() if args.with_shim else None

    loops = ["gemm", "control"] if args.loop == "both" else [args.loop]
    results = []
    for loop_name in loops:
        res = run_loop(
            loop_name,
            device,
            iters=args.iters,
            warmup=args.warmup,
            shim=shim,
            gemm_dim=args.gemm_dim,
            control_elems=args.control_elems,
        )
        results.append(res)
        print("TESSERA_JSON " + json.dumps(res, sort_keys=True), flush=True)

    if shim is not None and args.shim_dump:
        shim.dump(args.shim_dump)

    props = torch.cuda.get_device_properties(device)
    summary = {
        "kind": "torch_matmul_loop_summary",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessor_count": props.multi_processor_count,
        "cublas_workspace_config": _CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_setter_notes": tf32_setter_notes,
        "with_shim": bool(shim),
        "shim_abi_version": shim.abi_version if shim else None,
        "loops": results,
        "pass": all(r["pass"] for r in results),
    }
    print("TESSERA_JSON " + json.dumps(summary, sort_keys=True), flush=True)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)

    if not summary["pass"]:
        for res in results:
            for failure in res["failures"]:
                print(f"torch_matmul_loop: {res['loop']}: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
