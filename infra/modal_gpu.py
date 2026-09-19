"""Tessera's Modal GPU runner (M0).

Written against the installed modal client 1.5.5 (`modal --version`), using
that package's own type stubs for the API surface rather than recollection.

Design rules this file follows, because GPU seconds are the expensive resource
and the M0 cap is $5 of real money:

  * Everything that can happen without a GPU happens in the image build: apt,
    pip, the CUDA header fetch, both CPU builds (g++-13 and clang-18), and the
    nvcc build of the sample. The GPU functions only run already-built binaries.
  * Every function declares an explicit timeout, all <= 1200 s.
  * `cpu_tests` requests no GPU at all. Modal supports CPU-only functions, so
    proving the minimum compilers does not need to rent an L4.
  * The local entrypoint - never the container - appends one line per launch to
    results/spend/ledger.jsonl, so a container that dies still leaves a record.
  * Nothing is detached. `modal run --detach` is banned: a detached app can keep
    a GPU attached after this process exits.

Usage:

    modal run infra/modal_gpu.py --action platform_probe
    modal run infra/modal_gpu.py --action cpu-tests          # no GPU attached
    modal run infra/modal_gpu.py --action m0-gate            # no-shim baseline
    modal run infra/modal_gpu.py --action m0-gate --with-shim   # once WP-0.5 lands

`TESSERA_GPU` overrides the GPU type (default L4).
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from typing import Any, Optional

import modal

# ---------------------------------------------------------------------------
# Paths. Two worlds: the dev host (repo checkout) and the image (/opt/tessera).
# ---------------------------------------------------------------------------
LOCAL_REPO = pathlib.Path(__file__).resolve().parent.parent
IMAGE_REPO = "/opt/tessera"
IMAGE_DEPS = "/opt/tessera-deps"
IMAGE_BUILD_DIR = "/opt/tessera-build"
IMAGE_BIN = "/opt/tessera-bin"
IMAGE_META_DIR = "/opt/tessera-image"
IMAGE_MANIFEST = f"{IMAGE_META_DIR}/manifest.json"


def _load_versions() -> dict[str, Any]:
    """Read infra/versions.lock.

    This module is imported twice: on the dev host, where the repo is at
    LOCAL_REPO, and inside the container, where modal mounts only this .py file
    but the image carries the whole repo at IMAGE_REPO. Both are tried.
    """
    candidates = [
        pathlib.Path(__file__).resolve().parent / "versions.lock",
        pathlib.Path(IMAGE_REPO) / "infra" / "versions.lock",
    ]
    env_repo = os.environ.get("TESSERA_REPO")
    if env_repo:
        candidates.append(pathlib.Path(env_repo) / "infra" / "versions.lock")
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("rb") as fh:
                return tomllib.load(fh)
    raise RuntimeError(
        "infra/versions.lock not found; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


VERSIONS = _load_versions()

BASE_IMAGE_REF = VERSIONS["base_image"]["reference"]
PYTHON_VERSION = VERSIONS["python"]["version"]
CLANG_MAJOR = VERSIONS["compilers"]["clang"]["major"]
GCC_MAJOR = VERSIONS["compilers"]["gcc"]["major"]
CMAKE_VERSION = VERSIONS["build_tools"]["cmake"]["version"]
NINJA_VERSION = VERSIONS["build_tools"]["ninja"]["version"]
TORCH_VERSION = VERSIONS["python_packages"]["torch"]["version"]
NUMPY_VERSION = VERSIONS["python_packages"]["numpy"]["version"]
VLLM_VERSION = VERSIONS["python_packages"]["vllm"]["version"]

GPU = os.environ.get("TESSERA_GPU", "L4")
MILESTONE = "M0"

# Timeouts, seconds. The spending rule is <= 1200 s for every function.
PROBE_TIMEOUT_S = 600
CPU_TESTS_TIMEOUT_S = 1200
GATE_TIMEOUT_S = 1200

app = modal.App("tessera-m0")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
_IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        "build",  # a symlink to tmpfs on the dev host; never goes in the image
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        ".cache",
        "node_modules",
    }
)
_IGNORED_SUFFIXES = (".pyc", ".pyo", ".o", ".a", ".so", ".nsys-rep", ".qdrep")

# This module is deliberately NOT copied into the image. modal mounts the
# entrypoint file itself, so the container already has it, and leaving it out of
# the copied layer means editing the runner does not invalidate the layer that
# compiles the repo - which is the difference between a 3-second iteration and a
# five-minute one. Everything the container reads out of /opt/tessera (notably
# infra/versions.lock and infra/image_build.sh) is still copied.
_IGNORED_REPO_FILES = frozenset({"infra/modal_gpu.py"})


def _repo_ignore(rel_path: pathlib.Path) -> bool:
    """True for repo paths that must not enter the image.

    modal calls this with paths relative to the repo root and skips anything
    that returns True. `results/` is excluded on purpose: run artefacts are
    written on the dev host, and leaving them out means a second GPU run does
    not invalidate the (expensive) image layers.
    """
    parts = rel_path.parts
    if any(part in _IGNORED_DIR_NAMES for part in parts):
        return True
    if parts and parts[0] == "results":
        return True
    if rel_path.as_posix() in _IGNORED_REPO_FILES:
        return True
    return rel_path.name.endswith(_IGNORED_SUFFIXES)


def _repo_source_for_image() -> str:
    """The directory handed to add_local_dir.

    Not the repo itself: several agents work in this checkout at once, and modal
    hashes the tree, uploads it, then re-hashes it, aborting the whole build
    with "<file> was modified during build process" if anything changed in
    between. That already happened once here. The tree is ~300 KB, so it is
    copied to a fixed scratch path first and modal uploads the copy, which no
    one else is writing to.

    The path is fixed (not a mkdtemp) so repeated runs produce the same image
    layer. The copy only happens on the dev host; inside a container the image
    already carries the repo.
    """
    if not modal.is_local():
        return IMAGE_REPO

    dest = pathlib.Path(
        os.environ.get("TESSERA_SNAPSHOT_DIR", "/tmp/tessera-modal-src")
    ) / "tessera"

    def _copytree_ignore(directory: str, names: list[str]) -> set[str]:
        base = pathlib.Path(directory)
        return {
            name
            for name in names
            if _repo_ignore((base / name).relative_to(LOCAL_REPO))
        }

    # A concurrent writer can still delete a file between listdir and copy, so
    # the copy is retried. A persistent failure is raised, never swallowed.
    last_error: Optional[Exception] = None
    for _ in range(3):
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(LOCAL_REPO, dest, symlinks=True, ignore=_copytree_ignore)
            return str(dest)
        except (OSError, shutil.Error) as exc:
            last_error = exc
    raise RuntimeError(
        f"could not snapshot {LOCAL_REPO} into {dest} after 3 attempts: {last_error}"
    )


BASE_IMAGE = (
    modal.Image.from_registry(BASE_IMAGE_REF, add_python=PYTHON_VERSION)
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "TESSERA_REPO": IMAGE_REPO,
            "TESSERA_DEPS": IMAGE_DEPS,
            "TESSERA_BUILD_DIR": IMAGE_BUILD_DIR,
            "TESSERA_BIN": IMAGE_BIN,
            "TESSERA_IMAGE_DIR": IMAGE_META_DIR,
            # cuBLAS needs this set before its handle is created for
            # torch.use_deterministic_algorithms(True) to be usable at all.
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    .apt_install(
        "ca-certificates",
        "curl",
        "gnupg",
        "dirmngr",
        "git",
        "strace",
        "xz-utils",
        "software-properties-common",
        "lsb-release",
    )
    .run_commands(
        # clang-18 is not in jammy; it comes from apt.llvm.org.
        "curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key"
        " | gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg",
        f"echo 'deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg]"
        f" https://apt.llvm.org/jammy/ llvm-toolchain-jammy-{CLANG_MAJOR} main'"
        f" > /etc/apt/sources.list.d/llvm-{CLANG_MAJOR}.list",
        # g++-13 is not in jammy either; it comes from the toolchain PPA.
        "add-apt-repository -y ppa:ubuntu-toolchain-r/test",
        "apt-get update",
        "apt-get install -y --no-install-recommends"
        f" clang-{CLANG_MAJOR} clang-tools-{CLANG_MAJOR}"
        f" libclang-rt-{CLANG_MAJOR}-dev lld-{CLANG_MAJOR}"
        f" gcc-{GCC_MAJOR} g++-{GCC_MAJOR} libstdc++-{GCC_MAJOR}-dev"
        " ninja-build",
        "rm -rf /var/lib/apt/lists/*",
    )
    # jammy's cmake is 3.22 and CMakeLists.txt needs >= 3.28, so cmake comes
    # from the pinned wheel. ninja is pinned the same way and shadows the apt
    # one on PATH; the image manifest records which binary actually resolved.
    .pip_install(f"cmake=={CMAKE_VERSION}", f"ninja=={NINJA_VERSION}")
    .pip_install(f"torch=={TORCH_VERSION}")
    # Separate layer, deliberately after torch: numpy changes must not force
    # the multi-GB torch layer to rebuild. torch warns at import without it and
    # Tensor.numpy() - which is how the G8 hashes are taken - needs it.
    .pip_install(f"numpy=={NUMPY_VERSION}")
    # The repo goes in as a copied layer so the builds below are cached. The
    # source is a snapshot, not the live checkout: see _repo_source_for_image.
    .add_local_dir(_repo_source_for_image(), IMAGE_REPO, copy=True, ignore=_repo_ignore)
    .run_commands(f"bash {IMAGE_REPO}/infra/image_build.sh")
)

# M1's layer, defined here so the vllm pin lives with everything else. It is
# NOT referenced by any M0 function, so modal never builds it during M0.
VLLM_IMAGE = BASE_IMAGE.pip_install(f"vllm=={VLLM_VERSION}")


# ---------------------------------------------------------------------------
# Container-side helpers
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(
    cmd: list[str],
    timeout: int,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> dict[str, Any]:
    """Run a command and record everything about it.

    The exit code is returned, never swallowed: callers that treat a non-zero
    code as a finding (the MPS probe) say so explicitly, and callers that
    require success check `ok` and raise (I-9).
    """
    started = time.monotonic()
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
            cwd=cwd,
        )
        return {
            "cmd": cmd,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "timed_out": False,
            "wall_s": round(time.monotonic() - started, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "exit_code": None,
            "ok": False,
            "timed_out": True,
            "wall_s": round(time.monotonic() - started, 3),
            "stdout": exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
        }
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "exit_code": None,
            "ok": False,
            "timed_out": False,
            "wall_s": round(time.monotonic() - started, 3),
            "stdout": "",
            "stderr": f"not found: {exc}",
        }


def _require(step: str, result: dict[str, Any]) -> dict[str, Any]:
    if not result["ok"]:
        raise RuntimeError(
            f"{step} failed (exit {result['exit_code']}, timed_out="
            f"{result['timed_out']})\n--- stdout ---\n{result['stdout']}"
            f"\n--- stderr ---\n{result['stderr']}"
        )
    return result


def _json_safe(obj: Any) -> Any:
    """Force a return value through JSON before it crosses back to the caller.

    modal pickles return values, and a pickled library type needs that library
    importable on the *local* side to come back. torch.__version__ is a str
    subclass (torch.torch_version.TorchVersion), so returning it raised
    "Deserialization failed because the 'torch' module is not available in the
    local environment" after the GPU work had already been done and paid for.
    Round-tripping through JSON makes that class of failure impossible, and
    these results are written out as JSON anyway.
    """
    return json.loads(json.dumps(obj, default=str, sort_keys=True))


def _parse_tessera_json(stdout: str) -> list[dict[str, Any]]:
    """Pull every `TESSERA_JSON {...}` line out of a program's stdout."""
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("TESSERA_JSON "):
            out.append(json.loads(line[len("TESSERA_JSON ") :]))
    return out


def _image_manifest() -> dict[str, Any]:
    with open(IMAGE_MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def _driver_probe() -> dict[str, Any]:
    """Ask libcuda.so.1 directly, via ctypes: this is the same library the shim
    will masquerade as, so knowing exactly what it reports matters."""
    import ctypes

    info: dict[str, Any] = {}
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return {"libcuda_loadable": False, "error": str(exc)}
    info["libcuda_loadable"] = True

    version = ctypes.c_int(0)
    rc = int(lib.cuDriverGetVersion(ctypes.byref(version)))
    info["cuDriverGetVersion_rc"] = rc
    info["cuda_driver_version"] = version.value if rc == 0 else None

    rc_init = int(lib.cuInit(0))
    info["cuInit_rc"] = rc_init
    if rc_init != 0:
        return info

    count = ctypes.c_int(0)
    info["cuDeviceGetCount_rc"] = int(lib.cuDeviceGetCount(ctypes.byref(count)))
    info["device_count"] = count.value

    devices = []
    for ordinal in range(count.value):
        dev = ctypes.c_int(0)
        if int(lib.cuDeviceGet(ctypes.byref(dev), ordinal)) != 0:
            continue
        name = ctypes.create_string_buffer(256)
        lib.cuDeviceGetName(name, 256, dev)

        def attribute(attr_id: int) -> Optional[int]:
            value = ctypes.c_int(0)
            if int(lib.cuDeviceGetAttribute(ctypes.byref(value), attr_id, dev)) != 0:
                return None
            return value.value

        devices.append(
            {
                "ordinal": ordinal,
                "name": name.value.decode(errors="replace"),
                # Enum values read from cuda.h 12.6 (CUdevice_attribute).
                "multiprocessor_count": attribute(16),  # MULTIPROCESSOR_COUNT
                "compute_capability_major": attribute(75),
                "compute_capability_minor": attribute(76),
                "clock_rate_khz": attribute(13),
                "async_engine_count": attribute(40),
                "compute_mode": attribute(20),  # CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
            }
        )
    info["devices"] = devices
    return info


def _torch_probe() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not out["cuda_available"]:
        return out
    props = torch.cuda.get_device_properties(0)
    cudnn = torch.backends.cudnn.version()
    out.update(
        {
            "device_name": str(props.name),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": int(props.multi_processor_count),
            "total_memory_bytes": int(props.total_memory),
            "cudnn_version": int(cudnn) if cudnn is not None else None,
        }
    )
    return out


def _gpu_identity() -> dict[str, Any]:
    """The identity block every results JSON must carry (I-1 traceability)."""
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,compute_mode,memory.total",
            "--format=csv,noheader",
        ],
        timeout=60,
    )
    driver = _driver_probe()
    identity: dict[str, Any] = {
        "nvidia_smi_query": smi["stdout"].strip(),
        "nvidia_smi_exit_code": smi["exit_code"],
        "gpu_requested": GPU,
        "cuda_driver_version": driver.get("cuda_driver_version"),
        "driver_probe": driver,
    }
    if smi["ok"] and smi["stdout"].strip():
        fields = [f.strip() for f in smi["stdout"].strip().splitlines()[0].split(",")]
        if len(fields) >= 6:
            identity.update(
                {
                    "gpu_model": fields[1],
                    "gpu_uuid": fields[2],
                    "driver_version": fields[3],
                    "compute_mode": fields[4],
                    "memory_total": fields[5],
                }
            )
    return identity


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
@app.function(image=BASE_IMAGE, gpu=GPU, timeout=PROBE_TIMEOUT_S)
def platform_probe() -> dict[str, Any]:
    """M1 pre-flight (H8): what does this box actually let us do?

    The M1 lever ladder depends on whether MPS and compute-policy are usable
    inside a Modal container. CLAUDE.md says that if MPS does not start here,
    the M1 MPS rows have to run on a root-capable hourly instance - so finding
    out now, while a GPU is already attached for a few cheap seconds, is what
    lets the M1 budget be written from a measurement instead of a hope.
    """
    started = time.monotonic()
    result: dict[str, Any] = {
        "kind": "platform_probe",
        "started_utc": _utcnow(),
        "image_manifest": _image_manifest(),
        "gpu_identity": _gpu_identity(),
    }

    devices = result["gpu_identity"].get("driver_probe", {}).get("devices") or []
    result["sm_count"] = devices[0]["multiprocessor_count"] if devices else None

    result["nvidia_smi"] = _run(["nvidia-smi"], timeout=60)
    result["nvidia_smi_compute"] = _run(["nvidia-smi", "-q", "-d", "COMPUTE"], timeout=60)
    result["torch"] = _torch_probe()

    # --- privileges and device nodes ---
    cap_eff = ""
    try:
        for line in open("/proc/self/status", encoding="utf-8"):
            if line.startswith("CapEff:"):
                cap_eff = line.split()[1]
                break
    except OSError as exc:  # pragma: no cover - /proc always exists on Linux
        cap_eff = f"unreadable: {exc}"
    result["privileges"] = {
        "euid": os.geteuid(),
        "cap_eff": cap_eff,
        "dev_nvidia_nodes": sorted(
            p for p in os.listdir("/dev") if p.startswith("nvidia")
        )
        if os.path.isdir("/dev")
        else [],
    }

    # --- nvidia-smi compute-policy ---
    # Non-zero here is a finding, not an error: the question the probe answers
    # is precisely "is this subcommand available in a Modal container?".
    policy = _run(["nvidia-smi", "compute-policy", "--list"], timeout=60)
    result["compute_policy"] = {
        "available": policy["ok"],
        "exit_code": policy["exit_code"],
        "stdout": policy["stdout"],
        "stderr": policy["stderr"],
    }

    # --- MPS ---
    mps: dict[str, Any] = {}
    mps_control = shutil.which("nvidia-cuda-mps-control")
    mps["control_binary"] = mps_control
    mps["control_binary_present"] = mps_control is not None
    mps["server_binary"] = shutil.which("nvidia-cuda-mps-server")

    if mps_control is None:
        mps["starts"] = False
        mps["reason"] = (
            "nvidia-cuda-mps-control is not present in the container. It is part "
            "of the driver's userspace utilities, which libnvidia-container "
            "injects only for some capability sets."
        )
    else:
        pipe_dir = "/tmp/tessera-mps/pipe"
        log_dir = "/tmp/tessera-mps/log"
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        mps_env = {
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
        }
        mps["env"] = mps_env

        # Every timeout below is deliberately tight. A GPU is attached for the
        # whole of this function, so a control command that hangs waiting for a
        # daemon that never came up would burn the milestone's budget.
        start = _run([mps_control, "-d"], timeout=45, env=mps_env)
        mps["start"] = start
        mps["starts"] = start["ok"]

        # Ask the control daemon whether it is actually there. `echo <cmd> |`
        # is how nvidia-cuda-mps-control takes non-interactive commands.
        status = _run(
            ["bash", "-c", f"echo get_server_list | {mps_control}"],
            timeout=20,
            env=mps_env,
        )
        mps["get_server_list"] = status

        if start["ok"]:
            # Does a CUDA client attach through MPS? Run the already-built
            # sample with the MPS env set; if the server is up it becomes an
            # MPS client, which is the only way to see whether
            # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is honoured at all.
            client = _run(
                [f"{IMAGE_BIN}/vector_add"],
                timeout=120,
                env={**mps_env, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50"},
            )
            mps["client_run"] = {
                "exit_code": client["exit_code"],
                "ok": client["ok"],
                "stdout_tail": client["stdout"][-2000:],
                "stderr_tail": client["stderr"][-2000:],
                "parsed": _parse_tessera_json(client["stdout"]),
            }
            mps["get_server_list_after_client"] = _run(
                ["bash", "-c", f"echo get_server_list | {mps_control}"],
                timeout=20,
                env=mps_env,
            )
        else:
            mps["client_run"] = {
                "skipped": True,
                "reason": "the MPS control daemon did not start, so there is no "
                "server for a client to attach to; m0_gate runs the same sample "
                "without MPS anyway.",
            }

        quit_result = _run(
            ["bash", "-c", f"echo quit | {mps_control}"], timeout=45, env=mps_env
        )
        mps["quit"] = quit_result

        logs = {}
        for name in ("control.log", "server.log"):
            path = os.path.join(log_dir, name)
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    logs[name] = fh.read()[-4000:]
        mps["logs"] = logs
        mps["pipe_dir_contents"] = sorted(os.listdir(pipe_dir)) if os.path.isdir(pipe_dir) else []

    # Whether each CUDA_MPS_* knob was actually honoured, decided from evidence
    # rather than from documentation. PIPE/LOG_DIRECTORY are settled by the
    # daemon having created files there. ACTIVE_THREAD_PERCENTAGE is settled by
    # the MPS client's own view of the device: a client capped at P% sees
    # roughly P% of the SMs, so the sample's multiprocessor_count under MPS is
    # compared against the count the driver reports outside it.
    client_parsed = (mps.get("client_run") or {}).get("parsed") or []
    sm_under_mps = client_parsed[0].get("multiprocessor_count") if client_parsed else None
    sm_from_driver = result.get("sm_count")
    if sm_under_mps is None or sm_from_driver is None:
        thread_pct_honoured: Optional[bool] = None
    else:
        thread_pct_honoured = sm_under_mps < sm_from_driver

    mps["env_knobs"] = {
        "CUDA_MPS_PIPE_DIRECTORY": {
            "set_to": mps.get("env", {}).get("CUDA_MPS_PIPE_DIRECTORY"),
            "honoured": bool(mps.get("pipe_dir_contents")),
            "evidence": "the control daemon created its socket in that directory",
        },
        "CUDA_MPS_LOG_DIRECTORY": {
            "set_to": mps.get("env", {}).get("CUDA_MPS_LOG_DIRECTORY"),
            "honoured": bool(mps.get("logs")),
            "evidence": "the control daemon wrote control.log there",
        },
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": {
            "set_to": "50" if mps.get("control_binary_present") else None,
            "honoured": thread_pct_honoured,
            "sm_count_seen_by_mps_client": sm_under_mps,
            "sm_count_seen_by_driver": sm_from_driver,
            "evidence": (
                "bench/samples/vector_add reports the device's "
                "multiProcessorCount; run as an MPS client with the percentage "
                "set it sees a reduced count, which is the cap taking effect. "
                "null means there was no MPS server to attach to."
            ),
        },
    }
    result["mps"] = mps

    # Pre-flight for G9's ground truth. torch.profiler's CUDA activity comes
    # from CUPTI, and if CUPTI cannot initialise in this container the profiler
    # reports zero kernels instead of failing - which would turn the gate's
    # independent counter into a silent zero. Five iterations of the control
    # loop settle it, and they exercise the whole eval script (determinism
    # knobs, trace parsing, tensor hashing, numpy) in a container that already
    # has the GPU attached, rather than discovering it during the gate run.
    smoke = _run(
        [
            sys.executable,
            f"{IMAGE_REPO}/eval/m0/torch_matmul_loop.py",
            "--loop",
            "control",
            "--iters",
            "5",
            "--warmup",
            "2",
        ],
        timeout=300,
    )
    result["cupti_preflight"] = {
        "command": smoke["cmd"],
        "exit_code": smoke["exit_code"],
        "ok": smoke["ok"],
        "wall_s": smoke["wall_s"],
        "parsed": _parse_tessera_json(smoke["stdout"]),
        "stderr_tail": smoke["stderr"][-4000:],
    }

    result["wall_s"] = round(time.monotonic() - started, 3)
    result["finished_utc"] = _utcnow()
    # The MPS answer is informational either way (the M0 gate doc says the
    # pre-flight is not an M0 criterion), but a GPU that could not be
    # identified, or a CUPTI that does not work, must be loud: the gate run
    # depends on both.
    result["pass"] = bool(result["gpu_identity"].get("gpu_model")) and smoke["ok"]
    return _json_safe(result)


@app.function(image=BASE_IMAGE, timeout=CPU_TESTS_TIMEOUT_S)
def cpu_tests() -> dict[str, Any]:
    """Run the CPU test suite in-image with BOTH minimum compilers.

    No GPU: modal allows CPU-only functions, and nothing here touches a device.
    The compiles already happened in the image build (that is where -Werror was
    proved); this runs the resulting ctest suites.
    """
    started = time.monotonic()
    manifest = _image_manifest()
    result: dict[str, Any] = {
        "kind": "cpu_tests",
        "started_utc": _utcnow(),
        "image_manifest": manifest,
        "gpu_requested": None,
        "suites": {},
    }

    for tree in manifest["paths"]["build_trees"]:
        test_dir = os.path.join(IMAGE_BUILD_DIR, tree)
        run = _run(
            [
                "ctest",
                "--test-dir",
                test_dir,
                "--output-on-failure",
                "--label-exclude",
                "gpu",
                "-j",
                "4",
            ],
            timeout=900,
        )
        result["suites"][tree] = {
            "exit_code": run["exit_code"],
            "ok": run["ok"],
            "timed_out": run["timed_out"],
            "wall_s": run["wall_s"],
            "stdout_tail": run["stdout"][-8000:],
            "stderr_tail": run["stderr"][-4000:],
        }

    result["wall_s"] = round(time.monotonic() - started, 3)
    result["finished_utc"] = _utcnow()
    result["pass"] = bool(result["suites"]) and all(
        suite["ok"] for suite in result["suites"].values()
    )
    return _json_safe(result)


@app.function(image=BASE_IMAGE, gpu=GPU, timeout=GATE_TIMEOUT_S)
def m0_gate(
    with_shim: bool = False, iters: int = 200, warmup: int = 20
) -> dict[str, Any]:
    """G8/G9 acceptance.

    The shim does not exist yet (WP-0.5). `with_shim` is therefore a flag: with
    it off - which is all M0 can run today - this establishes the baseline half
    of G8 (the hashes a shimmed run must reproduce) and the CUPTI ground truth
    for G9. A later agent flips the flag on and the same function compares the
    two halves.
    """
    started = time.monotonic()
    result: dict[str, Any] = {
        "kind": "m0_gate",
        "started_utc": _utcnow(),
        "with_shim": with_shim,
        "window_iters": iters,
        "warmup_iters": warmup,
        "image_manifest": _image_manifest(),
        "gpu_identity": _gpu_identity(),
    }

    vector_add = f"{IMAGE_BIN}/vector_add"
    loop_script = f"{IMAGE_REPO}/eval/m0/torch_matmul_loop.py"
    with_shim_sh = f"{IMAGE_REPO}/infra/with_shim.sh"

    def run_baseline() -> dict[str, Any]:
        sample = _require("baseline vector_add", _run([vector_add], timeout=300))
        loop = _require(
            "baseline torch_matmul_loop",
            _run(
                [
                    sys.executable,
                    loop_script,
                    "--loop",
                    "both",
                    "--iters",
                    str(iters),
                    "--warmup",
                    str(warmup),
                ],
                timeout=900,
            ),
        )
        return {
            "vector_add": {
                "command": [vector_add],
                "exit_code": sample["exit_code"],
                "wall_s": sample["wall_s"],
                "parsed": _parse_tessera_json(sample["stdout"]),
                "stdout_tail": sample["stdout"][-4000:],
            },
            "torch_matmul_loop": {
                "command": loop["cmd"],
                "exit_code": loop["exit_code"],
                "wall_s": loop["wall_s"],
                "parsed": _parse_tessera_json(loop["stdout"]),
                "stderr_tail": loop["stderr"][-4000:],
            },
        }

    result["baseline"] = run_baseline()

    # Lift the versions the results envelope needs out of what the workloads
    # themselves reported, so the envelope quotes measurements rather than
    # re-deriving them.
    summary = next(
        (
            entry
            for entry in result["baseline"]["torch_matmul_loop"]["parsed"]
            if entry.get("kind") == "torch_matmul_loop_summary"
        ),
        {},
    )
    sample_json = result["baseline"]["vector_add"]["parsed"]
    result["torch"] = {
        "torch_version": summary.get("torch_version"),
        "torch_cuda_version": summary.get("torch_cuda_version"),
        "cudnn_version": summary.get("cudnn_version"),
        "deterministic_algorithms": summary.get("deterministic_algorithms"),
        "allow_tf32_matmul": summary.get("allow_tf32_matmul"),
        "cublas_workspace_config": summary.get("cublas_workspace_config"),
    }
    if sample_json:
        result["gpu_identity"]["cuda_runtime_version"] = sample_json[0].get(
            "cuda_runtime_version"
        )

    if not with_shim:
        result["shim"] = {
            "requested": False,
            "reason": (
                "libtessera does not exist yet (WP-0.5). This run records the "
                "no-shim baseline: the hashes a shimmed run must reproduce for "
                "G8, and the CUPTI launch counts a shimmed run must match for "
                "G9."
            ),
        }
        result["g8"] = {
            "status": "baseline_only",
            "baseline_hashes": _collect_hashes(result["baseline"]),
        }
        result["g9"] = {
            "status": "baseline_only",
            "cupti_counts": _collect_cupti(result["baseline"]),
        }
        result["pass"] = True
    else:
        masq_dir = os.environ.get(
            "TESSERA_MASQ_DIR", f"{IMAGE_BUILD_DIR}/gcc-{GCC_MAJOR}/shim/masq"
        )
        shim_env = {"TESSERA_MASQ_DIR": masq_dir}
        sample = _require(
            "shim vector_add",
            _run([with_shim_sh, vector_add], timeout=300, env=shim_env),
        )
        loop = _require(
            "shim torch_matmul_loop",
            _run(
                [
                    with_shim_sh,
                    sys.executable,
                    loop_script,
                    "--loop",
                    "both",
                    "--iters",
                    str(iters),
                    "--warmup",
                    str(warmup),
                    "--with-shim",
                ],
                timeout=900,
                env=shim_env,
            ),
        )
        result["shim"] = {
            "requested": True,
            "masq_dir": masq_dir,
            "vector_add": {
                "command": [with_shim_sh, vector_add],
                "exit_code": sample["exit_code"],
                "wall_s": sample["wall_s"],
                "parsed": _parse_tessera_json(sample["stdout"]),
                "stdout_tail": sample["stdout"][-4000:],
            },
            "torch_matmul_loop": {
                "command": loop["cmd"],
                "exit_code": loop["exit_code"],
                "wall_s": loop["wall_s"],
                "parsed": _parse_tessera_json(loop["stdout"]),
                "stderr_tail": loop["stderr"][-4000:],
            },
        }
        result["g8"] = _compare_g8(result["baseline"], result["shim"])
        result["g9"] = _compare_g9(result["shim"])
        result["pass"] = result["g8"]["pass"] and result["g9"]["pass"]

    result["wall_s"] = round(time.monotonic() - started, 3)
    result["finished_utc"] = _utcnow()
    return _json_safe(result)


def _collect_hashes(half: dict[str, Any]) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    sample = half["vector_add"]["parsed"]
    if sample:
        hashes["vector_add"] = sample[0].get("sha256")
    for entry in half["torch_matmul_loop"]["parsed"]:
        if entry.get("kind") == "torch_matmul_loop_summary":
            for loop in entry.get("loops", []):
                hashes[f"torch::{loop['loop']}"] = loop.get("sha256")
    return hashes


def _collect_cupti(half: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    sample = half["vector_add"]["parsed"]
    if sample:
        counts["vector_add_self_reported_launches"] = sample[0].get("kernel_launches")
    for entry in half["torch_matmul_loop"]["parsed"]:
        if entry.get("kind") == "torch_matmul_loop_summary":
            for loop in entry.get("loops", []):
                counts[f"torch::{loop['loop']}"] = loop.get("cupti")
    return counts


def _compare_g8(baseline: dict[str, Any], shim: dict[str, Any]) -> dict[str, Any]:
    """G8: outputs must be bitwise identical with and without the shim."""
    base_hashes = _collect_hashes(baseline)
    shim_hashes = _collect_hashes(shim)
    mismatches = []
    for key, value in base_hashes.items():
        if shim_hashes.get(key) != value:
            mismatches.append(
                {"artifact": key, "baseline": value, "with_shim": shim_hashes.get(key)}
            )
    missing = sorted(set(base_hashes) ^ set(shim_hashes))
    return {
        "status": "compared",
        "baseline_hashes": base_hashes,
        "shim_hashes": shim_hashes,
        "mismatches": mismatches,
        "missing_on_one_side": missing,
        "pass": not mismatches and not missing and bool(base_hashes),
    }


def _compare_g9(shim: dict[str, Any]) -> dict[str, Any]:
    """G9: the shim's launch-count delta must equal the CUPTI ground truth."""
    rows = []
    ok = True
    sample = shim["vector_add"]["parsed"]
    if sample:
        rows.append(
            {
                "workload": "vector_add",
                "expected": sample[0].get("kernel_launches"),
                "shim_launches": None,
                "note": (
                    "Self-reported launch count from the sample itself. The "
                    "sample is a plain C++ binary and never calls "
                    "tessera_get_stats_v1, so its shim-side count has to come "
                    "from the shim dumping stats at process exit. Whoever "
                    "lands WP-0.5 must add that (an env knob honoured by the "
                    "shim, read back here) before this row can be compared; "
                    "until then only the torch rows decide G9."
                ),
            }
        )
    for entry in shim["torch_matmul_loop"]["parsed"]:
        if entry.get("kind") != "torch_matmul_loop_summary":
            continue
        for loop in entry.get("loops", []):
            cupti = loop.get("cupti", {})
            delta = (loop.get("shim") or {}).get("shim_delta")
            row = {
                "workload": f"torch::{loop['loop']}",
                "cupti_device_kernels": cupti.get("cupti_device_kernels"),
                "shim_launches": (delta or {}).get("launches"),
                "shim_graph_launches": (delta or {}).get("graph_launches"),
                "loop_pass": loop.get("pass"),
                "failures": loop.get("failures", []),
            }
            if not loop.get("pass"):
                ok = False
            rows.append(row)
    return {"status": "compared", "rows": rows, "pass": ok and bool(rows)}


# ---------------------------------------------------------------------------
# Local entrypoint: results JSON + spend ledger
# ---------------------------------------------------------------------------
_ACTIONS = ("platform_probe", "cpu_tests", "m0_gate")

_RESULT_FILENAMES = {
    "platform_probe": "platform.json",
    "cpu_tests": "cpu_tests.json",
    "m0_gate_baseline": "gpu_baseline.json",
    "m0_gate_shim": "gpu_gate.json",
}


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        proc = subprocess.run(
            ["git", "-C", str(LOCAL_REPO), *args],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_paths": status.splitlines() if status else [],
    }


def _append_ledger(entry: dict[str, Any]) -> pathlib.Path:
    """One line per launch, written locally so a dead container still leaves a
    record (I-8). Ordered keys keep the file diff-friendly."""
    path = LOCAL_REPO / "results" / "spend" / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {
        "ts_start": entry["ts_start"],
        "ts_end": entry["ts_end"],
        "wall_s": entry["wall_s"],
        "milestone": entry["milestone"],
        "gpu": entry["gpu"],
        "function": entry["function"],
        "command": entry["command"],
        "exit_code": entry["exit_code"],
        "app_id": entry["app_id"],
    }
    for key, value in entry.items():
        if key not in ordered:
            ordered[key] = value
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ordered, sort_keys=False) + "\n")
    return path


@app.local_entrypoint()
def main(
    action: str = "platform_probe",
    with_shim: bool = False,
    iters: int = 200,
    warmup: int = 20,
) -> None:
    action = action.replace("-", "_")
    if action not in _ACTIONS:
        raise SystemExit(
            f"unknown --action '{action}'; expected one of {', '.join(_ACTIONS)}"
        )
    if with_shim and action != "m0_gate":
        raise SystemExit("--with-shim only applies to --action m0_gate")

    gpu_for_ledger = None if action == "cpu_tests" else GPU
    command = (
        f"TESSERA_GPU={GPU} modal run infra/modal_gpu.py --action {action}"
        + (" --with-shim" if with_shim else "")
        + (f" --iters {iters} --warmup {warmup}" if action == "m0_gate" else "")
    )

    # Note on ledger coverage: this body runs inside `app.run()`, which means
    # modal has already built (or reused) the image by the time the first line
    # below executes. A failure during the image build therefore leaves no
    # ledger line - correctly, because an image build attaches no GPU and
    # spends no GPU budget. The ledger records launches, which is what the cap
    # is measured in.
    ts_start = _utcnow()
    started = time.monotonic()
    payload: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    exit_code = 0

    try:
        if action == "platform_probe":
            payload = platform_probe.remote()
        elif action == "cpu_tests":
            payload = cpu_tests.remote()
        else:
            payload = m0_gate.remote(with_shim=with_shim, iters=iters, warmup=warmup)
        if not payload.get("pass", False):
            exit_code = 1
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised below
        error = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    wall_s = round(time.monotonic() - started, 3)
    ts_end = _utcnow()

    ledger_path = _append_ledger(
        {
            "ts_start": ts_start,
            "ts_end": ts_end,
            "wall_s": wall_s,
            "milestone": MILESTONE,
            "gpu": gpu_for_ledger,
            "function": action,
            "command": command,
            "exit_code": exit_code,
            "app_id": app.app_id,
            "error": error,
            "remote_wall_s": payload.get("wall_s") if payload else None,
            "timeout_s": {
                "platform_probe": PROBE_TIMEOUT_S,
                "cpu_tests": CPU_TESTS_TIMEOUT_S,
                "m0_gate": GATE_TIMEOUT_S,
            }[action],
        }
    )
    print(f"ledger: appended to {ledger_path}")

    identity = (payload or {}).get("gpu_identity", {})
    manifest = (payload or {}).get("image_manifest", {})
    key = action
    if action == "m0_gate":
        key = "m0_gate_shim" if with_shim else "m0_gate_baseline"
    out_path = LOCAL_REPO / "results" / MILESTONE.lower() / _RESULT_FILENAMES[key]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "kind": f"tessera_{action}_result",
        "milestone": MILESTONE,
        "utc": ts_start,
        "finished_utc": ts_end,
        "wall_s_local": wall_s,
        "git": _git_state(),
        "command": command,
        "argv": sys.argv,
        "modal": {
            "client_version": modal.__version__,
            "app_name": app.name,
            "app_id": app.app_id,
            "function": action,
            "timeout_s": {
                "platform_probe": PROBE_TIMEOUT_S,
                "cpu_tests": CPU_TESTS_TIMEOUT_S,
                "m0_gate": GATE_TIMEOUT_S,
            }[action],
            "gpu_requested": gpu_for_ledger,
            "detached": False,
        },
        "image": {
            "base_image": BASE_IMAGE_REF,
            "base_image_digest": VERSIONS["base_image"]["digest"],
            "manifest": manifest,
        },
        "gpu": {
            "model": identity.get("gpu_model"),
            "uuid": identity.get("gpu_uuid"),
            "driver_version": identity.get("driver_version"),
            "cuda_driver_version": identity.get("cuda_driver_version"),
            "cuda_runtime_version": identity.get("cuda_runtime_version"),
            "cuda_runtime_version_torch": ((payload or {}).get("torch") or {}).get(
                "torch_cuda_version"
            ),
            "compute_mode": identity.get("compute_mode"),
        },
        "error": error,
        "exit_code": exit_code,
        "result": payload,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2, sort_keys=True)
    print(f"results: wrote {out_path}")

    if error:
        print(f"modal_gpu: {action} failed: {error}", file=sys.stderr)
    if exit_code != 0:
        sys.exit(exit_code)
