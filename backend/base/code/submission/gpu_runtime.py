"""Shared process and artifact helpers for the native Latti GPU submission."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


RESULT_PREFIX = "__LATTI_STAGE_RESULT__ "
SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent
ASSET_DIR = SUBMISSION_DIR / "latti"
MODEL_DIR = ASSET_DIR / "model"
ARTIFACT_ROOT = Path(os.environ.get("CRYPTOFACE_ARTIFACT_ROOT") or
                     (REPO_ROOT / ".cryptoface-artifact-root").read_text().strip()).resolve()
SUBMISSION_BUILD_DIR = Path(os.environ.get("CRYPTOFACE_BUILD_DIR") or
                            str(ARTIFACT_ROOT / "submission-build"))
NATIVE_BINARY = SUBMISSION_BUILD_DIR / "latti_stage_runtime"
LATTI_AI_SOURCE = ARTIFACT_ROOT / ".deps" / "latti-ai"
LATTI_AI_BUILD = LATTI_AI_SOURCE / "build-h200"
RUNTIME_LIBRARY = LATTI_AI_BUILD / "inference" / "lattisense" / "liblattisense.so"
LATTIGO_LIBRARY_DIR = (
    LATTI_AI_SOURCE / "inference" / "lattisense" / "fhe_ops_lib" / "lattigo" / "go_sdk"
)
HEONGPU_LIBRARY_DIR = (
    LATTI_AI_SOURCE / "inference" / "lattisense" / "backends" / "HEonGPU" / "install" / "lib"
)
BUILD_MANIFEST = SUBMISSION_BUILD_DIR / "build_manifest.json"


def describe_returncode(returncode: int | None) -> str:
    if returncode is None:
        return "return code unavailable"
    if returncode >= 0:
        return f"exit code {returncode}"
    signum = -returncode
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = "UNKNOWN"
    return f"signal {signum} ({signal_name})"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_runtime(cfg: dict) -> None:
    required = {
        MODEL_DIR / "server" / "model_parameters.h5": cfg["model_parameters_sha256"],
        MODEL_DIR / "server" / "mega_ag.json": cfg["mega_ag_sha256"],
    }
    for path, expected in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing GPU submission artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch for {path}: {actual} != {expected}")

    if not BUILD_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing source-build manifest: {BUILD_MANIFEST}; "
            "run scripts/build_submission.sh"
        )
    manifest = json.loads(BUILD_MANIFEST.read_text())
    expected_metadata = {
        "schema_version": 1,
        "cuda_architecture": cfg["cuda_architecture"],
        "latti_ai_commit": cfg["latti_ai_commit"],
        "lattisense_commit": cfg["lattisense_commit"],
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Build manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}"
            )
    built_artifacts = {
        "native_binary": NATIVE_BINARY,
        "lattisense_runtime": RUNTIME_LIBRARY,
    }
    for name, path in built_artifacts.items():
        record = manifest.get("artifacts", {}).get(name, {})
        if not path.is_file():
            raise FileNotFoundError(f"Missing source-built artifact: {path}")
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise ValueError(
                f"Source-built artifact hash mismatch for {path}: "
                f"{actual} != {record.get('sha256')}"
            )
    if not (LATTIGO_LIBRARY_DIR / "liblattigo.so").is_file():
        raise FileNotFoundError(
            f"Missing source-built Lattigo runtime: {LATTIGO_LIBRARY_DIR / 'liblattigo.so'}"
        )
    if not HEONGPU_LIBRARY_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing source-built HEonGPU library directory: {HEONGPU_LIBRARY_DIR}"
        )


def runtime_environment(gpu: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    library_dir = RUNTIME_LIBRARY.parent
    cuda_dirs = [
        Path("/usr/local/lib/ollama/cuda_v12"),
        Path("/usr/local/cuda-12.9/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda-12.9/lib64"),
        Path("/usr/local/cuda-12/lib64"),
        Path("/usr/local/cuda-13.0/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda-13.0/lib64"),
        Path("/usr/local/cuda-13/lib64"),
        Path("/usr/local/cuda/lib64"),
    ]
    library_paths = [
        str(library_dir),
        str(LATTIGO_LIBRARY_DIR),
        str(HEONGPU_LIBRARY_DIR),
        *(str(path) for path in cuda_dirs if path.exists()),
    ]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    preload = str(RUNTIME_LIBRARY)
    if env.get("LD_PRELOAD"):
        preload += ":" + env["LD_PRELOAD"]
    env["LD_PRELOAD"] = preload
    env["LATTISENSE_GPU_KEY_CACHE"] = os.environ.get(
        "CRYPTOFACE_GPU_KEY_CACHE", "1"
    )
    env["LATTISENSE_GPU_KEY_CACHE_STATS"] = env.get(
        "LATTISENSE_GPU_KEY_CACHE_STATS", "1"
    )
    env["LATTISENSE_GPU_HOIST_ROTATIONS"] = os.environ.get(
        "CRYPTOFACE_GPU_HOIST_ROTATIONS", "0"
    )
    env["LATTISENSE_GPU_PIN_REQUEST_H2D"] = "1"
    env["LATTISENSE_GPU_EVENT_CACHE"] = os.environ.get(
        "CRYPTOFACE_GPU_EVENT_CACHE", "0"
    )
    env["LATTISENSE_GPU_BOOTSTRAP_SUBMISSION_LOCK"] = "1"
    env.pop("LATTISENSE_GPU_OPERATOR_SUBMISSION_LOCK", None)
    env.pop("LATTISENSE_GPU_ABI_CACHE_MAX_BYTES", None)
    env.pop("LATTISENSE_GPU_ABI_CACHE_STATS", None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _discover_gpu_ids(cfg: dict) -> list[int]:
    minimum_free_mib = int(cfg["minimum_gpu_free_memory_mib"])
    maximum_utilization = int(cfg.get("maximum_gpu_utilization_percent", 100))
    required_compute_capability = str(cfg["required_gpu_compute_capability"])
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,compute_cap,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Unable to discover compatible GPUs with nvidia-smi; set "
            "CRYPTOFACE_GPUS explicitly only after verifying the hardware"
        ) from exc

    discovered: list[dict[str, int | str]] = []
    eligible: list[int] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi output: {raw_line!r}")
        index = int(fields[0])
        free_mib = int(fields[1])
        compute_capability = fields[2]
        utilization = int(fields[3])
        discovered.append(
            {
                "index": index,
                "free_mib": free_mib,
                "compute_capability": compute_capability,
                "utilization_percent": utilization,
            }
        )
        if (
            free_mib >= minimum_free_mib
            and compute_capability == required_compute_capability
            and utilization <= maximum_utilization
        ):
            eligible.append(index)

    if not eligible:
        raise RuntimeError(
            "No compatible GPU has enough free memory: "
            f"required compute capability {required_compute_capability} and "
            f"at least {minimum_free_mib} MiB free with utilization at most "
            f"{maximum_utilization}%; discovered {discovered}"
        )
    print(
        "[gpu_runtime] Auto-selected GPUs "
        f"{eligible} (compute capability {required_compute_capability}, "
        f"minimum free memory {minimum_free_mib} MiB, maximum utilization "
        f"{maximum_utilization}%)",
        flush=True,
    )
    return eligible


def gpu_ids(cfg: dict) -> list[int]:
    configured = os.environ.get("CRYPTOFACE_GPUS")
    if configured:
        result = [int(value.strip()) for value in configured.split(",")]
        if (
            not result
            or len(set(result)) != len(result)
            or any(value < 0 for value in result)
        ):
            raise ValueError(f"Invalid GPU list: {result}")
        print(
            f"[gpu_runtime] Using CRYPTOFACE_GPUS override: {result}",
            flush=True,
        )
        return result
    return _discover_gpu_ids(cfg)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def server_task_dir(params, cfg: dict) -> Path:
    """Validate and return the stage-3 server-owned task directory."""
    reference_path = params.iodir() / "server_model.json"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing server model reference: {reference_path}; run stage 3"
        )
    reference = json.loads(reference_path.read_text())
    task_dir = Path(reference["task_dir"]).resolve()
    server_data_root = params.get_server_data_dir().resolve()
    if task_dir != server_data_root and server_data_root not in task_dir.parents:
        raise ValueError(f"Server task directory escapes io/server_data: {task_dir}")
    expected = {
        task_dir / "server" / "model_parameters.h5": cfg["model_parameters_sha256"],
        task_dir / "server" / "mega_ag.json": cfg["mega_ag_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Server-owned packed model failed validation: {path}")
    return task_dir


class NativeSession:
    """Line-delimited JSON session with one native client or GPU server."""

    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        log_path: Path,
        ready_marker: str,
    ) -> None:
        self.command = command
        self.env = env
        self.log_path = log_path
        self.ready_marker = ready_marker
        self.process: subprocess.Popen[str] | None = None
        self.log_stream = None
        self.lock = threading.Lock()
        self.startup_seconds: float | None = None
        self.request_count = 0

    def _exit_detail(self, wait_seconds: float = 1.0) -> str:
        if self.process is None:
            return "process unavailable"
        returncode = self.process.poll()
        if returncode is None:
            try:
                returncode = self.process.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                returncode = self.process.poll()
        return describe_returncode(returncode)

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.log_stream.write("# command=" + json.dumps(self.command) + "\n")
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self.env,
        )
        self.log_stream.write(
            f"# pid={self.process.pid} "
            f"cuda_visible_devices={self.env.get('CUDA_VISIBLE_DEVICES', '')}\n"
        )
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "Native session exited during startup via "
                    f"{self._exit_detail()}; see {self.log_path}"
                )
            self.log_stream.write(line)
            if line.rstrip("\r\n") == self.ready_marker:
                self.startup_seconds = time.monotonic() - started
                return

    def request(self, payload: dict, timeout_seconds: float | None = None) -> dict:
        del timeout_seconds  # Native stages are additionally bounded by harness timeout.
        request_started = time.monotonic()
        with self.lock:
            lock_acquired = time.monotonic()
            if self.process is None or self.process.stdin is None or self.process.stdout is None:
                raise RuntimeError("Native session not running")
            self.request_count += 1
            request_count = self.request_count
            request_id = payload.get("id")
            assert self.log_stream is not None
            self.log_stream.write(
                f"# request_begin count={request_count} id={request_id!r}\n"
            )

            encode_started = time.monotonic()
            encoded_payload = json.dumps(payload, separators=(",", ":")) + "\n"
            encode_finished = time.monotonic()
            try:
                write_started = time.monotonic()
                self.process.stdin.write(encoded_payload)
                self.process.stdin.flush()
                write_finished = time.monotonic()
            except (BrokenPipeError, OSError) as error:
                detail = self._exit_detail()
                self.log_stream.write(
                    f"# request_write_failed count={request_count} "
                    f"id={request_id!r} detail={detail} error={error!r}\n"
                )
                raise RuntimeError(
                    f"Native session failed writing request {request_count} "
                    f"{request_id!r} via {detail}; see {self.log_path}"
                ) from error

            response_wait_started = time.monotonic()
            while True:
                line = self.process.stdout.readline()
                if line == "":
                    detail = self._exit_detail()
                    self.log_stream.write(
                        f"# request_eof count={request_count} id={request_id!r} "
                        f"detail={detail}\n"
                    )
                    raise RuntimeError(
                        f"Native session exited during request {request_count} "
                        f"{request_id!r} via {detail}; see {self.log_path}"
                    )
                self.log_stream.write(line)
                if not line.startswith(RESULT_PREFIX):
                    continue

                response_received = time.monotonic()
                parse_started = time.monotonic()
                response = json.loads(line[len(RESULT_PREFIX) :])
                parse_finished = time.monotonic()
                if response.get("id") != payload.get("id"):
                    raise RuntimeError(
                        f"Native response id {response.get('id')!r} does not match "
                        f"{payload.get('id')!r}"
                    )
                if not response.get("ok", False):
                    raise RuntimeError(response.get("error", "native request failed"))

                request_finished = time.monotonic()
                response["python_transport_timing_seconds"] = {
                    "session_lock_wait": lock_acquired - request_started,
                    "request_json_encode": encode_finished - encode_started,
                    "request_pipe_write": write_finished - write_started,
                    "native_response_wait": response_received - response_wait_started,
                    "response_json_parse": parse_finished - parse_started,
                    "round_trip_total": request_finished - request_started,
                }
                self.log_stream.write(
                    f"# request_end count={request_count} id={request_id!r} "
                    f"elapsed_seconds={request_finished - request_started:.6f}\n"
                )
                return response
    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps({"op": "quit", "id": "quit"}) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None

    def __enter__(self) -> "NativeSession":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def client_command(secret_context: Path) -> list[str]:
    return [
        str(NATIVE_BINARY), "client",
        "--task-dir", str(MODEL_DIR),
        "--secret-context", str(secret_context),
    ]


def server_command(eval_context: Path, task_dir: Path) -> list[str]:
    return [
        str(NATIVE_BINARY), "server",
        "--task-dir", str(task_dir),
        "--eval-context", str(eval_context),
        "--gpu", "--gpu-device", "0",
    ]


def scorer_command(eval_context: Path, task_dir: Path) -> list[str]:
    return [
        str(NATIVE_BINARY), "scorer",
        "--task-dir", str(task_dir),
        "--eval-context", str(eval_context),
    ]
