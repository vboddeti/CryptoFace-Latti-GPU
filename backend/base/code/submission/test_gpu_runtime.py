import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Some cluster Python installations enable safe-path mode, which omits the
# script directory from sys.path. Make the sibling adapter import explicit so
# this preflight behaves the same on login and compute nodes.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_runtime


CONFIG = {
    "minimum_gpu_free_memory_mib": 92160,
    "maximum_gpu_utilization_percent": 0,
    "required_gpu_compute_capability": "9.0",
}


class NativeExitDescriptionTest(unittest.TestCase):
    def test_reports_exit_code_and_signal(self):
        self.assertEqual(gpu_runtime.describe_returncode(None), "return code unavailable")
        self.assertEqual(gpu_runtime.describe_returncode(1), "exit code 1")
        self.assertEqual(gpu_runtime.describe_returncode(-11), "signal 11 (SIGSEGV)")


class NativeSessionTimingTest(unittest.TestCase):
    def test_records_structured_transport_timing(self):
        payload = {"op": "infer", "id": "image-0"}
        native_response = {
            "id": "image-0",
            "ok": True,
            "inference_seconds": 1.25,
        }
        stdin = io.StringIO()
        stdout = io.StringIO(
            gpu_runtime.RESULT_PREFIX + json.dumps(native_response) + "\n"
        )
        session = gpu_runtime.NativeSession(
            ["fake-native"],
            env={},
            log_path=Path("unused.log"),
            ready_marker="ready",
        )
        session.process = SimpleNamespace(stdin=stdin, stdout=stdout)
        session.log_stream = io.StringIO()

        response = session.request(payload)

        self.assertEqual(json.loads(stdin.getvalue()), payload)
        timings = response["python_transport_timing_seconds"]
        self.assertEqual(
            set(timings),
            {
                "session_lock_wait",
                "request_json_encode",
                "request_pipe_write",
                "native_response_wait",
                "response_json_parse",
                "round_trip_total",
            },
        )
        self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertGreaterEqual(
            timings["round_trip_total"], timings["native_response_wait"]
        )


class GpuDiscoveryTest(unittest.TestCase):
    @mock.patch.dict("os.environ", {}, clear=True)
    @mock.patch("gpu_runtime.subprocess.run")
    def test_selects_every_compatible_gpu_with_enough_free_memory(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "0, 91000, 9.0, 0\n"
                "1, 97000, 9.0, 0\n"
                "2, 96000, 9.0, 0\n"
                "3, 100000, 9.0, 20\n"
                "4, 48000, 8.6, 0\n"
            ),
        )

        self.assertEqual(gpu_runtime.gpu_ids(CONFIG), [1, 2])

    @mock.patch.dict("os.environ", {"CRYPTOFACE_GPUS": "3, 1"}, clear=True)
    def test_explicit_override_is_preserved(self):
        self.assertEqual(gpu_runtime.gpu_ids(CONFIG), [3, 1])

    @mock.patch.dict("os.environ", {}, clear=True)
    @mock.patch("gpu_runtime.subprocess.run")
    def test_fails_when_no_gpu_is_eligible(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, 50000, 9.0, 0\n1, 97000, 8.6, 0\n",
        )

        with self.assertRaisesRegex(RuntimeError, "No compatible GPU"):
            gpu_runtime.gpu_ids(CONFIG)


class RuntimeEnvironmentTest(unittest.TestCase):
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_disables_event_cache_by_default(self):
        environment = gpu_runtime.runtime_environment(2)
        self.assertEqual(environment["LATTISENSE_GPU_KEY_CACHE"], "1")
        self.assertEqual(environment["LATTISENSE_GPU_EVENT_CACHE"], "0")

    @mock.patch.dict(
        "os.environ",
        {
            "CRYPTOFACE_GPU_KEY_CACHE": "0",
            "CRYPTOFACE_GPU_EVENT_CACHE": "0",
        },
        clear=False,
    )
    def test_supports_cache_ab_diagnostics(self):
        environment = gpu_runtime.runtime_environment(2)
        self.assertEqual(environment["LATTISENSE_GPU_KEY_CACHE"], "0")
        self.assertEqual(environment["LATTISENSE_GPU_EVENT_CACHE"], "0")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")


class ServerModelReferenceTest(unittest.TestCase):
    def test_validates_task_inside_server_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            iodir = root / "single"
            task_dir = root / "server_data" / "latti-test"
            server_dir = task_dir / "server"
            iodir.mkdir()
            server_dir.mkdir(parents=True)
            model = server_dir / "model_parameters.h5"
            graph = server_dir / "mega_ag.json"
            model.write_bytes(b"model")
            graph.write_bytes(b"graph")
            (iodir / "server_model.json").write_text(
                json.dumps({"task_dir": str(task_dir)})
            )

            class Params:
                def iodir(self):
                    return iodir

                def get_server_data_dir(self):
                    return root / "server_data"

            cfg = {
                "model_parameters_sha256": gpu_runtime.sha256_file(model),
                "mega_ag_sha256": gpu_runtime.sha256_file(graph),
            }
            self.assertEqual(gpu_runtime.server_task_dir(Params(), cfg), task_dir)


class ScorerCommandTest(unittest.TestCase):
    def test_uses_public_context_without_gpu_mode(self):
        command = gpu_runtime.scorer_command(Path("eval.bin"), Path("task"))
        self.assertEqual(command[1], "scorer")
        self.assertIn("--eval-context", command)
        self.assertNotIn("--gpu", command)


if __name__ == "__main__":
    unittest.main()
