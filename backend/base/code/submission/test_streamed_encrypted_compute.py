import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import server_encrypted_compute


class _Params:
    def __init__(self, root: Path, pair_count: int):
        self.root = root
        self._iodir = root / "io"
        self._pair_count = pair_count

    def iodir(self):
        return self._iodir

    def get_batch_size(self):
        return self._pair_count


class _FakeNativeSession:
    inference_finished = []
    score_started = []
    event_lock = threading.Lock()

    def __init__(self, command, *, env, log_path, ready_marker):
        del env
        self.is_scorer = command[1] == "scorer"
        self.log_path = log_path
        self.ready_marker = ready_marker
        self.startup_seconds = None
        self.request_count = 0

    def start(self):
        self.startup_seconds = 0.001

    def request(self, payload):
        self.request_count += 1
        if self.is_scorer:
            self.assert_score_payload(payload)
            with self.event_lock:
                self.score_started.append(time.monotonic())
            Path(payload["output"]).write_bytes(b"encrypted-score")
            time.sleep(0.005)
            return {
                "id": payload["id"],
                "ok": True,
                "matching_seconds": 0.005,
                "native_timing_seconds": {
                    "ciphertext_input_read": 0.001,
                    "fhe_inner_product": 0.003,
                    "ciphertext_output_write": 0.001,
                    "request_total": 0.005,
                },
                "python_transport_timing_seconds": {
                    "native_response_wait": 0.005,
                    "round_trip_total": 0.0051,
                },
            }

        delay = 0.08 if payload["id"].startswith("p000000000001") else 0.01
        time.sleep(delay)
        Path(payload["output"]).write_bytes(b"encrypted-embedding")
        with self.event_lock:
            self.inference_finished.append(time.monotonic())
        return {
            "id": payload["id"],
            "ok": True,
            "inference_seconds": delay,
            "native_timing_seconds": {
                "ciphertext_input_read": 0.001,
                "fhe_evaluate": delay - 0.002,
                "ciphertext_output_write": 0.001,
                "request_total": delay,
            },
            "python_transport_timing_seconds": {
                "native_response_wait": delay,
                "round_trip_total": delay + 0.0001,
            },
        }

    @staticmethod
    def assert_score_payload(payload):
        if payload["op"] != "score":
            raise AssertionError(f"scorer received {payload['op']!r}")
        if not Path(payload["left"]).is_file() or not Path(payload["right"]).is_file():
            raise AssertionError("score scheduled before both encrypted embeddings existed")

    def close(self):
        return None


class StreamedEncryptedComputeTest(unittest.TestCase):
    def setUp(self):
        _FakeNativeSession.inference_finished = []
        _FakeNativeSession.score_started = []

    def test_parses_per_request_latti_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "server.log"
            log_path.write_text(
                "# request_begin count=1 id='image-0'\n"
                "[Timing] server deserialize inputs: 0.12 s\n"
                "[Timing] server run_task total: 10.75 s\n"
                "[Timing] server serialize outputs: 0.08 s\n"
                "# request_end count=1 id='image-0' elapsed_seconds=10.95\n"
            )

            parsed = server_encrypted_compute._parse_latti_request_timings(log_path)

        self.assertEqual(
            parsed["image-0"],
            {
                "input_deserialization": 0.12,
                "run_task": 10.75,
                "output_serialization": 0.08,
            },
        )

    def _run_compute(self, environment: dict[str, str]):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = _Params(root, pair_count=2)
            eval_context = params.iodir() / "public_keys" / "eval_context.bin"
            eval_context.parent.mkdir(parents=True)
            eval_context.write_bytes(b"public-only-context")

            upload = params.iodir() / "ciphertexts_upload"
            upload.mkdir()
            for pair_index in range(2):
                for image_index in range(2):
                    image_id = f"p{pair_index:012d}_i{image_index}"
                    (upload / f"{image_id}__input.ct").write_bytes(b"ciphertext")

            cfg = {"input_names": ["input"], "embedding_dim": 256}
            task_dir = root / "server_data" / "task"

            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(
                    server_encrypted_compute,
                    "parse_stage_args",
                    return_value=(0, cfg, params),
                ),
                mock.patch.object(server_encrypted_compute, "require_runtime"),
                mock.patch.object(
                    server_encrypted_compute.os,
                    "sched_getaffinity",
                    return_value={10, 2, 6},
                    create=True,
                ),
                mock.patch.object(
                    server_encrypted_compute.os,
                    "sched_setaffinity",
                    create=True,
                ) as set_affinity,
                mock.patch.object(
                    server_encrypted_compute,
                    "server_task_dir",
                    return_value=task_dir,
                ),
                mock.patch.object(
                    server_encrypted_compute, "gpu_ids", return_value=[0, 1]
                ),
                mock.patch.object(
                    server_encrypted_compute, "runtime_environment", return_value={}
                ),
                mock.patch.object(
                    server_encrypted_compute,
                    "NativeSession",
                    _FakeNativeSession,
                ),
            ):
                server_encrypted_compute.main()
                set_affinity.assert_not_called()

            scores = sorted((params.iodir() / "ciphertexts_download").glob("*.ct"))
            report = json.loads((params.iodir() / "server_reported.json").read_text())
            return report["additional_measurements"], len(scores)

    def test_scores_ready_pair_before_later_inference_finishes(self):
        measurements, score_count = self._run_compute(
            {"CRYPTOFACE_STREAM_SCORING": "1"}
        )

        self.assertEqual(len(_FakeNativeSession.inference_finished), 4)
        self.assertEqual(measurements["workers"]["cpu_affinity"], [2, 6, 10])
        self.assertEqual(len(_FakeNativeSession.score_started), 2)
        self.assertLess(
            min(_FakeNativeSession.score_started),
            max(_FakeNativeSession.inference_finished),
        )
        self.assertEqual(score_count, 2)
        self.assertEqual(measurements["Scheduling mode"], "streamed_encrypted_score")
        self.assertGreater(measurements["Encrypted inference/matching overlap"], 0)
        native = measurements["native"]
        self.assertEqual(len(native["request_timings"]["inference"]), 4)
        self.assertEqual(len(native["request_timings"]["matching"]), 2)
        self.assertEqual(
            native["timing_summary"]["inference"]["native_timing_seconds"][
                "fhe_evaluate"
            ]["count"],
            4,
        )
        self.assertEqual(
            native["timing_summary"]["inference_hot"]["native_timing_seconds"][
                "fhe_evaluate"
            ]["count"],
            2,
        )
        self.assertEqual(
            native["timing_summary"]["matching"]["scorer_dispatch_timing_seconds"][
                "dispatch_total"
            ]["count"],
            2,
        )
        self.assertIn("stage_timing_seconds", measurements)
        self.assertIn(
            "unattributed_before_final_write", measurements["stage_timing_seconds"]
        )

    def test_sequential_mode_waits_for_all_inference(self):
        measurements, score_count = self._run_compute(
            {"CRYPTOFACE_STREAM_SCORING": "0"}
        )

        self.assertEqual(len(_FakeNativeSession.inference_finished), 4)
        self.assertEqual(len(_FakeNativeSession.score_started), 2)
        self.assertGreaterEqual(
            min(_FakeNativeSession.score_started),
            max(_FakeNativeSession.inference_finished),
        )
        self.assertEqual(score_count, 2)
        self.assertEqual(measurements["Scheduling mode"], "sequential_encrypted_score")
        self.assertEqual(measurements["Encrypted inference/matching overlap"], 0)

    def test_pair_limit_restricts_diagnostic_work(self):
        measurements, score_count = self._run_compute(
            {
                "CRYPTOFACE_PAIR_LIMIT": "1",
                "CRYPTOFACE_STREAM_SCORING": "0",
            }
        )

        self.assertEqual(len(_FakeNativeSession.inference_finished), 2)
        self.assertEqual(len(_FakeNativeSession.score_started), 1)
        self.assertEqual(score_count, 1)
        self.assertEqual(measurements["Pair count"], 1)


if __name__ == "__main__":
    unittest.main()
