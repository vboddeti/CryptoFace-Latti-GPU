#!/usr/bin/env python3
"""Stage 7: run secret-free native H200 inference and encrypted matching."""

from __future__ import annotations

import json
import os
import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import pair_stem, parse_stage_args, release_file_cache
from gpu_runtime import (
    NativeSession,
    atomic_write_json,
    gpu_ids,
    require_runtime,
    runtime_environment,
    scorer_command,
    server_command,
    server_task_dir,
)


def _current_cpu_affinity() -> list[int] | None:
    """Report the scheduler-provided CPU set without narrowing it."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(os.sched_getaffinity(0))


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _pair_count(default: int) -> int:
    raw_limit = os.environ.get("CRYPTOFACE_PAIR_LIMIT")
    if raw_limit is None:
        return default
    limit = int(raw_limit)
    if limit <= 0 or limit > default:
        raise ValueError(
            f"CRYPTOFACE_PAIR_LIMIT must be between 1 and {default}, got {limit}"
        )
    return limit


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing_summary(responses: list[dict]) -> dict:
    samples: dict[str, dict[str, list[float]]] = {}
    for response in responses:
        for group_name, timings in response.items():
            if not group_name.endswith("_timing_seconds") or not isinstance(timings, dict):
                continue
            group = samples.setdefault(group_name, {})
            for field, value in timings.items():
                if isinstance(value, (int, float)):
                    group.setdefault(field, []).append(float(value))

    return {
        group_name: {
            field: {
                "count": len(values),
                "min": min(values),
                "mean": sum(values) / len(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": max(values),
                "total": sum(values),
            }
            for field, values in sorted(fields.items())
        }
        for group_name, fields in sorted(samples.items())
    }


_REQUEST_BEGIN_PATTERN = re.compile(r"^# request_begin .* id='([^']+)'$")
_LATTI_TIMING_PATTERN = re.compile(
    r"^\[Timing\] server (deserialize inputs|run_task total|serialize outputs): "
    r"([0-9]+(?:\.[0-9]+)?) s$"
)
_LATTI_TIMING_FIELDS = {
    "deserialize inputs": "input_deserialization",
    "run_task total": "run_task",
    "serialize outputs": "output_serialization",
}


def _parse_latti_request_timings(log_path: Path) -> dict[str, dict[str, float]]:
    if not log_path.is_file():
        return {}
    parsed: dict[str, dict[str, float]] = {}
    request_id: str | None = None
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            request_match = _REQUEST_BEGIN_PATTERN.match(line)
            if request_match:
                request_id = request_match.group(1)
                parsed.setdefault(request_id, {})
                continue
            timing_match = _LATTI_TIMING_PATTERN.match(line)
            if request_id is not None and timing_match:
                parsed[request_id][_LATTI_TIMING_FIELDS[timing_match.group(1)]] = (
                    float(timing_match.group(2))
                )
    return parsed


def main() -> None:
    stage_started = time.monotonic()
    _size, cfg, params = parse_stage_args(resolve_checkpoint=False)
    require_runtime(cfg)
    runtime_validation_finished = time.monotonic()
    inherited_cpu_affinity = _current_cpu_affinity()
    pair_count = _pair_count(params.get_batch_size())
    stream_scoring = _env_enabled("CRYPTOFACE_STREAM_SCORING", True)
    keep_stage_inputs = _env_enabled("CRYPTOFACE_KEEP_STAGE_INPUTS", False)
    input_names = cfg["input_names"]
    upload_dir = params.iodir() / "ciphertexts_upload"
    download_dir = params.iodir() / "ciphertexts_download"
    work_dir = params.iodir() / "server_work"
    log_dir = params.iodir() / "logs"
    download_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    eval_context = params.iodir() / "public_keys" / "eval_context.bin"
    if not eval_context.is_file():
        raise FileNotFoundError(f"Missing public evaluation context: {eval_context}")
    if (params.iodir() / "client_secret").exists():
        # The harness normally hides this directory from stage 7. Refuse to use it
        # even when invoking the stage manually from a shared development tree.
        print("[server_encrypted_compute] Client-secret directory is present but ignored", flush=True)

    task_dir = server_task_dir(params, cfg)
    physical_gpus = gpu_ids(cfg)
    sessions = [
        NativeSession(
            server_command(eval_context, task_dir),
            env=runtime_environment(gpu),
            log_path=log_dir / f"server_gpu_{gpu}.log",
            ready_marker="[StageServer] Ready.",
        )
        for gpu in physical_gpus
    ]
    scorer_env = runtime_environment()
    scorer_env["CUDA_VISIBLE_DEVICES"] = ""
    scorer = NativeSession(
        scorer_command(eval_context, task_dir),
        env=scorer_env,
        log_path=log_dir / "server_scorer.log",
        ready_marker="[StageScorer] Ready.",
    )
    preparation_finished = time.monotonic()
    print(
        f"[server_encrypted_compute] Starting {len(sessions)} persistent GPU workers "
        f"on physical GPUs {physical_gpus} and one public-context CPU scorer",
        flush=True,
    )
    setup_started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=len(sessions) + 1) as executor:
            starts = [executor.submit(session.start) for session in sessions]
            starts.append(executor.submit(scorer.start))
            for future in as_completed(starts):
                future.result()
        setup_seconds = time.monotonic() - setup_started
        pool: queue.Queue[NativeSession] = queue.Queue()
        for session in sessions:
            pool.put(session)

        session_gpu = dict(zip(sessions, physical_gpus))

        def dispatch(payload: dict, submitted_at: float) -> dict:
            dispatch_started = time.monotonic()
            session = pool.get()
            worker_acquired = time.monotonic()
            try:
                response = session.request(payload)
                dispatch_finished = time.monotonic()
                response["physical_gpu"] = session_gpu[session]
                response["worker_request_index"] = getattr(
                    session, "request_count", None
                )
                response["inference_dispatch_timing_seconds"] = {
                    "executor_queue_wait": dispatch_started - submitted_at,
                    "worker_pool_wait": worker_acquired - dispatch_started,
                    "worker_request": dispatch_finished - worker_acquired,
                    "dispatch_total": dispatch_finished - submitted_at,
                }
                return response
            finally:
                pool.put(session)

        def score_dispatch(payload: dict, submitted_at: float) -> dict:
            dispatch_started = time.monotonic()
            response = scorer.request(payload)
            dispatch_finished = time.monotonic()
            response["scorer_dispatch_timing_seconds"] = {
                "executor_queue_wait": dispatch_started - submitted_at,
                "scorer_request": dispatch_finished - dispatch_started,
                "dispatch_total": dispatch_finished - submitted_at,
            }
            return response

        compute_started = time.monotonic()
        inference_started = compute_started
        inference_responses = []
        match_responses = []
        ready_embeddings = [0] * pair_count
        matching_started: float | None = None
        match_futures = []
        with (
            ThreadPoolExecutor(max_workers=len(sessions)) as inference_executor,
            ThreadPoolExecutor(max_workers=1) as score_executor,
        ):
            submission_started = time.monotonic()
            future_to_pair = {}
            for pair_index in range(pair_count):
                for image_index in range(2):
                    image_id = f"{pair_stem(pair_index)}_i{image_index}"
                    inputs = {
                        name: str(upload_dir / f"{image_id}__{name}.ct")
                        for name in input_names
                    }
                    missing = [path for path in inputs.values() if not Path(path).is_file()]
                    if missing:
                        raise FileNotFoundError(f"Missing encrypted input(s): {missing[:2]}")
                    submitted_at = time.monotonic()
                    future = inference_executor.submit(
                        dispatch,
                        {
                            "op": "infer",
                            "id": image_id,
                            "inputs": inputs,
                            "output": str(work_dir / f"{image_id}_embedding.ct"),
                        },
                        submitted_at,
                    )
                    future_to_pair[future] = pair_index
            submission_finished = time.monotonic()
            completed = 0
            for future in as_completed(future_to_pair):
                inference_responses.append(future.result())
                completed += 1
                pair_index = future_to_pair[future]
                ready_embeddings[pair_index] += 1
                if stream_scoring and ready_embeddings[pair_index] == 2:
                    if matching_started is None:
                        matching_started = time.monotonic()
                    stem = pair_stem(pair_index)
                    submitted_at = time.monotonic()
                    match_futures.append(
                        score_executor.submit(
                            score_dispatch,
                            {
                                "op": "score",
                                "id": stem,
                                "left": str(work_dir / f"{stem}_i0_embedding.ct"),
                                "right": str(work_dir / f"{stem}_i1_embedding.ct"),
                                "output": str(download_dir / f"{stem}_score.ct"),
                                "embedding_dim": int(cfg["embedding_dim"]),
                            },
                            submitted_at,
                        )
                    )
                print(
                    f"[server_encrypted_compute] embeddings {completed}/{2 * pair_count}",
                    flush=True,
                )
            inference_finished = time.monotonic()

            if not stream_scoring:
                matching_started = time.monotonic()
                for pair_index in range(pair_count):
                    stem = pair_stem(pair_index)
                    submitted_at = time.monotonic()
                    match_futures.append(
                        score_executor.submit(
                            score_dispatch,
                            {
                                "op": "score",
                                "id": stem,
                                "left": str(work_dir / f"{stem}_i0_embedding.ct"),
                                "right": str(work_dir / f"{stem}_i1_embedding.ct"),
                                "output": str(download_dir / f"{stem}_score.ct"),
                                "embedding_dim": int(cfg["embedding_dim"]),
                            },
                            submitted_at,
                        )
                    )

            if len(match_futures) != pair_count:
                raise RuntimeError(
                    f"Scheduled {len(match_futures)} encrypted scores; expected {pair_count}"
                )
            completed = 0
            for future in as_completed(match_futures):
                match_responses.append(future.result())
                completed += 1
                print(
                    f"[server_encrypted_compute] encrypted scores {completed}/{pair_count}",
                    flush=True,
                )
        compute_finished = time.monotonic()
        inference_seconds = inference_finished - inference_started
        matching_seconds = (
            0.0 if matching_started is None else compute_finished - matching_started
        )
        encrypted_compute_seconds = compute_finished - compute_started
        overlap_seconds = (
            0.0
            if matching_started is None
            else max(0.0, inference_finished - matching_started)
        )
    finally:
        shutdown_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(sessions) + 1) as executor:
            closes = [executor.submit(session.close) for session in sessions]
            closes.append(executor.submit(scorer.close))
            for future in as_completed(closes):
                future.result()
        shutdown_finished = time.monotonic()

    cleanup_started = time.monotonic()
    for path in upload_dir.glob("*.ct"):
        release_file_cache(path)
        if not keep_stage_inputs:
            path.unlink()
    for path in work_dir.glob("*.ct"):
        release_file_cache(path)
        if not keep_stage_inputs:
            path.unlink()
    cleanup_finished = time.monotonic()
    encrypted_output_bytes = sum(path.stat().st_size for path in download_dir.glob("*.ct"))
    report_path = params.iodir() / "submission_reported.json"
    bandwidth = json.loads(report_path.read_text()) if report_path.exists() else {
        "schema_version": 1,
        "Bandwidth": {},
    }
    bandwidth["Bandwidth"]["Encrypted outputs"] = encrypted_output_bytes
    atomic_write_json(report_path, bandwidth)
    accounting_finished = time.monotonic()

    report_build_started = time.monotonic()
    latti_timings_by_id: dict[str, dict[str, float]] = {}
    for session in sessions:
        latti_timings_by_id.update(_parse_latti_request_timings(session.log_path))
    for response in inference_responses:
        request_id = response.get("id")
        if request_id in latti_timings_by_id:
            response["latti_timing_seconds"] = latti_timings_by_id[request_id]
    all_responses = inference_responses + match_responses
    stage_seconds_before_report_write = report_build_started - stage_started
    report = {
        "Encrypted computation": encrypted_compute_seconds,
        "Total": stage_seconds_before_report_write,
        "additional_measurements": {
            "stage_timing_seconds": {
                "argument_parse_and_runtime_validation": (
                    runtime_validation_finished - stage_started
                ),
                "preparation": preparation_finished - runtime_validation_finished,
                "worker_setup": setup_seconds,
                "inference_task_submission": submission_finished - submission_started,
                "encrypted_compute": encrypted_compute_seconds,
                "worker_shutdown": shutdown_finished - shutdown_started,
                "stage_input_cleanup": cleanup_finished - cleanup_started,
                "output_accounting": accounting_finished - cleanup_finished,
            },
            "GPU worker setup": setup_seconds,
            "Encrypted embedding inference": inference_seconds,
            "Encrypted pair matching": matching_seconds,
            "Encrypted inference/matching overlap": overlap_seconds,
            "Scheduling mode": (
                "streamed_encrypted_score"
                if stream_scoring
                else "sequential_encrypted_score"
            ),
            "Pair count": pair_count,
            "GPU key cache": os.environ.get("CRYPTOFACE_GPU_KEY_CACHE", "1"),
            "GPU event cache": os.environ.get("CRYPTOFACE_GPU_EVENT_CACHE", "1"),
            "Scorer startup": scorer.startup_seconds,
            "Throughput": {
                "images_per_second_during_inference": (
                    2 * pair_count
                ) / inference_seconds,
                "pairs_per_second_end_to_end_compute": (
                    pair_count / encrypted_compute_seconds
                ),
            },
            "workers": {
                "gpu_count": len(physical_gpus),
                "physical_gpus": physical_gpus,
                "cpu_affinity": inherited_cpu_affinity,
                "startup_seconds": [
                    session.startup_seconds for session in sessions
                ],
                "scorer_startup_seconds": scorer.startup_seconds,
            },
            "native": {
                "inference_core_seconds": [
                    response.get("inference_seconds")
                    for response in inference_responses
                ],
                "matching_core_seconds": [
                    response.get("matching_seconds")
                    for response in match_responses
                ],
                "request_timings": {
                    "inference": sorted(
                        inference_responses, key=lambda response: response.get("id", "")
                    ),
                    "matching": sorted(
                        match_responses, key=lambda response: response.get("id", "")
                    ),
                },
                "timing_summary": {
                    "inference": _timing_summary(inference_responses),
                    "inference_hot": _timing_summary(
                        [
                            response
                            for response in inference_responses
                            if (response.get("worker_request_index") or 0) > 1
                        ]
                    ),
                    "matching": _timing_summary(match_responses),
                    "all_requests": _timing_summary(all_responses),
                },
            },
        },
    }
    report_build_finished = time.monotonic()
    report["additional_measurements"]["stage_timing_seconds"]["report_build"] = (
        report_build_finished - report_build_started
    )
    report_write_started = time.monotonic()
    atomic_write_json(params.iodir() / "server_reported.json", report)
    report_write_finished = time.monotonic()
    report["additional_measurements"]["stage_timing_seconds"]["report_write"] = (
        report_write_finished - report_write_started
    )
    stage_seconds = report_write_finished - stage_started
    report["Total"] = stage_seconds
    stage_timings = report["additional_measurements"]["stage_timing_seconds"]
    # Task submission is a nested part of encrypted_compute, so do not
    # double-count it in the stage-level accounting reconciliation.
    accounted_seconds = sum(
        value
        for name, value in stage_timings.items()
        if name != "inference_task_submission"
    )
    stage_timings["accounted_before_final_write"] = accounted_seconds
    stage_timings["unattributed_before_final_write"] = (
        stage_seconds - accounted_seconds
    )
    atomic_write_json(params.iodir() / "server_reported.json", report)
    print(
        f"[server_encrypted_compute] Completed {pair_count} pairs in {stage_seconds:.1f}s "
        f"({pair_count / encrypted_compute_seconds * 3600:.2f} pairs/hour)",
        flush=True,
    )


if __name__ == "__main__":
    main()
