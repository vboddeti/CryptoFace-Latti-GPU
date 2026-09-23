{
    case: $case_name,
    gpu_list: $gpu_list,
    pair_count: $pair_limit,
    encrypted_compute_seconds: .["Encrypted computation"],
    seconds_per_pair: (.["Encrypted computation"] / $pair_limit),
    gpu_worker_setup_seconds: .additional_measurements["GPU worker setup"],
    inference_core_seconds: .additional_measurements.native.inference_core_seconds,
    hot_inference_seconds: .additional_measurements.native.inference_core_seconds[$warm_count:],
    hot_inference_mean_seconds:
        ((.additional_measurements.native.inference_core_seconds[$warm_count:] | add) /
         (.additional_measurements.native.inference_core_seconds[$warm_count:] | length)),
    matching_core_seconds: .additional_measurements.native.matching_core_seconds,
    workers: .additional_measurements.workers,
    scheduling_mode: .additional_measurements["Scheduling mode"]
}
