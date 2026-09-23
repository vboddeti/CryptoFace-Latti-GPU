{
    cases: .,
    four_vs_one_before_hot_ratio:
        (.[1].hot_inference_mean_seconds / .[0].hot_inference_mean_seconds),
    four_vs_one_after_hot_ratio:
        (.[1].hot_inference_mean_seconds / .[2].hot_inference_mean_seconds),
    one_gpu_drift_ratio:
        (.[2].hot_inference_mean_seconds / .[0].hot_inference_mean_seconds)
}
