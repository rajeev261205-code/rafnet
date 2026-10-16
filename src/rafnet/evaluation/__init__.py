from .metrics import (
    compute_eer,
    compute_roc_curve,
    compute_ece,
    compute_brier,
    compute_reliability_diagram,
    compute_risk_coverage,
    compute_coverage_error_curve,
    compute_calibration_metrics,
    compute_latency_throughput,
    compute_model_size,
    evaluate_all,
    bootstrap_confidence_intervals,
    EvaluationConfig,
)

# Lazy import for ablation to avoid circular dependencies
def __getattr__(name):
    if name in ("AblationModel", "AblationConfig", "run_ablation_study"):
        from .ablation import AblationModel, AblationConfig, run_ablation_study
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")