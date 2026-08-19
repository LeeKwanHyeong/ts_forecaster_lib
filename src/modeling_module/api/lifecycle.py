"""Public fit and forecast APIs for lifecycle-specific models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from modeling_module._internal.lifecycle_runtime import (
    CGMM_MODEL_KEY,
    CGMMConfig,
    CGMMCorrectionState,
    CGMMPrediction,
    CGMMPreprocessingConfig,
    ConditionalGaussianMixtureForecaster,
    LifecycleSample,
    SIMILAR_LIFECYCLE_MODEL_KEY,
    SimilarLifecycleConfig,
    SimilarLifecycleForecaster,
    SimilarLifecyclePrediction,
    default_similar_lifecycle_preprocessing,
    load_cgmm_artifact,
    load_similar_lifecycle_artifact,
    require_sha256,
)


@dataclass(frozen=True, slots=True)
class CGMMFitRequest:
    """Complete fitting request with no database or engine ownership."""

    samples: tuple[LifecycleSample, ...]
    dataset_fingerprint: str
    config: CGMMConfig = field(default_factory=CGMMConfig)
    preprocessing: CGMMPreprocessingConfig = field(
        default_factory=CGMMPreprocessingConfig
    )
    correction_state: CGMMCorrectionState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("samples must be a non-empty tuple")
        if any(not isinstance(sample, LifecycleSample) for sample in self.samples):
            raise TypeError("samples must contain LifecycleSample values")
        require_sha256(
            self.dataset_fingerprint,
            field_name="dataset_fingerprint",
        )
        if not isinstance(self.config, CGMMConfig):
            raise TypeError("config must be CGMMConfig")
        if not isinstance(self.preprocessing, CGMMPreprocessingConfig):
            raise TypeError("preprocessing must be CGMMPreprocessingConfig")
        if self.correction_state is not None and not isinstance(
            self.correction_state,
            CGMMCorrectionState,
        ):
            raise TypeError("correction_state must be CGMMCorrectionState")


@dataclass(frozen=True, slots=True)
class CGMMFitResult:
    """Fitted model and stable provenance returned by the public API."""

    model: ConditionalGaussianMixtureForecaster
    model_key: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    dataset_fingerprint: str
    correction_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class CGMMForecastRequest:
    """Forecast request using either an in-memory model or artifact directory."""

    model: ConditionalGaussianMixtureForecaster | str | Path
    samples: tuple[LifecycleSample, ...]
    apply_correction: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.model,
            (ConditionalGaussianMixtureForecaster, str, Path),
        ):
            raise TypeError("model must be a fitted CGMM or artifact path")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("samples must be a non-empty tuple")
        if any(not isinstance(sample, LifecycleSample) for sample in self.samples):
            raise TypeError("samples must contain LifecycleSample values")
        if not isinstance(self.apply_correction, bool):
            raise TypeError("apply_correction must be bool")


def fit_cgmm(request: CGMMFitRequest) -> CGMMFitResult:
    """Fit one CGMM from completed lifecycle samples."""

    if not isinstance(request, CGMMFitRequest):
        raise TypeError("request must be CGMMFitRequest")
    model = ConditionalGaussianMixtureForecaster(
        request.config,
        preprocessing_config=request.preprocessing,
    ).fit(
        request.samples,
        dataset_fingerprint=request.dataset_fingerprint,
    )
    model.attach_correction(request.correction_state)
    return CGMMFitResult(
        model=model,
        model_key=CGMM_MODEL_KEY,
        model_fingerprint=model.model_fingerprint,
        preprocessing_fingerprint=model.preprocessing_state.fingerprint,
        dataset_fingerprint=request.dataset_fingerprint,
        correction_fingerprint=(
            None
            if model.correction_state is None
            else model.correction_state.fingerprint
        ),
    )


def forecast_cgmm(request: CGMMForecastRequest) -> CGMMPrediction:
    """Return a conditional lifecycle distribution without side effects."""

    if not isinstance(request, CGMMForecastRequest):
        raise TypeError("request must be CGMMForecastRequest")
    model = (
        request.model
        if isinstance(request.model, ConditionalGaussianMixtureForecaster)
        else load_cgmm_artifact(request.model)
    )
    return model.predict(
        request.samples,
        apply_correction=request.apply_correction,
    )


@dataclass(frozen=True, slots=True)
class SimilarLifecycleFitRequest:
    """Fit request for a completed lifecycle retrieval repository."""

    samples: tuple[LifecycleSample, ...]
    dataset_fingerprint: str
    config: SimilarLifecycleConfig = field(
        default_factory=SimilarLifecycleConfig
    )
    preprocessing: CGMMPreprocessingConfig = field(
        default_factory=default_similar_lifecycle_preprocessing
    )
    correction_state: CGMMCorrectionState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("samples must be a non-empty tuple")
        if any(not isinstance(sample, LifecycleSample) for sample in self.samples):
            raise TypeError("samples must contain LifecycleSample values")
        require_sha256(
            self.dataset_fingerprint,
            field_name="dataset_fingerprint",
        )
        if not isinstance(self.config, SimilarLifecycleConfig):
            raise TypeError("config must be SimilarLifecycleConfig")
        if not isinstance(self.preprocessing, CGMMPreprocessingConfig):
            raise TypeError(
                "preprocessing must be SimilarLifecyclePreprocessingConfig"
            )
        if self.preprocessing.feature_profile != "static_observed_v1":
            raise ValueError(
                "Similar Lifecycle preprocessing must use static_observed_v1"
            )
        if self.correction_state is not None and not isinstance(
            self.correction_state,
            CGMMCorrectionState,
        ):
            raise TypeError(
                "correction_state must be SimilarLifecycleCorrectionState"
            )


@dataclass(frozen=True, slots=True)
class SimilarLifecycleFitResult:
    """Fitted retrieval model and stable provenance."""

    model: SimilarLifecycleForecaster
    model_key: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    dataset_fingerprint: str
    correction_fingerprint: str | None
    distance_feature_names: tuple[str, ...]
    training_sample_count: int


@dataclass(frozen=True, slots=True)
class SimilarLifecycleForecastRequest:
    """Forecast using an in-memory retrieval model or artifact directory."""

    model: SimilarLifecycleForecaster | str | Path
    samples: tuple[LifecycleSample, ...]
    apply_correction: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.model,
            (SimilarLifecycleForecaster, str, Path),
        ):
            raise TypeError(
                "model must be a fitted Similar Lifecycle model or artifact path"
            )
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("samples must be a non-empty tuple")
        if any(not isinstance(sample, LifecycleSample) for sample in self.samples):
            raise TypeError("samples must contain LifecycleSample values")
        if not isinstance(self.apply_correction, bool):
            raise TypeError("apply_correction must be bool")


def fit_similar_lifecycle(
    request: SimilarLifecycleFitRequest,
) -> SimilarLifecycleFitResult:
    """Fit one retrieval repository from completed lifecycle samples."""

    if not isinstance(request, SimilarLifecycleFitRequest):
        raise TypeError("request must be SimilarLifecycleFitRequest")
    model = SimilarLifecycleForecaster(
        request.config,
        preprocessing_config=request.preprocessing,
    ).fit(
        request.samples,
        dataset_fingerprint=request.dataset_fingerprint,
    )
    model.attach_correction(request.correction_state)
    return SimilarLifecycleFitResult(
        model=model,
        model_key=SIMILAR_LIFECYCLE_MODEL_KEY,
        model_fingerprint=model.model_fingerprint,
        preprocessing_fingerprint=model.preprocessing_state.fingerprint,
        dataset_fingerprint=request.dataset_fingerprint,
        correction_fingerprint=(
            None
            if model.correction_state is None
            else model.correction_state.fingerprint
        ),
        distance_feature_names=model.distance_feature_names,
        training_sample_count=model.training_sample_count,
    )


def forecast_similar_lifecycle(
    request: SimilarLifecycleForecastRequest,
) -> SimilarLifecyclePrediction:
    """Return a retrieval forecast and its neighbor evidence."""

    if not isinstance(request, SimilarLifecycleForecastRequest):
        raise TypeError("request must be SimilarLifecycleForecastRequest")
    model = (
        request.model
        if isinstance(request.model, SimilarLifecycleForecaster)
        else load_similar_lifecycle_artifact(request.model)
    )
    return model.predict(
        request.samples,
        apply_correction=request.apply_correction,
    )


__all__ = [
    "CGMMFitRequest",
    "CGMMFitResult",
    "CGMMForecastRequest",
    "SimilarLifecycleFitRequest",
    "SimilarLifecycleFitResult",
    "SimilarLifecycleForecastRequest",
    "fit_cgmm",
    "fit_similar_lifecycle",
    "forecast_cgmm",
    "forecast_similar_lifecycle",
]
