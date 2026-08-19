"""Conditional Gaussian mixture lifecycle forecaster."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from modeling_module.data_loader.lifecycle_contracts import (
    LTB_FORECAST_MONTHS,
    LifecycleSample,
)
from modeling_module.models.CGMM.configs import (
    CGMMConfig,
    CGMMPreprocessingConfig,
)
from modeling_module.models.CGMM.contracts import (
    CGMM_MODEL_ID,
    CGMM_MODEL_KEY,
    CGMMContractError,
    CGMMCorrectionState,
    CGMMPrediction,
    CGMMPreprocessingState,
    freeze_float_array,
)
from modeling_module.models.CGMM.preprocessing import CGMMPreprocessor


_LOG_2PI = float(np.log(2.0 * np.pi))


class CGMMModelError(CGMMContractError):
    """Raised when a CGMM cannot be fitted, restored, or evaluated."""


@dataclass(frozen=True, slots=True)
class _PCAState:
    components: np.ndarray
    mean: np.ndarray
    noise_variance: float

    def __post_init__(self) -> None:
        components = np.asarray(self.components, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        if components.ndim != 2 or mean.shape != (components.shape[1],):
            raise CGMMModelError("PCA state has incompatible dimensions")
        if components.shape[0] == 0 or components.shape[1] != LTB_FORECAST_MONTHS:
            raise CGMMModelError("PCA state has an invalid target shape")
        object.__setattr__(
            self,
            "components",
            freeze_float_array(
                components,
                field_name="pca_components",
                shape=components.shape,
            ),
        )
        object.__setattr__(
            self,
            "mean",
            freeze_float_array(
                mean,
                field_name="pca_mean",
                shape=mean.shape,
            ),
        )
        if not np.isfinite(self.noise_variance) or self.noise_variance < 0.0:
            raise CGMMModelError("PCA noise variance must be non-negative")


@dataclass(frozen=True, slots=True)
class _MixtureState:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    converged: bool
    iteration_count: int

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        covariances = np.asarray(self.covariances, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise CGMMModelError("mixture weights must contain components")
        component_count = weights.size
        if means.ndim != 2 or means.shape[0] != component_count:
            raise CGMMModelError("mixture means have incompatible dimensions")
        if covariances.shape != (
            component_count,
            means.shape[1],
            means.shape[1],
        ):
            raise CGMMModelError("mixture covariances have incompatible dimensions")
        weights = freeze_float_array(
            weights,
            field_name="mixture_weights",
            shape=weights.shape,
            non_negative=True,
        )
        if not np.isclose(weights.sum(), 1.0, rtol=1e-8, atol=1e-10):
            raise CGMMModelError("mixture weights must sum to one")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "means",
            freeze_float_array(
                means,
                field_name="mixture_means",
                shape=means.shape,
            ),
        )
        object.__setattr__(
            self,
            "covariances",
            freeze_float_array(
                covariances,
                field_name="mixture_covariances",
                shape=covariances.shape,
            ),
        )
        if not isinstance(self.converged, bool) or not self.converged:
            raise CGMMModelError("restored mixture must be converged")
        if (
            isinstance(self.iteration_count, bool)
            or not isinstance(self.iteration_count, int)
            or self.iteration_count <= 0
        ):
            raise CGMMModelError("iteration_count must be positive")


class ConditionalGaussianMixtureForecaster:
    """Infer a 72-month distribution from a completed 12-month prefix."""

    model_key = CGMM_MODEL_KEY
    model_id = CGMM_MODEL_ID

    def __init__(
        self,
        config: CGMMConfig | Mapping[str, object] | None = None,
        *,
        preprocessing_config: CGMMPreprocessingConfig | None = None,
    ) -> None:
        self.config = (
            CGMMConfig()
            if config is None
            else CGMMConfig.from_config(config)
        )
        self.preprocessor = CGMMPreprocessor(preprocessing_config)
        self._pca_state: _PCAState | None = None
        self._mixture_state: _MixtureState | None = None
        self._model_fingerprint: str | None = None
        self._correction_state: CGMMCorrectionState | None = None

    @property
    def is_fitted(self) -> bool:
        return self._pca_state is not None and self._mixture_state is not None

    @property
    def preprocessing_state(self) -> CGMMPreprocessingState:
        return self.preprocessor.state

    @property
    def model_fingerprint(self) -> str:
        if self._model_fingerprint is None:
            raise CGMMModelError("model has not been fitted")
        return self._model_fingerprint

    @property
    def correction_state(self) -> CGMMCorrectionState | None:
        return self._correction_state

    @property
    def converged(self) -> bool:
        return self._require_mixture_state().converged

    @property
    def iteration_count(self) -> int:
        return self._require_mixture_state().iteration_count

    @property
    def target_component_count(self) -> int:
        return int(self._require_pca_state().components.shape[0])

    def fit(
        self,
        samples: Iterable[LifecycleSample],
        *,
        dataset_fingerprint: str,
    ) -> "ConditionalGaussianMixtureForecaster":
        if self.is_fitted or self.preprocessor.is_fitted:
            raise CGMMModelError(
                "model is already fitted; create a new instance to refit"
            )
        materialized = tuple(samples)
        prepared = self.preprocessor.fit_transform(
            materialized,
            dataset_fingerprint=dataset_fingerprint,
        )
        if prepared.normalized_future is None:
            raise CGMMModelError("CGMM fitting requires future targets")
        sample_count = prepared.condition_matrix.shape[0]
        if sample_count < self.config.component_count:
            raise CGMMModelError(
                "training sample count must be at least component_count"
            )
        target_component_count = min(
            self.config.target_component_count,
            sample_count,
            LTB_FORECAST_MONTHS,
        )
        pca = PCA(n_components=target_component_count, svd_solver="full")
        target_latent = pca.fit_transform(prepared.normalized_future)
        joint = np.concatenate((prepared.condition_matrix, target_latent), axis=1)
        mixture = GaussianMixture(
            n_components=self.config.component_count,
            covariance_type="full",
            reg_covar=self.config.covariance_regularization,
            n_init=self.config.initialization_count,
            max_iter=self.config.max_iterations,
            random_state=self.config.random_seed,
        ).fit(joint)
        if not mixture.converged_:
            raise CGMMModelError(
                "Gaussian mixture did not converge within max_iterations"
            )
        self._pca_state = _PCAState(
            components=pca.components_,
            mean=pca.mean_,
            noise_variance=max(float(pca.noise_variance_), 0.0),
        )
        self._mixture_state = _MixtureState(
            weights=mixture.weights_,
            means=mixture.means_,
            covariances=mixture.covariances_,
            converged=bool(mixture.converged_),
            iteration_count=int(mixture.n_iter_),
        )
        self._model_fingerprint = self._build_model_fingerprint()
        return self

    def attach_correction(
        self,
        state: CGMMCorrectionState | None,
    ) -> "ConditionalGaussianMixtureForecaster":
        if not self.is_fitted:
            raise CGMMModelError("model must be fitted before correction is attached")
        if state is not None and not isinstance(state, CGMMCorrectionState):
            raise TypeError("state must be CGMMCorrectionState or None")
        self._correction_state = state
        return self

    def predict(
        self,
        samples: Iterable[LifecycleSample],
        *,
        apply_correction: bool = True,
    ) -> CGMMPrediction:
        materialized = tuple(samples)
        prediction = self._predict_raw(materialized)
        if not apply_correction or self._correction_state is None:
            return prediction
        from modeling_module.models.CGMM.correction import apply_cgmm_correction

        return apply_cgmm_correction(
            prediction,
            materialized,
            self._correction_state,
        )

    def _predict_raw(
        self,
        samples: tuple[LifecycleSample, ...],
    ) -> CGMMPrediction:
        pca = self._require_pca_state()
        mixture = self._require_mixture_state()
        prepared = self.preprocessor.transform(samples)
        probabilities, conditional_means, conditional_covariances = (
            self._conditional_target_distribution(prepared.condition_matrix)
        )
        sample_count = len(samples)
        component_count = self.config.component_count
        candidate_curves = np.empty(
            (sample_count, component_count, LTB_FORECAST_MONTHS),
            dtype=np.float64,
        )
        component_variances = np.empty_like(candidate_curves)
        components = pca.components
        for component in range(component_count):
            log_mean = (
                conditional_means[:, component, :] @ components + pca.mean
            )
            latent_covariance = conditional_covariances[component]
            log_variance = np.einsum(
                "ph,pq,qh->h",
                components,
                latent_covariance,
                components,
                optimize=True,
            )
            log_variance = np.clip(
                log_variance + pca.noise_variance,
                0.0,
                20.0,
            )
            expected_exponent = np.clip(
                log_mean + 0.5 * log_variance[None, :],
                -50.0,
                50.0,
            )
            expected_normalized = np.maximum(
                np.expm1(expected_exponent),
                0.0,
            )
            variance_exponent = np.clip(
                2.0 * log_mean + log_variance[None, :],
                -50.0,
                50.0,
            )
            normalized_variance = np.maximum(
                np.expm1(log_variance)[None, :]
                * np.exp(variance_exponent),
                0.0,
            )
            candidate_curves[:, component, :] = (
                expected_normalized * prepared.quantity_scale[:, None]
            )
            component_variances[:, component, :] = (
                normalized_variance
                * np.square(prepared.quantity_scale[:, None])
            )

        mean_forecast = np.einsum(
            "nk,nkh->nh",
            probabilities,
            candidate_curves,
        )
        second_moment = np.einsum(
            "nk,nkh->nh",
            probabilities,
            component_variances + np.square(candidate_curves),
        )
        forecast_variance = np.maximum(
            second_moment - np.square(mean_forecast),
            0.0,
        )
        forecast_std = np.sqrt(forecast_variance)
        lower_bound = np.maximum(
            mean_forecast - self.config.interval_z * forecast_std,
            0.0,
        )
        upper_bound = mean_forecast + self.config.interval_z * forecast_std
        return CGMMPrediction(
            sample_ids=prepared.sample_ids,
            component_probabilities=probabilities,
            candidate_curves=candidate_curves,
            mean_forecast=mean_forecast,
            forecast_std=forecast_std,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_key=CGMM_MODEL_KEY,
            model_id=CGMM_MODEL_ID,
            model_fingerprint=self.model_fingerprint,
            preprocessing_fingerprint=self.preprocessing_state.fingerprint,
        )

    def _conditional_target_distribution(
        self,
        condition: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mixture = self._require_mixture_state()
        pca = self._require_pca_state()
        sample_count, condition_dim = condition.shape
        target_dim = pca.components.shape[0]
        component_count = mixture.weights.size
        if mixture.means.shape[1] != condition_dim + target_dim:
            raise CGMMModelError(
                "restored model condition width does not match preprocessing"
            )
        log_probabilities = np.empty(
            (sample_count, component_count),
            dtype=np.float64,
        )
        conditional_means = np.empty(
            (sample_count, component_count, target_dim),
            dtype=np.float64,
        )
        conditional_covariances = np.empty(
            (component_count, target_dim, target_dim),
            dtype=np.float64,
        )
        for component in range(component_count):
            mean = mixture.means[component]
            covariance = mixture.covariances[component]
            mean_x = mean[:condition_dim]
            mean_y = mean[condition_dim:]
            covariance_xx = covariance[:condition_dim, :condition_dim]
            covariance_xy = covariance[:condition_dim, condition_dim:]
            covariance_yy = covariance[condition_dim:, condition_dim:]
            try:
                cholesky = np.linalg.cholesky(covariance_xx)
            except np.linalg.LinAlgError:
                cholesky = np.linalg.cholesky(
                    covariance_xx
                    + np.eye(condition_dim)
                    * self.config.covariance_regularization
                )
            regression = np.linalg.solve(
                cholesky.T,
                np.linalg.solve(cholesky, covariance_xy),
            )
            delta = condition - mean_x
            conditional_means[:, component, :] = mean_y + delta @ regression
            conditional_covariance = covariance_yy - covariance_xy.T @ regression
            conditional_covariance = 0.5 * (
                conditional_covariance + conditional_covariance.T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(conditional_covariance)
            conditional_covariances[component] = (
                eigenvectors * np.maximum(eigenvalues, 0.0)[None, :]
            ) @ eigenvectors.T
            solved = np.linalg.solve(cholesky, delta.T)
            mahalanobis = np.square(solved).sum(axis=0)
            log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
            log_probabilities[:, component] = (
                np.log(mixture.weights[component])
                - 0.5
                * (
                    condition_dim * _LOG_2PI
                    + log_determinant
                    + mahalanobis
                )
            )
        maximum = log_probabilities.max(axis=1, keepdims=True)
        unnormalized = np.exp(log_probabilities - maximum)
        probabilities = unnormalized / unnormalized.sum(axis=1, keepdims=True)
        return probabilities, conditional_means, conditional_covariances

    def _build_model_fingerprint(self) -> str:
        pca = self._require_pca_state()
        mixture = self._require_mixture_state()
        digest = hashlib.sha256()
        metadata = {
            "model_id": CGMM_MODEL_ID,
            "config": self.config.to_dict(),
            "preprocessing_fingerprint": self.preprocessing_state.fingerprint,
            "target_component_count": self.target_component_count,
            "iteration_count": mixture.iteration_count,
        }
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        for array in (
            pca.components,
            pca.mean,
            np.asarray([pca.noise_variance], dtype=np.float64),
            mixture.weights,
            mixture.means,
            mixture.covariances,
        ):
            contiguous = np.ascontiguousarray(array, dtype="<f8")
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def export_state_arrays(self) -> dict[str, np.ndarray]:
        pca = self._require_pca_state()
        mixture = self._require_mixture_state()
        return {
            "pca_components": np.asarray(pca.components),
            "pca_mean": np.asarray(pca.mean),
            "pca_noise_variance": np.asarray(
                [pca.noise_variance],
                dtype=np.float64,
            ),
            "mixture_weights": np.asarray(mixture.weights),
            "mixture_means": np.asarray(mixture.means),
            "mixture_covariances": np.asarray(mixture.covariances),
        }

    @classmethod
    def restore(
        cls,
        *,
        config: CGMMConfig,
        preprocessing_state: CGMMPreprocessingState,
        arrays: Mapping[str, np.ndarray],
        converged: bool,
        iteration_count: int,
        expected_model_fingerprint: str,
        correction_state: CGMMCorrectionState | None = None,
    ) -> "ConditionalGaussianMixtureForecaster":
        required = {
            "pca_components",
            "pca_mean",
            "pca_noise_variance",
            "mixture_weights",
            "mixture_means",
            "mixture_covariances",
        }
        if set(arrays) != required:
            raise CGMMModelError("artifact arrays have an invalid schema")
        instance = cls(config)
        instance.preprocessor = CGMMPreprocessor.from_state(preprocessing_state)
        noise = np.asarray(arrays["pca_noise_variance"], dtype=np.float64)
        if noise.shape != (1,):
            raise CGMMModelError("pca_noise_variance must have shape (1,)")
        instance._pca_state = _PCAState(
            components=arrays["pca_components"],
            mean=arrays["pca_mean"],
            noise_variance=float(noise[0]),
        )
        instance._mixture_state = _MixtureState(
            weights=arrays["mixture_weights"],
            means=arrays["mixture_means"],
            covariances=arrays["mixture_covariances"],
            converged=converged,
            iteration_count=iteration_count,
        )
        instance._model_fingerprint = instance._build_model_fingerprint()
        if instance._model_fingerprint != expected_model_fingerprint:
            raise CGMMModelError("restored model fingerprint mismatch")
        instance._correction_state = correction_state
        return instance

    def _require_pca_state(self) -> _PCAState:
        if self._pca_state is None:
            raise CGMMModelError("model has not been fitted")
        return self._pca_state

    def _require_mixture_state(self) -> _MixtureState:
        if self._mixture_state is None:
            raise CGMMModelError("model has not been fitted")
        return self._mixture_state


CGMMForecaster = ConditionalGaussianMixtureForecaster


__all__ = [
    "CGMMForecaster",
    "CGMMModelError",
    "ConditionalGaussianMixtureForecaster",
]
