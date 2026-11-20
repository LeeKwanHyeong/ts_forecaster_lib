"""출력 보정의 배열·시간·기준 모델 계약. 모델 구현에는 의존하지 않는 경계."""

from dataclasses import dataclass
import re

import numpy as np


def real_array(value, name, ndim):
    source = np.asarray(value)
    if source.dtype.kind not in "fiu" or source.ndim != ndim:
        raise TypeError(f"{name}은 {ndim}차원 실수 수치 배열이어야 합니다.")
    result = np.array(source, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name}에 결측·Inf가 있습니다. v1은 조밀한 관측 창만 지원합니다.")
    result.setflags(write=False)
    return result


def _names(value, count, name):
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{name}의 길이가 배열과 다릅니다.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name}에는 비어 있지 않은 문자열이 필요합니다.")
    return tuple(value)


@dataclass(frozen=True)
class ForecastQuery:
    """origin은 첫 예측 위치의 연속 정수 인덱스. ISO YYYYWW를 직접 더하지 않음."""

    history: np.ndarray
    backbone_prediction: np.ndarray
    origins: np.ndarray
    channel_names: tuple[str, ...]
    series_ids: tuple[str, ...]
    backbone_id: str
    scale_id: str = "original_units"

    def __post_init__(self):
        history = real_array(self.history, "history", 3)
        prediction = real_array(self.backbone_prediction, "backbone_prediction", 3)
        n, length, channels = history.shape
        if min(n, length, channels) < 1 or prediction.shape[0] != n or prediction.shape[2] != channels or prediction.shape[1] < 1:
            raise ValueError("history=(N,L,C), backbone_prediction=(N,H,C)가 서로 일치해야 합니다.")
        origins = np.asarray(self.origins)
        if origins.dtype.kind not in "iu" or origins.shape != (n,):
            raise TypeError("origins는 길이 N의 정수 배열이어야 합니다.")
        if any(abs(int(value)) >= 2**50 for value in origins):
            raise ValueError("시간 인덱스 범위가 너무 큽니다.")
        origins = np.array(origins, dtype=np.int64, copy=True)
        origins.setflags(write=False)
        names = _names(self.channel_names, channels, "channel_names")
        if len(set(names)) != channels:
            raise ValueError("채널 이름은 중복될 수 없습니다.")
        ids = _names(self.series_ids, n, "series_ids")
        if len(set(zip(ids, origins.tolist()))) != n:
            raise ValueError("동일 시계열·예측 시작 위치가 중복되었습니다.")
        for name in ("backbone_id", "scale_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name}은 비어 있지 않은 문자열이어야 합니다.")
        for name, value in (("history", history), ("backbone_prediction", prediction), ("origins", origins),
                            ("channel_names", names), ("series_ids", ids)):
            object.__setattr__(self, name, value)

    def take(self, indices):
        indices = np.asarray(indices)
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise TypeError("indices는 1차원 정수 배열이어야 합니다.")
        if np.any(indices < 0) or np.any(indices >= len(self.origins)):
            raise ValueError("query 선택 인덱스가 범위를 벗어났습니다.")
        indices = indices.astype(np.int64)
        return ForecastQuery(self.history[indices], self.backbone_prediction[indices], self.origins[indices],
                             self.channel_names, tuple(self.series_ids[i] for i in indices), self.backbone_id, self.scale_id)

    @property
    def signature(self):
        return {"backbone_id": self.backbone_id, "scale_id": self.scale_id, "channel_names": list(self.channel_names),
                "lookback": self.history.shape[1], "horizon": self.backbone_prediction.shape[1]}


@dataclass(frozen=True)
class ForecastFitData:
    """정답은 fit 전용. 기준 예측의 OOF 여부는 호출자가 제공한 선언·증적과 구분."""

    query: ForecastQuery
    target: np.ndarray
    backbone_provenance: str
    oof_manifest_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.query, ForecastQuery):
            raise TypeError("query는 ForecastQuery여야 합니다.")
        target = real_array(self.target, "target", 3)
        if target.shape != self.query.backbone_prediction.shape:
            raise ValueError("정답과 기준 예측 shape가 다릅니다.")
        if self.backbone_provenance not in {"frozen_pretrained", "synthetic_history_only", "blocked_oof"}:
            raise ValueError("기준 예측은 frozen_pretrained, synthetic_history_only, blocked_oof 중 하나로 선언해야 합니다.")
        if self.backbone_provenance == "blocked_oof" and not re.fullmatch(r"[0-9a-f]{64}", self.oof_manifest_sha256 or ""):
            raise ValueError("학습형 backbone의 blocked_oof 선언에는 manifest SHA-256이 필요합니다.")
        object.__setattr__(self, "target", target)

    def take(self, indices):
        return ForecastFitData(self.query.take(indices), self.target[indices], self.backbone_provenance, self.oof_manifest_sha256)


@dataclass(frozen=True)
class AdaptedForecast:
    """보정 점예측과 원본을 분리. 확률 분위수의 보정 결과로 표시하지 않음."""

    point: np.ndarray
    backbone_point: np.ndarray
    correction: np.ndarray
    metadata: dict
