"""pickle 없는 단일 NPZ 상태. 무결성 해시는 서명이나 신뢰할 수 없는 파일의 보안 인증과 구분."""

from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .configs import MORAConfig
from .fitting import MORAState, canonical_state
from .native.basis import coarse_operator, tapered_basis
from .native.coarse_future import CoarseEstimator, RidgeView


def _arrays(state):
    result = {name: getattr(state, name) for name in
              ("operator", "basis", "first_mapping", "second_mapping", "first_gain", "second_gain", "calibration")}
    for view in ("absolute", "relative"):
        for name in ("feature_mean", "feature_scale", "target_mean", "target_scale", "weight"):
            result[f"{view}_{name}"] = getattr(getattr(state.estimator, view), name)
    return result


def state_digest(metadata, arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=True, allow_nan=False).encode())
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value, dtype="<f8")
        digest.update(f"{name}:{value.shape}:float64\n".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validate(config, metadata, arrays):
    signature = metadata["fit"]["signature"]
    names = signature["channel_names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("저장된 채널 계약이 유효하지 않습니다.")
    if signature["lookback"] != config.lookback or signature["horizon"] != config.horizon:
        raise ValueError("저장된 길이 계약이 설정과 다릅니다.")
    if any(not isinstance(signature[name], str) or not signature[name] for name in ("backbone_id", "scale_id")):
        raise ValueError("저장된 기준 모델·단위 계약이 유효하지 않습니다.")
    if metadata["fit"]["scope"] != config.scope or not metadata["fit"]["fit_series_ids"]:
        raise ValueError("저장된 시계열 범위가 유효하지 않습니다.")
    channels, rank, count = len(names), config.correction_rank, config.coarse_count
    shapes = {"operator": (count, config.horizon), "basis": (config.horizon, rank),
              "first_mapping": (channels, rank, 2 * count), "second_mapping": (channels, rank, count),
              "first_gain": (rank,), "second_gain": (rank,), "calibration": (channels, rank)}
    for view in ("absolute", "relative"):
        shapes.update({f"{view}_feature_mean": (channels, 78), f"{view}_feature_scale": (channels, 78),
                       f"{view}_target_mean": (channels, count), f"{view}_target_scale": (channels, count),
                       f"{view}_weight": (channels, 78, count)})
    if set(arrays) != set(shapes):
        raise ValueError("저장 상태의 배열 이름이 계약과 다릅니다.")
    for name, shape in shapes.items():
        if arrays[name].shape != shape or arrays[name].dtype.kind != "f" or not np.isfinite(arrays[name]).all():
            raise ValueError(f"저장 배열의 shape 또는 수량 오류: {name}")
        if name.endswith("_scale") and np.any(arrays[name] <= 0):
            raise ValueError("ridge 표준화 scale은 양수여야 합니다.")
    if not np.allclose(arrays["operator"], coarse_operator(config.horizon, count), rtol=0, atol=1e-14):
        raise ValueError("저장된 구간 평균 연산자가 설정과 다릅니다.")
    expected = tapered_basis(config.horizon, rank, config.taper_steps)
    if not np.allclose(arrays["basis"], expected, rtol=0, atol=1e-12):
        raise ValueError("저장된 보정 기저가 설정과 다릅니다.")
    for name in ("first_gain", "second_gain"):
        if np.any((arrays[name] < 0) | (arrays[name] > config.gain_max)):
            raise ValueError("gain이 허용 범위를 벗어났습니다.")
    if np.any((arrays["calibration"] < 0) | (arrays["calibration"] > config.calibration_max)):
        raise ValueError("calibration scale이 허용 범위를 벗어났습니다.")
    if type(metadata["fit"]["calibration_target_end"]) is not int:
        raise ValueError("저장된 시간 cutoff가 정수가 아닙니다.")


def save_adapter(adapter, path):
    if adapter._state is None:
        raise RuntimeError("적합 전에는 저장할 수 없습니다.")
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(f"기존 보정기를 덮어쓰지 않습니다: {path}")
    metadata = {"format_version": 1, "method": "MORA_paper_independent_v1", "config": asdict(adapter.config),
                "fit": adapter.diagnostics}
    arrays = _arrays(adapter._state)
    _validate(adapter.config, metadata, arrays)
    metadata["state_sha256"] = state_digest(metadata, arrays)
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays, metadata_json=np.frombuffer(encoded, dtype=np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        # 완성된 파일만 공개. 동시 저장에서도 기존 파일을 교체하지 않는 local filesystem 계약.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_adapter(path):
    from .adapter import MORAAdapter
    try:
        with np.load(path, allow_pickle=False) as archive:
            encoded = archive["metadata_json"]
            if encoded.dtype != np.uint8 or encoded.ndim != 1 or len(encoded) > 2_000_000:
                raise ValueError("보정기 metadata 형식 또는 크기가 유효하지 않습니다.")
            metadata = json.loads(encoded.tobytes().decode())
            if type(metadata["format_version"]) is not int or metadata["format_version"] != 1 or metadata["method"] != "MORA_paper_independent_v1":
                raise ValueError("지원하지 않는 보정기 형식입니다.")
            digest = metadata.pop("state_sha256")
            arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
        if state_digest(metadata, arrays) != digest:
            raise ValueError("보정기 상태 SHA-256이 일치하지 않습니다.")
        config = MORAConfig(**metadata["config"])
        _validate(config, metadata, arrays)
        views = []
        for view in ("absolute", "relative"):
            views.append(RidgeView(**{name: arrays[f"{view}_{name}"] for name in
                                      ("feature_mean", "feature_scale", "target_mean", "target_scale", "weight")}))
        estimator = CoarseEstimator(*views, config.period)
        state = MORAState(**{name: arrays[name] for name in
                            ("operator", "basis", "first_mapping", "second_mapping", "first_gain", "second_gain", "calibration")},
                          estimator=estimator, metadata=metadata["fit"])
    except (KeyError, TypeError, UnicodeError) as error:
        raise ValueError("보정기 파일의 필수 계약이 손상되었습니다.") from error
    result = MORAAdapter(config)
    result._state = canonical_state(state)
    return result
