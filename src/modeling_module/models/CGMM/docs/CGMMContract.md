# CGMM Lifecycle Forecast Contract

현재 LTB 모델 역할과 비교 결과는 [LTBBaseline.md](LTBBaseline.md)에 고정한다.

## 역할

CGMM은 Lifecycle 시작 후 12개월의 관측값으로 M12~M83의 72개월 수요
분포를 예측하는 통계 모델이다. 모델 수식, train-only 전처리, rolling
검증, 분포 보정, artifact 저장과 복원은 `ts_forecaster_lib`가 소유한다.

Demand Engine은 DSDM 데이터를 `LifecycleSample`로 변환하고 모델 artifact를
선택한 뒤 공개 API를 호출한다. DB 조회, tenant, Run 이력, timeout, 결과 적재는
라이브러리에 포함하지 않는다.

## 모델 식별자

- Registry key: `cgmm`
- Model contract: `modeling-module.cgmm.v1`
- Artifact contract: `modeling-module.cgmm-artifact.v1`
- Input window: M0~M11 관측, M12~M83 예측

`cgmm`은 기존 딥러닝 `train()`과 데이터 로더를 사용하지 않는다. 공개
`fit_cgmm()`과 `forecast_cgmm()`을 사용하며, `build_model("cgmm", config)`은
모델 객체 생성만 담당한다.

## 학습 입력

`CGMMFitRequest`는 다음 값을 받는다.

- 완료된 `LifecycleSample` 목록
- 원본 학습 데이터의 SHA-256 fingerprint
- `CGMMConfig`
- `CGMMPreprocessingConfig`
- 선택적인 `CGMMCorrectionState`

모든 학습 표본은 같은 순서의 `LifecycleFeatureSchema`를 사용해야 한다.
수량 정규화 통계, 연속형 결측 대체값, 카테고리 사전, 조건 변수 평균과
표준편차는 학습 표본에서만 계산한다. 추론에서 처음 등장한 카테고리는
해당 feature slot의 `<UNK>` 열로 변환된다.

정적 관측 전처리는 다음 두 profile을 제공한다.

- `static_observed_v1`: 기존 조건 벡터를 그대로 유지하는 호환 profile
- `static_observed_m0_v1`: 기존 조건 벡터에
  `year * 12 + month - 1` 형태의 Lifecycle 시작 월 ordinal을 추가하는 profile

M0 ordinal은 예측 요청 시 이미 알 수 있는 값이다. 학습 표본에서 계산한 평균과
표준편차로 정규화하며, 실제 미래 72개월의 감소 속도나 정답 값은 조건 벡터에
포함하지 않는다. 선택한 profile과 정규화 상태는 artifact에 저장된다.

## 예측 출력

`CGMMPrediction`은 다음 값을 함께 반환한다.

- mixture별 조건부 확률 `(N, K)`
- mixture별 72개월 후보 곡선 `(N, K, 72)`
- 확률 가중 평균 `(N, 72)`
- 표준편차와 하한·상한 `(N, 72)`
- model key, contract ID, model/preprocessing/correction fingerprint

확률 가중 후보곡선은 항상 평균 예측과 같아야 한다. 모든 수량 출력은
유한한 0 이상의 값이어야 하며 하한은 상한을 넘을 수 없다.

## Rolling 검증과 보정

`build_cgmm_rolling_evidence()`는 각 Validation cohort를 예측할 때 최초 학습
표본과 그보다 앞선 cohort만 사용한다. 현재 또는 이후 cohort가 전처리나
모델 fitting에 들어가면 안 된다.

`fit_cgmm_correction()`은 rolling evidence에서 horizon block별 cohort 추세와
선택적인 scale gate, 장기 tail 감쇠를 추정한다. 보정 시 하나의 양수 계수를
후보곡선, 평균, 표준편차, 하한, 상한에 모두 곱한다. mixture 확률은 바꾸지
않으므로 보정 후에도 후보곡선 가중합과 평균이 일치한다.

`short_horizon_cohort_strength`를 설정하면 M1~M12 block에만 별도의 cohort
반영 비율을 적용한다. M25 이후에는 기존 `cohort_strength`와 같은 계수를
사용하므로 M37~M72 Tail 계약은 변하지 않는다. 기본값 `None`은 기존 세 block
공통 strength와 직렬화 fingerprint를 그대로 유지한다.

## Artifact

`save_cgmm_artifact()`는 pickle을 사용하지 않고 두 파일을 만든다.

- `manifest.json`: config, 전처리 state, correction state, schema와 fingerprint
- `model_arrays.npz`: PCA와 Gaussian mixture 수치 파라미터

Manifest는 전체 payload fingerprint를 포함하고 NPZ 파일은 SHA-256으로
봉인한다. `load_cgmm_artifact()`는 schema, artifact version, SHA, 전처리
fingerprint, correction fingerprint와 최종 model fingerprint를 모두 확인한
후 모델을 복원한다.

## 공개 호출 예시

```python
from modeling_module import (
    CGMMFitRequest,
    CGMMForecastRequest,
    fit_cgmm,
    forecast_cgmm,
)

fit_result = fit_cgmm(
    CGMMFitRequest(
        samples=training_samples,
        dataset_fingerprint=dataset_sha256,
    )
)

prediction = forecast_cgmm(
    CGMMForecastRequest(
        model=fit_result.model,
        samples=inference_samples,
    )
)
```
