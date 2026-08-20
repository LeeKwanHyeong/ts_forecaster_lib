# Similar Lifecycle Forecast Contract

## 역할

Similar Lifecycle은 Lifecycle 시작 후 12개월의 관측값과 당시 알 수 있는
조건 특성을 이용해, 완료된 과거 Lifecycle 중 가장 가까운 이웃을 검색하고
M12~M83의 72개월 수요를 예측한다.

모델, 학습 구간 전용 전처리, 이웃 저장소, cohort·장기 Tail 보정, artifact
저장과 복원은 `ts_forecaster_lib`가 소유한다. Demand Engine은 DSDM 데이터를
`LifecycleSample`로 바꾸고 공개 API를 호출하며, DB 조회와 실행 이력 및 결과
적재는 라이브러리에 포함하지 않는다.

## 모델 식별자

- Registry key: `similar_lifecycle`
- Model contract: `modeling-module.similar-lifecycle.v1`
- Artifact contract: `modeling-module.similar-lifecycle-artifact.v1`
- Input window: M0~M11 관측
- Forecast window: M12~M83 72개월

`similar_lifecycle`은 일반 딥러닝 `train()` 경로를 사용하지 않는다.
`fit_similar_lifecycle()`과 `forecast_similar_lifecycle()`을 사용하며,
`build_model("similar_lifecycle", config)`은 빈 모델 객체 생성만 담당한다.

## 전처리와 거리

CGMM과 동일한 train-only `static_observed_v1` 또는
`static_observed_m0_v1` 전처리 상태를 공유한다. M0 profile은 예측 시 이미
알 수 있는 Lifecycle 시작 월을 ordinal로 추가하며 실제 미래 감소 속도는
사용하지 않는다. 수량 scale, 연속형 결측 대체값, 카테고리 사전, 조건 특성
평균과 표준편차는 학습 Lifecycle에서만 계산한다. 검증과 추론에서 처음
등장한 카테고리는 `<UNK>`로 처리한다.

M0 ordinal은 `demand_shape_static`과 `all` 거리 profile에서만 이웃 검색
거리에 포함된다. shape 전용 profile의 기존 거리 계약과
`static_observed_v1`의 feature layout은 변경하지 않는다.

기본 설정은 전체 조건 특성을 사용해 15개 이웃을 찾는다. 각 이웃의 거리에
역수를 적용해 가중치를 계산하고, 거리가 사실상 0인 이웃이 있으면 해당
이웃들에만 균등 가중치를 준다. 학습 표본을 다시 예측할 때는 같은
`sample_id`를 이웃 후보에서 제외한다.

## 예측 출력

`SimilarLifecyclePrediction`은 다음 값을 함께 반환한다.

- 72개월 가중 평균과 가중 표준편차
- nominal 90% 하한과 상한
- 이웃 sample ID, 가중치와 거리
- model, preprocessing, correction fingerprint

예측구간은 이웃 곡선의 가중 분산을 정규분포로 근사한 구간이다. 따라서
점 예측 설명에는 유용하지만, 확률분포 자체를 학습하는 CGMM보다 coverage가
낮을 수 있다.

## Tail 보정

`build_similar_lifecycle_rolling_evidence()`는 각 cohort를 예측할 때 그보다
이른 완료 Lifecycle만 검색 저장소에 포함한다. 보정 상태는 이 rolling 오차로
학습하며, 현재 기준 정책은 다음과 같다.

- cohort 추세 반영 비율: 0.5
- Tail 시작: 미래 37번째 월
- Tail half-life: 72개월
- 보정 범위: 0.20~1.50

동일한 양수 계수를 평균, 표준편차, 하한과 상한에 적용한다. 이웃 ID, 거리와
가중치는 바꾸지 않는다.

## Artifact

`save_similar_lifecycle_artifact()`는 pickle을 사용하지 않고 두 파일을 만든다.

- `manifest.json`: 모델 설정, 전처리 상태, correction state, 저장소 메타데이터
- `repository_arrays.npz`: 이웃 검색 조건 행렬과 완료된 72개월 곡선 비율

Manifest 전체 payload와 NPZ 파일은 각각 SHA-256으로 봉인한다.
`load_similar_lifecycle_artifact()`는 schema, version, checksum, 전처리와 모델
fingerprint를 확인한 뒤 복원한다. correction state도 artifact에 포함되므로
외부 보정 파일 없이 같은 예측을 재현한다.

## 공개 호출 예시

```python
from modeling_module import (
    SimilarLifecycleFitRequest,
    SimilarLifecycleForecastRequest,
    fit_similar_lifecycle,
    forecast_similar_lifecycle,
)

fit_result = fit_similar_lifecycle(
    SimilarLifecycleFitRequest(
        samples=training_samples,
        dataset_fingerprint=dataset_sha256,
    )
)

prediction = forecast_similar_lifecycle(
    SimilarLifecycleForecastRequest(
        model=fit_result.model,
        samples=inference_samples,
    )
)
```

## 이관 검증

기존 Demand Engine 구현과 현재 DSDM 7,000개 Lifecycle에서 비교했을 때 평균,
표준편차, 이웃 ID, 가중치와 거리가 모두 bitwise 동일했다. 전처리와 모델
fingerprint는 새 소유권과 artifact 계약을 반영하므로 의도적으로 달라진다.

현재 정확도와 qualification 결과는
`../../CGMM/docs/LTBBaseline.md`에 함께 고정한다.
