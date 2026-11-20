# 구현 출처와 범위

## 원본 코드 복사 없는 독립 구현

- 논문: [MORA: Correcting Time Series Forecasts with Coarse Future Signals, arXiv:2611.15413v2](https://arxiv.org/abs/2611.15413v2)
- 구현 기준: 논문의 식 (3)~(11), 부록 C.1·C.3·C.4·C.5
- 논문 저자 코드: https://github.com/hugerli-1/MORA
- 확인일: 2026-11-20
- 확인한 저자 저장소 HEAD: `0f9d1f0f8c63ace7de75aec2f9ccf8a557fb6f81`

저자 저장소의 [NOTICE.md](https://github.com/hugerli-1/MORA/blob/0f9d1f0f8c63ace7de75aec2f9ccf8a557fb6f81/NOTICE.md)에
영구적인 소프트웨어 라이선스가 아직 선택되지 않았다는 설명 존재.
사용자 확인 후 해당 저장소 코드를 복사하지 않고 논문 수식으로 독립 구현한 버전.
`native/`는 외부 패키지 래퍼나 저자 코드의 vendoring 경로가 아닌 내부 연산 소스의 위치.
저자 라이선스를 임의로 부여하거나 저자 구현과 수치 동등성을 주장하지 않는 기준.

## 수식과 파일의 대응

| 논문 개념 | 구현 파일 |
|---|---|
| 구간 평균 연산 A | `native/basis.py:coarse_operator` |
| tapered IDCT + QR와 첫 예측값 보존 | `native/basis.py:tapered_basis` |
| compact 과거 특징·시간 인코딩 | `native/features.py:compact_features` |
| 현재 수준·최근 패턴 외삽 | `native/coarse_future.py:extrapolated` |
| absolute·last-centered ridge 평균 | `native/coarse_future.py:CoarseEstimator` |
| 비중심 가중 모멘트·shrinkage·선형 시스템 | `native/stagewise.py:fit_mapping` |
| pooled·보수적 block gain | `native/stagewise.py:reliability` |
| 학습 내부 OOF와 2단계 순차 보정 | `fitting.py` |
| 검증 자료에만 적합하는 그룹별 calibration | `native/calibration.py` |

## 구현 선택과 미포함 범위

1. 기본 L52/H27, 구간 6개, 보정 rank 16은 프로젝트 실험 설정. 합성 예제는 period 13·rank 12 사용.
2. Compact 78개 특징만 구현. 최근 24개 관측의 23개 차분을 12개 비어 있지 않은 블록으로 분할.
3. 초기 수준과 최근 패턴의 두 외삽만 사용. 학습형 harmonic template, factorized estimator, 확장 252/348개 특징은 미구현.
4. 정규화 강도는 명시적 고정 설정. 논문의 후보 그리드 탐색·benchmark별 설정 자동 재현은 미포함.
5. OOF는 전체 학습 origin을 연속 블록으로 나누고 raw history/target 중첩 제거. 논문의 후반 origin·stride 6 설정을 그대로 복제하지 않은 구현 선택.
6. 2단계 gain 평가에서 outer held 블록의 정답이 내부 coarse·gain·매핑에 섞이지 않도록 nested OOF 적용.
7. Calibration은 식 (39) 형태의 목적함수와 bounded IRLS 사용. 검증 목적함수가 더 좋은 경우 고정된 무보정 scale 0도 선택하는 추가 안전 장치 포함. 테스트별 fallback은 없음.
8. 기본 `single_series`는 같은 시계열 ID와 채널 순서만 허용. `pooled_series`는 서로 다른 시계열의 query를 채널 역할별로 함께 적합하는 명시적 프로젝트 연구 확장.
9. v1은 모두 유한한 조밀 관측 창만 지원. NaN·Inf를 0으로 채우지 않고 거부하며 실제 관측 0은 그대로 사용.
10. 추론은 calibration 정답 종료 이후 query만 허용. 통계·파라미터 갱신과 미래 정답 입력은 없음.

합성 오차 개선은 계산 검증에 한정. 논문 전체 재현, 실제 부품 성능 개선, cold-part 일반화 검증과 구분.
