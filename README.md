# ts_forecaster_lib
> LTB(Last-Time-Buy) / 부품 수요예측을 위한 시계열 Forecasting 라이브러리 (Draft)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-red)](#)
[![Polars](https://img.shields.io/badge/Polars-supported-0c7)](#)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#)

---

## 1) 개요 (Overview)

`ts_forecaster_lib`는 **제조/서비스 부품(Spare Parts) 도메인**에서 발생하는 **일반적 수요** 및 **간헐적(intermittent) 수요**를 대상으로,
학습/추론/평가 파이프라인을 일관된 인터페이스로 제공하는 시계열 예측 라이브러리입니다.

특히 제조 도메인 수명주기 기반 예측 의사결정에서 요구되는:
- 긴 히스토리 기반의 **장기 예측**
- 품목별 편차가 큰 **lumpy/erratic** 패턴
- 캘린더/이벤트/보증(warranty) 등 **외생변수(exogenous) 결합**
- 운영 환경에서의 **대량 품목 일괄 실행(멀티파트)**

을 목표로 설계되었습니다.

---

## 2) 핵심 기능 (Key Features)

- **Multi-Part Forecasting**
  - 다수 품목(part_id / prod_no 등)을 한 번에 학습/추론하는 배치 파이프라인
  - 품목별 lookback/horizon, 필터링, 샘플링 전략 확장 가능

- **모델 지원 (Pluggable Models)**
  - 통계 모델(예: MA/SES/Holt/ARIMA 계열)과 딥러닝 모델을 동일한 실행 인터페이스로 구성
  - 포인트/분위수(Quantile) 예측을 공통 스키마로 저장/평가 가능

- **외생변수(Exogenous)**
  - 캘린더(월/주/요일/휴일), 이벤트, 보증 기간 기반 피처 등 확장
  - 모델별 exogenous 입력 지원/미지원 차이를 어댑터에서 흡수

- **평가/시각화**
  - MAE/RMSE/SMAPE 등 기본 지표 + 분위수 기반 지표 확장
  - 결과 parquet 저장 및 plot-only 유틸로 시각화 분리

- **운영지향 구조**
  - 서비스/도메인/데이터 레이어 분리(확장 예정)
  - 결과 스키마 표준화(모델 교체/엔SEMBLE/리포팅 용이)

> Draft Note: 실제 지원 모델/클래스명/패키지 구조는 현재 코드베이스 기준으로 맞춰 업데이트가 필요합니다(TODO 표기 참조).

---

## 3) 설치 (Installation)

### Option A. Editable install (개발 모드)
```bash
git clone <REPO_URL>
cd ts_forecaster_lib
pip install -e .