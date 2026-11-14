# PatchTST pretrain loss 안전성 수정

## 확인한 문제

기존 pretrain은 학습 가능한 RevIN affine을 입력과 재구성 정답 양쪽에 적용하는 구조.
affine scale이 0에 가까워지면 정답 스케일도 축소되어, 시계열 복원 성능 향상 없이 reconstruction loss가 낮아지는 경로 존재.

합성 시계열 CPU 실험에서 학습 가능한 affine은 검증 MSE 약 3.2e-15,
affine 고정/미사용 대조군은 약 0.9574로 관찰된 결과.
이 결과는 소스의 문제 경로를 재현한 것이며 삼성 운영 현상의 원인 확정이나 동일 epoch 재현 결과는 아님.

## 수정 범위

- pretrain RevIN만 affine=False로 변경
- 기존 pretrain의 평균 중심 정규화, use_std, fine-tuning 모델 및 LR/dropout/Spike 정책 유지
- pretrain 정답을 학습 가능한 affine 파라미터에서 분리
- 무작위 마스크는 샘플별 최소 한 패치 보장
- 0 이하/범위 초과/NaN/Inf mask_ratio 및 외부에서 전달한 빈 마스크 거부
- 재구성 loss와 학습/검증 loss의 NaN/Inf는 명시적 예외 처리
- 빈 loader와 유효 처리 배치 0개는 loss=0 대신 명시적 예외 처리
- 실제 처리한 배치 수로 epoch 평균 산출, 조기 종료되는 profiler 경로에도 적용
- loss는 과학적 표기법으로 출력하고 처리 배치 수 및 pretrain 마스크 수 기록

공통 Trainer의 정상적인 0 loss는 허용하는 정책.
문제가 있는 입력이나 실패한 계산을 0으로 바꾸어 정상 학습처럼 기록하지 않는 것이 수정 목적.

## 체크포인트 주의사항

- fine-tuning의 기존 affine=False 및 supervised checkpoint 로딩 경로는 유지
- 새 pretrain state_dict에는 revin_layer.affine_weight/affine_bias가 없는 구조
- 과거 affine=True pretrain state를 새 pretrain 모델에 strict=True로 직접 복원하면 예상치 못한 key 오류가 날 수 있는 변경
- 과거 pretrain의 affine이 이미 축소된 경우 설정 변경만으로 backbone 학습 품질이 복구된다고 볼 수 없음
- 기존 checkpoint 자동 변환, 파일 교체, 운영 재학습 및 Runtime 배포는 미수행 범위
- 기존 pretrained artifact와 새 artifact는 분리 관리하고, 필요한 경우 새 pretrain부터 검증하는 순서

## 브랜치별 적용 기준

- samsung_gcs: 본 수정만 독립 커밋, 기존 신규 모델 이식 작업은 별도 보존
- ts_forecaster_lib_v2: 고정 검증 마스크 RNG, 미래 범주형 외생변수와 production_refit 경로 보존
- V2 production_refit은 검증 loader=None을 정상적으로 허용하되, 비어 있거나 비정상인 학습은 거부
- main: 검증한 V2 수정만 PR을 통해 병합

## 검증

- tests/test_patchtst_loss_safety.py: affine 제거, 정답 스케일 고정, 가중치 업데이트, 마스크 및 NaN/Inf/빈 loader 방어, 실제 배치 수 평균 검증
- V2 전용 테스트: production_refit에서 검증 비활성화 유지 확인
- 기존 checkpoint, 모델 학습, 외생변수 및 SSL 관련 회귀 테스트 실행
- 설치 버전과 원본 로그가 없는 삼성 운영 실행에 대한 원인 확정은 미수행 범위
