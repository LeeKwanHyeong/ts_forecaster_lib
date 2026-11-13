"""합성 수요로 마스킹 사전학습 → 가중치 전이 → 예측 학습을 확인한다.

저장소 루트에서 다음 명령으로 실행한다.
PYTHONPATH=src python -m modeling_module.models.TinyTimeMixer.examples.train_from_scratch

단계별 연산 확인용 예제이며, 출력 손실은 실제 부품 예측 성능의 증거가 아니다.
"""

from dataclasses import replace

import torch

from modeling_module.models.TinyTimeMixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerModel,
    TinyTimeMixerPretrainModel,
)


def main() -> None:
    """CPU에서 짧게 실행하며 데이터·체크포인트 파일은 생성하지 않는다."""
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = TinyTimeMixerConfig(
        d_model=16, decoder_d_model=16, num_layers=1, decoder_num_layers=1,
        adaptive_patching_levels=1, dropout=0,
        categorical_vocab_size_list=(4, 8),
    )
    # 같은 수요 모양에도 수준·위상을 달리한 샘플을 만들고 정적 코드를 붙인다.
    part = torch.arange(8, dtype=torch.float32).view(-1, 1)
    week = torch.arange(cfg.lookback + cfg.horizon, dtype=torch.float32).view(1, -1)
    demand = 100 + 3 * part + 12 * torch.sin(week / 4 + part / 8) + week / 10
    history = demand[:, :cfg.lookback].unsqueeze(-1)
    future = demand[:, cfg.lookback:].unsqueeze(-1)
    static_ids = torch.stack((torch.arange(8) % 4, torch.arange(8)), dim=1)

    pretrainer = TinyTimeMixerPretrainModel(cfg)
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad()
        result = pretrainer(history, static_categorical_values=static_ids, mask_ratio=0.3)
        result["loss"].backward()
        optimizer.step()
        print(f"복원 사전학습 {step + 1}: loss={result['loss'].item():.6f}")

    # 복원 헤드는 옮기지 않는다. 새 미래 예측 헤드는 다음 단계에서 학습한다.
    forecaster = TinyTimeMixerModel(replace(cfg, horizon=26))
    pretrainer.transfer_to(forecaster, freeze_backbone=False)
    optimizer = torch.optim.AdamW(forecaster.parameters(), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad()
        prediction = forecaster(history, static_categorical_values=static_ids)
        loss = torch.nn.functional.mse_loss(prediction, future)
        loss.backward()
        optimizer.step()
        print(f"미래 예측 학습 {step + 1}: loss={loss.item():.6f}")

    forecaster.eval()
    with torch.no_grad():
        prediction = forecaster(history, static_categorical_values=static_ids)
    print(f"과거 입력: {tuple(history.shape)}, 정적 입력: {tuple(static_ids.shape)}")
    print(f"미래 출력: {tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
