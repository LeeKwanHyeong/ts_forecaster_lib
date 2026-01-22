# modeling_module/training/model_trainers/patchtst_pretrain.py
from __future__ import annotations

import os
import json
from dataclasses import asdict, is_dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage


def _json_safe(obj):
    """
    JSON 직렬화가 불가능한 객체(Torch Device, Tensor, Path 등)를 문자열이나 기본 타입으로 변환.
    설정(Config) 로깅 시 오류 방지 목적.
    """
    # torch.device 처리
    try:
        import torch
        if isinstance(obj, torch.device):
            return str(obj)
    except Exception:
        pass

    # numpy / torch scalar 처리
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass

    # pathlib.Path 등 처리
    try:
        import os
        if isinstance(obj, os.PathLike):
            return str(obj)
    except Exception:
        pass

    # set/tuple -> list 변환
    if isinstance(obj, (set, tuple)):
        return list(obj)

    # 기본 문자열 변환
    return str(obj)


def _dump_cfg(cfg, save_dir: str, name: str) -> None:
    """
    학습 설정을 JSON 파일로 저장.
    """
    import json, os
    from dataclasses import asdict, is_dataclass

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)

    payload = asdict(cfg) if is_dataclass(cfg) else dict(cfg.__dict__)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_safe)


def _getattr(cfg, key: str, default):
    """안전한 속성 접근 유틸리티."""
    return getattr(cfg, key, default)


def _extract_x(batch):
    """
    배치 데이터에서 입력 시계열(x) 추출.

    기능:
    - 지도학습용 데이터셋((x, y, exo...) 튜플 형태)에서도 사전학습에 필요한 x만 분리.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _to_device(x, device: torch.device):
    """
    텐서 또는 텐서 컨테이너를 지정된 장치로 재귀적 이동.
    """
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


@torch.no_grad()
def _eval_pretrain(model, loader: DataLoader, device: torch.device, *, mask_ratio: float, loss_type: str) -> float:
    """
    검증 데이터셋에 대한 사전학습 손실(Reconstruction Loss) 평가.
    """
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = _to_device(_extract_x(batch), device)
        # 마스킹 비율 및 손실 함수 설정 적용
        out = model(x, mask_ratio=mask_ratio, return_loss=True, loss_type=loss_type)
        loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
        total += float(loss.detach().cpu().item())
        n += 1
    return total / max(n, 1)


def train_patchtst_pretrain(
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        stages: list[StageConfig] | None = None,
        train_cfg: Optional[TrainingConfig] = None,
        # Self-supervised 설정
        mask_ratio: float = 0.3,
        loss_type: str = "mse",
        # 입출력 설정
        save_dir: Optional[str] = None,
        ckpt_name: str = "patchtst_pretrain_best.pt",
):
    """
    PatchTST 자기지도 사전학습(Masked Patch Reconstruction) 실행 함수.

    기능:
    - 마스킹된 패치를 복원하는 태스크 수행.
    - CommonTrainer에 의존하지 않고 독립적인 PyTorch 학습 루프 구현.
    - AMP(Mixed Precision), Gradient Clipping, Multi-stage 학습 지원.
    - Best Validation Loss 기준 체크포인트 저장.

    요구사항:
    - 모델의 forward 시그니처가 `mask_ratio`, `return_loss` 인자를 지원해야 함.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."
    # 기본 스테이지 설정 (없을 경우 10 에포크 단일 스테이지)
    stages = stages or [StageConfig(epochs=10)]

    # 학습 환경 설정 (Device, AMP, Logging)
    device = torch.device(_getattr(train_cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(_getattr(train_cfg, "use_amp", _getattr(train_cfg, "amp", True)))
    grad_clip = float(_getattr(train_cfg, "grad_clip", 0.0))
    log_every = int(_getattr(train_cfg, "log_every", 100))

    model = model.to(device)

    best_val = float("inf")
    best_state = None

    # 설정 저장
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _dump_cfg(train_cfg, save_dir, "pretrain_cfg.json")

    # 스테이지별 학습 루프
    for si, stg in enumerate(stages):
        cfg_i = apply_stage(train_cfg, stg)
        epochs = int(_getattr(cfg_i, "epochs", _getattr(cfg_i, "max_epochs", 10)))
        lr = float(_getattr(cfg_i, "lr", 1e-3))
        weight_decay = float(_getattr(cfg_i, "weight_decay", 0.0))

        # 옵티마이저 초기화
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # AMP Scaler 초기화
        use_cuda_amp = bool(use_amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp) if use_cuda_amp else None

        print(
            f"[Pretrain] stage={si} epochs={epochs} lr={lr} wd={weight_decay} mask_ratio={mask_ratio} loss={loss_type}")

        for ep in range(1, epochs + 1):
            model.train()
            running = 0.0
            step = 0

            for batch in train_loader:
                step += 1
                # 입력 데이터 준비
                x = _to_device(_extract_x(batch), device)

                optimizer.zero_grad(set_to_none=True)

                # 순전파 및 손실 계산 (AMP 적용)
                if use_cuda_amp:
                    with torch.amp.autocast("cuda"):
                        out = model(x, mask_ratio=mask_ratio, return_loss=True, loss_type=loss_type)
                        loss = out["loss"] if isinstance(out, dict) and "loss" in out else out

                    # 역전파 및 가중치 업데이트
                    scaler.scale(loss).backward()

                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # AMP 미사용 시 일반적인 학습 루틴
                    out = model(x, mask_ratio=mask_ratio, return_loss=True, loss_type=loss_type)
                    loss = out["loss"] if isinstance(out, dict) and "loss" in out else out

                    loss.backward()

                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    optimizer.step()

                running += float(loss.detach().cpu().item())

                if (step % log_every) == 0:
                    print(f"[Pretrain][stage={si} ep={ep}/{epochs} step={step}] loss={running / step:.6f}")

            train_loss = running / max(step, 1)

            # 검증 및 체크포인트 저장
            if val_loader is not None:
                val_loss = _eval_pretrain(model, val_loader, device, mask_ratio=mask_ratio, loss_type=loss_type)
                print(f"[Pretrain][stage={si} ep={ep}/{epochs}] train={train_loss:.6f} val={val_loss:.6f}")

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if save_dir is not None:
                        ckpt_path = os.path.join(save_dir, ckpt_name)
                        torch.save(
                            {"state_dict": best_state, "best_val": best_val,
                             "cfg": asdict(cfg_i) if is_dataclass(cfg_i) else None},
                            ckpt_path,
                        )
            else:
                print(f"[Pretrain][stage={si} ep={ep}/{epochs}] train={train_loss:.6f}")

    # 학습 완료 후 최적 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    print(f"[Pretrain] done | best_val={best_val:.6f}" if val_loader is not None else "[Pretrain] done")
    return {"model": model, "best_val": best_val if val_loader is not None else None}