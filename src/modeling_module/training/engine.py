import copy
import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple

import torch
from torch.amp import autocast, GradScaler

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.model_losses.losses import LossComputer


class CommonTrainer:
    """
    LossComputer를 래핑하여 학습 전반을 관장하는 범용 트레이너.

    기능:
    - 다양한 배치 형식((x,y)부터 (x,y,exo...)까지)의 표준화 처리.
    - 외생 변수(Exogenous Variable)의 우선순위 조정 (Batch > Callback).
    - 어댑터를 통한 모델 입출력 인터페이스 통일.
    - Spike-aware Loss 디버깅 및 비교 분석 지원.
    """

    def __init__(
            self,
            cfg,
            adapter: DefaultAdapter,
            *,
            metrics_fn=None,
            logger=print,
            future_exo_cb=None,
            autocast_input=None,
            extra_loss_fn=None,
            use_exogenous_mode=False,
    ):
        """
        학습 트레이너 초기화 및 필수 컴포넌트 구성.

        기능:
        - 설정(Config), 어댑터, 로거, 손실 계산기 등 핵심 객체 연결.
        - AMP(Automatic Mixed Precision) 환경 및 Autocast 옵션 설정.
        - 디버깅용 상태 변수 초기화 및 확정된 설정 정보 로깅.
        """
        self.cfg = cfg
        self.adapter: DefaultAdapter = adapter
        self.logger = logger
        self.loss_comp = LossComputer(cfg)
        self.metrics_fn = metrics_fn
        self.future_exo_cb = future_exo_cb

        # AMP(Mixed Precision) 활성화 여부 결정 (CUDA 가용성 확인 포함)
        self.amp_enabled = (self.cfg.amp_device == "cuda" and torch.cuda.is_available())
        self.autocast_input = autocast_input or {}
        self.extra_loss_fn = extra_loss_fn
        self.use_exogenous_mode = use_exogenous_mode

        # Autocast 실행을 위한 세부 옵션(장치, 활성화 여부, 데이터 타입) 파싱
        self.amp_device = self.autocast_input.get("device_type", self.cfg.amp_device)
        self.enabled = self.autocast_input.get("enabled", self.amp_enabled)
        self.dtype = self.autocast_input.get("dtype", None)

        # Spike Loss 효과 분석 및 디버깅을 위한 내부 카운터/비교군 초기화
        self._dbg_spike_seen = 0
        self._dbg_loss_comp_base = None  # 비교용 (Spike OFF) 손실 계산기
        self._dbg_max_print = 3  # 초기 N개 배치에 대해서만 상세 로그 출력 제한

        def _dump(obj, title):
            """설정 객체를 JSON 형식으로 직렬화하여 로깅하는 내부 헬퍼."""
            data = asdict(obj) if is_dataclass(obj) else obj.__dict__
            self.logger(f"[CommonTrainer] {title}")
            self.logger(json.dumps(data, indent=2, ensure_ascii=False, default=str))

        # 최종 확정된 학습 설정 및 어댑터 설정 로깅
        _dump(self.cfg, "TrainingConfig (final)")
        if hasattr(self.adapter, "cfg"):
            _dump(self.adapter.cfg, "Adapter Config")

    def _get_spike_enabled(self) -> bool:
        """현재 설정에서 Spike Loss 활성화 여부 확인."""
        sl = self.cfg.get("spike_loss") if isinstance(self.cfg, dict) else getattr(self.cfg, "spike_loss", None)
        if sl is None:
            return False
        if isinstance(sl, dict):
            return bool(sl.get("enabled", False))
        return bool(getattr(sl, "enabled", False))

    def _clone_cfg_disable_spike(self):
        """Spike Loss를 비활성화한 설정 복제본 생성 (비교 분석용).

        NOTE:
        - cfg 안에 nn.Module(loss 등) / Tensor가 들어갈 수 있어 deepcopy가 깨질 수 있음.
        - 여기서는 'spike_loss.enabled'만 끄는 얕은 복제(shallow clone)로 충분함.
        """
        cfg = self.cfg

        if isinstance(cfg, dict):
            cfg2 = dict(cfg)
            spike = dict(cfg2.get("spike_loss", {}))
            spike["enabled"] = False
            cfg2["spike_loss"] = spike
            return cfg2

        try:
            import dataclasses
            if dataclasses.is_dataclass(cfg) and hasattr(cfg, "spike_loss"):
                sl = getattr(cfg, "spike_loss")
                if isinstance(sl, dict):
                    sl2 = dict(sl)
                    sl2["enabled"] = False
                    return dataclasses.replace(cfg, spike_loss=sl2)
                cfg2 = dataclasses.replace(cfg)
                try:
                    cfg2.spike_loss.enabled = False
                except Exception:
                    pass
                return cfg2
        except Exception:
            pass

        import copy as _copy
        cfg2 = _copy.copy(cfg)
        if hasattr(cfg2, "spike_loss"):
            sl = getattr(cfg2, "spike_loss")
            if isinstance(sl, dict):
                sl2 = dict(sl)
                sl2["enabled"] = False
                setattr(cfg2, "spike_loss", sl2)
            else:
                try:
                    sl.enabled = False
                except Exception:
                    pass
        return cfg2

    def _debug_spike_breakdown(self, pred, y, *, is_val: bool, tag: str):
        """
        Spike Loss 적용 전후의 손실 값 비교 디버깅.

        기능:
        - 동일한 예측값에 대해 Spike ON/OFF 손실을 각각 계산.
        - 두 손실의 차이(Delta)를 로깅하여 Spike 가중치의 영향력 모니터링.
        """
        # 1) 타겟 데이터 통계 확인
        with torch.no_grad():
            y_f = y.detach()
            y_abs_max = float(y_f.abs().max().item())
            y_mean = float(y_f.mean().item())
            y_zero_ratio = float((y_f == 0).float().mean().item())

        # 2) 비교군(Spike OFF) LossComputer 준비
        if self._dbg_loss_comp_base is None:
            cfg_no_spike = self._clone_cfg_disable_spike()
            self._dbg_loss_comp_base = LossComputer(cfg_no_spike)

        # 3) 손실 계산 및 비교
        loss_on = self.loss_comp.compute(pred, y, is_val=is_val)
        loss_off = self._dbg_loss_comp_base.compute(pred, y, is_val=is_val)

        # 4) 스칼라 변환 및 로깅
        def _scalar(v):
            if torch.is_tensor(v):
                return float(v.detach().float().mean().item()) if v.numel() > 1 else float(v.detach().item())
            return float(v)

        lon = _scalar(loss_on)
        loff = _scalar(loss_off)
        delta = lon - loff

        self.logger(
            f"[DBG-{tag}] spike_enabled=True | loss_on={lon:.6e} | loss_off={loff:.6e} | delta={delta:.6e} | "
            f"y_mean={y_mean:.3e} y_abs_max={y_abs_max:.3e} y_zero_ratio={y_zero_ratio:.3f}"
        )

    # ----------------- 내부 유틸 -----------------
    @staticmethod
    def _to_tensor(x, device):
        """입력을 지정된 장치의 텐서로 변환."""
        if x is None:
            raise RuntimeError("[Loss None] loss is None. Check LossComputer and model output.")
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _normalize_future_exo_from_cb(self, x, y, *, device) -> Optional[torch.Tensor]:
        """
        Callback으로부터 생성된 외생 변수를 모델 입력 형태 [B, H, E]로 정규화.

        기능:
        - 차원 불일치(1차원, 2차원 등) 자동 보정.
        - 배치 크기(B)에 맞춰 브로드캐스팅 수행.
        """
        if self.future_exo_cb is None:
            return None

        B = x.size(0)
        H = y.size(1)
        t0 = 0

        exo = self.future_exo_cb(t0, H, device=device)
        if not torch.is_tensor(exo):
            raise TypeError(f"future_exo_cb must return torch.Tensor, got {type(exo)}")

        # 불필요한 차원 제거
        while exo.dim() >= 3 and exo.size(0) == 1:
            exo = exo.squeeze(0)

        # 차원 확장 및 배치 맞춤
        if exo.dim() == 2:
            exo = exo.unsqueeze(0)  # (1,H,E)
        elif exo.dim() == 3:
            pass  # (B' or 1, H, E)
        elif exo.dim() == 4 and exo.size(0) == 1 and exo.size(1) == 1:
            exo = exo.squeeze(0).squeeze(0).unsqueeze(0)  # -> (1,H,E)
        else:
            raise RuntimeError(f"future_exo_cb returned unsupported shape={tuple(exo.shape)}")

        if exo.size(0) == 1 and B > 1:
            exo = exo.expand(B, -1, -1)
        elif exo.size(0) not in (1, B):
            raise RuntimeError(f"[EXO] batch mismatch: exo.shape[0]={exo.size(0)} vs B={B}")

        exo = exo.to(device)
        if not hasattr(self, "_logged_exo_shape"):
            print(f"[EXO-batch] exo normalized to shape={tuple(exo.shape)} (expect [B,H,E])")
            self._logged_exo_shape = True
        return exo

    def _resolve_future_exo(
            self,
            batch_future_exo: Optional[torch.Tensor],
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        미래 외생 변수 결정 로직.
        우선순위: 배치 데이터(Loader 제공) > Callback 생성 > None.
        """
        exo = None
        if batch_future_exo is not None:
            # (B,H,E) 가정. 빈 채널(E=0)이면 무시
            if torch.is_tensor(batch_future_exo) and batch_future_exo.ndim >= 2 and batch_future_exo.size(-1) > 0:
                exo = batch_future_exo.to(device)
        if exo is None:
            exo = self._normalize_future_exo_from_cb(x, y, device=device)

        if exo is not None:
            exo = torch.nan_to_num(exo, nan=0.0, posinf=1e6, neginf=-1e6)
            self._nan_stat("future_exo", exo)
        return exo

    def _nan_stat(self, name, t):
        """텐서 내 NaN/Inf 존재 여부 검사 및 로깅."""
        if not torch.is_tensor(t):
            return
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        finite_mask = torch.isfinite(t)
        if finite_mask.any():
            try:
                mx = t[finite_mask].abs().max().item()
            except Exception:
                mx = t[finite_mask].to(torch.float32).abs().max().item()
        else:
            mx = float("inf")
        if has_nan or has_inf:
            print(f"[NaN-{name}] has_nan={has_nan} has_inf={has_inf} max|x|={mx}")

    def _unpack_batch(self, batch) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[list], Optional[torch.Tensor], Optional[torch.Tensor], Optional[
            torch.Tensor]]:
        """
        가변 길이의 배치 튜플을 표준화된 6개 변수로 언패킹.
        Returns:
            x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat
        """
        part_ids = None
        future_exo_cont = None
        past_exo_cont = None
        past_exo_cat = None

        if len(batch) == 6:
            x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat = batch
        elif len(batch) == 5:
            x, y, part_ids, future_exo_cont, past_exo_cont = batch
        elif len(batch) == 3:
            x, y, part_ids = batch
        elif len(batch) == 2:
            x, y = batch
        else:
            raise RuntimeError(f"Unsupported batch tuple length: {len(batch)}")
        return x, y, part_ids, future_exo_cont, past_exo_cont, past_exo_cat

    def _forward_with_adapter(
            self,
            model,
            x,
            *,
            future_exo,
            past_exo_cont,
            past_exo_cat,
            part_ids,
            mode: str,
    ):
        """
        Adapter의 forward 메서드 호환성 처리 (레거시 vs 확장).
        """
        try:
            return self.adapter.forward(
                model,
                x,
                future_exo=future_exo,
                past_exo_cont=past_exo_cont,
                past_exo_cat=past_exo_cat,
                part_ids=part_ids,
                mode=mode,
            )
        except TypeError:
            return self.adapter.forward(
                model,
                x,
                future_exo=future_exo,
                mode=mode,
            )

    # ----------------- 에폭 루프 -----------------
    def _run_epoch(self, model, loader, *, train: bool):
        """
        단일 에폭(Epoch)에 대한 학습 또는 검증 루프 실행.

        기능:
        - 배치 데이터 언패킹 및 장치 이동.
        - 외생 변수 처리.
        - AMP(Mixed Precision) 기반 순전파 및 손실 계산.
        - 역전파 및 가중치 업데이트 (Train 모드 시).
        """
        device = self.cfg.device
        total = 0.0
        # 모델 모드(학습/평가) 전환
        model.train() if train else model.eval()

        with torch.set_grad_enabled(train):
            for batch in loader:
                # 1. 배치 데이터 구조 분해 및 텐서 장치 이동
                x, y, part_ids, fe_cont, pe_cont, pe_cat = self._unpack_batch(batch)
                x, y = x.to(device), y.to(device)

                # 입력 데이터 수치 안정성(NaN/Inf) 검사
                self._nan_stat("x(in)", x)
                self._nan_stat("y(in)", y)
                if pe_cont is not None: self._nan_stat("past_exo_cont", pe_cont)
                if pe_cat is not None: self._nan_stat("past_exo_cat", pe_cat)

                if train:
                    self.opt.zero_grad(set_to_none=True)

                # 2. 설정에 따른 미래 외생 변수 주입 전략 결정
                if self.use_exogenous_mode:
                    future_exo = self._resolve_future_exo(fe_cont, x, y, device=device)
                else:
                    future_exo = None

                # 3. AMP(Mixed Precision) 컨텍스트 내 순전파 및 손실 계산
                with autocast(
                        device_type=self.cfg.amp_device,
                        enabled=self.amp_enabled,
                        dtype=self.dtype if self.dtype is not None else "fp32",
                ):
                    # 어댑터를 경유한 모델 순전파 실행
                    pred = self._forward_with_adapter(
                        model,
                        x,
                        future_exo=future_exo,
                        past_exo_cont=(pe_cont.to(device) if torch.is_tensor(pe_cont) else None),
                        past_exo_cat=(pe_cat.to(device) if torch.is_tensor(pe_cat) else None),
                        part_ids=part_ids,
                        mode=("train" if train else "eval"),
                    )
                    self._nan_stat("pred", pred)

                    # 손실 함수 계산 (Validation 여부 반영)
                    loss = self.loss_comp.compute(pred, y, is_val=(not train))

                    # 디버깅: Spike Loss 상세 분석 (초기 배치 한정)
                    if self._get_spike_enabled():
                        self._dbg_spike_seen += 1
                        if self._dbg_spike_seen <= self._dbg_max_print:
                            self._debug_spike_breakdown(pred, y, is_val=(not train), tag=("train" if train else "eval"))

                    # 추가 손실 함수(Extra Loss) 합산
                    if self.extra_loss_fn is not None:
                        loss = loss + self.extra_loss_fn(x, pred, self.cfg)

                    self._nan_stat("loss_raw", loss)

                    # 정규화 손실(Regularization Loss) 합산
                    reg = self.adapter.reg_loss(model)
                    if reg is not None:
                        self._nan_stat("reg", reg)
                        loss = loss + reg

                # 4. 역전파 및 파라미터 최적화 (학습 모드 시)
                if train:
                    loss_t = self._to_tensor(loss, device)

                    # 손실 값 무결성 체크
                    if torch.isnan(loss_t):
                        self.logger("[Warn] NaN loss. step skipped.")
                        continue

                    # Scaler를 이용한 역전파 (Gradient Scaling)
                    self.scaler.scale(loss_t).backward()

                    # Gradient Clipping을 위한 Unscale 및 최적화
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()

                total += float(loss.detach())

        # 평균 손실 값 반환
        return total / max(1, len(loader))

    # ----------------- 학습 진입 -----------------
    def fit(self, model, train_loader, val_loader, *, tta_steps: int = 0):
        """
        전체 학습 프로세스 실행 관리.

        기능:
        - 옵티마이저/스케줄러 설정.
        - 에폭 반복 및 조기 종료(Early Stopping) 체크.
        - 검증 루프 및 TTA(Test-Time Adaptation) 수행.
        """
        device = self.cfg.device
        model.to(device)
        from modeling_module.training.optim import build_optimizer_and_scheduler

        # 최적화 도구 및 스케줄러 빌드
        self.opt, self.sched = build_optimizer_and_scheduler(model, self.cfg)
        self.scaler = GradScaler(self.cfg.amp_device)

        # 조기 종료(Early Stopping) 추적 변수 초기화
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        counter = 0

        # TTA(Test-Time Adaptation) 상태 초기화
        if self.adapter.uses_tta():
            self.adapter.tta_reset(model)

        for epoch in range(self.cfg.epochs):

            # 1. 학습 루프 실행
            train_loss = self._run_epoch(model, train_loader, train=True)

            # 2. 검증 루프 진입
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, part_ids, fe_cont, pe_cont, pe_cat = self._unpack_batch(batch)
                    x_val, y_val = x.to(device), y.to(device)

                    # 외생 변수 처리
                    if self.use_exogenous_mode:
                        future_exo = self._resolve_future_exo(fe_cont, x_val, y_val, device=device)
                    else:
                        future_exo = None

                    # TTA 적용 여부에 따른 분기 처리
                    if tta_steps > 0 and self.adapter.uses_tta():
                        # 테스트 데이터에 대한 모델 적응(Adaptation) 수행
                        loss = self.adapter.tta_adapt(model, x_val, y_val, steps=tta_steps)

                        if loss is None:  # TTA 실패 또는 지원 안함 -> 일반 평가로 전환
                            with autocast(
                                    device_type=self.cfg.amp_device,
                                    enabled=self.amp_enabled,
                                    dtype=self.dtype if self.dtype is not None else "fp32",
                            ):
                                pred = self._forward_with_adapter(
                                    model,
                                    x_val,
                                    future_exo=future_exo,
                                    past_exo_cont=(pe_cont.to(device) if torch.is_tensor(pe_cont) else None),
                                    past_exo_cat=(pe_cat.to(device) if torch.is_tensor(pe_cat) else None),
                                    part_ids=part_ids,
                                    mode="eval",
                                )
                                loss = self.loss_comp.compute(pred, y_val, is_val=True)
                                if self.extra_loss_fn is not None:
                                    loss = loss + self.extra_loss_fn(x_val, pred, self.cfg)
                                loss = float(loss.detach())
                        val_total += loss
                    else:
                        # 일반 평가 (Standard Validation)
                        with autocast(
                                device_type=self.cfg.amp_device,
                                enabled=self.amp_enabled,
                                dtype=self.dtype if self.dtype is not None else "fp32",
                        ):
                            pred = self._forward_with_adapter(
                                model,
                                x_val,
                                future_exo=future_exo,
                                past_exo_cont=(pe_cont.to(device) if torch.is_tensor(pe_cont) else None),
                                past_exo_cat=(pe_cat.to(device) if torch.is_tensor(pe_cat) else None),
                                part_ids=part_ids,
                                mode="eval",
                            )
                            vloss = self.loss_comp.compute(pred, y_val, is_val=True)

                            if self.extra_loss_fn is not None:
                                vloss = vloss + self.extra_loss_fn(x_val, pred, self.cfg)
                            val_total += float(vloss.detach())

                    # 메트릭 계산 (선택 사항)
                    if self.metrics_fn:
                        _ = self.metrics_fn(pred, y_val)

            # 에폭별 검증 손실 집계 및 스케줄러 갱신
            val_loss = val_total / max(1, len(val_loader))
            self.sched.step()

            # 3. 조기 종료(Early Stopping) 체크 및 최적 모델 저장
            if val_loss < best_loss:
                best_loss, counter = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= self.cfg.patience:
                    self.logger(f"Early stopping at epoch {epoch + 1}")
                    break

            cur_lr = self.sched.get_last_lr()[0]
            self.logger(
                f"Epoch {epoch + 1}/{self.cfg.epochs} | LR {cur_lr:.6f} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        # 학습 종료 후 최적 가중치 복원
        model.load_state_dict(best_state)
        return model