import os
import inspect
from typing import Any, Dict, List, Sequence, Optional, Callable, Tuple, Union
import numpy as np
import torch

try:
    import polars as pl
except ImportError:
    pl = None

DEBUG_FCAST = True


# -------------------------------------------------------------------------
# Device helpers
# -------------------------------------------------------------------------
def _to_device_any(obj: Any, device: torch.device) -> Any:
    """
    임의의 객체(텐서 및 컨테이너)를 대상 장치(Device)로 재귀적 이동.

    기능:
    - 텐서, 리스트, 튜플, 딕셔너리 등 중첩된 데이터 구조를 순회하며 장치 동기화.
    - CPU/CUDA 장치 불일치(Device Mismatch)로 인한 런타임 오류 방지.
    - 텐서가 아닌 객체(int, str 등)는 변경 없이 원본 반환.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device_any(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device_any(v, device) for k, v in obj.items()}
    return obj


def _infer_d_future_expected(model: torch.nn.Module) -> Optional[int]:
    """
    모델이 기대하는 미래 외생 변수(Future Exo)의 차원(Dimension) 추론.

    기능:
    - 다양한 모델 아키텍처(PatchTST, Titan 등)의 구현 차이를 고려한 최선(Best-effort) 탐색.
    - 모델 속성, 설정(Config), 헤드(Head) 순으로 `d_future` 또는 `exo_dim` 속성 확인.

    반환:
        Optional[int]: 추론된 차원 (0 이상), 추론 불가 시 None 반환.
    """
    # 1. 모델 직접 속성 확인
    for attr in ('d_future', 'exo_dim'):
        if hasattr(model, attr):
            try:
                v = int(getattr(model, attr))
                if v >= 0:
                    return v
            except Exception:
                pass

    # 2. Config 객체 속성 확인
    cfg = getattr(model, 'cfg', None)
    if cfg is not None:
        for attr in ('d_future', 'exo_dim'):
            if hasattr(cfg, attr):
                try:
                    v = int(getattr(cfg, attr))
                    if v >= 0:
                        return v
                except Exception:
                    pass

    # 3. Head 모듈 속성 확인
    head = getattr(model, 'head', None)
    if head is not None and hasattr(head, 'd_future'):
        try:
            v = int(getattr(head, 'd_future'))
            if v >= 0:
                return v
        except Exception:
            pass

    return None


# -------------------------------------------------------------------------
# Standalone Helpers (Stateless)
# -------------------------------------------------------------------------
def _safe_forward(model: torch.nn.Module, x: torch.Tensor, **kwargs):
    """
    모델의 forward 시그니처를 분석하여 호환되는 인자만 전달하는 안전 호출 래퍼.

    기능:
    - `inspect` 모듈을 통한 forward 메서드 파라미터 검사.
    - 지원하지 않는 키워드 인자(kwargs)를 필터링하여 `TypeError` 방지.
    - 실패 시 전체 인자 전달 또는 기본 호출(x만 전달)로 폴백(Fallback) 처리.
    """
    try:
        # 시그니처 기반 인자 필터링
        sig = inspect.signature(model.forward)
        allowed = set(sig.parameters.keys())
        fkwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return model(x, **fkwargs)
    except Exception:
        # 필터링 실패 시 폴백 메커니즘
        try:
            return model(x, **kwargs)
        except TypeError:
            return model(x)

def _first_usable(out: Any) -> Any:
    """
    모델의 다양한 출력 형식(Tuple, List 등)에서 유효한 첫 번째 데이터 추출.

    기능:
    - 시퀀스형 출력의 경우, 내부 요소를 순회하여 Tensor 또는 Dict 타입을 우선 반환.
    - 주요 결과값이 튜플 등에 래핑되어 있을 때 메인 데이터를 식별하여 꺼내는 역할.
    """
    if isinstance(out, (tuple, list)):
        for t in out:
            if torch.is_tensor(t) or isinstance(t, dict):
                return t
        return out[0] if out else out
    return out


def _normalize_point_to_BH(y_any: Any, B: int, H_hint: Optional[int] = None) -> torch.Tensor:
    """
    다양한 모델 출력 형태를 표준 점 예측 텐서 [B, H]로 정규화.

    기능:
    - Dict/Tuple/List 등 다양한 컨테이너 래핑 해제.
    - 분위수(Quantile) 출력 시 중앙값(Median) 추출.
    - 3차원 텐서 [B, H, C] 또는 [B, C, H] 형태를 H_hint를 이용해 [B, H]로 차원 축소.
    """
    # 1. 래핑 해제 (Tuple/List -> Tensor/Dict)
    y_any = _first_usable(y_any)

    # 2. 딕셔너리 처리: 'point' > 'q'(중앙값) > 첫 번째 값 순으로 탐색
    if isinstance(y_any, dict):
        if "point" in y_any:
            y_any = y_any["point"]
        elif "q" in y_any:
            q = y_any["q"]
            if torch.is_tensor(q):
                # [B, H, Q] 형태 가정: Q>=3이면 중간값(인덱스 1), 아니면 0번 인덱스
                if q.dim() == 3 and q.size(-1) >= 3:
                    y_any = q[..., 1]
                else:
                    y_any = q[..., 0]
            elif isinstance(q, dict) and "q50" in q:
                y_any = q["q50"]
        else:
            y_any = y_any[next(iter(y_any))]

    # 3. 텐서 차원 정규화
    if torch.is_tensor(y_any):
        y = y_any
        if y.dim() == 1:
            return y.view(B, -1)
        if y.dim() == 2:
            return y

        # 3차원 처리: [B, ?, ?] -> [B, H]
        if y.dim() == 3:
            d1, d2 = y.size(1), y.size(2)

            # 힌트(Horizon 길이)가 주어지면 해당 차원을 유지하고 나머지 차원 축소
            if H_hint is not None:
                if d1 == H_hint and d2 != H_hint:
                    return y[:, :, 0]
                if d2 == H_hint and d1 != H_hint:
                    return y[:, 0, :]
                if d1 == H_hint and d2 == H_hint:
                    return y[:, :, 0]

            # 힌트가 없으면 채널 수(d2)를 보고 판단 (3채널이면 분위수로 보고 중앙값 선택)
            if d2 in (1, 3):
                return y[:, :, 1] if d2 == 3 else y[:, :, 0]
            return y[:, 0, :]

        return y.reshape(B, -1)

    raise RuntimeError(f"Unsupported point output type={type(y_any)}")


def _extract_quantile_block(out_any: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    모델 출력에서 주요 분위수(10%, 50%, 90%)를 추출하여 (q10, q50, q90) 튜플로 반환.

    기능:
    - Dictionary 출력 처리: "q10", "q50", "q90" 키 직접 접근 또는 중첩된 "q" 키 탐색.
    - Tensor 출력 처리: 3차원 텐서 [B, Q, H] 또는 [B, H, Q] 형태 자동 인식.
    - 분위수 개수(3, 5, 9)에 따른 적절한 인덱스 선택 로직 적용.
    """
    out_any = _first_usable(out_any)

    # 1. Dictionary 기반 출력 처리
    if isinstance(out_any, dict):
        if all(k in out_any for k in ("q10", "q50", "q90")):
            q10, q50, q90 = out_any["q10"], out_any["q50"], out_any["q90"]

            # 텐서 차원 정규화 헬퍼 (Batch, Horizon 형태로 변환)
            def _S(t: torch.Tensor):
                if t.dim() == 1: return t.unsqueeze(0)
                if t.dim() == 2: return t
                if t.dim() == 3 and t.size(-1) == 1: return t.squeeze(-1)
                if t.dim() == 3 and t.size(1) == 1: return t[:, 0, :]
                return t.reshape(t.size(0), -1)

            return _S(q10), _S(q50), _S(q90)

        # 중첩된 "q" 키 처리
        if "q" in out_any:
            q = out_any["q"]
            if isinstance(q, dict) and all(k in q for k in ("q10", "q50", "q90")):
                return _extract_quantile_block(q)
            out_any = q

    # 2. Tensor 기반 출력 처리
    if not torch.is_tensor(out_any):
        raise RuntimeError(f"Unsupported quantile output type={type(out_any)}")

    q3d = out_any
    if q3d.dim() != 3:
        raise RuntimeError(f"Quantile output must be 3D, got {tuple(q3d.shape)}")

    # (B, Q, H) 형태 추론
    if q3d.shape[1] in (3, 5, 9):
        Qn = q3d.shape[1]
        # 분위수 개수에 따른 인덱스 매핑 (예: 5개일 경우 1,2,3번 인덱스 선택 등 정책 반영)
        i10, i50, i90 = (0, 1, 2) if Qn == 3 else (1, 2, 3) if Qn == 5 else (1, 4, 7)
        return q3d[:, i10, :], q3d[:, i50, :], q3d[:, i90, :]

    # (B, H, Q) 형태 추론
    if q3d.shape[2] in (3, 5, 9):
        Qn = q3d.shape[2]
        i10, i50, i90 = (0, 1, 2) if Qn == 3 else (1, 2, 3) if Qn == 5 else (1, 4, 7)
        return q3d[:, :, i10], q3d[:, :, i50], q3d[:, :, i90]

    raise RuntimeError(f"Cannot infer quantile axis from shape {tuple(q3d.shape)}")


# -------------------------------------------------------------------------
# Main Class Refactored
# -------------------------------------------------------------------------
class DMSForecaster:
    """
    Unified Forecaster for DMS + IMS extension.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            *,
            target_channel: int = 0,
            fill_mode: str = "copy_last",
            ttm: Optional[object] = None,
            # Guards Config
            use_winsor: bool = False,
            use_multi_guard: bool = False,
            use_dampen: bool = False,
            winsor_q: Tuple[float, float] = (0.05, 0.95),
            winsor_mul: float = 2.0,
            winsor_growth: float = 1.2,
            max_step_up: float = 0.05,
            max_step_down: float = 0.40,
            damp: float = 0.30,
            # Quantile Specific Config
            quantile_feed: str = "q50",  # 'q10' or 'q50' for feedback
    ):
        """
        예측 모델 래퍼(Wrapper) 초기화 및 후처리 안전 장치(Guard) 설정.

        기능:
        - PyTorch 모델 및 타겟 채널, 결측 보간(Fill Mode) 방식 설정.
        - 이상치 제어(Winsorization) 및 급격한 변동 제한(Step Guard)을 위한 파라미터 구성.
        - 분위수 예측(Quantile) 시 피드백 루프에 사용할 기준 분위수(q10/q50) 지정.
        """
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.ttm = ttm

        # Guard config (안전 장치 설정 딕셔너리 구성)
        self.guard_cfg = dict(
            use_winsor=use_winsor, use_multi_guard=use_multi_guard, use_dampen=use_dampen,
            winsor_q=winsor_q, winsor_mul=winsor_mul, winsor_growth=winsor_growth,
            max_step_up=max_step_up, max_step_down=max_step_down, damp=damp
        )
        self.quantile_feed = quantile_feed
        self.global_t0 = 0
        # ---------------------------------------------------------------------

    # Public Entry Point
    # ---------------------------------------------------------------------

    def predict(
            self,
            x_init: torch.Tensor,
            *,
            horizon: int,
            device: Union[str, torch.device, None] = None,
            mode: str = "eval",
            # Optional Exogenous / IDs
            part_ids: Optional[Sequence[Any]] = None,
            past_exo_cont: Optional[torch.Tensor] = None,
            past_exo_cat: Optional[torch.Tensor] = None,
            future_exo_batch: Optional[torch.Tensor] = None,
            future_exo_cb: Optional[Callable[[int, int, torch.device], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        통합 예측 메서드 (Unified Prediction Entry Point).

        기능:
        - 입력 데이터 및 보조 데이터(Exo, IDs)의 장치 동기화.
        - 미래 외생 변수(Future Exo) 소스 결정 (Batch vs Callback) 및 검증.
        - 모델 시그니처에 맞는 인자 필터링 및 전달.
        - 모델 출력 분석을 통한 예측 전략(Quantile vs Point) 자동 분기.
        """
        device = torch.device(device or next(self.model.parameters()).device)
        self.model.to(device).eval()

        # 보조 텐서들을 모델과 동일한 장치로 이동하여 연산 호환성 확보
        part_ids = _to_device_any(part_ids, device)
        past_exo_cont = _to_device_any(past_exo_cont, device)
        past_exo_cat = _to_device_any(past_exo_cat, device)
        future_exo_batch = _to_device_any(future_exo_batch, device)

        # 입력 시계열 전처리 (복사 및 차원 보정)
        x_raw = x_init.to(device).float().clone()
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(-1)
        B = x_raw.size(0)

        # --------------------------------------------------------------
        # 1) 미래 외생 변수(Future Exogenous) 소스 결정 및 정책 적용
        # --------------------------------------------------------------
        # 정책:
        # - 모델이 d_future == 0을 기대하면, 외생 변수를 전달하지 않음 (강제 비활성화).
        # - 모델이 d_future > 0을 기대하면, 유효한 소스(Batch 또는 Callback)가 필수이며 차원 일치 검증 수행.
        d_future_expected = _infer_d_future_expected(self.model)

        cb_final = future_exo_cb

        # 사용자가 텐서 형태로 Future Exo를 전달한 경우, 이를 동적 콜백으로 래핑(Wrapping)
        if torch.is_tensor(future_exo_batch):
            exb = future_exo_batch

            if exb.dim() == 2:
                # (H, E) 형태: 모든 샘플에 공통 적용되는 프로파일
                def _cb_from_batch(t0: int, h_req: int, dev: torch.device):
                    s, e = int(t0), int(t0) + int(h_req)
                    Htot = exb.size(0)
                    if Htot >= e:
                        return exb[s:e, :].detach().to(dev)
                    if Htot <= s:
                        return exb[-1:, :].expand(h_req, -1).detach().to(dev)
                    tail = exb[s:, :]
                    pad = exb[-1:, :].expand(e - Htot, -1)
                    return torch.cat([tail, pad], dim=0).detach().to(dev)

                cb_final = _cb_from_batch

            elif exb.dim() == 3:
                # (B, H, E) 형태: 샘플별 개별 프로파일
                if exb.size(0) not in (1, B):
                    raise RuntimeError(
                        f"future_exo_batch has incompatible batch dim: got {exb.size(0)}, expected 1 or {B}."
                    )

                def _cb_from_batch_b(t0: int, h_req: int, dev: torch.device):
                    s, e = int(t0), int(t0) + int(h_req)
                    Htot = exb.size(1)
                    if Htot >= e:
                        out = exb[:, s:e, :]
                    elif Htot <= s:
                        out = exb[:, -1:, :].expand(exb.size(0), h_req, -1)
                    else:
                        tail = exb[:, s:, :]
                        pad = exb[:, -1:, :].expand(exb.size(0), e - Htot, -1)
                        out = torch.cat([tail, pad], dim=1)

                    # (1, H, E)인 경우 배치 크기(B)에 맞춰 확장
                    if out.size(0) == 1 and B > 1:
                        out = out.expand(B, -1, -1)
                    return out.detach().to(dev)

                cb_final = _cb_from_batch_b

        # 내부 시그니처 (t0, H) -> Tensor 로 최종 래핑
        self.future_exo_cb = None
        if cb_final is not None:
            self.future_exo_cb = lambda t, h: cb_final(t, h, device)

        # 모델의 기대 차원(d_future)과 실제 제공 여부 간의 정합성 검사 (Fail-Fast)
        if d_future_expected is not None:
            if d_future_expected <= 0:
                # 모델이 외생 변수를 사용하지 않음 -> 강제 비활성화
                self.future_exo_cb = None
            else:
                if self.future_exo_cb is None:
                    raise RuntimeError(
                        f"Model expects future_exo dim d_future={d_future_expected}, "
                        f"but neither future_exo_batch nor future_exo_cb was provided."
                    )

        # 2. 순전파 인자(Kwargs) 구성
        fwd_kwargs = {}
        # IDs (모델에 따라 part_ids 또는 기타 식별자 사용)
        if part_ids is not None:
            fwd_kwargs["part_ids"] = part_ids

        # 과거 외생 변수 (별칭 포함하여 전달, _safe_forward가 필터링함)
        if past_exo_cont is not None:
            fwd_kwargs["past_exo_cont"] = past_exo_cont
            fwd_kwargs["pe_cont"] = past_exo_cont
        if past_exo_cat is not None:
            fwd_kwargs["past_exo_cat"] = past_exo_cat
            fwd_kwargs["pe_cat"] = past_exo_cat

        if mode is not None:
            fwd_kwargs["mode"] = mode

        # 3. 모델 출력 타입 및 Horizon 탐지 (Probe 실행)
        H_hint = int(getattr(self.model, "horizon", getattr(self.model, "output_horizon", 0)) or 0)
        probe_H = int(horizon if H_hint == 0 else max(1, H_hint))  # Probe용 Exo 길이 결정

        exo_probe = None
        if self.future_exo_cb is not None:
            ex = self.future_exo_cb(0, probe_H)

            # Future Exo 차원 및 형태 검증
            if d_future_expected is not None and d_future_expected > 0:
                if ex is None:
                    raise RuntimeError(f"future_exo_cb returned None, but d_future={d_future_expected} is required")
                if ex.ndim not in (2, 3):
                    raise RuntimeError(
                        f"future_exo_cb must return (H,E) or (B,H,E); got shape={tuple(ex.shape)}"
                    )
                if int(ex.shape[-1]) != int(d_future_expected):
                    raise RuntimeError(
                        f"future_exo dim mismatch: got E={int(ex.shape[-1])}, expected d_future={int(d_future_expected)}"
                    )

            # (B, H, E) 형태로 정규화
            if ex.ndim == 2:
                # (H, E) -> (B, H, E) 확장
                exo_probe = ex.to(device).unsqueeze(0).expand(B, -1, -1)
            else:
                # (B, H, E)
                exo_probe = ex.to(device)

        # Probe Forward 실행 (Gradient 비활성화)
        with torch.no_grad():
            out0 = _safe_forward(
                self.model,
                x_raw,
                future_exo=exo_probe,
                fe_cont=exo_probe,
                **fwd_kwargs,
            )

        # 4. 출력 타입에 따른 예측 전략 분기 (Quantile vs Point)
        is_quantile = False
        try:
            _extract_quantile_block(out0)
            is_quantile = True
        except Exception:
            is_quantile = False

        if is_quantile:
            return self._predict_quantile_strategy(x_raw, out0, horizon, device, fwd_kwargs)
        else:
            return self._predict_point_strategy(x_raw, out0, horizon, device, fwd_kwargs, probe_H)

    # ---------------------------------------------------------------------
    # Internal Strategies
    # ---------------------------------------------------------------------
    def _predict_point_strategy(self, x_raw, out0, horizon, device, fwd_kwargs, probe_H):
        """
        점 예측(Point Prediction) 전략 실행 및 결과 후처리.

        기능:
        - 초기 출력 정규화 및 모델의 실제 출력 길이(Hm) 확인.
        - 요청된 Horizon 길이에 따른 분기 처리:
          1) Case A (Horizon <= Hm): 단순 슬라이싱(Slicing) 반환.
          2) Case B (Horizon > Hm): 자기회귀(Autoregressive) 방식을 통한 예측 구간 확장.
        """
        B = x_raw.size(0)
        y0 = _normalize_point_to_BH(out0, B, H_hint=probe_H)
        Hm = int(y0.size(1))

        if int(horizon) <= Hm:
            # Case A: Simple Slice (모델 출력이 충분히 긴 경우)
            y_hat = y0[:, :int(horizon)]
        else:
            # Case B: Autoregressive Extension (모델 출력이 짧아 반복 예측 필요)
            y_hat = self._impl_point_DMS_to_IMS(
                x_init=x_raw,
                horizon=int(horizon),
                model_horizon=Hm,  # <--- PASS Hm HERE
                device=device,
                fwd_kwargs=fwd_kwargs
            )

        return {"point": y_hat.detach().cpu().numpy().reshape(-1)}

    def _predict_quantile_strategy(self, x_raw, out0, horizon, device, fwd_kwargs):
        """
        분위수 예측(Quantile Prediction) 전략 실행 및 결과 패킹.

        기능:
        - 초기 출력에서 주요 분위수(10%, 50%, 90%) 블록 추출.
        - 모델의 출력 길이(Hm)와 요청 Horizon 비교를 통한 분기 처리:
          1) Case A: 단순 슬라이싱 (Simple Slice).
          2) Case B: 자기회귀 확장을 통한 장기 예측 (Autoregressive Extension).
        - q50(중앙값)을 점 예측(Point) 값으로 매핑하여 반환.
        """
        q10_blk, q50_blk, q90_blk = _extract_quantile_block(out0)
        Hm = int(q50_blk.size(1))

        if int(horizon) <= Hm:
            # Case A: Simple Slice (모델 출력이 충분할 경우)
            q10 = q10_blk[:, :int(horizon)]
            q50 = q50_blk[:, :int(horizon)]
            q90 = q90_blk[:, :int(horizon)]
        else:
            # Case B: Autoregressive Extension (반복 예측 수행)
            # Quantile impl extracts Hm from block, but better to pass it if structured like point
            # Currently Quantile impl handles block extraction inside, so we rely on block shape.
            q10, q50, q90 = self._impl_quantile_DMS_to_IMS(
                x_init=x_raw,
                horizon=int(horizon),
                model_horizon=Hm,  # <--- PASS Hm HERE
                device=device,
                fwd_kwargs=fwd_kwargs
            )

        return {
            "q10": q10.detach().cpu().numpy().reshape(-1),
            "q50": q50.detach().cpu().numpy().reshape(-1),
            "q90": q90.detach().cpu().numpy().reshape(-1),
            "point": q50.detach().cpu().numpy().reshape(-1),
        }

    # ---------------------------------------------------------------------
    # Implementation: Point Autoregression (Logic from point_DMS_to_IMS)
    # ---------------------------------------------------------------------
    def _impl_point_DMS_to_IMS(
            self, x_init: torch.Tensor, horizon: int, model_horizon: int, device: torch.device, fwd_kwargs: Dict
    ) -> torch.Tensor:
        """
        점 예측(Point Prediction)을 위한 하이브리드 DMS-to-IMS 전략 실행.

        기능:
        - DMS(Direct Multi-step): 모델의 기본 출력 길이(Hm)만큼 초기 블록 확보.
        - IMS(Iterative Multi-step): 초기 블록 소진 후, 1-step씩 자기회귀(Autoregressive) 방식으로 확장.
        - 각 스텝마다 안전 장치(Guard) 적용 및 입력 윈도우(Context) 갱신 수행.
        """
        x_raw = x_init.clone()
        B, L, C = x_raw.shape
        Hm = model_horizon  # 전달받은 모델의 정확한 Horizon 사용

        # 헬퍼: 현재 시점(step_offset)에 맞춰 외생 변수를 준비하고 모델을 호출하여 점 예측 수행
        def _call_point(xr, need_h, step_offset):
            exo = None
            if self.future_exo_cb is not None:
                # 콜백을 통해 필요한 길이만큼 미래 외생 변수 생성
                t0 = self.global_t0 + step_offset
                ex = self.future_exo_cb(t0, need_h).to(xr.device)

                # 배치 차원 브로드캐스팅 처리
                if ex.ndim == 2:
                    exo = ex.unsqueeze(0).expand(B, -1, -1)
                elif ex.ndim == 3:
                    if ex.size(0) == 1 and B > 1:
                        exo = ex.expand(B, -1, -1)
                    elif ex.size(0) == B:
                        exo = ex
                    else:
                        raise RuntimeError(f"future_exo batch dim mismatch: got {ex.size(0)}, expected 1 or {B}")
                else:
                    raise RuntimeError(f"future_exo must be (H,E) or (B,H,E), got shape={tuple(ex.shape)}")

            # 모델 순전파 및 출력 정규화
            out = _safe_forward(self.model, xr, future_exo=exo, fe_cont=exo, **fwd_kwargs)
            return _normalize_point_to_BH(out, B, H_hint=need_h)

        # 1. 초기 블록 예측 (DMS 단계) - 모델의 전체 Horizon만큼 예측
        y_block_raw = _call_point(x_raw, Hm, 0)

        if DEBUG_FCAST:
            print(f"[DMS] Point AR Start. Hm={Hm}, H_req={horizon}")

        outputs = []
        use_len = min(Hm, horizon)

        # DMS 구간 처리: 초기 예측 블록을 스텝별로 순회하며 가드 적용 및 컨텍스트 갱신
        for t in range(use_len):
            if self.ttm: self.ttm.add_context(x_raw)
            y_step = y_block_raw[:, t]
            y_adj = self._apply_guards(x_raw, y_step)
            outputs.append(y_adj.unsqueeze(1))
            x_raw = self._prepare_next_input(x_raw, y_adj)

        # IMS 구간 처리: 모델 Horizon을 초과하는 구간에 대해 1-step 자기회귀 반복
        if horizon > Hm:
            for t in range(horizon - Hm):
                if self.ttm: self.ttm.add_context(x_raw)

                # 다음 스텝 예측 (가장 첫 번째 시점 값만 사용)
                y_full = _call_point(x_raw, Hm, step_offset=(use_len + t))
                y_step = y_full[:, 0]
                y_adj = self._apply_guards(x_raw, y_step)
                outputs.append(y_adj.unsqueeze(1))
                x_raw = self._prepare_next_input(x_raw, y_adj)

        return torch.cat(outputs, dim=1)

    # ---------------------------------------------------------------------
    # Implementation: Quantile Autoregression (Logic from quantile_DMS_to_IMS)
    # ---------------------------------------------------------------------
    def _impl_quantile_DMS_to_IMS(
            self, x_init: torch.Tensor, horizon: int, model_horizon: int, device: torch.device, fwd_kwargs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        분위수 예측(Quantile Prediction)을 위한 하이브리드 DMS-to-IMS 전략 실행.

        기능:
        - 점 예측과 유사하게 초기 Horizon(Hm)까지는 DMS로, 초과분은 IMS(자기회귀)로 처리.
        - 자기회귀 시 입력으로 재주입(Feedback)할 값(q10 또는 q50)을 설정에 따라 선택.
        - 성장 제한(Growth Guard)을 적용하여 이상치 전파 방지.
        """
        x_raw = x_init.clone()
        B, L, C = x_raw.shape
        Hm = model_horizon

        # 헬퍼: 특정 시점의 외생 변수를 준비하고 모델을 호출하여 분위수 블록 반환
        def _call_quantile(xr, step_offset):
            exo = None
            if self.future_exo_cb is not None:
                # 현재 오프셋에 맞춰 외생 변수 생성
                ex = self.future_exo_cb(step_offset, Hm).to(xr.device)

                # 배치 차원 브로드캐스팅
                if ex.ndim == 2:
                    exo = ex.unsqueeze(0).expand(B, -1, -1)
                elif ex.ndim == 3:
                    if ex.size(0) == 1 and B > 1:
                        exo = ex.expand(B, -1, -1)
                    elif ex.size(0) == B:
                        exo = ex
                    else:
                        raise RuntimeError(f"future_exo batch dim mismatch: got {ex.size(0)}, expected 1 or {B}")
                else:
                    raise RuntimeError(f"future_exo must be (H,E) or (B,H,E), got shape={tuple(ex.shape)}")

            out = _safe_forward(self.model, xr, future_exo=exo, fe_cont=exo, **fwd_kwargs)
            return _extract_quantile_block(out)

        # 1. 초기 블록 예측 (DMS 단계 시작)
        q10_blk, q50_blk, q90_blk = _call_quantile(x_raw, 0)
        use_len = min(horizon, Hm)

        q10_seq, q50_seq, q90_seq = [], [], []

        # DMS 구간 처리: 초기 블록을 순회하며 피드백 및 입력 갱신
        for t in range(use_len):
            q10, q50, q90 = q10_blk[:, t], q50_blk[:, t], q90_blk[:, t]

            # 피드백 값 결정 (보수적 예측을 위해 q10을 사용할지, 중앙값 q50을 사용할지 선택)
            y_feed = q10 if self.quantile_feed == "q10" else q50
            y_next = self._apply_growth_guard(x_raw, y_feed)

            q10_seq.append(q10.unsqueeze(1))
            q50_seq.append(q50.unsqueeze(1))
            q90_seq.append(q90.unsqueeze(1))

            x_raw = self._prepare_next_input(x_raw, y_next)

        # IMS 구간 처리: 모델 Horizon 초과분에 대한 반복적 자기회귀 예측
        if horizon > Hm:
            for k in range(horizon - Hm):
                offset = use_len + k
                qb10, qb50, qb90 = _call_quantile(x_raw, offset)
                # 첫 번째 스텝만 취하여 다음 입력으로 활용
                q10, q50, q90 = qb10[:, 0], qb50[:, 0], qb90[:, 0]

                y_feed = q10 if self.quantile_feed == "q10" else q50
                y_next = self._apply_growth_guard(x_raw, y_feed)

                q10_seq.append(q10.unsqueeze(1))
                q50_seq.append(q50.unsqueeze(1))
                q90_seq.append(q90.unsqueeze(1))

                x_raw = self._prepare_next_input(x_raw, y_next)

        return torch.cat(q10_seq, 1), torch.cat(q50_seq, 1), torch.cat(q90_seq, 1)

    # ---------------------------------------------------------------------
    # Internal Logic: Input Preparation & Guards
    # ---------------------------------------------------------------------
    def _prepare_next_input(self, x_raw, y_next_val):
        """
        자기회귀(Autoregressive) 예측을 위한 다음 입력 시퀀스 갱신.

        기능:
        - 슬라이딩 윈도우(Sliding Window) 방식으로 가장 오래된 시점을 제거하고 예측값을 끝에 추가.
        - 다변량 시계열의 경우 타겟 채널(Target Channel) 값만 업데이트하고 나머지는 유지(또는 복사).
        """
        B, L, C = x_raw.shape
        y_r = y_next_val.reshape(B, 1, 1)

        if C == 1:
            new_token = y_r
        else:
            last = x_raw[:, -1:, :].clone()
            if self.fill_mode == 'zeros':
                new_token = torch.zeros_like(last)
            else:
                new_token = last
            new_token[:, 0, self.target_channel] = y_r[:, 0, 0]

        return torch.cat([x_raw[:, 1:, :], new_token], dim=1)

    def _apply_guards(self, x_raw, y_step):
        """
        점 예측(Point Forecast) 결과에 대한 후처리 안전 장치(Guard) 적용.

        기능:
        - 설정된 옵션에 따라 Winsorization, 급격한 변동 제한(Multi-guard), 댐핑(Damping) 순차 적용.
        """
        cfg = self.guard_cfg
        hist_raw = x_raw[:, :, self.target_channel]
        last_raw = hist_raw[:, -1]
        y_adj = y_step.float()

        if cfg['use_winsor']:
            y_adj = self._winsorize_clamp_raw(hist_raw, y_adj, **cfg)
        if cfg['use_multi_guard']:
            y_adj = self._guard_multiplicative_raw(last_raw, y_adj, **cfg)
        if cfg['use_dampen']:
            y_adj = self._dampen_to_last_raw(last_raw, y_adj, **cfg)
        return y_adj

    def _apply_growth_guard(self, x_raw, y_step):
        """
        분위수 예측(Quantile Forecast)을 위한 성장률 기반 가드 적용.
        (현재 구현은 점 예측 가드 로직을 공유함).
        """
        # Simplification: use same point guards
        return self._apply_guards(x_raw, y_step)

    def _winsorize_clamp_raw(self, hist_raw, y, winsor_q, winsor_mul, winsor_growth, **kwargs):
        """
        이력 데이터의 분포(Quantile)를 기반으로 한 이상치 제어(Winsorization) 및 상한 클램핑.

        기능:
        - 과거 데이터의 분포($q_{lo}, q_{hi}$)를 계산하여 허용 범위 설정.
        - 직전 값 대비 허용 성장률(winsor_growth)과 비교하여 최종 상한 결정.
        """
        last = hist_raw[:, -1]
        # NaN/Inf 처리: 유효하지 않은 값은 직전 값으로 대체
        hist_safe = torch.where(torch.isfinite(hist_raw), hist_raw, last.unsqueeze(1))

        q_lo = torch.quantile(hist_safe, winsor_q[0], dim=1)
        q_hi = torch.quantile(hist_safe, winsor_q[1], dim=1)

        cap_quant = q_hi * winsor_mul
        cap_growth = torch.where(last > 0, last * winsor_growth, cap_quant)
        max_cap = torch.minimum(cap_quant, cap_growth)

        y = torch.clamp(y, max=max_cap)
        return y

    def _guard_multiplicative_raw(self, last_raw, y, max_step_up, max_step_down, **kwargs):
        """
        직전 값 대비 급격한 변동(Step Change) 제한.

        기능:
        - 로그 비율(Log Ratio)을 계산하여 상승/하락폭을 설정된 임계값 내로 제한.
        - 비현실적인 급등이나 급락을 방지하여 예측 안정성 확보.
        """
        eps = 1e-6
        last_safe = torch.clamp(last_raw, min=eps)
        y_safe = torch.clamp(y, min=eps)

        ratio = y_safe / last_safe
        log_ratio = torch.log(ratio)

        log_min = torch.log(torch.tensor(1.0 - max_step_down, device=y.device))
        log_max = torch.log(torch.tensor(1.0 + max_step_up, device=y.device))

        log_ratio = torch.clamp(log_ratio, min=log_min, max=log_max)
        return last_safe * torch.exp(log_ratio)

    def _dampen_to_last_raw(self, last_raw, y, damp, **kwargs):
        """
        직전 관측값과의 가중 평균을 통한 예측값 평활화(Smoothing/Damping).

        기능:
        - 예측값(y)과 직전 값(last_raw)을 `damp` 비율로 혼합하여 변동성 완화.
        """
        if damp <= 0.0: return y
        return (1.0 - damp) * last_raw + damp * y


# -------------------------------------------------------------------------
# Export Function (Unchanged except imports)
# -------------------------------------------------------------------------
def _unpack_batch_for_export(batch: Any) -> Dict[str, Any]:
    """
    내보내기(Export) 및 로깅을 위한 배치 데이터 구조화 및 딕셔너리 변환.

    기능:
    - 튜플/리스트 형태의 배치를 명시적인 키(Key)를 가진 딕셔너리로 매핑.
    - 배치 길이에 따라 가변적인 요소(Part IDs, 외생 변수 등)의 유동적 처리 및 결측(None) 할당.
    """

    x = batch[0]
    y = batch[1]
    part_ids = batch[2] if len(batch) >= 3 else None
    future_exo = batch[3] if len(batch) >= 4 else None
    past_exo_cont = batch[4] if len(batch) >= 5 else None
    past_exo_cat = batch[5] if len(batch) >= 6 else None

    return dict(x=x, y=y, part_ids=part_ids, future_exo=future_exo,
                past_exo_cont=past_exo_cont, past_exo_cat=past_exo_cat)


def _to_py_id(v) -> str:
    if v is None: return "NA"
    if torch.is_tensor(v):
        return str(v.item()) if v.numel() == 1 else str(v.tolist())
    return str(v)


def forecast_to_parquet(
        model_dict: Dict[str, torch.nn.Module],
        loader,
        *,
        parquet_path: str,
        horizon: int,
        freq: str = "unknown",
        mode: str = "infer",
        plan_dt: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_channel: int = 0,
        fill_mode: str = "copy_last",
        max_samples: int = 200,
        future_exo_cb: Optional[Callable] = None,
):
    """
    여러 모델의 예측 결과를 일괄 생성하여 Parquet 파일로 저장.

    기능:
    - 입력된 모델 딕셔너리(`model_dict`)를 순회하며 `DMSForecaster` 인스턴스화 및 설정.
    - 데이터 로더에서 배치를 순회하며 모델별 예측 수행 (Point 및 Quantile).
    - 예측 결과(part_id, sample_idx, model, horizon, predictions)를 수집하여 구조화.
    - Polars DataFrame으로 변환 후 지정된 경로에 Parquet 포맷으로 저장.
    """
    if pl is None: raise ImportError("polars required")

    rows = []
    device = torch.device(device) if device else None
    sample_idx = 0

    # 1. 모델별 Forecaster(예측기) 인스턴스 생성 및 안전 장치(Guard) 설정
    forecasters = {
        name: DMSForecaster(
            model,
            target_channel=target_channel,
            fill_mode=fill_mode,
            use_winsor=True,
            use_multi_guard=True
        )
        for name, model in model_dict.items()
    }

    # 2. 데이터 로더 순회 및 예측 수행
    for batch in loader:
        b = _unpack_batch_for_export(batch)
        xb = b["x"]

        B = xb.size(0)
        for i in range(B):
            if sample_idx >= max_samples: break

            # 개별 샘플 추출 및 준비
            x1 = xb[i:i + 1]
            y1 = b["y"][i] if b["y"] is not None else None

            pid = b["part_ids"][i] if b["part_ids"] is not None else None
            fe1 = b["future_exo"][i:i + 1] if b["future_exo"] is not None else None
            pec1 = b["past_exo_cont"][i:i + 1] if b["past_exo_cont"] is not None else None
            pek1 = b["past_exo_cat"][i:i + 1] if b["past_exo_cat"] is not None else None

            # 등록된 모든 모델에 대해 예측 실행
            for name, fcaster in forecasters.items():
                pred = fcaster.predict(
                    x1,
                    horizon=int(horizon),
                    device=device,
                    mode=mode,
                    part_ids=[pid] if pid is not None else None,
                    past_exo_cont=pec1,
                    past_exo_cat=pek1,
                    future_exo_batch=fe1,
                    future_exo_cb=future_exo_cb
                )

                # 결과 행 추가 (리스트 변환 포함)
                rows.append({
                    "part_id": _to_py_id(pid),
                    "sample_idx": int(sample_idx),
                    "model": str(name),
                    "horizon": int(horizon),
                    "y_pred_point": pred.get("point").tolist(),
                    "y_pred_q50": pred.get("q50", pred.get("point")).tolist(),
                    "y_pred_q10": pred.get("q10", pred.get('point')).to_list(),
                    "y_pred_q90": pred.get("q90", pred.get('point')).to_list()
                })

            sample_idx += 1
        if sample_idx >= max_samples: break

    # 3. 결과 저장 (Polars -> Parquet)
    df = pl.DataFrame(rows)
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    df.write_parquet(parquet_path)
    return df