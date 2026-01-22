from typing import Protocol, Any, Optional, List, Dict
import inspect
import torch
import torch.nn as nn

PREFERRED_KEYS = ("pred", "yhat", "output", "logits")


class ModelAdapter(Protocol):
    """
    모델과 학습 엔진(Trainer) 간의 상호작용을 추상화하는 인터페이스 프로토콜.

    기능:
    - 다양한 모델 아키텍처(PatchTST, Titan 등)의 상이한 입출력 형식을 표준화.
    - 순전파(Forward), 정규화 손실 계산, TTA(Test-Time Adaptation) 기능 명세 정의.
    """

    def forward(
            self,
            model: nn.Module,
            x_batch: Any,
            *,
            future_exo: Optional[torch.Tensor] = None,
            past_exo_cont: Optional[torch.Tensor] = None,  # [B,L,E_c] float32
            past_exo_cat: Optional[torch.Tensor] = None,  # [B,L,E_k] long
            part_ids: Optional[List[str]] = None,  # 길이 B
            mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        모델 순전파 실행.

        기능:
        - 메인 입력(x_batch)과 선택적 외생 변수(Exo), 메타데이터(part_ids)를 모델에 전달.
        - 모델의 출력 텐서(예측값) 반환.
        """
        ...

    def reg_loss(self, model: nn.Module) -> Optional[torch.Tensor]:
        """모델 자체의 추가적인 정규화 손실(Regularization Loss) 산출."""
        ...

    def uses_tta(self) -> bool:
        """테스트 시점 적응(Test-Time Adaptation) 기능 지원 여부 확인."""
        ...

    def tta_reset(self, model: nn.Module):
        """TTA 수행 전 모델의 내부 상태(State) 초기화."""
        ...

    def tta_adapt(self, model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, steps: int) -> Optional[float]:
        """
        검증 데이터를 사용한 테스트 시점 적응(TTA) 수행.
        지정된 스텝(steps)만큼 모델 파라미터를 미세 조정(Adaptation).
        """
        ...


def _model_accepts_kw(model: nn.Module, kw: str) -> bool:
    """
    모델의 `forward` 메서드가 특정 키워드 인자(kw)를 수용하는지 검사.

    기능:
    - `inspect` 모듈을 이용한 함수 시그니처 분석.
    - 모델이 지원하지 않는 인자를 전달하여 발생하는 런타임 오류 방지(Safe Forwarding).
    """
    try:
        sig = inspect.signature(model.forward)
        return kw in sig.parameters
    except Exception:
        return False


class DefaultAdapter:
    """
    다양한 모델의 입출력 형식을 표준화하여 학습 엔진과 연결하는 기본 어댑터.

    기능:
    - 모델의 `forward` 메서드 시그니처(Signature)를 분석하여 지원하는 인자만 안전하게 전달.
    - Dict, Tuple, List 등 다양한 형태의 배치 입력을 모델이 기대하는 텐서 형태로 변환.
    - 모델의 출력을 단일 텐서로 정규화하여 반환.
    """

    # -------------------- utils --------------------
    def _call_model(
            self,
            model: nn.Module,
            x: Any,
            *,
            future_exo: Optional[torch.Tensor] = None,
            past_exo_cont: Optional[torch.Tensor] = None,
            past_exo_cat: Optional[torch.Tensor] = None,
            part_ids: Optional[List[str]] = None,
            mode: Optional[str] = None,
    ):
        """
        모델이 수용 가능한 인자만 선별하여 호출(Safe Forwarding).

        기능:
        - `inspect` 기반의 `_model_accepts_kw` 함수를 통해 인자 지원 여부 확인.
        - 지원하지 않는 인자는 제외하여 런타임 에러(Unexpected Keyword Argument) 방지.
        """
        accepts: Dict[str, bool] = {
            "future_exo": _model_accepts_kw(model, "future_exo"),
            "past_exo_cont": _model_accepts_kw(model, "past_exo_cont"),
            "past_exo_cat": _model_accepts_kw(model, "past_exo_cat"),
            "part_ids": _model_accepts_kw(model, "part_ids"),
            "mode": _model_accepts_kw(model, "mode"),
        }

        # 수용 가능한 인자만 kwargs에 구성
        kwargs = {}
        if future_exo is not None and accepts["future_exo"]:
            kwargs["future_exo"] = future_exo
        if past_exo_cont is not None and accepts["past_exo_cont"]:
            kwargs["past_exo_cont"] = past_exo_cont
        if past_exo_cat is not None and accepts["past_exo_cat"]:
            kwargs["past_exo_cat"] = past_exo_cat
        if part_ids is not None and accepts["part_ids"]:
            kwargs["part_ids"] = part_ids
        if mode is not None and accepts["mode"]:
            kwargs["mode"] = mode

        return model(x, **kwargs)

    def _as_tensor(self, out):
        """
        모델의 다양한 출력 형식(Tuple, List, Dict)에서 메인 텐서를 추출하여 반환 형식을 통일.
        """
        # 1. Tuple/List: 첫 번째 텐서 요소 반환
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    return item
            raise TypeError(f"Model returned tuple/list without a Tensor: {type(out)}")

        # 2. Dict: 우선순위 키(PREFERRED_KEYS) 또는 첫 번째 텐서 값 반환
        if isinstance(out, dict):
            for k in PREFERRED_KEYS:  # PREFERRED_KEYS는 외부 정의 가정 (예: 'pred', 'logits')
                v = out.get(k, None)
                if torch.is_tensor(v):
                    return v
            for v in out.values():
                if torch.is_tensor(v):
                    return v
            raise TypeError(f"Model returned dict without a Tensor value: keys={list(out.keys())}")

        # 3. Tensor: 그대로 반환
        if torch.is_tensor(out):
            return out
        raise TypeError(f"Model output is not a Tensor/tuple/dict: {type(out)}")

    # -------------------- main --------------------
    def forward(
            self,
            model: nn.Module,
            x_batch: Any,
            *,
            future_exo: Optional[torch.Tensor] = None,
            past_exo_cont: Optional[torch.Tensor] = None,
            past_exo_cat: Optional[torch.Tensor] = None,
            part_ids: Optional[List[str]] = None,
            mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        어댑터의 메인 순전파(Forward) 진입점.

        기능:
        - 입력 배치(`x_batch`)의 타입(Dict, Tuple, Tensor)에 따라 적절한 언패킹 수행.
        - 모델 호출(`_call_model`) 및 출력 정규화(`_as_tensor`) 연결.
        """

        # 1. Dict 입력 처리: **kwargs 언패킹 시도
        if isinstance(x_batch, dict):
            try:
                return self._as_tensor(model(**x_batch))
            except TypeError:
                # 언패킹 실패 시 'x' 키를 메인 입력으로 간주
                x_batch = x_batch.get("x", x_batch)

        # 2. Tuple/List 입력 처리
        if isinstance(x_batch, (tuple, list)):
            try:
                # *args 언패킹 시도
                return self._as_tensor(model(*x_batch))
            except TypeError:
                # 언패킹 실패 시, (Input, Future_Exo) 형태의 튜플로 간주하여 처리
                if len(x_batch) == 2 and future_exo is None:
                    x_only, exo = x_batch
                    out = self._call_model(
                        model, x_only,
                        future_exo=exo,
                        past_exo_cont=past_exo_cont,
                        past_exo_cat=past_exo_cat,
                        part_ids=part_ids,
                        mode=mode,
                    )
                    return self._as_tensor(out)
                # 그 외의 경우 첫 번째 요소를 메인 입력으로 사용
                x_batch = x_batch[0]

        # 3. 일반적인 호출 (Tensor 입력 등)
        out = self._call_model(
            model, x_batch,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            part_ids=part_ids,
            mode=mode,
        )
        return self._as_tensor(out)

    # -------------------- optional hooks --------------------
    def reg_loss(self, model):
        """정규화 손실(Regularization Loss) 계산 (기본값: 없음)."""
        return None

    def uses_tta(self):
        """TTA(Test-Time Adaptation) 사용 여부 (기본값: False)."""
        return False

    def tta_reset(self, model):
        """TTA 상태 초기화 (기본값: 패스)."""
        pass

    def tta_adapt(self, model, x_val, y_val, steps):
        """TTA 수행 로직 (기본값: 없음)."""
        return None

class PatchMixerAdapter(DefaultAdapter):
    def reg_loss(self, model):
        try:
            patcher = getattr(getattr(model, 'backbone', None), 'patcher', None)
            if patcher is not None and hasattr(patcher, 'last_reg_loss'):
                return patcher.last_reg_loss()
        except Exception:
            pass
        return None


class TitanAdapter(DefaultAdapter):
    def __init__(self, tta_manager_factory=None):
        self._tta = None
        self._factory = tta_manager_factory

    def uses_tta(self):
        return self._factory is not None

    def tta_reset(self, model):
        self._tta = None

    def tta_adapt(self, model, x_val, y_val, steps):
        if not self._tta:
            return None
        self._tta.add_context(x_val)
        return float(self._tta.adapt(x_val, y_val, steps=steps))