import copy
import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple

import torch
from torch.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity

import os
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict

# -----------------------------------------------------------------------------
# Device / AMP helpers (robust to str | torch.device)
# -----------------------------------------------------------------------------
def _normalize_device_type(x) -> str:
    """Normalize 'cuda:0' / torch.device('cuda:0') / 'CUDA' -> 'cuda'."""
    if x is None:
        return "cpu"
    s = str(x).lower()
    if s.startswith("cuda"):
        return "cuda"
    if s.startswith("cpu"):
        return "cpu"
    if s.startswith("mps"):
        return "mps"
    return s

def _resolve_device(device_like) -> torch.device:
    """Resolve a device-like input into torch.device, falling back to CPU."""
    if isinstance(device_like, torch.device):
        return device_like
    if device_like is None:
        return torch.device("cpu")
    try:
        return torch.device(device_like)
    except Exception:
        return torch.device("cpu")

def _resolve_autocast_dtype(dtype_like, device_type: str) -> torch.dtype:
    """Resolve autocast dtype from str | torch.dtype. CPU autocast defaults to bfloat16."""
    if isinstance(dtype_like, torch.dtype):
        dt = dtype_like
    elif dtype_like is None:
        dt = torch.float16
    else:
        s = str(dtype_like).lower()
        mapping = {
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        dt = mapping.get(s, torch.float16)

    # CPU autocast: float16 is often unsupported/inefficient; prefer bfloat16.
    if device_type == "cpu" and dt == torch.float16:
        dt = torch.bfloat16
    return dt

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.model_losses.losses import LossComputer



# -----------------------------------------------------------------------------
# Debug stack tracing (code-level) for torch ops
# - Useful when IDE "View Call Stack" is disabled / not clickable.
# - Captures Python call sites for common sync-heavy ops: Tensor.to / Tensor.item
# -----------------------------------------------------------------------------
def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _short_stack(*, limit: int = 25, skip: int = 0, keep_keywords=None) -> str:
    """
    Return a compact Python stack string.
    - keep_keywords: if provided, only keep frames whose filename contains any keyword.
    """
    st = traceback.extract_stack(limit=limit)
    st = st[:-1 - skip]  # drop this function frame + optional skip
    frames = []
    for fr in reversed(st):
        fn = fr.filename.replace("\\", "/")
        if keep_keywords:
            if not any(k in fn for k in keep_keywords):
                continue
        # hide torch internals by default
        if "/site-packages/torch/" in fn or "/python" in fn and "site-packages" in fn:
            continue
        frames.append(f"{fn}:{fr.lineno} in {fr.name} -> {fr.line}".rstrip())
        if len(frames) >= 12:
            break
    return "\n".join(frames) if frames else "(no python frames kept)"

class TorchStackTracer:
    """
    Monkeypatch Tensor.to / Tensor.item to print code-level stacks.
    - Extremely useful to locate accidental CPU<->GPU transfers and scalar sync points.
    - Enable via env TSF_TRACE_STACK=1 (default off).
    """
    def __init__(
        self,
        *,
        enabled: bool,
        max_prints_per_key: int = 20,
        stack_limit: int = 30,
        keep_keywords=None,
        logger=print,
    ):
        self.enabled = bool(enabled)
        self.max_prints_per_key = int(max_prints_per_key)
        self.stack_limit = int(stack_limit)
        self.keep_keywords = keep_keywords or ["DSIODemand", "modeling_module", "engine.py", "PycharmProjects"]
        self.logger = logger

        self._counts = defaultdict(int)
        self._orig_to = None
        self._orig_item = None

    def _should_print(self, key: str) -> bool:
        self._counts[key] += 1
        return self._counts[key] <= self.max_prints_per_key

    def __enter__(self):
        if not self.enabled:
            return self

        # Guard: patch only once
        if self._orig_to is None:
            self._orig_to = torch.Tensor.to
        if self._orig_item is None:
            self._orig_item = torch.Tensor.item

        tracer = self

        def _to_patched(t: torch.Tensor, *args, **kwargs):
            # Print before executing (so failures still show stack)
            if tracer._should_print("Tensor.to"):
                dev = kwargs.get("device", None)
                dtype = kwargs.get("dtype", None)
                # args can include (device) or (dtype) etc; keep lightweight
                tracer.logger(
                    "[TRACE][Tensor.to] "
                    f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} -> "
                    f"args={args} kwargs={{'device':{dev}, 'dtype':{dtype}}}\n"
                    f"{_short_stack(limit=tracer.stack_limit, keep_keywords=tracer.keep_keywords)}\n"
                )
            return tracer._orig_to(t, *args, **kwargs)

        def _item_patched(t: torch.Tensor, *args, **kwargs):
            if tracer._should_print("Tensor.item"):
                tracer.logger(
                    "[TRACE][Tensor.item] "
                    f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device}\n"
                    f"{_short_stack(limit=tracer.stack_limit, keep_keywords=tracer.keep_keywords)}\n"
                )
            return tracer._orig_item(t, *args, **kwargs)

        torch.Tensor.to = _to_patched
        torch.Tensor.item = _item_patched
        self.logger(
            f"[TRACE] TorchStackTracer enabled "
            f"(max_prints_per_key={self.max_prints_per_key}, keep_keywords={self.keep_keywords})"
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        # restore
        try:
            if self._orig_to is not None:
                torch.Tensor.to = self._orig_to
            if self._orig_item is not None:
                torch.Tensor.item = self._orig_item
        finally:
            self.logger(f"[TRACE] TorchStackTracer disabled. counts={dict(self._counts)}")
        return False

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
            device
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

        # AMP / device canonicalization (robust to str | torch.device)
        # - `config.device` is kept as-is to minimize ripple effects.
        # - `self.device` becomes the single source of truth inside the trainer.
        self.device = _resolve_device(device if device is not None else getattr(self.cfg, "device", "cpu"))
        self.device_type = self.device.type

        # Extra hooks / flags
        self.autocast_input = autocast_input or {}
        self.extra_loss_fn = extra_loss_fn

        # Exogenous mode: keep backward-compat (explicit arg overrides cfg)
        cfg_exo = bool(getattr(self.cfg, "use_exogenous_mode", False))
        self.use_exogenous_mode = bool(use_exogenous_mode or cfg_exo)

        # Autocast (AMP) settings
        # - device_type: explicit override > cfg.amp_device > resolved device.type
        amp_device_req = self.autocast_input.get("device_type", getattr(self.cfg, "amp_device", self.device_type))
        self.amp_device = _normalize_device_type(amp_device_req)

        # - dtype: explicit override > cfg.autocast_dtype
        dtype_req = self.autocast_input.get("dtype", getattr(self.cfg, "autocast_dtype", "float16"))
        self.dtype = _resolve_autocast_dtype(dtype_req, self.amp_device)

        # - enabled: explicit override > cfg.use_amp
        requested_amp = bool(self.autocast_input.get("enabled", getattr(self.cfg, "use_amp", False)))

        # 실제로 AMP를 켤 수 있는지 최종 결정:
        #   1) 요청(enabled)이 True
        #   2) amp_device == 실제 device.type (mismatch 방지)
        #   3) CUDA일 경우 CUDA 가용성 확인
        self.enabled = bool(
            requested_amp
            and (self.amp_device == self.device_type)
            and (self.amp_device != "cuda" or torch.cuda.is_available())
        )

        # Legacy alias (기존 코드 호환)
        self.amp_enabled = self.enabled

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
        """텐서 내 NaN/Inf 존재 여부를 선택적으로 검사하고 이상 시에만 로깅."""
        cfg = self.cfg
        enabled = False
        if isinstance(cfg, dict):
            enabled = bool(cfg.get("debug_nan_stats", False))
        else:
            enabled = bool(getattr(cfg, "debug_nan_stats", False))

        # Keep the hot path cheap in normal training runs.
        if not enabled or not torch.is_tensor(t):
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
            self.logger(f"[NaN-{name}] has_nan={has_nan} has_inf={has_inf} max|x|={mx}")

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
        - (선택) 간이 프로파일(load vs compute) 및 torch.profiler 정밀 프로파일 지원.

        Profiler 사용 방법(예):
            cfg.profile = True
            cfg.profile_dir = "tb_prof/patchtst"
            cfg.profile_warmup = 10
            cfg.profile_steps = 30
            # 실행 후:
            #   tensorboard --logdir tb_prof
        """
        device = self.device
        total_scalar = 0.0
        total_tensor = None

        # 모델 모드(학습/평가) 전환
        model.train() if train else model.eval()

        # ----------------------
        # profiler 옵션 (cfg 기반)
        # ----------------------
        def _cfg_get(key, default):
            if isinstance(self.cfg, dict):
                return self.cfg.get(key, default)
            return getattr(self.cfg, key, default)

        do_prof = bool(_cfg_get("profile", False))
        prof_steps = int(_cfg_get("profile_steps", 30))
        prof_warmup = int(_cfg_get("profile_warmup", 10))
        prof_dir = str(_cfg_get("profile_dir", r"C:\Users\USER\PycharmProjects\ts_forecaster_lib\tb_prof"))
        simple_prof_enabled = bool(_cfg_get("simple_profile", False))

        # CUDA synchronize는 CUDA에서만
        is_cuda = (hasattr(device, "type") and device.type == "cuda") or str(device).lower().startswith("cuda")
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if is_cuda else [ProfilerActivity.CPU]

        def _sync():
            if is_cuda and (do_prof or simple_prof_enabled):
                torch.cuda.synchronize()

        # ----------------------
        # 간이 profile(load vs compute) 준비
        # ----------------------
        import time
        if simple_prof_enabled and not hasattr(self, "_prof_simple"):
            self._prof_simple = {"load": 0.0, "compute": 0.0, "n": 0}
            self._prof_simple_warmup = int(_cfg_get("simple_profile_warmup", 10))
            self._prof_simple_steps = int(_cfg_get("simple_profile_steps", 200))
            self._prof_simple_i = 0

        def _finalize_epoch_total(denom: int) -> float:
            total = float(total_scalar)
            if total_tensor is not None:
                total += float(total_tensor.item())
            return total / max(1, denom)

        def _print_top_ops_with_stack(prof, *, sort_by="cpu_time_total", topk=15, stack_depth=8):
            """
            key_averages(group_by_stack_n=...) 로 'op + stack' 기준 집계 테이블 출력
            """
            try:
                tbl = prof.key_averages(group_by_stack_n=stack_depth).table(
                    sort_by=sort_by,
                    row_limit=topk
                )
                self.logger(f"[Profiler] key_averages(group_by_stack_n={stack_depth}) sort_by={sort_by}")
                self.logger(tbl)
            except Exception as e:
                self.logger(f"[Profiler] key_averages stack table failed: {e}")

        def _dump_event_stacks(prof, *,
                               keys=("aten::to", "aten::item", "aten::_local_scalar_dense", "aten::to_copy"),
                               max_events_per_key=3, max_stack_lines=12):
            """
            개별 이벤트에서 stack을 직접 뽑아 프린트 (UI 없이도 '어느 코드라인에서 호출했는지' 확인용)
            """
            try:
                # 이벤트가 많을 수 있으니 cpu_time_total 큰 순으로 정렬 후 필터
                evs = list(prof.events())
                evs.sort(key=lambda e: getattr(e, "cpu_time_total", 0), reverse=True)

                seen = {k: 0 for k in keys}
                for e in evs:
                    k = getattr(e, "key", None)
                    if k not in seen:
                        continue
                    if seen[k] >= max_events_per_key:
                        continue

                    st = getattr(e, "stack", None)
                    if not st:
                        continue

                    seen[k] += 1
                    self.logger(f"\n[STACK-DUMP] op={k} cpu_time_total(us)={getattr(e, 'cpu_time_total', 0)} "
                                f"cuda_time_total(us)={getattr(e, 'cuda_time_total', 0)}")
                    # stack은 보통 "file:line - func" 문자열 리스트
                    for line in st[:max_stack_lines]:
                        self.logger(f"  {line}")

                self.logger(f"\n[STACK-DUMP] done. counts={seen}")
            except Exception as e:
                self.logger(f"[Profiler] event stack dump failed: {e}")

        # NOTE:
        # - for batch in loader 구조에서는 next(it) 시간 자체를 분리하기 어려움.
        # - 여기서는 "unpack + H2D(to(device))"를 load로 근사하고,
        #   "forward~(backward/step)"을 compute로 근사합니다.
        def _run_one_step(batch):
            nonlocal total_scalar, total_tensor

            t0 = time.perf_counter()

            # 1) 배치 데이터 구조 분해
            x, y, part_ids, fe_cont, pe_cont, pe_cat = self._unpack_batch(batch)
            t1 = time.perf_counter()

            # 2) H2D (가능하면 non_blocking 사용; pin_memory=True일 때 효과)
            if torch.is_tensor(x):
                x = x.to(device, non_blocking=True)
            if torch.is_tensor(y):
                y = y.to(device, non_blocking=True)
            t2 = time.perf_counter()

            # 입력 데이터 수치 안정성(NaN/Inf) 검사
            # self._nan_stat("x(in)", x)
            # self._nan_stat("y(in)", y)
            # if pe_cont is not None:
            #     self._nan_stat("past_exo_cont", pe_cont)
            # if pe_cat is not None:
            #     self._nan_stat("past_exo_cat", pe_cat)

            if train:
                self.opt.zero_grad(set_to_none=True)

            # 3) 미래 외생 변수
            if self.use_exogenous_mode:
                future_exo = self._resolve_future_exo(fe_cont, x, y, device=device)
            else:
                future_exo = None

            # 4) AMP 컨텍스트 내 순전파 및 손실 계산
            _ac_kwargs = {"device_type": self.amp_device, "enabled": self.enabled}
            if self.dtype is not None:
                _ac_kwargs["dtype"] = self.dtype

            with autocast(**_ac_kwargs):
                pred = self._forward_with_adapter(
                    model,
                    x,
                    future_exo=future_exo,
                    past_exo_cont=(pe_cont.to(device, non_blocking=True) if torch.is_tensor(pe_cont) else None),
                    past_exo_cat=(pe_cat.to(device, non_blocking=True) if torch.is_tensor(pe_cat) else None),
                    part_ids=part_ids,
                    mode=("train" if train else "eval"),
                )
                # self._nan_stat("pred", pred)

                loss = self.loss_comp.compute(pred, y, is_val=(not train))

                # 디버깅: Spike Loss 상세 분석 (초기 배치 한정)
                if self._get_spike_enabled():
                    self._dbg_spike_seen += 1
                    if self._dbg_spike_seen <= self._dbg_max_print:
                        self._debug_spike_breakdown(
                            pred, y, is_val=(not train),
                            tag=("train" if train else "eval")
                        )

                # 추가 손실 함수(Extra Loss) 합산
                if self.extra_loss_fn is not None:
                    loss = loss + self.extra_loss_fn(x, pred, self.cfg)

                # self._nan_stat("loss_raw", loss)

                # 정규화 손실(Regularization Loss) 합산
                reg = self.adapter.reg_loss(model)
                if reg is not None:
                    # self._nan_stat("reg", reg)
                    loss = loss + reg

            # 5) 역전파 및 최적화 (train only)
            if train:
                loss_t = self._to_tensor(loss, device)

                if torch.isnan(loss_t):
                    self.logger("[Warn] NaN loss. step skipped.")
                    return

                # detect_anomaly는 매우 느림. 필요 시 cfg로 켜는 것을 권장.
                if bool(_cfg_get("detect_anomaly", False)):
                    torch.autograd.set_detect_anomaly(True)

                self.scaler.scale(loss_t).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

            _sync()
            t3 = time.perf_counter()

            # ---- 간이 프로파일 누적 ----
            if train and simple_prof_enabled:
                self._prof_simple_i += 1
                if self._prof_simple_i > self._prof_simple_warmup and self._prof_simple["n"] < self._prof_simple_steps:
                    self._prof_simple["load"] += (t2 - t0)       # unpack + H2D
                    self._prof_simple["compute"] += (t3 - t2)    # forward ~ step
                    self._prof_simple["n"] += 1

                    if self._prof_simple["n"] == self._prof_simple_steps:
                        avg_load = self._prof_simple["load"] / self._prof_simple["n"] * 1000
                        avg_comp = self._prof_simple["compute"] / self._prof_simple["n"] * 1000
                        ratio = self._prof_simple["load"] / (self._prof_simple["load"] + self._prof_simple["compute"]) * 100
                        self.logger(f"[SIMPLE-PROFILE] avg load   = {avg_load:.2f} ms/step")
                        self.logger(f"[SIMPLE-PROFILE] avg compute = {avg_comp:.2f} ms/step")
                        self.logger(f"[SIMPLE-PROFILE] load ratio  = {ratio:.1f}%")

            # loss 누적
            if torch.is_tensor(loss):
                loss_detached = loss.detach()
                if loss_detached.numel() != 1:
                    loss_detached = loss_detached.float().mean()
                total_tensor = loss_detached if total_tensor is None else total_tensor + loss_detached
            else:
                total_scalar += float(loss)

        # ----------------------
        # 루프 실행 (profiler on/off)
        # ----------------------
        import os
        tb_logdir = r"C:\Users\USER\PycharmProjects\ts_forecaster_lib\src\model_test\tb_prof"
        profile_dir = os.path.join(tb_logdir, "plugins", "profile")
        os.makedirs(profile_dir, exist_ok=True)
        with torch.set_grad_enabled(train):
            if train and do_prof:
                # warmup + measure만 수행 (trace 파일 크기 제한)
                max_steps = prof_warmup + prof_steps
                with profile(
                        activities=activities,
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,  # 핵심
                        with_flops=True,
                        with_modules=True,
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                ) as prof:
                    _tracer = TorchStackTracer(enabled=_env_bool('TSF_TRACE_STACK', True), max_prints_per_key=int(os.getenv('TSF_TRACE_MAX', '20')))
                    _tracer.__enter__()
                    for i, batch in enumerate(loader):
                        if i >= max_steps:
                            break
                        if i >= prof_warmup:
                            with record_function("train_step"):
                                _run_one_step(batch)
                        else:
                            _run_one_step(batch)
                        prof.step()

                    _tracer.__exit__(None, None, None)

                # --- 기존 summary 출력 대신 아래를 호출 ---
                _print_top_ops_with_stack(prof, sort_by="cpu_time_total", topk=25, stack_depth=10)
                _print_top_ops_with_stack(prof, sort_by="cuda_time_total", topk=25, stack_depth=10)

                # 'to/item' 류가 어디서 터지는지 라인 단위로 보고 싶으면 이것도:
                _dump_event_stacks(prof)

                # 요약 테이블 출력
                try:
                    self.logger(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
                    self.logger(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
                except Exception as e:
                    self.logger(f"[Profiler] summary print failed: {e}")

                # 프로파일 모드에서는 epoch 전체를 다 돌지 않았으므로 len(loader) 대신 실제 step로 평균
                denom = max(1, max_steps)
                return _finalize_epoch_total(denom)

            # 일반 학습/검증 (전체 epoch)
            for batch in loader:
                _run_one_step(batch)

        return _finalize_epoch_total(len(loader))

    # ----------------- 학습 진입 -----------------
    def fit(self, model, train_loader, val_loader, *, tta_steps: int = 0):
        """
        전체 학습 프로세스 실행 관리.

        기능:
        - 옵티마이저/스케줄러 설정.
        - 에폭 반복 및 조기 종료(Early Stopping) 체크.
        - 검증 루프 및 TTA(Test-Time Adaptation) 수행.
        """
        device = self.device
        model.to(device)
        from modeling_module.training.optim import build_optimizer_and_scheduler

        # 최적화 도구 및 스케줄러 빌드
        self.opt, self.sched = build_optimizer_and_scheduler(model, self.cfg)
        self.scaler = GradScaler(device="cuda", enabled=(self.enabled and self.amp_device == "cuda"))
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
            val_total_scalar = 0.0
            val_total_tensor = None
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
                            _ac_kwargs = {"device_type": self.amp_device, "enabled": self.enabled}

                            if self.dtype is not None:
                                _ac_kwargs["dtype"] = self.dtype

                            with autocast(**_ac_kwargs):
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
                        if torch.is_tensor(loss):
                            loss_detached = loss.detach()
                            if loss_detached.numel() != 1:
                                loss_detached = loss_detached.float().mean()
                            val_total_tensor = loss_detached if val_total_tensor is None else val_total_tensor + loss_detached
                        else:
                            val_total_scalar += float(loss)
                    else:
                        # 일반 평가 (Standard Validation)
                        _ac_kwargs = {"device_type": self.amp_device, "enabled": self.enabled}

                        if self.dtype is not None:
                            _ac_kwargs["dtype"] = self.dtype

                        with autocast(**_ac_kwargs):
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
                            vloss_detached = vloss.detach()
                            if vloss_detached.numel() != 1:
                                vloss_detached = vloss_detached.float().mean()
                            val_total_tensor = vloss_detached if val_total_tensor is None else val_total_tensor + vloss_detached

                    # 메트릭 계산 (선택 사항)
                    if self.metrics_fn:
                        _ = self.metrics_fn(pred, y_val)


            # 에폭별 검증 손실 집계 및 스케줄러 갱신
            val_total = float(val_total_scalar)
            if val_total_tensor is not None:
                val_total += float(val_total_tensor.item())
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
