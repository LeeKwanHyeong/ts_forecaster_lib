from modeling_module.utils.metrics import mae, smape

from modeling_module.utils.metrics import rmse
import torch
import numpy as np

def load_model_ckpt(model, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def eval_on_loader(model, loader, device: str, future_exo_cb=None):
    model.eval()
    ys, yhats = [], []

    for batch in loader:
        x = batch[0].to(device)              # [B, 52, 1]
        y = batch[1].to(device)              # [B, 27]
        start_idx = batch[2]  # <-- 여기! 실제 batch 포맷에 맞게 수정 (현재 batch[2]가 part_ids일 수도 있음)

        if future_exo_cb is not None:
            # start_idx가 tensor면 python int로
            if torch.is_tensor(start_idx):
                start_idx = int(start_idx[0].item()) if start_idx.numel() > 0 else int(start_idx.item())
            fe = future_exo_cb(start_idx, model.horizon, device=device)  # [H, D] 또는 [B,H,D] 규약 확인 필요
            if fe.dim() == 2:
                fe = fe.unsqueeze(0).expand(x.size(0), -1, -1)  # [B,H,D]
            future_exo = fe.to(device)
        else:
            future_exo = None
        past_exo_cont = batch[4].to(device)  # [B, 52, 11]
        past_exo_cat = batch[5].to(device)   # [B, 52, 0]

        # PatchTSTPointModel.forward 시그니처에 맞춰 전달
        yhat = model(
            x,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
        )  # [B, 27]

        ys.append(y.detach().cpu().numpy())
        yhats.append(yhat.detach().cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    yhat_all = np.concatenate(yhats, axis=0)
    return y_all, yhat_all


def _extract_pred_from_output(out, prefer_q=0.5):
    """
    Quantile 모델 출력(out)이 dict/tuple/tensor 등일 수 있으므로
    예측 텐서를 안전하게 꺼낸다.

    반환:
      - yhat_point: [B,H] (예: q=0.5 또는 첫 번째 출력)
      - extra: 원본 out (필요 시 분석용)
    """
    # 1) Tensor면 그대로
    if torch.is_tensor(out):
        return out, out

    # 2) tuple/list면 첫 원소를 예측으로 가정
    if isinstance(out, (tuple, list)):
        # 가장 흔한 케이스: (pred, aux) 또는 (q_pred, ...)
        first = out[0]
        if torch.is_tensor(first):
            return first, out
        # 더 복잡하면 아래 dict 로직으로 넘기기 위해 out를 dict처럼 처리 불가 -> 에러
        raise TypeError(f"Unsupported tuple/list output types: {[type(x) for x in out]}")

    # 3) dict면 키 후보들에서 찾기
    if isinstance(out, dict):
        # 흔한 키 후보들
        key_candidates = [
            "yhat", "pred", "prediction", "y_pred", "output",
            "q_pred", "quantiles", "yq", "y_hat"
        ]
        for k in key_candidates:
            if k in out and torch.is_tensor(out[k]):
                t = out[k]
                # [B,H] 또는 [B,H,Q] 또는 [B,Q,H] 등 가능
                return _select_point_from_quantile_tensor(t, prefer_q=prefer_q), out

        # dict 안에 텐서가 하나뿐이면 그걸 쓰기
        tensor_items = [(k, v) for k, v in out.items() if torch.is_tensor(v)]
        if len(tensor_items) == 1:
            t = tensor_items[0][1]
            return _select_point_from_quantile_tensor(t, prefer_q=prefer_q), out

        raise KeyError(f"Cannot find tensor prediction in dict output. keys={list(out.keys())}")

    raise TypeError(f"Unsupported model output type: {type(out)}")


def _select_point_from_quantile_tensor(t: torch.Tensor, prefer_q=0.5):
    """
    t가
      - [B,H]이면 그대로
      - [B,H,Q]이면 Q축에서 median(0.5)에 해당하는 index 선택
      - [B,Q,H]이면 Q축에서 선택 후 [B,H]로
    """
    if t.ndim == 2:
        return t

    if t.ndim == 3:
        B, d1, d2 = t.shape

        # [B,H,Q] 케이스 가정: d1이 horizon, d2가 quantile
        # [B,Q,H] 케이스 가정: d1이 quantile, d2가 horizon
        # heuristic: horizon은 보통 27 같은 값, quantile은 보통 3/5/9 등 작은 값
        if d1 > d2:
            # [B,H,Q]
            q_dim = 2
            q_len = d2
            h_dim = 1
        else:
            # [B,Q,H]
            q_dim = 1
            q_len = d1
            h_dim = 2

        # prefer_q=0.5에 해당하는 index를 선택 (가능하면 중앙)
        # q 값 리스트를 out에 포함하지 않는 경우가 많으니 중앙 index를 사용
        q_idx = int(round((q_len - 1) * prefer_q))  # median index

        if q_dim == 2:
            # [B,H,Q] -> [B,H]
            return t[:, :, q_idx]
        else:
            # [B,Q,H] -> [B,H]
            return t[:, q_idx, :]

    raise ValueError(f"Unexpected prediction tensor shape: {t.shape}")

@torch.no_grad()
def eval_on_loader_quantile(model, loader, device, prefer_q=0.5, future_exo_cb=None):
    model.eval()
    ys, yhats = [], []

    for batch in loader:
        # 기존 unpack 유지 (당신 eval에서 batch[0], batch[1] ... 쓰는 구조)
        x = batch[0].to(device)
        y = batch[1].to(device)

        # ---- 중요: future_exo 생성
        # dataset이 start_idx를 batch에 포함하지 않으면, 아래 두 줄은 dataset에서 가져오는 방식으로 조정 필요
        # 보통은 batch에 start_idx 또는 date_idx 같은 메타가 들어있거나, part_ids가 들어있습니다.
        start_idx = batch[2]  # <-- 여기! 실제 batch 포맷에 맞게 수정 (현재 batch[2]가 part_ids일 수도 있음)
        if future_exo_cb is not None:
            # start_idx가 tensor면 python int로
            if torch.is_tensor(start_idx):
                start_idx = int(start_idx[0].item()) if start_idx.numel() > 0 else int(start_idx.item())
            fe = future_exo_cb(start_idx, model.horizon, device=device)  # [H, D] 또는 [B,H,D] 규약 확인 필요
            if fe.dim() == 2:
                fe = fe.unsqueeze(0).expand(x.size(0), -1, -1)  # [B,H,D]
            future_exo = fe.to(device)
        else:
            future_exo = None

        past_exo_cont = batch[4].to(device)
        past_exo_cat  = batch[5].to(device)

        out = model(x, future_exo=future_exo, past_exo_cont=past_exo_cont, past_exo_cat=past_exo_cat)

        yhat_point, _ = _extract_pred_from_output(out, prefer_q=prefer_q)
        ys.append(y.detach().cpu().numpy())
        yhats.append(yhat_point.detach().cpu().numpy())

    return np.concatenate(ys), np.concatenate(yhats)

import numpy as np
import torch

def _pick_q50_from_q(q: torch.Tensor, *, prefer_q=0.5, quantiles=(0.1,0.5,0.9), horizon=27) -> torch.Tensor:
    """
    q: (B,Q,H) or (B,H,Q)
    return: (B,H) q50
    """
    qs = list(quantiles)
    idx = int(min(range(len(qs)), key=lambda i: abs(qs[i]-prefer_q)))

    if q.ndim != 3:
        raise RuntimeError(f"q ndim={q.ndim}, shape={tuple(q.shape)}")

    # normalize to (B,Q,H)
    if q.shape[1] == horizon and q.shape[2] == len(qs):   # (B,H,Q)
        q = q.transpose(1, 2)                              # -> (B,Q,H)
    elif q.shape[1] == len(qs) and q.shape[2] == horizon:  # (B,Q,H)
        pass
    else:
        raise RuntimeError(f"Unrecognized q shape={tuple(q.shape)} (H={horizon}, Q={len(qs)})")

    return q[:, idx, :]  # (B,H)

@torch.no_grad()
def eval_on_loader_quantile_v2(
    model, loader, device,
    *,
    prefer_q=0.5,
    use_exo_inputs: bool,
    quantiles_fallback=(0.1,0.5,0.9),
    horizon: int = 27,
):
    model.eval()
    ys, yhats = [], []

    # 가능하면 model.configs.quantiles 사용
    qs = getattr(getattr(model, "configs", None), "quantiles", None)
    quantiles = tuple(qs) if qs is not None else tuple(quantiles_fallback)

    for batch in loader:
        # loader 포맷: (x, y, uid, fe, pe_cont, pe_cat)
        x = batch[0].to(device)
        y = batch[1].to(device)

        future_exo = None
        past_exo_cont = None
        past_exo_cat  = None

        if use_exo_inputs:
            # future exo는 loader가 이미 batch[3]로 줌
            if len(batch) > 3:
                fe = batch[3].to(device)
                # (B,H,0) 같은 경우 None 처리
                if fe.ndim == 3 and fe.shape[-1] == 0:
                    future_exo = None
                else:
                    future_exo = fe

            if len(batch) > 4:
                pe_cont = batch[4].to(device)
                past_exo_cont = None if (pe_cont.ndim == 3 and pe_cont.shape[-1] == 0) else pe_cont

            if len(batch) > 5:
                pe_cat = batch[5].to(device)
                past_exo_cat = None if (pe_cat.ndim == 3 and pe_cat.shape[-1] == 0) else pe_cat

        out = model(
            x,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
        )

        # PatchMixerQuantileModel: {"q": (B,Q,H)} 로 반환
        if isinstance(out, dict) and "q" in out:
            pred = _pick_q50_from_q(out["q"], prefer_q=prefer_q, quantiles=quantiles, horizon=horizon)  # (B,H)
        else:
            # PatchTST 등 기존 유틸과 호환이 필요하면 여기서 _extract_pred_from_output 사용
            yhat_point, _ = _extract_pred_from_output(out, prefer_q=prefer_q)  # (B,H) 기대
            pred = yhat_point

        ys.append(y.squeeze(-1).detach().cpu().numpy())   # (B,H)
        yhats.append(pred.detach().cpu().numpy())         # (B,H)

    return np.concatenate(ys, axis=0), np.concatenate(yhats, axis=0)
