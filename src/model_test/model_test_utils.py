import numpy as np
import polars as pl
import torch
from datetime import date, timedelta
from typing import Callable, Optional
from modeling_module.training.forecater import DMSForecaster


def add_week(yyyyww: int, add: int) -> int:
    """
    YYYYWW에 주(week) 단위로 add만큼 더한 YYYYWW 반환.

    예) add_week(202601, 1) -> 202602

    [작업자 변경 가능 포인트]
    - 예측 결과를 plan_week 포함(H개) vs plan_week+1부터(H개)로 만들지 정책에 따라 달라짐
    """
    return monday_to_yyyyww(yyyyww_to_monday(yyyyww) + timedelta(weeks=int(add)))

def yyyyww_to_monday(yyyyww: int) -> date:
    """
    YYYYWW(정수) -> 해당 ISO week의 월요일(date)로 변환.

    예) 202601 -> 2026년 ISO week 1의 월요일 날짜

    [주의]
    - ISO week는 연말/연초에 '연도'가 달라질 수 있음(ISO year 기준)
      예: 2025년 12월 말이 ISO 기준으로 2026년 1주에 포함될 수 있음
    - 입력 주차가 존재하지 않는 값이면 date.fromisocalendar에서 ValueError 발생
    """
    y = int(yyyyww) // 100
    w = int(yyyyww) % 100
    return date.fromisocalendar(y, w, 1)  # ISO week에서 월요일=1


def monday_to_yyyyww(d: date) -> int:
    """
    date -> YYYYWW(정수)로 변환(ISO year-week 기준)

    예) 2026-01-05(월) -> 202601
    """
    iso_y, iso_w, _ = d.isocalendar()
    return int(iso_y) * 100 + int(iso_w)


# -----------------------------
# 결과 테이블 생성 함수
# -----------------------------
@torch.no_grad()
def make_forecast_result_table(
    *,
    inference_loader,
    base_model: torch.nn.Module,
    quantile_model: torch.nn.Module,
    plan_week: int,
    horizon: int,
    device: str = "cuda",
    max_parts: int = 10_000,          # 필요시 제한
    future_exo_cb: Optional[Callable] = None,
) -> pl.DataFrame:
    """
    [기능 요약]
    inference_loader에서 배치를 순회하며 각 시계열(파트/스토어 등)별로
    - base_model: point forecast
    - quantile_model: q50 forecast
    을 생성하고,
    (plan_week, oper_part_no, forecast_week) 단위 long table로 반환한다.

    [입력 가정]
    inference_loader는 batch마다 아래 형태를 반환한다고 가정:
        (x, uid, fe, pe_cont, pe_cat)
    - x: (B, L, C)  과거 타깃(및 채널)
    - uid: (B,)     각 샘플 식별자
    - fe: (B, H, E) 미래 외생(future exo). 없으면 None 또는 shape[-1]=0 가능
    - pe_cont: (B, L, E_past_cont) 과거 연속 외생
    - pe_cat: (B, L, E_past_cat)   과거 범주 외생

    [출력]
    polars DataFrame, long format:
        plan_week: int
        oper_part_no: str
        forecast_week: int
        base_forecast: float
        quantile_forecast: float

    [작업자 주의사항]
    1) plan_week 포함 여부:
       - 현재 weeks = [plan_week + 0, ..., plan_week + H-1]
       - 만약 "예측은 다음 주부터"라면 range를 1..H 로 바꿔야 함

    2) uid 타입:
       - uid가 torch.Tensor scalar면 pid.item() 사용 가능
       - uid가 이미 문자열이면 그대로 str(uid) 처리해야 함

    3) 성능:
       - 배치 내 샘플을 i 루프로 1개씩 예측 -> 느림
       - production/대량 inference면 DMSForecaster가 batch 입력을 받도록 개선 권장
    """

    # --------------------------------------------------------
    # (1) Forecaster 래핑
    # --------------------------------------------------------
    # DMSForecaster는 모델 추론을 위한 공통 wrapper로 보이며,
    # - target_channel=0: y 채널이 다변량일 때 어떤 채널을 예측 대상으로 볼지 지정
    # - fill_mode="copy_last": 입력 부족/결측 시 마지막 값 복사 등 보정 전략 (프로젝트 정책 확인)
    # Align inference stabilizers with the library export helper so raw negative
    # or explosive predictions are guarded before final_df clips negatives to 0.
    base_fc = DMSForecaster(
        base_model,
        target_channel=0,
        fill_mode="copy_last",
        use_winsor=True,
        use_multi_guard=True,
    )
    q_fc = DMSForecaster(
        quantile_model,
        target_channel=0,
        fill_mode="copy_last",
        use_winsor=True,
        use_multi_guard=True,
    )

    rows = []
    n_parts = 0

    # --------------------------------------------------------
    # (2) inference_loader 순회
    # --------------------------------------------------------
    for batch in inference_loader:
        # [중요] 여기서 배치 언패킹 형태가 loader 구현과 다르면 바로 에러 납니다.
        # - 만약 loader가 (x, y, uid, fe, pe_cont, pe_cat) 형태면 아래 줄을 수정해야 합니다.
        if len(batch) == 5:
            x, uid, fe, pe_cont, pe_cat = batch
        elif len(batch) == 6:
            x, _, uid, fe, pe_cont, pe_cat = batch
        else:
            raise RuntimeError(f"Unsupported inference batch tuple length: {len(batch)}")

        # B: 배치 크기
        B = x.size(0)

        # ----------------------------------------------------
        # (3) 배치 내 샘플(=파트/시계열) 단위 순회
        # ----------------------------------------------------
        for i in range(B):
            # max_parts를 넘기면 early stop
            if n_parts >= max_parts:
                break

            # (3-1) 샘플 1개만 분리하여 (1, L, C) 형태로 만듦
            # - DMSForecaster.predict가 batch=1 입력을 받아 처리하도록 구성된 것으로 가정
            x1 = x[i:i+1]

            # (3-2) 파트/시계열 ID 추출
            # - uid가 tensor scalar라면 uid[i]는 tensor(0-d) 또는 tensor(1,)일 수 있음
            # - uid가 str/list면 uid[i] 자체가 string일 수 있음
            pid = uid[i]  # oper_part_no로 사용(필요하면 mapping)

            # (3-3) 외생변수도 같은 인덱스로 1개 샘플만 분리
            # - fe/pe_cont/pe_cat가 None이면 그대로 None
            fe1  = fe[i:i+1] if fe is not None else None
            if torch.is_tensor(fe1) and fe1.dim() >= 3 and fe1.shape[-1] <= 0:
                fe1 = None
            pec1 = pe_cont[i:i+1] if pe_cont is not None else None
            pek1 = pe_cat[i:i+1] if pe_cat is not None else None

            # ------------------------------------------------
            # (4) Base(Point) 예측
            # ------------------------------------------------
            pred_base = base_fc.predict(
                x1,
                horizon=int(horizon),
                device=device,
                mode="eval",
                # part_ids는 forecaster 내부에서 로깅/조건 분기 등에 사용될 수 있음
                # - uid 타입이 tensor면 list에 tensor가 들어갈 수 있어 내부에서 str 변환 필요할 수 있음
                part_ids=[pid] if pid is not None else None,
                # 과거 외생
                past_exo_cont=pec1,
                past_exo_cat=pek1,
                # 미래 외생 (H, E)
                future_exo_batch=fe1,
                future_exo_cb=future_exo_cb,
            )

            # pred_base["point"]를 (H,)로 변환
            # - 반환 shape이 (1, H) or (H,) 등 다양할 수 있으므로 reshape(-1)로 평탄화
            y_base = np.asarray(pred_base["point"]).reshape(-1)

            # ------------------------------------------------
            # (5) Quantile(q50) 예측
            # ------------------------------------------------
            pred_q = q_fc.predict(
                x1,
                horizon=int(horizon),
                device=device,
                mode="eval",
                part_ids=[pid] if pid is not None else None,
                past_exo_cont=pec1,
                past_exo_cat=pek1,
                future_exo_batch=fe1,
                future_exo_cb=future_exo_cb,
            )

            # - quantile 모델은 q50 키가 있을 수도 있고 없을 수도 있음(프로젝트 구현체에 따라 다름)
            # - q50이 없으면 point를 fallback으로 사용
            y_q50 = np.asarray(pred_q.get("q50", pred_q["point"])).reshape(-1)

            # ------------------------------------------------
            # (6) forecast_week 생성
            # ------------------------------------------------
            # 현재는 plan_week 포함하여 horizon개 생성:
            #   [plan_week, plan_week+1, ..., plan_week+(H-1)]
            # 정책상 "다음주부터" 예측이면:
            #   weeks = [add_week(plan_week, h) for h in range(1, H+1)]
            weeks = [add_week(plan_week, h) for h in range(int(horizon))]

            # ------------------------------------------------
            # (7) long-format row 적재
            # ------------------------------------------------
            # uid가 tensor(1개 원소)라면 pid.item()으로 파이썬 스칼라 변환 후 문자열화
            # uid가 이미 문자열이면 str(pid)로 충분
            pid_str = str(pid.item()) if torch.is_tensor(pid) and pid.numel() == 1 else str(pid)

            # weeks, y_base, y_q50 길이가 모두 horizon인지 확인 필요
            # - 불일치 시 zip이 짧은 쪽에 맞춰 잘리므로 조용히 데이터가 누락될 수 있음
            for w, bval, qval in zip(weeks, y_base.tolist(), y_q50.tolist()):
                rows.append({
                    "plan_week": int(plan_week),
                    "oper_part_no": pid_str,
                    "forecast_week": int(w),
                    "base_forecast": float(bval),
                    "quantile_forecast": float(qval),
                })

            n_parts += 1

        if n_parts >= max_parts:
            break

    # --------------------------------------------------------
    # (8) polars DataFrame 반환
    # --------------------------------------------------------
    # rows가 비어있으면 빈 DF가 생성됨
    return pl.DataFrame(rows)


def calc_accuracy(df):
    col_nm = '_base_forecast'
    df = df.with_columns(
        pl.when((pl.col('8' + col_nm) == 0) | (pl.col('8' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('8' + col_nm) - pl.col('demand_qty')) / pl.col('8' + col_nm))
        .alias('far_pre8'),
        pl.when((pl.col('7' + col_nm) == 0) | (pl.col('7' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('7' + col_nm) - pl.col('demand_qty')) / pl.col('7' + col_nm))
        .alias('far_pre7'),
        pl.when((pl.col('6' + col_nm) == 0) | (pl.col('6' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('6' + col_nm) - pl.col('demand_qty')) / pl.col('6' + col_nm))
        .alias('far_pre6'),
        pl.when((pl.col('5' + col_nm) == 0) | (pl.col('5' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('5' + col_nm) - pl.col('demand_qty')) / pl.col('5' + col_nm))
        .alias('far_pre5'),
        pl.when((pl.col('4' + col_nm) == 0) | (pl.col('4' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('4' + col_nm) - pl.col('demand_qty')) / pl.col('4' + col_nm))
        .alias('far_pre4'),
        pl.when((pl.col('3' + col_nm) == 0) | (pl.col('3' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('3' + col_nm) - pl.col('demand_qty')) / pl.col('3' + col_nm))
        .alias('far_pre3'),
        pl.when((pl.col('2' + col_nm) == 0) | (pl.col('2' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('2' + col_nm) - pl.col('demand_qty')) / pl.col('2' + col_nm))
        .alias('far_pre2'),
        pl.when((pl.col('1' + col_nm) == 0) | (pl.col('1' + col_nm) * 2 < pl.col('demand_qty')))
        .then(pl.lit(0))
        .otherwise(1 - np.abs(pl.col('1' + col_nm) - pl.col('demand_qty')) / pl.col('1' + col_nm))
        .alias('far_pre1'),
    )

    df = df.with_columns([
        (pl.col('far_pre8') * 0.25 + pl.col('far_pre7') * 0.25 + pl.col('far_pre6') * 0.15 + pl.col('far_pre5') * 0.15 +
         pl.col('far_pre4') * 0.07 + pl.col('far_pre3') * 0.07 + pl.col('far_pre2') * 0.03 + pl.col('far_pre1') * 0.03)
        .alias('accu'),
        (pl.col('8' + col_nm) * 0.25 + pl.col('7' + col_nm) * 0.25 + pl.col('6' + col_nm) * 0.15 + pl.col('5' + col_nm) * 0.15
         + pl.col('4' + col_nm) * 0.07 + pl.col('3' + col_nm) * 0.07 + pl.col('2' + col_nm) * 0.03 + pl.col('1' + col_nm) * 0.03)
        .alias('fcst_qty')
    ])
    return df
