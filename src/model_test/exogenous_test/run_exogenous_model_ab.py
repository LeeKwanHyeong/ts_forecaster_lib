from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_test.exogenous_test.exogenous_ab_utils import (
    attach_actuals,
    build_callback_future_exo_components,
    compute_metric_tables,
    ensure_dir,
    make_point_forecast_result_table,
    plot_latest_revision_aggregate,
    plot_latest_revision_part_grid,
    plot_metric_summary,
    resolve_single_plan_week,
    save_json,
    select_eval_ids_with_full_actual_coverage,
    select_latest_revision,
    select_plot_parts,
)
from model_test.total_train.dsio_total_running import (
    DATE_COL,
    EXO_SOURCE_FALLBACK,
    FUTURE_EXO_CONT_COLS,
    FREQ,
    ID_COL,
    PAST_EXO_CAT_COLS,
    PAST_EXO_CONT_COLS,
    TARGET_EXO_SOURCE,
    TARGET_SOURCE,
    Y_COL,
    configure_torch_runtime,
    load_polars_table,
    prepare_exo_one_table,
    prepare_target_df,
    resolve_exo_source,
    set_global_seed,
)
from modeling_module import (
    ArchitectureConfig,
    ArtifactConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    ExoTSTArchitectureConfig,
    ExogenousConfig,
    LoaderConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TimexerArchitectureConfig,
    TrainRequest,
    TrainerConfig,
    load_predictor,
    train,
)
from modeling_module.api.data import build_datamodule


@dataclass(frozen=True)
class ModelSpec:
    # label: artifact/plot/metric 파일명에 노출되는 사용자 친화 이름
    # request_key: public train API에 넘길 canonical artifact key
    # use_future_exogenous: timexer처럼 past-only 모델을 분기하기 위한 플래그
    label: str
    request_key: str
    use_future_exogenous: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "patchtst_exo": ModelSpec(label="patchtst_exo", request_key="patchtst_base", use_future_exogenous=True),
    "timexer": ModelSpec(label="timexer", request_key="timexer_base", use_future_exogenous=False),
    "exotst": ModelSpec(label="exotst", request_key="exotst_base", use_future_exogenous=True),
}

CALLBACK_CALENDAR_FUTURE_COLS: tuple[str, ...] = ("sin_annual", "cos_annual")
CALLBACK_LOOKUP_FUTURE_COLS: tuple[str, ...] = tuple(
    col for col in FUTURE_EXO_CONT_COLS if col not in CALLBACK_CALENDAR_FUTURE_COLS
)


def build_architecture_config(args: argparse.Namespace) -> ArchitectureConfig:
    # 세 모델을 같은 실험 런타임에서 비교하되,
    # 모델별 아키텍처 override는 한 곳에서 모아 관리한다.
    return ArchitectureConfig(
        patchtst=PatchTSTArchitectureConfig(
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.patchtst_d_model,
            n_layers=args.patchtst_layers,
            d_ff=args.patchtst_d_ff,
            dropout=0.1,
            norm="LayerNorm",
            pre_norm=True,
            act="gelu",
            use_revin=True,
            pe="sincos",
            learn_pe=True,
            padding_patch="end",
        ),
        timexer=TimexerArchitectureConfig(
            patch_len=args.timexer_patch_len,
            d_model=args.timexer_d_model,
            n_heads=args.timexer_heads,
            d_ff=args.timexer_d_ff,
            e_layers=args.timexer_e_layers,
            dropout=args.timexer_dropout,
            factor=args.timexer_factor,
            activation=args.timexer_activation,
            use_norm=args.timexer_use_norm,
        ),
        exotst=ExoTSTArchitectureConfig(
            d_model=args.exotst_d_model,
            n_heads=args.exotst_heads,
            d_ff=args.exotst_d_ff,
            dropout=args.exotst_dropout,
            attn_dropout=args.exotst_attn_dropout,
            exo_enc_layers=args.exotst_exo_enc_layers,
            fusion_layers=args.exotst_fusion_layers,
            endo_dec_layers=args.exotst_endo_dec_layers,
            exo_memory_mode=args.exotst_exo_memory_mode,
            exo_nan_policy=args.exotst_exo_nan_policy,
            use_revin=args.exotst_use_revin,
            subtract_last=args.exotst_subtract_last,
        ),
    )


def build_exo_data_request(
    *,
    df: pl.DataFrame,
    lookback: int,
    horizon: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool,
    use_future_exogenous: bool,
    future_exo_source: str,
    future_exo_cb=None,
    part_future_exo_fn=None,
) -> DataRequest:
    # AB 테스트에서는 미래 외생을 두 방식으로 비교할 수 있다.
    # 1) columns  : 테이블의 future_exo_cont_cols를 그대로 사용
    # 2) callback : calendar callback + part별 lookup callback을 collate에서 합성
    if future_exo_source not in {"columns", "callback"}:
        raise ValueError(f"Unsupported future_exo_source={future_exo_source!r}")

    future_cols = []
    effective_future_exo_cb = None
    effective_part_future_exo_fn = None
    if use_future_exogenous:
        if future_exo_source == "columns":
            future_cols = list(FUTURE_EXO_CONT_COLS)
        else:
            # callback 경로는 두 콜백이 함께 있어야 최종 future_exo가 완성된다.
            if future_exo_cb is None or part_future_exo_fn is None:
                raise ValueError(
                    "Callback future exogenous mode requires both `future_exo_cb` and `part_future_exo_fn`."
                )
            effective_future_exo_cb = future_exo_cb
            effective_part_future_exo_fn = part_future_exo_fn

    return DataRequest(
        df=df,
        window=DataWindowConfig(
            lookback=lookback,
            horizon=horizon,
            freq=FREQ,
        ),
        columns=DataColumnConfig(
            id_col=ID_COL,
            date_col=DATE_COL,
            y_col=Y_COL,
        ),
        exogenous=ExogenousConfig(
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=use_future_exogenous,
            past_exo_cont_cols=list(PAST_EXO_CONT_COLS),
            past_exo_cat_cols=list(PAST_EXO_CAT_COLS),
            future_exo_cont_cols=future_cols,
            future_exo_cb=effective_future_exo_cb,
            part_future_exo_fn=effective_part_future_exo_fn,
        ),
        loader=LoaderConfig(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )


def resolve_model_specs(names: list[str]) -> list[ModelSpec]:
    # notebook/CLI에서 받은 모델 이름을 검증하고,
    # 이후 로직에서는 일관된 ModelSpec만 다루도록 정규화한다.
    out: list[ModelSpec] = []
    for name in names:
        key = str(name).strip().lower()
        if key not in MODEL_SPECS:
            raise ValueError(f"Unsupported model name: {name}. Expected one of {sorted(MODEL_SPECS)}")
        out.append(MODEL_SPECS[key])
    return out


def maybe_clean_dir(path: Path, clean_output: bool) -> None:
    if clean_output and path.exists():
        shutil.rmtree(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare PatchTST-Exo, TimeXer, and ExoTST on the DSIO exogenous dataset."
    )
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts" / "exogenous_test")
    parser.add_argument("--models", nargs="+", default=["patchtst_exo", "timexer", "exotst"])
    parser.add_argument("--lookback", type=int, default=104)
    parser.add_argument("--horizon", type=int, default=27)
    # 이 스크립트는 단일 plan week backtest를 전제로 한다.
    parser.add_argument("--plan-week", type=int, required=True)
    parser.add_argument("--plot-part-count", type=int, default=12)
    parser.add_argument("--sample-part-count", type=int, default=256)
    parser.add_argument("--max-parts-per-plan", type=int, default=100_000)
    # 같은 모델이라도 future exo를 column / callback 중 어떤 계약으로 넣는지 비교 가능
    parser.add_argument("--future-exo-source", choices=["columns", "callback"], default="columns")
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--spike-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ssl-mode", choices=["off", "sl_only", "full"], default="off")
    parser.add_argument("--ssl-pretrain-epochs", type=int, default=2)
    parser.add_argument("--ssl-mask-ratio", type=float, default=0.3)
    parser.add_argument("--ssl-loss-type", type=str, default="mse")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--skip-batch-check", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-len", type=int, default=13)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--patchtst-d-model", type=int, default=384)
    parser.add_argument("--patchtst-layers", type=int, default=5)
    parser.add_argument("--patchtst-d-ff", type=int, default=1536)
    parser.add_argument("--timexer-patch-len", type=int, default=13)
    parser.add_argument("--timexer-d-model", type=int, default=128)
    parser.add_argument("--timexer-heads", type=int, default=8)
    parser.add_argument("--timexer-d-ff", type=int, default=256)
    parser.add_argument("--timexer-e-layers", type=int, default=3)
    parser.add_argument("--timexer-dropout", type=float, default=0.1)
    parser.add_argument("--timexer-factor", type=int, default=5)
    parser.add_argument("--timexer-activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--timexer-use-norm", action="store_true", default=True)
    parser.add_argument("--no-timexer-use-norm", action="store_false", dest="timexer_use_norm")
    parser.add_argument("--exotst-d-model", type=int, default=128)
    parser.add_argument("--exotst-heads", type=int, default=8)
    parser.add_argument("--exotst-d-ff", type=int, default=256)
    parser.add_argument("--exotst-dropout", type=float, default=0.1)
    parser.add_argument("--exotst-attn-dropout", type=float, default=0.1)
    parser.add_argument("--exotst-exo-enc-layers", type=int, default=2)
    parser.add_argument("--exotst-fusion-layers", type=int, default=2)
    parser.add_argument("--exotst-endo-dec-layers", type=int, default=2)
    parser.add_argument("--exotst-exo-memory-mode", type=str, default="all")
    parser.add_argument("--exotst-exo-nan-policy", type=str, default="zero+indicator")
    parser.add_argument("--exotst-use-revin", action="store_true", default=True)
    parser.add_argument("--no-exotst-use-revin", action="store_false", dest="exotst_use_revin")
    parser.add_argument("--exotst-subtract-last", action="store_true", default=True)
    parser.add_argument("--no-exotst-subtract-last", action="store_false", dest="exotst_subtract_last")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.artifact_root = args.artifact_root.expanduser().resolve()
    ensure_dir(args.artifact_root)

    set_global_seed(args.seed)
    default_device, device_note = configure_torch_runtime()
    device = default_device if args.device == "auto" else args.device
    architecture = build_architecture_config(args)
    model_specs = resolve_model_specs(args.models)

    print("REPO_ROOT         :", REPO_ROOT)
    print("DEVICE            :", device)
    if args.device == "auto" and device_note:
        print("DEVICE_NOTE       :", device_note)
    print("MODELS            :", [spec.label for spec in model_specs])
    print("TARGET_SOURCE     :", TARGET_SOURCE)
    print("TARGET_EXO_SOURCE :", TARGET_EXO_SOURCE)
    print("LOOKBACK          :", args.lookback)
    print("HORIZON           :", args.horizon)
    print("PLAN_WEEK         :", args.plan_week)
    print("SAMPLE_PART_COUNT :", args.sample_part_count)
    print("FUTURE_EXO_SOURCE :", args.future_exo_source)

    target_raw = load_polars_table(TARGET_SOURCE, "tb_master_target")
    # 전체 master target이 크기 때문에,
    # 빠른 AB 비교를 위해 eligible part 중 일부만 샘플링해서 실험할 수 있다.
    target_df = prepare_target_df(
        target_raw,
        id_col=ID_COL,
        date_col=DATE_COL,
        y_col=Y_COL,
        use_id_sample=False,
        max_ids=args.sample_part_count if args.sample_part_count is not None else 256,
        sample_part_count=args.sample_part_count,
        min_obs=args.lookback + args.horizon,
        seed=args.seed,
    )

    exo_source = resolve_exo_source(TARGET_EXO_SOURCE, EXO_SOURCE_FALLBACK)
    exo_raw = load_polars_table(exo_source, exo_source.name)
    exo_one_table = prepare_exo_one_table(
        target_df=target_df,
        exo_df=exo_raw,
        id_col=ID_COL,
        date_col=DATE_COL,
        y_col=Y_COL,
        past_exo_cont_cols=list(PAST_EXO_CONT_COLS),
        future_exo_cont_cols=list(FUTURE_EXO_CONT_COLS),
        past_exo_cat_cols=list(PAST_EXO_CAT_COLS),
    )

    future_exo_cb = None
    part_future_exo_fn = None
    if args.future_exo_source == "callback":
        # callback 모드에서는 future_exo_cont_cols를 비우고,
        # collate 단계에서 calendar + part-specific future exo를 배치 단위로 합성한다.
        future_exo_cb, part_future_exo_fn = build_callback_future_exo_components(
            exo_one_table,
            id_col=ID_COL,
            date_col=DATE_COL,
            lookup_future_cols=CALLBACK_LOOKUP_FUTURE_COLS,
        )

    plan_week = resolve_single_plan_week(
        target_df,
        date_col=DATE_COL,
        horizon=args.horizon,
        plan_week=args.plan_week,
    )
    # leakage를 피하기 위해 명시한 계획주 이전까지만 학습 테이블로 사용한다.
    train_cutoff = int(plan_week)
    train_one_table = exo_one_table.filter(pl.col(DATE_COL) < train_cutoff)
    eval_ids = select_eval_ids_with_full_actual_coverage(
        target_df,
        id_col=ID_COL,
        date_col=DATE_COL,
        plan_weeks=[plan_week],
        horizon=args.horizon,
    )
    if not eval_ids:
        raise ValueError(
            "No evaluation ids have full actual coverage for the selected plan week. "
            "Move `plan_week` earlier or increase `sample_part_count`."
        )
    eval_target_df = target_df.filter(pl.col(ID_COL).cast(pl.String).is_in(eval_ids))
    infer_one_table = exo_one_table.filter(pl.col(ID_COL).cast(pl.String).is_in(eval_ids))

    print("target_raw shape  :", target_raw.shape)
    print("target_df shape   :", target_df.shape)
    print("exo_raw shape     :", exo_raw.shape)
    print("exo_one_table     :", exo_one_table.shape)
    print("train_cutoff      :", train_cutoff)
    print("plan_week         :", plan_week)
    print("eval_id_count     :", len(eval_ids))
    print("eval_target_df    :", eval_target_df.shape)
    print("infer_one_table   :", infer_one_table.shape)

    output_dirs = {
        "training": ensure_dir(args.artifact_root / "training"),
        "forecasts": ensure_dir(args.artifact_root / "forecasts"),
        "metrics": ensure_dir(args.artifact_root / "metrics"),
        "plots": ensure_dir(args.artifact_root / "plots"),
        "manifest": ensure_dir(args.artifact_root / "manifest"),
    }

    save_json(
        {
            "args": vars(args),
            "models": [asdict(spec) for spec in model_specs],
            "plan_week": plan_week,
            "train_cutoff": train_cutoff,
            "device": device,
            "eval_id_count": len(eval_ids),
            "future_exo_source": args.future_exo_source,
            "callback_calendar_future_cols": list(CALLBACK_CALENDAR_FUTURE_COLS),
            "callback_lookup_future_cols": list(CALLBACK_LOOKUP_FUTURE_COLS),
        },
        output_dirs["manifest"] / "run_config.json",
    )

    all_forecasts: list[pl.DataFrame] = []
    ckpt_manifest: dict[str, str] = {}

    for spec in model_specs:
        print(f"\n=== TRAIN {spec.label} ({spec.request_key}) ===")
        # 모든 모델이 같은 train dataframe을 보되,
        # future exo 사용 여부는 spec별로 갈라진다.
        model_train_df = train_one_table
        train_data_req = build_exo_data_request(
            df=model_train_df,
            lookback=args.lookback,
            horizon=args.horizon,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=not args.no_shuffle,
            use_future_exogenous=spec.use_future_exogenous,
            future_exo_source=args.future_exo_source,
            future_exo_cb=future_exo_cb,
            part_future_exo_fn=part_future_exo_fn,
        )

        if not args.skip_batch_check:
            # AB 실험에서 가장 먼저 확인하고 싶은 것은
            # "이 모델이 실제로 어떤 future_exo shape를 받는가" 이므로 배치 shape를 출력한다.
            train_loader = build_datamodule(train_data_req).get_train_loader(
                batch_size=args.train_batch_size,
                shuffle=not args.no_shuffle,
                drop_last=False,
            )
            batch = next(iter(train_loader))
            print(f"[{spec.label}] train batch future_exo shape:", tuple(batch[3].shape))

        model_save_dir = output_dirs["training"] / spec.label
        maybe_clean_dir(model_save_dir, args.clean_output)
        ensure_dir(model_save_dir)

        ssl_mode = args.ssl_mode if spec.request_key.startswith("patchtst") else "off"
        train_result = train(
            TrainRequest(
                data=train_data_req,
                freq=FREQ,
                models=[spec.request_key],
                architecture=architecture,
                trainer=TrainerConfig(
                    warmup_epochs=args.warmup_epochs,
                    spike_epochs=args.spike_epochs,
                    lr=args.lr,
                ),
                ssl=SSLConfig(
                    mode=ssl_mode,
                    pretrain_epochs=args.ssl_pretrain_epochs,
                    mask_ratio=args.ssl_mask_ratio,
                    loss_type=args.ssl_loss_type,
                ),
                runtime=RuntimeConfig(device=device),
                artifacts=ArtifactConfig(
                    save_dir=str(model_save_dir),
                    auto_save_dir=False,
                ),
            )
        )

        ckpt_path = train_result.primary_ckpt_path or train_result.ckpt_paths.get(spec.request_key)
        if not ckpt_path:
            raise RuntimeError(f"No checkpoint produced for {spec.label} ({spec.request_key})")
        ckpt_manifest[spec.label] = ckpt_path
        print(f"[{spec.label}] ckpt: {ckpt_path}")

        predictor = load_predictor(
            ckpt_path,
            device=device,
            forecaster_kwargs={
                "target_channel": 0,
                "fill_mode": "copy_last",
                "use_winsor": True,
                "use_multi_guard": True,
            },
        )

        infer_data_req = build_exo_data_request(
            df=infer_one_table,
            lookback=args.lookback,
            horizon=args.horizon,
            batch_size=args.infer_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=False,
            use_future_exogenous=spec.use_future_exogenous,
            future_exo_source=args.future_exo_source,
            future_exo_cb=future_exo_cb,
            part_future_exo_fn=part_future_exo_fn,
        )
        infer_dm = build_datamodule(infer_data_req)

        if not args.skip_batch_check:
            infer_batch = next(iter(infer_dm.get_inference_loader_at_plan(int(plan_week))))
            print(f"[{spec.label}] infer batch future_exo shape:", tuple(infer_batch[3].shape))

        print(f"[{spec.label}] infer plan_week={plan_week}")
        infer_loader = infer_dm.get_inference_loader_at_plan(int(plan_week))
        model_forecast_df = make_point_forecast_result_table(
            inference_loader=infer_loader,
            predictor=predictor,
            model_name=spec.label,
            plan_week=int(plan_week),
            horizon=args.horizon,
            device=device,
            max_parts=args.max_parts_per_plan,
        )
        # 최종 평가/플롯을 위해 forecast long table에 GT(actual)를 붙인다.
        model_forecast_df = attach_actuals(
            model_forecast_df,
            eval_target_df,
            id_col=ID_COL,
            date_col=DATE_COL,
            y_col=Y_COL,
        )
        model_forecast_df.write_parquet(output_dirs["forecasts"] / f"{spec.label}_forecast_long.parquet")
        all_forecasts.append(model_forecast_df)

    save_json({"ckpt_paths": ckpt_manifest}, output_dirs["manifest"] / "checkpoint_manifest.json")

    combined_forecast_df = pl.concat(all_forecasts) if all_forecasts else pl.DataFrame()
    combined_forecast_df.write_parquet(output_dirs["forecasts"] / "all_models_forecast_long.parquet")

    overall_df, by_horizon_df, latest_summary_df = compute_metric_tables(combined_forecast_df)
    # latest revision 기준 테이블은 "현재 시점에서 실제로 어떤 모델이 더 쓸만한가"를 보기 쉽다.
    latest_df = select_latest_revision(combined_forecast_df.filter(pl.col("actual").is_not_null()))

    if overall_df.height > 0:
        overall_df.write_parquet(output_dirs["metrics"] / "overall_metrics.parquet")
        overall_df.write_csv(output_dirs["metrics"] / "overall_metrics.csv")
    if by_horizon_df.height > 0:
        by_horizon_df.write_parquet(output_dirs["metrics"] / "metrics_by_horizon.parquet")
        by_horizon_df.write_csv(output_dirs["metrics"] / "metrics_by_horizon.csv")
    if latest_summary_df.height > 0:
        latest_summary_df.write_parquet(output_dirs["metrics"] / "latest_revision_metrics.parquet")
        latest_summary_df.write_csv(output_dirs["metrics"] / "latest_revision_metrics.csv")
    if latest_df.height > 0:
        latest_df.write_parquet(output_dirs["forecasts"] / "latest_revision_forecast_long.parquet")

    sampled_parts = select_plot_parts(latest_df, plot_part_count=args.plot_part_count)
    save_json({"sampled_parts": sampled_parts}, output_dirs["manifest"] / "sampled_parts.json")

    # 요약 플롯은 latest revision을 우선 사용한다.
    # 계획 revision을 전부 섞은 전체 평균보다 실무 해석이 직관적이기 때문이다.
    metric_plot_df = latest_summary_df if latest_summary_df.height > 0 else overall_df
    plot_metric_summary(metric_plot_df, output_dirs["plots"] / "metric_summary.png")
    plot_latest_revision_aggregate(latest_df, output_dirs["plots"] / "aggregate_latest_revision.png")
    plot_latest_revision_part_grid(
        latest_df,
        sampled_parts=sampled_parts,
        save_path=output_dirs["plots"] / "parts_latest_revision.png",
        ncols=3,
    )

    if overall_df.height > 0:
        print("\n=== OVERALL METRICS ===")
        print(overall_df)
    if latest_summary_df.height > 0:
        print("\n=== LATEST REVISION METRICS ===")
        print(latest_summary_df)


if __name__ == "__main__":
    main()
