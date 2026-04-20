from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

from model_test.exogenous_test.exogenous_ab_utils import (
    attach_actuals,
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
from model_test.exogenous_test.run_exogenous_model_ab import build_architecture_config
from modeling_module import (
    ArtifactConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    ExogenousConfig,
    LoaderConfig,
    RuntimeConfig,
    SSLConfig,
    TrainerConfig,
    TrainRequest,
    load_predictor,
    train,
)
from modeling_module.api.data import build_datamodule
from modeling_module.api.train import ArchitectureConfig
ID_COL = "unique_id"
DATE_COL = "date"
Y_COL = "y"
FREQ = "weekly"


@dataclass(frozen=True)
class CompareSpec:
    label: str
    request_key: str
    use_future_exogenous: bool


def _default_specs(include_baselines: bool) -> list[CompareSpec]:
    specs = [
        CompareSpec("patchtst_no_future", "patchtst_base", False),
        CompareSpec("patchtst_token_cross_attn", "patchtst_base", True),
    ]
    if include_baselines:
        specs.extend(
            [
                CompareSpec("timexer", "timexer_base", False),
                CompareSpec("exotst", "exotst_base", True),
            ]
        )
    return specs


def _summarize_train_result(train_result) -> dict:
    return {
        "requested_models": list(getattr(train_result, "requested_models", ()) or ()),
        "save_dir": getattr(train_result, "save_dir", None),
        "ckpt_paths": dict(getattr(train_result, "ckpt_paths", {}) or {}),
        "pretrain_ckpt_paths": dict(getattr(train_result, "pretrain_ckpt_paths", {}) or {}),
        "manifest_path": getattr(train_result, "manifest_path", None),
        "primary_result_name": getattr(train_result, "primary_result_name", None),
        "primary_ckpt_path": getattr(train_result, "primary_ckpt_path", None),
        "best_ckpt_path": getattr(train_result, "best_ckpt_path", None),
        "results": dict(getattr(train_result, "results", {}) or {}),
    }


def _configure_torch_runtime() -> tuple[str, str | None]:
    if torch.cuda.is_available():
        return "cuda", None
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", None
    return "cpu", None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_walmart_frame(path: Path) -> pl.DataFrame:
    preview_cols = pl.read_parquet(path, n_rows=1).columns
    return (
        pl.read_parquet(path)
        .with_columns(
            [
                pl.col(ID_COL).cast(pl.String),
                pl.col(DATE_COL).cast(pl.Int64),
                pl.col(Y_COL).cast(pl.Float64),
                *[
                    pl.col(c).cast(pl.Float64)
                    for c in preview_cols
                    if c.startswith("exo_c_")
                ],
            ]
        )
        .sort([ID_COL, DATE_COL])
    )


def _build_data_request(
    *,
    df: pl.DataFrame,
    lookback: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    past_exo_cont_cols: list[str],
    past_exo_cat_cols: list[str],
    future_exo_cont_cols: list[str],
    use_future_exogenous: bool,
) -> DataRequest:
    return DataRequest(
        df=df,
        window=DataWindowConfig(
            lookback=int(lookback),
            horizon=int(horizon),
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
            use_future_exogenous=bool(use_future_exogenous),
            past_exo_cont_cols=list(past_exo_cont_cols),
            past_exo_cat_cols=list(past_exo_cat_cols),
            future_exo_cont_cols=list(future_exo_cont_cols) if use_future_exogenous else [],
            future_exo_cb=None,
            part_future_exo_fn=None,
        ),
        loader=LoaderConfig(
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )


def _build_architecture(args: argparse.Namespace, spec: CompareSpec) -> ArchitectureConfig:
    arch_args = SimpleNamespace(
        patch_len=args.patch_len,
        stride=args.stride,
        patchtst_d_model=args.patchtst_d_model,
        patchtst_layers=args.patchtst_layers,
        patchtst_d_ff=args.patchtst_d_ff,
        patchtst_future_exo_fusion_dropout=args.patchtst_future_exo_fusion_dropout,
        timexer_patch_len=args.timexer_patch_len,
        timexer_d_model=args.timexer_d_model,
        timexer_heads=args.timexer_heads,
        timexer_d_ff=args.timexer_d_ff,
        timexer_e_layers=args.timexer_e_layers,
        timexer_dropout=args.timexer_dropout,
        timexer_factor=args.timexer_factor,
        timexer_activation=args.timexer_activation,
        timexer_use_norm=args.timexer_use_norm,
        exotst_d_model=args.exotst_d_model,
        exotst_heads=args.exotst_heads,
        exotst_d_ff=args.exotst_d_ff,
        exotst_dropout=args.exotst_dropout,
        exotst_attn_dropout=args.exotst_attn_dropout,
        exotst_exo_enc_layers=args.exotst_exo_enc_layers,
        exotst_fusion_layers=args.exotst_fusion_layers,
        exotst_endo_dec_layers=args.exotst_endo_dec_layers,
        exotst_exo_memory_mode=args.exotst_exo_memory_mode,
        exotst_exo_nan_policy=args.exotst_exo_nan_policy,
        exotst_use_revin=args.exotst_use_revin,
        exotst_subtract_last=args.exotst_subtract_last,
    )
    architecture = build_architecture_config(arch_args)
    return architecture


def _run_single_model(
    *,
    spec: CompareSpec,
    train_df: pl.DataFrame,
    infer_df: pl.DataFrame,
    eval_target_df: pl.DataFrame,
    past_exo_cont_cols: list[str],
    past_exo_cat_cols: list[str],
    future_exo_cont_cols: list[str],
    args: argparse.Namespace,
    run_root: Path,
    device: str,
) -> tuple[dict, pl.DataFrame]:
    architecture = _build_architecture(args, spec)
    model_dir = ensure_dir(run_root / spec.label)

    train_req = _build_data_request(
        df=train_df,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        past_exo_cont_cols=past_exo_cont_cols,
        past_exo_cat_cols=past_exo_cat_cols,
        future_exo_cont_cols=future_exo_cont_cols,
        use_future_exogenous=spec.use_future_exogenous,
    )

    train_result = train(
        TrainRequest(
            data=train_req,
            freq=FREQ,
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=spec.use_future_exogenous,
            models=[spec.request_key],
            architecture=architecture,
            trainer=TrainerConfig(
                warmup_epochs=args.warmup_epochs,
                spike_epochs=args.spike_epochs,
                lr=args.lr,
                use_intermittent=args.use_intermittent,
                val_use_weights=args.val_use_weights,
            ),
            ssl=SSLConfig(
                mode="off",
                pretrain_epochs=0,
                mask_ratio=args.ssl_mask_ratio,
                loss_type=args.ssl_loss_type,
            ),
            runtime=RuntimeConfig(device=device),
            artifacts=ArtifactConfig(
                save_dir=str(model_dir),
                auto_save_dir=False,
            ),
        )
    )

    ckpt_path = train_result.primary_ckpt_path or train_result.ckpt_paths.get(spec.request_key)
    if not ckpt_path:
        raise RuntimeError(f"No checkpoint produced for {spec.label}")

    predictor = load_predictor(
        ckpt_path,
        device=device,
        forecaster_kwargs={
            "target_channel": 0,
            "fill_mode": "copy_last",
            "use_winsor": False,
            "use_multi_guard": False,
        },
    )

    infer_req = _build_data_request(
        df=infer_df,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.infer_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        past_exo_cont_cols=past_exo_cont_cols,
        past_exo_cat_cols=past_exo_cat_cols,
        future_exo_cont_cols=future_exo_cont_cols,
        use_future_exogenous=spec.use_future_exogenous,
    )
    infer_dm = build_datamodule(infer_req)
    infer_loader = infer_dm.get_inference_loader_at_plan(int(args.plan_week))

    forecast_df = make_point_forecast_result_table(
        inference_loader=infer_loader,
        predictor=predictor,
        model_name=spec.label,
        plan_week=int(args.plan_week),
        horizon=int(args.horizon),
        device=device,
        max_parts=int(args.max_parts_per_plan),
    )
    forecast_df = attach_actuals(
        forecast_df,
        eval_target_df,
        id_col=ID_COL,
        date_col=DATE_COL,
        y_col=Y_COL,
    )
    return _summarize_train_result(train_result), forecast_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PatchTST future exo fusion modes on Walmart weekly data.")
    p.add_argument("--parquet-path", type=Path, default=Path("raw_data/train_data/walmart_best_feature_train.parquet"))
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/exogenous_test/patchtst_future_fusion_compare"))
    p.add_argument("--lookback", type=int, default=52)
    p.add_argument("--horizon", type=int, default=13)
    p.add_argument("--plan-week", type=int, default=201231)
    p.add_argument("--plot-part-count", type=int, default=12)
    p.add_argument("--max-parts-per-plan", type=int, default=10000)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--infer-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--warmup-epochs", type=int, default=30)
    p.add_argument("--spike-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use-intermittent", action="store_true", default=False)
    p.add_argument("--val-use-weights", action="store_true", default=False)
    p.add_argument("--ssl-mask-ratio", type=float, default=0.3)
    p.add_argument("--ssl-loss-type", type=str, default="mse")
    p.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--patch-len", type=int, default=13)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--patchtst-d-model", type=int, default=64)
    p.add_argument("--patchtst-layers", type=int, default=3)
    p.add_argument("--patchtst-d-ff", type=int, default=128)
    p.add_argument("--patchtst-future-exo-fusion-dropout", type=float, default=0.1)

    p.add_argument("--timexer-patch-len", type=int, default=13)
    p.add_argument("--timexer-d-model", type=int, default=128)
    p.add_argument("--timexer-heads", type=int, default=8)
    p.add_argument("--timexer-d-ff", type=int, default=256)
    p.add_argument("--timexer-e-layers", type=int, default=3)
    p.add_argument("--timexer-dropout", type=float, default=0.1)
    p.add_argument("--timexer-factor", type=int, default=5)
    p.add_argument("--timexer-activation", type=str, default="gelu")
    p.add_argument("--timexer-use-norm", action="store_true", default=True)

    p.add_argument("--exotst-d-model", type=int, default=128)
    p.add_argument("--exotst-heads", type=int, default=8)
    p.add_argument("--exotst-d-ff", type=int, default=256)
    p.add_argument("--exotst-dropout", type=float, default=0.1)
    p.add_argument("--exotst-attn-dropout", type=float, default=0.1)
    p.add_argument("--exotst-exo-enc-layers", type=int, default=2)
    p.add_argument("--exotst-fusion-layers", type=int, default=2)
    p.add_argument("--exotst-endo-dec-layers", type=int, default=2)
    p.add_argument("--exotst-exo-memory-mode", type=str, default="all")
    p.add_argument("--exotst-exo-nan-policy", type=str, default="zero+indicator")
    p.add_argument("--exotst-use-revin", action="store_true", default=True)
    p.add_argument("--exotst-subtract-last", action="store_true", default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(int(args.seed))
    device, device_note = _configure_torch_runtime()

    df = _load_walmart_frame(args.parquet_path)
    target_df = df.select([ID_COL, DATE_COL, Y_COL])

    past_exo_cont_cols = [c for c in df.columns if c.startswith("exo_p_")] + [
        c for c in df.columns if c.startswith("exo_c_")
    ]
    past_exo_cat_cols: list[str] = []
    future_exo_cont_cols = [
        c for c in df.columns
        if c.startswith("exo_") and not c.startswith("exo_p_") and not c.startswith("exo_c_")
    ]

    plan_week = resolve_single_plan_week(
        target_df,
        date_col=DATE_COL,
        horizon=int(args.horizon),
        plan_week=int(args.plan_week),
    )
    args.plan_week = plan_week

    eval_ids = select_eval_ids_with_full_actual_coverage(
        target_df,
        id_col=ID_COL,
        date_col=DATE_COL,
        plan_weeks=[plan_week],
        horizon=int(args.horizon),
    )
    eval_target_df = target_df.filter(pl.col(ID_COL).cast(pl.String).is_in(eval_ids))
    train_one_table = df.filter(pl.col(DATE_COL) < int(plan_week))
    infer_one_table = df.filter(pl.col(ID_COL).cast(pl.String).is_in(eval_ids))

    specs = _default_specs(include_baselines=bool(args.include_baselines))
    run_root = ensure_dir(args.artifact_root / f"lb{args.lookback}_h{args.horizon}_pw{plan_week}")

    print("parquet_path        :", args.parquet_path)
    print("artifact_root       :", run_root)
    print("device              :", device)
    if device_note:
        print("device_note         :", device_note)
    print("plan_week           :", plan_week)
    print("eval_id_count       :", len(eval_ids))
    print("past_exo_cont_cols  :", len(past_exo_cont_cols))
    print("future_exo_cont_cols:", len(future_exo_cont_cols))
    print("model_specs         :", [s.label for s in specs])

    results: dict[str, dict] = {}
    forecast_tables: list[pl.DataFrame] = []

    for spec in specs:
        print(f"=== RUN {spec.label} ({spec.request_key}) ===")
        train_result, forecast_df = _run_single_model(
            spec=spec,
            train_df=train_one_table,
            infer_df=infer_one_table,
            eval_target_df=eval_target_df,
            past_exo_cont_cols=past_exo_cont_cols,
            past_exo_cat_cols=past_exo_cat_cols,
            future_exo_cont_cols=future_exo_cont_cols,
            args=args,
            run_root=run_root,
            device=device,
        )
        results[spec.label] = train_result
        forecast_tables.append(forecast_df)
        print(spec.label, "forecast shape:", forecast_df.shape)

    combined_forecast_df = pl.concat(forecast_tables, how="vertical_relaxed")
    overall_df, by_horizon_df, latest_summary_df = compute_metric_tables(combined_forecast_df)
    latest_df = select_latest_revision(
        combined_forecast_df.filter(pl.col("actual").is_not_null())
    )

    forecast_path = run_root / "forecast_long.parquet"
    overall_path = run_root / "overall_metrics.csv"
    latest_summary_path = run_root / "latest_revision_metrics.csv"
    manifest_path = run_root / "run_manifest.json"
    metric_plot_path = run_root / "metric_summary.png"
    agg_plot_path = run_root / "latest_revision_aggregate.png"
    part_plot_path = run_root / "latest_revision_parts.png"

    combined_forecast_df.write_parquet(forecast_path)
    overall_df.write_csv(overall_path)
    latest_summary_df.write_csv(latest_summary_path)

    plot_metric_summary(overall_df, metric_plot_path)
    plot_latest_revision_aggregate(latest_df, agg_plot_path)
    sampled_parts = select_plot_parts(latest_df, plot_part_count=int(args.plot_part_count))
    plot_latest_revision_part_grid(
        latest_df,
        sampled_parts=sampled_parts,
        save_path=part_plot_path,
    )

    save_json(
        {
            "parquet_path": str(args.parquet_path),
            "artifact_root": str(run_root),
            "lookback": int(args.lookback),
            "horizon": int(args.horizon),
            "plan_week": int(plan_week),
            "device": device,
            "specs": [
                {
                    "label": s.label,
                    "request_key": s.request_key,
                    "use_future_exogenous": bool(s.use_future_exogenous),
                }
                for s in specs
            ],
            "paths": {
                "forecast_long": str(forecast_path),
                "overall_metrics": str(overall_path),
                "latest_revision_metrics": str(latest_summary_path),
                "metric_plot": str(metric_plot_path),
                "aggregate_plot": str(agg_plot_path),
                "part_plot": str(part_plot_path),
            },
        },
        manifest_path,
    )

    print("\n== Overall Metrics ==")
    print(overall_df)
    print("\n== Latest Revision Metrics ==")
    print(latest_summary_df)
    print("\nSaved:")
    print(" -", forecast_path)
    print(" -", overall_path)
    print(" -", latest_summary_path)
    print(" -", metric_plot_path)
    print(" -", agg_plot_path)
    print(" -", part_plot_path)
    print(" -", manifest_path)


if __name__ == "__main__":
    main()
