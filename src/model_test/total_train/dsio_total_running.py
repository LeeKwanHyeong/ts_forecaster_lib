from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch


REPO_ROOT = Path(os.environ.get("TS_FORECASTER_REPO_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling_module import (
    ArchitectureConfig,
    ArtifactConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    ExoTSTArchitectureConfig,
    ExogenousConfig,
    LoaderConfig,
    PatchMixerArchitectureConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TimexerArchitectureConfig,
    TitanArchitectureConfig,
    TrainRequest,
    TrainerConfig,
    build_dataloader,
    train,
)
from modeling_module._internal.model_registry import expand_training_targets, family_for_artifact_key
from modeling_module.utils.device import select_default_device


DATA_ROOT = REPO_ROOT / "raw_data" / "master"
TARGET_SOURCE = DATA_ROOT / "tb_master_target.parquet"
TARGET_EXO_SOURCE = DATA_ROOT / "tb_master_target_exo.parquet"
EXO_SOURCE_FALLBACK = DATA_ROOT / "tb_master_exo.parquet"

FREQ = "weekly"
ID_COL = "oper_part_no"
DATE_COL = "demand_dt"
Y_COL = "demand_qty"

PAST_EXO_CONT_COLS = [
    "sin_annual",
    "cos_annual",
    "sin_semi",
    "cos_semi",
    "sin_quarter",
    "cos_quarter",
    "weather_index",
    "macro_index",
    "promo_strength",
    "part_len",
    "week_of_year",
]

FUTURE_EXO_CONT_COLS = [
    "sin_annual",
    "cos_annual",
    "sin_semi",
    "cos_semi",
    "sin_quarter",
    "cos_quarter",
    "weather_index",
    "macro_index",
    "promo_strength",
    "week_of_year",
    "promo_flag",
    "supply_outage_flag",
    "peak_season_flag",
    "is_year_start",
    "is_year_end",
    "is_q_start",
    "is_q_end",
]

PAST_EXO_CAT_COLS: list[str] = []
DEFAULT_ENDO_FAMILY_MODELS = [
    "patchtst",
    "patchmixer",
    "titan"
]
DEFAULT_EXO_FAMILY_MODELS = [
    # "patchtst",
    # "patchmixer",
    # "titan",
    "exotst",
    "timexer"
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_torch_runtime() -> tuple[str, str | None]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return select_default_device()


def load_polars_table(source: Path, table_name: str) -> pl.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"{table_name} source not found: {source}")
    if source.suffix.lower() == ".parquet":
        return pl.read_parquet(source)
    if source.suffix.lower() in {".csv", ".txt"}:
        return pl.read_csv(source)
    raise ValueError(f"Unsupported file type for {table_name}: {source}")


def assert_columns(df: pl.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def sample_ids_by_observed_target(
    df: pl.DataFrame,
    *,
    id_col: str,
    y_col: str,
    min_obs: int,
    sample_size: int,
    seed: int,
) -> list[str]:
    eligible = (
        df.group_by(id_col)
        .agg(pl.col(y_col).is_not_null().sum().alias("observed_target_rows"))
        .filter(pl.col("observed_target_rows") >= min_obs)
        .sort(id_col)
    )
    ids = eligible[id_col].cast(pl.String).to_list()
    if not ids:
        raise ValueError(
            "No ids have enough observed target history. "
            f"Need at least {min_obs} non-null `{y_col}` rows per id."
        )

    take = min(int(sample_size), len(ids))
    if take <= 0:
        raise ValueError(f"`sample_size` must be positive. got={sample_size}")
    if take >= len(ids):
        return ids

    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(ids, dtype=object), size=take, replace=False).tolist()
    return sorted(str(x) for x in sampled)


def prepare_target_df(
    raw_df: pl.DataFrame,
    *,
    id_col: str,
    date_col: str,
    y_col: str,
    use_id_sample: bool,
    max_ids: int,
    sample_part_count: int | None,
    min_obs: int,
    seed: int,
) -> pl.DataFrame:
    assert_columns(raw_df, [id_col, date_col, y_col], "target table")
    df = (
        raw_df.sort([id_col, date_col]).with_columns(
            [
                pl.col(date_col).cast(pl.Int64),
                pl.col(y_col).cast(pl.Float64),
            ]
        )
    )
    effective_sample_size = None
    if sample_part_count is not None:
        effective_sample_size = int(sample_part_count)
    elif use_id_sample:
        effective_sample_size = int(max_ids)

    if effective_sample_size is not None:
        keep_ids = sample_ids_by_observed_target(
            df,
            id_col=id_col,
            y_col=y_col,
            min_obs=min_obs,
            sample_size=effective_sample_size,
            seed=seed,
        )
        df = df.filter(pl.col(id_col).is_in(keep_ids))
    return df


def resolve_exo_source(preferred: Path, fallback: Path) -> Path:
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find exogenous source. Checked: {preferred} and {fallback}"
    )


def prepare_exo_one_table(
    *,
    target_df: pl.DataFrame,
    exo_df: pl.DataFrame,
    id_col: str,
    date_col: str,
    y_col: str,
    past_exo_cont_cols: list[str],
    future_exo_cont_cols: list[str],
    past_exo_cat_cols: list[str],
) -> pl.DataFrame:
    assert_columns(exo_df, [id_col, date_col], "exogenous table")

    valid_ids = target_df.select(id_col).unique()
    one_table = exo_df.join(valid_ids, on=id_col, how="inner")
    if y_col not in one_table.columns:
        target_y = target_df.select([id_col, date_col, y_col])
        one_table = one_table.join(target_y, on=[id_col, date_col], how="left")

    required = [id_col, date_col, y_col, *past_exo_cont_cols, *future_exo_cont_cols, *past_exo_cat_cols]
    assert_columns(one_table, required, "one-table exogenous input")

    float_cols = sorted(
        set([y_col, *past_exo_cont_cols, *future_exo_cont_cols]).intersection(one_table.columns)
    )
    one_table = one_table.with_columns(
        [pl.col(date_col).cast(pl.Int64)] + [pl.col(col).cast(pl.Float64) for col in float_cols]
    )
    return one_table.sort([id_col, date_col])


def describe_batch(batch, label: str) -> None:
    names = ["x", "y", "uid_list", "future_exo", "past_exo_cont", "past_exo_cat"]
    print(f"=== {label} batch ===")
    for name, item in zip(names, batch):
        if torch.is_tensor(item):
            print(f"{name:14s}: shape={tuple(item.shape)}, dtype={item.dtype}")
        elif isinstance(item, list):
            print(f"{name:14s}: list(len={len(item)})")
        else:
            print(f"{name:14s}: {type(item).__name__}")


def print_training_result(train_result, label: str) -> None:
    print(f"[{label}] requested_models:", train_result.requested_models)
    print(f"[{label}] save_dir       :", train_result.save_dir)
    print(f"[{label}] manifest_path  :", train_result.manifest_path)
    print(f"[{label}] total_ckpts    :", len(train_result.ckpt_paths))
    for model_key, ckpt_path in sorted(train_result.ckpt_paths.items()):
        pretrain_ckpt = train_result.pretrain_ckpt_paths.get(model_key)
        print(f"[{label}] {model_key:20s} ckpt={ckpt_path}")
        if pretrain_ckpt:
            print(f"[{label}] {'':20s} pretrain={pretrain_ckpt}")


def build_model_architecture(args: argparse.Namespace) -> ArchitectureConfig:
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
            future_exo_fusion_dropout=getattr(args, "patchtst_future_exo_fusion_dropout", None),
        ),
        patchmixer=PatchMixerArchitectureConfig(
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.patchmixer_d_model,
            e_layers=args.patchmixer_layers,
            f_out=args.patchmixer_f_out,
            head_hidden=args.patchmixer_head_hidden,
            dropout=0.1,
            head_dropout=0.02,
            use_revin=True,
            final_nonneg=True,
            expander_n_harmonics=24,
        ),
        titan=TitanArchitectureConfig(
            d_model=args.titan_d_model,
            n_layers=args.titan_layers,
            n_heads=args.titan_heads,
            d_ff=args.titan_d_ff,
            dropout=0.1,
            contextual_mem_size=args.titan_contextual_mem_size,
            persistent_mem_size=args.titan_persistent_mem_size,
            use_revin=True,
            final_clamp_nonneg=False,
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
    )


def build_loader_config(
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool,
) -> LoaderConfig:
    return LoaderConfig(
        stage="train",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def _resolve_model_groups(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.models:
        shared = [str(m).strip() for m in args.models if str(m).strip()]
        return shared, shared

    endo_models = [str(m).strip() for m in args.endo_models if str(m).strip()]
    exo_models = [str(m).strip() for m in args.exo_models if str(m).strip()]
    if not endo_models:
        endo_models = list(DEFAULT_ENDO_FAMILY_MODELS)
    if not exo_models:
        exo_models = list(DEFAULT_EXO_FAMILY_MODELS)
    return endo_models, exo_models


def _clean_output_dirs(*dirs: Path) -> None:
    for directory in dirs:
        if directory.exists():
            print(f"[clean] removing existing output directory: {directory}")
            shutil.rmtree(directory)


def _split_exo_training_targets(models: list[str]) -> tuple[list[str], list[str]]:
    future_required: list[str] = []
    past_only: list[str] = []

    for key in expand_training_targets(models):
        family = family_for_artifact_key(key)
        if family == "timexer":
            past_only.append(key)
        else:
            future_required.append(key)

    return future_required, past_only


def _build_exo_data_request(
    *,
    args: argparse.Namespace,
    one_table: pl.DataFrame,
    use_future_exogenous: bool,
) -> DataRequest:
    return DataRequest(
        df=one_table,
        window=DataWindowConfig(
            lookback=args.lookback,
            horizon=args.horizon,
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
            past_exo_cont_cols=PAST_EXO_CONT_COLS,
            past_exo_cat_cols=PAST_EXO_CAT_COLS,
            future_exo_cont_cols=(FUTURE_EXO_CONT_COLS if use_future_exogenous else []),
        ),
        loader=build_loader_config(
            batch_size=args.exo_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=not args.no_shuffle,
        ),
    )


def _run_exo_stage(
    *,
    label: str,
    args: argparse.Namespace,
    device: str,
    architecture: ArchitectureConfig,
    models: list[str],
    data_req: DataRequest,
    save_dir: Path,
) -> None:
    if not models:
        print(f"[{label}] skipped: no models requested")
        return

    if not args.skip_batch_check:
        batch = next(iter(build_dataloader(data_req)))
        describe_batch(batch, f"{label}/train")

    if args.clean_output:
        _clean_output_dirs(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    result = train(
        TrainRequest(
            data=data_req,
            freq=FREQ,
            models=models,
            architecture=architecture,
            trainer=TrainerConfig(
                warmup_epochs=args.warmup_epochs,
                spike_epochs=args.spike_epochs,
                lr=args.lr,
            ),
            ssl=SSLConfig(
                mode=args.ssl_mode,
                pretrain_epochs=args.ssl_pretrain_epochs,
                mask_ratio=args.ssl_mask_ratio,
                loss_type=args.ssl_loss_type,
            ),
            runtime=RuntimeConfig(device=device),
            artifacts=ArtifactConfig(
                save_dir=str(save_dir),
                auto_save_dir=False,
            ),
        )
    )
    print_training_result(result, label)


def run_endo(
    args: argparse.Namespace,
    *,
    device: str,
    architecture: ArchitectureConfig,
    models: list[str],
) -> None:
    min_obs = args.lookback + args.horizon
    target_raw = load_polars_table(TARGET_SOURCE, "tb_master_target")
    target_df = prepare_target_df(
        target_raw,
        id_col=ID_COL,
        date_col=DATE_COL,
        y_col=Y_COL,
        use_id_sample=args.use_id_sample,
        max_ids=args.max_ids,
        sample_part_count=args.sample_part_count,
        min_obs=min_obs,
        seed=args.seed,
    )

    print("[ENDO] target_raw shape:", target_raw.shape)
    print("[ENDO] target_df shape :", target_df.shape)
    print("[ENDO] n_unique ids    :", target_df.select(ID_COL).n_unique())

    data_req = DataRequest(
        df=target_df,
        window=DataWindowConfig(
            lookback=args.lookback,
            horizon=args.horizon,
            freq=FREQ,
        ),
        columns=DataColumnConfig(
            id_col=ID_COL,
            date_col=DATE_COL,
            y_col=Y_COL,
        ),
        loader=build_loader_config(
            batch_size=args.endo_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=not args.no_shuffle,
        ),
    )

    if not args.skip_batch_check:
        batch = next(iter(build_dataloader(data_req)))
        describe_batch(batch, "ENDO/train")

    save_dir = args.artifact_root / "endo_only"
    if args.clean_output:
        _clean_output_dirs(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    result = train(
        TrainRequest(
            data=data_req,
            freq=FREQ,
            models=models,
            architecture=architecture,
            trainer=TrainerConfig(
                warmup_epochs=args.warmup_epochs,
                spike_epochs=args.spike_epochs,
                lr=args.lr,
            ),
            ssl=SSLConfig(
                mode=args.ssl_mode,
                pretrain_epochs=args.ssl_pretrain_epochs,
                mask_ratio=args.ssl_mask_ratio,
                loss_type=args.ssl_loss_type,
            ),
            runtime=RuntimeConfig(device=device),
            artifacts=ArtifactConfig(
                save_dir=str(save_dir),
                auto_save_dir=False,
            ),
        )
    )
    print_training_result(result, "ENDO")


def run_exo(
    args: argparse.Namespace,
    *,
    device: str,
    architecture: ArchitectureConfig,
    models: list[str],
) -> None:
    min_obs = args.lookback + args.horizon
    target_raw = load_polars_table(TARGET_SOURCE, "tb_master_target")
    target_df = prepare_target_df(
        target_raw,
        id_col=ID_COL,
        date_col=DATE_COL,
        y_col=Y_COL,
        use_id_sample=args.use_id_sample,
        max_ids=args.max_ids,
        sample_part_count=args.sample_part_count,
        min_obs=min_obs,
        seed=args.seed,
    )
    exo_source = resolve_exo_source(TARGET_EXO_SOURCE, EXO_SOURCE_FALLBACK)
    exo_raw = load_polars_table(exo_source, exo_source.name)
    future_required_models, past_only_models = _split_exo_training_targets(models)

    print("[EXO] exo_source        :", exo_source)
    print("[EXO] target_df shape   :", target_df.shape)
    print("[EXO] exo_raw shape     :", exo_raw.shape)
    print("[EXO] future_models     :", future_required_models)
    print("[EXO] past_only_models  :", past_only_models)

    if future_required_models:
        exo_one_table_future = prepare_exo_one_table(
            target_df=target_df,
            exo_df=exo_raw,
            id_col=ID_COL,
            date_col=DATE_COL,
            y_col=Y_COL,
            past_exo_cont_cols=PAST_EXO_CONT_COLS,
            future_exo_cont_cols=FUTURE_EXO_CONT_COLS,
            past_exo_cat_cols=PAST_EXO_CAT_COLS,
        )
        print("[EXO/FUTURE] one_table   :", exo_one_table_future.shape)
        print("[EXO/FUTURE] n_unique ids:", exo_one_table_future.select(ID_COL).n_unique())
        future_req = _build_exo_data_request(
            args=args,
            one_table=exo_one_table_future,
            use_future_exogenous=True,
        )
        _run_exo_stage(
            label="EXO/FUTURE",
            args=args,
            device=device,
            architecture=architecture,
            models=future_required_models,
            data_req=future_req,
            save_dir=args.artifact_root / "exo_future",
        )

    if past_only_models:
        exo_one_table_past = prepare_exo_one_table(
            target_df=target_df,
            exo_df=exo_raw,
            id_col=ID_COL,
            date_col=DATE_COL,
            y_col=Y_COL,
            past_exo_cont_cols=PAST_EXO_CONT_COLS,
            future_exo_cont_cols=[],
            past_exo_cat_cols=PAST_EXO_CAT_COLS,
        )
        print("[EXO/PAST_ONLY] one_table   :", exo_one_table_past.shape)
        print("[EXO/PAST_ONLY] n_unique ids:", exo_one_table_past.select(ID_COL).n_unique())
        past_req = _build_exo_data_request(
            args=args,
            one_table=exo_one_table_past,
            use_future_exogenous=False,
        )
        _run_exo_stage(
            label="EXO/PAST_ONLY",
            args=args,
            device=device,
            architecture=architecture,
            models=past_only_models,
            data_req=past_req,
            save_dir=args.artifact_root / "exo_past_only",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DSIO weekly total training for endogenous-only and/or exogenous models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["endo", "exo", "both"], default="both")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts" / "total_train")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--endo-models", nargs="+", default=list(DEFAULT_ENDO_FAMILY_MODELS))
    parser.add_argument("--exo-models", nargs="+", default=list(DEFAULT_EXO_FAMILY_MODELS))
    parser.add_argument("--lookback", type=int, default=104)
    parser.add_argument("--horizon", type=int, default=27)
    parser.add_argument("--endo-batch-size", type=int, default=1024)
    parser.add_argument("--exo-batch-size", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--spike-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--ssl-mode",
        choices=["off", "sl_only", "full"],
        default="sl_only",
        help=(
            "PatchTST SSL mode. Use full only for a request containing PatchTST; "
            "the default ExoTST/TimeXer stages are supervised-only."
        ),
    )
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
    parser.add_argument("--use-id-sample", action="store_true")
    parser.add_argument("--max-ids", type=int, default=256)
    parser.add_argument("--sample-part-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-len", type=int, default=13)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--patchtst-d-model", type=int, default=384)
    parser.add_argument("--patchtst-layers", type=int, default=5)
    parser.add_argument("--patchtst-d-ff", type=int, default=1536)
    parser.add_argument("--patchmixer-d-model", type=int, default=192)
    parser.add_argument("--patchmixer-layers", type=int, default=6)
    parser.add_argument("--patchmixer-f-out", type=int, default=256)
    parser.add_argument("--patchmixer-head-hidden", type=int, default=256)
    parser.add_argument("--titan-d-model", type=int, default=384)
    parser.add_argument("--titan-layers", type=int, default=4)
    parser.add_argument("--titan-heads", type=int, default=8)
    parser.add_argument("--titan-d-ff", type=int, default=1536)
    parser.add_argument("--titan-contextual-mem-size", type=int, default=384)
    parser.add_argument("--titan-persistent-mem-size", type=int, default=96)
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    default_device, device_note = configure_torch_runtime()
    device = default_device if args.device == "auto" else args.device
    architecture = build_model_architecture(args)
    endo_models, exo_models = _resolve_model_groups(args)

    print("REPO_ROOT           :", REPO_ROOT)
    print("PYTHON              :", sys.executable)
    print("DEVICE              :", device)
    if args.device == "auto" and device_note:
        print("DEVICE_NOTE         :", device_note)
    print("TARGET_SOURCE       :", TARGET_SOURCE)
    print("TARGET_EXO_SOURCE   :", TARGET_EXO_SOURCE, "(exists=", TARGET_EXO_SOURCE.exists(), ")")
    print("EXO_SOURCE_FALLBACK :", EXO_SOURCE_FALLBACK, "(exists=", EXO_SOURCE_FALLBACK.exists(), ")")
    print("MODE                :", args.mode)
    print("LOOKBACK            :", args.lookback)
    print("HORIZON             :", args.horizon)
    print("SAMPLE_PART_COUNT   :", args.sample_part_count)
    print("ENDO_BATCH_SIZE     :", args.endo_batch_size)
    print("EXO_BATCH_SIZE      :", args.exo_batch_size)
    print("NUM_WORKERS         :", args.num_workers)
    print("PREFETCH_FACTOR     :", args.prefetch_factor)
    print("PIN_MEMORY          :", args.pin_memory)
    print("PERSISTENT_WORKERS  :", args.persistent_workers)
    print("WARMUP_EPOCHS       :", args.warmup_epochs)
    print("SPIKE_EPOCHS        :", args.spike_epochs)
    print("SSL_MODE            :", args.ssl_mode)
    print("SSL_PRETRAIN_EPOCHS :", args.ssl_pretrain_epochs)
    print("ENDO_MODELS         :", endo_models)
    print("EXO_MODELS          :", exo_models)
    print("ARTIFACT_ROOT       :", args.artifact_root)
    print("CLEAN_OUTPUT        :", args.clean_output)
    print("ARCHITECTURE        :", architecture)

    if args.mode in {"endo", "both"}:
        run_endo(args, device=device, architecture=architecture, models=endo_models)
    if args.mode in {"exo", "both"}:
        run_exo(args, device=device, architecture=architecture, models=exo_models)


if __name__ == "__main__":
    main()
