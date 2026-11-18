from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Final


REPO_ROOT: Final = Path(__file__).resolve().parents[3]
AB_RUNNER: Final = (
    REPO_ROOT
    / "src"
    / "model_test"
    / "exogenous_test"
    / "run_exogenous_model_ab.py"
)


@dataclass(frozen=True)
class ArchitectureCase:
    d_model: int
    n_layers: int
    d_ff: int


@dataclass(frozen=True)
class SweepRun:
    case_name: str
    artifact_root: str
    command: tuple[str, ...]
    architecture: str
    batch_size: int
    seed: int
    plan_week: int


ARCHITECTURE_CASES: Final[dict[str, ArchitectureCase]] = {
    "arch_base": ArchitectureCase(d_model=256, n_layers=4, d_ff=1024),
    "arch_small": ArchitectureCase(d_model=128, n_layers=3, d_ff=512),
    "arch_deep": ArchitectureCase(d_model=256, n_layers=6, d_ff=1024),
    "arch_wide": ArchitectureCase(d_model=384, n_layers=4, d_ff=1536),
}

SAMSUNG_SOURCE_COMMITS: Final[tuple[str, ...]] = (
    "52ce0f3",
    "7cb4384",
    "217d3b3",
    "c68aba3",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Samsung GCS PatchTST capacity sweep through the maintained "
            "public AB runner."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "samsung_gcs_patchtst_sweep",
    )
    parser.add_argument("--plan-weeks", type=int, nargs="+", required=True)
    parser.add_argument("--target-source", type=Path)
    parser.add_argument("--exo-source", type=Path)
    parser.add_argument("--scope-label", default="samsung_gcs")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=tuple(ARCHITECTURE_CASES),
        default=list(ARCHITECTURE_CASES),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--lookback", type=int, default=52)
    parser.add_argument("--horizon", type=int, default=27)
    parser.add_argument("--sample-part-count", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--spike-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--future-exo-source",
        choices=("columns", "callback"),
        default="columns",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser


def build_sweep_runs(args: argparse.Namespace) -> list[SweepRun]:
    artifact_root_base = args.artifact_root.expanduser().resolve()
    # YYYYWW를 정수 범위로 늘리지 않고 실제 ISO 주차인지 확인한다.
    for week in args.plan_weeks:
        date.fromisocalendar(int(week) // 100, int(week) % 100, 1)
    for name in ("architectures", "batch_sizes", "seeds", "plan_weeks"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be non-empty and contain no duplicates")
    if args.lookback <= 0 or args.horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    runs: list[SweepRun] = []
    for architecture_name in args.architectures:
        architecture = ARCHITECTURE_CASES[architecture_name]
        for batch_size in args.batch_sizes:
            if batch_size <= 0:
                raise ValueError("batch sizes must be positive")
            for seed in args.seeds:
                for plan_week in args.plan_weeks:
                    case_name = (
                        f"{architecture_name}__b{batch_size}__s{seed}"
                        f"__p{plan_week}"
                    )
                    artifact_root = artifact_root_base / case_name
                    command = [
                        sys.executable,
                        str(AB_RUNNER),
                        "--artifact-root",
                        str(artifact_root),
                        "--models",
                        "patchtst_no_future",
                        "patchtst_token_cross_attn",
                        "--lookback",
                        str(args.lookback),
                        "--horizon",
                        str(args.horizon),
                        "--plan-week",
                        str(plan_week),
                        "--sample-part-count",
                        str(args.sample_part_count),
                        "--train-batch-size",
                        str(batch_size),
                        "--infer-batch-size",
                        str(batch_size),
                        "--warmup-epochs",
                        str(args.warmup_epochs),
                        "--spike-epochs",
                        str(args.spike_epochs),
                        "--num-workers",
                        str(args.num_workers),
                        "--device",
                        str(args.device),
                        "--future-exo-source",
                        str(args.future_exo_source),
                        "--seed",
                        str(seed),
                        "--patchtst-d-model",
                        str(architecture.d_model),
                        "--patchtst-layers",
                        str(architecture.n_layers),
                        "--patchtst-d-ff",
                        str(architecture.d_ff),
                    ]
                    if args.clean_output:
                        command.append("--clean-output")
                    for name in ("target_source", "exo_source"):
                        source = getattr(args, name, None)
                        if source is not None:
                            command.extend([
                                "--" + name.replace("_", "-"),
                                str(Path(source).expanduser().resolve()),
                            ])
                    runs.append(
                        SweepRun(
                            case_name=case_name,
                            artifact_root=str(artifact_root),
                            command=tuple(command),
                            architecture=architecture_name,
                            batch_size=batch_size,
                            seed=seed,
                            plan_week=plan_week,
                        )
                    )
    return runs


def write_manifest(
    args: argparse.Namespace,
    runs: list[SweepRun],
) -> Path:
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.artifact_root / "sweep_manifest.json"
    payload = {
        "format_version": 2,
        "scope_label": getattr(args, "scope_label", "samsung_gcs"),
        "lookback": args.lookback,
        "horizon": args.horizon,
        "source_commits": list(SAMSUNG_SOURCE_COMMITS),
        "architectures": {
            name: asdict(ARCHITECTURE_CASES[name])
            for name in args.architectures
        },
        "models": [
            "patchtst_no_future",
            "patchtst_token_cross_attn",
        ],
        "legacy_paths_excluded": [
            "head_flatten",
            "direct MultiPartExoDataModule construction",
            "direct run_total_train_weekly calls",
            "hard-coded Windows data paths",
        ],
        "runs": [asdict(run) for run in runs],
    }
    # 다른 설정의 manifest로 기존 실행 증적을 덮어쓰지 않는다.
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != json.loads(json.dumps(payload)):
            raise FileExistsError(f"Different sweep manifest already exists: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return manifest_path


def main() -> int:
    args = build_parser().parse_args()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    runs = build_sweep_runs(args)
    manifest_path = write_manifest(args, runs)
    print(f"manifest={manifest_path}")
    print(f"run_count={len(runs)}")
    for run in runs:
        print(" ".join(run.command))
        if args.execute:
            subprocess.run(run.command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
