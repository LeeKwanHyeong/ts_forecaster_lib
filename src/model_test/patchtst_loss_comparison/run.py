"""데이터 고정, 별도 프로세스 실행, 비교 조건 확인을 순서대로 수행하는 실행기."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__:
    from .common import SETTINGS, prepare, save_json, sha256
else:
    from common import SETTINGS, prepare, save_json, sha256


def validate_cached_summary(summary, *, code_commit, data_sha256, seed, lr, smoke):
    """다른 코드나 입력으로 만든 결과를 이번 실행 결과로 재사용하지 않음."""
    expected = {"code_commit": code_commit, "data_sha256": data_sha256,
                "seed": seed, "pretrain_lr": lr, "smoke": smoke,
                "settings": {**SETTINGS, "pretrain_lr": lr}}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Cached comparison result mismatch: {key}; use a new output directory")


def resolve_clean_commit(code_root):
    """실제 입력 코드와 커밋 번호가 일치하도록 깨끗한 비교 worktree 요구."""
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=code_root, text=True)
    if dirty.strip():
        raise ValueError(f"Comparison code worktree must be clean: {code_root}")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()


def validate_source_override(manifest, source):
    if source is None:
        return
    source = source.expanduser().resolve()
    if Path(manifest["source"]).resolve() != source or manifest["source_sha256"] != sha256(source):
        raise ValueError("Explicit target source differs from cached dataset; use a new output directory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-source", type=Path)
    parser.add_argument("--exo-source", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.expanduser().resolve()
    commits = {}
    for variant in ("before", "after"):
        code_root = getattr(args, variant).expanduser().resolve()
        setattr(args, variant, code_root)
        commits[variant] = resolve_clean_commit(code_root)
    root = args.output.resolve()
    dataset = root / "data"
    if not dataset.exists():
        source = args.target_source or args.repo / "raw_data/master/tb_master_target.parquet"
        exogenous = args.exo_source or args.repo / "raw_data/master/tb_master_exo.parquet"
        if args.exo_source is not None and not exogenous.is_file():
            raise FileNotFoundError(exogenous)
        prepare(source, dataset, exogenous if exogenous.is_file() else None)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["settings"] != SETTINGS:
        raise ValueError("저장된 데이터의 설정과 현재 코드 설정이 다름. 새 결과 폴더 필요")
    validate_source_override(manifest, args.target_source)
    data_hash = sha256(dataset / "dataset.npz")
    if manifest["dataset_sha256"] != data_hash:
        raise ValueError("Stored dataset SHA-256 does not match manifest")
    experiments = [("primary", SETTINGS["lr"], seed) for seed in SETTINGS["seeds"]]
    experiments.append(("lr_sensitivity", SETTINGS["sensitivity_lr"], SETTINGS["seeds"][0]))
    if args.smoke:
        experiments = [("smoke", SETTINGS["lr"], 42)]
    for group, lr, seed in experiments:
        pair = []
        for variant in ["before", "after"]:
            folder = root / group / f"seed_{seed}" / variant
            if not (folder / "summary.json").exists():
                folder.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(Path(__file__).with_name("worker.py")),
                           "--code-root", str(getattr(args, variant)), "--data", str(dataset / "dataset.npz"),
                           "--output", str(folder), "--variant", variant, "--seed", str(seed), "--pretrain-lr", str(lr)]
                if args.smoke:
                    command.append("--smoke")
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TSF_TRACE_STACK": "0", "TSF_PROFILE": "0"}
                with (folder.parent / f"{variant}.log").open("w", encoding="utf-8") as log:
                    print("START", group, seed, variant, flush=True)
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((folder / "summary.json").read_text())
            validate_cached_summary(summary, code_commit=commits[variant], data_sha256=data_hash,
                                    seed=seed, lr=lr, smoke=args.smoke)
            pair.append(summary)
            print("DONE", group, seed, variant, summary["test_metrics"], flush=True)
        for key in ["data_sha256", "initial_shared_pretrain_hash", "initial_finetune_hash", "settings"]:
            if pair[0][key] != pair[1][key]:
                raise AssertionError(f"비교 조건 불일치: {key}")
        save_json(root / group / f"seed_{seed}" / "pair_validation.json", {"matched": True, "checked": ["dataset", "common initial weights", "FT initial weights", "settings"]})
    print("ALL PAIRS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
