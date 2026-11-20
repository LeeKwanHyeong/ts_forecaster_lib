"""학습 없이 parquet의 식별자·날짜만 확인해 70:15:15 분할 manifest 저장.

수요 수량의 값, 패턴, 모델 오차를 분할 기준으로 읽지 않으며 원천은 변경하지 않음.
실제 수량 품질과 L52/H27 창의 가용성 검사는 후속 데이터셋 단계의 책임.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import polars as pl

from modeling_module.data_loader.temporal import normalize_period_key
from modeling_module.models.Chronos2.training.contracts import Chronos2DataContract, SeriesKey, make_split_contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    source = args.source_parquet.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"기존 계약을 덮어쓰지 않습니다: {output}")
    digest = sha256_file(source)
    contract = Chronos2DataContract(source_id=source.name, source_sha256=digest,
        series_columns=tuple(args.id_columns), zero_semantics=args.zero_semantics, provenance=args.provenance,
        observed_mask_column=args.observed_mask_column)
    schema = pl.scan_parquet(source).collect_schema()
    required = [*contract.series_columns, contract.time_column, contract.target_column]
    if contract.observed_mask_column is not None:
        required.append(contract.observed_mask_column)
    if set(required) - set(schema.names()):
        raise ValueError("원천에 식별자·주차·수량·관측 마스크 필수 칼럼이 없습니다.")
    if any(schema[column] != pl.String for column in contract.series_columns):
        raise TypeError("식별자는 앞자리 0을 보존한 String 칼럼이어야 합니다.")
    if not schema[contract.time_column].is_integer() or not schema[contract.target_column].is_numeric():
        raise TypeError("주차는 ISO YYYYWW 정수, 수량은 수치형 칼럼이어야 합니다.")
    if contract.observed_mask_column and schema[contract.observed_mask_column] != pl.Boolean:
        raise TypeError("실제 관측 여부 칼럼은 Boolean이어야 합니다.")
    # 수량 칼럼은 schema만 검사. 실제 값은 이 단계에서 materialize하지 않음.
    identity = pl.scan_parquet(source).select(*contract.series_columns, contract.time_column)
    if identity.select(pl.any_horizontal(pl.all().is_null()).any()).collect().item():
        raise ValueError("식별자 또는 주차에 null이 있습니다.")
    rows = identity.select(pl.len()).collect().item()
    if rows == 0:
        raise ValueError("원천 파일에 분할할 부품·주차 행이 없습니다.")
    unique_rows = identity.select(pl.struct(pl.all()).n_unique()).collect().item()
    if rows != unique_rows:
        raise ValueError("동일 부품·주차에 중복 행이 있습니다.")
    weeks = identity.select(contract.time_column).unique().collect()[contract.time_column].to_list()
    for week in weeks:
        normalize_period_key(week, "weekly")
    keys = [SeriesKey(tuple(row)) for row in identity.select(*contract.series_columns).unique().collect().iter_rows()]
    excluded, exclusion_evidence = set(), []
    for filename in args.exclude_evaluated_cohort:
        filename = filename.expanduser().resolve()
        before = sha256_file(filename)
        frame = pl.read_csv(filename, schema_overrides={column: pl.String for column in contract.series_columns})
        if set(contract.series_columns) - set(frame.columns):
            raise ValueError("이전 평가 CSV에 동일한 복합 식별자 칼럼이 필요합니다.")
        excluded.update(SeriesKey(tuple(row)) for row in frame.select(*contract.series_columns).unique().iter_rows())
        if sha256_file(filename) != before:
            raise RuntimeError("계약 준비 도중 이전 평가 CSV가 변경되었습니다.")
        exclusion_evidence.append({"file": filename.name, "sha256": before, "kind": "evaluated_cohort"})
    for filename in args.exclude_smoke_summary:
        filename = filename.expanduser().resolve()
        if len(contract.series_columns) != 1:
            raise ValueError("series_ids 형식의 이전 smoke 증적은 단일 식별자만 지원합니다.")
        before = sha256_file(filename)
        content = json.loads(filename.read_text(encoding="utf-8"))
        excluded.update(SeriesKey((value,)) for value in content["series_ids"])
        if sha256_file(filename) != before:
            raise RuntimeError("계약 준비 도중 이전 smoke 증적이 변경되었습니다.")
        exclusion_evidence.append({"file": filename.name, "sha256": before, "kind": "smoke_summary"})
    split = make_split_contract(contract, keys, min(weeks), max(weeks), excluded_from_test=excluded,
                                seed=args.seed, context_length=args.context_length, horizon=args.horizon)
    manifest = split.manifest(contract)
    manifest["allocation"] = {"ratios": [70, 15, 15], "rounding": "floor validation/test; remainder train",
        "algorithm": "seeded role-salted SHA-256 ranking of identity only", "excluded_from_test": [asdict(key) for key in sorted(excluded)],
        "exclusion_evidence": exclusion_evidence, "fresh_holdout_exclusions_provided": bool(exclusion_evidence)}
    # 추가된 배정·제외 기록까지 포함한 최종 해시로 다시 고정.
    from modeling_module.models.Chronos2.training.contracts import manifest_sha256
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    if sha256_file(source) != digest:
        raise RuntimeError("계약 준비 도중 원천 파일이 변경되었습니다.")
    summary = {"status": "contract_prepared_not_trained", "source_rows": rows, "source_series": len(keys),
        "source_sha256": digest, "source_unchanged": True, "zero_semantics": contract.zero_semantics,
        "provenance": contract.provenance, "context_length": split.context_length, "horizon": split.horizon,
        "counts": {name: len(getattr(split, name).series) for name in ("train", "validation", "test")},
        "target_periods": {name: [getattr(split, name).target_start, getattr(split, name).target_end]
                           for name in ("train", "validation", "test")},
        "prior_evaluated_ids_excluded_from_test": len(excluded & set(keys)),
        "test_overlap_with_prior_evaluation": len(set(split.test.series) & excluded),
        "manifest_sha256": manifest["manifest_sha256"], "source_access": "identity/calendar projection; quantity values not loaded",
        "ready_for_training": False, "pending": ["quantity and window eligibility validation", "dataset/sampler/trainer implementation"]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("source_contract.json", asdict(contract)), ("split_manifest.json", manifest), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-columns", nargs="+", default=["oper_part_no"])
    parser.add_argument("--zero-semantics", choices=["unconfirmed", "observed_zero", "explicit_mask"], default="unconfirmed")
    parser.add_argument("--provenance", choices=["unverified", "synthetic", "user_provided"], default="unverified")
    parser.add_argument("--observed-mask-column")
    parser.add_argument("--exclude-evaluated-cohort", type=Path, action="append", default=[])
    parser.add_argument("--exclude-smoke-summary", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=52)
    parser.add_argument("--horizon", type=int, default=27)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))
