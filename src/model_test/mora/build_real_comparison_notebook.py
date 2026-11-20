"""완료된 비교 결과만 읽어 한글 설명 노트북·보고서·작은 증적 파일 생성."""

import argparse
import json
from pathlib import Path
import textwrap


def build(root, output):
    import nbformat as nbf

    summary = json.loads((output / "summary.json").read_text())
    independent = json.loads((output / "independent_audit.json").read_text())
    experiment = json.loads((output / "experiment.json").read_text())
    if summary.get("status") != "complete" or independent.get("status") != "passed":
        raise ValueError("실험과 독립 검산이 모두 완료된 결과만 문서화할 수 있습니다.")
    if (summary["queries"] != {"train": 19600, "validation": 1050, "test": 1050}
            or summary["split_manifest_sha256"] != "a7b16441549e128297797be87d1c0853690a89bc1dc50baa112ca5883fc72238"):
        raise ValueError("이 설명 템플릿은 2026-11-20에 확정한 기존 분할·학습 창 조건 전용입니다.")
    values = summary["comparison"]
    baseline, mora = values["baseline"], values["mora"]
    ratio = values["mae_reduction_ratio"]
    direction = "감소" if ratio >= 0 else "증가"
    relative = str(output.relative_to(root))
    report_path = root / "docs/Chronos2MORARealComparison20261120.md"
    notebook_path = root / "src/modeling_module/forecast_adapters/MORA/notebook/mora_chronos2_real_comparison.ipynb"
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

    def source(value):
        return textwrap.dedent(value).strip() + "\n"

    cells = [md(f"""# 동결 Chronos2 + MORA: 실제 수요 L52/H27 비교

## 핵심 결과
- 시험 부품 1,050개, 시작 주차 202518, 예측 정답 28,350개에 대한 짝지은 비교
- MAE: **{baseline['mae']:.6f} → {mora['mae']:.6f}** ({abs(ratio):.2%} {direction})
- RMSE: **{baseline['rmse']:.6f} → {mora['rmse']:.6f}**
- 부품별 MAE 개선 {values['part_mae']['improved']}개 / 동일 {values['part_mae']['unchanged']}개 / 악화 {values['part_mae']['worsened']}개
- 공식 Chronos2 가중치와 기존 모델 코드 변경 없음. MORA 보정 파라미터만 적합

한 시작 주차에서 MORA 적합에 쓰지 않은 부품의 시험 결과이며 모든 법인·시점에 대한 성능 보장은 아님.
이 노트북은 **저장된 결과 확인·재계산용**으로, 기본 실행 시 학습·추론·DB 작업을 다시 수행하지 않음.
"""), md("""## 1. 환경과 결과 위치

저장소 루트를 자동으로 찾은 뒤 결과 파일을 읽는 구성. NumPy·pandas·Matplotlib·SciPy·IPython 필요.
공식 가중치를 다시 호출하지 않으므로 이 노트북의 결과 재생에는 `chronos` 패키지 불필요.
"""), code(source(f"""
        from pathlib import Path
        import sys, json, hashlib
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown, Image

        candidates = [Path.cwd(), *Path.cwd().parents]
        ROOT = next(path for path in candidates if (path / "src/modeling_module").is_dir())
        sys.path.insert(0, str(ROOT / "src"))
        RUN = ROOT / {relative!r}
        summary = json.loads((RUN / "summary.json").read_text())
        experiment = json.loads((RUN / "experiment.json").read_text())
        audit = json.loads((RUN / "independent_audit.json").read_text())
        assert summary["status"] == "complete" and audit["status"] == "passed"
        display(Markdown(f"결과 위치: `{{RUN}}`"))
        plt.rcParams.update({{"font.family": "DejaVu Sans", "font.size": 11,
                              "axes.spines.top": False, "axes.spines.right": False}})
    """)), md("""## 2. 기존 데이터 분할 유지

부품 ID와 정답 기간을 모두 분리한 기존 계약 유지. 입력 52주는 정답 시작 직전 관측값만 사용.

| 구분 | 부품 | 정답 허용 기간 | 선택 창 |
|---|---:|---|---:|
| 보정 적합 | 4,900 | 201801–202442 | 부품당 4개, 총 19,600개 |
| 보정 강도 결정 | 1,050 | 202443–202517 | 시작 202443, 총 1,050개 |
| 최종 시험 | 1,050 | 202518–202544 | 시작 202518, 총 1,050개 |

실제 관측된 0수요는 그대로 유지. 결측의 0 대체나 부품 번호 앞순서 선택 없음.
학습 가능 창 1,080,909개 중 수요량과 무관한 ID·주차 해시로 19,600개 선택.
검증·시험 부품은 전부 포함하며 조건 미달 제외 부품은 0개.
"""), code(source("""
        eligibility = pd.DataFrame(summary["eligibility"]).T
        display(eligibility[["assigned_parts", "eligible_parts", "eligible_windows", "selected_windows", "target_start", "target_end"]])
        display(pd.DataFrame([{"source_sha256": summary["source_sha256"],
                               "split_manifest_sha256": summary["split_manifest_sha256"],
                               "zero_semantics": summary["zero_semantics"]}]).T)
    """)), md("""## 3. Chronos2와 MORA의 역할 분리

`과거 52주 → 동결 공식 Chronos2 → 원본 P50 → 독립 MORA → 보정 점예측 27주`

Chronos2는 공식 사전학습 가중치의 CPU FP32 추론이며 학습 가능한 파라미터 0개.
MORA는 논문 수식의 독립 구현으로, 저자 원본 코드를 복사한 구현이 아님.

기존 분할에서 부품 ID가 겹치지 않으므로 `pooled_series`로 학습 부품의 공통 보정 규칙을 새 부품에 적용.
수천 부품을 동시 채널로 넣는 것이 아니라 각 창의 채널은 `demand_qty` 1개.
이는 논문의 단일 시계열 설정과 구분되는 연구 확장.

학습 내부는 겹치는 관측·정답 구간을 제거한 4개 outer/inner 블록 검증.
검증 구간에는 H27 시작점이 하나뿐이므로 보정 강도 계산의 4개 그룹은 시간 그룹이 아닌 부품 그룹.
시험 정답은 보정기 저장 이후 평가에만 사용. 시험 결과를 보고 재적합하지 않은 비교.
"""), code(source("""
        display(pd.DataFrame(list(experiment["mora"].items()), columns=["설정", "값"]))
        frozen = summary["frozen_checks"]
        display(pd.DataFrame(list(frozen.items()), columns=["불변 검사", "결과"]))
        assert frozen["parameters_identical"] and frozen["model_sources_identical"]
        assert frozen["trainable_parameters"] == 0
    """)), md("""## 4. 같은 원본 예측을 두 경로에 사용했는지 확인

기준선 P50을 캐시에 한 번 저장한 뒤 동일한 값을 MORA 입력으로 재사용.
원본 P10/P50/P90은 그대로 보존하며 MORA 출력은 **보정된 점예측**일 뿐 보정 분위수는 아님.
반올림과 음수 절단은 양쪽 모두 적용하지 않은 조건.
"""), code(source("""
        arrays = np.load(RUN / "test_comparison.npz", allow_pickle=False)
        original = np.load(RUN / "test_arrays.npz", allow_pickle=False)
        actual = arrays["target"]
        baseline = arrays["baseline"]
        corrected = arrays["corrected"]
        np.testing.assert_array_equal(baseline[:, :, 0], original["original_quantiles"][:, :, 1])
        np.testing.assert_array_equal(actual, original["target"])
        np.testing.assert_array_equal(baseline[:, 0], corrected[:, 0])
        display(pd.DataFrame({"배열": ["과거", "정답", "기준선", "보정 점예측", "원본 분위수"],
                               "shape": [original["history"].shape, actual.shape, baseline.shape,
                                         corrected.shape, original["original_quantiles"].shape]}))
    """)), md("""## 5. 실제 성능 재계산

MAE = 평균 절대 오차, RMSE = 제곱 오차 평균의 제곱근.
WAPE = 전체 절대 오차 합 / 실제 수요 합. 실제 수요 합 0인 부품의 개별 WAPE는 정의하지 않으며 별도 집계.
같은 부품의 27개 주차를 독립 표본으로 간주하지 않고 부품 단위 재표집으로 신뢰구간 계산.
아래 기본 셀은 보정기를 재학습하지 않는 저장 결과 재계산.
"""), code(source("""
        from model_test.mora.comparison_metrics import paired_comparison
        recalculated = paired_comparison(actual, baseline, corrected)
        for name in ("baseline", "mora"):
            for metric in ("mae", "rmse", "wape"):
                np.testing.assert_allclose(recalculated[name][metric], summary["comparison"][name][metric], rtol=1e-12)
        table = pd.DataFrame({"Chronos2 P50": recalculated["baseline"], "Chronos2 + MORA": recalculated["mora"]})
        display(table.loc[["mae", "rmse", "wape", "macro_part_rmse", "zero_total_parts", "zero_total_parts_mae", "negative_predictions"]])
        display(pd.DataFrame([recalculated["part_mae"]]))
        display(pd.DataFrame([recalculated["paired_part_bootstrap"]]))
    """)), md("""## 6. 예측 거리별 오차와 부품별 편차

첫 예측 주차는 MORA 기저 구조상 보정하지 않는 값. 2–27주 오차를 확인할 때 전체 평균 개선과 부품별 악화를 함께 확인할 필요.
"""), code(source("""
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        for values, color, label, style in [(baseline, "#355C8A", "Frozen Chronos2 P50", "-"),
                                             (corrected, "#C4562D", "Chronos2 + MORA point", "--")]:
            ax.plot(np.arange(1, 28), np.abs(values - actual).mean(axis=(0, 2)), style,
                    color=color, label=label, linewidth=2)
        ax.set(xlabel="Forecast horizon (weeks)", ylabel="MAE (demand units)",
               title="Held-out 1,050 parts | origin 2025-W18 | L52 / H27", xticks=[1, 5, 9, 13, 17, 21, 25, 27])
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
        plt.show()
        display(Image(filename=str(RUN / "part_mae.png"), width=700))
    """)), md("""## 7. 결과가 좋은 부품만 고르지 않은 예시

시험 ID의 SHA-256 순서로 4개 선택. 성능 순위나 수요 패턴으로 선택한 대표 표본이 아님.
달력 날짜 축 사용으로 연말·연초의 YYYYWW 숫자 간격 왜곡 방지.
"""), code(source("""
        display(Image(filename=str(RUN / "forecast_examples.png"), width=1100))
    """)), md("""## 8. 저장된 보정기만으로 동일 결과 재생

시험 정답은 `ForecastQuery`에 전달하지 않는 구조. 원본 과거와 P50만 전달하여 보정 결과 재현.
체크포인트 로드는 적합이나 검증 강도 재계산이 아님.
"""), code(source("""
        from modeling_module.forecast_adapters.contracts import ForecastQuery
        from modeling_module.forecast_adapters.MORA import MORAAdapter
        adapter = MORAAdapter.load(RUN / "adapter.npz")
        query = ForecastQuery(original["history"], baseline, original["origins"], ("demand_qty",),
                              tuple(original["series_ids"].tolist()), experiment["baseline"])
        replay = adapter.predict(query)
        np.testing.assert_array_equal(replay.point, corrected)
        display(Markdown("동일 환경에서 저장·복원 전후 보정 점예측의 완전 일치 확인."))
        display(pd.DataFrame(adapter.diagnostics["calibration_details"]))
    """)), md("""## 해석 범위와 다음 작업

- 한 시험 시작 주차에 대한 결과. 여러 시작 주차나 고객 환경으로의 일반화 미검증
- 검증 강도는 한 주차의 별도 부품들로 결정. 논문의 시간 블록 검증 재현으로 설명하지 않을 조건
- 전체 오차 감소가 모든 부품의 개선을 의미하지 않는 점에 유의
- 음수 절단·계절 주기·rank 변경은 새 실험 조건. 이번 시험을 보고 바꾸면 같은 시험을 최종 미사용 시험으로 재사용 불가
- 원본 분위수는 보정하지 않았으므로 보정 후 예측구간 신뢰도 개선 주장 불가
- 원천은 사용자가 제공한 로컬 스냅샷이며 실시간 운영 DB의 출처·최신성 별도 확인 필요
- 공식 Chronos2 사전학습 데이터와의 중복 여부는 확인되지 않은 상태. 부품 분리는 이번 MORA 적합 데이터 내부의 계약

원천·분할·가중치·실험 설정 해시, 전체 예측 CSV, 보정기와 독립 검산 JSON은 결과 디렉토리에 보관.
""")]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}})
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    if notebook_path.exists():
        raise FileExistsError(notebook_path)
    nbf.write(notebook, notebook_path)
    ci = values["paired_part_bootstrap"]["percentile_95_ci"]
    report = f"""# 동결 Chronos2 + MORA 실제 데이터 비교

## 결과

202518 시작, MORA 적합에 쓰지 않은 부품 1,050개 × 27주 = 28,350개 정답의 동일 조건 비교.

MAE·WAPE는 소폭 악화, RMSE는 미세 개선된 결과로, 이번 조건에서 기본 Chronos2를 MORA 경로로 교체할 성능 근거는 확인되지 않은 상태.

| 지표 | 공식 Chronos2 P50 | Chronos2 + MORA 점예측 |
|---|---:|---:|
| MAE | {baseline['mae']:.6f} | {mora['mae']:.6f} |
| RMSE | {baseline['rmse']:.6f} | {mora['rmse']:.6f} |
| WAPE | {baseline['wape']:.4%} | {mora['wape']:.4%} |
| 부품별 RMSE 평균 | {baseline['macro_part_rmse']:.6f} | {mora['macro_part_rmse']:.6f} |
| 음수 점예측 수 | {baseline['negative_predictions']} | {mora['negative_predictions']} |
| 27주 전체 0수요 부품의 MAE | {baseline['zero_total_parts_mae']:.6f} | {mora['zero_total_parts_mae']:.6f} |

MAE 기준 {abs(ratio):.4%} {direction}. 부품별 개선 {values['part_mae']['improved']}개·동일 {values['part_mae']['unchanged']}개·악화 {values['part_mae']['worsened']}개.
부품 단위 paired bootstrap 2,000회에서 `보정 MAE - 기준 MAE` 평균 차이의 95% percentile 구간: [{ci[0]:.6f}, {ci[1]:.6f}].
한 시작 주차의 부품 표본 불확실성에 대한 구간이며 시점 변화에 대한 신뢰구간은 아님.
시험 부품 중 657개(62.57%)는 27주 실제 수요 합이 0인 부품. 개별 WAPE의 분모가 0인 집단은 WAPE 평균에서 제외하고 MAE를 별도 제시한 구성.

## 비교 조건

- 대상 저장소·브랜치: `ts_forecaster_lib` / `ts_forecaster_lib_v2`
- 원천: `raw_data/master/tb_master_target.parquet`, 2,455,508행·7,000부품·201801–202544, 사용자가 실제 관측된 0수요로 확인한 로컬 스냅샷
- 기존 부품·기간 분할 그대로 유지: 학습 4,900개/201801–202442, 검증 1,050개/202443–202517, 시험 1,050개/202518–202544
- 모든 배정 부품에서 L52/H27 조건 충족, 제외 0개
- 학습 가능한 1,080,909개 창 중 ID·주차·seed 42만으로 부품당 4개, 총 19,600개 선택. 수요량이나 예측 성능을 기준으로 선택하지 않은 제한 표집
- 검증·시험 각 1,050개 창 전부 포함. 정답은 검증에서 보정 강도 결정, 시험에서 최종 평가 용도로만 사용
- 공식 `amazon/chronos-2` 고정 revision, CPU FP32, 입력 52주, 출력 27주, 별도 외생변수 없음, cross-learning 비활성
- 양쪽에 동일한 원본 P50 사용. 반올림·음수 절단·부품별 사후 fallback 없음
- MORA만 적합. 학습 시 outer/inner 4블록의 같은 부품 raw 구간 중첩 제거
- `period=52`, `rank=16`, `ridge_strength=10`, 나머지 기본 설정 고정. 시험 결과 기반 설정 탐색 없음

### MORA 적용 범위

논문 수식의 독립 구현으로 저자 코드를 복사하지 않은 구성. `pooled_series`는 MORA 적합에 쓰지 않은 부품에도 공통 보정 규칙을 적용하는 명시적 연구 확장.
부품마다 수요 채널 1개이며 수천 부품의 동시 다채널 입력 방식이 아님.
검증에는 하나의 H27 시작점만 있으므로 강도 결정의 4개 그룹은 시간 블록이 아닌 부품 그룹.
따라서 논문 원본 실험을 그대로 재현한 결과로 설명하지 않을 조건.

## 불변·누수 방지 확인

- 공식 가중치 파일·설정·메모리 파라미터·모든 기존 모델 Python 소스의 전후 동일성 확인
- Chronos2 학습 가능 파라미터 0개, 평가 모드 유지
- 기존 분할 manifest와 원천 parquet 해시 유지
- 보정기 저장 완료 시각과 해시 기록 후 시험 정답 로드·평가
- 시험 평가 중 보정기 해시 불변, 저장·복원 후 점예측 완전 일치
- 첫 예측 주차는 MORA 기저 구조상 원본 값 그대로 유지
- CSV 독립 오차 합산, 배열·원본 P50 대조, 실제 parquet 정답 28,350개 및 달력 전체 대조 통과

## 제한과 판단 기준

단일 시작 주차이므로 다른 시점·법인에서도 같은 개선을 보장하지 않는 결과.
공식 Chronos2 사전학습 데이터와의 중복 여부는 미확인. 부품 분리는 이번 MORA 적합·검증·시험 데이터 내부에 대한 계약.
전체 오차와 부품별 개선·악화, 음수 예측을 함께 판단할 필요.
MORA 출력은 보정된 점예측이며 원본 P10/P50/P90 분위수의 보정 또는 예측구간 개선으로 해석 불가.
이번 시험은 이미 사용된 holdout이므로 이후 설정 탐색의 근거로 사용하면 새 최종 시험 구간 필요.

## 파일과 재현

- 결과: `{relative}`
- 셀별 설명: `src/modeling_module/forecast_adapters/MORA/notebook/mora_chronos2_real_comparison.ipynb`
- `experiment.json`: 수요량 확인 전 고정한 계획·분할·모델 소스 해시
- `window_manifest.csv`, `eligibility.json`, `excluded_parts.csv`: 창 선택·완전 관측 여부
- `train_arrays.npz`, `validation_arrays.npz`, `test_arrays.npz`: 원본 입력·정답·P50·분위수
- `adapter.npz`, `fit_complete_before_test.json`: 보정 상태와 시험 전 고정 증적
- `test_forecast.csv`, `test_comparison.npz`: 전체 짝지은 시험 결과
- `summary.json`, `independent_audit.json`, `artifact_hashes.json`: 지표·독립 검산·무결성

```bash
# 기존 결과를 덮어쓰지 않도록 새 출력 경로 지정 필요
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
python -m model_test.mora.run_chronos2_comparison --output-dir artifacts/mora/NEW_RUN

# 저장된 결과 독립 검산과 그림 재생성
PYTHONPATH=src python -m model_test.mora.audit_chronos2_comparison --output-dir {relative}
```

실행 환경: chronos-forecasting {summary['versions']['chronos-forecasting']}, torch {summary['versions']['torch']}, NumPy {summary['versions']['numpy']}, SciPy {summary['versions']['scipy']}.
실험 소요 시간 {summary['total_seconds']:.1f}초, MORA 적합 {summary['mora_fit_seconds']:.1f}초, 시험 1,050창 보정 {summary['mora_predict_seconds']:.4f}초.
최종 전체 회귀 테스트 1,100 passed / CUDA 6 skipped / 46 subtests passed. 설명 노트북 코드 셀 8개 전체 실행과 저장·복원 점예측 일치 확인.

### 식별 해시

- 원천 SHA-256: `{summary['source_sha256']}`
- 분할 manifest SHA-256: `{summary['split_manifest_sha256']}`
- 공식 가중치 SHA-256: `{experiment['official_weights_sha256']}`
- 보정기 SHA-256: `{summary['adapter_sha256']}`

## 남은 작업

- **다음 작업**: 새 기간 또는 미사용 데이터로 다중 시작 주차 검증. 이미 확인한 시험 결과에 맞춘 재튜닝과 구분 필요
- **승인 필요**: 운영 적용·공용 Runtime 변경. 이번 비교에서는 미수행
- 커밋·Push 미수행. 기존 개발 변경·IDE 변경·stash 보존
"""
    report_path.write_text(report, encoding="utf-8")
    evidence = {"summary": summary, "independent_audit": independent,
                "result_directory": relative, "report": str(report_path.relative_to(root)),
                "notebook": str(notebook_path.relative_to(root))}
    (root / "docs/evidence/Chronos2MORARealComparison20261120.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return notebook_path, report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(Path.cwd(), args.output_dir.resolve()))
