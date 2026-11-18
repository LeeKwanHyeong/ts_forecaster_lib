"""삼성 실험의 기존 파일명을 유지하되 실행은 v2 공개 API 경로에 위임."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_test.exogenous_test.run_samsung_gcs_patchtst_sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
