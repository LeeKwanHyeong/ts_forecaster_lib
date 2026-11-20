# 내부 수식 구현

이 디렉토리의 Python 소스는 논문 수식으로 독립 작성한 코드.
공개 저장소의 원본 코드나 라이선스를 복사한 상태가 아님.

읽는 순서:

1. `basis.py`: 구간 평균 A와 첫 예측 위치가 0인 직교 기저 U
2. `features.py`: context만으로 만드는 compact 78개 특징
3. `coarse_future.py`: 두 외삽과 absolute/centered ridge
4. `stagewise.py`: 비중심 모멘트 매핑 K와 reliability gain G
5. `calibration.py`: 검증 전용 그룹 scale S

부록 C.1·C.3·C.4·C.5와 파일 대응·구현 차이는 [../SOURCE.md](../SOURCE.md) 참고.
저자 코드와의 비트 단위 대조 또는 논문 전체 재현을 수행한 것으로 표시하지 않는 기준.
