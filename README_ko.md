# AGILEQ-Training

[English](README.md)

CARLA 기반 자율주행 데이터 생성 및 모델 학습을 위한 저장소입니다.

| 폴더 | 설명 | 상태 |
| --- | --- | --- |
| [data_gen](data_gen/README_ko.md) | CARLA에서 카메라 이미지, BEV 정답 및 주행 정보를 수집합니다. | 포함 |
| [bevformer](bevformer/README_ko.md) | 다중 카메라 이미지를 이용한 BEV 공간 인지 모델을 학습합니다. | 포함 |
| [TLCformer](TLCformer/README_ko.md) | 연속 전방 영상으로 신호등 GO/STOP 상태와 거리 구간을 학습합니다. | 포함 |
| [AGILEQ-train](AGILEQ-train/README_ko.md) | CrossQ++, CrossQ 등 강화학습 알고리즘으로 CARLA 자율주행 정책을 학습합니다. | 포함 |

설정과 실행 방법은 각 폴더의 README를 참고하고, 명령은 해당 폴더에서 실행합니다. 수집 데이터와 학습 코드를 연결할 때 카메라 이름, 라벨, 공간 좌표 기준을 확인해야 합니다.

학습된 BEV·신호등 인지 모델을 사용하는 주행 정책 평가는 별도 [AGILEQ-Evaluation](https://github.com/SungjinDavidLee/AGILEQ-Evaluation) 저장소에서 관리합니다.

기본 문서는 영어 `README.md`, 한국어 문서는 `README_ko.md`입니다. Python 주석과 docstring은 제거했으며, 제3자 Python 헤더 고지는 각 모듈의 `THIRD_PARTY_NOTICES.txt`에 보존했습니다. Python 캐시와 BEVFormer의 환경별 빌드 산출물은 제외하고 확장 연산 소스는 포함했습니다.
