# AGILEQ-Training

CARLA 기반 자율주행 데이터 생성 및 모델 학습을 위한 저장소입니다.

| 폴더 | 설명 | 상태 |
| --- | --- | --- |
| `data_gen/` | CARLA에서 카메라 이미지, BEV 정답 및 주행 정보를 수집합니다. | 포함 |
| `bevformer/` | 다중 카메라 이미지를 이용한 BEV 공간 인지 모델을 학습합니다. | 추가 예정 |
| `TLCformer/` | 신호등 인식 모델을 학습합니다. | 추가 예정 |
| `RL_Training/` | 강화학습 기반 자율주행 정책을 학습합니다. | 추가 예정 |

데이터 생성 설정과 실행 방법은 [data_gen/README.md](data_gen/README.md)를 참고하세요.
