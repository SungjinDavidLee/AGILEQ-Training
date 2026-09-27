# BEVFormer

[English](README.md)

마지막으로 제공한 모델과 DDP 학습 코드로 교체했습니다. 단일 `BEVFormer`와 단일 `train.py`를 사용하며 출력은 도로, 점선, 실선, 차량, 보행자, 정지선의 6개 BEV logits입니다.

## 구조

| 경로 | 역할 |
| --- | --- |
| `models/bevformer.py` | 기하, 인코더, attention, 6개 head 디코더를 포함한 단일 `BEVFormer` |
| `models/backbone/` | InternImage 구현 및 생성 함수 |
| `train.py` | 단일 GPU/DDP 학습, 검증, 타운별 지표, 체크포인트 |
| `configs/backbone.py` | `load_backbone_config()`와 InternImage 기본값 |
| `configs/internimage_t_1k_224.yaml` | 실제 사용하는 InternImage-T 설정 |
| `ops/dcnv3/` | DCNv3 래퍼와 C++/CUDA 확장 |
| `ops/deformable_attention/` | Deformable attention 래퍼와 C++/CUDA 확장 |
| `data.py` | 6개 카메라, 6종 BEV 정답, 타운 정보를 반환하는 학습 데이터로더 |


## 데이터셋

제공된 데이터로더를 `data.py`로 반영했으며 train이 직접 불러옵니다.

```python
from data import Drive_Dataset

train_set = Drive_Dataset(train=True, conf=conf, v=2)
val_set = Drive_Dataset(train=False, conf=conf, v=2)
```

샘플 반환 순서는 다음과 같아야 합니다.

```text
bev_imgs, cam_pam, road_gt, lane_broken_gt, lane_solid_gt,
veh_gt, ped_gt, stop_gt, town_idx, town_name
```

`cam_pam`은 `intrins`, `rots`, `trans`를 제공하고 영상 배치는 `[B,N,3,H,W]`입니다. Train은 `data.py`의 `TOWNS`를 함께 불러오므로 `Town01`, `Town02`, `Town03`, `Town04`, `Town05`의 인덱스 순서를 공유합니다.

`data/train/Town01`~`Town05`, 대응되는 `data/val` 폴더와 `data/sensor_config.yaml`을 읽습니다. Scene별로 정렬된 첫 5개 파일을 제외합니다. 카메라 순서는 `right_cam`, `front_cam`, `left_cam`, `rear_right_cam`, `rear_cam`, `rear_left_cam`이며 영상 폴더는 `<camera>_rgb`입니다.

BEV 정답 폴더는 `das`, `lane_broken`, `solid`, `vehicle_mask`, `walker_mask`, `stopline`입니다. 실선 폴더 이름은 `lane_solid`가 아닌 `solid`입니다. 값이 255인 흰색 픽셀을 정답으로 사용합니다. 학습에서는 6개 카메라에 같은 색상 증강을 적용하고 검증에는 적용하지 않습니다. 현재 샘플 경로는 `get_lidar()`를 호출하지 않으므로 LiDAR 파일은 필요하지 않습니다.

점선·실선 정답은 별도로 준비해야 하며 수집기의 합쳐진 `lane` mask를 이 로더가 자동 분리하지 않습니다. 학습·검증 scene은 분리하세요.

## 환경

명령은 `bevformer/`에서 실행합니다. 호환되는 PyTorch, torchvision, CUDA toolkit, C++ 컴파일러, setuptools, wheel을 먼저 준비하세요. 기존 코드에는 완전히 고정된 환경 정보가 없습니다.

```bash
python -m pip install numpy opencv-python PyYAML yacs pyquaternion tqdm segmentation-models-pytorch efficientnet-pytorch timm wandb
python -m pip install --no-build-isolation ./ops/dcnv3
python -m pip install --no-build-isolation ./ops/deformable_attention
```

DCNv3에서 사용하는 `pkg_resources`를 제공하는 setuptools가 필요합니다. 두 빌드 스크립트는 CUDA PyTorch, 사용 가능한 GPU, CUDA toolkit을 요구합니다. DCNv3는 설치된 패키지 버전을 읽으므로 제자리 빌드뿐 아니라 설치해야 합니다. 확장 설치 시 최상위 `functions`, `modules`를 중복 설치하지 않고 저장소 `ops`에서 Python 래퍼를 읽도록 정리했습니다.

`model` 키 아래 가중치가 있는 InternImage-T 체크포인트를 `bevformer/internimage_t_1k_224.pth`에 준비하세요. 디코더는 `EfficientNet.from_pretrained("efficientnet-b0")`를 사용하므로 다운로드가 발생할 수 있습니다. 사전학습 파일은 포함되어 있지 않습니다.

## 학습

데이터셋과 실행 환경을 준비한 뒤 실행합니다.

```bash
python train.py --wandb_mode disabled
python train.py --wandb_mode disabled --resume
torchrun --standalone --nproc_per_node=2 train.py --wandb_mode disabled
```

`--batch_size`는 GPU당 배치입니다. 제공된 코드의 gradient accumulation, 재개 방식, 6개 정답 손실, 타운별 검증을 유지했으며 기본 결과 경로는 `run_422/bev_<v>`입니다.

Disabled 모드에서도 wandb 설치는 필요합니다. 첨부 train의 `WANDB_API_KEY`에는 `?`가 대입되고 기존 project/entity 기본값도 있으므로 온라인 로깅 전 해당 대입을 제거하고 본인의 인증·계정 설정을 적용하세요.

Backbone YAML은 `configs/backbone.py` 기준으로 찾습니다. 데이터·가중치·결과 경로는 실행 디렉터리 기준입니다.

## 모델 및 호환성

새 모델의 InternImage-T, attention 6개 층, EfficientNet-B0, 추가 디코더 블록의 GroupNorm을 유지했습니다. BEV는 200×200, 수직 샘플은 8개, 수평 범위는 -20~20m, scene centroid는 `(0, 1, 0)`입니다. 40m 범위는 수집기의 명목 범위와 같지만 카메라 보정·좌표 기준은 실제 데이터로 확인해야 합니다.

```python
from configs import load_backbone_config
from models import BEVFormer

model = BEVFormer({"config": load_backbone_config()}).cuda()
road, lane_broken, lane_solid, vehicle, pedestrian, stopline = model(images, camera_parameters)
```

새 클래스명과 폴더명으로 바꿔도 마지막 첨부 모델의 매개변수 속성명과 `state_dict` 키 구조는 유지됩니다. 이전 5개 출력/BatchNorm 모델의 체크포인트는 새 6개 출력/GroupNorm 모델과 호환된다고 가정하면 안 됩니다. 옛 모듈 경로에 의존하는 모델 전체 pickle은 별도 이전이 필요합니다.

## 검증

Python 구문, 첨부 데이터로더의 주석·docstring 제거 전후 AST, 샘플 반환 10개 항목, train의 data import 및 타운 목록 공유를 확인했습니다. 모델·train 연결은 이전 작업에서 첨부 원본과 대조했습니다. 이번 작업에서 CUDA 빌드, 실제 데이터 로딩, 전체 학습은 실행하지 않았습니다.

확장 연산 설치 후 기존 검사는 다음처럼 실행합니다.

```bash
python -m ops.dcnv3.test
python -m ops.deformable_attention.test
```

Python 주석과 docstring은 제거했습니다. 제3자 Python 고지는 `THIRD_PARTY_NOTICES.txt`, C++/CUDA 고지는 소스에 유지했습니다.
