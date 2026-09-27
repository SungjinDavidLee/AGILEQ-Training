# BEVFormer 학습

[English](README.md)

다중 카메라 기반 BEV 인지 학습 소스입니다. 기본 `train.py`는 `models_carla/bevformer.py`를 사용하며, 첨부된 다른 모델 변형도 유지했습니다.

| 파일 또는 폴더 | 역할 |
| --- | --- |
| `train.py` | 학습, 검증, 체크포인트, W&B 로깅 |
| `data.py` | 6개 카메라, 카메라 기하, BEV 정답, LiDAR 로딩 |
| `models_carla/bevformer.py` | 기본 BEV 인지 모델 |
| `models_carla/` | 첨부된 다른 모델 변형 |
| `models_carla/models/` | InternImage backbone 및 생성 함수 |
| `models_carla/ops/` | Deformable attention Python 바인딩과 C++/CUDA 소스 |
| `configs/` | InternImage 설정 생성 함수 및 YAML 프리셋 |
| `ops_dcnv3/` | DCNv3 Python 바인딩, 설치 스크립트, C++/CUDA 소스 |

## 실행 준비

설정과 DCNv3 소스가 포함되어 있습니다. 학습 전에 실행 환경을 준비해야 합니다.

- `train.py`가 불러오는 `get_config_b`, `get_config_t`는 `configs/config.py`에 있으며 B/T YAML 프리셋도 포함되어 있습니다. YAML 경로가 실행 디렉터리 기준이므로 `bevformer/`에서 실행하세요. 루트의 `config.py`는 다른 설정 코드입니다.
- InternImage에서 사용하는 `ops_dcnv3`가 포함되어 있습니다. 네이티브 `DCNv3` 확장 연산은 대상 환경에서 빌드·설치해야 합니다.
- CUDA에 의존하는 코드가 있으므로 PyTorch, torchvision, CUDA toolkit, 컴파일러 버전을 맞춰야 합니다. 완전한 실행 환경의 버전은 고정되어 있지 않습니다.
- NumPy, OpenCV, PyYAML, yacs, pyquaternion, tqdm, segmentation-models-pytorch, efficientnet-pytorch, timm, wandb 등을 사용합니다. DCNv3 래퍼는 호환되는 setuptools의 `pkg_resources`도 사용합니다. 다른 모델 변형은 추가 의존성이 있을 수 있습니다.

Python 3.7/3.9용 Linux 바이너리, object 파일, 빌드 메타데이터, egg, Python 캐시는 제외했습니다. 호환되는 PyTorch, CUDA, 컴파일러, setuptools, wheel을 준비한 뒤 `bevformer/`에서 두 확장 연산을 설치합니다.

```bash
python -m pip install yacs
python -m pip install --no-build-isolation ./ops_dcnv3
python -m pip install --no-build-isolation ./models_carla/ops
```

DCNv3 래퍼는 최상위 `DCNv3` 확장 모듈과 설치된 패키지 버전을 읽으므로 제자리 빌드만 하지 말고 설치해야 합니다. 제공된 빌드 스크립트는 사용 가능한 GPU와 CUDA toolkit이 있는 CUDA PyTorch 환경을 요구합니다.

환경과 데이터를 준비한 뒤 실행합니다.

```bash
python train.py --wandb_mode disabled
```

현재 코드는 disabled 모드에서도 wandb 설치가 필요합니다. `WANDB_API_KEY`에 `???`를 대입하며 기존 프로젝트·entity 기본값이 있으므로 온라인 로깅 전에 설정해야 합니다. 이번 업로드에서는 주석과 docstring을 제거하고 학습 동작은 유지했습니다.

## 데이터

명령은 이 폴더에서 실행합니다. `data.py`는 `data/train/Town01`~`Town05`, 대응되는 `data/val` 경로와 `data/sensor_config.yaml`을 사용합니다.

카메라 이름은 `left_cam`, `front_cam`, `right_cam`, `rear_left_cam`, `rear_cam`, `rear_right_cam`이며 이미지는 `<camera>_rgb` 폴더에서 읽습니다. BEV 정답, 메타데이터, LiDAR도 읽으므로 전체 규약은 `data.py`를 확인하세요. 학습·검증 scene은 분리해야 합니다.

`data_gen` 결과와 바로 호환된다고 가정하면 안 됩니다. 카메라 이름을 일치시키고 필요한 LiDAR를 수집해야 하며, 모델의 고정 공간 범위와 수집기의 40m×40m BEV 범위를 맞춰야 합니다. 현재 기본 모델의 수평 범위는 -50~50m입니다.

## 검증

추가된 설정·DCNv3 Python 소스까지 구문 검사와 주석·docstring 제거 전후의 실행 AST 동일성을 확인했습니다. 두 설정 생성 함수와 참조 YAML도 포함되어 있습니다. CUDA 확장 빌드와 실제 학습은 수행하지 않았습니다. Python 저작권·라이선스 헤더는 `THIRD_PARTY_NOTICES.txt`에 보존했고 C++/CUDA 고지는 유지했습니다.
