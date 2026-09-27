# AGILEQ-train

[English](README.md) · [학습 저장소](../README_ko.md)

CARLA에서 자율주행 정책을 강화학습하는 모듈입니다. CrossQ++ 구현, 주행 환경, 보상 함수 및 BEV 지도 자료를 포함합니다. 설정 이름은 `PPO`, `SAC`, `DDPG`, `TD3`, `TQC`, `crossq`, `crossq_pp`입니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `train.py` | 주행 정책 학습·재개 및 체크포인트 저장 |
| `config.py` | 알고리즘 설정, 관측 특징 및 보상 파라미터 |
| `crossq_pp/` | CrossQ++ 알고리즘과 정책 코드 |
| `carla_env/` | CARLA 환경, 경로 계획, 센서, 보상 및 래퍼 |
| `BEV/` | 지도 렌더링 유틸리티, 지도 이미지 및 좌표 오프셋 |
| `start_train.sh` | 단일 학습 실행 설정 |
| `run_experiments.py` | 여러 학습 실험을 실행하는 스크립트 |
| `environment_agileq_train.yml` | Conda 환경 내보내기 파일 |

## 환경 설정

명령은 `AGILEQ-Training/AGILEQ-train/`에서 실행합니다. Linux, 필요한 Town 맵이 설치된 CARLA 0.9.15, NVIDIA GPU가 필요합니다. 제공 환경은 Python 3.9와 PyTorch 2.7 및 CUDA 12.8 패키지를 사용하며, 코드에서 CUDA를 직접 지정합니다.

```bash
conda env create -n agileq_train -f environment_agileq_train.yml
conda activate agileq_train
export CARLA_ROOT=/absolute/path/to/CARLA_0.9.15
```

YAML에는 특정 플랫폼의 빌드와 CUDA wheel 버전이 기록되어 있습니다. 해당 빌드를 설치할 수 없다면 사용 환경에 맞게 패키지 공급 경로나 버전을 조정해야 합니다. 코드에서 불러오는 `CrossQ`와 `BatchRenorm1d`를 제공하는 `sb3_contrib` 환경이 필요합니다.

학습 전에 `train.py`의 W&B `entity`와 `project`를 자신의 작업 공간으로 수정하고 `wandb login`으로 인증합니다. 오프라인 기록은 `WANDB_MODE=offline`으로 설정합니다.

환경은 `CARLA_ROOT/CarlaUE4.sh`로 CARLA를 실행합니다. 기본값인 `activate_model=False`에서는 시뮬레이터에서 생성한 BEV 관측을 사용합니다. 선택적으로 인지 모델 분기를 켜면 `run/cls/best_loss.pt`가 필요하며, 가중치는 포함되어 있지 않습니다.

## 학습

```bash
python train.py \
  --config crossq_pp \
  --town town01 \
  --port 2000 \
  --total_timesteps 600000 \
  --save_dir ./results \
  --no_render
```

학습을 재개하려면 `--reload_model /absolute/path/to/model_100000_steps.zip`을 추가합니다. 알고리즘 설정과 관측 공간은 체크포인트와 호환되어야 합니다. `--fps` 기본값은 15, `--num_checkpoints` 기본값은 12입니다. `--no_render`는 환경 화면 표시를 끕니다.

Town 이름은 `BEV/`의 폴더 이름과 대소문자까지 일치해야 합니다. 포함된 이름은 `town01`, `town02`, `town03`, `town04`, `town05`, `town07`, `town10HD`이며, 대응하는 CARLA 맵도 설치되어 있어야 합니다.

`start_train.sh`를 사용하려면 내부 `CARLA_ROOT`와 실행 설정을 먼저 수정합니다. 일괄 실행은 `run_experiments.py`의 CARLA 경로, Town 목록 및 실험 목록을 수정한 뒤 사용합니다. 일괄 실행 스크립트는 실행 사이에 CARLA 서버 프로세스를 종료합니다.

## 결과 및 평가

체크포인트, `config.json`, CSV/TensorBoard 로그, `training_log_dw_reward.json`은 `--save_dir/<실행 폴더>/`에 저장됩니다. W&B의 추가 모델 스냅샷은 `models/<실행 폴더>/`에 저장됩니다.

학습된 정책은 [AGILEQ-Evaluation](https://github.com/SungjinDavidLee/AGILEQ-Evaluation)에서 평가합니다. 평가에는 호환되는 BEV·신호등 인지 모델 체크포인트도 필요하며, 이 폴더에는 가중치를 포함하지 않습니다.

Python 주석과 docstring은 제거했으며 원본의 제3자 헤더 고지는 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)에 보존했습니다. 반영 과정에서 Python 문법과 실행 구문의 보존을 확인했으며, 실제 CARLA 학습은 실행하지 않았습니다.
