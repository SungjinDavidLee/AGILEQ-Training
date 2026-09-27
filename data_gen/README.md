# CARLA 이미지 및 최종 BEV 수집

`carla_run.py`가 `data/sensor_config.yaml`을 읽어 CARLA를 실행하고, 카메라 이미지와 최종 BEV를 같은 프레임에서 저장합니다. 별도의 GUI 설정이나 수집 종료 후 BEV 후처리가 필요하지 않습니다.

## 실행

CARLA 서버 버전과 일치하는 CARLA Python 패키지를 먼저 설치하고 나머지 의존성을 설치합니다. CARLA 실행 파일은 별도로 준비합니다.

```bash
python -m pip install -r requirements.txt
python carla_run.py --check-config
python carla_run.py --carla-root /path/to/CARLA
```

프로젝트의 `carla/CarlaUE4.sh`에 CARLA가 있으면 `python carla_run.py`만 실행하면 됩니다. 설정과 기본 출력 경로는 현재 터미널 위치가 아니라 프로젝트 위치를 기준으로 합니다.

기존 전용 서버를 사용하려면 다음과 같이 실행합니다. 이 모드는 YAML의 타운으로 월드를 다시 로드합니다.

```bash
python carla_run.py --connect --host 127.0.0.1 --port 2000 --tm-port 8000
```

먼저 소량만 수집하려면 다음과 같이 실행합니다.

```bash
python carla_run.py --carla-root /path/to/CARLA --samples 20 --output ./sample_data
```

`--config`, `--output`, `--map-root`로 YAML, 저장 위치, 전역 지도 위치를 바꿀 수 있습니다. 기본 전역 지도 폴더는 실제 첨부 파일에 있던 `data5`입니다. 로컬 폴더명이 `data05`라면 `--map-root ./data05`를 지정합니다.

## 최종 BEV 기준

추가로 제공한 후처리 코드의 전역 지도 샘플링을 수집 루프에 통합했습니다.

- 크기: 200×200
- 해상도: 5픽셀/m, 즉 0.2m/픽셀
- 범위: 자차를 중심으로 약 40m×40m
- 회전: ego yaw + 90도
- 중심: `int((world_coordinate - world_offset) * 5)`
- 샘플링: 원본 코드와 같은 최근접 정수 샘플링, 범위 밖은 0
- Town04: ego z > 5일 때 `das_full_high.png`, `lane_full_high.png`
- Town05: ego z > 7일 때 `das_full_high.png`, `lane_full_high.png`

`road_network`는 원본 지도의 색상과 알파 채널을 유지합니다. `das`, `lane`, `stopline`은 원본처럼 회색조 지도를 읽습니다. 원본 기본 설정대로 이진화와 침식은 하지 않습니다.

기존 500×500 전역 지도 렌더링, 도로/차선/정지선의 중복 생성, 신호등 BEV 렌더링을 제거했습니다. 정적 BEV는 `data5`에서 직접 잘라냅니다. 차량·보행자와 경로는 같은 200×200 좌표계에 바로 그립니다. 수집기에 PyTorch, CUDA 텐서 렌더러, pygame이 필요하지 않습니다. CARLA 서버 자체의 그래픽 실행 환경은 별도로 필요합니다.

동적 마스크는 이전 footprint 규칙을 유지합니다. 차량은 actor 중심·yaw와 bounding box 크기(최소 1m)를 사용하고, 보행자는 1.4m 정사각형을 사용합니다. 자차는 제외합니다. GPU 보간 대신 OpenCV 폴리곤으로 그리므로 원본 500×500 마스크를 잘라낸 결과와 경계 픽셀이 완전히 같지는 않습니다. 차량과 보행자의 footprint 세계 좌표를 JSON에도 보관합니다.

## 출력

기본 저장 위치는 `data/<Town>/scene_0001/`입니다. 모든 출력은 `000000_a` 같은 공통 파일명을 사용합니다.

| 폴더 | 내용 |
| --- | --- |
| `<camera>_rgb` | BGR 배열을 PNG로 저장한 카메라 RGB 이미지 |
| `das` | 최종 주행 가능 영역, 제공한 코드의 `das2`에 해당 |
| `lane` | 최종 차선, 제공한 코드의 `lane2`에 해당 |
| `stopline` | 최종 정지선, 제공한 코드의 `stopline2`에 해당 |
| `road_network` | 최종 컬러/알파 지도 crop |
| `vehicle_mask` | 200×200 차량 마스크 |
| `walker_mask` | 200×200 보행자 마스크 |
| `route` | 200×200 주행 경로 |
| `info` | ego 상태, 제어, IMU, 프레임 번호, BEV 메타데이터 |
| `<camera>_seg` | `seg: true`일 때 CARLA 색상 segmentation |
| `<camera>_seg_ids` | `seg: true`일 때 원본 semantic ID |
| `<camera>_depth` | `depth: true`일 때 미터 단위 float32 NPY 및 mm 단위 uint16 PNG |
| `<camera>_bbs` | `od: true`일 때 `[x1,y1,x2,y2,class]` NPY |
| `lidar` | YAML에 lidar가 있을 때 `[x,y,z,intensity]` NPY |

최종 폴더명을 `das/lane/stopline`으로 통일했으므로 `das2/lane2/stopline2`를 중복 저장하지 않습니다. 기존 중간 BEV도 생성하지 않습니다.

깊이값의 정확한 학습용 원본은 NPY입니다. PNG는 밀리미터 단위이며 65.535m 초과 값이 잘립니다. 기존 코드처럼 float 깊이를 8비트 PNG에 넣지 않습니다. LiDAR는 CARLA의 미터 좌표와 intensity를 그대로 저장하며, 원본의 전체 배열 `* 2.0`은 제거했습니다. 회전 주파수 기본값은 20Hz이고 lidar 설정에서 변경할 수 있습니다.

OD는 기존 차량·보행자 bbox 기능을 유지하며 신호등 bbox는 생성하지 않습니다. 각 카메라의 개별 FOV·크기와 현재 CARLA 패키지의 semantic ID를 사용합니다. 카메라 각도는 모두 도(degree) 단위입니다.

## 설정과 실행 동작

기본 YAML은 원본의 6개 카메라, Town05, 타운당 10000개 샘플 설정입니다. 숫자 문자열 `data_num: '10000'`도 허용합니다. 카메라별 `seg/depth/od`는 각각 독립적으로 적용됩니다. 키를 생략하면 false입니다.

- 기본 시뮬레이션 간격 0.05초, 2틱마다 저장하여 시뮬레이션 시간 기준 10Hz입니다.
- 카메라·IMU·LiDAR의 프레임 번호가 모두 현재 world frame과 같아야 저장합니다. 타임아웃이나 프레임 불일치는 오류로 종료합니다.
- 충돌이나 기본 1000틱마다 날씨를 바꿀 때 새 scene을 시작합니다. 기존 scene을 덮어쓰지 않습니다.
- 기본 조향 노이즈 ±0.2는 원본과 같습니다. `--steer-noise 0`으로 끌 수 있습니다.
- `--weather-every 0`으로 날씨 변경을 끌 수 있습니다.
- `--vehicles`, `--walkers`, `--seed`로 NPC 수와 난수 시드를 지정할 수 있습니다.
- 이미지 저장이 모두 성공한 뒤 JSON을 마지막에 저장합니다. 저장 오류가 나면 해당 샘플의 부분 파일을 지웁니다.
- 정상 종료, 예외, Ctrl+C 시 생성한 actor를 정리하고 월드 설정을 복원합니다. 직접 실행한 서버만 종료합니다.

scene별 `collection.yaml`에는 실제 적용한 설정을 기록합니다. 재실행은 남은 개수의 자동 재개가 아니라 새 scene에서 설정된 개수만큼 추가 수집합니다.

## 이전 인지 학습 코드 연결

이 수집 결과는 이전에 정리한 학습 코드의 `legacy` 폴더명과 200×200 입력 크기에 맞습니다. `train/val`은 scene 단위로 나누어 배치합니다. 같은 scene의 인접 프레임을 양쪽에 나누지 않는 것이 좋습니다.

학습 설정은 이 데이터의 범위에 맞게 조정합니다.

```bash
python train.py --data-root ./dataset --layout legacy --yaw-unit degrees --bev-extent 20 --projection-frame ego --scene-centroid 0 0 0 --encoder-config encoder_config.json
```

위 명령은 별도의 학습 프로젝트에서 실행합니다. 이번 압축에는 수집 파일만 포함합니다. `--bev-extent 50`이나 500→200 리사이즈는 이 데이터의 40m 범위와 맞지 않습니다. 실제 카메라 투영 축과 모델 학습 품질은 CARLA 수집 샘플로 별도 확인해야 합니다.

## 포함 및 제외

`carla_run.py`, `collector/`, 필요한 `carla_data_gan/` 주행·센서 의존 코드, 센서 YAML, 실제 사용되는 `data5` 지도 파일만 남겼습니다. GUI `start.py`, 재생 코드, 학습/모델 코드, 중복 main, 미사용 지도 변형, 캐시를 제외했습니다. 코드 주석과 docstring을 제거했고, 제3자 라이선스는 별도 문서에 보존했습니다. W&B, 계정명, API 키 설정은 없습니다.

## 검증 범위

첨부된 전역 지도에서 원본 crop 함수와 최종 정적 BEV의 픽셀 일치, Town04/05 고가도로 분기, 경계 밖 0 처리, 동적 마스크 방향·크기, 프레임 버퍼, 깊이값, 저장 실패 정리, 이전 학습 데이터로더의 샘플 로딩을 검사했습니다. 실제 CARLA 서버 주행·센서 스트리밍은 이 환경에서 실행하지 못했습니다.
