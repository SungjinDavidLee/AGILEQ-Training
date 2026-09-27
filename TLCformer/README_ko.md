# TLCFormer

[English](README.md)

전방 카메라의 연속 프레임으로 신호등의 GO/STOP 상태와 거리 구간을 예측합니다. 첨부된 `xa.zip`의 `LightFormer` 구현을 기준으로 정리했습니다. 별도 논문 구현을 새로 재현한 코드는 아닙니다.

**원본 라벨은 적색·황색·녹색 3종 분류가 아닙니다.** `traffic_light_20`의 0/1을 원본 표시 방식인 GO/STOP에 대응합니다. 원본 데이터 생성기가 이 필드에 어떤 조건을 기록하는지는 생성 코드를 별도로 확인해야 합니다.

## 실행

Python 3.10 이상을 사용합니다. GPU 환경에서는 해당 CUDA 환경에 맞는 PyTorch와 torchvision을 먼저 설치하세요.

```bash
pip install -r requirements.txt
python train.py --config configs/tlcformer.yaml
```

`configs/tlcformer.yaml`의 `data.root`를 데이터 경로로 바꿉니다. 경로는 실행 디렉터리 기준입니다. 기본값은 원본에서 실제 사용하던 10프레임, 높이 512 × 너비 960입니다. 원본 학습 코드에 적힌 224/480 값은 데이터셋의 실제 리사이즈에 사용되지 않았습니다.

`pretrained_backbone: true`는 torchvision의 ResNet18 사전학습 가중치를 사용하므로 처음 실행할 때 다운로드가 필요할 수 있습니다. 다운로드 없이 시작하려면 `false`로 설정합니다. 체크포인트 로딩 시에는 사전학습 가중치를 다운로드하지 않습니다. 기본 배치 8이 GPU 메모리에 맞지 않으면 `train.batch_size`를 줄이세요.

```bash
python train.py --config configs/tlcformer.yaml --resume runs/tlcformer/last.pt
python train.py --config configs/tlcformer.yaml --weights previous_model.pth
python predict.py --checkpoint runs/tlcformer/best_loss.pt --images data/val/Town01/scene_0001/front_cam_pv_rgb --output prediction.json
```

- `--resume`: 모델·optimizer·scheduler·완료 epoch·best 지표를 복구합니다. `epochs`는 추가 횟수가 아닌 전체 목표 epoch입니다. 난수 진행 위치까지 완전히 복구하는 기능은 아닙니다.
- `--weights`: 가중치만 읽어 새 학습을 시작합니다.
- 추론 폴더에서는 마지막 `sequence_length`장의 PNG를 숫자 순서로 사용합니다. 여러 파일을 지정하면 전달한 순서를 사용하므로 과거→현재 순서로 입력합니다.
- 원본 `state_dict`로 추론할 때는 `--config configs/tlcformer.yaml`도 지정합니다.

W&B, 계정명, API 키는 필요하지 않습니다. 로그는 `train.log`와 `metrics.jsonl`에 저장합니다.

## 데이터 계약

파일 경로 예시:

```text
data/train/Town01/scene_0001/front_cam_pv_rgb/000001.png
data/train/Town01/scene_0001/info/000001.json
data/val/Town01/scene_0002/front_cam_pv_rgb/000001.png
data/val/Town01/scene_0002/info/000001.json
```

현재 프레임의 JSON:

```json
{"traffic_light_20": true, "traffic_light_distance": 12.4}
```

| 항목 | 규칙 |
| --- | --- |
| 신호등 상태 | 0/false → GO, 1/true → STOP |
| 거리 0 또는 20m 초과 | 클래스 21: Far 또는 원본의 거리 없음 값 |
| 0보다 크고 20m 이하 | 반올림한 0~20 정수 클래스 |
| 반올림 | 원본 `torch.round`처럼 정확히 .5일 때 짝수 쪽으로 반올림 |
| 음수·NaN·무한대·누락 라벨 | 오류 발생, 임의의 정답으로 대체하지 않음 |
| 입력 | `[B,T,3,H,W]`, RGB, ImageNet 정규화 |
| 시퀀스 | 같은 scene 안에서 연속된 T개의 파일, 마지막 프레임 라벨 사용 |
| 프레임 순서 | 파일명의 숫자를 고려한 정렬; 누락 프레임의 시간 간격은 보간하지 않음 |
| skip_first | scene마다 첫 10장을 제외하는 원본 기본값 유지; 필요하면 0으로 변경 |
| sample_stride | 시퀀스 끝 지점을 이동하는 간격; 시퀀스 내부 프레임 간격이 아님 |

`data.scene_glob` 기본값은 `Town*/scene_*`입니다. 원본처럼 Town01만 쓰려면 `Town01/scene_*`로 변경합니다. `sensor_config.yaml`은 사용하지 않습니다. 학습과 검증 scene은 서로 겹치지 않게 준비해야 합니다.

거리 클래스 21은 21m를 뜻하지 않습니다. ±1m/±2m 정확도에서 Far와 20m를 정답으로 혼동하지 않도록 별도 처리합니다. 추론 JSON의 `distance_bin_m`는 정확한 거리 회귀값이 아닌 분류 구간이며 Far일 때 `null`입니다.

## 구조

| 파일 | 역할 |
| --- | --- |
| `train.py` | 학습·검증·체크포인트 저장·재개 |
| `predict.py` | 연속 전방 영상에 대한 추론, JSON 출력 |
| `configs/tlcformer.yaml` | 모델·데이터·학습 설정 |
| `tlcformer/model.py` | TLCFormer 조립 및 forward |
| `tlcformer/attention.py` | 시간 self-attention, 공간 deformable cross-attention |
| `tlcformer/heads.py` | 다중 중심 ArcFace 신호등 분류 |
| `tlcformer/data.py` | 시퀀스 구성·이미지 전처리·라벨 변환 |
| `tlcformer/engine.py` | 손실·역전파·전체 표본 기준 지표 집계 |
| `tlcformer/checkpoint.py` | 체크포인트 저장·복구·원본 가중치 호환 |
| `tests/test_tlcformer.py` | 데이터·모델·학습·체크포인트 검증 |

입력 프레임을 ResNet18과 두 개의 3×3 convolution으로 인코딩합니다. 두 query가 프레임별 시간/공간 어텐션을 거친 후, 첫 토큰은 신호등 상태를, 두 번째 토큰은 22개 거리 클래스를 예측합니다.

원본의 학습된 reference point, offset/attention score의 ReLU, 고정 query와 이전 프레임 토큰을 사용하는 시간 어텐션 방식을 유지했습니다. 공간 어텐션은 원본이 호출하던 PyTorch 연산과 같은 단일 feature-level `grid_sample` 계산으로 분리해 MMCV 설치를 제거했습니다. 모델을 임의로 일반 Transformer로 교체하지 않았습니다.

```python
from tlcformer import TLCFormer
from tlcformer.config import ModelConfig

model = TLCFormer(ModelConfig(pretrained_backbone=False))
output = model(images, light_targets)
loss_logits = output['light_loss_logits']
light_logits = output['light_logits']
distance_logits = output['distance_logits']

model.eval()
prediction = model.predict(images)
```

`light_targets`는 학습 때만 전달하는 `[B]` int64입니다. 학습 손실에는 ArcFace margin을 적용한 logits을, 정확도와 검증·추론에는 정답에 의존하지 않는 logits을 사용합니다. 모델은 softmax 전 logits을 반환하며 `predict()`가 확률로 변환합니다. 학습과 검증 손실은 margin 적용 여부가 다르므로 절대값을 그대로 비교하지 마세요.

## 정리 및 수정 사항

- 중복된 두 `traffic_model.py` 중 동일한 구현을 하나로 통합하고 클래스명을 `TLCFormer`로 정리했습니다.
- 신호등과 관련 없는 BEV, InternImage, 경로 생성 함수, 분산학습 설정, 빌드 산출물과 캐시를 제외했습니다.
- 정의되지 않은 `vehicle_front_gt`와 이에 대응하는 미사용 차량/표지판 head·손실을 제거했습니다.
- 존재하지 않는 `models_carla` import와 불필요한 Lightning·segmentation·OpenCV·MMCV 의존성을 제거했습니다.
- ArcFace의 무조건적 `squeeze()`를 제거해 배치 1 추론을 지원합니다.
- 검증 시 정답 라벨을 입력하던 문제를 수정했습니다. 추론 시에도 학습과 같은 ArcFace scale=20을 적용하므로 원본 무라벨 경로와 확률값은 달라질 수 있습니다.
- `log(softmax)` 대신 `cross_entropy(logits)`를 사용합니다. ArcFace sine 계산에는 0 지점의 미분 발산을 방지하는 작은 하한을 사용합니다.
- epoch 손실의 그래프 누적, 누락된 scheduler.step, 마지막 배치만 출력하던 정확도, 배치 평균의 불균등 가중을 수정했습니다.
- 검증 마지막 배치를 버리지 않으며 모든 표본으로 지표를 집계합니다.
- CUDA 강제 호출을 없애고 입력 장치를 따라 계산합니다. 이전 프레임 토큰은 forward 내부 지역변수로 관리합니다.
- 모델 입력 T는 실제 텐서 shape로 처리하며 학습·추론의 기본 T는 공통 설정으로 관리합니다.
- 코드 주석은 넣지 않았습니다. 설명은 이 문서에 모았습니다.

## 원본 체크포인트

기본 구조와 매개변수 이름을 유지해 원본 `LightFormer.state_dict()`를 로딩할 수 있게 했습니다. 내부 `Light_img_encoder`, `head1`, `Light_decoder` 이름은 가중치 호환을 위해 유지했습니다. 삭제된 `head2.*`, `signs_decoder.*` 5개 텐서만 명시적으로 제외하며, 나머지 누락·오류·크기 불일치는 strict loading으로 차단합니다. `module.` 접두사도 지원합니다.

호환 검증은 첨부 소스로 생성한 가중치에 대해 수행했습니다. 사용자의 실제 학습 완료 체크포인트는 첨부되지 않았으므로 직접 검증하지 않았습니다. 원본 forward의 4개 tuple 반환 API는 신호등 중심 dict API로 변경되었습니다.

## 검증

```bash
python -m unittest discover -s tests -v
```

CPU에서 7개 테스트 통과: 배치 1/2 추론, 호출 간 상태 독립성, 주요 모듈의 유한·비영 gradient 및 optimizer 업데이트, 이미지/JSON 시퀀스 로딩, 마지막 작은 배치를 포함한 학습·검증, 체크포인트 복구, 거리 경계 및 Far 지표.

추가로 첨부 원본과 동일 가중치를 사용해 신호등 margin 확률·거리 logits·최종 토큰의 수치 일치를 확인했습니다. 토큰 최대 절대오차는 약 2.98e-7이었습니다. 원본의 불필요한 CUDA reference 생성과 사전학습 가중치 다운로드만 비교 과정에서 우회했습니다.

CLI 학습→체크포인트→재개→JSON 추론은 합성 데이터로 확인했습니다. 실제 CARLA 데이터 정확도, CUDA 실행, 기본 10×512×960 입력의 성능·메모리는 검증하지 않았습니다.
