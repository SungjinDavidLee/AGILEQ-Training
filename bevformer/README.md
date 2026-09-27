# BEVFormer

[한국어](README_ko.md)

One BEVFormer model and one training entry point, replaced with the latest supplied model and DDP training code. The model returns six BEV logits: road, broken lane, solid lane, vehicle, pedestrian, and stop line.

## Structure

| Path | Purpose |
| --- | --- |
| `models/bevformer.py` | The single `BEVFormer` implementation, including geometry, encoder, attention, and six-head decoder |
| `models/backbone/` | InternImage backbone implementation and builder |
| `train.py` | Single-GPU/DDP training, validation, per-town metrics, and checkpoints |
| `configs/backbone.py` | `load_backbone_config()` and InternImage defaults |
| `configs/internimage_t_1k_224.yaml` | The active InternImage-T preset |
| `ops/dcnv3/` | DCNv3 wrappers and C++/CUDA extension |
| `ops/deformable_attention/` | Deformable-attention wrappers and C++/CUDA extension |
| `data.py` | Previous dataset loader, retained for reference; not used by the new trainer |

The old `models_carla` and `ops_dcnv3` directories, alternative BEV models, duplicate model files, unrelated traffic/RL models, and unused configuration presets have been removed. Previous versions remain in Git history.

## Required dataset file

**The new trainer requires `data2.py`, which was not included in the supplied files. Training cannot start until it is provided.** Its expected interface is:

```python
from data2 import Drive_Dataset

train_set = Drive_Dataset(train=True, conf=conf, v=2)
val_set = Drive_Dataset(train=False, conf=conf, v=2)
```

Each sample must provide, in order:

```text
bev_imgs, cam_pam, road_gt, lane_broken_gt, lane_solid_gt,
veh_gt, ped_gt, stop_gt, town_idx, town_name
```

`cam_pam` supplies `intrins`, `rots`, and `trans`. Batched images have shape `[B,N,3,H,W]`. The trainer aggregates town indices in the order `Town01`, `Town02`, `Town03`, `Town04`, `Town05`, `Town10HD`.

The previous `data.py` has a different constructor and return contract, including a single lane target. It is not silently substituted for `data2.py`. Separate broken/solid lane ground truth must come from the actual dataset; it cannot be inferred from a merged lane mask without additional information.

## Environment

Run from `bevformer/`. Install compatible PyTorch, torchvision, CUDA toolkit, a C++ compiler, setuptools, and wheel first. The inherited code does not specify a complete pinned environment.

```bash
python -m pip install numpy opencv-python PyYAML yacs pyquaternion tqdm segmentation-models-pytorch efficientnet-pytorch timm wandb
python -m pip install --no-build-isolation ./ops/dcnv3
python -m pip install --no-build-isolation ./ops/deformable_attention
```

The DCNv3 wrapper uses `pkg_resources`, so setuptools must provide it. Both extension build scripts require CUDA-enabled PyTorch, an available GPU, and the CUDA toolkit. Install DCNv3 rather than only compiling it in place: the wrapper reads its installed package version. The extension installers do not install conflicting top-level `functions` or `modules` packages; wrappers are loaded from this repository's `ops` package.

Place the original InternImage-T checkpoint at `bevformer/internimage_t_1k_224.pth`, with weights under its `model` key. The decoder uses `EfficientNet.from_pretrained("efficientnet-b0")`, which may download weights. These pretrained files are not included.

## Training

After supplying `data2.py` and preparing its dataset:

```bash
python train.py --wandb_mode disabled
python train.py --wandb_mode disabled --resume
torchrun --standalone --nproc_per_node=2 train.py --wandb_mode disabled
```

`--batch_size` is per GPU. The supplied trainer keeps its gradient accumulation, resume behavior, six-target losses, and per-town validation. Output defaults to `run_422/bev_<v>`.

W&B is still an import dependency in disabled mode. The supplied trainer assigns `?` to `WANDB_API_KEY` and retains project/entity defaults. Remove that placeholder assignment and configure your own credentials and account settings before online logging.

Backbone YAML paths resolve relative to `configs/backbone.py`. Dataset, checkpoint, and run paths still resolve from the working directory.

## Model and compatibility

The supplied model uses InternImage-T, six attention layers, an EfficientNet-B0 decoder, and GroupNorm in the added decoder blocks. Its BEV grid is 200 × 200 with eight vertical samples, horizontal bounds of -20 to 20 m, and scene centroid `(0, 1, 0)`. These settings and the six-output order are preserved. The 40 m horizontal extent matches the collector's nominal extent; camera calibration and coordinate conventions still require dataset validation.

```python
from configs import load_backbone_config
from models import BEVFormer

model = BEVFormer({"config": load_backbone_config()}).cuda()
road, lane_broken, lane_solid, vehicle, pedestrian, stopline = model(images, camera_parameters)
```

The new class name and module paths do not change the latest supplied model's parameter attribute names. Its `state_dict` layout is preserved. Old five-output/BatchNorm checkpoints are not assumed compatible with this six-output/GroupNorm model. Full pickled objects referring to old module paths require migration.

## Verification

Python syntax, retained computation ASTs against the two supplied files, and the relocated local import paths were checked. `data2.py` remains the known missing local dependency. C++/CUDA source is unchanged. CUDA extension builds and full model training were not run.

After installing the extensions, run their original checks from this directory:

```bash
python -m ops.dcnv3.test
python -m ops.deformable_attention.test
```

Python comments and docstrings are omitted. Third-party Python notices are preserved in `THIRD_PARTY_NOTICES.txt`; C++/CUDA notices remain in place.
