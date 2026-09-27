# BEVFormer training

[한국어](README_ko.md)

Multi-camera BEV perception training source. The default training entry point imports `models_carla/bevformer.py`; additional model variants are retained from the supplied archive.

| File or directory | Purpose |
| --- | --- |
| `train.py` | Training, validation, checkpointing, and W&B logging |
| `data.py` | Six-camera CARLA dataset, camera geometry, BEV targets, and LiDAR loading |
| `models_carla/bevformer.py` | Default BEV perception model |
| `models_carla/` | Additional supplied model variants |
| `models_carla/models/` | InternImage backbone and model builder |
| `models_carla/ops/` | Multi-scale deformable-attention Python bindings and C++/CUDA source |

## Setup status

This archive is not a self-contained runnable training package. In particular:

- `train.py` imports `get_config_b` and `get_config_t` from `configs.config`, which is not included. The root `config.py` is a different configuration file and is not a substitute.
- The InternImage backbone imports `ops_dcnv3`, which must be supplied and built separately.
- The default model includes CUDA-specific operations. Install mutually compatible PyTorch, torchvision, CUDA toolkit, and compiler versions. The archive does not pin a complete working environment.
- Python dependencies include NumPy, OpenCV, PyYAML, pyquaternion, tqdm, segmentation-models-pytorch, efficientnet-pytorch, timm, and wandb. Some alternative modules require additional dependencies.

Prebuilt Python 3.9 Linux extensions, object files, build metadata, eggs, and Python caches are excluded. Build the supplied deformable-attention extension in the target environment:

```bash
cd models_carla/ops
python setup.py build_ext --inplace
cd ../..
```

After supplying the missing configuration and dependencies, the training entry point is:

```bash
python train.py --wandb_mode disabled
```

The current source still imports `wandb` even in disabled mode. It also assigns the placeholder `???` to `WANDB_API_KEY` and contains existing project/entity defaults; configure these before using online logging. This upload removes comments and docstrings without changing training behavior.

## Dataset

Run from this directory. `data.py` looks under `data/train/Town01` through `Town05` and the corresponding `data/val` directories. It loads `data/sensor_config.yaml` and uses these camera prefixes:

`left_cam`, `front_cam`, `right_cam`, `rear_left_cam`, `rear_cam`, `rear_right_cam`.

Camera images use `<camera>_rgb` directories. The loader also reads BEV labels, metadata, and LiDAR files; inspect `data.py` for the complete contract before collecting a training dataset. Keep training and validation scenes separate.

Do not assume the `data_gen` output is already fully aligned with this archive. Its default sensor names must match the loader, LiDAR must be collected if required, and the model's hard-coded spatial bounds must be reconciled with the collector's 40 m by 40 m BEV extent. The default model currently uses horizontal bounds of -50 to 50 m.

## Verification

Python syntax and executable AST equivalence after comment/docstring removal were checked during repository preparation. CUDA extension builds and actual model training were not run. Original Python copyright and license headers are preserved in `THIRD_PARTY_NOTICES.txt`; C++/CUDA source notices remain in place.
