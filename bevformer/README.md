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
| `configs/` | InternImage configuration factories and YAML presets |
| `ops_dcnv3/` | DCNv3 Python bindings, installation script, and C++/CUDA source |

## Setup status

The configuration and DCNv3 sources are included. Prepare the runtime environment before training:

- `train.py` imports `get_config_b` and `get_config_t` from the included `configs/config.py`. Their InternImage B/T YAML presets are in `configs/`. Run from `bevformer/` because these YAML paths are relative to the working directory. The root `config.py` is a different configuration file.
- The InternImage backbone imports the included `ops_dcnv3` package. Its native `DCNv3` extension must be built and installed for your environment.
- The default model includes CUDA-specific operations. Install mutually compatible PyTorch, torchvision, CUDA toolkit, and compiler versions. The archive does not pin a complete working environment.
- Python dependencies include NumPy, OpenCV, PyYAML, yacs, pyquaternion, tqdm, segmentation-models-pytorch, efficientnet-pytorch, timm, and wandb. The DCNv3 wrapper also uses `pkg_resources` from a compatible setuptools release. Some alternative modules require additional dependencies.

Prebuilt Linux extensions for Python 3.7/3.9, object files, build metadata, eggs, and Python caches are excluded. After installing compatible PyTorch, CUDA, compiler, setuptools, and wheel dependencies, build and install both extensions from `bevformer/`:

```bash
python -m pip install yacs
python -m pip install --no-build-isolation ./ops_dcnv3
python -m pip install --no-build-isolation ./models_carla/ops
```

DCNv3 must be installed, not only compiled in place: the wrapper imports the top-level `DCNv3` extension and reads its installed package version. The supplied build scripts require a CUDA-enabled PyTorch environment with an available GPU and CUDA toolkit.

After preparing the environment and dataset, the training entry point is:

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

Python syntax and executable AST equivalence after comment/docstring removal were checked during repository preparation, including the added configuration and DCNv3 Python sources. Both configuration factories and their referenced YAML files are present. CUDA extension builds and actual model training were not run. Original Python copyright and license headers are preserved in `THIRD_PARTY_NOTICES.txt`; C++/CUDA source notices remain in place.
