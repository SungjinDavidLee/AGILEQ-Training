# TLCFormer

[한국어](README_ko.md)

Predict traffic-light GO/STOP state and distance bins from consecutive front-camera frames. This module reorganizes the `LightFormer` implementation supplied in `xa.zip`; it is not a new reproduction of a separate paper.

The original labels are **not red/yellow/green classes**. The `traffic_light_20` field maps 0/1 to the original GO/STOP display convention. Verify the conditions used by your data generator to produce this field.

## Run

Use Python 3.10 or newer. For GPU execution, install PyTorch and torchvision appropriate for your CUDA environment first. Run from `TLCformer/`:

```bash
pip install -r requirements.txt
python train.py --config configs/tlcformer.yaml
```

Set `data.root` in `configs/tlcformer.yaml`. Paths are relative to the working directory. Defaults are 10 frames at height 512 × width 960, matching the original dataset's actual resizing rather than unused 224/480 settings in the old training script.

`pretrained_backbone: true` uses torchvision's pretrained ResNet18 weights and may download them on first use. Set it to false to start without that download. Loading a checkpoint does not download pretrained backbone weights. Reduce `train.batch_size` if the default batch size of 8 exceeds GPU memory.

```bash
python train.py --config configs/tlcformer.yaml --resume runs/tlcformer/last.pt
python train.py --config configs/tlcformer.yaml --weights previous_model.pth
python predict.py --checkpoint runs/tlcformer/best_loss.pt --images data/val/Town01/scene_0001/front_cam_pv_rgb --output prediction.json
```

- `--resume` restores the model, optimizer, scheduler, completed epoch, and best metrics. `epochs` is the total target, not an additional count. Exact random-number progress is not restored.
- `--weights` loads model weights only and starts a new training run.
- Folder inference uses the last `sequence_length` PNGs in numeric order. Explicit file arguments preserve the supplied order; pass oldest to newest.
- For an original plain `state_dict`, also pass `--config configs/tlcformer.yaml` during inference.

No W&B account or API key is required. Logs are saved to `train.log` and `metrics.jsonl`.

## Data contract

Example paths:

```text
data/train/Town01/scene_0001/front_cam_pv_rgb/000001.png
data/train/Town01/scene_0001/info/000001.json
data/val/Town01/scene_0002/front_cam_pv_rgb/000001.png
data/val/Town01/scene_0002/info/000001.json
```

Current-frame JSON:

```json
{"traffic_light_20": true, "traffic_light_distance": 12.4}
```

| Item | Rule |
| --- | --- |
| Light state | 0/false → GO; 1/true → STOP |
| Distance 0 or above 20 m | Class 21: Far or the original no-distance value |
| Distance above 0 and at most 20 m | Rounded integer class from 0 to 20 |
| Rounding | Half-to-even, matching the original `torch.round` behavior |
| Negative, NaN, infinite, or missing labels | Raise an error; never silently substitute labels |
| Input | `[B,T,3,H,W]`, RGB with ImageNet normalization |
| Sequence | T consecutive files within one scene; use the final frame's label |
| Frame order | Numeric-aware filename ordering; missing-frame time gaps are not interpolated |
| `skip_first` | Skip the first 10 images per scene by default; set to 0 if needed |
| `sample_stride` | Step between sequence endpoints, not between frames within a sequence |

`data.scene_glob` defaults to `Town*/scene_*`. Use `Town01/scene_*` to restrict it to Town01. This module does not read `sensor_config.yaml`. Prepare non-overlapping training and validation scenes and verify your collector provides the configured image directory and both required JSON labels.

Class 21 does not mean 21 m. Far is handled separately in ±1 m/±2 m accuracy, so confusing Far with 20 m does not count as correct. In inference JSON, `distance_bin_m` is a classification bin, not continuous distance regression; it is null for Far.

## Structure

| File | Purpose |
| --- | --- |
| `train.py` | Training, validation, checkpoint saving, and resume |
| `predict.py` | Sequence inference and JSON output |
| `configs/tlcformer.yaml` | Model, dataset, and training settings |
| `tlcformer/model.py` | Model assembly and forward pass |
| `tlcformer/attention.py` | Temporal self-attention and spatial deformable cross-attention |
| `tlcformer/heads.py` | Multi-center ArcFace light classification |
| `tlcformer/data.py` | Sequences, preprocessing, and label conversion |
| `tlcformer/engine.py` | Losses, backpropagation, and sample-weighted metrics |
| `tlcformer/checkpoint.py` | Checkpoint persistence and original-weight compatibility |
| `tests/test_tlcformer.py` | Data, model, training, and checkpoint checks |

ResNet18 and two 3 × 3 convolutions encode the frames. Two queries pass through temporal and spatial attention for each frame. The first output token predicts light state; the second predicts one of 22 distance classes.

The original learned reference points, ReLU on offsets/attention scores, and temporal attention using fixed queries and previous-frame tokens are retained. Spatial attention uses the same single-feature-level PyTorch `grid_sample` computation as the original call, removing the MMCV installation requirement.

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

`light_targets` is an int64 `[B]` tensor supplied only during training. Training loss uses ArcFace-margin logits; metrics, validation, and inference use target-independent logits. The model returns logits, and `predict()` converts them to probabilities. Training and validation loss values differ in margin application, so they are not directly comparable.

## Refactoring included in the supplied module

- Consolidated duplicate traffic-model implementations under `TLCFormer`.
- Removed unrelated BEV, InternImage, path-generation, distributed-training, build, and cache files.
- Removed undefined `vehicle_front_gt` and unused vehicle/sign heads and losses.
- Removed broken `models_carla` imports and unnecessary Lightning, segmentation, OpenCV, and MMCV dependencies.
- Replaced unconditional ArcFace `squeeze()` to support batch-one inference.
- Removed target-label input from validation. Inference also applies ArcFace scale 20, so probabilities may differ from the original unlabeled path.
- Uses `cross_entropy(logits)` instead of `log(softmax)`. ArcFace's sine calculation has a small lower bound to avoid divergent derivatives at zero.
- Fixed retained epoch-loss graphs, missing scheduler steps, last-batch-only accuracy, and unequal weighting of batch means.
- Keeps the final validation batch and aggregates over all samples.
- Follows the input device instead of unconditional CUDA calls. Previous-frame tokens are local to each forward call.
- Uses the actual input tensor's T dimension; training and inference defaults share one configuration.

Explanations are in the documentation; Python comments and docstrings are removed.

## Original checkpoints

The default architecture and parameter names support loading the original `LightFormer.state_dict()`. Internal names `Light_img_encoder`, `head1`, and `Light_decoder` are retained for compatibility. Only the five explicitly removed `head2.*` and `signs_decoder.*` tensors are ignored. All other missing, unexpected, or incompatible tensors fail strict loading. A `module.` prefix is supported.

The supplied module's compatibility checks used weights generated from the supplied original source, not a user's trained checkpoint. The original four-element forward tuple is replaced with a traffic-light-focused dictionary API.

## Verification

```bash
python -m unittest discover -s tests -v
```

The original Korean documentation reports seven CPU tests covering batch-one/two inference, independence between calls, finite nonzero gradients and optimizer updates, image/JSON loading, training/validation with a short final batch, checkpoint restoration, and distance-boundary/Far metrics.

It also reports equal-weight numerical comparison with the original implementation, with approximately 2.98e-7 maximum absolute token error, and a synthetic-data CLI training → checkpoint → resume → JSON inference check. Those are prior preparation results, not newly executed tests for this upload. Real CARLA accuracy, CUDA execution, and performance/memory at the default 10 × 512 × 960 input were not validated there.

This repository update checked Python syntax and executable AST equivalence after removing comments and docstrings.
