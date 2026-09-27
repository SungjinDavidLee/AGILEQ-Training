# CARLA image and final BEV collection

[한국어](README_ko.md)

`carla_run.py` reads `data/sensor_config.yaml`, starts CARLA, and saves camera images and final BEV labels from the same frame. A GUI or separate post-collection BEV processing step is not required.

## Run

Install the CARLA Python package matching your server version, then install the remaining dependencies. Supply the CARLA executable separately. Run these commands from `data_gen/`:

```bash
python -m pip install -r requirements.txt
python carla_run.py --check-config
python carla_run.py --carla-root /path/to/CARLA
```

If CARLA is at `carla/CarlaUE4.sh` inside this module, `python carla_run.py` is sufficient. Default configuration and output paths are resolved relative to the module, rather than the current working directory.

Connect to an existing dedicated server, reloading the town specified in the YAML:

```bash
python carla_run.py --connect --host 127.0.0.1 --port 2000 --tm-port 8000
```

Collect a small sample first:

```bash
python carla_run.py --carla-root /path/to/CARLA --samples 20 --output ./sample_data
```

Use `--config`, `--output`, and `--map-root` to override the YAML, output directory, and global maps. The supplied map directory is `data5`. If your local maps are named `data05`, pass `--map-root ./data05`.

## Final BEV specification

The supplied global-map sampling procedure is integrated into collection:

- Size: 200 × 200 pixels.
- Resolution: 5 pixels/m, or 0.2 m/pixel.
- Extent: approximately 40 m × 40 m, centered on the ego vehicle.
- Rotation: ego yaw + 90 degrees.
- Center: `int((world_coordinate - world_offset) * 5)`.
- Sampling: the original integer nearest-neighbor sampling; out-of-bounds values are zero.
- Town04: use `das_full_high.png` and `lane_full_high.png` when ego z > 5.
- Town05: use the high maps when ego z > 7.

`road_network` preserves the map's color and alpha channels. `das`, `lane`, and `stopline` use grayscale maps. Binarization and erosion are disabled, matching the supplied defaults.

Static BEVs are cropped directly from `data5`; the intermediate 500 × 500 renderer, duplicate road/lane/stop-line generation, and traffic-light BEV rendering are removed. Vehicles, pedestrians, and routes are drawn directly in the final coordinate system. The collector does not require PyTorch, CUDA tensor rendering, or pygame. The CARLA server still needs its own graphics environment.

Dynamic masks preserve the original footprint conventions: vehicles use actor center, yaw, and bounding-box dimensions with a 1 m minimum; pedestrians use 1.4 m squares. The ego vehicle is excluded. OpenCV polygons may differ at boundary pixels from the old GPU-rendered and cropped masks. World-coordinate footprints are also retained in JSON.

## Output

The default output is `data/<Town>/scene_0001/`. All outputs share a basename such as `000000_a`.

| Directory | Contents |
| --- | --- |
| `<camera>_rgb` | Camera PNG images saved from BGR arrays |
| `das` | Final drivable-area mask, corresponding to the old `das2` output |
| `lane` | Final lane mask, corresponding to `lane2` |
| `stopline` | Final stop-line mask, corresponding to `stopline2` |
| `road_network` | Final color/alpha map crop |
| `vehicle_mask` | 200 × 200 vehicle mask |
| `walker_mask` | 200 × 200 pedestrian mask |
| `route` | 200 × 200 route |
| `info` | Ego state, control, IMU, frame identifiers, and BEV metadata |
| `<camera>_seg` | CARLA color segmentation when `seg: true` |
| `<camera>_seg_ids` | Raw semantic IDs when `seg: true` |
| `<camera>_depth` | Float32 NPY in meters and uint16 PNG in millimeters when `depth: true` |
| `<camera>_bbs` | `[x1,y1,x2,y2,class]` NPY when `od: true` |
| `lidar` | `[x,y,z,intensity]` NPY when LiDAR is configured |

There are no duplicate `das2`, `lane2`, or `stopline2` directories or intermediate BEVs.

Use depth NPY files for the original training values. Depth PNGs clip beyond 65.535 m. LiDAR preserves CARLA's meter coordinates and intensity; the old array-wide multiplication by 2 is removed. Its default rotation frequency is 20 Hz and can be configured.

Object detection retains vehicle and pedestrian boxes; traffic-light boxes are not generated. Projection uses each camera's own FOV and dimensions and the installed CARLA package's semantic IDs. All configured camera angles are in degrees.

## Configuration and collection behavior

The default YAML uses six cameras, Town05, and 10,000 samples per town. Numeric strings such as `data_num: '10000'` are accepted. Each camera's `seg`, `depth`, and `od` switches are independent and default to false when absent.

- Simulation step: 0.05 s. Saving every two ticks gives 10 Hz in simulation time.
- Camera, IMU, and LiDAR frame IDs must match the current world frame. Timeouts or mismatches stop collection with an error.
- Collisions and periodic weather changes start new scenes. Existing scenes are not overwritten.
- Steering noise defaults to ±0.2; disable it with `--steer-noise 0`.
- Weather changes default to every 1,000 ticks; disable them with `--weather-every 0`.
- Configure NPC counts and randomness with `--vehicles`, `--walkers`, and `--seed`.
- JSON is saved last, after successful image writes. Partial sample files are removed on save failures.
- Normal exit, exceptions, and Ctrl+C clean up created actors and restore world settings. Only a server launched by the collector is terminated.

Each scene's `collection.yaml` records the effective configuration. Restarting creates new scenes and collects the configured number of additional samples; it does not resume a partially completed sample count.

## Training integration

Split data into training and validation sets by scene. Do not place adjacent frames from the same scene in both splits. The final masks are 200 × 200 with a 40 m horizontal extent; a 100 m extent or resizing an old 500 × 500 target is not equivalent.

The Korean source document also describes a separate refactored training project's `legacy` loader. Its example command is retained here for that separate project only:

```bash
python train.py --data-root ./dataset --layout legacy --yaw-unit degrees --bev-extent 20 --projection-frame ego --scene-centroid 0 0 0 --encoder-config encoder_config.json
```

Those options are not implemented by the BEVFormer archive currently included in this repository. Check [the BEVFormer setup and data contract](../bevformer/README.md) before connecting the modules. Camera names, calibration conventions, label keys, and model projection bounds require alignment. TLCFormer's expected traffic-light labels must also be checked separately.

## Included files and verification

This module includes `carla_run.py`, `collector/`, the required `carla_data_gan/` driving and sensor utilities, sensor YAML, and the supplied `data5` maps. GUI startup, playback, model training, duplicate entry points, unused map variants, and caches are excluded. W&B accounts and API keys are not required. Third-party notices are in `THIRD_PARTY_NOTICES.txt`.

The original Korean document reports checks for static-map crop equivalence, elevated-road selection, zero padding, dynamic masks, frame buffering, depth encoding, partial-write cleanup, and loading samples with the separate refactored loader. Live CARLA driving and sensor streaming were not tested in that preparation environment. This repository update checked Python syntax and executable AST equivalence after removing comments and docstrings; it did not run CARLA.
