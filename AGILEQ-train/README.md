# AGILEQ-train

[한국어](README_ko.md) · [Training repository](../README.md)

Reinforcement-learning training for autonomous driving in CARLA. The module contains the CrossQ++ implementation, driving environment, reward functions, and BEV map assets. Available configuration names are `PPO`, `SAC`, `DDPG`, `TD3`, `TQC`, `crossq`, and `crossq_pp`.

## Contents

| Path | Purpose |
| --- | --- |
| `train.py` | Train or resume a driving policy and save checkpoints. |
| `config.py` | Algorithm settings, observation features, and reward parameters. |
| `crossq_pp/` | CrossQ++ algorithm and policy code. |
| `carla_env/` | CARLA environment, navigation, sensors, rewards, and wrappers. |
| `BEV/` | Map rendering utilities, map images, and world offsets. |
| `start_train.sh` | Single-run training settings. |
| `run_experiments.py` | Batch training launcher. |
| `environment_agileq_train.yml` | Exported Conda environment. |

## Setup

Run commands from `AGILEQ-Training/AGILEQ-train/`. Use Linux, CARLA 0.9.15 with the required towns, and an NVIDIA GPU. The supplied environment uses Python 3.9 and PyTorch 2.7 with CUDA 12.8 packages; the code selects CUDA directly.

```bash
conda env create -n agileq_train -f environment_agileq_train.yml
conda activate agileq_train
export CARLA_ROOT=/absolute/path/to/CARLA_0.9.15
```

The YAML is an environment export, with platform-specific builds and CUDA wheel versions. Adjust package sources or versions for your machine if those builds are unavailable. `sb3_contrib` must expose `CrossQ` and `BatchRenorm1d`, which the supplied code imports.

Before training, change the W&B `entity` and `project` in `train.py` to your workspace, then authenticate with `wandb login`. For offline logging, set `WANDB_MODE=offline`.

The environment starts CARLA using `CARLA_ROOT/CarlaUE4.sh`. Its default `activate_model=False` uses simulator-derived BEV observations. The optional perception branch requires `run/cls/best_loss.pt`; that checkpoint is not included.

## Train

```bash
python train.py \
  --config crossq_pp \
  --town town01 \
  --port 2000 \
  --total_timesteps 600000 \
  --save_dir ./results \
  --no_render
```

To resume, add `--reload_model /absolute/path/to/model_100000_steps.zip`. Keep the algorithm configuration and observation space compatible with the checkpoint. `--fps` defaults to 15; `--num_checkpoints` defaults to 12. `--no_render` disables the environment display.

Town names must match the directories under `BEV/` exactly. The bundled names are `town01`, `town02`, `town03`, `town04`, `town05`, `town07`, and `town10HD`; matching CARLA maps must also be installed.

To use `start_train.sh`, first update its `CARLA_ROOT` and run settings. For batch runs, edit `run_experiments.py`, including its CARLA path, towns, and experiments. The batch launcher terminates CARLA server processes between runs.

## Outputs and evaluation

Training writes model checkpoints, `config.json`, CSV/TensorBoard logs, and `training_log_dw_reward.json` under `--save_dir/<run>/`. Additional W&B model snapshots use `models/<run>/`.

Evaluate trained policies in [AGILEQ-Evaluation](https://github.com/SungjinDavidLee/AGILEQ-Evaluation). That repository also requires compatible BEV and traffic-light perception checkpoints; weights are not bundled here.

Python comments and docstrings are removed, with original third-party header notices preserved in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Python syntax and preservation of executable statements were checked during import; CARLA training has not been run as part of this repository update.
